# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Tool channel progress through the session mailbox, with a paused microphone."""

import asyncio
from copy import deepcopy

import pytest
import torch

from tests.engine.duplex.test_session_runner import append_audio, types
from tests.engine.duplex.test_session_runner_nemotron_voicechat import (
    answer,
    deliver,
    error_codes,
    expose_function_call,
    open_nemotron_harness,
)
from vllm_omni.engine.duplex import commands
from vllm_omni.engine.duplex.config import DuplexSessionState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def envelope(h):
    return h.port.submissions[-1].prompt["model_intermediate_buffer"]["duplex"]


@pytest.mark.asyncio
async def test_call_and_result_advance_without_microphone_or_silence_timer(monkeypatch):
    async with open_nemotron_harness(monkeypatch, extra_body={"silence_continuation": False}) as h:
        await h.run(append_audio(samples=1280))
        call_id, _ = await expose_function_call(h)
        assert [s.prompt["model_intermediate_buffer"]["duplex"]["seq"] for s in h.port.submissions] == [1, 2, 3]
        assert envelope(h)["nvc_tool_step"] is True
        # The result has three forced tokens, each requiring one scheduler wake.
        h.tokenizer.encode = lambda text, **kwargs: [40, 41, 42]
        before = len(h.port.submissions)
        assert error_codes(await answer(h, call_id)) == []
        assert len(h.port.submissions) == before + 1
        for index, token in enumerate((40, 41, 42)):
            await deliver(
                h,
                nvc_function_token=[token],
                nvc_function_response_consumed_generation=[1],
                nvc_function_token_forced=True,
            )
            assert h.session.runtime_config["nvc_tool_result_steps"] == 2 - index
        assert len(h.port.submissions) == before + 3
        assert envelope(h)["seq"] == before + 3
        assert not h.runner.plugin.has_fast_tool_steps(h.session.runtime_config)
        # Supported microphone input resumes on the next scheduler sequence.
        await h.run(append_audio(samples=1280))
        assert envelope(h)["nvc_tool_step"] is False
        assert envelope(h)["seq"] == before + 4
    assert h.session.state is DuplexSessionState.CLOSED
    assert not h.runner.tasks.append_tasks


@pytest.mark.asyncio
async def test_microphone_is_discarded_and_not_replayed_after_eotc(monkeypatch):
    async with open_nemotron_harness(monkeypatch, extra_body={"silence_continuation": False}) as h:
        await h.run(append_audio(samples=1280))
        await deliver(h, nvc_function_token=[20])
        before = len(h.port.submissions)
        events = await h.run(append_audio(samples=2560))
        assert "error" not in types(events)
        assert len(h.port.submissions) == before
        assert not h.runner.model_state.audio_buffer.has_pending()
        await deliver(h, nvc_function_token=[21])
        await h.run(append_audio(samples=1280))
        assert len(h.port.submissions) == before + 1
        assert envelope(h)["nvc_tool_step"] is False


@pytest.mark.asyncio
async def test_real_output_and_progress_cannot_drain_or_duplicate_result_steps(monkeypatch):
    async with open_nemotron_harness(monkeypatch, extra_body={"silence_continuation": False}) as h:
        await h.run(append_audio(samples=1280))
        call_id, _ = await expose_function_call(h)
        h.tokenizer.encode = lambda text, **kwargs: [40, 41]
        await answer(h, call_id)
        before = len(h.port.submissions)
        await deliver(h, nvc_tool_step=False, nvc_function_token=[12])
        await deliver(h, nvc_tool_step=True)  # no function token: pure progress
        await h.runner.model.continue_fast_tool_steps(h.session.epoch)
        assert len(h.port.submissions) == before
        assert h.session.runtime_config["nvc_tool_result_steps"] == 2
        await deliver(
            h, nvc_function_token=[40], nvc_function_response_consumed_generation=[1], nvc_function_token_forced=True
        )
        assert len(h.port.submissions) == before + 1
        assert h.session.runtime_config["nvc_tool_result_steps"] == 1


@pytest.mark.asyncio
async def test_cancel_fences_pending_synthetic_step_and_resets_tool_state(monkeypatch):
    async with open_nemotron_harness(monkeypatch, extra_body={"silence_continuation": False}) as h:
        await h.run(append_audio(samples=1280))
        await deliver(h, nvc_function_token=[20])
        old_epoch = h.session.epoch
        old_runtime = dict(h.session.runtime_config)
        await h.run(commands.CancelInput())
        assert h.session.epoch > old_epoch
        assert not h.runner.plugin.has_fast_tool_steps(h.session.runtime_config)
        assert not await h.runner._schedule_fast_tool_step(
            h.runner.plugin.fast_tool_step_payload(old_runtime), old_epoch
        )
        await h.run(append_audio(samples=1280))
        assert envelope(h)["epoch"] == h.session.epoch
        assert envelope(h)["seq"] == 1
        assert envelope(h)["nvc_tool_step"] is False


@pytest.mark.asyncio
async def test_synthetic_steps_are_charged_against_max_model_len(monkeypatch):
    async with open_nemotron_harness(monkeypatch, extra_body={"silence_continuation": False}) as h:
        await h.run(append_audio(samples=1280))
        runtime = dict(h.session.runtime_config)
        runtime["nvc_max_model_len"] = len(runtime["nvc_prompt_token_ids"]) + 2
        h.session.replace_runtime_config(runtime)
        events = await deliver(h, nvc_function_token=[20])
        assert "runtime_append_failed" in error_codes(events)
        assert len(h.port.submissions) == 1
        assert h.session.state is DuplexSessionState.CLOSED


@pytest.mark.asyncio
async def test_call_budget_cannot_be_extended_by_repeated_sotc(monkeypatch):
    async with open_nemotron_harness(monkeypatch) as h:
        plugin = h.runner.plugin
        runtime = dict(h.session.runtime_config)
        runtime = plugin.runtime_config_after_model_output(runtime, {"nvc_function_token": [20]})
        for remaining in range(511, -1, -1):
            runtime = plugin.runtime_config_after_model_output(
                runtime, {"nvc_function_token": [20], "nvc_tool_step": True}
            )
            assert runtime["nvc_tool_call_steps"] == remaining
        assert not plugin.has_fast_tool_steps(runtime)
        assert plugin.fast_tool_step_payload(runtime) is None
        runtime = plugin.runtime_config_after_model_output(runtime, {"nvc_function_token": [21]})
        runtime = plugin.runtime_config_after_model_output(runtime, {"nvc_function_token": [20]})
        assert runtime["nvc_tool_call_steps"] == 512


@pytest.mark.asyncio
async def test_marker_is_trusted_and_no_tool_output_keeps_runtime_untouched(monkeypatch):
    async with open_nemotron_harness(monkeypatch) as h:
        plugin = h.runner.plugin
        runtime = dict(h.session.runtime_config)
        assert plugin.runtime_config_after_model_output(runtime, {"nvc_function_token": [12]}) is None
        assert not plugin.is_fast_tool_step({"_nvc_tool_step": True, "nvc_tool_step": True})
        runtime["nvc_tool_call_steps"] = 1
        payload = plugin.fast_tool_step_payload(runtime)
        assert plugin.is_fast_tool_step(deepcopy(payload))
        assert "nvc_tool_step" in plugin.private_runtime_config_keys


@pytest.mark.asyncio
async def test_queued_step_cancelled_by_real_eotc_releases_its_slot(monkeypatch):
    async with open_nemotron_harness(monkeypatch, extra_body={"silence_continuation": False}) as h:
        await h.run(append_audio(samples=1280))
        gate = asyncio.Event()

        async def predecessor():
            await gate.wait()
            return True

        h.runner.tasks.append_tail = asyncio.create_task(predecessor())
        try:
            # A real EOTC can already be queued before the SOTC-triggered
            # synthetic append passes its predecessor and reaches the stage.
            task = asyncio.create_task(deliver(h, nvc_function_token=[20], nvc_tool_step=False))
            await asyncio.sleep(0.02)
            await deliver(h, nvc_function_token=[7], nvc_tool_step=False)
            events_task = asyncio.create_task(deliver(h, nvc_function_token=[21], nvc_tool_step=False))
            await asyncio.sleep(0.02)
        finally:
            gate.set()
        events = await events_task
        await task
        await h.settle()
        call_id = next(event.call_id for event in events if event.type == "response.function_call_arguments.done")
        assert len(h.port.submissions) == 1
        assert h.runner.model._fast_tool_step_epoch is None
        assert error_codes(await answer(h, call_id)) == []
        assert len(h.port.submissions) == 2
        assert envelope(h)["nvc_tool_step"] is True


@pytest.mark.asyncio
async def test_synthetic_output_without_forced_token_ack_keeps_result_budget(monkeypatch):
    async with open_nemotron_harness(monkeypatch) as h:
        plugin = h.runner.plugin
        runtime = dict(h.session.runtime_config)
        runtime["nvc_tool_result_steps"] = 2
        # This step was submitted before generation 3 arrived.
        updated = plugin.runtime_config_after_model_output(
            runtime,
            {"nvc_tool_step": True, "nvc_function_token": [12], "nvc_function_response_consumed_generation": [2]},
        )
        assert updated["nvc_tool_result_steps"] == 2


@pytest.mark.asyncio
async def test_already_queued_microphone_reservation_is_consumed_without_replay(monkeypatch):
    async with open_nemotron_harness(monkeypatch, extra_body={"silence_continuation": False}) as h:
        await h.run(append_audio(samples=1280))
        gate = asyncio.Event()

        async def predecessor():
            await gate.wait()
            return True

        h.runner.tasks.append_tail = asyncio.create_task(predecessor())
        try:
            h.submit(append_audio(samples=1280))
            await asyncio.sleep(0.02)
            output_task = asyncio.create_task(deliver(h, nvc_function_token=[20], nvc_tool_step=False))
            await asyncio.sleep(0.02)
        finally:
            gate.set()
        await output_task
        assert len(h.port.submissions) == 2
        assert envelope(h)["nvc_tool_step"] is True
        assert not h.runner.model_state.audio_buffer.has_pending()
        await deliver(h, nvc_function_token=[21])
        await h.run(append_audio(samples=1280))
        assert len(h.port.submissions) == 3
        assert envelope(h)["nvc_tool_step"] is False


@pytest.mark.asyncio
async def test_failed_predecessor_closes_fast_work_without_retry_loop(monkeypatch):
    async with open_nemotron_harness(monkeypatch, extra_body={"silence_continuation": False}) as h:
        await h.run(append_audio(samples=1280))
        gate = asyncio.Event()

        async def predecessor():
            await gate.wait()
            return False

        h.runner.tasks.append_tail = asyncio.create_task(predecessor())
        try:
            output_task = asyncio.create_task(deliver(h, nvc_function_token=[20], nvc_tool_step=False))
            await asyncio.sleep(0.02)
        finally:
            gate.set()
        await output_task
        assert h.session.state is DuplexSessionState.CLOSED
        assert h.runner.model._fast_tool_step_epoch is None
        assert len(h.port.submissions) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("synthetic", [False, True])
async def test_boundary_valued_result_tokens_never_start_extraction(monkeypatch, synthetic):
    async with open_nemotron_harness(monkeypatch) as h:
        plugin = h.runner.plugin
        runtime = dict(h.session.runtime_config)
        runtime["nvc_tool_result_steps"] = 2
        for token, remaining in ((20, 1), (21, 0)):
            runtime = plugin.runtime_config_after_model_output(
                runtime,
                {
                    "nvc_tool_step": synthetic,
                    "nvc_function_token": [token],
                    "nvc_function_response_consumed_generation": [1],
                    "nvc_function_token_forced": [torch.tensor([True])],
                },
            )
            assert runtime["nvc_tool_result_steps"] == remaining
            assert not runtime.get("nvc_tool_call_active")
            assert runtime["nvc_tool_call_steps"] == 0
        assert plugin.fast_tool_step_payload(runtime) is None
