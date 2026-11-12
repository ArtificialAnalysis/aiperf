# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""agentperf-v2 locks output length to the dataset's per-turn max_tokens."""

from pathlib import Path
from typing import Any

import pytest
from pytest import param

from aiperf.common.scenario import ScenarioLockError, apply_scenario
from aiperf.config.config import BenchmarkConfig
from aiperf.config.resolution.plan import BenchmarkRun


def _build_run(
    extra: dict[str, Any] | None = None, endpoint_type: str = "chat"
) -> BenchmarkRun:
    endpoint: dict[str, Any] = {
        "urls": ["http://localhost:8000/v1/chat/completions"],
        "type": endpoint_type,
        "streaming": True,
    }
    if extra is not None:
        endpoint["extra"] = extra
    if endpoint_type == "template":
        endpoint["template"] = {
            "body": '{"messages": {{ messages | tojson }}, "max_tokens": 16}'
        }
    cfg = BenchmarkConfig.model_validate(
        {
            "models": ["my-model"],
            "endpoint": endpoint,
            "datasets": [{"name": "main", "type": "public", "dataset": "agentperf_v2"}],
            "phases": [
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "concurrency": 8,
                    "duration": 1800,
                }
            ],
            "scenario": "agentperf-v2",
        }
    )
    return BenchmarkRun(
        benchmark_id="test-run",
        cfg=cfg,
        artifact_dir=Path("/tmp/aiperf-scenario-test"),
    )


def test_agentperf_v2_injects_ignore_eos() -> None:
    run = _build_run()
    outcome = apply_scenario(run)
    assert outcome.submission_valid is True
    assert run.cfg.endpoint.extra["ignore_eos"] is True


@pytest.mark.parametrize(
    "key, value",
    [
        param("max_tokens", 4096, id="max_tokens"),
        param("max_completion_tokens", 4096, id="max_completion_tokens"),
        param("min_tokens", 16, id="min_tokens"),
        param("stop", ["</s>"], id="stop"),
        param("reasoning_effort", "low", id="reasoning_effort"),
        param("chat_template_kwargs", {"enable_thinking": False}, id="chat_template_kwargs"),
        param("temperature", 0.0, id="temperature"),
        param("payload_template", "{}", id="payload_template"),
        param("stop_regex", "x", id="stop_regex"),
    ],
)  # fmt: skip
def test_agentperf_v2_rejects_output_shaping_extra_inputs(key: str, value: Any) -> None:
    run = _build_run(extra={key: value})
    with pytest.raises(ScenarioLockError) as exc:
        apply_scenario(run)
    assert f"extra_inputs.{key}" in str(exc.value)


@pytest.mark.parametrize("endpoint_type", ["template", "completions"])
def test_agentperf_v2_accepts_only_chat_endpoint(endpoint_type: str) -> None:
    run = _build_run(endpoint_type=endpoint_type)
    with pytest.raises(ScenarioLockError) as exc:
        apply_scenario(run)
    assert "--endpoint-type" in str(exc.value)
