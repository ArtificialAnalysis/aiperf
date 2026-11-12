# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the aggregate run-metadata helpers."""

import pytest
from pytest import param

from aiperf.common.environment import Environment
from aiperf.exporters.aggregate.aggregate_base_exporter import (
    CONTEXT_OVERFLOW_REASON,
    RUN_CANCELLED_REASON,
    _build_run_metadata_dict,
    compute_submission_outcome,
)


def test_submission_valid_omitted_when_scenario_unset() -> None:
    md = _build_run_metadata_dict(scenario_name=None, submission_valid=None)
    assert "submission_valid" not in md
    assert md == {}


def test_submission_valid_true_when_scenario_set_and_clean() -> None:
    md = _build_run_metadata_dict(
        scenario_name="inferencex-agentx-mvp", submission_valid=True
    )
    assert md["submission_valid"] is True
    assert md["scenario"] == "inferencex-agentx-mvp"
    assert "submission_invalid_reasons" not in md


def test_submission_valid_false_with_reason() -> None:
    md = _build_run_metadata_dict(
        scenario_name="inferencex-agentx-mvp",
        submission_valid=False,
        submission_invalid_reasons=[
            "unsafe_override",
            "context_overflow_rate_exceeded",
        ],
    )
    assert md["submission_valid"] is False
    assert "unsafe_override" in md["submission_invalid_reasons"]
    assert "context_overflow_rate_exceeded" in md["submission_invalid_reasons"]


@pytest.mark.parametrize(
    "scenario_name, validator_submission_valid, was_cancelled, expected_valid, expected_reasons",
    [
        param(
            "inferencex-agentx-mvp",
            True,
            True,
            False,
            [RUN_CANCELLED_REASON],
            id="cancelled_flips_false",
        ),
        param(
            "inferencex-agentx-mvp",
            True,
            False,
            True,
            [],
            id="not_cancelled_keeps_true",
        ),
        param(None, None, True, None, [], id="cancelled_without_scenario_omits"),
    ],
)  # fmt: skip
def test_cancellation_submission_outcome(
    scenario_name,
    validator_submission_valid,
    was_cancelled,
    expected_valid,
    expected_reasons,
) -> None:
    valid, reasons = compute_submission_outcome(
        scenario_name=scenario_name,
        validator_submission_valid=validator_submission_valid,
        was_cancelled=was_cancelled,
    )
    assert valid is expected_valid
    assert reasons == expected_reasons


def test_cancelled_run_appends_reason_to_existing_reasons() -> None:
    # 11 / 500 = 2.2% overflow rate already flips submission_valid;
    # cancellation adds its own reason exactly once.
    valid, reasons = compute_submission_outcome(
        scenario_name="inferencex-agentx-mvp",
        validator_submission_valid=False,
        validator_reasons=["unsafe_override"],
        total_responses=500,
        context_overflow_count=11,
        was_cancelled=True,
    )
    assert valid is False
    assert reasons == ["unsafe_override", CONTEXT_OVERFLOW_REASON, RUN_CANCELLED_REASON]


def test_overflow_rate_boundary_without_double_count() -> None:
    """101 overflows / 10_000 responses must flip submission_valid (regression against double-counting overflows into the denominator)."""
    valid, reasons = compute_submission_outcome(
        scenario_name="inferencex-agentx-mvp",
        validator_submission_valid=True,
        total_responses=10_000,
        context_overflow_count=101,
    )
    assert valid is False
    assert reasons == [CONTEXT_OVERFLOW_REASON]

    # The double-counted denominator incorrectly accepted this run.
    valid_inflated, reasons_inflated = compute_submission_outcome(
        scenario_name="inferencex-agentx-mvp",
        validator_submission_valid=True,
        total_responses=10_101,
        context_overflow_count=101,
    )
    assert valid_inflated is True
    assert reasons_inflated == []


@pytest.mark.parametrize(
    "scenario_name, overflow, total, expected_valid",
    [
        param("agentperf-v2", 0, 10_000, True, id="agentperf_v2_zero_overflow_valid"),
        param("agentperf-v2", 1, 10_000, False, id="agentperf_v2_single_overflow_invalid"),
        param("inferencex-agentx-mvp", 1, 10_000, True, id="agentx_keeps_one_percent_rule"),
        param("inferencex-agentx-mvp", 101, 10_000, False, id="agentx_over_one_percent_invalid"),
    ],
)  # fmt: skip
def test_scenario_overflow_rate_ceiling(
    scenario_name: str, overflow: int, total: int, expected_valid: bool
) -> None:
    valid, reasons = compute_submission_outcome(
        scenario_name=scenario_name,
        validator_submission_valid=True,
        total_responses=total,
        context_overflow_count=overflow,
    )
    assert valid is expected_valid
    assert (CONTEXT_OVERFLOW_REASON in reasons) is (not expected_valid)


def test_env_limit_cannot_loosen_scenario_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Environment.AGENTX, "CONTEXT_OVERFLOW_RATE_LIMIT", 0.5)
    valid, reasons = compute_submission_outcome(
        scenario_name="agentperf-v2",
        validator_submission_valid=True,
        total_responses=10_000,
        context_overflow_count=1,
    )
    assert valid is False
    assert CONTEXT_OVERFLOW_REASON in reasons
