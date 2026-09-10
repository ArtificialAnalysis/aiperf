# SPDX-FileCopyrightText: Copyright (c) 2026 Artificial Analysis
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest

from aiperf.dataset.aa_rwlt.analyze import decode_record, reconcile_capture, stop_checks
from aiperf.dataset.aa_rwlt.bundle import Methodology


@pytest.mark.parametrize("missing_capture", [True, False])
def test_analysis_cli_rejects_incomplete_capture_or_unverified_timeout_policy(
    tmp_path, monkeypatch, missing_capture
):
    import sys

    from aiperf.dataset.aa_rwlt import analyze
    from aiperf.dataset.aa_rwlt.bundle import read_json, write_json

    write_json(
        tmp_path / "aa_phase.json",
        {
            "final": True,
            "phase_start": 100,
            "measurement_start": 105,
            "sending_stop": 165,
            "measurement_end": 166,
            "concurrency": 1,
            "completed_trajectories": {"0": 0},
            "sessions": {},
            "stop_reason": "criteria_met",
            "methodology": Methodology(
                min_measurement_seconds=60,
                min_total_trajectories=0,
                min_trajectories_per_user=0,
            ).model_dump(),
            "native_counts": {
                "requests_sent": int(missing_capture),
                "requests_completed": 0,
                "requests_cancelled": int(missing_capture),
                "request_errors": 0,
            },
        },
    )
    (tmp_path / "profile_export_raw.jsonl").write_bytes(b"")
    monkeypatch.setattr(analyze, "tokenizer", lambda _: len)
    monkeypatch.setattr(sys, "argv", ["analyze", str(tmp_path)])
    assert analyze.main() == 2
    summary = read_json(tmp_path / "aa_summary.json")
    assert summary["valid_stop"]
    assert summary["complete_capture"] == (not missing_capture)
    assert not summary["timeout_policy_verified"]


def test_capture_distinguishes_cancelled_requests_from_unknown_loss():
    counts = {
        "requests_sent": 10,
        "requests_completed": 8,
        "requests_cancelled": 2,
        "request_errors": 0,
    }
    stock = reconcile_capture(counts, 8, 0)
    assert not stock["complete_capture"] and stock["cancelled_requests"] == 2
    assert stock["unaccounted_native_requests"] == 0
    counts["requests_cancelled"] = 0
    lost = reconcile_capture(counts, 8, 0)
    assert lost["cancelled_requests"] == 0
    assert lost["unaccounted_native_requests"] == 2
    counts.update(requests_completed=10, request_errors=2)
    fixed = reconcile_capture(counts, 10, 2)
    assert fixed["complete_capture"] and fixed["cancelled_requests"] == 2
    with pytest.raises(ValueError, match="missing completed"):
        reconcile_capture(counts, 9, 2)
    with pytest.raises(ValueError, match="more records"):
        reconcile_capture(counts, 11, 2)


def test_stop_checks_independently_verify_the_reported_stop_reason():
    phase = {
        "phase_start": 100,
        "measurement_start": 105,
        "sending_stop": 165,
        "concurrency": 2,
        "completed_trajectories": {"0": 3, "1": 3},
        "methodology": Methodology(
            min_measurement_seconds=60, min_total_trajectories=6
        ).model_dump(),
    }
    assert all(stop_checks(phase).values())
    phase["completed_trajectories"]["1"] = 2
    assert not stop_checks(phase)["minimum_trajectories_per_user"]
    phase["sending_stop"] = 164
    assert not stop_checks(phase)["minimum_measurement"]
    phase["completed_trajectories"] = {"0": 30}
    assert not stop_checks(phase)["user_coverage"]


def test_native_empty_response_validation_retains_legacy_quality_semantics():
    raw = {
        "metadata": {
            "x_correlation_id": "session",
            "request_start_ns": 1000000000,
            "request_end_ns": 2000000000,
            "conversation_id": "fixture",
            "turn_index": 0,
            "error": {
                "type": "InvalidInferenceResultError",
                "message": "No responses with actual content were received",
            },
        },
        "start_perf_ns": 0,
        "status": 200,
        "responses": [],
    }
    session = {"session": {"user_id": 0, "cache_bust_marker": "[rid:123456789abc]"}}
    record, _ = decode_record(raw, len, session)
    assert record["success"] and record["num_content_chunks"] == 0
    assert record["native_error"] is not None
    other = copy.deepcopy(raw)
    other["metadata"]["error"] = {"type": "TimeoutError", "message": "timeout"}
    record, _ = decode_record(other, len, session)
    assert not record["success"]


def test_offline_sse_validation_skips_malformed_and_error_only_frames():
    import orjson

    from aiperf.dataset.aa_rwlt.analyze import iter_events

    good = {
        "id": "test",
        "object": "chat.completion.chunk",
        "model": "fixture",
        "created": 0,
        "choices": [{"index": 0, "delta": {"content": "hello"}, "finish_reason": None}],
    }
    values = [
        "{malformed",
        orjson.dumps({"error": {"message": "server error"}}).decode(),
        orjson.dumps(good).decode(),
        "[DONE]",
    ]
    raw = {
        "responses": [
            {"perf_ns": i, "packets": [{"name": "data", "value": v}]}
            for i, v in enumerate(values)
        ]
    }
    parsed = list(iter_events(raw))
    assert len(parsed) == 1
    assert parsed[0][0] == 2
    assert parsed[0][1]["choices"][0]["delta"]["content"] == "hello"
