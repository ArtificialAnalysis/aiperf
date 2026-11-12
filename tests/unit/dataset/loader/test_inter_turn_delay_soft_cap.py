# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smooth (tanh) inter-turn delay cap: the curve, the tracker, and the loaders that honor it."""

import json
import logging
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
from pytest import param

from aiperf.config.dataset import PublicDataset
from aiperf.dataset.loader._delay_cap import (
    DelayCapTracker,
    clamp_inter_turn_delay_ms,
    soft_cap_ms,
)
from aiperf.dataset.loader.dag_jsonl import DagJsonlLoader, _resolve_delay_cap
from tests.unit.dataset.loader.conftest import make_weka_run

CAP_MS = 10_000.0


class TestSoftCapCurve:
    @pytest.mark.parametrize(
        "delay_ms,expected_ms",
        [
            param(0.0, 0.0, id="zero"),
            param(1_000.0, CAP_MS * math.tanh(0.1), id="tenth_of_cap"),
            param(CAP_MS, CAP_MS * math.tanh(1.0), id="at_cap"),
            param(60_000.0, CAP_MS * math.tanh(6.0), id="six_caps"),
        ],
    )
    def test_matches_cap_times_tanh(self, delay_ms: float, expected_ms: float) -> None:
        assert soft_cap_ms(delay_ms, CAP_MS) == pytest.approx(expected_ms)

    def test_short_delays_almost_unchanged(self) -> None:
        assert soft_cap_ms(500.0, CAP_MS) == pytest.approx(500.0, rel=1e-3)

    def test_never_exceeds_cap(self) -> None:
        for delay_ms in (1e4, 1e5, 5.2e6, 1e12):
            assert soft_cap_ms(delay_ms, CAP_MS) <= CAP_MS

    def test_monotonic(self) -> None:
        xs = [i * 250.0 for i in range(400)]
        ys = [soft_cap_ms(x, CAP_MS) for x in xs]
        assert all(b >= a for a, b in zip(ys, ys[1:], strict=False))

    def test_zero_cap_maps_to_zero(self) -> None:
        assert soft_cap_ms(5_000.0, 0.0) == 0.0


class TestClampFunctionModes:
    def test_hard_is_default(self) -> None:
        assert clamp_inter_turn_delay_ms(60_000.0, 10.0) == CAP_MS
        assert clamp_inter_turn_delay_ms(5_000.0, 10.0) == 5_000.0

    def test_tanh_mode(self) -> None:
        assert clamp_inter_turn_delay_ms(60_000.0, 10.0, "tanh") == pytest.approx(
            CAP_MS * math.tanh(6.0)
        )
        assert clamp_inter_turn_delay_ms(5_000.0, 10.0, "tanh") == pytest.approx(
            CAP_MS * math.tanh(0.5)
        )

    @pytest.mark.parametrize("mode", ["hard", "tanh"])
    def test_passthroughs(self, mode: str) -> None:
        assert clamp_inter_turn_delay_ms(None, 10.0, mode) is None
        assert clamp_inter_turn_delay_ms(float("nan"), 10.0, mode) is None
        assert clamp_inter_turn_delay_ms(-5.0, 10.0, mode) == -5.0
        assert clamp_inter_turn_delay_ms(60_000.0, None, mode) == 60_000.0


class TestTrackerTanh:
    def test_counts_and_removed_time(self) -> None:
        tracker = DelayCapTracker(cap_seconds=10.0, mode="tanh")
        out = [tracker.clamp(d) for d in (500.0, 9_000.0, 60_000.0, 120_000.0)]
        assert out[0] == pytest.approx(soft_cap_ms(500.0, CAP_MS))
        assert all(o < CAP_MS for o in out)
        assert tracker.capped_count == 2  # the two delays above the cap
        assert tracker.removed_ms == pytest.approx(
            sum((500.0, 9_000.0, 60_000.0, 120_000.0)) - sum(out)
        )
        assert tracker.max_observed_ms == 120_000.0

    def test_reset_clears_removed_time(self) -> None:
        tracker = DelayCapTracker(cap_seconds=10.0, mode="tanh")
        tracker.clamp(60_000.0)
        tracker.reset()
        assert tracker.removed_ms == 0.0
        assert tracker.mode == "tanh"

    def test_rejects_unknown_mode(self) -> None:
        with pytest.raises(ValueError, match="hard' or 'tanh"):
            DelayCapTracker(cap_seconds=10.0, mode="soft")  # type: ignore[arg-type]

    def test_logs_soft_cap_summary(self, caplog) -> None:
        tracker = DelayCapTracker(cap_seconds=10.0, mode="tanh")
        tracker.clamp(60_000.0)
        with caplog.at_level(logging.INFO):
            tracker.log_summary(logger_name=__name__)
        assert any("Soft-capped" in r.getMessage() for r in caplog.records)


def _write_dag(tmp_path: Path, delays: list[float]) -> Path:
    path = tmp_path / "dag.jsonl"
    turns = [
        {"messages": [{"role": "user", "content": f"t{i}"}], "delay": d}
        for i, d in enumerate(delays)
    ]
    path.write_text(json.dumps({"session_id": "s1", "turns": turns}) + "\n")
    return path


class TestDagLoader:
    def test_tanh_mode_applies_to_dag_turns(self, tmp_path: Path) -> None:
        run = make_weka_run(
            inter_turn_delay_cap_seconds=10.0, inter_turn_delay_cap_mode="tanh"
        )
        loader = DagJsonlLoader(
            filename=str(_write_dag(tmp_path, [0.0, 2_000.0, 600_000.0])), run=run
        )
        conv = loader.convert_to_conversations(loader.load_dataset())[0]
        assert conv.turns[1].delay == pytest.approx(soft_cap_ms(2_000.0, CAP_MS))
        assert conv.turns[2].delay == pytest.approx(soft_cap_ms(600_000.0, CAP_MS))

    def test_hard_mode_unchanged(self, tmp_path: Path) -> None:
        run = make_weka_run(inter_turn_delay_cap_seconds=10.0)
        loader = DagJsonlLoader(
            filename=str(_write_dag(tmp_path, [0.0, 2_000.0, 600_000.0])), run=run
        )
        conv = loader.convert_to_conversations(loader.load_dataset())[0]
        assert conv.turns[1].delay == 2_000.0
        assert conv.turns[2].delay == CAP_MS

    @pytest.mark.parametrize("mode", ["hard", "tanh"])
    def test_cap_resolves_from_public_dataset(self, mode: str) -> None:
        """Public loaders (agentperf_v2) hand their file to DagJsonlLoader; the cap must follow."""
        ds = PublicDataset(
            name="default",
            type="public",
            dataset="agentperf_v2",
            inter_turn_delay_cap_seconds=10.0,
            inter_turn_delay_cap_mode=mode,
        )
        run = SimpleNamespace(cfg=SimpleNamespace(datasets=[ds]))
        assert _resolve_delay_cap(run) == (10.0, mode)

    def test_no_cap_resolves_to_disabled(self) -> None:
        ds = PublicDataset(name="default", type="public", dataset="agentperf_v2")
        assert _resolve_delay_cap(
            SimpleNamespace(cfg=SimpleNamespace(datasets=[ds]))
        ) == (None, "hard")
