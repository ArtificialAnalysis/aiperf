# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scenario-required per-turn max_tokens in the DAG JSONL loader."""

import json
from pathlib import Path

import pytest
from pytest import param

from aiperf.dataset.loader.dag_jsonl import DagJsonlLoader, DagLoadError
from tests.unit.dataset.loader.conftest import make_weka_run


def _write(tmp_path: Path, max_tokens: list[int | None]) -> Path:
    turns = []
    for value in max_tokens:
        turn: dict = {"messages": [{"role": "user", "content": "hi"}]}
        if value is not None:
            turn["max_tokens"] = value
        turns.append(turn)
    path = tmp_path / "dag.jsonl"
    path.write_text(json.dumps({"session_id": "s1", "turns": turns}) + "\n")
    return path


def _loader(path: Path, scenario: str | None) -> DagJsonlLoader:
    run = make_weka_run()
    run.cfg.scenario = scenario
    return DagJsonlLoader(filename=str(path), run=run)


def test_agentperf_v2_accepts_turns_with_max_tokens(tmp_path: Path) -> None:
    loader = _loader(_write(tmp_path, [120, 45]), "agentperf-v2")
    convs = loader.convert_to_conversations(loader.load_dataset())
    assert [turn.max_tokens for turn in convs[0].turns] == [120, 45]


def test_agentperf_v2_rejects_turn_without_max_tokens(tmp_path: Path) -> None:
    loader = _loader(_write(tmp_path, [120, None]), "agentperf-v2")
    with pytest.raises(
        DagLoadError, match=r"requires max_tokens on every request turn.*s1\[1\]"
    ):
        loader.load_dataset()


@pytest.mark.parametrize(
    "scenario",
    [
        param(None, id="no_scenario"),
        param("inferencex-agentx-mvp", id="other_scenario"),
    ],
)  # fmt: skip
def test_missing_max_tokens_allowed_without_requirement(
    tmp_path: Path, scenario: str | None
) -> None:
    loader = _loader(_write(tmp_path, [None, None]), scenario)
    convs = loader.convert_to_conversations(loader.load_dataset())
    assert [turn.max_tokens for turn in convs[0].turns] == [None, None]


def test_agentperf_v2_skips_turns_without_requests(tmp_path: Path) -> None:
    rows = [
        {"session_id": "orch", "orchestrator": True, "spawns": ["child"]},
        {
            "session_id": "child",
            "turns": [
                {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8}
            ],
        },
    ]
    path = tmp_path / "dag.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    loader = _loader(path, "agentperf-v2")
    convs = loader.convert_to_conversations(loader.load_dataset())
    assert {conv.session_id for conv in convs} == {"orch", "child"}


@pytest.mark.parametrize(
    "extra",
    [
        param({"ignore_eos": False}, id="ignore_eos_false"),
        param({"min_tokens": 4}, id="min_tokens"),
        param({"stop": ["\\n"]}, id="stop"),
    ],
)  # fmt: skip
def test_agentperf_v2_rejects_turn_extra_overriding_length(
    tmp_path: Path, extra: dict
) -> None:
    turn = {
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 8,
        "extra": extra,
    }
    path = tmp_path / "dag.jsonl"
    path.write_text(json.dumps({"session_id": "s1", "turns": [turn]}) + "\n")
    loader = _loader(path, "agentperf-v2")
    with pytest.raises(DagLoadError, match=r"overrides it.*s1\[0\]"):
        loader.load_dataset()
