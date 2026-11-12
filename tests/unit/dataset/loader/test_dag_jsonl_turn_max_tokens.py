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
        DagLoadError, match=r"requires max_tokens on every turn.*s1\[1\]"
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
