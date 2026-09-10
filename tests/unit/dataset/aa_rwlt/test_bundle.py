# SPDX-FileCopyrightText: Copyright (c) 2026 Artificial Analysis
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import orjson
import pytest

from aiperf.dataset.aa_rwlt.bundle import (
    Methodology,
    bundle_file,
    load_assignments,
    load_manifest,
    read_json,
    sha256,
    write_json,
)
from aiperf.dataset.aa_rwlt.loader import RWLTLoader
from aiperf.dataset.aa_rwlt.prepare import prepare


def test_bundle_is_relocatable_and_pins_content(bundle: Path, tmp_path: Path):
    moved = tmp_path / "moved"
    bundle.parent.rename(moved)
    manifest = load_manifest(moved / "manifest.json")
    assert manifest["request_count"] == 9
    for name, digest in manifest["files"].items():
        assert sha256(moved / name) == digest
    assert load_assignments(moved / "assignments.json")[1] == ["fixture-2", "fixture-0"]


def test_loader_preserves_histories_and_post_response_delay(bundle: Path):
    loader = RWLTLoader(filename=bundle)
    conversations = loader.convert_to_conversations(loader.load_dataset())
    first = next(c for c in conversations if c.session_id == "fixture-0")
    assert [t.delay for t in first.turns] == [0, 25, 50]
    assert [t.source_outer_idx for t in first.turns] == [0, 1, 2]
    assert [m["role"] for m in first.turns[1].extra_body["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert first.turns[1].extra_body["tools"][0]["function"]["name"] == "add"


@pytest.mark.parametrize(
    "field,value",
    [
        ("tool_call_delay_multiplier", -1),
        ("tool_call_delay_multiplier", float("nan")),
        ("user_spawn_rate", 0),
        ("min_measurement_seconds", 59),
        ("phase_timeout_seconds", 1),
    ],
)
def test_invalid_methodology_fails(field, value):
    with pytest.raises(ValueError):
        Methodology.model_validate({field: value})


def test_bundle_paths_cannot_escape(bundle: Path):
    with pytest.raises(ValueError, match="inside"):
        bundle_file(bundle, "../outside")


def test_duplicate_source_turn_is_rejected(tmp_path: Path, fixture_path: Path):
    lines = (fixture_path / "trajectories.jsonl").read_bytes().splitlines(keepends=True)
    source = tmp_path / "duplicate.jsonl"
    source.write_bytes(b"".join(lines + lines[:1]))
    with pytest.raises(ValueError, match="duplicate"):
        prepare(
            source, fixture_path / "assignments.json", tmp_path / "bad", model="fixture"
        )
    assert not (tmp_path / "bad" / "manifest.json").exists()


def test_missing_assignment_rejected(tmp_path: Path, fixture_path: Path):
    assignments = read_json(fixture_path / "assignments.json")
    assignments["assignments"]["1"] = ["missing"]
    path = tmp_path / "assignments.json"
    write_json(path, assignments)
    with pytest.raises(ValueError, match="missing"):
        prepare(
            fixture_path / "trajectories.jsonl", path, tmp_path / "bad", model="fixture"
        )


@pytest.mark.parametrize("delay", [-1, 10001, 1.5, True, False, "25"])
def test_malformed_delay_fails(tmp_path: Path, fixture_path: Path, delay):
    rows = [
        orjson.loads(line)
        for line in (fixture_path / "trajectories.jsonl").read_bytes().splitlines()
    ]
    rows[0]["tool_call_delay_ms"] = delay
    source = tmp_path / "invalid.jsonl"
    source.write_bytes(b"".join(orjson.dumps(row) + b"\n" for row in rows))
    with pytest.raises(ValueError, match="delay"):
        prepare(
            source, fixture_path / "assignments.json", tmp_path / "bad", model="fixture"
        )


@pytest.mark.parametrize("name", ["requests.jsonl", "assignments.json"])
def test_modified_bundle_is_rejected_before_running(bundle: Path, name: str):
    with (bundle.parent / name).open("ab") as file:
        file.write(b"\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_manifest(bundle)


def test_assignment_value_must_be_a_list(tmp_path: Path):
    path = tmp_path / "assignments.json"
    write_json(path, {"assignments": {"1": "fixture-0"}})
    with pytest.raises(ValueError, match="lists"):
        load_assignments(path)
