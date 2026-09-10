# SPDX-FileCopyrightText: Copyright (c) 2026 Artificial Analysis
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from aiperf.dataset.aa_rwlt.bundle import Methodology
from aiperf.dataset.aa_rwlt.prepare import prepare


@pytest.fixture
def fixture_path() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def bundle(tmp_path: Path, fixture_path: Path) -> Path:
    prepare(
        fixture_path / "trajectories.jsonl",
        fixture_path / "assignments.json",
        tmp_path / "bundle",
        model="fixture",
        methodology=Methodology(
            user_spawn_rate=10,
            settling_time_seconds=0,
            min_measurement_seconds=60,
            phase_timeout_seconds=80,
            max_starting_line_offset=0,
        ),
    )
    return tmp_path / "bundle" / "manifest.json"


# These tests exercise scheduled callbacks against real monotonic timestamps.
@pytest.fixture(autouse=True)
def no_sleep():
    pass
