# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.common.scenario.agentperf_v2 import AGENTPERF_V2
from aiperf.common.scenario.base import ScenarioSpec, UnknownScenarioError
from aiperf.common.scenario.inferencex_agentx_mvp import INFERENCEX_AGENTX_MVP

SCENARIOS: dict[str, ScenarioSpec] = {
    INFERENCEX_AGENTX_MVP.name: INFERENCEX_AGENTX_MVP,
    AGENTPERF_V2.name: AGENTPERF_V2,
}


def get_scenario(name: str) -> ScenarioSpec:
    if name not in SCENARIOS:
        valid = ", ".join(sorted(SCENARIOS.keys()))
        raise UnknownScenarioError(
            f"Unknown scenario {name!r}. Valid scenarios: {valid}"
        )
    return SCENARIOS[name]
