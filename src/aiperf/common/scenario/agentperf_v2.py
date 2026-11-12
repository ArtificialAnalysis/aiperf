# SPDX-FileCopyrightText: Copyright (c) 2026 Artificial Analysis. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.common.enums import CacheBustTarget
from aiperf.common.scenario.base import LoopRules, ScenarioSpec
from aiperf.plugin.enums import TimingMode

AGENTPERF_V2 = ScenarioSpec(
    name="agentperf-v2",
    timing_mode=TimingMode.AGENTIC_REPLAY,
    require_ignore_eos=False,
    require_streaming=True,
    forbid_ignore_trace_delays=True,
    forbid_input_truncation=True,
    require_loader=("agentperf_v2", "dag_jsonl"),
    min_benchmark_duration_seconds=1,
    default_benchmark_duration_seconds=43200,
    default_trajectory_start_min_ratio=0.0,
    default_trajectory_start_max_ratio=1.0,
    require_cache_bust=CacheBustTarget.FIRST_TURN_PREFIX,
    # Every recorded request fits a 1M-token window, so a correctly
    # configured server never overflows; any overflow means skipped turns.
    max_context_overflow_rate=0.0,
    loop=LoopRules(
        burn_in_seconds=900,
        ramp_seconds=300,
        set_hours=(16, 32, 64),
        longest_pass_seconds=4500,
    ),
)
