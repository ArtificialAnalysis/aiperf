# SPDX-FileCopyrightText: Copyright (c) 2026 Artificial Analysis. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.common.enums import CacheBustTarget
from aiperf.common.scenario.base import LoopRules, ScenarioSpec
from aiperf.plugin.enums import TimingMode

AGENTPERF_V2 = ScenarioSpec(
    name="agentperf-v2",
    timing_mode=TimingMode.AGENTIC_REPLAY,
    # Every turn carries its recorded output length as max_tokens, and
    # ignore_eos makes each reply exactly that long, so reply length cannot
    # move the score.
    require_ignore_eos=True,
    require_turn_max_tokens=True,
    forbid_extra_input_keys=(
        "max_tokens",
        "max_completion_tokens",
        "min_tokens",
        "min_new_tokens",
        "stop",
        "stop_token_ids",
        "reasoning_effort",
        "chat_template_kwargs",
        "temperature",
        "top_p",
        "top_k",
    ),
    require_streaming=True,
    forbid_ignore_trace_delays=True,
    forbid_input_truncation=True,
    require_loader=("agentperf_v2", "dag_jsonl"),
    min_benchmark_duration_seconds=1,
    default_benchmark_duration_seconds=43200,
    default_trajectory_start_min_ratio=0.0,
    default_trajectory_start_max_ratio=1.0,
    require_cache_bust=CacheBustTarget.FIRST_TURN_PREFIX,
    loop=LoopRules(
        burn_in_seconds=900,
        ramp_seconds=300,
        set_hours=(16, 32, 64),
        longest_pass_seconds=4500,
    ),
)
