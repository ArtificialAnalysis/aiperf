# SPDX-FileCopyrightText: Copyright (c) 2026 Artificial Analysis
# SPDX-License-Identifier: Apache-2.0

import asyncio
import random
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.credit.structs import Credit
from aiperf.dataset.aa_rwlt.bundle import read_json
from aiperf.dataset.aa_rwlt.loader import RWLTLoader
from aiperf.dataset.aa_rwlt.timing import RWLTTiming
from aiperf.plugin.enums import TimingMode
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.conversation_source import SampledSession


class Scheduler:
    def __init__(self):
        self.pending = []

    def schedule_later(self, delay, coroutine):
        self.pending.append((delay, coroutine))

    def close(self):
        for _, coroutine in self.pending:
            coroutine.close()
        self.pending.clear()


@pytest.fixture
def strategy(bundle: Path, tmp_path: Path):
    loader = RWLTLoader(filename=bundle)
    metadata = {
        c.session_id: c.to_metadata()
        for c in loader.convert_to_conversations(loader.load_dataset())
    }
    number = 0

    def new_session(cid):
        nonlocal number
        number += 1
        return SampledSession(cid, metadata[cid], f"session-{number}")

    source = SimpleNamespace(
        get_metadata=metadata.__getitem__,
        session_for_conversation=new_session,
        get_next_turn_metadata=lambda credit: metadata[credit.conversation_id].turns[
            credit.turn_index + 1
        ],
    )
    scheduler = Scheduler()
    config = CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode("aa_rwlt"),
        concurrency=2,
        expected_duration_sec=80,
        grace_period_sec=0,
    )
    now = time.time_ns()
    lifecycle = SimpleNamespace(
        started_at_perf_ns=time.perf_counter_ns(),
        started_at_ns=now,
        now_ns=time.time_ns,
    )
    issuer = SimpleNamespace(
        issue_credit=AsyncMock(return_value=True),
        stop_issuing=lambda: None,
        _credit_router=SimpleNamespace(cancel_all_credits=AsyncMock()),
    )
    s = RWLTTiming(
        config=config,
        conversation_source=source,
        scheduler=scheduler,
        stop_checker=SimpleNamespace(can_send_any_turn=lambda: True),
        credit_issuer=issuer,
        lifecycle=lifecycle,
        run=SimpleNamespace(
            cfg=SimpleNamespace(
                get_default_dataset=lambda: SimpleNamespace(path=bundle)
            ),
            benchmark_id="test-phase",
            artifact_dir=tmp_path,
        ),
        progress=SimpleNamespace(
            in_flight=0,
            all_credits_sent_event=asyncio.Event(),
            counter=SimpleNamespace(
                requests_sent=0,
                requests_completed=0,
                requests_cancelled=0,
                request_errors=0,
            ),
            fatal_error=None,
            record_fatal_error=lambda error: None,
        ),
    )
    yield s
    scheduler.close()


def credit_from(turn, index=None):
    return Credit(
        id=0,
        phase=CreditPhase.PROFILING,
        conversation_id=turn.conversation_id,
        x_correlation_id=turn.x_correlation_id,
        turn_index=turn.turn_index if index is None else index,
        num_turns=turn.num_turns,
        issued_at_ns=0,
        cache_bust_marker=turn.cache_bust_marker,
        cache_bust_target=turn.cache_bust_target,
    )


async def test_fixed_user_assignments_cache_marker_and_recycling(strategy):
    await strategy.start_trajectory(strategy.users[0])
    first = strategy.issuer.issue_credit.call_args.args[0]
    assert first.conversation_id == "fixture-0"
    assert first.turn_index == 0
    await strategy.advance(strategy.users[0], credit_from(first, 2))
    second = strategy.issuer.issue_credit.call_args.args[0]
    assert second.conversation_id == "fixture-1"
    assert second.cache_bust_marker != first.cache_bust_marker
    await strategy.advance(strategy.users[0], credit_from(second, 2))
    third = strategy.issuer.issue_credit.call_args.args[0]
    assert third.conversation_id == "fixture-0"
    assert third.cache_bust_marker not in {
        first.cache_bust_marker,
        second.cache_bust_marker,
    }
    assert strategy.users[0].pass_index == 1
    assert strategy.users[1].completed == 0


async def test_delay_uses_returned_turn_not_next_turn(strategy):
    await strategy.start_trajectory(strategy.users[0])
    first = strategy.issuer.issue_credit.call_args.args[0]
    await strategy.handle_credit_return(credit_from(first, 1))
    delay, continuation = strategy.scheduler.pending.pop()
    assert delay == 0.025
    await continuation
    next_turn = strategy.issuer.issue_credit.call_args.args[0]
    assert next_turn.turn_index == 2
    assert next_turn.cache_bust_marker == first.cache_bust_marker


async def test_final_delay_precedes_completion_count_and_next_trajectory(strategy):
    await strategy.start_trajectory(strategy.users[0])
    first = strategy.issuer.issue_credit.call_args.args[0]
    await strategy.handle_credit_return(credit_from(first, 2))
    assert strategy.users[0].completed == 0
    delay, continuation = strategy.scheduler.pending.pop()
    assert delay == 0.05
    await continuation
    assert strategy.users[0].completed == 1


async def test_seeded_offset_only_applies_to_first_trajectory(strategy):
    strategy.method.max_starting_line_offset = 10
    await strategy.start_trajectory(strategy.users[0])
    first = strategy.issuer.issue_credit.call_args.args[0]
    assert first.turn_index == min(random.Random(42).randint(0, 10), 2)
    await strategy.advance(strategy.users[0], credit_from(first, 2))
    assert strategy.issuer.issue_credit.call_args.args[0].turn_index == 0


async def test_criteria_stop_signals_native_runner_without_mutating_frozen_config(
    strategy,
):
    strategy.elapsed = lambda: 61
    for user in strategy.users:
        user.completed = 30
    assert strategy.should_stop_user()
    await strategy.execute_phase()
    assert strategy.stop_reason == "criteria_met"
    assert strategy.progress.all_credits_sent_event.is_set()
    assert strategy.config.expected_duration_sec == 80
    assert strategy.config.grace_period_sec == 0
    assert (
        read_json(Path(strategy.run.artifact_dir) / "aa_phase.json")["stop_reason"]
        == "criteria_met"
    )


async def test_grace_expiry_cancels_aiperf_requests(strategy):
    strategy.elapsed = lambda: 61
    strategy.method.grace_period_seconds = 0
    strategy.progress.in_flight = 1
    for user in strategy.users:
        user.completed = 30
    assert strategy.should_stop_user()

    async def cancel():
        strategy.progress.in_flight = 0

    strategy.issuer._credit_router.cancel_all_credits.side_effect = cancel
    await strategy.execute_phase()
    strategy.issuer._credit_router.cancel_all_credits.assert_awaited_once()


def test_all_three_stop_criteria_are_required(strategy):
    strategy.elapsed = lambda: 61
    strategy.users[0].completed = 30
    assert not strategy.criteria_met()
    strategy.users[1].completed = 3
    assert strategy.criteria_met()
    strategy.elapsed = lambda: 59
    assert not strategy.criteria_met()


async def test_scheduled_failure_unblocks_native_phase(strategy):
    error = RuntimeError("cannot dispatch")
    observed = []
    strategy.progress.record_fatal_error = observed.append
    strategy.issuer.issue_credit.side_effect = error
    strategy.schedule(0, strategy.start_trajectory, strategy.users[0])
    _, pending = strategy.scheduler.pending.pop()
    with pytest.raises(RuntimeError, match="cannot dispatch"):
        await pending
    assert observed == [error]
    assert strategy.stop_reason == "error"
    assert strategy.changed.is_set()


async def test_cancelled_credit_is_recorded_even_after_sending_stops(strategy):
    await strategy.start_trajectory(strategy.users[0])
    credit = credit_from(strategy.issuer.issue_credit.call_args.args[0])
    strategy.stopped = True
    await strategy.handle_credit_result(SimpleNamespace(credit=credit, cancelled=True))
    assert strategy.cancelled_credits[0]["x_correlation_id"] == credit.x_correlation_id


async def test_final_count_is_observed_at_next_user_boundary(strategy):
    strategy.elapsed = lambda: 61
    strategy.method.min_total_trajectories = 1
    strategy.method.min_trajectories_per_user = 0
    await strategy.start_trajectory(strategy.users[0])
    first = strategy.issuer.issue_credit.call_args.args[0]
    await strategy.advance(strategy.users[0], credit_from(first, 2))
    assert strategy.users[0].completed == 1
    assert strategy.issuer.issue_credit.call_count == 2
    assert not strategy.stopped
    next_turn = strategy.issuer.issue_credit.call_args.args[0]
    await strategy.advance(strategy.users[0], credit_from(next_turn))
    assert strategy.stopped
    assert strategy.stop_reason == "criteria_met"
    assert strategy.issuer.issue_credit.call_count == 2


async def test_tool_delay_remains_pending_after_stop(strategy):
    await strategy.start_trajectory(strategy.users[0])
    first = strategy.issuer.issue_credit.call_args.args[0]
    strategy.stopped = True
    await strategy.handle_credit_return(credit_from(first, 2))
    assert strategy.pending_delays == {0}
    delay, pending = strategy.scheduler.pending.pop()
    assert delay == 0.05
    await pending
    assert not strategy.pending_delays
    assert strategy.users[0].completed == 0
    assert strategy.issuer.issue_credit.call_count == 1
