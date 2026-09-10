# SPDX-FileCopyrightText: Copyright (c) 2026 Artificial Analysis
# SPDX-License-Identifier: Apache-2.0

"""Deterministic closed-loop users implemented as an AIPerf timing strategy."""

import asyncio
import contextlib
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aiperf.common.enums import CacheBustTarget
from aiperf.credit.structs import Credit, TurnToSend
from aiperf.dataset.aa_rwlt.bundle import (
    Methodology,
    bundle_file,
    load_assignments,
    load_manifest,
    sha256,
    write_json,
)
from aiperf.dataset.aa_rwlt.payload import cache_marker
from aiperf.dataset.aa_rwlt.transport_settings import REQUEST_TIMEOUT_POLICY


@dataclass
class UserState:
    user_id: int
    assignments: list[str]
    trajectory_index: int = 0
    pass_index: int = 0
    completed: int = 0
    first: bool = True


class RWLTTiming:
    wants_returns_after_sending_complete = True

    def __init__(
        self,
        *,
        config: Any,
        conversation_source: Any,
        scheduler: Any,
        stop_checker: Any,
        credit_issuer: Any,
        lifecycle: Any,
        run: Any,
        progress: Any,
        **kwargs: Any,
    ) -> None:
        self.config = config
        self.source = conversation_source
        self.scheduler = scheduler
        self.stop_checker = stop_checker
        self.issuer = credit_issuer
        self.lifecycle = lifecycle
        self.run = run
        self.progress = progress
        dataset = run.cfg.get_default_dataset()
        self.manifest_path = Path(getattr(dataset, "rwlt_bundle", None) or dataset.path)
        self.manifest = load_manifest(self.manifest_path)
        self.method = Methodology.model_validate(self.manifest["methodology"])
        assignments = load_assignments(
            bundle_file(self.manifest_path, self.manifest["assignments"])
        )
        concurrency = config.concurrency
        if concurrency is None or not 1 <= concurrency <= len(assignments):
            raise ValueError("Concurrency must fit the published user assignments")
        self.users = [UserState(i, assignments[i]) for i in range(concurrency)]
        self.measurement_offset = (
            concurrency - 1
        ) / self.method.user_spawn_rate + self.method.settling_time_seconds
        if (
            self.measurement_offset + self.method.min_measurement_seconds
            > self.method.phase_timeout_seconds
        ):
            raise ValueError(
                "Phase timeout does not accommodate ramp, settling, and measurement"
            )
        if config.expected_duration_sec != self.method.phase_timeout_seconds:
            raise ValueError(
                "AIPerf phase duration must equal methodology.phase_timeout_seconds"
            )
        if (
            config.total_expected_requests is not None
            or config.expected_num_sessions is not None
        ):
            raise ValueError(
                "RWLT uses duration AND trajectory minima; request/session caps are unsupported"
            )
        self.sessions: dict[str, UserState] = {}
        self.session_ledger: dict[str, dict] = {}
        self.changed = asyncio.Event()
        self.stopped = False
        self.stop_reason = "running"
        self.started_wall = 0.0
        self.stopped_wall = 0.0
        self.cancelled_credits: list[dict] = []
        self.pending_delays: set[int] = set()

    async def setup_phase(self) -> None:
        for user in self.users:
            for cid in user.assignments:
                if not self.source.get_metadata(cid).turns:
                    raise ValueError(f"Empty assigned trajectory: {cid}")

    def elapsed(self) -> float:
        return (time.perf_counter_ns() - self.lifecycle.started_at_perf_ns) / 1e9

    def schedule(self, delay: float, callback: Any, *args: Any) -> None:
        async def guarded() -> None:
            try:
                await callback(*args)
            except Exception as error:
                self.stop_reason = "error"
                self.stopped = True
                self.progress.record_fatal_error(error)
                self.changed.set()
                raise

        self.scheduler.schedule_later(delay, guarded())

    def criteria_met(self) -> bool:
        return (
            self.elapsed()
            >= self.measurement_offset + self.method.min_measurement_seconds
            and sum(user.completed for user in self.users)
            >= self.method.min_total_trajectories
            and all(
                user.completed >= self.method.min_trajectories_per_user
                for user in self.users
            )
        )

    def should_stop_user(self) -> bool:
        if self.stopped or not self.stop_checker.can_send_any_turn():
            return True
        if self.elapsed() >= self.method.phase_timeout_seconds:
            self.stopped = True
            self.stop_reason = "hard_timeout"
            self.stopped_wall = self.lifecycle.now_ns() / 1e9
            self.changed.set()
            return True
        if self.criteria_met():
            self.stopped = True
            self.stop_reason = "criteria_met"
            self.stopped_wall = self.lifecycle.now_ns() / 1e9
            self.changed.set()
            return True
        return False

    async def start_trajectory(self, user: UserState) -> None:
        if self.stopped or not self.stop_checker.can_send_any_turn():
            return
        cid = user.assignments[user.trajectory_index]
        session = self.source.session_for_conversation(cid)
        session.x_correlation_id = f"aa-u{user.user_id}-{session.x_correlation_id}"
        marker = cache_marker(
            self.run.benchmark_id, user.pass_index, user.user_id, user.trajectory_index
        )
        session.cache_bust_marker = marker
        session.cache_bust_target = CacheBustTarget.FIRST_TURN_PREFIX
        offset = 0
        if user.first:
            offset = min(
                random.Random(self.method.seed + user.user_id).randint(
                    0, self.method.max_starting_line_offset
                ),
                len(session.metadata.turns) - 1,
            )
            user.first = False
        self.sessions[session.x_correlation_id] = user
        self.session_ledger[session.x_correlation_id] = {
            "user_id": user.user_id,
            "conversation_id": cid,
            "trajectory_index": user.trajectory_index,
            "pass_index": user.pass_index,
            "start_turn_index": offset,
            "cache_bust_marker": marker,
        }
        await self.issuer.issue_credit(session.build_turn_at_index(offset))

    async def execute_phase(self) -> None:
        self.started_wall = self.lifecycle.started_at_ns / 1e9
        for user in self.users:
            self.schedule(
                user.user_id / self.method.user_spawn_rate, self.start_trajectory, user
            )
        try:
            while self.stop_checker.can_send_any_turn() and not self.stopped:
                if self.progress.fatal_error is not None:
                    raise self.progress.fatal_error
                self.changed.clear()
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self.changed.wait(), timeout=1.0)
            if self.stop_reason == "running":
                self.stop_reason = "hard_timeout"
            self.stopped = True
            self.stopped_wall = self.stopped_wall or self.lifecycle.now_ns() / 1e9
            self.issuer.stop_issuing()
            deadline = min(
                self.elapsed() + self.method.grace_period_seconds,
                self.method.phase_timeout_seconds,
            )
            while self.progress.in_flight and self.elapsed() < deadline:
                self.changed.clear()
                try:
                    await asyncio.wait_for(
                        self.changed.wait(),
                        timeout=max(0.001, deadline - self.elapsed()),
                    )
                except TimeoutError:
                    break
            if self.progress.in_flight:
                # The timing plugin API exposes issuance but no cancellation
                # operation. This single pinned-version adapter uses AIPerf's
                # own router to abort HTTP requests; it never touches a stream.
                await self.issuer._credit_router.cancel_all_credits()
                while (
                    self.progress.in_flight
                    and self.elapsed() < self.method.phase_timeout_seconds
                ):
                    self.changed.clear()
                    with contextlib.suppress(TimeoutError):
                        await asyncio.wait_for(self.changed.wait(), timeout=1.0)
            # A user sleeps after every response, including a grace-aborted
            # response. The original measurement ends only after those sleeps.
            while (
                self.pending_delays
                and self.elapsed() < self.method.phase_timeout_seconds
            ):
                self.changed.clear()
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self.changed.wait(), timeout=1.0)
        finally:
            self.stopped = True
            elapsed = self.elapsed()
            self.stopped_wall = self.stopped_wall or self.lifecycle.now_ns() / 1e9
            if self.stop_reason == "running":
                self.stop_reason = (
                    "hard_timeout"
                    if elapsed >= self.method.phase_timeout_seconds
                    else "cancelled"
                )
            self.progress.all_credits_sent_event.set()
            await self.save_state(final=False)

    async def handle_credit_result(self, credit_return: Any) -> None:
        if credit_return.cancelled:
            credit = credit_return.credit
            self.cancelled_credits.append(
                {
                    "credit_id": credit.id,
                    "x_correlation_id": credit.x_correlation_id,
                    "conversation_id": credit.conversation_id,
                    "turn_index": credit.turn_index,
                }
            )
        self.changed.set()

    async def handle_credit_return(
        self, credit: Credit, *, error: str | None = None
    ) -> None:
        user = self.sessions[credit.x_correlation_id]
        metadata = self.source.get_metadata(credit.conversation_id).turns[
            credit.turn_index
        ]
        delay = (
            0
            if self.method.no_tool_call_delay
            else (metadata.delay_ms or 0) * self.method.tool_call_delay_multiplier
        )
        self.pending_delays.add(user.user_id)
        self.schedule(delay / 1000, self.advance, user, credit)

    async def advance(self, user: UserState, credit: Credit) -> None:
        self.pending_delays.discard(user.user_id)
        self.changed.set()
        if self.should_stop_user():
            return
        if credit.is_final_turn:
            # Legacy counts attempted trajectories after their final tool delay,
            # including failed turns. Quality/error rates are reported separately.
            user.completed += 1
            self.sessions.pop(credit.x_correlation_id, None)
            user.trajectory_index += 1
            if user.trajectory_index == len(user.assignments):
                user.trajectory_index = 0
                user.pass_index += 1
            self.changed.set()
            # Legacy checks stop before reporting the completed trajectory,
            # so the user that meets the final count starts one more request.
            await self.start_trajectory(user)
        else:
            metadata = self.source.get_next_turn_metadata(credit)
            await self.issuer.issue_credit(
                TurnToSend.from_previous_credit(credit, metadata)
            )

    async def save_state(self, *, final: bool) -> None:
        state = {
            "dataset": {
                "manifest_sha256": sha256(self.manifest_path),
                "source_sha256": self.manifest["source_sha256"],
                "files": self.manifest["files"],
            },
            "schema_version": 1,
            "benchmark_id": self.run.benchmark_id,
            "engine": "aiperf",
            "methodology": self.method.model_dump(),
            "request_timeout_policy": REQUEST_TIMEOUT_POLICY,
            "concurrency": len(self.users),
            "stop_reason": self.stop_reason,
            "phase_start": self.started_wall,
            "measurement_start": self.started_wall + self.measurement_offset,
            "sending_stop": self.stopped_wall,
            "measurement_end": min(
                self.lifecycle.now_ns() / 1e9,
                self.started_wall + self.method.phase_timeout_seconds,
            ),
            "final": final,
            "completed_trajectories": {str(u.user_id): u.completed for u in self.users},
            "sessions": self.session_ledger,
            "native_counts": {
                key: getattr(self.progress.counter, key)
                for key in (
                    "requests_sent",
                    "requests_completed",
                    "requests_cancelled",
                    "request_errors",
                )
            },
            "cancelled_credits": self.cancelled_credits,
        }
        await asyncio.to_thread(
            write_json, Path(self.run.artifact_dir) / "aa_phase.json", state
        )

    async def finalize_phase(self) -> None:
        await self.save_state(final=True)
