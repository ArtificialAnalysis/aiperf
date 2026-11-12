# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inter-turn delay clamping + a sanity warning for absurd uncapped delays.

The clamp is opt-in (``--inter-turn-delay-cap-seconds``). Two modes
(``--inter-turn-delay-cap-mode``): ``hard`` (default) clamps with
``min(delay, cap)``; ``tanh`` maps every delay through ``cap * tanh(delay / cap)``,
which is close to the identity for delays well below the cap, bends smoothly,
and approaches but never exceeds the cap. The sanity warning
is always-on: when an authored delay exceeds :data:`UNCAPPED_DELAY_WARN_MS`
and no cap is set, the loader emits a single warning at load time so a
typo doesn't silently hang the run for hours/days.
"""

from __future__ import annotations

import math
from typing import Literal

from aiperf.common.aiperf_logger import AIPerfLogger

DelayCapMode = Literal["hard", "tanh"]

# Threshold (ms) above which an UNCAPPED delay triggers a load-time warning.
# Chosen at 60s: anything longer than a minute almost certainly isn't what
# the author meant on a per-turn basis, and silently waiting that long when
# the cap is unset is a known footgun (multi-day "hangs" from typos).
UNCAPPED_DELAY_WARN_MS: float = 60_000.0


def soft_cap_ms(delay_ms: float, cap_ms: float) -> float:
    """Smoothly clip a non-negative delay to below ``cap_ms``: ``cap * tanh(delay / cap)``.

    Slope 1 at zero (short delays are almost unchanged: 0.1 x cap loses 0.3%),
    0.76 x cap at the cap, and it approaches but never exceeds ``cap_ms``. A
    zero cap is the limit of the curve and maps every delay to 0.
    """
    if cap_ms <= 0:
        return 0.0
    return cap_ms * math.tanh(delay_ms / cap_ms)


def clamp_inter_turn_delay_ms(
    delay_ms: float | None,
    cap_seconds: float | None,
    mode: DelayCapMode = "hard",
) -> float | None:
    """Clamp ``delay_ms`` to at most ``cap_seconds * 1000`` ms.

    ``mode="hard"`` returns ``min(delay, cap)``; ``mode="tanh"`` returns
    :func:`soft_cap_ms`. Returns the input unchanged when either value is
    ``None``. Negative values pass through unchanged. Non-finite values
    (NaN / ±Inf) map to ``None`` (absent delay), matching timestamp scrubbing
    elsewhere.
    """
    if delay_ms is None:
        return None
    if not math.isfinite(delay_ms):
        return None
    if cap_seconds is None or delay_ms < 0:
        return delay_ms
    cap_ms = cap_seconds * 1000.0
    if mode == "tanh":
        return soft_cap_ms(delay_ms, cap_ms)
    if delay_ms > cap_ms:
        return cap_ms
    return delay_ms


class DelayCapTracker:
    """Per-loader counter that clamps inter-turn delays and logs a summary.

    Note: not a settable cap value — a stateful clamp+counter+summary helper.

    Subscribers call :meth:`clamp` on every per-turn delay value (ms or
    ``None``); the tracker returns the clamped value, increments the
    capped-count when clamping actually fires, and records the largest
    pre-clamp delay seen. Loaders call :meth:`log_summary` once after a
    load completes to emit a single info-level summary if any clamp
    happened, or a warning if an absurd delay slipped through with no cap.
    """

    __slots__ = (
        "cap_seconds",
        "capped_count",
        "max_observed_ms",
        "mode",
        "non_finite_count",
        "removed_ms",
        "uncapped_huge_count",
    )

    def __init__(self, cap_seconds: float | None, mode: DelayCapMode = "hard") -> None:
        """Initialize with an optional cap (seconds); ``None`` disables clamping.

        ``mode`` picks ``hard`` (``min``) or ``tanh`` (smooth) clipping.
        """
        if mode not in ("hard", "tanh"):
            raise ValueError(
                f"inter-turn delay cap mode must be 'hard' or 'tanh', got {mode!r}"
            )
        self.cap_seconds = cap_seconds
        self.mode = mode
        self.capped_count = 0
        self.max_observed_ms = 0.0
        self.non_finite_count = 0
        self.removed_ms = 0.0
        self.uncapped_huge_count = 0

    def clamp(self, delay_ms: float | None) -> float | None:
        """Return ``delay_ms`` clamped to the cap, updating counters.

        Returns ``None`` when ``delay_ms`` is ``None``. Non-finite values
        (NaN / ±Inf) map to ``None`` and increment ``non_finite_count`` so
        :meth:`log_summary` can warn. When ``cap_seconds`` is ``None`` the
        input passes through unchanged but ``max_observed_ms`` and
        ``uncapped_huge_count`` still update so :meth:`log_summary` can
        warn about absurd uncapped delays. Negative values pass through
        unchanged (matching :func:`clamp_inter_turn_delay_ms`) and do not
        count toward any tracker.
        """
        if delay_ms is None:
            return None
        if not math.isfinite(delay_ms):
            self.non_finite_count += 1
            return None
        if delay_ms < 0:
            return delay_ms
        if delay_ms > self.max_observed_ms:
            self.max_observed_ms = float(delay_ms)
        if self.cap_seconds is None:
            if delay_ms > UNCAPPED_DELAY_WARN_MS:
                self.uncapped_huge_count += 1
            return delay_ms
        cap_ms = self.cap_seconds * 1000.0
        if delay_ms > cap_ms:
            self.capped_count += 1
        if self.mode == "tanh":
            out = soft_cap_ms(delay_ms, cap_ms)
        else:
            out = cap_ms if delay_ms > cap_ms else delay_ms
        self.removed_ms += delay_ms - out
        return out

    def reset(self) -> None:
        """Zero per-load counters (cap value and mode untouched)."""
        self.capped_count = 0
        self.max_observed_ms = 0.0
        self.non_finite_count = 0
        self.removed_ms = 0.0
        self.uncapped_huge_count = 0

    def log_summary(self, *, logger_name: str) -> None:
        """Emit one log line if either the clamp fired or a huge uncapped
        delay was observed; otherwise no-op."""
        log = AIPerfLogger(logger_name)
        if self.non_finite_count > 0:
            log.warning(
                f"Scrubbed {self.non_finite_count:,} non-finite inter-turn "
                f"delay(s) to None (NaN/Inf)"
            )
        if self.cap_seconds is not None and self.mode == "tanh" and self.removed_ms > 0:
            log.info(
                f"Soft-capped inter-turn delays with {self.cap_seconds}s * tanh(delay / "
                f"{self.cap_seconds}s): {self.capped_count:,} exceeded the cap, "
                f"{self.removed_ms / 1000.0:,.1f} s of delay removed in total "
                f"(max observed: {self.max_observed_ms:,.1f} ms)"
            )
            return
        if self.cap_seconds is not None and self.capped_count > 0:
            log.info(
                f"Capped {self.capped_count:,} inter-turn delays exceeding "
                f"{self.cap_seconds}s (max observed: {self.max_observed_ms:,.1f} ms)"
            )
            return
        if self.cap_seconds is None and self.uncapped_huge_count > 0:
            log.warning(
                f"Authored inter-turn delay exceeds {UNCAPPED_DELAY_WARN_MS / 1000.0:.0f}s "
                f"on {self.uncapped_huge_count:,} turn(s) (max observed: "
                f"{self.max_observed_ms:,.1f} ms). The benchmark will sleep for "
                f"that long before dispatching. Pass --inter-turn-delay-cap-seconds "
                f"to clamp, or fix the authored delay."
            )
