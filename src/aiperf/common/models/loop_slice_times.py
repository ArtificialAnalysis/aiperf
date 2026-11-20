# SPDX-FileCopyrightText: Copyright (c) 2026 Artificial Analysis. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pydantic import Field

from aiperf.common.models.base_models import AIPerfBaseModel


class LoopSliceTimes(AIPerfBaseModel):
    """Time inside each loop lane's slice that the span of its measured requests leaves out."""

    edge_ms: list[float] = Field(
        description="Per lane, the recorded pause inside its slice before its first "
        "measured request and before the slice end after its last; for a slice with "
        "no measured request, the pause inside it. Pauses replay in real time."
    )
    idle_skips: list[tuple[int, float]] = Field(
        default_factory=list,
        description="Each global idle skip as (wall clock time.time_ns(), seconds skipped).",
    )
