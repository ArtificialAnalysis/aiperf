# SPDX-FileCopyrightText: Copyright (c) 2026 Artificial Analysis
# SPDX-License-Identifier: Apache-2.0

"""Uniform level interface.

Every validation level is a module exposing a module-level ``LEVEL = LevelSpec(...)``.
``run`` receives ALL successful model turns (for reference-free, model-only checks) and
the matched pairs (for reference comparisons), plus the shared EvalContext, and returns
a LevelResult. Callers run any selected subset and the report composes them generically.
"""

from collections.abc import Callable

import numpy as np
from pydantic import ConfigDict, Field

from aiperf.common.models import AIPerfBaseModel
from aiperf.dataset.aa_rwlt.quality.datatypes import (
    EvalContext,
    LevelResult,
    MatchedPair,
    ModelTurn,
)


class LevelSpec(AIPerfBaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int = Field(ge=0, description="Quality-check level number.")
    name: str = Field(description="Quality-check level name.")
    requires_reference: bool = Field(
        description="Whether this level compares to recorded responses."
    )
    run: Callable[[list[ModelTurn], list[MatchedPair], EvalContext], LevelResult] = (
        Field(description="Offline quality evaluator.")
    )
    description: str = Field(default="", description="Human-readable level purpose.")


def pct(values, p: float) -> float:
    """Linear-interpolated percentile over non-None values; NaN if empty."""
    v = [x for x in values if x is not None]
    if not v:
        return float("nan")
    return float(np.percentile(v, p))


def rate(num: int, den: int) -> str:
    return f"{num}/{den} = {100 * num / den:.1f}%" if den else "n/a"
