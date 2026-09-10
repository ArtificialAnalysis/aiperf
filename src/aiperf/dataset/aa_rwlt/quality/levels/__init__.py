# SPDX-FileCopyrightText: Copyright (c) 2026 Artificial Analysis
# SPDX-License-Identifier: Apache-2.0

"""Ordered registry of validation levels (L0 hygiene -> L1 length)."""

from aiperf.dataset.aa_rwlt.quality.levels import level0_hygiene, level1_length
from aiperf.dataset.aa_rwlt.quality.levels.base import LevelSpec, pct, rate

# id -> LevelSpec, in increasing order of depth/cost.
REGISTRY = {m.LEVEL.id: m.LEVEL for m in (level0_hygiene, level1_length)}

DEFAULT_LEVELS = "0,1"


def get_levels(spec: str) -> list[LevelSpec]:
    """Parse a comma list like '0,1' into ordered LevelSpecs."""
    ids = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok not in (str(i) for i in REGISTRY):
            raise ValueError(f"unknown level {tok!r}; available: {sorted(REGISTRY)}")
        ids.append(int(tok))
    return [REGISTRY[i] for i in sorted(set(ids))]


__all__ = ["REGISTRY", "DEFAULT_LEVELS", "get_levels", "LevelSpec", "pct", "rate"]
