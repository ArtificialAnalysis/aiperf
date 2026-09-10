# SPDX-FileCopyrightText: Copyright (c) 2026 Artificial Analysis
# SPDX-License-Identifier: Apache-2.0

"""Build matched (model, reference) pairs.

Each model record is one replay of a conversation turn; a phase replays turns many
times across users. We keep every replay as its own pair (vs the single reference for
its key) so downstream metrics are distributions, not single points.
"""

from aiperf.dataset.aa_rwlt.quality.datatypes import MatchedPair, ModelTurn, RefTurn


def build_pairs(
    model_turns: list[ModelTurn], ref_map: dict[tuple, RefTurn]
) -> list[MatchedPair]:
    return [
        MatchedPair(key=t.key, model=t, ref=ref_map[t.key])
        for t in model_turns
        if t.key in ref_map
    ]
