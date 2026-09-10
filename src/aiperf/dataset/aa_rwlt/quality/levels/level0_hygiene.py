# SPDX-FileCopyrightText: Copyright (c) 2026 Artificial Analysis
# SPDX-License-Identifier: Apache-2.0

"""L0 — Hygiene / well-formedness (reference-free).

Model-only sanity over all successful outputs: how many are tool calls vs pure-text
(BLOCKED), valid-JSON-arg rate, truncated/malformed calls, multi-tool rate. Also
reports how many turns came from the structured tool_call channel vs the fallback
text parser (`structured_rate`), so a low tool-call rate can be told apart from a
server that simply doesn't emit structured tool calls.
"""

from aiperf.dataset.aa_rwlt.quality.datatypes import (
    EvalContext,
    LevelResult,
    MatchedPair,
    ModelTurn,
)
from aiperf.dataset.aa_rwlt.quality.levels.base import LevelSpec, rate


def run(
    model_turns: list[ModelTurn], pairs: list[MatchedPair], ctx: EvalContext
) -> LevelResult:
    n = len(model_turns)
    n_tool = sum(1 for t in model_turns if t.tool.name)
    n_text = n - n_tool
    n_jsonbad = sum(1 for t in model_turns if t.tool.name and not t.tool.json_ok)
    n_trunc = sum(1 for t in model_turns if t.tool.name and t.tool.truncated)
    n_multi = sum(1 for t in model_turns if t.tool.name and t.multi)
    n_structured = sum(1 for t in model_turns if t.source == "structured")

    metrics = dict(
        total_success=n,
        tool_calls=n_tool,
        pure_text_blocked=n_text,
        json_invalid=n_jsonbad,
        truncated=n_trunc,
        multi_tool=n_multi,
        valid_json_rate=round(1 - n_jsonbad / n_tool, 4) if n_tool else None,
        blocked_rate=round(n_text / n, 4) if n else None,
        structured_rate=round(n_structured / n, 4) if n else None,
    )
    table = "| metric | value |\n|---|---|\n"
    table += f"| tool-call responses | {rate(n_tool, n)} |\n"
    table += f"| pure-text / BLOCKED | {rate(n_text, n)} |\n"
    table += (
        f"| valid-JSON args (of tool calls) | {rate(n_tool - n_jsonbad, n_tool)} |\n"
    )
    table += f"| truncated/malformed calls | {rate(n_trunc, n_tool)} |\n"
    table += f"| multi-tool responses | {rate(n_multi, n_tool)} |\n"
    table += f"| structured tool_call channel | {rate(n_structured, n)} |\n"
    return LevelResult(
        level_id=0, name="hygiene", metrics=metrics, tables={"hygiene": table}
    )


LEVEL = LevelSpec(
    id=0,
    name="hygiene",
    requires_reference=False,
    run=run,
    description="Well-formedness: JSON validity, truncation, blocked & multi-tool rates",
)
