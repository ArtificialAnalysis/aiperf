# SPDX-FileCopyrightText: Copyright (c) 2026 Artificial Analysis
# SPDX-License-Identifier: Apache-2.0

"""Build ModelTurn objects from phase output — in-memory RequestMetrics or metrics JSONL.

Both paths funnel through `turn_from_record` on the record shape written by
`agentperf.runner.save_metrics_jsonl` (whose `response_text` comes from the shared
`ResponseChannels.to_record()`), so post-phase (in-memory) and offline (JSONL) evals
see identical turns.

Tool calls are taken from the structured `response_text.tool_call` channel when present;
otherwise the fallback text parser runs over the content channel (some server types leave
tool calls inline in content instead of emitting structured deltas).
"""

from typing import TYPE_CHECKING

import orjson

from aiperf.dataset.aa_rwlt.quality.datatypes import ModelTurn, ToolCall
from aiperf.dataset.aa_rwlt.quality.parse import decode_tool_args, parse_response

if TYPE_CHECKING:
    from pathlib import Path


def _structured_tool(entry: dict) -> ToolCall:
    fn = entry.get("function") or {}
    args, json_ok, truncated = decode_tool_args(fn.get("arguments") or "")
    return ToolCall(
        name=fn.get("name"), args=args, json_ok=json_ok, truncated=truncated
    )


def turn_from_record(
    d: dict, tool_names: tuple[str, ...] | None = None
) -> ModelTurn | None:
    """One metrics record -> ModelTurn, or None for records excluded from eval (failures)."""
    if not d.get("success"):
        return None
    rt = d.get("response_text") or {}
    content = rt.get("content") or ""
    tool_calls = rt.get("tool_call") or []

    if tool_calls:
        calls = [_structured_tool(entry) for entry in tool_calls]
        multi = len(calls) > 1
        source = "structured"
    else:
        # the text parser yields the primary call only; secondary calls in a multi-call
        # text response are not retained (they only set the `multi` flag)
        _, tool, multi = parse_response(content, tool_names)
        calls = [tool] if tool.name else []
        source = "text"

    return ModelTurn(
        conversation_id=d["conversation_id"],
        conversation_idx=d["conversation_idx"],
        content=content,
        reasoning=rt.get("reasoning") or "",
        server_output_tokens=d.get("server_output_tokens"),
        multi=multi,
        source=source,
        tool_calls=calls,
    )


def load_model_outputs(
    path: "Path", tool_names: tuple[str, ...] | None = None
) -> list[ModelTurn]:
    """Read a per-request metrics JSONL (save_metrics_jsonl output) into ModelTurns."""
    turns: list[ModelTurn] = []
    with open(path, "rb") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            turn = turn_from_record(orjson.loads(line), tool_names)
            if turn is not None:
                turns.append(turn)
    return turns
