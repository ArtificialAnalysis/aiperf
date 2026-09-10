# SPDX-FileCopyrightText: Copyright (c) 2026 Artificial Analysis
# SPDX-License-Identifier: Apache-2.0

"""Load the reference trajectory dataset and align reference actions to model turns.

The trajectory JSONL that drives replay doubles as the reference: record (cid, idx) for
idx >= 1 carries the previous turn's assistant action. Alignment rule: the reference
action for a model turn (cid, idx) is the assistant message at `messages[-2]` of the
(cid, idx+1) record (the recorded assistant turn, with the following tool result as
messages[-1]).

The file is large (multi-GB JSONL) but a full streaming scan is only seconds with
orjson, and the eval runs after the measurement window, so each eval simply scans once
for the conversations it needs.
"""

# Reference tool JSON uses the original benchmark decoder, including its numeric acceptance rules.
import json
from collections.abc import Callable, Iterable
from pathlib import Path

import orjson

from aiperf.dataset.aa_rwlt.quality.datatypes import RefTurn, ToolCall
from aiperf.dataset.aa_rwlt.quality.tokenization import osl_estimate


def tool_vocabulary(path: Path | str) -> tuple[str, ...] | None:
    """Tool names declared by the dataset's first record, or None if it declares none.

    Anchors the fallback text parser's `toolname{` format to the workload's actual
    tools instead of a hardcoded vocabulary (the set is constant across records).
    """
    try:
        with open(path, "rb") as fh:
            for line in fh:
                if not line.strip():
                    continue
                tools = (orjson.loads(line).get("payload") or {}).get("tools") or []
                names = tuple(t.get("function", {}).get("name") for t in tools)
                return tuple(n for n in names if n) or None
    except Exception:
        pass
    return None


def _parse_assistant(msg: dict) -> dict:
    """Extract a raw action from an assistant message."""
    tcs = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {})
        try:
            args = json.loads(fn.get("arguments", "") or "{}")
        except Exception:
            args = None
        tcs.append([fn.get("name"), args if isinstance(args, dict) else None])
    return {
        "reasoning": msg.get("reasoning_content", "") or "",
        "content": msg.get("content", "") or "",
        "tool_calls": tcs,
    }


def _raw_to_refturn(
    cid: str, idx: int, raw: dict, tok_fn: Callable[[str], int] | None
) -> RefTurn:
    tcs = [
        ToolCall(name=n, args=a, json_ok=isinstance(a, dict))
        for n, a in raw.get("tool_calls", [])
    ]
    rt = RefTurn(
        conversation_id=cid,
        conversation_idx=idx,
        reasoning=raw.get("reasoning", ""),
        content=raw.get("content", ""),
        tool_calls=tcs,
    )
    if tok_fn is not None:
        rt.osl_est = osl_estimate(rt, tok_fn)
    return rt


def load_reference(
    path: Path | str,
    keys: Iterable[tuple],
    tok_fn: Callable[[str], int] | None = None,
) -> dict[tuple, RefTurn]:
    """Stream the reference file once; return {(cid, idx): RefTurn} for the given keys."""
    keys = set(keys)
    need_cids = {c for c, _ in keys}
    if not need_cids:
        return {}

    extracted: dict[str, dict[int, dict]] = {}
    with open(path, "rb") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = orjson.loads(line)
            cid = d.get("conversation_id")
            if cid not in need_cids:
                continue
            msgs = (d.get("payload") or {}).get("messages") or []
            if len(msgs) >= 2 and msgs[-2].get("role") == "assistant":
                # record (cid, J) provides the action for model turn (cid, J-1)
                extracted.setdefault(cid, {})[d["conversation_idx"] - 1] = (
                    _parse_assistant(msgs[-2])
                )

    ref: dict[tuple, RefTurn] = {}
    for cid, idx in keys:
        raw = extracted.get(cid, {}).get(idx)
        if raw is not None:
            ref[(cid, idx)] = _raw_to_refturn(cid, idx, raw, tok_fn)
    return ref
