# SPDX-FileCopyrightText: Copyright (c) 2026 Artificial Analysis
# SPDX-License-Identifier: Apache-2.0

"""Convert recorded agent trajectories into a portable AIPerf workload bundle."""

import argparse
import math
from pathlib import Path

import orjson

from aiperf.dataset.aa_rwlt.bundle import (
    FORMAT,
    Methodology,
    load_assignments,
    read_json,
    sha256,
    write_json,
)
from aiperf.dataset.aa_rwlt.payload import request_body


def prepare(
    source: Path,
    assignments_path: Path,
    output: Path,
    *,
    model: str,
    users: int | None = None,
    max_tokens: int = 16384,
    reasoning_effort: str | None = None,
    server_type: str | None = None,
    methodology: Methodology | None = None,
) -> dict:
    assignments = load_assignments(assignments_path)
    if users is not None:
        if not 1 <= users <= len(assignments):
            raise ValueError("Requested users exceed available assignments")
        assignments = {i: assignments[i] for i in range(users)}
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    needed = {cid for cids in assignments.values() for cid in cids}
    output.mkdir(parents=True, exist_ok=False)
    counts: dict[str, int] = {}
    seen: set[tuple[str, int]] = set()
    try:
        with (
            source.open("rb") as stream,
            (output / "requests.jsonl").open("wb") as dest,
        ):
            for lineno, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = orjson.loads(line)
                cid, idx = row["conversation_id"], row["conversation_idx"]
                if not isinstance(cid, str) or type(idx) is not int or idx < 0:
                    raise ValueError(
                        f"{source}:{lineno}: invalid conversation ID/index"
                    )
                if cid not in needed:
                    continue
                if (cid, idx) in seen:
                    raise ValueError(f"{source}:{lineno}: duplicate turn {cid}:{idx}")
                seen.add((cid, idx))
                delay = row.get("tool_call_delay_ms")
                if delay is None:
                    delay = 0
                if (
                    isinstance(delay, bool)
                    or not isinstance(delay, int | float)
                    or not math.isfinite(delay)
                ):
                    raise ValueError(f"{source}:{lineno}: invalid tool delay")
                if int(delay) != delay or not 0 <= delay <= 10000:
                    raise ValueError(
                        f"{source}:{lineno}: tool delay must be an integer in [0, 10000] ms"
                    )
                if not isinstance(row.get("messages"), list) or not row["messages"]:
                    raise ValueError(f"{source}:{lineno}: missing messages")
                converted = {
                    "conversation_id": cid,
                    "conversation_idx": idx,
                    "tool_call_delay_ms": int(delay),
                    "payload": request_body(
                        row,
                        model,
                        max_tokens=max_tokens,
                        reasoning_effort=reasoning_effort,
                        server_type=server_type,
                    ),
                }
                dest.write(orjson.dumps(converted) + b"\n")
                counts[cid] = counts.get(cid, 0) + 1
        if missing := needed - counts.keys():
            raise ValueError(
                f"Assigned conversations missing from source: {sorted(missing)}"
            )
        write_json(
            output / "assignments.json",
            {
                "metadata": {
                    "user_count": len(assignments),
                    "source_sha256": sha256(assignments_path),
                },
                "assignments": {str(i + 1): cids for i, cids in assignments.items()},
            },
        )
        manifest = {
            "format": FORMAT,
            "requests": "requests.jsonl",
            "assignments": "assignments.json",
            "model": model,
            "max_tokens": max_tokens,
            "reasoning_effort": reasoning_effort,
            "server_type": server_type,
            "methodology": (methodology or Methodology()).model_dump(),
            "source_sha256": sha256(source),
            "user_count": len(assignments),
            "conversation_count": len(counts),
            "request_count": sum(counts.values()),
            "turn_counts": counts,
            "files": {
                name: sha256(output / name)
                for name in ("requests.jsonl", "assignments.json")
            },
        }
        # AIPerf probes the first JSONL record even for manifest-based loaders.
        # One complete JSON line avoids a misleading parse warning at startup.
        (output / "manifest.json").write_bytes(orjson.dumps(manifest) + b"\n")
        return manifest
    except Exception:
        # An incomplete directory remains inspectable; no manifest marks it valid.
        (output / "INCOMPLETE").write_text(
            "Conversion failed. Inspect inputs and use a fresh output directory.\n"
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--users", type=int)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--reasoning-effort")
    parser.add_argument("--server-type")
    parser.add_argument(
        "--methodology", type=Path, help="JSON with explicit methodology overrides"
    )
    args = parser.parse_args()
    manifest = prepare(
        args.trajectories,
        args.assignments,
        args.out,
        model=args.model,
        users=args.users,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
        server_type=args.server_type,
        methodology=Methodology.model_validate(read_json(args.methodology))
        if args.methodology
        else None,
    )
    print(
        f"Prepared {manifest['conversation_count']} trajectories, {manifest['request_count']} turns, "
        f"{manifest['user_count']} users: {args.out / 'manifest.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
