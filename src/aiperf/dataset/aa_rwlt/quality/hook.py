# SPDX-FileCopyrightText: Copyright (c) 2026 Artificial Analysis
# SPDX-License-Identifier: Apache-2.0

"""Post-phase eval entry point plus the evaluation core shared with the offline CLI."""

import random
import sys
from collections.abc import Callable
from pathlib import Path

import orjson

from aiperf.dataset.aa_rwlt.quality.align import build_pairs
from aiperf.dataset.aa_rwlt.quality.datatypes import EvalContext, ModelTurn, RefTurn
from aiperf.dataset.aa_rwlt.quality.levels import LevelSpec
from aiperf.dataset.aa_rwlt.quality.reference import load_reference
from aiperf.dataset.aa_rwlt.quality.report import compose


def _summary_line(jd: dict) -> str:
    parts = [f"n={jd['meta']['n_success']}"]
    l0 = jd["levels"].get("L0_hygiene")
    if l0:
        if l0["blocked_rate"] is not None:
            parts.append(f"blocked={100 * l0['blocked_rate']:.1f}%")
        if l0["valid_json_rate"] is not None:
            parts.append(f"json_ok={100 * l0['valid_json_rate']:.1f}%")
    l1 = jd["levels"].get("L1_length")
    if l1 and l1["n"]:
        parts.append(f"osl_ratio_p50={l1['ratio_p50']}")
    return " ".join(parts)


_SAMPLE_SEED = 0  # fixed: same inputs -> same sample -> reproducible reports


def sample_turns(model_turns: list[ModelTurn], k: int) -> list[ModelTurn]:
    """Deterministic sample of turns for the reference-based checks; k=0 -> all turns."""
    if not k or len(model_turns) <= k:
        return model_turns
    return random.Random(_SAMPLE_SEED).sample(model_turns, k)


def evaluate(
    model_turns: list[ModelTurn],
    *,
    output_path: Path | str,
    reference_path: Path | str | None,
    levels: list[LevelSpec],
    tokenizer: tuple[Callable[[str], int], str],
    model_label: str,
    ref_map: dict[tuple, RefTurn] | None = None,
    sample: int = 0,
) -> tuple[str, dict, str]:
    """Run the selected levels over loaded turns -> (markdown, json_dict, summary_line).

    `tokenizer` is the (tok_fn, backend_label) pair from get_tokenizer(). A preloaded
    `ref_map` (any superset of this file's keys) skips the reference load. Reference-
    requiring levels are skipped (with a stderr notice) when no reference turns loaded.

    `sample` caps the turns used for reference-based levels (loading, tokenizing, and
    pairing scale with the sample, not the phase). The eval is an endpoint sanity check,
    not exhaustive scoring — a few hundred turns is plenty. Reference-free levels (L0)
    always cover every turn; their per-turn work already happened at load time.
    """
    tok_fn, tok_backend = tokenizer
    sampled = sample_turns(model_turns, sample)
    keys = {t.key for t in sampled}
    needs_ref = any(spec.requires_reference for spec in levels)
    if ref_map is not None:
        ref_map = {
            k: v for k, v in ref_map.items() if k in keys
        }  # coverage counts stay per-file
    else:
        ref_map = (
            load_reference(reference_path, keys, tok_fn)
            if (needs_ref and reference_path)
            else {}
        )
    pairs = build_pairs(sampled, ref_map)
    ctx = EvalContext(tok_fn=tok_fn, tok_backend=tok_backend)

    results = []
    for spec in levels:
        if spec.requires_reference and not ref_map:
            print(
                f"[skip] L{spec.id} ({spec.name}): no reference loaded", file=sys.stderr
            )
            continue
        results.append(
            spec.run(sampled if spec.requires_reference else model_turns, pairs, ctx)
        )

    meta = dict(
        output_path=str(output_path),
        reference_path=str(reference_path) if reference_path else "(none)",
        model_label=model_label,
        osl_backend=tok_backend,
        levels_run=",".join(str(r.level_id) for r in results),
        n_success=len(model_turns),
        sampled=len(sampled),
        unique_keys=len(keys),
        with_reference=len(ref_map),
        matched=len(pairs),
    )
    md, jd = compose(meta, results, pairs)
    return md, jd, _summary_line(jd)


def write_reports(md: str, jd: dict, out_dir: Path, stem: str) -> Path:
    """Write <stem>__eval.md/.json into out_dir; return the JSON path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / (stem + "__eval.json")
    md_path = out_dir / (stem + "__eval.md")
    json_path.write_bytes(
        orjson.dumps(jd, option=orjson.OPT_INDENT_2, default=lambda o: None)
    )
    md_path.write_text(md, encoding="utf-8")
    return json_path
