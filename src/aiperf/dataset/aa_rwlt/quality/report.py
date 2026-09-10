# SPDX-FileCopyrightText: Copyright (c) 2026 Artificial Analysis
# SPDX-License-Identifier: Apache-2.0

"""Compose level results into a markdown + JSON report."""

from aiperf.dataset.aa_rwlt.quality.datatypes import LevelResult, MatchedPair


def worked_examples(pairs: list[MatchedPair], k: int = 8) -> list[str]:
    out = []
    for p in pairs[:k]:
        mo, ro = p.model.server_output_tokens, p.ref.osl_est
        ratio = f"{mo / ro:.2f}x" if mo and ro else "n/a"
        out.append(
            f"**cid={p.key[0]} idx={p.key[1]}** — model OSL: `{mo}` · ref-est OSL: `{ro}` · ratio: `{ratio}`"
        )
    return out


def compose(
    meta: dict, results: list[LevelResult], pairs: list[MatchedPair]
) -> tuple[str, dict]:
    """Return (markdown_str, json_dict)."""
    jd = {"meta": meta, "levels": {}}
    for r in results:
        jd["levels"][f"L{r.level_id}_{r.name}"] = r.metrics

    L = []
    L.append(
        f"# agentperf eval report — {meta.get('model_label', 'model')} vs reference trajectories\n"
    )
    L.append(f"**Output:** `{meta['output_path']}`")
    L.append(f"**Reference:** `{meta['reference_path']}`")
    L.append(f"**Backend:** OSL=`{meta['osl_backend']}`")
    L.append(f"**Levels run:** {meta['levels_run']}\n")

    L.append("## Coverage")
    sampled = meta.get("sampled", meta["n_success"])
    sample_note = f" (sampled: **{sampled}**)" if sampled < meta["n_success"] else ""
    L.append(
        f"- successful outputs: **{meta['n_success']}**{sample_note} · unique (cid,idx): **{meta['unique_keys']}** "
        f"· with reference: **{meta['with_reference']}** · matched records: **{meta['matched']}**\n"
    )

    titles = {
        0: "Layer 0 — hygiene (well-formedness)",
        1: "Layer 1 — output-length (OSL) match",
    }
    for r in results:
        L.append(f"## {titles.get(r.level_id, f'Layer {r.level_id} — {r.name}')}")
        for t in r.tables.values():
            L.append(t)
        L.append("")

    if pairs:
        L.append("## Worked examples")
        L.extend(worked_examples(pairs))
        L.append("")

    L.append("## Caveats")
    L.append("- L0 is reference-free and only checks structural well-formedness.")
    L.append(
        "- L1 compares model `server_output_tokens` to an estimated reference OSL; treat the "
        "ratio distribution as more meaningful than absolute token counts."
    )
    return "\n".join(L) + "\n", jd
