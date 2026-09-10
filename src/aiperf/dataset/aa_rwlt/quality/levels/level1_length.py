# SPDX-FileCopyrightText: Copyright (c) 2026 Artificial Analysis
# SPDX-License-Identifier: Apache-2.0

"""L1 — Output-length (OSL) fidelity.

Two comparisons:

1. **Model vs reference**: model output length (server_output_tokens) vs the reference
   action's estimated output length. The basic quantitative check: is the model
   generating responses of a similar size to the reference? Flags both under- and
   over-generation.

2. **Client vs server token accounting**: the model's own output re-tokenized
   client-side (reasoning + content + serialized tool calls, same serialization as the
   reference estimate) vs the server-reported completion token count. Ratios far outside
   [0.5, 2.0] mean the stream and the server's accounting disagree — dropped SSE
   content, un-streamed reasoning tokens, or a wrong tokenizer. Small responses get a
   fixed absolute buffer since prompt/format scaffolding adds constant overhead.
"""

from aiperf.dataset.aa_rwlt.quality.datatypes import (
    EvalContext,
    LevelResult,
    MatchedPair,
    ModelTurn,
)
from aiperf.dataset.aa_rwlt.quality.levels.base import LevelSpec, pct
from aiperf.dataset.aa_rwlt.quality.tokenization import client_osl_estimate

RATIO_LO = (
    0.5  # "within reasonable bound" for both the ref-OSL and client-vs-server checks
)
RATIO_HI = 2.0
TOKEN_ABS_BUFFER = 20  # tokens; absorbs fixed overhead on small responses


def _within(client: int, server: int) -> bool:
    ratio = client / server
    return RATIO_LO <= ratio <= RATIO_HI or abs(client - server) <= TOKEN_ABS_BUFFER


def run(
    model_turns: list[ModelTurn], pairs: list[MatchedPair], ctx: EvalContext
) -> LevelResult:
    model_osls, ref_osls, ratios = [], [], []
    for p in pairs:
        mo, ro = p.model.server_output_tokens, p.ref.osl_est
        if mo and ro:
            model_osls.append(mo)
            ref_osls.append(ro)
            ratios.append(mo / ro)
    within = sum(1 for x in ratios if RATIO_LO <= x <= RATIO_HI)

    # client-side reconstruction vs server-reported completion tokens (reference-free)
    client_ratios, client_ok = [], 0
    for t in model_turns:
        if not t.server_output_tokens:
            continue
        client = client_osl_estimate(t, ctx.tok_fn)
        client_ratios.append(client / t.server_output_tokens)
        client_ok += _within(client, t.server_output_tokens)

    metrics = dict(
        n=len(ratios),
        model_osl_p50=round(pct(model_osls, 50), 1),
        ref_osl_est_p50=round(pct(ref_osls, 50), 1),
        ratio_p10=round(pct(ratios, 10), 3),
        ratio_p50=round(pct(ratios, 50), 3),
        ratio_p90=round(pct(ratios, 90), 3),
        within_0_5x_2x=round(within / len(ratios), 4) if ratios else None,
        osl_token_backend=ctx.tok_backend,
        client_token_n=len(client_ratios),
        client_token_ratio_p10=round(pct(client_ratios, 10), 3),
        client_token_ratio_p50=round(pct(client_ratios, 50), 3),
        client_token_ratio_p90=round(pct(client_ratios, 90), 3),
        client_token_within=round(client_ok / len(client_ratios), 4)
        if client_ratios
        else None,
    )
    lines = [
        f"- model OSL p50 = **{pct(model_osls, 50):.0f}** · ref-est OSL p50 = **{pct(ref_osls, 50):.0f}** "
        f"(tokenizer: `{ctx.tok_backend}`)",
        f"- length ratio model/ref: p10=**{pct(ratios, 10):.2f}** p50=**{pct(ratios, 50):.2f}** "
        f"p90=**{pct(ratios, 90):.2f}** · within 0.5–2×: **"
        + (f"{100 * within / len(ratios):.1f}%" if ratios else "n/a")
        + "**",
        f"- client-tokenized vs server-reported tokens: p10=**{pct(client_ratios, 10):.2f}** "
        f"p50=**{pct(client_ratios, 50):.2f}** p90=**{pct(client_ratios, 90):.2f}** "
        f"· within 0.5–2× (±{TOKEN_ABS_BUFFER} tok buffer): **"
        + (f"{100 * client_ok / len(client_ratios):.1f}%" if client_ratios else "n/a")
        + "**",
    ]
    return LevelResult(
        level_id=1, name="length", metrics=metrics, tables={"length": "\n".join(lines)}
    )


LEVEL = LevelSpec(
    id=1,
    name="length",
    requires_reference=True,
    run=run,
    description="Output-length (OSL) fidelity vs reference; client-vs-server token accounting",
)
