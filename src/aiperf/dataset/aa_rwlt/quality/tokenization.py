# SPDX-FileCopyrightText: Copyright (c) 2026 Artificial Analysis
# SPDX-License-Identifier: Apache-2.0

"""Token counting for output-length (OSL) estimates.

The reference output-length estimate must be tokenized with the **evaluated model's**
tokenizer, so the model-vs-reference length ratio in L1 is apples-to-apples. The spec is:

- a tiktoken encoding name — ``o200k_base`` (default), ``o200k_harmony`` (gpt-oss), ...;
- a path to a local HF ``tokenizer.json`` (DeepSeek, GLM, ...), loaded via the fast
  ``tokenizers`` library (no network, no trust_remote_code).

A spec that cannot be loaded is a **hard error**. Substituting another vocabulary
would change the client token counts and invalidate comparisons with the model's
reference lengths and server-reported token accounting.
"""

# Legacy L1 token counts depend on the standard encoder's spaces and ASCII escapes.
import json
from collections.abc import Callable
from pathlib import Path

from aiperf.dataset.aa_rwlt.quality.datatypes import ModelTurn, RefTurn

DEFAULT_TOKENIZER = "o200k_base"


class TokenizerUnavailable(RuntimeError):
    """The configured eval tokenizer could not be loaded."""


def _tiktoken(name: str) -> tuple[Callable[[str], int], str]:
    import tiktoken

    enc = tiktoken.get_encoding(name)
    # disallowed_special=() counts a special-token literal such as "<|endoftext|>" as
    # ordinary text instead of raising. We are measuring the length of text that already
    # exists -- a prompt from the trajectory set, or a model response -- never building a
    # prompt, so there is nothing for the guard to protect. Left at its default it aborts
    # the whole run mid-sweep the first time such a literal appears in the corpus.
    return (
        lambda s: len(enc.encode(s or "", disallowed_special=()))
    ), f"tiktoken {name}"


def _hf_file(path: Path) -> tuple[Callable[[str], int], str]:
    from tokenizers import Tokenizer

    tk = Tokenizer.from_file(str(path))
    return (lambda s: len(tk.encode(s or "").ids)), f"hf tokenizers ({path})"


def get_tokenizer(spec: str = DEFAULT_TOKENIZER) -> tuple[Callable[[str], int], str]:
    """Resolve (tok_fn, backend_label) for the given spec.

    spec: a tiktoken encoding name, or a path to a tokenizer.json. Raises
    :class:`TokenizerUnavailable` if it cannot be loaded — see the module docstring for
    why this does not fall back.
    """
    try:
        if spec.endswith(".json"):
            p = Path(spec)
            if not p.is_file():
                raise FileNotFoundError(spec)
            return _hf_file(p)
        return _tiktoken(spec)
    except Exception as e:
        raise TokenizerUnavailable(
            f"eval_tokenizer {spec!r} could not be loaded ({type(e).__name__}: {e}). "
            "A different vocabulary would invalidate client token accounting. Fix the spec: "
            "a tiktoken encoding name (o200k_base, o200k_harmony) or an absolute path to "
            "the evaluated model's own tokenizer.json."
        ) from e


def osl_estimate(turn: RefTurn | ModelTurn, tok_fn: Callable[[str], int]) -> int:
    """Estimate a turn's output length: reasoning + content + serialized tool calls.

    Works on both reference turns and model turns (same channel shape), so L1 can
    compare a client-side reconstruction against server-reported token counts.
    """
    serialized = (turn.reasoning or "") + (turn.content or "")
    for tc in turn.tool_calls:
        serialized += (tc.name or "") + json.dumps(tc.args or {})
    return tok_fn(serialized)


def client_osl_estimate(turn: ModelTurn, tok_fn: Callable[[str], int]) -> int:
    """Client-side reconstruction of a model turn's output length.

    Structured turns serialize like the reference estimate (content is narration only;
    tool calls live in their own channel). Text-source turns already carry the tool
    call inline in content, so serializing it again would double-count.
    """
    if turn.source == "structured":
        return osl_estimate(turn, tok_fn)
    return tok_fn((turn.reasoning or "") + (turn.content or ""))
