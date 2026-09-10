# SPDX-FileCopyrightText: Copyright (c) 2026 Artificial Analysis
# SPDX-License-Identifier: Apache-2.0

"""Fallback parser: extract narration + a structured ToolCall from raw response text.

Used when a record carries no structured tool_call entries — e.g. server types that do
not parse tool calls into streamed deltas, leaving them inline in the content channel.
Two serializations are supported (auto-detected):

1. **gpt-oss / agentperf-v2** — optional narration then `toolname{<json-args>}`, e.g.
       Let's read the file.read{"filePath":"/testbed/x.py","offset":540,"limit":200}

2. **DeepSeek DSML** — a `tool_calls` block of `invoke`/`parameter` tags, e.g.
       <｜DSML｜tool_calls><｜DSML｜invoke name="read">
       <｜DSML｜parameter name="filePath" string="true">/testbed/x.py</｜DSML｜parameter>...

3. **Raw harmony (gpt-oss)** — endpoints whose channel separation breaks (e.g. Together
   with tool history in the request) stream the model's harmony channels as plain text:
       analysis<reasoning>assistantcommentary to=functions.grep json{"pattern":...}
       assistantcommentary"<fabricated tool result>"assistantanalysis...assistantfinal<answer>
   The *first* `to=functions.*` call is the model's actual action for the turn — later
   calls follow self-fabricated tool results, since the server never stopped generation.

The *primary* tool call is first for DSML/harmony, last for the `toolname{}` form
(where narration precedes the call). `tool.name is None` => BLOCKED.
"""

# Recorded inline tool calls can have trailing prose; JSONDecoder.raw_decode preserves the benchmark parser.
import json
import re
from functools import lru_cache

from aiperf.dataset.aa_rwlt.quality.datatypes import ToolCall
from aiperf.dataset.aa_rwlt.quality.tools import TOOLS


# --- gpt-oss style: `toolname{json}` needs a vocabulary to anchor the regex.
# Preferably the dataset's own `tools` field (see reference.tool_vocabulary);
# TOOLS is the OpenCode default used as fallback.
@lru_cache(maxsize=8)
def _toolname_re(names: tuple[str, ...]) -> re.Pattern:
    return re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\{")


TOOL_RE = _toolname_re(tuple(TOOLS))

# --- raw harmony (gpt-oss channels leaked into content) ---
# channel varies (commentary/analysis) and the constrain word varies (json/code/absent):
#   "commentary to=functions.grep json{...}", "analysis to=functions.grep code{...}",
#   "commentary to=functions.grep{...}"
HARMONY_TOOL_RE = re.compile(
    r"to=functions\.([\w.-]+)[ \t]*(?:<\|constrain\|>)?[ \t]*\w*\{"
)
_HARMONY_FINAL = "assistantfinal"

# --- DeepSeek DSML style (｜ is U+FF5C fullwidth vertical bar) ---
DSML_INVOKE_RE = re.compile(r'<｜DSML｜invoke name="([^"]+)"\s*>')
DSML_PARAM_RE = re.compile(
    r'<｜DSML｜parameter name="([^"]+)"[^>]*>(.*?)</｜DSML｜parameter>', re.DOTALL
)
_DSML_TOOLCALLS = "<｜DSML｜tool_calls>"


def decode_tool_args(raw: str) -> tuple[dict | None, bool, bool]:
    """Decode a serialized tool-argument blob -> (args, json_ok, truncated).

    The L0 semantics shared by every format: args count as valid only when they parse
    to a JSON object; an unparseable blob is truncated iff it never closed its brace.
    """
    args, json_ok, truncated = None, False, False
    try:
        obj, _ = json.JSONDecoder().raw_decode(raw)
        if isinstance(obj, dict):
            args, json_ok = obj, True
    except Exception:
        truncated = "}" not in raw
    return args, json_ok, truncated


def _parse_toolname_json(
    text: str, tool_names: tuple[str, ...]
) -> tuple[str, ToolCall, bool]:
    matches = list(_toolname_re(tool_names).finditer(text))
    if not matches:
        return text.strip(), ToolCall(name=None, args=None), False
    multi = len(matches) > 1
    m = matches[-1]
    narration = text[: m.start()].strip()
    args, json_ok, truncated = decode_tool_args(text[m.end() - 1 :])  # from the '{'
    return (
        narration,
        ToolCall(name=m.group(1), args=args, json_ok=json_ok, truncated=truncated),
        multi,
    )


def _parse_dsml(text: str) -> tuple[str, ToolCall, bool] | None:
    invokes = list(DSML_INVOKE_RE.finditer(text))
    if not invokes:
        return None
    multi = len(invokes) > 1
    first = invokes[0]
    name = first.group(1)
    # narration = anything before the tool_calls block (usually empty for DeepSeek)
    cut = text.find(_DSML_TOOLCALLS)
    narration = text[:cut].strip() if cut != -1 else text[: first.start()].strip()
    block = text[first.end() : (invokes[1].start() if multi else len(text))]
    args = {pm.group(1): pm.group(2) for pm in DSML_PARAM_RE.finditer(block)}
    # an opened <parameter> with no matching close => truncated generation
    truncated = block.count("<｜DSML｜parameter") > block.count("</｜DSML｜parameter>")
    json_ok = (not truncated) and (name is not None)
    return (
        narration,
        ToolCall(name=name, args=args or None, json_ok=json_ok, truncated=truncated),
        multi,
    )


def _parse_harmony(text: str) -> tuple[str, ToolCall, bool]:
    calls = list(HARMONY_TOOL_RE.finditer(text))
    fin = text.rfind(_HARMONY_FINAL)
    narration = text[fin + len(_HARMONY_FINAL) :].strip() if fin != -1 else ""
    if not calls:
        return narration or text.strip(), ToolCall(name=None, args=None), False
    multi = len(calls) > 1
    m = calls[0]  # first call = the actual action; later ones follow fabricated results
    args, json_ok, truncated = decode_tool_args(text[m.end() - 1 :])  # from the '{'
    return (
        narration,
        ToolCall(name=m.group(1), args=args, json_ok=json_ok, truncated=truncated),
        multi,
    )


def parse_response(
    text: str, tool_names: tuple[str, ...] | None = None
) -> tuple[str, ToolCall, bool]:
    """-> (narration, ToolCall, multi_flag). Auto-detects gpt-oss/DSML/raw-harmony; BLOCKED if none.

    `tool_names` anchors the inline `toolname{` format; defaults to the OpenCode set.
    DSML and harmony are self-describing and ignore it.
    """
    if not text:
        return "", ToolCall(name=None, args=None), False
    if "DSML｜invoke" in text:
        dsml = _parse_dsml(text)
        if dsml is not None:
            return dsml
    if _HARMONY_FINAL in text or HARMONY_TOOL_RE.search(text):
        return _parse_harmony(text)
    return _parse_toolname_json(text, tool_names or tuple(TOOLS))
