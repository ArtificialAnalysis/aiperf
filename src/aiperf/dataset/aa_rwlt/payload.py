# SPDX-FileCopyrightText: Copyright (c) 2026 Artificial Analysis
# SPDX-License-Identifier: Apache-2.0

"""Versioned wire semantics, independent of the legacy benchmark package."""

import hashlib

HARMONY_PREFIX = "You are ChatGPT, a large language model trained by OpenAI."


def is_prelude(message: dict) -> bool:
    return message.get("role") == "system" and str(
        message.get("content", "")
    ).startswith(HARMONY_PREFIX)


def normalize_messages(
    messages: list[dict], system_prompt: str | None = None
) -> list[dict]:
    result = [dict(message) for message in messages]
    prompt = str(system_prompt or "")
    if not prompt:
        return result
    for message in result:
        if message.get("role") == "system" and not is_prelude(message):
            content = str(message.get("content", ""))
            if content != prompt:
                message["content"] = f"{prompt}\n\n{content}" if content else prompt
            return result
    result.insert(
        1 if result and is_prelude(result[0]) else 0,
        {"role": "system", "content": prompt},
    )
    return result


def cache_marker(
    phase_id: str, pass_index: int, user_id: int, trajectory_index: int
) -> str:
    digest = hashlib.sha256(
        f"{phase_id}:{pass_index}:{user_id}:{trajectory_index}".encode()
    ).hexdigest()[:12]
    return f"[rid:{digest}]"


def apply_cache_marker(messages: list[dict], marker: str) -> list[dict]:
    result = [dict(message) for message in messages]
    for message in result:
        if message.get("role") == "user":
            content = str(message.get("content", ""))
            message["content"] = f"{marker}\n\n{content}" if content else marker
            return result
    result.append({"role": "user", "content": marker})
    return result


def request_body(
    row: dict,
    model: str,
    *,
    max_tokens: int = 16384,
    reasoning_effort: str | None = None,
    server_type: str | None = None,
) -> dict:
    """Match agentperf's sampling and role rules as of migration version 1."""
    name = model.lower().replace("_", "-")
    leaf = name.rsplit("/", 1)[-1]
    gpt = "gpt-oss" in name or "gptoss" in name
    deepseek = "deepseek" in name and "v4" in name
    glm = leaf in {"glm-5.2", "glm-52"}
    kimi = "kimi-k3" in name
    qwen38 = leaf == "qwen3.8-2.4t-a95b"
    messages = normalize_messages(row["messages"], row.get("system_prompt"))
    if gpt:
        messages = [
            {**m, "role": "developer"}
            if m.get("role") == "system" and not is_prelude(m)
            else m
            for m in messages
        ]
    body = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 1.0,
        "max_tokens": max_tokens,
    }
    if gpt or deepseek:
        body["top_p"] = 1.0
    elif glm or kimi or qwen38:
        body["top_p"] = 0.95
    if qwen38:
        body.update(top_k=20, min_p=0.0)
    if row.get("tools"):
        body["tools"] = row["tools"]
        if gpt and (server_type or "").lower().replace("_", "-").startswith("trtllm"):
            body["tool_choice"] = "auto"
    if reasoning_effort != "none":
        if glm:
            body["chat_template_kwargs"] = {
                "enable_thinking": True,
                "reasoning_effort": reasoning_effort or "max",
            }
        elif kimi:
            body["chat_template_kwargs"] = {
                "thinking": True,
                "thinking_effort": reasoning_effort or "max",
            }
        elif deepseek:
            body["chat_template_kwargs"] = {"thinking": True, "reasoning_effort": "max"}
        elif qwen38 or gpt or reasoning_effort is not None:
            body["reasoning_effort"] = reasoning_effort or (
                "xhigh" if qwen38 else "high"
            )
    return body
