# SPDX-FileCopyrightText: Copyright (c) 2026 Artificial Analysis
# SPDX-License-Identifier: Apache-2.0

"""Shared data contracts for the eval subpackage."""

from collections.abc import Callable
from typing import Annotated

from pydantic import ConfigDict, Field

from aiperf.common.models import AIPerfBaseModel


class ToolCall(AIPerfBaseModel):
    """A single tool invocation (model-side parsed, or reference-side)."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(
        description="tool name, or None when no tool was called (BLOCKED)"
    )
    args: dict | None = Field(
        description="parsed JSON args, or None if unparseable/absent"
    )
    json_ok: bool = Field(
        default=False, description="args parsed cleanly as a JSON object"
    )
    truncated: bool = Field(
        default=False, description="tool call present but its JSON had no closing brace"
    )


class ModelTurn(AIPerfBaseModel):
    """One successful model output record (one replay of a conversation turn)."""

    model_config = ConfigDict(extra="forbid")

    conversation_id: str = Field(description="Recorded conversation identifier.")
    conversation_idx: int = Field(ge=0, description="Recorded zero-based turn index.")
    content: str = Field(description="Assistant content channel.")
    reasoning: str = Field(description="Assistant reasoning channel.")
    server_output_tokens: int | None = Field(
        ge=0, description="Completion tokens reported by the server."
    )
    multi: bool = Field(description="more than one tool call in the response")
    source: str = Field(
        description='"structured" (tool_call channel) or "text" (parsed from content)'
    )
    tool_calls: list[ToolCall] = Field(
        default_factory=list, description="all calls; [] when BLOCKED"
    )

    @property
    def key(self) -> tuple[str, int]:
        return (self.conversation_id, self.conversation_idx)

    @property
    def tool(self) -> ToolCall:
        """The primary tool call; name is None when BLOCKED."""
        return self.tool_calls[0] if self.tool_calls else ToolCall(name=None, args=None)


class RefTurn(AIPerfBaseModel):
    """The reference (ground-truth-ish) assistant action for a turn."""

    model_config = ConfigDict(extra="forbid")

    conversation_id: str = Field(description="Recorded conversation identifier.")
    conversation_idx: int = Field(ge=0, description="Recorded zero-based turn index.")
    reasoning: str = Field(description="Assistant reasoning channel.")
    content: str = Field(description="Assistant content channel.")
    tool_calls: list[ToolCall] = Field(description="Structured reference tool calls.")
    osl_est: int | None = Field(
        default=None, ge=0, description="estimated output length in tokens"
    )

    @property
    def key(self) -> tuple[str, int]:
        return (self.conversation_id, self.conversation_idx)

    @property
    def primary_tool(self) -> ToolCall:
        return self.tool_calls[0] if self.tool_calls else ToolCall(name=None, args=None)


class MatchedPair(AIPerfBaseModel):
    model_config = ConfigDict(extra="forbid")

    key: tuple[str, Annotated[int, Field(ge=0)]] = Field(
        description="Recorded conversation and turn key."
    )
    model: ModelTurn = Field(description="Generated model turn.")
    ref: RefTurn = Field(description="Recorded reference turn.")


class LevelResult(AIPerfBaseModel):
    """Uniform output of every validation level."""

    model_config = ConfigDict(extra="forbid")

    level_id: int = Field(ge=0, description="Quality-check level number.")
    name: str = Field(description="Quality-check level name.")
    metrics: dict = Field(
        default_factory=dict, description="scalar metrics (JSON-serializable)"
    )
    tables: dict = Field(
        default_factory=dict, description="name -> markdown table string"
    )


class EvalContext(AIPerfBaseModel):
    """Shared resources passed to every level."""

    model_config = ConfigDict(extra="forbid")

    tok_fn: Callable[[str], int] = Field(
        description="Exact tokenizer counting callback."
    )
    tok_backend: str = Field(description="Tokenizer backend label.")
