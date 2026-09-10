# SPDX-FileCopyrightText: Copyright (c) 2026 Artificial Analysis
# SPDX-License-Identifier: Apache-2.0

"""AIPerf loader: preserve each recorded full history as dispatch-only data."""

from pathlib import Path
from typing import Any

import orjson

from aiperf.common.enums import ConversationContextMode
from aiperf.common.models import Conversation, Text, Turn
from aiperf.dataset.aa_rwlt.bundle import (
    FORMAT,
    bundle_file,
    load_assignments,
    load_manifest,
)
from aiperf.dataset.loader.base_loader import BaseFileLoader
from aiperf.plugin.enums import DatasetSamplingStrategy


class RWLTLoader(BaseFileLoader):
    @classmethod
    def can_load(
        cls, data: dict | None = None, filename: str | Path | None = None
    ) -> bool:
        return isinstance(data, dict) and data.get("format") == FORMAT

    @classmethod
    def get_default_context_mode(cls) -> ConversationContextMode:
        return ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES

    @classmethod
    def get_preferred_sampling_strategy(cls) -> Any:
        return DatasetSamplingStrategy.SEQUENTIAL

    def load_dataset(self) -> dict[str, list[Any]]:
        if self.filename is None:
            raise ValueError("aa_rwlt requires a bundle manifest file")
        path = Path(self.filename)
        manifest = load_manifest(path)
        self.routing = {
            key: manifest["methodology"].get(key)
            for key in (
                "sgl_conversation_dp_size",
                "send_conversation_routing_headers",
                "use_dynamo_conv_aware_routing",
                "send_dynamo_request_id",
            )
        }
        assignments = load_assignments(bundle_file(path, manifest["assignments"]))
        needed = {cid for cids in assignments.values() for cid in cids}
        result: dict[str, list[dict]] = {}
        seen: set[tuple[str, int]] = set()
        with bundle_file(path, manifest["requests"]).open("rb") as stream:
            for line in stream:
                row = orjson.loads(line)
                cid, idx = row["conversation_id"], row["conversation_idx"]
                if cid not in needed:
                    continue
                if (cid, idx) in seen:
                    raise ValueError(f"Duplicate turn: {cid}:{idx}")
                seen.add((cid, idx))
                result.setdefault(cid, []).append(row)
        missing = needed - result.keys()
        if missing:
            raise ValueError(
                f"Assigned conversations missing from bundle: {sorted(missing)}"
            )
        for rows in result.values():
            rows.sort(key=lambda row: row["conversation_idx"])
        return result

    def convert_to_conversations(
        self, custom_data: dict[str, list[Any]]
    ) -> list[Conversation]:
        return [
            Conversation(
                session_id=cid,
                context_mode=self.get_default_context_mode(),
                turns=[
                    Turn(
                        role="user",
                        texts=[Text(contents=["aa-rwlt"])],
                        extra_body={**row["payload"], "__aa_routing": self.routing},
                        delay=row["tool_call_delay_ms"],
                        source_trace_id=cid,
                        source_outer_idx=row["conversation_idx"],
                    )
                    for row in rows
                ],
            )
            for cid, rows in custom_data.items()
        ]
