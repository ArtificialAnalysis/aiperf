# SPDX-FileCopyrightText: Copyright (c) 2026 Artificial Analysis
# SPDX-License-Identifier: Apache-2.0

"""Portable workload bundle validation and integrity metadata."""

import hashlib
from pathlib import Path

import orjson
from pydantic import ConfigDict, Field, model_validator

from aiperf.common.models import AIPerfBaseModel

FORMAT = "aa-rwlt-aiperf-v1"


class Methodology(AIPerfBaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    seed: int = Field(
        default=42,
        ge=0,
        description="Seed for deterministic per-user starting offsets.",
    )
    max_starting_line_offset: int = Field(
        default=10, ge=0, description="Largest initial recorded-turn offset."
    )
    user_spawn_rate: float = Field(
        default=1.0, gt=0, description="Users started per second during ramp-up."
    )
    settling_time_seconds: float = Field(
        default=5,
        ge=0,
        description="Delay after ramp-up before the measurement window.",
    )
    min_measurement_seconds: float = Field(
        default=600, ge=60, description="Minimum steady-state measurement duration."
    )
    min_total_trajectories: int = Field(
        default=30, ge=0, description="Minimum completed trajectories across all users."
    )
    min_trajectories_per_user: int = Field(
        default=3, ge=0, description="Minimum completed trajectories for every user."
    )
    phase_timeout_seconds: float = Field(
        default=3600,
        gt=0,
        description="Hard wall-clock limit including ramp and drain.",
    )
    grace_period_seconds: float = Field(
        default=30,
        ge=0,
        description="Maximum drain after the sampling criteria are met.",
    )
    request_timeout_seconds: float = Field(
        default=300,
        ge=1,
        description="Connect and read-inactivity timeout; no total healthy-stream duration limit",
    )
    tool_call_delay_multiplier: float = Field(
        default=1.0, ge=0, description="Scale recorded post-response tool delays."
    )
    no_tool_call_delay: bool = Field(
        default=False, description="Disable recorded post-response tool delays."
    )
    sgl_conversation_dp_size: int | None = Field(
        default=None,
        ge=1,
        description="SGLang data parallel ranks for deterministic user routing.",
    )
    send_conversation_routing_headers: bool = Field(
        default=False,
        description="Send the recorded-workload conversation routing headers.",
    )
    use_dynamo_conv_aware_routing: bool = Field(
        default=False, description="Send Dynamo conversation-affinity routing headers."
    )
    send_dynamo_request_id: bool = Field(
        default=False,
        description="Preserve a unique native request ID for Dynamo tracing.",
    )

    @model_validator(mode="after")
    def check_window(self) -> "Methodology":
        if (
            self.phase_timeout_seconds
            < self.settling_time_seconds + self.min_measurement_seconds
        ):
            raise ValueError("phase timeout must accommodate settling and measurement")
        return self


def read_json(path: Path) -> dict:
    return orjson.loads(path.read_bytes())


def write_json(path: Path, value: object) -> None:
    path.write_bytes(orjson.dumps(value, option=orjson.OPT_INDENT_2) + b"\n")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bundle_file(manifest_path: Path, relative: str) -> Path:
    root = manifest_path.resolve().parent
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("bundle files must be inside the bundle directory")
    return path


def load_manifest(path: Path, *, verify: bool = True) -> dict:
    manifest = read_json(path)
    if manifest.get("format") != FORMAT:
        raise ValueError(f"Expected {FORMAT!r} bundle: {path}")
    Methodology.model_validate(manifest["methodology"])
    for key in ("requests", "assignments"):
        relative = manifest[key]
        file = bundle_file(path, relative)
        digest = manifest.get("files", {}).get(relative)
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"Missing SHA-256 for bundle file: {relative}")
        if verify and sha256(file) != digest:
            raise ValueError(f"SHA-256 mismatch for bundle file: {relative}")
    if (path.parent / "INCOMPLETE").exists():
        raise ValueError("Incomplete workload bundle")
    return manifest


def load_assignments(path: Path) -> dict[int, list[str]]:
    data = read_json(path)
    raw = data["assignments"]
    if not isinstance(raw, dict) or any(not isinstance(v, list) for v in raw.values()):
        raise ValueError("Assignments must map user IDs to lists of conversation IDs")
    result = {int(k) - 1: list(v) for k, v in raw.items()}
    if len(result) != len(raw):
        raise ValueError("Duplicate assignment user IDs after integer normalization")
    if sorted(result) != list(range(len(result))):
        raise ValueError("Assignment user IDs must be contiguous and 1-indexed")
    if not result or any(
        not v or any(not isinstance(cid, str) for cid in v) for v in result.values()
    ):
        raise ValueError("Every user must have a nonempty list of conversation IDs")
    return result
