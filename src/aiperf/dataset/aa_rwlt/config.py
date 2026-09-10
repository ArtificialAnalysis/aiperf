# SPDX-FileCopyrightText: Copyright (c) 2026 Artificial Analysis
# SPDX-License-Identifier: Apache-2.0

"""Dataset-owned replay defaults, shared by native CLI and YAML configuration."""

from copy import deepcopy
from pathlib import Path
from typing import Any

from aiperf.common.path_safety import safe_read_template_path
from aiperf.dataset.aa_rwlt.bundle import Methodology, load_manifest


def _require(section: dict[str, Any], key: str, expected: Any) -> None:
    actual = section.get(key)
    if actual is not None and actual != expected:
        raise ValueError(
            f"aa_rwlt requires {key}={expected!r}; got {actual!r}. Change the workload bundle to change methodology."
        )
    section[key] = expected


def configure_rwlt(data: Any) -> Any:
    """Fill only absent settings and reject incompatible explicit overrides."""
    if not isinstance(data, dict):
        return data
    datasets = data.get("datasets", [])
    if (
        not isinstance(datasets, list)
        or len(datasets) != 1
        or not isinstance(datasets[0], dict)
    ):
        return data
    dataset = datasets[0]
    if str(dataset.get("dataset", "")) != "aa_rwlt":
        if dataset.get("rwlt_bundle") or dataset.get("rwlt_metrics_tokenizer"):
            raise ValueError(
                "--rwlt-bundle and --rwlt-metrics-tokenizer require --public-dataset aa_rwlt"
            )
        return data
    data = deepcopy(data)
    dataset = data["datasets"][0]
    path = dataset.get("rwlt_bundle")
    if not path:
        raise ValueError(
            "aa_rwlt requires --rwlt-bundle /path/to/manifest.json; public downloading is not available in this local preview"
        )
    path = Path(path).expanduser()
    if safe_read_template_path(str(path)) is None:
        raise ValueError(f"Cannot read --rwlt-bundle: {path}")
    path = path.resolve()
    manifest = load_manifest(path, verify=False)
    dataset["rwlt_bundle"] = str(path)
    method = Methodology.model_validate(manifest["methodology"])
    spec = dataset.get("rwlt_metrics_tokenizer")
    if not spec:
        raise ValueError(
            "aa_rwlt requires --rwlt-metrics-tokenizer: an exact tokenizer.json or tiktoken encoding"
        )
    if str(spec).endswith(".json"):
        tokenizer_path = Path(spec).expanduser().resolve()
        if not tokenizer_path.is_file():
            raise ValueError(f"Missing --rwlt-metrics-tokenizer: {tokenizer_path}")
        dataset["rwlt_metrics_tokenizer"] = str(tokenizer_path)
    for key in (
        "entries",
        "osl",
        "filters",
        "system_prompt",
        "system_prompt_file",
        "max_context_length",
        "inter_turn_delay_cap_seconds",
        "trace_idle_gap_cap_seconds",
    ):
        if dataset.get(key):
            raise ValueError(
                f"aa_rwlt does not support dataset {key}; prepare a different bundle"
            )
    for key in ("ignore_trace_delays", "use_think_time_only"):
        if dataset.get(key):
            raise ValueError(
                f"aa_rwlt delay controls belong in the bundle methodology, not {key}"
            )
    _require(dataset, "sampling", "sequential")
    models = data.get("models", {})
    if isinstance(models, dict):
        items = models.get("items", [])
        if len(items) != 1 or items[0].get("name") != manifest["model"]:
            raise ValueError(
                f"aa_rwlt bundle requires --model {manifest['model']!r}; prepare a new bundle for another model"
            )
    endpoint = data.setdefault("endpoint", {})
    for key, value in {
        "type": "aa_rwlt_chat",
        "transport": "aa_rwlt_http",
        "streaming": True,
        "use_server_token_count": True,
        "timeout": method.request_timeout_seconds,
    }.items():
        _require(endpoint, key, value)
    if endpoint.get("extra"):
        raise ValueError(
            "aa_rwlt request controls belong in the bundle; --extra-inputs is unsupported"
        )
    phases = data.get("phases", [])
    if len(phases) != 1 or phases[0].get("kind", "profiling") != "profiling":
        raise ValueError(
            "aa_rwlt requires one profiling phase; ramp and settling are part of its methodology"
        )
    phase = phases[0]
    for key in (
        "requests",
        "sessions",
        "users",
        "rate",
        "concurrency_ramp",
        "prefill_concurrency",
        "cancellation",
    ):
        if phase.get(key):
            raise ValueError(
                f"aa_rwlt does not support phase {key}; it owns assignment scheduling and stopping"
            )
    for key, value in {
        "type": "concurrency",
        "timing_mode": "aa_rwlt",
        "duration": method.phase_timeout_seconds,
        "grace_period": 0,
    }.items():
        _require(phase, key, value)
    concurrency = phase.setdefault("concurrency", 1)
    if (
        not isinstance(concurrency, int)
        or not 1 <= concurrency <= manifest["user_count"]
    ):
        raise ValueError("aa_rwlt concurrency must fit the bundle user assignments")
    ramp = (concurrency - 1) / method.user_spawn_rate
    if (
        ramp + method.settling_time_seconds + method.min_measurement_seconds
        > method.phase_timeout_seconds
    ):
        raise ValueError(
            "aa_rwlt phase timeout does not accommodate ramp, settling, and measurement"
        )
    artifacts = data.setdefault("artifacts", {})
    if artifacts.get("prefix"):
        raise ValueError(
            "aa_rwlt requires standard artifact filenames; omit --profile-export-prefix"
        )
    if artifacts.get("raw") is False:
        raise ValueError("aa_rwlt requires raw export for exact offline metrics")
    artifacts["raw"] = True
    data.setdefault("gpu_telemetry", {}).setdefault("enabled", False)
    data.setdefault("server_metrics", {}).setdefault("enabled", False)
    return data
