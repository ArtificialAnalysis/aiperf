# SPDX-FileCopyrightText: Copyright (c) 2026 Artificial Analysis
# SPDX-License-Identifier: Apache-2.0

"""Native CLI selection, configuration and public loader coverage."""

from pathlib import Path

import pytest

from aiperf.common.enums import ConversationContextMode
from aiperf.config import AIPerfConfig, BenchmarkRun
from aiperf.config.flags import CLIConfig
from aiperf.config.flags.resolver import resolve_config
from aiperf.dataset.aa_rwlt.bundle import write_json
from aiperf.dataset.aa_rwlt.public import RWLTPublicDatasetLoader
from aiperf.dataset.composer.public import PublicDatasetComposer
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType


def cli(bundle: Path, **kwargs) -> CLIConfig:
    return CLIConfig(
        model_names=["fixture"],
        public_dataset="aa_rwlt",
        rwlt_bundle=bundle,
        rwlt_metrics_tokenizer="o200k_base",
        concurrency=2,
        **kwargs,
    )


def test_native_cli_sets_methodology_and_roundtrips(bundle: Path) -> None:
    config = resolve_config(cli(bundle))
    config = AIPerfConfig.model_validate_json(config.model_dump_json())
    assert config.benchmark.endpoint.type == "aa_rwlt_chat"
    assert config.benchmark.endpoint.transport == "aa_rwlt_http"
    assert config.benchmark.endpoint.streaming
    phase = config.benchmark.phases[0]
    assert phase.timing_mode == "aa_rwlt"
    assert phase.duration == 80
    assert phase.requests is None
    assert phase.sessions is None
    assert phase.grace_period == 0
    assert config.benchmark.artifacts.raw
    assert not config.benchmark.gpu_telemetry.enabled
    assert not config.benchmark.server_metrics.enabled
    assert (
        plugins.get_class(PluginType.PUBLIC_DATASET_LOADER, "aa_rwlt")
        is RWLTPublicDatasetLoader
    )


@pytest.mark.asyncio
async def test_public_composer_preserves_full_histories(
    bundle: Path, tmp_path: Path
) -> None:
    cfg = resolve_config(cli(bundle)).benchmark
    run = BenchmarkRun(benchmark_id="rwlt-test", cfg=cfg, artifact_dir=tmp_path)
    conversations = await PublicDatasetComposer(
        run=run, tokenizer=None
    ).create_dataset_async()
    assert len(conversations) == 3
    first = next(c for c in conversations if c.session_id == "fixture-0")
    assert first.context_mode == ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES
    assert [t.delay for t in first.turns] == [0, 25, 50]
    assert first.turns[1].extra_body["messages"][-1]["role"] == "tool"


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"request_count": 10}, "entries|requests"),
        ({"benchmark_duration": 10}, "duration"),
        ({"streaming": False}, "streaming"),
        ({"endpoint_type": "chat"}, "type"),
        ({"transport": "http"}, "transport"),
        ({"timeout_seconds": 5}, "timeout"),
        ({"extra_inputs": [("temperature", 0)]}, "extra"),
        ({"conversation_num_dataset_entries": 1}, "entries"),
        ({"ignore_trace_delays": True}, "delay"),
    ],
)
def test_conflicting_cli_flags_fail(bundle: Path, overrides: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        resolve_config(cli(bundle, **overrides))


def test_missing_public_release_explains_local_input() -> None:
    with pytest.raises(ValueError, match="rwlt-bundle"):
        resolve_config(CLIConfig(model_names=["fixture"], public_dataset="aa_rwlt"))


def test_missing_metrics_tokenizer_rejected(bundle: Path) -> None:
    with pytest.raises(ValueError, match="rwlt-metrics-tokenizer"):
        resolve_config(
            CLIConfig(
                model_names=["fixture"], public_dataset="aa_rwlt", rwlt_bundle=bundle
            )
        )


def test_yaml_and_cli_override_apply_same_dataset_preset(
    bundle: Path, tmp_path: Path
) -> None:
    path = tmp_path / "config.json"
    write_json(
        path,
        {
            "benchmark": {
                "model": "fixture",
                "endpoint": {"url": "http://localhost:8000"},
                "dataset": {
                    "type": "public",
                    "dataset": "aa_rwlt",
                    "rwlt_bundle": str(bundle),
                    "rwlt_metrics_tokenizer": "o200k_base",
                },
                "phases": {"type": "concurrency", "concurrency": 1},
            }
        },
    )
    cfg = resolve_config(
        CLIConfig(
            config_file=path, concurrency=2, rwlt_metrics_tokenizer="o200k_harmony"
        )
    )
    assert cfg.benchmark.phases[0].concurrency == 2
    assert cfg.benchmark.datasets[0].rwlt_metrics_tokenizer == "o200k_harmony"
    assert cfg.benchmark.phases[0].duration == 80


def test_stock_dataset_defaults_unchanged() -> None:
    cfg = resolve_config(CLIConfig(model_names=["fixture"], public_dataset="sharegpt"))
    assert cfg.benchmark.phases[0].requests == 10
    assert cfg.benchmark.endpoint.transport != "aa_rwlt_http"


def test_rwlt_options_rejected_for_other_dataset(bundle: Path) -> None:
    with pytest.raises(ValueError, match="require --public-dataset aa_rwlt"):
        resolve_config(
            CLIConfig(
                model_names=["fixture"], public_dataset="sharegpt", rwlt_bundle=bundle
            )
        )


@pytest.mark.asyncio
async def test_aa_exporter_waits_for_raw_files_and_precedes_uploads(
    bundle: Path, tmp_path: Path, monkeypatch
) -> None:
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from aiperf.common.models import ProfileResults
    from aiperf.exporters.exporter_manager import ExporterManager

    order = []

    class Raw:
        def __init__(self, **kwargs):
            pass

        async def export(self):
            order.append("raw")

    class Derived(Raw):
        runs_after_local_exports = True

        async def export(self):
            assert order == ["raw"]
            order.append("derived")

    class Upload(Raw):
        is_deferred = True

        async def export(self):
            assert order == ["raw", "derived"]
            order.append("upload")

    cfg = resolve_config(cli(bundle)).benchmark
    run = BenchmarkRun(benchmark_id="rwlt-test", cfg=cfg, artifact_dir=tmp_path)
    manager = ExporterManager(
        results=ProfileResults(
            records=[],
            start_ns=0,
            end_ns=0,
            completed=0,
            was_cancelled=False,
            error_summary=[],
        ),
        run=run,
        telemetry_results=None,
    )
    monkeypatch.setattr(
        plugins,
        "iter_all",
        lambda category: [
            (SimpleNamespace(name="upload"), Upload),
            (SimpleNamespace(name="derived"), Derived),
            (SimpleNamespace(name="raw"), Raw),
        ],
    )
    monkeypatch.setattr(
        manager, "_export_phase_metric_artifacts", AsyncMock(return_value=[])
    )
    assert await manager.export_data() == []
    assert order == ["raw", "derived", "upload"]
