# SPDX-FileCopyrightText: Copyright (c) 2026 Artificial Analysis
# SPDX-License-Identifier: Apache-2.0

"""Offline AA metrics and quality after native AIPerf services have stopped."""

from aiperf.config.resolution.plan import BenchmarkRun
from aiperf.dataset.aa_rwlt.analyze import analyze
from aiperf.dataset.aa_rwlt.bundle import bundle_file, load_manifest
from aiperf.dataset.aa_rwlt.quality.hook import evaluate, write_reports
from aiperf.dataset.aa_rwlt.quality.levels import get_levels
from aiperf.dataset.aa_rwlt.quality.model_output import load_model_outputs
from aiperf.dataset.aa_rwlt.quality.reference import tool_vocabulary
from aiperf.dataset.aa_rwlt.quality.tokenization import get_tokenizer


def finalize_rwlt(run: BenchmarkRun) -> None:
    """Write comparable measurements and quality; fail if capture or stopping is invalid."""
    dataset = run.cfg.get_default_dataset()
    stats = analyze(run.artifact_dir, dataset.rwlt_metrics_tokenizer)
    manifest = load_manifest(dataset.rwlt_bundle)
    reference = bundle_file(dataset.rwlt_bundle, manifest["requests"])
    metrics = run.artifact_dir / "aa_metrics.jsonl"
    turns = load_model_outputs(metrics, tool_vocabulary(reference))
    md, report, _ = evaluate(
        turns,
        output_path=metrics,
        reference_path=reference,
        levels=get_levels("0,1"),
        tokenizer=get_tokenizer(dataset.rwlt_metrics_tokenizer),
        model_label=manifest["model"],
        sample=512,
    )
    write_reports(md, report, run.artifact_dir, "aa_metrics")
    if not stats["valid_for_legacy_comparison"]:
        raise ValueError(
            "AA RWLT phase did not satisfy stopping/capture checks; inspect aa_summary.json"
        )
