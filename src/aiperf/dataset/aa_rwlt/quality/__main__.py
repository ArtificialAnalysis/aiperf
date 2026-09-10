# SPDX-FileCopyrightText: Copyright (c) 2026 Artificial Analysis
# SPDX-License-Identifier: Apache-2.0

"""Run the benchmark's L0/L1 quality checks over AIPerf-derived AA metrics."""

import argparse
from pathlib import Path

from aiperf.dataset.aa_rwlt.bundle import bundle_file, load_manifest
from aiperf.dataset.aa_rwlt.quality.hook import evaluate, write_reports
from aiperf.dataset.aa_rwlt.quality.levels import get_levels
from aiperf.dataset.aa_rwlt.quality.model_output import load_model_outputs
from aiperf.dataset.aa_rwlt.quality.reference import tool_vocabulary
from aiperf.dataset.aa_rwlt.quality.tokenization import get_tokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--tokenizer", default="o200k_base")
    parser.add_argument("--sample", type=int, default=512)
    args = parser.parse_args()
    if args.sample < 0:
        parser.error("--sample must be nonnegative; 0 means every turn")
    manifest = load_manifest(args.bundle)
    reference = bundle_file(args.bundle, manifest["requests"])
    metrics = args.artifacts / "aa_metrics.jsonl"
    turns = load_model_outputs(metrics, tool_vocabulary(reference))
    md, report, summary = evaluate(
        turns,
        output_path=metrics,
        reference_path=reference,
        levels=get_levels("0,1"),
        tokenizer=get_tokenizer(args.tokenizer),
        model_label=manifest["model"],
        sample=args.sample,
    )
    write_reports(md, report, args.artifacts, "aa_metrics")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
