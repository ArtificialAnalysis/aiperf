# SPDX-FileCopyrightText: Copyright (c) 2026 Artificial Analysis
# SPDX-License-Identifier: Apache-2.0

"""AA measurement-window metrics as a native AIPerf data exporter."""

import asyncio
from typing import Any

from aiperf.common.exceptions import DataExporterDisabled
from aiperf.dataset.aa_rwlt.post_process import finalize_rwlt
from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo


class RWLTExporter:
    """Run after the controller's raw-record barrier, outside the measurement window."""

    runs_after_local_exports = True

    def __init__(self, exporter_config: ExporterConfig, **kwargs: Any) -> None:
        if (
            str(getattr(exporter_config.cfg.get_default_dataset(), "dataset", ""))
            != "aa_rwlt"
        ):
            raise DataExporterDisabled("AA metrics apply only to the aa_rwlt dataset")
        if exporter_config.run is None:
            raise ValueError("AA RWLT exporter requires a benchmark run")
        self.run = exporter_config.run

    def get_export_info(self) -> FileExportInfo:
        return FileExportInfo(
            export_type="AA RWLT Metrics",
            file_path=self.run.artifact_dir / "aa_summary.json",
        )

    async def export(self) -> None:
        await asyncio.to_thread(finalize_rwlt, self.run)
