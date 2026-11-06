# SPDX-FileCopyrightText: Copyright (c) 2026 Artificial Analysis. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AgentPerf v2 datasets, downloaded from Hugging Face and loaded as dag_jsonl."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, ClassVar

from huggingface_hub import hf_hub_download

from aiperf.common.exceptions import DatasetLoaderError
from aiperf.common.models import Conversation
from aiperf.dataset.loader.dag_jsonl import DagJsonlLoader
from aiperf.plugin.enums import DatasetSamplingStrategy

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun


class AgentPerfV2Loader:
    """Downloads ``<hf_subset>.dag.jsonl`` from a dataset repo and loads it.

    ``hf_subset`` names one model and side, such as ``glm-5.3-flash/public``. Sessions
    are joined end to end by their turn timestamps, which the ``agentperf-v2``
    scenario reads to space lanes along the loop.
    """

    tag: ClassVar[str] = "AgentPerfV2"

    def __init__(
        self,
        *,
        run: BenchmarkRun | None = None,
        hf_dataset_name: str,
        hf_subset: str | None = None,
        **_: Any,
    ) -> None:
        if hf_subset is None:
            raise DatasetLoaderError(
                "AgentPerf v2 needs --hf-subset <model>/<side>, for example "
                "glm-5.3-flash/public."
            )
        self.run = run
        self.hf_dataset_name = hf_dataset_name
        self.filename = f"{hf_subset}.dag.jsonl"
        self._dag: DagJsonlLoader | None = None

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        return DatasetSamplingStrategy.SEQUENTIAL

    async def load_dataset(self) -> dict[str, list[Conversation]]:
        """Side effects: downloads the file into the Hugging Face cache."""
        path = await asyncio.to_thread(
            hf_hub_download,
            repo_id=self.hf_dataset_name,
            filename=self.filename,
            repo_type="dataset",
        )
        self._dag = DagJsonlLoader(filename=path, run=self.run)
        return await asyncio.to_thread(self._dag.load_dataset)

    async def convert_to_conversations(
        self, data: dict[str, list[Conversation]]
    ) -> list[Conversation]:
        return self._dag.convert_to_conversations(data)
