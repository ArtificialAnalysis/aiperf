# SPDX-FileCopyrightText: Copyright (c) 2026 Artificial Analysis
# SPDX-License-Identifier: Apache-2.0

"""Built-in public dataset entry for the local AA RWLT release preview."""

import asyncio
from typing import Any

from aiperf.common.models import Conversation
from aiperf.dataset.aa_rwlt.loader import RWLTLoader
from aiperf.dataset.loader.base_public_dataset import BasePublicDatasetLoader


class RWLTPublicDatasetLoader(BasePublicDatasetLoader):
    """Load a verified local release bundle without external plugin discovery."""

    get_default_context_mode = RWLTLoader.get_default_context_mode
    get_preferred_sampling_strategy = RWLTLoader.get_preferred_sampling_strategy

    async def load_dataset(self) -> dict[str, Any]:
        dataset = self.run.cfg.get_default_dataset()
        self._loader = RWLTLoader(filename=dataset.rwlt_bundle, run=self.run)
        return await asyncio.to_thread(self._loader.load_dataset)

    async def convert_to_conversations(
        self, dataset: dict[str, Any]
    ) -> list[Conversation]:
        return await asyncio.to_thread(self._loader.convert_to_conversations, dataset)
