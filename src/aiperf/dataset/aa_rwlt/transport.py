# SPDX-FileCopyrightText: Copyright (c) 2026 Artificial Analysis
# SPDX-License-Identifier: Apache-2.0

"""Configure native AIPerf HTTP timeouts to preserve long healthy streams."""

import aiohttp

from aiperf.common.hooks import on_init
from aiperf.plugin.schema.schemas import TransportMetadata
from aiperf.transports.aiohttp_transport import AioHttpTransport


class RWLTHttpTransport(AioHttpTransport):
    derive_session_affinity_headers = False

    @on_init
    async def _configure_rwlt_timeouts(self) -> None:
        # Native lifecycle hooks run base classes first. Configure its client
        # after initialization; every request and stream method stays inherited.
        if self.aiohttp_client is None:
            raise RuntimeError("Native AIPerf HTTP client was not initialized")
        timeout = self.model_endpoint.endpoint.timeout
        if timeout is None or timeout <= 0:
            raise ValueError("RWLT requires a positive connect/read timeout")
        self.aiohttp_client.timeout = aiohttp.ClientTimeout(
            total=None,
            connect=timeout,
            sock_connect=timeout,
            sock_read=timeout,
            ceil_threshold=float("inf"),
        )

    @classmethod
    def metadata(cls) -> TransportMetadata:
        # Explicit selection only. Do not change native URL auto-detection for
        # unrelated AIPerf workloads when this package is installed.
        return TransportMetadata(transport_type="aa_rwlt_http", url_schemes=[])
