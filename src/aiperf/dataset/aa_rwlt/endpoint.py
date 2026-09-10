# SPDX-FileCopyrightText: Copyright (c) 2026 Artificial Analysis
# SPDX-License-Identifier: Apache-2.0

"""Request construction only; streaming and response parsing remain AIPerf's."""

from typing import Any

from aiperf.common.models import ModelEndpointInfo, RequestInfo
from aiperf.dataset.aa_rwlt.payload import apply_cache_marker
from aiperf.endpoints.openai_chat import ChatEndpoint


class RWLTChatEndpoint(ChatEndpoint):
    def __init__(self, model_endpoint: ModelEndpointInfo, **kwargs: Any) -> None:
        if str(model_endpoint.transport) != "aa_rwlt_http":
            raise ValueError(
                "aa_rwlt_chat requires aa_rwlt_http transport for connect/read timeout semantics"
            )
        super().__init__(model_endpoint, **kwargs)

    def format_payload(self, request_info: RequestInfo) -> dict:
        body = dict(request_info.turns[-1].extra_body or {})
        routing = body.pop("__aa_routing", {})
        if not isinstance(body.get("messages"), list):
            raise ValueError(
                "aa_rwlt_chat requires an aa_rwlt bundle with full recorded request bodies"
            )
        if request_info.cache_bust_marker:
            body["messages"] = apply_cache_marker(
                body["messages"], request_info.cache_bust_marker
            )
            if routing.get("sgl_conversation_dp_size"):
                user_id = int(request_info.x_correlation_id.split("-", 2)[1][1:])
                body["routed_dp_rank"] = user_id % routing["sgl_conversation_dp_size"]
        if body["model"] != request_info.model_endpoint.primary_model_name:
            raise ValueError(
                "AIPerf model differs from the bundle; regenerate the bundle for the requested model"
            )
        if request_info.model_endpoint.endpoint.extra:
            raise ValueError(
                "AIPerf extra inputs would change the benchmark request contract; regenerate the bundle"
            )
        return body

    def get_endpoint_headers(self, request_info: RequestInfo) -> dict[str, str]:
        headers = super().get_endpoint_headers(request_info)
        marker = request_info.cache_bust_marker
        if not marker or not request_info.turns:
            return headers
        routing = (request_info.turns[-1].extra_body or {}).get("__aa_routing", {})
        request_id = marker[5:-1]
        cid = request_info.conversation_id
        idx = request_info.turns[-1].source_outer_idx
        routing_id = f"{cid}:{request_id}"
        if routing.get("send_conversation_routing_headers"):
            headers.update(
                {
                    "X-SMG-Routing-Key": routing_id,
                    "X-Session-ID": routing_id,
                    "X-Correlation-ID": routing_id,
                    "X-AGENTPERF-Conversation-ID": cid,
                    "X-AGENTPERF-Conversation-Idx": str(idx),
                    "X-AGENTPERF-Session-ID": request_id,
                    "X-AGENTPERF-Request-ID": f"{routing_id}:{idx}",
                }
            )
        if routing.get("use_dynamo_conv_aware_routing"):
            headers["X-Dynamo-Session-ID"] = routing_id
        # AIPerf already supplies a unique X-Request-ID per request. Retain it
        # rather than generating a second correlation ID for Dynamo traces.
        return headers
