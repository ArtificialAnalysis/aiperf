# SPDX-FileCopyrightText: Copyright (c) 2026 Artificial Analysis
# SPDX-License-Identifier: Apache-2.0

"""Recompute AA measurement-window metrics from AIPerf's raw exports, offline."""

import argparse
from collections import defaultdict
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import orjson
from openai.types.chat import ChatCompletionChunk

from aiperf.dataset.aa_rwlt.bundle import Methodology, read_json, sha256, write_json
from aiperf.dataset.aa_rwlt.transport_settings import REQUEST_TIMEOUT_POLICY


def tokenizer(spec: str) -> Callable[[str], int]:
    if spec.endswith(".json"):
        from tokenizers import Tokenizer

        hf_encoder = Tokenizer.from_file(spec)
        return lambda text: len(hf_encoder.encode(text or "").ids)
    import tiktoken

    encoder = tiktoken.get_encoding(spec)
    return lambda text: len(encoder.encode(text or "", disallowed_special=()))


def iter_events(raw: dict) -> Iterator[tuple[int, dict]]:
    for response in raw.get("responses", []):
        packets = response.get("packets", [])
        data = "\n".join(p["value"] for p in packets if p.get("name") == "data")
        if not data or data == "[DONE]":
            continue
        try:
            # Use the original benchmark's offline schema validation, including
            # its tolerant skip of malformed/error-only SSE frames. No SDK
            # request or streaming code is used by this package.
            event = ChatCompletionChunk.model_validate_json(data)
        except Exception:
            continue
        yield response["perf_ns"], event.model_dump(exclude_unset=True)


def decode_record(
    raw: dict, count: Callable[[str], int], sessions: dict[str, dict]
) -> tuple[dict, list[tuple[float, int]]]:
    meta = raw["metadata"]
    session = sessions[meta["x_correlation_id"]]
    start_ns, end_ns = meta["request_start_ns"], meta["request_end_ns"]
    start = start_ns / 1e9
    chunks: list[tuple[float, int]] = []
    relative_times: list[float] = []
    content: list[str] = []
    reasoning: list[str] = []
    calls: dict[int, dict] = {}
    usage: dict = {}
    last_event: dict = {}
    finish_reason = None
    acceptance = None
    event_count = 0
    for perf_ns, event in iter_events(raw):
        event_count += 1
        last_event = event
        if event.get("usage"):
            usage = event["usage"]
        choices = event.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]
        value = choice.get("avg_decoded_tokens_per_iter")
        if isinstance(value, int | float) and np.isfinite(value) and value > 0:
            acceptance = float(value)
        delta = choice.get("delta") or {}
        text = delta.get("content") or ""
        thought = delta.get("reasoning") or delta.get("reasoning_content") or ""
        content.append(text)
        reasoning.append(thought)
        tc = delta.get("tool_calls") or []
        fragment = text + thought
        if tc:
            function = tc[0].get("function") or {}
            fragment += (function.get("name") or "") + (function.get("arguments") or "")
        for call in tc:
            index = call.get("index") or 0
            record = calls.setdefault(
                index,
                {
                    "id": None,
                    "type": "function",
                    "function": {"name": None, "arguments": ""},
                },
            )
            if call.get("id"):
                record["id"] = call["id"]
            function = call.get("function") or {}
            if function.get("name"):
                record["function"]["name"] = function["name"]
            record["function"]["arguments"] += function.get("arguments") or ""
        if fragment:
            relative = (perf_ns - raw["start_perf_ns"]) / 1e9
            relative_times.append(relative)
            chunks.append((start + relative, count(fragment)))
    local_tokens = sum(tokens for _, tokens in chunks)
    generation = relative_times[-1] - relative_times[0] if len(chunks) >= 2 else None
    first_tokens = chunks[0][1] if chunks else 0
    latency = (end_ns - start_ns) / 1e9
    # Legacy SSE speed uses the last parsed event's usage; server throughput
    # scans backwards for the latest usage independently.
    sse_tokens = (last_event.get("usage") or {}).get("completion_tokens")

    def speed(tokens: int | None, duration: float | None) -> float | None:
        return (
            tokens / duration
            if tokens is not None
            and tokens > 0
            and duration is not None
            and duration > 0
            else None
        )

    error = raw.get("error") or meta.get("error")
    status = raw.get("status")
    cancelled = bool(meta.get("was_cancelled"))
    # Legacy graceful aborts retain successfully received partial content.
    # Preserve the native cancellation error separately for auditing.
    native_error = error
    if (
        cancelled
        and isinstance(error, dict)
        and error.get("type") == "RequestCancellationError"
    ):
        error = None
    if (
        isinstance(error, dict)
        and error.get("type") == "InvalidInferenceResultError"
        and "No responses with actual content" in error.get("message", "")
    ):
        # AIPerf rejects empty 2xx streams during result processing. Legacy
        # retains transport success, then excludes them via has_output when
        # aggregating performance; quality still sees the empty response.
        error = None
    success = not error and (status is None or 200 <= status < 300)
    error_text = (
        (error.get("message") or orjson.dumps(error).decode())
        if isinstance(error, dict)
        else error
    )
    record = {
        "request_id": (raw.get("cache_bust_marker") or session["cache_bust_marker"])[
            5:-1
        ],
        "user_id": session["user_id"],
        "conversation_id": meta["conversation_id"],
        "conversation_idx": meta.get("source_outer_idx", meta["turn_index"]),
        "start_time": start,
        "end_time": end_ns / 1e9,
        "duration": latency,
        "success": success,
        "error": error_text,
        "has_output": bool(chunks),
        "cancelled": cancelled,
        "native_error": native_error,
        "native_x_request_id": meta.get("x_request_id"),
        "native_x_correlation_id": meta["x_correlation_id"],
        "ttft": relative_times[0] if chunks else None,
        "e2e_latency": latency,
        "generation_time": generation,
        "output_speed": speed(local_tokens - first_tokens, generation),
        "sse_output_speed": speed(
            sse_tokens - first_tokens if sse_tokens is not None else None, generation
        ),
        "e2e_speed": speed(local_tokens, latency),
        "sse_e2e_speed": speed(sse_tokens, latency),
        "total_tokens": local_tokens,
        "sse_total_tokens": sse_tokens,
        "server_input_tokens": usage.get("prompt_tokens"),
        "server_output_tokens": usage.get("completion_tokens"),
        "server_cached_tokens": (usage.get("prompt_tokens_details") or {}).get(
            "cached_tokens"
        ),
        "num_chunks": event_count,
        "num_content_chunks": len(chunks),
        "finish_reason": finish_reason,
        "acceptance_length": acceptance,
        "response_text": {
            "content": "".join(content),
            "reasoning": "".join(reasoning),
            "tool_call": [calls[i] for i in sorted(calls)],
        },
    }
    return record, chunks


def distribution(values: list[float]) -> dict:
    return {
        "count": len(values),
        "mean": float(np.mean(values)) if values else None,
        **{
            f"p{p}": float(np.percentile(values, p)) if values else None
            for p in (5, 10, 25, 50, 75, 95, 99)
        },
    }


def trajectory_stats(records: list[dict], k: int = 8) -> dict:
    per_user: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        if record["user_id"] < k:
            per_user[record["user_id"]].append(record)
    walls, inferences = [], []
    users = 0
    for values in per_user.values():
        visits: list[list[dict]] = []
        for record in sorted(values, key=lambda r: r["start_time"]):
            if (
                not visits
                or visits[-1][-1]["conversation_id"] != record["conversation_id"]
            ):
                visits.append([])
            visits[-1].append(record)
        if len(visits) > 1:
            users += 1
        for visit in visits[:-1]:
            walls.append(
                max(r["end_time"] for r in visit) - min(r["start_time"] for r in visit)
            )
            inferences.append(sum(r["end_time"] - r["start_time"] for r in visit))
    return {
        "avg_trajectory_wall_clock_s": float(np.mean(walls)) if walls else None,
        "avg_trajectory_inference_sum_s": float(np.mean(inferences))
        if inferences
        else None,
        "trajectory_count_in_avg": len(walls),
        "users_in_avg": users,
    }


def stop_checks(phase: dict) -> dict[str, bool]:
    method = Methodology.model_validate(phase["methodology"])
    completed = phase["completed_trajectories"]
    return {
        "user_coverage": set(completed)
        == {str(i) for i in range(phase["concurrency"])},
        "minimum_measurement": phase["sending_stop"] - phase["measurement_start"]
        >= method.min_measurement_seconds - 1e-6,
        "minimum_total_trajectories": sum(completed.values())
        >= method.min_total_trajectories,
        "minimum_trajectories_per_user": all(
            n >= method.min_trajectories_per_user for n in completed.values()
        ),
        "before_hard_timeout": phase["sending_stop"]
        <= phase["phase_start"] + method.phase_timeout_seconds,
    }


def reconcile_capture(counts: dict | None, exported: int, raw_cancelled: int) -> dict:
    if counts is None:
        return {
            "complete_capture": False,
            "omitted_native_requests": None,
            "unaccounted_native_requests": None,
            "cancelled_requests": None,
            "capture_note": "Native request counts unavailable; completeness is unverified",
        }
    for key in (
        "requests_sent",
        "requests_completed",
        "requests_cancelled",
        "request_errors",
    ):
        if type(counts.get(key)) is not int or counts[key] < 0:
            raise ValueError(f"Invalid native counter: {key}")
    sent, completed, cancelled = (
        counts[k] for k in ("requests_sent", "requests_completed", "requests_cancelled")
    )
    missing = sent - exported
    unaccounted = sent - completed - cancelled
    if missing < 0:
        raise ValueError(
            "Raw export contains more records than AIPerf sent; mixed or duplicate artifacts"
        )
    if exported < completed:
        raise ValueError("Raw export is missing completed requests")
    if unaccounted < 0 or counts["request_errors"] > completed:
        raise ValueError("Native terminal counters are inconsistent")
    complete = missing == 0 and unaccounted == 0
    note = None
    if not complete:
        note = (
            "AIPerf omits externally cancelled requests, including their partial tokens. "
            "These metrics cover exported requests only."
            if missing == cancelled and unaccounted == 0
            else "Native issued/terminal/exported counts do not reconcile. Missing requests are not assumed cancelled."
        )
    return {
        "complete_capture": complete,
        "omitted_native_requests": missing,
        "unaccounted_native_requests": unaccounted,
        "cancelled_requests": raw_cancelled + cancelled,
        "capture_note": note,
    }


def analyze(artifacts: Path, spec: str = "o200k_base") -> dict:
    phase = read_json(artifacts / "aa_phase.json")
    if not phase["final"]:
        raise ValueError(
            "AIPerf phase did not finalize; cannot publish benchmark metrics"
        )
    start, end = phase["measurement_start"], phase["measurement_end"]
    if end <= start:
        raise ValueError("No positive steady-state measurement window")
    checks = stop_checks(phase)
    valid_stop = phase["stop_reason"] == "criteria_met" and all(checks.values())
    count = tokenizer(spec)
    records: list[dict] = []
    window_tokens = 0
    raw_request_payload_count = 0
    first_chunk, last_chunk = float("inf"), float("-inf")
    request_ids: set[str] = set()
    with (
        (artifacts / "profile_export_raw.jsonl").open("rb") as stream,
        (artifacts / "aa_metrics.jsonl").open("wb") as out,
    ):
        for line in stream:
            raw = orjson.loads(line)
            raw_request_payload_count += raw.get("payload") is not None
            request_id = raw["metadata"]["x_request_id"]
            if request_id in request_ids:
                raise ValueError(f"Duplicate native request ID: {request_id}")
            request_ids.add(request_id)
            record, chunks = decode_record(raw, count, phase["sessions"])
            out.write(orjson.dumps(record) + b"\n")
            # Keep only scalar fields in memory; large outputs stay on disk.
            records.append({k: v for k, v in record.items() if k != "response_text"})
            if record["success"]:
                for timestamp, tokens in chunks:
                    if start <= timestamp < end:
                        window_tokens += tokens
                        first_chunk = min(first_chunk, timestamp)
                        last_chunk = max(last_chunk, timestamp)
    measured = [r for r in records if start <= r["start_time"] < end]
    successful = [r for r in measured if r["success"] and r["num_content_chunks"]]
    duration = end - start
    server_tokens = sum(r["server_output_tokens"] or 0 for r in successful)
    native_counts = phase.get("native_counts")
    capture = reconcile_capture(
        native_counts, len(records), sum(r["cancelled"] for r in records)
    )
    timeout_policy_verified = (
        phase.get("request_timeout_policy") == REQUEST_TIMEOUT_POLICY
    )
    stats = {
        "schema_version": 1,
        "engine": "aiperf",
        "metric_definition": "aa-agentperf-v1",
        "analysis_source_sha256": sha256(Path(__file__)),
        "native_raw_sha256": sha256(artifacts / "profile_export_raw.jsonl"),
        "phase_state_sha256": sha256(artifacts / "aa_phase.json"),
        "tokenizer": spec,
        "tokenizer_sha256": sha256(Path(spec)) if Path(spec).is_file() else None,
        "concurrency": phase["concurrency"],
        "stop_reason": phase["stop_reason"],
        "valid_stop": valid_stop,
        "stop_checks": checks,
        **capture,
        "request_timeout_policy": phase.get("request_timeout_policy"),
        "timeout_policy_verified": timeout_policy_verified,
        "compatibility_note": (
            None
            if timeout_policy_verified
            else "Connect/read-inactivity timeout policy is unverified; older runs used a total-duration timeout."
        ),
        "valid_for_legacy_comparison": valid_stop
        and capture["complete_capture"]
        and timeout_policy_verified,
        "native_counts": native_counts,
        "measurement_duration": duration,
        "total_requests": len(records),
        "raw_request_payload_count": raw_request_payload_count,
        "request_payloads_retained": bool(records)
        and raw_request_payload_count == len(records),
        "measured_requests": len(measured),
        "successful_requests": len(successful),
        "failed_requests": len(measured) - len(successful),
        "measured_cancelled_records": sum(r["cancelled"] for r in measured),
        "system_throughput": window_tokens / (last_chunk - first_chunk)
        if last_chunk > first_chunk
        else None,
        "server_output_throughput": server_tokens / duration if server_tokens else None,
        "server_output_tokens": server_tokens,
        "trajectory_stats": trajectory_stats(records),
        "distributions": {
            key: distribution([r[key] for r in successful if r[key] is not None])
            for key in (
                "ttft",
                "e2e_latency",
                "output_speed",
                "sse_output_speed",
                "e2e_speed",
                "sse_e2e_speed",
                "acceptance_length",
            )
        },
        "missing_server_usage": sum(
            r["server_output_tokens"] is None for r in successful
        ),
    }
    write_json(artifacts / "aa_summary.json", stats)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--tokenizer", default="o200k_base")
    args = parser.parse_args()
    stats = analyze(args.artifacts, args.tokenizer)
    print(orjson.dumps(stats, option=orjson.OPT_INDENT_2).decode())
    return 0 if stats["valid_for_legacy_comparison"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
