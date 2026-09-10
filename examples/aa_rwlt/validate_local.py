# SPDX-FileCopyrightText: Copyright (c) 2026 Artificial Analysis
# SPDX-License-Identifier: Apache-2.0

"""Run native AIPerf workers against a deterministic localhost SSE server.

From the fork: uv run python examples/aa_rwlt/validate_local.py --out /tmp/rwlt-smoke
The default check takes about 70 seconds and requires no GPU or model download.
"""

import argparse
import asyncio
import json
import os
import re
import signal
import sys
import time
from pathlib import Path

import orjson
from aiohttp import web
from tokenizers import Tokenizer, models, pre_tokenizers

from aiperf.dataset.aa_rwlt.bundle import Methodology, read_json, write_json
from aiperf.dataset.aa_rwlt.payload import apply_cache_marker
from aiperf.dataset.aa_rwlt.prepare import prepare


async def validate(args: argparse.Namespace) -> dict:
    args.out.mkdir(parents=True, exist_ok=False)
    fixtures = (
        Path(__file__).resolve().parents[2]
        / "tests"
        / "unit"
        / "dataset"
        / "aa_rwlt"
        / "fixtures"
    )
    assignment_path = args.out / "assignments.json"
    assignments = {"1": ["fixture-0", "fixture-1"], "2": ["fixture-2", "fixture-0"]}
    if args.mode in {"cancel", "hard-timeout", "errors"}:
        assignments = {"1": ["fixture-0"], "2": ["fixture-2"]}
    write_json(assignment_path, {"assignments": assignments})
    method = Methodology(
        user_spawn_rate=10,
        settling_time_seconds=0,
        min_measurement_seconds=60,
        min_total_trajectories=0 if args.mode in {"cancel", "slow-stream"} else 30,
        min_trajectories_per_user=0 if args.mode == "cancel" else 3,
        max_starting_line_offset=0,
        phase_timeout_seconds=90,
        grace_period_seconds=3 if args.mode == "slow-stream" else 1,
        request_timeout_seconds=1 if args.mode == "slow-stream" else 300,
        sgl_conversation_dp_size=8 if args.routing else None,
        send_conversation_routing_headers=args.routing,
        use_dynamo_conv_aware_routing=args.routing,
        send_dynamo_request_id=args.routing,
    )
    prepare(
        fixtures / "trajectories.jsonl",
        assignment_path,
        args.out / "bundle",
        model="fixture",
        methodology=method,
    )
    tokenizer_dir = args.out / "tokenizer"
    tokenizer_dir.mkdir()
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(tokenizer_dir / "tokenizer.json"))
    write_json(
        tokenizer_dir / "tokenizer_config.json",
        {
            "tokenizer_class": "TokenizersBackend",
            "backend": "tokenizers",
            "unk_token": "[UNK]",
        },
    )
    observed: list[dict] = []
    release_hangs = asyncio.Event()
    fake_key = "synthetic-local-validation-key-not-a-real-credential"
    if args.auth_test:
        os.environ["AA_LOCAL_HTTP_TEST_KEY"] = fake_key

    async def models_handler(request: web.Request) -> web.Response:
        return web.json_response({"data": [{"id": "fixture", "object": "model"}]})

    async def chat(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        observation: dict = {"body": body, "start": time.time(), "end": None}
        observation["headers"] = {
            k.lower(): v
            for k, v in request.headers.items()
            if k.lower() != "authorization"
        }
        observation["authenticated"] = (
            request.headers.get("Authorization") == f"Bearer {fake_key}"
        )
        if args.auth_test:
            assert observation["authenticated"], (
                "Configured API key did not reach the endpoint"
            )
        observed.append(observation)
        if args.mode == "errors" and "sum of 0 and 0" in body["messages"][1]["content"]:
            observation["end"] = time.time()
            observation["http_status"] = 429
            return web.json_response(
                {"error": {"message": "Synthetic rate limit"}}, status=429
            )
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)

        async def send(delta: dict, *, finish: bool = False) -> None:
            chunk = {
                "id": "fixture-response",
                "object": "chat.completion.chunk",
                "created": int(observation["start"]),
                "model": "fixture",
                "choices": [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": "stop" if finish else None,
                    }
                ],
            }
            if finish and args.mode != "no-usage":
                chunk["usage"] = {
                    "prompt_tokens": 100,
                    "completion_tokens": 10,
                    "total_tokens": 110,
                }
            await response.write(b"data: " + orjson.dumps(chunk) + b"\n\n")

        try:
            await send({"role": "assistant"})
            await asyncio.sleep(0.02)
            for _ in range(10):
                await send({"content": "hello "})
                await asyncio.sleep(0.2 if args.mode == "slow-stream" else 0.005)
            slow = (
                args.mode in {"cancel", "hard-timeout"}
                and "sum of 0 and 0" in body["messages"][1]["content"]
            )
            if slow:
                await release_hangs.wait()
            else:
                await send({}, finish=True)
                await response.write(b"data: [DONE]\n\n")
                await response.write_eof()
                observation["end"] = time.time()
        except (ConnectionError, asyncio.CancelledError):
            observation["disconnected"] = True
        return response

    app = web.Application()
    app.router.add_get("/v1/models", models_handler)
    app.router.add_post("/v1/chat/completions", chat)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    # AppRunner exposes the OS-assigned ephemeral address, avoiding fixed ports.
    port = runner.addresses[0][1]
    command = [
        sys.executable,
        "-m",
        "aiperf",
        "profile",
        "--model",
        "fixture",
        "--public-dataset",
        "aa_rwlt",
        "--rwlt-bundle",
        str(args.out / "bundle" / "manifest.json"),
        "--rwlt-metrics-tokenizer",
        "o200k_base",
        "--url",
        f"http://127.0.0.1:{port}",
        "--tokenizer",
        str(tokenizer_dir),
        "--max-workers",
        "2",
        "--concurrency",
        "2",
        "--artifact-dir",
        str(args.out / "run"),
        "--ui",
        "none",
    ]
    if args.omit_request_payloads:
        os.environ["AIPERF_RECORD_STRIP_PAYLOAD_BYTES"] = "true"
    if args.auth_test:
        command.extend(["--api-key", fake_key])
    write_json(
        args.out / "command.json",
        ["<redacted>" if v == fake_key else v for v in command],
    )
    try:
        with (args.out / "client.log").open("xb") as log:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=log,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                returncode = await asyncio.wait_for(process.wait(), timeout=180)
            except BaseException:
                if process.returncode is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        await asyncio.wait_for(process.wait(), timeout=20)
                    except TimeoutError:
                        os.killpg(process.pid, signal.SIGKILL)
                        await process.wait()
                raise
    finally:
        release_hangs.set()
        await runner.cleanup()
        with (args.out / "received.jsonl").open("xb") as out:
            for observation in observed:
                out.write(orjson.dumps(observation) + b"\n")
    phase_dir = args.out / "run"
    phase = read_json(phase_dir / "aa_phase.json")
    summary = read_json(phase_dir / "aa_summary.json")
    expected_complete = (
        args.mode not in {"cancel", "hard-timeout"} or args.expect_complete_capture
    )
    expected_valid_stop = args.mode != "hard-timeout"
    assert returncode == (0 if expected_complete and expected_valid_stop else 1), (
        returncode
    )
    assert summary["valid_stop"] == expected_valid_stop, summary["stop_checks"]
    assert summary["stop_reason"] == (
        "criteria_met" if expected_valid_stop else "hard_timeout"
    )
    assert summary["complete_capture"] == expected_complete
    assert summary["native_counts"]["requests_sent"] == len(observed)
    sessions = {s["cache_bust_marker"]: s for s in phase["sessions"].values()}
    rows = [
        orjson.loads(line)
        for line in (args.out / "bundle" / "requests.jsonl").read_bytes().splitlines()
    ]
    trajectories: dict[str, list[dict]] = {}
    for row in rows:
        trajectories.setdefault(row["conversation_id"], []).append(row)
    by_session: dict[str, list[dict]] = {}
    by_user: dict[int, list[dict]] = {}
    native_request_ids: set[str] = set()
    for observation in observed:
        content = next(
            m["content"] for m in observation["body"]["messages"] if m["role"] == "user"
        )
        marker_match = re.match(r"\[rid:[0-9a-f]{12}\]", content)
        assert marker_match, "Missing cache marker"
        marker = marker_match.group()
        session = sessions[marker]
        headers = observation["headers"]
        assert "x-session-affinity" not in headers, (
            "Unexpected native affinity can alter gateway routing"
        )
        native_id = headers["x-request-id"]
        assert native_id not in native_request_ids
        native_request_ids.add(native_id)
        routing_id = f"{session['conversation_id']}:{marker[5:-1]}"
        for key in ("x-session-id", "x-smg-routing-key", "x-dynamo-session-id"):
            if args.routing:
                assert headers[key] == routing_id, (
                    "Native transport changed the configured routing key"
                )
            else:
                assert key not in headers
        if args.routing:
            assert headers["x-correlation-id"] == routing_id
        by_session.setdefault(marker, []).append(observation)
        by_user.setdefault(session["user_id"], []).append(observation)
    for marker, requests in by_session.items():
        session = sessions[marker]
        source_rows = sorted(
            trajectories[session["conversation_id"]],
            key=lambda r: r["conversation_idx"],
        )
        for index, observation in enumerate(requests):
            row = source_rows[session["start_turn_index"] + index]
            expected = dict(row["payload"])
            expected["messages"] = apply_cache_marker(expected["messages"], marker)
            if args.routing:
                expected["routed_dp_rank"] = session["user_id"] % 8
            assert observation["body"] == expected, (
                "Wire body differs from the recorded history"
            )
            observation["returned_delay_seconds"] = row["tool_call_delay_ms"] / 1000
    for requests in by_user.values():
        requests.sort(key=lambda r: r["start"])
        for previous, current in zip(requests, requests[1:], strict=False):
            assert previous["end"] is not None, (
                "User issued a request before its previous stream closed"
            )
            gap = current["start"] - previous["end"]
            assert gap >= previous["returned_delay_seconds"] - 0.004, (
                "Returned-turn delay was shortened"
            )
    if args.mode in {"cancel", "hard-timeout"}:
        assert summary["cancelled_requests"] >= 1
        assert len(by_user[0]) == 1, (
            "Cancelled user unexpectedly advanced its trajectory"
        )
    else:
        assert all(value >= 3 for value in phase["completed_trajectories"].values())
        if args.mode == "errors":
            assert summary["failed_requests"] > 0
            assert summary["successful_requests"] > 0
            assert summary["native_counts"]["request_errors"] == sum(
                r.get("http_status") == 429 for r in observed
            )
        else:
            assert summary["failed_requests"] == 0
            if args.mode == "slow-stream":
                assert all(r["end"] - r["start"] > 1 for r in observed), (
                    "Healthy streams did not exceed timeout"
                )
            if args.mode == "no-usage":
                assert summary["missing_server_usage"] == summary["successful_requests"]
                assert summary["distributions"]["sse_output_speed"]["count"] == 0
    expected_payloads = 0 if args.omit_request_payloads else summary["total_requests"]
    assert summary["raw_request_payload_count"] == expected_payloads
    result = {
        "passed": True,
        "request_payloads_retained": summary["request_payloads_retained"],
        "raw_request_payload_count": summary["raw_request_payload_count"],
        "mode": args.mode,
        "requests_verified": len(observed),
        "user_lanes_verified": len(by_user),
        "complete_capture": summary["complete_capture"],
        "cancelled_requests": summary["cancelled_requests"],
        "runner_returncode": returncode,
        "stop_reason": summary["stop_reason"],
        "valid_stop": summary["valid_stop"],
        "routing_enabled": args.routing,
        "authentication_redaction_verified": args.auth_test,
        "checks": [
            "full_wire_payloads",
            "recorded_histories",
            "single_inflight_per_user",
            "returned_turn_delays",
            "cache_marker_lifetime",
            "stop_criteria",
            "native_capture_reconciliation",
            "routing_headers",
        ],
    }
    if args.auth_test:
        for artifact in args.out.rglob("*"):
            if artifact.is_file():
                assert fake_key.encode() not in artifact.read_bytes(), (
                    f"API key leaked into {artifact.name}"
                )
    write_json(args.out / "validation.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=[
            "normal",
            "cancel",
            "hard-timeout",
            "errors",
            "no-usage",
            "slow-stream",
        ],
        default="normal",
    )
    parser.add_argument("--omit-request-payloads", action="store_true")
    parser.add_argument(
        "--routing",
        action="store_true",
        help="Exercise all supported routing headers and DP affinity",
    )
    parser.add_argument(
        "--auth-test",
        action="store_true",
        help="Send a fake API key and verify export redaction",
    )
    parser.add_argument(
        "--expect-complete-capture",
        action="store_true",
        help="Require complete capture when exercising cancellation",
    )
    args = parser.parse_args()
    args.out = args.out.resolve()
    print(json.dumps(asyncio.run(validate(args)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
