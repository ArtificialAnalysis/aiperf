<!--
SPDX-FileCopyrightText: Copyright (c) 2026 Artificial Analysis. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AgentPerf v2

AgentPerf v2 measures an inference server under coding-agent load, by replaying recorded coding-agent sessions from many
users at once. Every request is sent with its recorded history, its recorded tools and its recorded pause, and each reply
is the server's own.

## Quick start

You need an OpenAI-compatible server and access to the dataset on Hugging Face, which is granted on request at
https://huggingface.co/datasets/ArtificialAnalysis/agentperf-v2. Set `HF_TOKEN` to a token of the account that has access.

```bash
git clone https://github.com/ArtificialAnalysis/aiperf.git
cd aiperf
git checkout agentperf-v2
uv sync

uv run aiperf profile \
    --scenario agentperf-v2 \
    --public-dataset agentperf_v2 \
    --hf-subset glm-5.3-flash/public \
    --url localhost:8000 \
    --model <served model name> \
    --endpoint-type chat \
    --tokenizer builtin \
    --use-server-token-count \
    --request-timeout-seconds 3600 \
    --concurrency 128
```

`--hf-subset` picks the recordings: `<model>/public`, where model is `deepseek-v4.1-flash`, `glm-5.3-flash` or `kimi-k3`.
`--concurrency` is the number of users. `--benchmark-duration` is a time limit, not the length of the run, and defaults
to 12 hours.

At high concurrency, cap aiperf's session cache so finished sessions cannot use up the client's memory: set
`AIPERF_WORKER_SESSION_CACHE_MAX_ENTRIES` to about 8 × users ÷ workers. The default was fine at 16 users.

Use the recordings of the model your server is serving. Your server must report cached tokens in its usage figures for the
cache results to appear: for example, `--enable-cache-report` on SGLang.

Every request carries its turn's recorded output length as `max_tokens` and is sent with `ignore_eos`, so each reply is
exactly as long as the recorded one and reply length cannot move the results. The dataset must set `max_tokens` on every
turn; the loader rejects a turn without one. `--extra-inputs` cannot set output length, stop, reasoning or sampling options.

## What a run does

The dataset's sessions sit end to end on one timeline, which is the loop. `N` users are spaced evenly along it, and each
replays the stretch up to the next user. Together they replay every request once.

| Stage | What happens | Measured |
| --- | --- | --- |
| Priming | Each user sends its history as one request with `max_tokens: 1`, over 5 minutes. Subagents running at that moment are primed too. | No |
| Burn-in | Each user replays the 15 minutes of the loop before its slice, so the server's cache fills as it would on a running system. | No |
| Slice | Each user replays its slice. | Yes |
| Overrun | A user that finishes early keeps replaying, so load stays at `N` until the last slice is done. | No |

The run stops when every user has finished its slice.

Short loops serve low concurrency. A run uses the longest of the first 16, 32 and 64 hours of the loop, or the whole loop,
that gives each user at most 75 minutes, and the 16-hour set if none does. Each set matches the whole dataset closely.

## Reading the results

- Latency and token figures count slice requests only.
- Rates, such as output tokens per second, divide by the average length of one slice, because slices overlap without lining
  up exactly.
- `profile_export.jsonl` keeps every request. Requests with `outside_slice: true` were sent for load and are not in the
  results. Slice requests carry `slice_lane`.

If the run reaches its time limit first, the log says how many slices were unfinished. Those results do not cover the whole
loop, so do not compare them with a finished run.

## Dataset format

The dataset is `dag_jsonl` with one addition: each turn has a `timestamp`, in milliseconds, on the loop's timeline. This is
how far along the loop the turn sits, assuming a fixed server speed, so that users can be spaced evenly by time.

To run your own file, use `--input-file <file> --custom-dataset-type dag_jsonl` in place of `--public-dataset` and
`--hf-subset`, and set `--benchmark-duration` yourself, or the run stops after one pass of the sessions. Every turn needs a
`timestamp`.
