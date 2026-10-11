# QCI runtime telemetry, request identity and receipts

What the engine exports for the QCI watchdog (investor-mvp #28), the economics view (#37),
trace correlation (#49) and compatibility receipts (#46). Field names follow the platform
contracts v1 (#41). The runtime's binding of them is one module:
`crates/qci-dump/src/contract/`.

## Contracts (#41)

`schemas/platform/` is Metrale/mfs `schemas/platform/` copied unchanged (Apache-2.0, its
own `LICENSE`). `schemas/platform.pin` records the mfs commit and the SHA-256 of the copy's
`SHA256SUMS`. Never edit the copy by hand. To take a newer bundle, copy it again and update
the pin in the same commit.

`crates/qci-dump/tests/platform_contracts.rs` runs in `cargo test` and checks three things:

1. The copy matches the pin: every file's SHA-256 is right, and no file is missing from
   `SHA256SUMS`.
2. Every valid fixture of every kind reproduces its `jcs_sha256`. This holds the runtime's
   canonical JSON to the platform's.
3. For the three kinds the runtime writes or reads (`inference-usage`,
   `hardware-fingerprint`, `compatibility-receipt`), every valid fixture parses, validates
   and round-trips to the same canonical digest, and every invalid fixture is refused.

The other kinds have no runtime type, so the runtime checks only their digests.

Rule for every field: **absent is never zero.** A quantity the engine did not measure is
left out of `/metrics`, or written as `{"absent": "<reason>"}` in a record. It is never
written as `0`.

## Request identity (#49)

Every `/v1/*` request is stamped before auth, rate limiting or the handler runs
(`main_modules/trace_layer.rs`, decisions in `qci-dump/src/trace.rs`).

| request header | accepted form | otherwise |
|---|---|---|
| `x-request-id` | 16 bytes: 32 hex digits or a UUID, either case | a random id is minted |
| `traceparent` | W3C Trace Context level 1 (`00-<32 hex>-<16 hex>-<2 hex>`) | a random trace id is minted |

A header id that parses is kept unchanged. Nothing is truncated or repaired. A repeated
header, or one that is not visible ASCII, counts as invalid.

| response header | value |
|---|---|
| `x-request-id` | the request id, lower-case hex |
| `traceresponse` | `00-<trace id>-<this runtime's span id>-<flags>` (`sampled` is carried through) |
| `x-metrale-id-source` | `request_id=<src>, trace_id=<src>`, where `src` is `header`, `generated-absent` or `generated-invalid` |

Every JSON error body under `/v1/*` gets `error.request_id` and `error.trace_id`. That
covers the handler's errors, 401 auth, 429 rate limit, 503 GPU fault and the
chat-completions SSE error frame. A body is rewritten only when it is JSON, at most 64 KiB, and has an
`error` object. Other bodies are sent byte for byte. The `/v1/completions` SSE error frame
is `{"error": "<text>"}` with no object, so for that frame the ids are in the headers only.

The ids are never metric labels.

## Per-request usage record (#28, #37, #49)

There is one `InferenceUsage` v1 record for each request the scheduler accepted. It is
written in two places:

- in the response, as `usage.inference_usage`: the blocking body, or the final SSE
  usage chunk. Only the OpenAI chat and completions surfaces write it in the response.
- in the log, as one JSON line under target `metrale::inference_usage`. Every surface
  writes it: chat, Responses, Messages and completions.

| field | unit / form | provenance |
|---|---|---|
| `request_id`, `trace_id` | 32 lower-case hex | from the stamp |
| `served_model` | the served model name, a route token `[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}` | a name outside it withholds the record (logged) |
| `recipe` | SHA-256 hex or `null` | `null`: no launcher passes a recipe digest yet |
| `runtime_commit` | 40 hex | build-time `METRALE_BUILD_COMMIT` |
| `outcome` | `completed`, `timeout`, `cancelled`, `{"engine_error": {"error": <kind>}}` | — |
| `tokens.prompt`, `.completion`, `.cached_prompt` | count | measured |
| `tokens.reasoning` | count | measured; `absent: unsupported` when the model has no reasoning channel |
| `timings.received_at_unix_ms` | Unix ms | HTTP receipt |
| `timings.queue_ms` | ms | measured: receipt → the scheduler's first prefill of the request |
| `timings.ttft_ms` | ms | derived: queue + scheduler TTFT |
| `timings.total_ms` | ms | derived: queue + TTFT + decode, up to the finish frame (serialization after it is not counted) |
| `timings.tpot_us` | µs | derived: decode / (completion − 1); `absent: not_applicable` below 2 tokens |
| `energy` | `{millijoules, rail, window_mono_ns}` | `{"absent": "unsupported"}` with no rail. The response path does not attribute energy to one request; the device-level energy is in `/metrics`. |
| `finished_at_unix_ms` | Unix ms | when the record was written |

When a record is written:

- **Not written:** a request refused before dispatch (validation 400, 401, 429, no model).
  Its error body and headers carry the ids, and the agent records the refusal.
- **`engine_error`:** an error after dispatch. The counts are `unavailable`.
- **`cancelled`:** the client went away after dispatch.
- **No record at all, for any request:** a binary built without a 40-hex
  `METRALE_BUILD_COMMIT`. Every record is keyed by runtime commit. The log says so once.
  Release builds (`release-build.yml`, `datacenter-binaries.yml`) and every `docker/`
  image build stamp it from `scripts/build-commit.sh` (via `--build-arg METRALE_GIT_SHA`);
  `docker/build_commit_stamp_test.sh` holds each path to that. A local `cargo build` is
  unstamped on purpose: it is not a commit anyone can name.

A record never holds prompt or response text.

## `/metrics` series added for QCI (#28, #37)

| series | type | meaning |
|---|---|---|
| `metrale_queue_time_seconds{model}` | histogram | receipt → first scheduled prefill |
| `metrale_request_outcomes_total{outcome}` | counter | accepted requests by outcome (4 label values) |
| `metrale_prefill_tokens_total` | counter | prompt tokens computed (prompt − prefix-cache hits) |
| `metrale_request_prefill_seconds_total` | counter | sum of per-request prefill windows. Mean prefill rate = Δprefill_tokens / Δthis. |
| `metrale_request_decode_seconds_total` | counter | sum of per-request decode windows. Mean decode rate = Δ(generation_tokens − completed) / Δthis. |
| `metrale_build_info{version,commit}` | gauge = 1 | `commit="unstamped"` without `METRALE_BUILD_COMMIT` |
| `metrale_process_start_time_seconds` | gauge | Unix s at process start. Uptime = scrape time − this. |
| `metrale_startup_phase` | gauge | index of the last startup phase entered (`11` = listening) |

`metrale_prompt_tokens_total`, `metrale_generation_tokens_total` and
`metrale_time_to_first_token_seconds` now count `/v1/completions` too. They are written in
one place, `request_accounting.rs`.

Already exported, unchanged:

- `metrale_requests_total` and `metrale_requests_active` (the active-sessions gauge).
- At `--telemetry basic|kernel` only: GPU power, temperature, memory and the energy counter
  (`metrale_gpu_*`), and scheduler queue depth and KV occupancy (`metrale_sched_*`,
  `metrale_kv_*`). At the default `off`, these series are absent.

Not exported:

- GPU utilization. The NVML layer does not read it.
- GPU count.
- Model revision.

## Compatibility receipt (#46)

`qci-dump --case <campaign> --receipt <run-spec.json>` prints one `CompatibilityReceipt`
v1. It is keyed by runtime commit × model revision × quantization × hardware-fingerprint
digest × topology.

**Where the key comes from.** The model repository and revision come from the campaign
recipe's pins, the same pins the text dump prints. The hardware-fingerprint digest is
SHA-256 over the canonical JSON of the `HardwareFingerprint` document named by the spec.
Each check's artifact is digested from its file bytes.

**When the receipt is refused.** The verdict is `pass` exactly when every check passed. No
receipt is written without at least one `coherence` check and one `benchmark` check. It is
also refused when a failed check names no failure kind, or the key cannot name a real
combination: adaptive weights or KV, a zero parallel degree, or a non-`owner/name`
repository.

**What the emitter does not do.** It runs nothing and measures nothing. It only binds the
artifacts a real run produced.

Run spec (all fields required; `failure` is a failure kind or `null`):

```json
{"runtime_commit": "<40 hex>", "recipe_digest": "<64 hex>",
 "quantization": {"weights": "int4", "activations": "bf16", "kv_cache": "bf16"},
 "topology": {"nodes": 1, "accelerators_per_node": 1, "tensor_parallel": 1,
              "pipeline_parallel": 1, "expert_parallel": 1},
 "hardware_fingerprint": "<path to HardwareFingerprint v1 JSON>",
 "started_at_unix_ms": 0, "finished_at_unix_ms": 0, "synthetic": false,
 "checks": [{"name": "coherence", "kind": "coherence", "passed": true, "failure": null,
             "parameters": {}, "artifact": "<path>", "metrics": []}]}
```
