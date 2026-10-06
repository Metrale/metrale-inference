# Laguna XS 2.1 native qualification

This is the cumulative bring-up record for `poolside/Laguna-XS-2.1-NVFP4`.
Add fixes, regression tests and qualification evidence to the same model branch
and draft PR until acceptance. Current status: native single-request smoke passed;
GPU batching is blocked by a reproduced correctness failure. No certification pass.

## Frozen baseline, October 6, 2026

- Engine: `3e954ac14786e614cb87c713dba1cc8735edab80`.
- Model: `d32afde8b09af1539b49ff96ff5551c674485f8e`.
- Hardware: one NVIDIA GB10 / DGX Spark; Rust 1.93.1, CUDA 13.0.
- Release binary SHA-256: `0a19d68408e7fa0cf2ae8ef7a0754072a2db197d6864c1f1495cf64d98e6f55b`.
- Build: `METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=laguna-xs-2.1
  METRALE_TARGET_QUANT=nvfp4 cargo build --locked --release -p metrale-server
  --bin met --no-default-features --features cuda,otlp -j4`.
- Kernel check: 296 lookups, zero unresolved, 35 expected absent;
  kernel set `a903f598945f`.
- Checkpoint: 17 files, 21,596,075,520 bytes, separately SHA-256 verified
  against a TrueNAS backup. Active weights retained. Backup is not runtime evidence.

## Reproduced failure and useful control

Serve the checkpoint with `--kernel-target laguna-xs-2.1`,
`--model-name poolside/Laguna-XS-2.1-NVFP4`, `--bind 127.0.0.1`,
`--port 8888`, `--gpu-memory-utilization 0.85`, `--max-seq-len 8192`,
`--max-batch-size 4`, `--kv-cache-dtype fp8`, `--telemetry basic`, `--no-tui`.

POST `/v1/chat/completions` with temperature 0 and max_tokens 64. Use two
user prompts: `Calculate 20 + 7. Answer only the integer.` and
`Calculate 21 + 7. Answer only the integer.` The independent oracle is exact
trimmed content `27` and `28`. Alternate sequential requests with two requests
released together using a client barrier; repeat three rounds. Record full
responses, finish reasons, usage, settings and engine/model identity.

| Configuration | Sequential responses | Concurrent responses |
| --- | --- | --- |
| Baseline, GPU batch limit 4 | 6/6 correct | 0/6 correct |
| Same binary, METRALE_NO_DECODE_GRAPHS_MULTISEQ=1 | 6/6 correct | 0/6 correct |
| Same binary, GPU batch limit 1 | 6/6 correct | 6/6 correct |

Failures repeated digits and exhausted the output budget. Disabling multi-sequence
CUDA graph replay did not fix them. This does not isolate the failing kernel.
The batch-one control serializes GPU execution despite concurrent HTTP clients;
it establishes a restricted operating mode, not working batched inference.

Before this diagnostic, arithmetic, a forced lookup_part tool round trip and SSE
completion passed. The first six-hour campaign stopped at its first C2 failure
(max_tokens 512). A replacement six-hour campaign is running with GPU batch limit
1 and HTTP concurrency 1/2/4. Its completion is pending; retain the failed campaign
separately rather than replacing it with the restricted result.

## Remaining acceptance

- [ ] Locate the batching defect using intermediate-output parity and pinned reference outputs.
- [ ] Review existing shared NVFP4 fixes before introducing another kernel change.
- [ ] Add a regression that fails on this baseline and passes with the correction.
- [ ] Repeat sequential and concurrent generation, tools, streaming and cancellation.
- [ ] Complete unrestricted soak, numerical parity and required benchmark certification.
- [ ] Record throughput, latency, memory and energy with exact hardware/build identity.

All correctness fixes and their evidence stay on this model PR. A workaround or
a reference-engine result must not be described as native batched qualification.
