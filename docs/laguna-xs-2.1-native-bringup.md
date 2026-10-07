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
(max_tokens 512). The replacement six-hour campaign used GPU batch limit 1 and HTTP concurrency
1/2/4, but stopped during cycle 10 at concurrency 4: the model returned
`33 + 7 = 40\n\n40` when the exact-format oracle required `40`. The arithmetic
was correct; this is an instruction/output-format failure, distinct from the
repeated-digit batching failure. Neither soak completed. Preserve both failures
and test arithmetic correctness and instruction compliance separately in the next run.

## Remaining acceptance

- [x] Locate the Q/K normalization dispatch defect with alternating failure/pass controls.
- [ ] Complete intermediate-output parity against a pinned reference.
- [ ] Review existing shared NVFP4 fixes before introducing another kernel change.
- [ ] Add a regression that fails on this baseline and passes with the correction.
- [ ] Repeat sequential and concurrent generation, tools, streaming and cancellation.
- [ ] Complete unrestricted soak, numerical parity and required benchmark certification.
- [ ] Record throughput, latency, memory and energy with exact hardware/build identity.

All correctness fixes and their evidence stay on this model PR. A workaround or
a reference-engine result must not be described as native batched qualification.

## Q/K dispatch correction, October 6

The scalar path selects vanilla RMS normalization for Laguna (`weight`), but the
strided batched kernel implements additive weights (`1 + weight`). Disabling
strided Q/K norm passed all 12 diagnostic responses; restoring the unchanged
baseline reproduced all six C2 failures. Disabling batched BF16 projections had
not helped. This isolates a normalization-policy mismatch.

Commit `d9d9879` prevents plain-weight models from selecting the additive strided
kernel. Existing additive-model admission is unchanged; Laguna uses the existing
per-sequence vanilla norm. Two focused dispatch tests passed on Spark2. The rebuilt
CUDA server, with no diagnostic environment override and GPU batch limit 4, passed
all 12 original reproducer responses, including six concurrent ones. Source was
baseline `3e954ac` plus this guard and BF16 diagnostic correction `b59ab41`; the
exact patch and binary digest were retained with operational evidence.

A new six-hour batch-four campaign exercises arithmetic, tools, SSE and HTTP
concurrency 1/2/4. It retains full arithmetic responses, reports format failures
separately, and stops on wrong last-line arithmetic or abnormal termination. A
correct last line cannot erase an exact-format failure or turn the whole campaign
into a pass. Completion, full numerical parity and speed qualification remain open.

The source guard fixes correctness through an existing scalar fallback; no speed
improvement is claimed. A matching parameterized strided vanilla kernel remains
a possible optimization after profiling and parity tests.
