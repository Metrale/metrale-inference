# Laguna XS 2.1 native qualification

This is the cumulative bring-up record for `poolside/Laguna-XS-2.1-NVFP4`.
The batched-decode correctness failure below is fixed ("Q/K dispatch correction");
later evidence is in the linked quality, coding and performance documents. No
certification pass.

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
- Checkpoint: 17 files, 21,596,075,520 bytes, separately SHA-256 verified.

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
repeated-digit batching failure. Neither soak completed; later runs score
arithmetic correctness and instruction compliance separately.

## Acceptance status

- Done: the Q/K normalization dispatch defect, located with alternating
  failure/pass controls, with focused dispatch regressions and the baseline/fixed
  GPU response difference reproduced.
- Open: intermediate-output parity against a pinned reference, unrestricted soak,
  required benchmark certification, and energy with exact hardware/build identity.

A workaround or a reference-engine result is not native batched qualification.

## Q/K dispatch correction, October 6

The scalar path selects vanilla RMS normalization for Laguna (`weight`), but the
strided batched kernel implements additive weights (`1 + weight`). Disabling
strided Q/K norm passed all 12 diagnostic responses; restoring the unchanged
baseline reproduced all six C2 failures. Disabling batched BF16 projections had
not helped. This isolates a normalization-policy mismatch.

Commit `d9d9879` prevents plain-weight models from selecting the additive strided
kernel. Existing additive-model admission is unchanged; Laguna uses the existing
per-sequence vanilla norm. Two focused dispatch tests passed on the GB10 host. The rebuilt
CUDA server, with no diagnostic environment override and GPU batch limit 4, passed
all 12 original reproducer responses, including six concurrent ones. Source was
baseline `3e954ac` plus this guard and BF16 diagnostic correction `b59ab41`.

A new six-hour batch-four campaign exercises arithmetic, tools, SSE and HTTP
concurrency 1/2/4. It retains full arithmetic responses, reports format failures
separately, and stops on wrong integer-only answers or abnormal termination.
Explanatory responses remain numerically unscored and fail the format check;
they cannot turn the whole campaign into a pass. Its completed result is in
[quality diagnostics](laguna-xs-2.1-quality-diagnostics.md).

The source guard fixes correctness through an existing scalar fallback; no speed
improvement is claimed. A strided vanilla-norm kernel would be a separate
optimization.

Version-three soak snapshot through cycle 302: 2,121 responses in completed
request groups, with 1,353 exact-format answers and 768 explanatory responses
classified `unscored_format`. Initial arithmetic, forced tool round-trip and
streaming checks passed. This intermediate snapshot is neither a completed soak
nor clean quality qualification.

A read-only replay found eight (input, concurrency) groups with multiple distinct
response texts, all at concurrency two or four. No single-request variation was
observed in this snapshot. Batch composition/order differs across groups, so this
does not establish a race or numerical cause. Snapshot SHA-256:
`c89ed7074e97930f48a27242913a0562c3e02406b22a93ea3e310e65228b5612`.
Client concurrency alone does not prove scheduler batch membership.

A separate constructed GB10 kernel experiment confirmed the policy distinction:
for two rows, two 128-wide heads, weights -1/0/1 and all-one inputs, the unchanged
plain and additive kernels differ at all 512 active coordinates while leaving
padding unchanged. Both kernels agree with their own equations. This supports
the dispatch diagnosis but is not a capture of the model's actual Q/K activations.
