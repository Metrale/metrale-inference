# Laguna XS 2.1: bounded serving comparison

Evidence captured October 7, 2026 on one DGX Spark. The checkpoint is
[poolside/Laguna-XS-2.1-NVFP4 at d32afde8b09af1539b49ff96ff5551c674485f8e](https://huggingface.co/poolside/Laguna-XS-2.1-NVFP4/tree/d32afde8b09af1539b49ff96ff5551c674485f8e).
These measurements identify remaining bottlenecks. They do not certify model
quality, numerical equivalence, energy efficiency, or the full concurrency ladder.

## Frozen implementations and workload

Native source `020057a974f1da08a8bef4fd8a1974edf524b0dd`, release binary SHA-256
`13732afc36bec64b878157c2bf2f5608a1913bf086901d9e0954dc1eac45b989`,
default asynchronous scheduler, no diagnostic capture. The native launch trace
confirms untransposed BF16-MMA NVFP4 expert prefill and shared BF16-activation
expert decode. An inventory entry alone was not used as proof of execution.

Reference image:
`nvcr.io/nvidia/vllm@sha256:fa68ef92f906e1b3770621625c5af539d15297fea15dacfc1466b853a567c5b6`,
actual vLLM `0.29.0+5013de39.nv26.9.69442229`. The reference used the same local,
read-only checkpoint, BF16 model dtype, FP8 KV cache, batch limit four, maximum
sequence length 8,192, memory fraction .85, and disabled prefix caching. Native
used those same bounds. Both actual chat renderers produced identical 64-token
and 1,111-token input fixtures, subsequently submitted as raw input IDs.

Each C1/C2/C4 and workload combination was warmed, then measured three times.
Short/long prefill workloads generated exactly one token; short-input decode
produced exactly 64. All 36 cohorts per runtime passed output-count and terminal
framing admission. Counts, partial errors, raw SSE and source identities were
retained. No builds, downloads or profiler collections overlapped these timings.

| Runtime | C1 short / long first-text, ms | C1 / C2 / C4 client TPOT, ms | C4 aggregate tokens/s |
|---|---:|---:|---:|
| Native | 315.283 / 813.104 | 19.602 / 28.711 / 39.119 | 78.685 |
| vLLM explicit Marlin | 68.187 / 189.197 | 22.707 / 23.478 / 23.311 | 158.622 |
| vLLM default FlashInfer CUTLASS | 71.233 / 176.817 | 22.844 / 23.792 / 23.926 | 154.490 |

Client TPOT is `(HTTP completion - first nonempty text)/(output count - 1)`;
it includes final drain. First text need not equal the first generated token.
Aggregate throughput divides all verified output tokens by the complete cohort
window. Client concurrency is not evidence of actual GPU batch membership.

Explicit `--moe-backend marlin --linear-backend marlin` selected Marlin MoE and
Marlin NVFP4 linears, verified in installed source and startup logs. This is the
nominal W4A16 comparison. The default reference selected FlashInfer CUTLASS W4A4
for quantized MoE and linears: a separate activation precision, not a matched
precision speed claim. Neither comparison establishes full numerical equivalence.

## Observed bottleneck

A separate native Nsight capture recorded 1,074 kernel calls and 309.751 ms of
summed GPU kernel time for short C1 prefill. The actual
`moe_w4a16_grouped_gemm_ptrtable` kernel consumed 234.901 ms, or 75.8%.
It dequantizes to BF16 and uses `m16n8k16` BF16 MMA. The transposed sibling uses
FP8 MMA and is not an interchangeable precision-preserving optimization.

The decode-64 capture, including prefill, recorded 49,017 kernel calls at C1;
`dense_gemv_bf16` consumed 801.266 of 1,528.030 ms. At C4 the corresponding
capture recorded 90,068 calls; `dense_gemv_bf16_batchm` consumed 1,071.089 of
3,132.878 ms, and grouped expert prefill consumed 933.432 ms. Summed kernel times
are diagnostic attribution, not substitutes for unprofiled request latency.

Native narrow C1 decode TPOT is competitive here. Prefill and C4 throughput are
not. The next experiments address small expert row tiles and small-batch dense
GEMV separately, preserving original arithmetic and testing full requests before
any promotion. Private prototype results are not production improvements.

## Quality boundary

Native and both reference configurations passed six strict JSON schema cases.
Each initial coding collection passed nine of twelve bounded semantic tasks;
the retry-delay editing task remains incomplete. Some reference candidates
exhausted the isolated grader's memory and remain failures. Correct tool names
and arguments in the reference did not satisfy the strict protocol gate because
its forced-tool response ended with `finish_reason=stop`. See
[quality diagnostics](laguna-xs-2.1-quality-diagnostics.md) and
[code acceptance](laguna-xs-2.1-code-acceptance.md) for the separate controls and
known failures. No throughput result converts these failures into passes.
