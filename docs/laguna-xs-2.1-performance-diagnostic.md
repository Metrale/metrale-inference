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

## Separate Marlin GPU attribution

A subsequent pinned-Marlin trace uses the same 64 input IDs and 64 output-token
requests at client C1/C4, original NVFP4 checkpoint and FP8 KV. Actual startup
selects `MarlinNvFp4LinearKernel` and `MARLIN` MoE; model metadata, shard sizes,
installed source/version receipts and the existing verified backup manifest
were checked. This capture did not independently rehash all weight bytes.
The official NVIDIA 26.09 ARM64 image remains pinned to
`sha256:fa68ef92f906e1b3770621625c5af539d15297fea15dacfc1466b853a567c5b6`.

| Captured kernel window | C1 | C4 |
|---|---:|---:|
| Kernel count | 65,310 | 69,037 |
| Kernel extent | 1,533.715 ms | 1,645.546 ms |
| GPU interval union | 1,520.568 ms | 1,630.442 ms |
| Summed kernel duration | 1,685.125 ms | 1,881.014 ms |
| Uncovered GPU interval | 13.147 ms | 15.104 ms |
| CUDA API interval outside GPU union | 9.145 ms | 8.518 ms |

An independent interval sweep verifies GPU/API unions and overlap. Kernel sums
exceed union because work overlaps; these quantities must not be added together.
The actual kernel `globalPid=281483314987008` joins the SQLite process table to
PID497, `VLLM::EngineCor`, context1/device0, rather than frontend PID253. Graph-node
kernel events are retained. Raw SSE confirms exact prompt IDs, usage64/64,
length termination, and overlapping client request intervals at C4. Client
concurrency is not proof of actual GPU batch membership.

Dense projection names account for 1,170.224 ms / 1,055.305 ms summed kernel time
at C1/C4. This is heuristic attribution of named kernels, not additive percentages
or isolated operator latency. For context, the **older native 020057a capture**
has GPU unions of 1,529.752 ms / 3,148.847 ms and kernel extents of
1,596.838 ms / 3,357.477 ms. That historical comparison includes prefill plus
decode and does not represent the later combined small-row executable. The
current combined executable requires its own trace.

Collection-off controls before and after capture also pass exact counts, but
retain profiler injection. They are not uninstrumented speed measurements.
Neither trace establishes numerical equivalence, energy efficiency or broad
model quality. The remaining C4 gap motivates further dense/expert execution
investigation rather than attributing all delay to host waiting.

Raw SQLite SHA-256 receipts:

- C1: `401db4522c79bb977a4937e9a57d32c668855db3c504b2e322aa495493e56359`
- C4: `b2a067876a60d24565e3de25513239a276c4c9c2c0b9bb5dd2c4710e9b459f0a`
