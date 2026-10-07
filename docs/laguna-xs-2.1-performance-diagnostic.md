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
container-namespace PID497, `VLLM::EngineCor`, context1/device0, rather than
frontend PID253. These are not host PIDs. Graph-node
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

## Longer fixed-count decode: current combined native

The current combined candidate, binary SHA-256
`5784c0fe1064b2dd7ab55dca39823d7ec3fef0c1ea205f6943edc783b63451b5`,
ran with both default-off small-row options explicitly enabled. Its source/build
receipt SHA-256 is
`13f6b248e09009e1af94d746c17c0c9a6f25844d4ffd497eb8f2d40bf9c8005f`.
The same pinned Marlin image and checkpoint above were used. Current checkpoint
config/index hashes matched the previously verified backup manifest; shard sizes
were checked without claiming a new full-weight checksum pass.

Native-A / Marlin-B / Marlin-B2 / native-A2 used fresh owned servers, identical
64 input IDs, temperature zero, and exactly 256 or 512 output tokens. Every
case and C1/C2/C4 rung was warmed before three measured repetitions. All
96 cohorts / 224 requests, including warmups, passed count and terminal admission.
No profiling, builds, downloads or remote probes overlapped timed requests.

| 512-output case | Native client TPOT, ms | Marlin client TPOT, ms | Native cohort tokens/s | Marlin cohort tokens/s |
|---|---:|---:|---:|---:|
| C1 | 20.550–20.651 | 23.274–23.330 | 47.23–47.47 | 42.69–42.79 |
| C2 | 24.621–25.049 | 22.221–23.914 | 77.45–78.69 | 82.93–89.57 |
| C4 | 31.505–32.210 | 23.989–24.010 | 117.56–120.97 | 164.89–165.11 |

Ranges retain both session orders. Across both lengths/orders, native C1 total
latency was 9.1–10.1% lower. C2 total latency was 5.5–19.1% higher and C4 was
36.4–45.4% higher. The reference's C2 variation remains visible; no run was
removed. Longer output amortizes prefill but does not eliminate the concurrent
serving gap. These client metrics retain final drain/interleaving, and actual
GPU batch membership is not inferred. This is a bounded C1 win, not overall
competitive serving, numerical equivalence, energy efficiency or broad quality
qualification. Coding remains 9/12.

The first orchestration attempt failed before requests because launching from
the SSH home directory selected the checkpoint template, whose `generation`
statement is unsupported by this native template parser. The successful sequence
retained the previously qualified repository working directory and bundled
`jinja-templates/laguna.jinja`, SHA-256
`cfa2d32aa24e16f63133abbe5b22fe8db06b9b4254feed1fe5abf0c8f6f72d9c`.
The startup refusal and corrected launch identity are preserved. This working
directory dependency remains a packaging limitation; no template fallback was
changed within the comparison.

## Current combined native attribution

A separate capture now verifies the same `5784c0fe…` combined executable with
both options enabled. Exact 64-input/64-output C1 and C4 requests, warmups and
collection-off controls passed. The profiled native host PID was 3330353;
SQLite global PID 337349028347904 joins that process, CUDA context 1/device 0,
and its recorded `/proc` executable hash. This replaces the older `020057a`
trace only for attribution of the current combined implementation.

| Profiled region | Kernel extent, ms | GPU active union, ms | Extent outside GPU union, ms |
|---|---:|---:|---:|
| Native combined C1 | 1577.67 | 1507.09 | 70.58 |
| Native combined C4 | 3015.59 | 2815.35 | 200.24 |

C4 kernel-duration sums include 937.37 ms in the capacity-four dense GEMV,
630.65 ms in the M16 expert-prefill entry, 172.35 ms in its M64 fallback,
252.13 ms in routed batch-two gate/up and 187.79 ms in routed batch-two down.
The two batch-two dispatches do not demonstrate shared expert IDs across requests;
source indexing treats each routed token/slot separately. Combining launches and
reusing weights are distinct opportunities. Dense and expert work both remain
material, and the prefill pair remains part of this 64-output region.

These are profiler-injected traces, not timing replacements for the longer
uninstrumented comparison. Kernel sums may overlap; the uncovered GPU interval
is not automatically CPU work or an attributable scheduler delay. No energy or
new quality qualification follows from attribution.

## Rejected follow-up screens

The following private experiments were retained as diagnostics and did not change
the production math or default dispatch:

- Loading 32 K values into shared memory before two ordered K16 BF16 MMA steps
  matched 31 constructed/learned comparisons, including the independent integer
  oracle and wrong-gather/grid controls. Eight microcases with captured expert
  load distributions were 31–51% slower than the qualified K16 hybrid; rejected.
- Direct activation loads in the capacity-four dense family preserved 63
  constructed and 27 legacy comparisons plus oracle/refusal controls. Actual C4
  shape screens regressed at N256/K2048 and N1024/K2048, with the other sampled
  shapes approximately flat; no general C4 promotion.
- Combining two independent batch-two routed-expert launches into one batch-four
  launch passed constructed same/distinct/partial expert-ID controls. It was
  restricted to exactly four live eager rows. The actual default-route profile
  observed **zero** new batch-four gate/down calls: graph execution correctly
  retained the original path. Twelve fixed-count C4 controls passed, but this
  does not qualify the unused candidate. Graph policy and live-row guards were
  not weakened to produce a speed result.

The last bounded run used binary
`a2017dff0afbdaffcdf22f6b08a99464f32ceb37ad5e65f70b129ec5a4414cd8`
from a new empty working directory. It also verified startup with the embedded
reviewed Laguna template added in `e9eefb6`; no working-directory template file
was present. Its profiler injection and private kernel overlay exclude it from
the frozen `5784c0fe…` speed comparisons above. The private overlay was restored
to its known parent files after capture, and the owned server stopped.

## Exact E2M1 lookup substitution

For the same pinned checkpoint linked above, the existing opt-in M16/M64 prefill
pair now constructs the E2M1 FP32 value bits directly. The legacy unfiltered
entry retains its original lookup policy. Both FP32 scale multiplications remain
ordered before the final BF16 cast; negative zero is preserved. This changes
neither the activation precision nor the K16 MMA accumulation order.

The frozen candidate executable is
`a9f01a9389ee456dc83f8aafcfc99316b0069fadb9649f102cda0a7fe23ef7f8`,
against combined incumbent `5784c0fe…`. Source/PTX receipts verify the complete
intended PTX bytes embedded in that binary. The source snapshot also contains the
template packaging fix, while both timing arms use the same qualified working
directory and reviewed template. No batch-four private overlay remains.

Four quiet unprofiled A/B/B2/A2 sessions admitted all 144 fixed-count cohorts.
Short-prefill total latency improved **7.20–8.06%**, and long-prefill
**3.73–4.23%**, across C1/C2/C4 in both orders. With 64 output tokens, C1 total
improved 1.17/1.34% and C2 1.87/3.03%. C4 was mixed: 0.38% slower in one order
and 4.28% faster in the other. There is no consistent C4 decode improvement.
All observed concatenated SSE text-hash sets match; this is not token-ID equality.

The constructed native decoder evidence covers 45,056 combinations of all
sixteen nibbles, all 256 FP8 scale bytes, and eleven second-scale values, including
signed zero and nonfinite cases. Raw bits match at every product/cast stage;
sign-removal and finite product-reassociation controls fail as intended. The
standalone `scripts/laguna/e2m1_decode_check.cu` preserves this regression gate
against the actual shared helper, without checkpoint data. Existing constructed
small-row tests independently exercise exact dyadic dots, tails, routing, the
934-row fallback and intentionally incorrect grids.

On an idle GB10 host, run the standalone gate from the repository root:

```sh
nvcc --ptx -O3 --fmad=false -arch=compute_121 scripts/laguna/e2m1_decode_check.cu -o /tmp/e2m1_decode_check.ptx
c++ -DE2M1_HOST -x c++ -I/usr/local/cuda/include scripts/laguna/e2m1_decode_check.cu -lcuda -o /tmp/e2m1_decode_check
/tmp/e2m1_decode_check /tmp/e2m1_decode_check.ptx
```

Six schemas, four JSON/tool controls and six concurrent schemas pass. Unequal
32/48/64/80-token draining, cancellation with three surviving requests and a
subsequent C1 request pass. All twelve generated coding outputs have the same
hashes as the incumbent; isolated semantic grading remains **9/12**, with the
same three retry-delay failures. The prefill improvement does not establish
general coding quality, energy efficiency, or competitive concurrent decoding.

## Dense precision audit and bounded pair reuse

CPU-only reads of the pinned checkpoint headers confirm **284 selected tensors**
(attention projections/norms, first-layer dense FFN and output head) are BF16.
The declared quantization configuration explicitly excludes attention Q/K/V/O,
the head, first-layer FFN and router from NVFP4. Representative shapes are
Q `[6144,2048]` / `[8192,2048]`, O `[2048,6144]` / `[2048,8192]`, and head
`[100352,2048]`. The native dense path is not expanding packed versions of those
weights. The reference trace's dominant dense entry is explicitly CUTLASS BF16
WMMA, accounting for 1004.925 ms across 7750 calls in the recorded C4 region.
Dense bandwidth remains relevant, but a checkpoint precision mismatch does not
explain this comparison. Trace regions include prompt processing; names alone
must not be used to classify individual launches as steady decode.

A separate private graph-compatible candidate reused gate/up weights when the
two routed rows selected the same expert in the same slot. Twenty-eight
constructed comparisons, duplicate/partial/null/non-null-shared controls and an
independent staged-FP32 oracle passed. A separate trace observed 4914 calls to
its new entry. This proves dispatch, not the fraction of matched experts.

Against qualified LUT binary `a9f01a…`, candidate `f9039fea…` completed four quiet
sessions: 160 cohorts and 400 fixed-count requests. C2 total latency for 64 output
tokens improved 2.53/2.59%; C4 was flat (-0.37/+0.07%). Four distinct rendered
prompts at C4/128 outputs were also flat (-0.327/+0.054%). Actual token fixtures
matched across all arms. Fixed-workload text hashes matched; diverse text varied
on both arms, without establishing a cause. Coding outputs remained identical
and isolated grading stayed 9/12. The candidate remains private: a narrow C2 win
does not resolve the C4 objective.

An occurrence-rank extension preserved one-to-one duplicate handling and passed
the same constructed controls, but only improved the half-overlap case about
4% while regressing the no-overlap kernel about 10% at the actual 128-thread
shape. It was not promoted or subjected to a blind full-model campaign. Neither
prototype changes the checked-in default dispatch.
