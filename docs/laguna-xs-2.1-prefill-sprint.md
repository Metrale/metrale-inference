# Laguna prefill investigation (2026-10-07)

Checkpoint: [poolside/Laguna-XS-2.1-NVFP4, d32afde8b09af1539b49ff96ff5551c674485f8e](https://huggingface.co/poolside/Laguna-XS-2.1-NVFP4/tree/d32afde8b09af1539b49ff96ff5551c674485f8e).
This investigation follows the [long-decode comparison](laguna-xs-2.1-performance-diagnostic.md). A competitive C1 long-decode result does not establish competitive prefill.

## Matched one-output baseline

Frozen native binary `3aeaa2f04ff185b4fd624ba6aa9ee6df8949d1a183b10b4d55d63bc98c7091d8`
(runtime source `5f4d7b9`) versus the pinned reference serving engine image
`sha256:fa68ef92f906e1b3770621625c5af539d15297fea15dacfc1466b853a567c5b6`,
with explicit W4A16 linear and expert backends. Both use the same original checkpoint,
BF16 activations, FP8 KV, batch limit 4, memory fraction 0.85, prefix caching disabled,
and exactly the same 64- or 1,111-token input IDs. Each request generates exactly one token.
Native enables the previously qualified small-row, dense-row and minimum-token sampling options;
its declared activation policy and canonical row tiers remain unchanged.

All 96 cohorts / 224 requests passed framing, identity and exact-count admission, including
warmups. Three measured cohorts per workload/concurrency were run in each fresh process,
in native/reference/reference/native order, without overlapping compilation or profiling.

| Input tokens | Client concurrency | Native/reference median total latency, first / reverse order |
|---:|---:|---:|
| 64 | 1 | 3.810× / 3.838× |
| 64 | 2 | 4.233× / 5.305× |
| 64 | 4 | 4.641× / 4.708× |
| 1,111 | 1 | 3.979× / 3.989× |
| 1,111 | 2 | 3.240× / 3.251× |
| 1,111 | 4 | 3.022× / 3.041× |

C1 native first text is approximately 263 ms / 743 ms for short / long prompts,
versus 69 ms / 186 ms in the reference. Native C4 first-text arrivals form a staircase:
approximately 262/524/783/1,045 ms and 745/1,488/2,229/2,971 ms.
These are client observations, not proof of GPU batch membership. Whole-cohort prompt throughput
includes queueing, one generation step and HTTP completion; it is not isolated GPU prefill throughput.
Raw evidence is preserved in `laguna-prefill-sprint-baseline`, including process/checkpoint receipts,
raw streams, `comparison.json` and `prefill-report.json`. No energy measurement or broad quality
qualification follows from this one-output workload.

## Batching diagnosis: separate activation-policy experiment

The existing declared-policy guard refuses prefill codispatch. That refusal was preserved.
A separate diagnostic explicitly selected adaptive activation policy **and canonical row tiers**,
comparing the same frozen binary with and without codispatch/variable-length batching.
Actual engine logs confirmed four live streams and dispatched row totals 256, 4,444 and 1,367;
the mixed-length case used 64/65/127/1,111 input tokens and distinct KV slots.

The short cohort's first text changed from a serial staircase to approximately 320 ms for all
four requests. Long requests arrived at approximately 2,350 ms together instead of
744/1,513/2,253/2,995 ms: cohort completion improved, but median individual first-text latency
worsened. These instrumented, single-cohort observations are not accepted performance measurements.
Equal-length eight-token outputs matched across the diagnostic arms; two mixed-length outputs
changed. The latter remains a row/batch-dependent behavior limitation.

Six structured-output cases, four tool/JSON controls, six concurrent structured cases,
unequal-length draining, cancellation survivors and a subsequent C1 request passed.
The twelve sequential coding outputs were byte-identical to the earlier declared-policy outputs
whose isolated semantic grade was 9/12; this is not concurrent coding qualification.
The adaptive route is not a replacement for the declared-policy baseline or evidence of
row-invariant precision. Its admission delay, paged prefill attention and batch-dependent outputs
need separate qualification before any default change.

## Remaining phase measurement boundary

A first CUDA graph launch is not a per-request prefill/decode boundary. Prior current-native
captures still contain other requests' prompt processing after the first graph launch.
Future attribution must join request/sequence membership to actual prefill calls and completion,
including queue time; neither a kernel name nor client C4 establishes a four-row GPU prefill.
The next declared-policy optimization is evaluated against the frozen native baseline above,
with full-request gates and unchanged per-output arithmetic.

## Qualified narrow-column point for the opt-in small-row pair

The existing BF16 MMA family now exposes an M16/N32, 64-thread point alongside its
unchanged M16/N64 and M64/N64 entry points. The already opt-in Laguna small-row path
uses N32 for experts with at most 16 rows; the complete M64 fallback remains 128 threads.
The launch reads each entry's published N tile, verifies the expected 32/64 metadata,
and retains all shape, pointer and BF16-activation refusals. No activation-policy default changes.

Frozen candidate binary `5aca2f9adb50e4084507219cde94109f07a151f785af215ad10426fa87b957b0`
was measured against `3aeaa2…` with identical declared-policy flags. The tested source overlay
and embedded PTX hashes are retained in the stage receipt; the checked-in
version only cleans formatting, error wording and a comment from that runtime overlay.
The exact legacy PTX gate passes on SM90a/100a/121f: all nine old entry bodies preserve
instructions, registers, constants and shared allocations. Only the additional default
parameters in compiler symbol names and per-function label ordinals are normalized.
GB10 is the only runtime-measured architecture.

The CUDA gate passes 31 constructed and captured-checkpoint slice cases, including zero,
1/15/16/17/63/64/65/934-row boundaries, N tails, gathers, sentinel padding, an independent
integer oracle, and deliberately incomplete fallback/wrong-gather controls. Five CPU wrapper
tests verify distinct metadata, 64/128-thread launches, full fallback coverage and refusals.
A separate injected one-output profile records 117 calls of the actual N32 entry on the
short prompt; that trace is dispatch evidence, not an accepted latency measurement.

All 144 fixed-work cohorts pass admission in incumbent/candidate/candidate/incumbent order.
There are three measured cohorts after a warmup per workload/concurrency in each arm.
Lower is better in this table; ranges cover both orders, without dropping outliers.

| Workload | Client concurrency | Median total latency change |
|---|---:|---:|
| 64 input, 1 output | 1 | −4.20% / −4.09% |
| 64 input, 1 output | 2 | −3.77% / −4.11% |
| 64 input, 1 output | 4 | −4.43% / −4.62% |
| 1,111 input, 1 output | 1–4 | −0.75% to −0.95% |
| 64 input, 64 outputs | 1–4 | −0.66% to −1.53% |

Measured fixed-work text-hash sets match in both orders. Six schema cases, four tool/JSON
controls, six concurrent schema cases, unequal-length draining and cancellation controls pass.
All twelve coding sources exactly match the earlier independently graded 9/12 baseline;
the three retry-delay failures remain. A separate four-topic C4/128-output diagnostic improves
only 0.24%/0.66%; one topic retains text variation in the reverse comparison. There is no
claim of broad quality equivalence or a material diverse-decode win. This narrow optimization
helps short-prompt latency; it does not close the long-prefill or reference-engine gap above.

## Further bounded screens retained as rejected evidence

The next screens (not in this tree) preserved the current N32 arithmetic but did not justify a serving
change. Constructed and captured-slice controls remained exact; speed qualification is separate.

| Screen | Observed boundary | Decision |
|---|---|---|
| M32/N64 for every expert above 16 rows | Long gate/down only 0.679–0.751× incumbent speed | Reject |
| M32 only for 17–32 rows, full M64 above 32 | 0.844–1.018× across measured shapes | Reject |
| M64/N32 large-expert tile | Long shapes 0.828–0.852× | Reject |
| Compact expert/tile worklist | Long 1.002–1.026× before CPU/map-upload cost; short 0.958–1.038× | No runtime integration |
| Transposed shared B storage | 0.976–1.055×; insufficient consistent benefit | Reject |
| Transposed shared B plus N-major cooperative loading | 0.312–0.589× | Reject |

A separate one-output C1 profile of the N32 executable records 117 small and 117 large
expert launches. The short prompt spends 128.83 ms in small tiles and 40.63 ms in large tiles;
the long prompt spends 116.55 ms and 358.69 ms respectively (summed injected kernel durations).
Those requests include first-output projection/sampling and HTTP completion; these sums are
not isolated uninstrumented prefill latency. A standalone Nsight Compute attempt refused
hardware counters with `ERR_NVGPUCTRPERM`. No driver permissions were changed, and no
occupancy/bandwidth-counter conclusion is claimed.

Concurrent prompt scheduling remains an open qualification item. Explicit adaptive activation
plus canonical tiers permits the existing codispatch route; declared activation still refuses it.
That is a distinct policy experiment, not a default change or evidence that the declared route
is qualified for batching. Equal-length and unequal-length cohorts, actual logged membership,
per-request latency, cohort completion and semantic controls must remain separate evidence.

The matched adaptive/canonical concurrent-code check uses the same frozen `5aca2f…`
executable in both arms, changing only codispatch/variable-length prefill flags. Four distinct
coding prompts start together, repeated three times. Logs prove four-request kernel-batched
prefill over 633 prompt tokens (117/205/194/117), rather than inferring batching from client
concurrency. Both arms pass 9/12 isolated semantic grades with identical per-test outcomes;
the same three retry-delay failures remain. Only 3/12 generated-source hashes match between
arms. This is bounded semantic evidence, not text/bit equivalence or a general coding pass.
The grader executes generated code only in the pinned, network-disabled, read-only Docker
sandbox with resource limits and no host mounts. Raw responses and grades are retained.

A subsequent unprofiled, counterbalanced control/codispatch/codispatch/control campaign
admits all 96 cohorts / 224 requests with exact 64/1,111 input IDs and one output. Both arms
use the frozen N32 executable, explicit adaptive activation and canonical tiers; only the two
prefill dispatch flags differ. Each workload/concurrency is warmed before three measured
cohorts. These results do not replace the declared-policy native/reference comparison above.

| Prompt | Clients | Median per-request total change | Median cohort completion change |
|---|---:|---:|---:|
| 64 tokens | 1 | +3.73% / +4.16% | +3.73% / +4.16% |
| 64 tokens | 2 | −25.41% / −25.14% | −43.83% / −43.92% |
| 64 tokens | 4 | −50.20% / −50.38% | −68.87% / −68.87% |
| 1,111 tokens | 1 | +1.87% / +1.11% | +1.87% / +1.11% |
| 1,111 tokens | 2 | +17.68% / +17.39% | −11.74% / −11.84% |
| 1,111 tokens | 4 | +27.84% / +27.82% | −20.01% / −20.20% |

Lower is better; both orders are shown. Median first-text changes follow the same direction.
The group finishing sooner does not mean each request benefits. No blanket default promotion
is justified: single requests regress, long concurrent requests trade worse individual latency
for earlier group completion, and semantic/output equivalence remains only bounded evidence.

The final M64/N128, 256-thread screen also preserves all 31 numerical controls but
runs long shapes at only 0.878–0.911× incumbent speed (short 0.960–0.996×). It is rejected;
no serving binary includes it.

## What must close before declared-policy batching

The refusal in `crates/server/src/main_modules/serve_load/act_quant_support.rs`
(`prefill_lever_refusal`) is intentional: fixed activation policy must not silently make a
prompt's output depend on its wave-mates. Passing a few semantic tasks does not discharge it.
The next qualification work has concrete source boundaries:

1. **Match attention operands and route.**
   `qwen3_attention/trait_impl/prefill_inner.rs` sends a single first chunk through contiguous
   Q/K/V attention, while admitted batched first chunks use paged attention. With this launch's
   FP8 KV cache, that is a real operand/rounding distinction. Capture the same prompt's Q/K/V,
   attention output and residual at the first differing layer under identical prefix conditions;
   prove the required invariant or implement a compatible shared attention route before relaxing
   the guard. Do not attribute all differences to activation tiers.
2. **Verify row-policy invariance, rather than assuming canonical tiers provide it.**
   `layers/row_tiers.rs` controls row-format policy, but joined waves still change matrix geometry,
   expert token counts and kernel dispatch. Compare same-operand dense/router/expert boundaries
   across wave size and ordering, with independent numerical controls. Retain actual generated
   output differences and the fixed-format contract; do not retrofit a tolerance to hide them.
3. **Prove admission and attribution.**
   `scheduler/phase_start_prefills.rs` requires an idle, non-EP, image-free chunked route and
   sufficiently co-arriving requests. `phase_continue_prefills/run_batched_prefill.rs` groups
   chunk start and last-chunk status, then enforces row budgets. The engine's
   `prefill_b/batch_kernel/eligible.rs` and `batch_kernel.rs` additionally enforce scratch/KV
   capacity and can fall back. Record stable sequence identity, actual membership, chunk spans,
   admission/fallback reason and first-output boundary; a client barrier or first CUDA graph
   is insufficient attribution.
4. **Exercise lifecycle and semantic controls.**
   Cover unequal prompts, arrival offsets, row/tile boundaries, prefix hits and misses,
   cancellation, draining, resumed chunks, schema/tool termination, and independent coding
   edge cases. Check no cross-request KV contamination, missing rows or duplicate ownership.
5. **Gate the latency policy separately.**
   Retain C1 and mixed short/long workloads, per-request first-text/total distributions and
   group completion in both orders. Batching's long-request median regression above must not
   disappear into aggregate throughput. A future queue policy needs an explicit latency
   objective and observed admission behavior, not an unconditional enablement.

The final bounded arrival/lifecycle diagnostic uses the same adaptive/canonical pair and
retains timestamp windows plus actual server dispatch lines:

| Arrivals | Observed candidate membership |
|---|---|
| Simultaneous 1,111 + 64 + 64 + 64 input tokens | One four-request/1,303-token prefill |
| Long first, others delayed 25/60/120 ms | No kernel-batched prefill; serial behavior |
| Short first, others delayed 25/60/120 ms | No kernel-batched prefill; serial behavior |
| Four short prompts delayed 0/5/20/40 ms | First two batched over 128 tokens; remaining requests serial |

All sixteen eight-output responses in each arm pass prompt/count/SSE checks; this particular
fixture's text matches across arms. It does not erase earlier unequal-prefix fixture output
changes. Each arm also passes 32/48/64/80-token draining, one intentional client disconnect
with three surviving requests, and a subsequent single request. These are single instrumented
lifecycle observations, not counterbalanced performance samples. The synchronized mixed
cohort illustrates head-of-line tradeoffs: its long request's first text arrived about 923 ms
instead of 809 ms, while its three short requests arrived about 922 ms instead of 1,090–1,591 ms.
That is not evidence that late-arriving short work will receive the same benefit.
