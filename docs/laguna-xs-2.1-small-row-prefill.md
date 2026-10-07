# Laguna small-row expert prefill candidate

Checkpoint: [poolside/Laguna-XS-2.1-NVFP4 at d32afde8b09af1539b49ff96ff5551c674485f8e](https://huggingface.co/poolside/Laguna-XS-2.1-NVFP4/tree/d32afde8b09af1539b49ff96ff5551c674485f8e).
Evidence date: October 7, 2026. This remains an opt-in candidate; the default
kernel and model qualification status are unchanged.

## Why partition expert rows

Actual routed-load captures across 39 MoE layers found that 4,590 of 4,881
nonempty expert/layer combinations had at most 16 rows for the 64-token fixture.
The 1,111-token fixture had 4,048 of 7,694 such combinations, but its largest
expert had 934 rows. A global M16 substitution regressed the long fixture.
The candidate instead partitions experts: counts 1–16 use an M16 tile; larger
counts retain M64. Separate entrypoints avoid allocating large-tile resources
for the small path. Empty experts launch no work inside either kernel.

The source uses one parameterized BF16-MMA body. The existing unfiltered M64
entrypoint remains available. Two added entrypoints are a disjoint pair, not
individually complete expert GEMMs. Kernel inventory records them as a named
row-partition residual; no automatic circuit lowering is claimed. Packed E2M1,
E4M3 scales, FP32 expert scale and the K16 accumulation order are preserved.
The FP8-activation transposed sibling is not used.

## Evidence from the private prototype

The prototype passed 31 constructed/learned-slice comparisons against the frozen
incumbent, including expert counts 0, 1, 15, 16, 17, 63, 64, 65 and 934,
N tails, sentinel padding, reversed launch order and independent integer-dot
controls. Learned checkpoint operands stayed on the host holding the checkpoint.
Wrong fallback-grid and gather controls were detected.

Frozen prototype binary SHA-256:
`1fdd8f2a2d1ce48d794fda5b8d6590756505e3897b9d09642223adbaf535790b`.
Baseline source and binary are recorded in the
[serving comparison](laguna-xs-2.1-performance-diagnostic.md).
Both orders of an A/B/B2/A2 request experiment admitted all 144 cohorts.
Short-prefill latency improved 9.2–10.1%; long-prefill latency improved 3.9–4.6%.
C4 decode was effectively flat in the reverse order, so no C4 decode win is
claimed. Six schema, four JSON/tool and six concurrent-client schema cases
passed. Initial coding remained 9/12, with the same retry-delay failures.

These results apply to the frozen private prototype. They do not automatically
qualify the parameterized checked-in implementation or a combination with any
other optimization.

## Integration and remaining gates

`METRALE_LAGUNA_SMALL_ROW_PREFILL=1` enables the pair only for the measured
2048-hidden, 512-intermediate, 256-expert, top-8 layout. It defaults off.
Unified/transposed layout, the alternate CUTLASS path, missing kernels, an
unexpected N tile or FP8-activation metadata refuse initialization. An average
expert-load cap is refused; the large-row grid must cover the complete maximum.
Resolved handles are stored per layer, with no per-launch host copy or lookup.

Three CPU controls verify dispatch, the 934-row grid, stream/argument identity,
missing-entry and invalid-shape/alignment refusal. The CUDA test
`nvfp4_small_rows_gpu` is deliberately ignored by ordinary CPU test runs.
It checks the compiled family against an independent exact integer-dot oracle,
legacy output, reversed entrypoint order, padding and a deliberately truncated
grid. It ran on the actual GB10 CUDA target and passed in 1.65 seconds.
All seven legacy PTX entry bodies also matched the frozen source after removing
comments/compiler labels and renaming only the two local shared-allocation
symbols introduced by the template; instruction and allocation contents matched.

The exact-source serving candidate (binary SHA-256
`5b6d84ec4ea76555ff5f886b7cc90b8d901b3983dc586eb361b7adbdc65f43f2`)
passed six schemas, four JSON/tool controls and six concurrent-client schemas.
All twelve extracted coding candidates matched the prototype byte-for-byte;
isolated grading remained 9/12. An earlier startup refused a missing new symbol
from a stale shared build artifact. That failure was retained; rebuilding the
owned kernel artifacts and checking source/PTX/embedded-symbol identities
preceded the successful run.

All 144 cohorts passed count/terminal admission in the exact-source A/B/B2/A2
run. Both orders measured 9.68–10.25% lower short-prefill latency and
4.15–4.67% lower long-prefill latency. C4 decode-64 total latency improved
3.80% and 4.64%; this includes prefill and is not a steady-decode claim.
No outliers were removed.

Scoped Linux Clippy passed with warnings denied for both the CPU and CUDA test
targets. The separately measured combined dense/prefill configuration is
recorded in [the dense-row report](laguna-xs-2.1-dense-small-rows.md). The full
concurrency ladder, broad coding/agentic quality and overall competitive prefill
remain open.
