# Kernel performance map

The living map for taking every GPU kernel of the engine to its hardware limit. For each kernel
entry point it records where it lives, which models launch it, which architectural component it
serves, what was traded away to get it where it is (with the pull requests behind it), and how far
its measured time is from the roofline floor of the hardware it runs on.

Most of this file is **generated** by [`scripts/kernel_perf.py`](scripts/kernel_perf.py) from the
tree itself plus three curated inputs under [`docs/kernel-perf/`](docs/kernel-perf/). CI runs
`python3 scripts/kernel_perf.py --check` in the cheap-checks job, so a kernel added, removed,
renamed or rewired without regenerating this file fails the PR. Edit the inputs, never the
generated block.

**Where to look**

| You want | Section |
|---|---|
| What is in the tree, in numbers | [Inventory at a glance](#inventory-at-a-glance) |
| Which checkpoints form each family | [Architecture families](#architecture-families) |
| Per-component counts (primary, launched from, unique, dead, measured) | [Components](#components) |
| Kernels every decoder architecture runs | [Shared by all LLM architectures](#shared-by-all-llm-architectures) |
| Every kernel a component launches | [Kernels by component](#kernels-by-component) |
| Kernels only one component launches | [Unique kernels by component](#unique-kernels-by-component) |
| Compiled code no engine path launches | [Compiled but not launched](#compiled-but-not-launched) |
| The trade-off notes behind the "Trade-offs" cells | [`docs/kernel-perf/TRADEOFFS.md`](docs/kernel-perf/TRADEOFFS.md) |
| Measured regimes at a glance | [Measurements](#measurements) |
| Every measured row behind a "% of floor" cell | [`docs/kernel-perf/MEASUREMENTS.md`](docs/kernel-perf/MEASUREMENTS.md) |

## Methodology

### 1. Inventory: derived from the tree, never typed in

1. **Targets.** Every `(hardware, model, quant)` target is enumerated and resolved with
   [`scripts/lib/kernel_layout.py`](scripts/lib/kernel_layout.py), the Python mirror of the
   resolver the build uses (`crates/closure/src/layout.rs`, held to the same answer on the real
   tree by `crates/kernels/tests/layout_mirror.rs`). `[sources] use`, `[hardware] inherits`,
   `[model] kernel_source` redirects and declared `[shadow]` forks are therefore applied exactly as
   `crates/kernels/build.rs` applies them. A source file that many model directories or hardware
   trees compile is **one row** listing all of them; a fork (a same-stem file with different bytes,
   see [`kernels/FORKS.md`](kernels/FORKS.md)) is a separate row because it is separate code.
2. **Modules.** A compiled file's module is its stem, renamed by the `[modules]` tables of the
   target's `KERNEL.toml`s merged least-specific first, as the build does. The table shows
   `module::function`, the pair the engine passes to `gpu.kernel(module, function)`.
3. **Entry points.** [`scripts/lib/kernel_perf_scan.py`](scripts/lib/kernel_perf_scan.py) reads
   each compiled source and the local headers it includes, with comments stripped, and takes every
   `extern "C" __global__` function (CUDA/HIP) or `kernel` function (Metal). Names built by the
   preprocessor are expanded: function-like macros (`WYN_INSTANTIATE(K)` → `gated_delta_rule_wy##K`),
   object-like names defined by the including file before a shared body (`#define KERNEL_NAME` +
   `prefill_paged_compute.cuh`), and pasting helpers (`PAGED_CONCAT(KERNEL_NAME, _64)`). A kernel
   whose body lives in a header is listed under the header, with the compiled `.cu` recorded as
   its source. `static` and template `__global__` functions are not loader-visible and are not
   rows. The scan was checked on 2026-09-27 against the `.entry` directives of a full nvcc PTX
   build of all 32 gb10 targets: the only differences were entry points added or removed in
   commits after that build and the six mangled internal kernels of the vendored MMQ code.
4. **Call sites.** Every string literal in `crates/**/*.rs` equal to an entry-point name is a call
   site; a `{}`-templated literal (`format!("w4a16_gemv_sw_moe_batchm_m{r}")`) counts for the names
   it matches when no exact literal exists. When the call spells its module (a literal or a
   `const X: &str`), the site is attached only to the entry points of that module, which
   separates same-named kernels in different files (the ten `paged_decode_attn_splitk_nvfp4`s).
   Sites in `tests/`, `*_tests.rs`, inline `#[cfg(test)] mod` blocks and `examples/` are counted
   but never make a kernel "used"; an entry point with no other site is listed under
   [Compiled but not launched](#compiled-but-not-launched).

### 2. Families and components: curated, and validated against the tree

[`docs/kernel-perf/taxonomy.toml`](docs/kernel-perf/taxonomy.toml) holds the only hand-written
classification:

- **Architecture families** group model directories by architecture (all model directories of the
  same name on every hardware tree belong to the same family). Each lists the
  architecture-specific components its layers have.
- **Components** are the architectural pieces the code is organised around: attention, MLA,
  sparse attention, GDN, KDA, Mamba2, causal conv1d, MoE, dense FFN, the projection GEMM/GEMV
  formats, norms, elementwise, RoPE, KV cache, quantization, embedding/LM head, sampling,
  speculative decoding, hyper-connections, n-gram memory, vision, NLLB, LoRA, one-time weight
  load, and diagnostics. A component is *generic* when every decoder family has it (attention,
  norms, projections, ...). Each component names the Rust paths that implement it (`sites`), e.g.
  `crates/model-layers/src/layers/qwen3_ssm/` for GDN.
- **Rules** give each entry point its *primary* component and a short *kind* (first matching rule
  by name/path glob wins).

The generator refuses to render while any entry point matches no rule, a rule matches nothing, a
model directory belongs to no family, or a `sites` path matches no file, so the taxonomy cannot
drift silently from `kernels/` and `crates/`.

### 3. "LLMs" column: which checkpoints launch a kernel

For each engine call site of a kernel, a family is admitted when all four hold:

1. a target of the family **compiles** the file (from the resolver, step 1.1);
2. the kernel's primary component is generic or **one the family has**;
3. the component owning the call-site path is generic or one the family has (a GEMV launched from
   the GDN layer counts only for GDN families, the same GEMV launched from the attention layer for
   every family);
4. when the path is **family-exclusive** (`[[family_site]]`, e.g. `crates/model-arch/src/kimi_k3`),
   the family is one of those.

A family whose forward pass does not use the shared transformer path (`shared_engine = false`:
NLLB) is admitted only through its exclusive paths. The cell names the families and the number of
checkpoints they contain; the [families table](#architecture-families) maps each to its model
directories and checkpoint ids. This is **reachability**, not a launch trace: a kernel is listed
for every family that compiles it and reaches its call site. Which checkpoint actually launches it
in a given run can narrow further by runtime choices (KV-cache dtype, head_dim, batch width, the
`METRALE_*` levers and `[defaults]` rows of `HARDWARE.toml`), which a profile of that run shows.
"none — its callers' targets compile another copy" marks a file whose call sites exist but whose
every caller resolves the same name to a different (forked) file.

### 4. Shared and unique

- **Shared by all LLM architectures** = used (step 3) by *every* decoder family (every family with
  `shared_engine = true`; there are 14). NLLB is an encoder-decoder with its own self-contained
  kernel set, so it is outside the quorum; a row also names it when NLLB uses the kernel.
- **Kernels by component** lists, under each component, a full row for every entry point whose
  *primary* component it is, plus the names of other components' entry points that the
  component's code launches (their full rows are under their own component).
- **Unique to a component** = every engine call site of the entry point belongs to that one
  component (the owner of the call-site path, or the primary component where no component owns
  the path).

### 5. Trade-offs and PRs

[`docs/kernel-perf/tradeoffs.toml`](docs/kernel-perf/tradeoffs.toml) holds one entry per known
trade-off, keyed by source file (applies to every entry point the file defines or compiles) or by
`file::function`: what was given up for what, known limits (registers, occupancy, spills, shape
restrictions, precision and bit-exactness across batch widths), and dated measurements, each with
its source (a code comment by `file:line`, a pull request, a perf document, or a dated
measurement note). PR numbers are pull requests of this repository only. The generator rejects an
entry whose key no longer names a kernel, so deleting or renaming a kernel forces its notes to be
moved or dropped. The notes are rendered to
[`docs/kernel-perf/TRADEOFFS.md`](docs/kernel-perf/TRADEOFFS.md); table cells link there and to the
PRs.

### 6. "% of floor": how far a kernel is from the hardware limit

For one kernel call in one regime:

```
floor_us     = max(bytes / BW_peak, flops / FLOP_peak(dtype))
pct_of_floor = floor_us / time_us * 100        (100 % = at the roofline floor)
```

- `time_us` is the **measured** median device time of the call (Nsight Systems kernel trace of a
  real serve run in the named regime, one row per kernel per regime).
- `bytes` is the **minimum** DRAM traffic the call must move, from the call's shapes: every weight
  byte it reads once (packed format including block scales), every activation / KV / state byte it
  must read or write once. Re-reads a better kernel could avoid are *not* counted, so cache misses
  and redundant passes show up as distance from the floor.
- `flops` is the **useful** arithmetic (2·M·N·K for a GEMM, the attention and recurrence FLOPs the
  math requires), at the precision the tensor cores execute.
- `bound` names the term that sets the floor. The peaks are the ones in
  [Peaks (GB10)](#peaks-gb10) below.
- **Above 100 %** means the kernel beat the modelled floor: the byte model counts traffic the
  hardware served from L2 (data reused across back-to-back calls), or the kernel streams faster
  than the peak in use. Treat it as a prompt to re-check that row's byte model, not as headroom
  below zero.
- **"not measured"** means no row exists for that kernel yet. It is never estimated, interpolated
  or copied from another kernel, and a kernel measured in one regime only shows that regime.

The summary cell shows the lowest and highest % over the kernel's rows and the regime of the
lowest; [`docs/kernel-perf/MEASUREMENTS.md`](docs/kernel-perf/MEASUREMENTS.md) (generated) has every
row with its shape, source (trace, commit, box, date) and notes. The generator re-derives `pct_of_floor` from `floor_us / time_us` and fails if
a stored value disagrees by more than 0.5 points, and fails on a row naming a kernel the tree does
not have.

#### Peaks (GB10)

Measured on one GB10 with the GPU otherwise idle, 2026-09-28, SM clock 2405–2496 MHz under load,
microbenchmarks built for `sm_121a` (plain `sm_121` rejects the block-scaled FP4 MMA). The floor uses
the **measured achievable** peak, not the datasheet, so 100 % is reachable; the datasheet column
converts (a row's % of the 273 GB/s datasheet bandwidth is `pct × 249/273 = pct × 0.912`).

| Resource | Floor uses | How it was measured | Datasheet / nominal |
|---|---|---|---|
| DRAM read | **249.0 GB/s** | one CTA streams one contiguous 8 KiB tile (512 threads, 16 B `ld.global.nc` per lane) over a 4 GiB buffer, median of 7; in-model GEMVs reach 246–253 GB/s | 273 GB/s (LPDDR5X-8533, 256-bit) |
| DRAM read, grid-stride / write / copy | 236.0 / 196.6 / 214.3 GB/s | streaming `__ldcs` / `__stcs` kernels; not used as the floor | |
| BF16 `mma.sync.m16n8k16`, FP32 acc | **123.7 TFLOPS** | registers only, 8 independent accumulators per warp, best of 48×{1,2,4} CTAs × {4,8} warps | 125 |
| E4M3 `mma.sync.m16n8k32`, FP32 acc | **243.6 TFLOPS** | same | 250 |
| NVFP4 `mma.sync…mxf4nvf4.block_scale.scale_vec::4X.m16n8k64` | **490.8 TFLOPS** | same (SASS `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`) | 500 (dense) |
| FP32 FFMA (CUDA cores) | **30.0 TFLOPS** | 8 independent FFMA chains; FMUL+FADD (the `--fmad=false` form) reaches 14.9 | 30.1 |

One bandwidth covers reads and writes, which flatters write-heavy kernels: a pure copy tops out at
86 % of its floor and a pure write at 79 %, so an elementwise kernel near 85 % is at the practical
limit. Tensor-core peaks need at least 8 resident warps per SM (4 warps on one CTA per SM reach 115
BF16 / 448 FP4 TFLOPS). Hopper, B200/B300, Strix and Metal have no measured peaks yet, so every
kernel on those targets is "not measured".

**Peak class.** `PEAK[class]` is the tensor-core rate of the operand formats the kernel
*implements*, not of the instruction it happens to issue: `fp8` for W8A8 and W4A8 (NVFP4 weight
dequantized to E4M3 × E4M3 activations), `fp4` for W4A4 NVFP4 × NVFP4, `bf16` for W8A16 / W4A16 /
BF16 GEMMs, attention and GDN matmuls, `fp32` only for the FP32-state GDN recurrent decode. So a W8A8
kernel that decodes E4M3 to BF16 in software and issues BF16 MMAs is judged against the FP8 peak,
and the exact-order FP32 router GEMMs against the BF16 peak (their notes say so). A floor never
assumes a lower precision than the kernel implements; a lever that changes the number format
(NVFP4 experts, W4A4 downcast) changes `bytes` itself, not the floor of the old kernel.

**No latency term.** Launch-latency-bound kernels (single-CTA sorts and worklists, top-k, argmax,
1–4-row norms) read near 0 %: their gap is recovered by fusing or removing the launch, not by
bandwidth. A row's `notes` give its share of the regime's GPU time, which says whether it matters.

#### Byte and FLOP models

The shape of each call is recovered from its launch grid through the launcher's grid formula
(`crates/model-layers/src/layers/ops/*.rs`), then:

| Family | Bytes of one call | FLOPs, class |
|---|---|---|
| FP8 block-scaled / unscaled FP8 GEMM | N·K weight + block scales + M·K activations (+ scales) + 2·M·N out | 2MNK, fp8 |
| W8A16 GEMM/GEMV | E4M3 weight + scales + 2·M·K + 2·M·N | 2MNK, bf16 |
| NVFP4 W4A16 / W4A8 GEMM/GEMV | 0.5625·N·K (E2M1 + UE4M3 per 16) + 2·M·K + 2·M·N | 2MNK, bf16 or fp8 |
| NVFP4 W4A4 MMQ | 0.5625·(N·K + M·K) + 2·M·N | 2MNK, fp4 |
| BF16 GEMV/GEMM | 2·N·K + 2·M·K + 2·M·N | 2MNK, bf16 |
| MoE grouped prefill (W8A8) | all routed experts' weights + scales, activations once, outputs once | 2·(top-k·T)·N·K, fp8 |
| MoE grouped decode | (distinct experts + shared) × expert bytes + activations; the distinct-expert count D comes from a routing log (R = 2 → 14.5, 8 → 45, 16 → 72, 32 → 121.5, 64 → 153) because the grid is a capacity | 2·rows·N·K, bf16 |
| Attention prefill | Q + O + K/V once ((prefix + chunk) · kv heads · head_dim · 2 · 2 B, BF16 KV) | 4·hd·nq·(T·P + T(T+1)/2), bf16 |
| Attention decode | each sequence's K and V once, plus q and o | 4·rows·nq·L·hd, bf16 |
| GDN chunked prefill (chunk 64) | per stage: its inputs once, its outputs (W/U, per-chunk states, o) once | per (chunk, v-head) matmul FLOPs, bf16 |
| GDN decode | recurrent state read + write per sequence (FP32 or FP16 pool), read-only for the lazy carried-state kernels whose write-back is deferred, plus q/k/v/o | 8·rows·nv·dk², fp32 |
| Conv, norms, RoPE, residual, cache writes | one read and one write of each tensor (RoPE: rotary dims only) | — |

`bytes` counts every weight byte (scales included) once, every activation input once, every output
once, and recurrent/KV state once (read, and written where the step must persist it). Re-reads a
kernel actually does (per-M-tile weight re-streaming, activation re-reads per N tile, L2 misses) are
not counted, so they show up as distance from the floor. `flops` excludes padding (a 64-row MMA
tile carrying 32 live rows), dequantization arithmetic and softmax work.

#### Regimes

A regime names the serving situation the time was taken in: `decode C=<concurrent sequences>`
with the speculative verify rows `R`, `prefill <prompt tokens>` for a single cold request, and the
model. A kernel's efficiency depends strongly on the regime (a decode GEMV at C=1 and C=16 moves the
same weights for up to 16x the useful work), so rows are never merged across regimes; a kernel that
runs at several shapes in one regime gets one row per shape (`regime · <shape>`). Measured so far,
all on GB10 at main `e37e3cb2` (kernel-equivalent binary), `nsys --trace=cuda --cuda-graph-trace=node`:

| Model | Regime | Serving configuration |
|---|---|---|
| Qwen/Qwen3.6-35B-A3B-FP8 | decode C=1, R=2 | concurrency-sweep flags (BF16 KV, MTP 1 draft, forced), 3 s window of a 1000-token greedy essay, KV ≈ 585 |
| Qwen/Qwen3.6-35B-A3B-FP8 | decode C=16, R=32 | same, 16 concurrent, KV ≈ 273 |
| Qwen/Qwen3.6-35B-A3B-FP8 | prefill 4k (4549 tok) and 32k (32772 tok), cold | TTFT-gate recipe (FP8 weights, BF16 LM head, BF16 KV) |
| unsloth/Qwen3.8-27B-NVFP4 | decode C=1, R=4 | decode-floor recipe (BF16 KV and LM head, 3 drafts), KV ≈ 245 |
| unsloth/Qwen3.8-27B-NVFP4 | decode C=16, R=32 | concurrency-sweep recipe (FP8 KV, FP16 SSM pool, batched recurrent), KV ≈ 308 |
| unsloth/Qwen3.8-27B-NVFP4 | prefill 4k (4103 tok) and 32k (32772 tok), cold | decode-floor recipe flags |

Between 99.8 % and 99.9 % of each regime's GPU kernel time has a floor model; a (kernel, shape) is
written as a row when it is at least 0.3 % of its regime's kernel time. Each window's purity is
checked from the grids (for example grouped `gridY = 264` means R = 32).

**Caveats.** The profiler inflates decode steps by about 3 % at C=1, about 8 % (dense) and 12–19 %
(MoE) at C=16, mostly inside kernel durations, so C=16 percentages are *understated* by up to that
factor; prefill inflation is 1–2 %. Per-call time is the median over the window's calls of that
(kernel, shape). Decode KV lengths are short (245–585 tokens), so long-context decode attention is a
separate, not yet measured regime. Byte counts are modelled, not counted; an `ncu` DRAM-bytes pass
over the largest rows would validate them.

### 7. Updating

```bash
python3 scripts/kernel_perf.py            # regenerate KERNEL-PERF.md, docs/kernel-perf/{TRADEOFFS,MEASUREMENTS}.md
python3 scripts/kernel_perf.py --check    # what CI runs: stale file or taxonomy drift -> exit 1
python3 scripts/kernel_perf.py --json     # the joined inventory (targets, call sites, families) as JSON
python3 scripts/kernel_perf_test.py       # generator self-test on a fixture tree (also run by CI)
```

- **A kernel was added, removed or renamed:** run the generator. If it reports `no rule
  classifies`, add or widen a rule in `taxonomy.toml`; if it reports a trade-off or measurement
  naming nothing, move or delete that entry.
- **A new model directory:** add it to its family in `taxonomy.toml` (or add a family).
- **A trade-off was made or learned:** add a `[[t]]` entry to `tradeoffs.toml` with its source
  and PR.
- **A kernel was measured:** append `[[m]]` rows to `measurements.toml` (schema below) and
  regenerate. Replace a kernel's rows for a regime when it is re-measured; keep the source line
  exact (trace, commit, box, date) so the number can be reproduced.

```toml
[[m]]
kernel = "<module>::<function>"        # as the loader names it
file = "kernels/gb10/common/x.cu"      # the defining file, or the .cu that compiles a header body
hardware = "gb10"
model = "Qwen/Qwen3.6-35B-A3B-FP8"
regime = "decode C=16 (R=32)"           # or "prefill 32k", "decode C=1", ...
time_us = 0.0                           # measured, per call (median)
bytes = 0                               # minimum DRAM bytes the call must move (model)
flops = 0                               # useful FLOPs (model)
bound = "memory"                        # or "compute"
floor_us = 0.0                          # max(bytes/BW_peak, flops/FLOP_peak) with the peaks above
pct_of_floor = 0.0                      # floor_us / time_us * 100
source = "nsys <file> @ <commit>, <box>, <date>"
notes = ""
```

<!-- kernel_perf.py: BEGIN GENERATED (edit the inputs, then run scripts/kernel_perf.py) -->

## Inventory at a glance

- **1406 kernel entry points** in **358 source files** across 7 hardware trees (b200, b300, gb10, hopper, metal, strix, strix-hip), compiled into 59 (hardware, model, quant) targets.
- **1144** have at least one engine call site; **262** are compiled but launched only from tests, examples or not at all (see [Compiled but not launched](#compiled-but-not-launched)).
- **16 architecture families**, **29 components**.
- **61** entry points have a measured % of floor; every other row reads “not measured”.

## Architecture families

| Label | Family | Checkpoints (model directory → checkpoint) | Components | Entry points used |
|---|---|---|---|---|
| Qwen-GDN | Qwen3.x GDN hybrid, dense FFN | `qwen3.5-27b` → Kbenkhaled/Qwen3.5-27B-NVFP4<br>`qwen3.6-27b` → Qwen/Qwen3.6-27B<br>`qwen3.8-27b` → Qwen/Qwen3.8-27B<br>`holo-3.1-0.8b` → Hcompany/Holo-3.1-0.8B<br>`holo-3.1-4b` → Hcompany/Holo-3.1-4B<br>`ornith-1.0-9b` → deepreinforce-ai/Ornith-1.0-9B<br>`qwen3-5-4b-vlm-mlx-int8` → mlx-community/Qwen3.5-4B-MLX-8bit | GDN, Causal conv1d, Dense FFN, Vision encoder | 598 |
| Qwen-GDN-MoE | Qwen3.x GDN hybrid, MoE (incl. Qwen3-Next) | `qwen3.5-35b-a3b` → Sehyo/Qwen3.5-35B-A3B-NVFP4<br>`qwen3.5-122b-a10b` → Sehyo/Qwen3.5-122B-A10B-NVFP4<br>`qwen3.5-397b-a17b` → nvidia/Qwen3.5-397B-A17B-NVFP4<br>`qwen3.6-35b-a3b` → Qwen/Qwen3.6-35B-A3B-FP8<br>`holo-3.1-35b-a3b` → Hcompany/Holo-3.1-35B-A3B-NVFP4<br>`qwen3-next-80b-a3b` → nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4 | GDN, Causal conv1d, MoE, Dense FFN, Vision encoder | 647 |
| Qwen3.8-FN | Qwen3.8-Flash-Next (GDN + QSA sparse attention + mHC + PLE + MoE) | `qwen3.8-flash-next` → Qwen/Qwen3.8-Flash-Next | GDN, Causal conv1d, Sparse / compressed attention, Hyper-connections, N-gram and memory embeddings, MoE, Dense FFN, Vision encoder | 549 |
| Qwen3-VL | Qwen3-VL MoE (full attention) | `qwen3-vl-30b-a3b` → ig1/Qwen3-VL-30B-A3B-Instruct-NVFP4 | MoE, Vision encoder | 376 |
| Gemma4 | Gemma 4 (sliding/full attention, dense and MoE) | `gemma-4-26b-a4b` → bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4A16<br>`gemma-4-31b` → nvidia/Gemma-4-31B-IT-NVFP4 | MoE, Dense FFN, Vision encoder | 404 |
| Nemotron-H | Nemotron-H (Mamba2 hybrid + MoE) | `nemotron-3-nano-30b-a3b` → nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4<br>`nemotron-super-120b-a12b` → nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4<br>`nemotron-labs-3-puzzle-75b-a9b` → nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-NVFP4 | Mamba2, Causal conv1d, MoE, Dense FFN | 436 |
| DeepSeek-V4 | DeepSeek-V4 (MLA + CSA/HCA + mHC + MoE + Engram) | `deepseek-v4-flash` → RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8<br>`deepseek-v4.1-flash` → deepseek-ai/DeepSeek-V4.1-Flash | MLA, Sparse / compressed attention, Hyper-connections, N-gram and memory embeddings, MoE, Dense FFN | 474 |
| Mistral4 | Mistral Small 4 (MLA + MoE) | `mistral-small-4` → mistralai/Mistral-Small-4-119B-2603-NVFP4 | MLA, MoE, Dense FFN | 383 |
| GLM-5.3 | GLM-5.3-Flash (KDA + DSA sparse MLA + mHC + MoE) | `glm-5.3-flash` → LibertAIDAI/GLM-5.3-Flash-NVFP4 | KDA, Causal conv1d, MLA, Sparse / compressed attention, Hyper-connections, MoE, Dense FFN, Vision encoder | 430 |
| Kimi-K3 | Kimi K3 (KDA + gated MLA + LatentMoE) | `kimi-k3` → inference-optimization/Kimi-K3-0.40B | KDA, Causal conv1d, MLA, MoE, Dense FFN | 389 |
| GPT-OSS | GPT-OSS (experimental eager C1, sink attention + MXFP4 MoE) | `gpt-oss-20b` → openai/gpt-oss-20b | MoE | 15 |
| Laguna | Laguna (full/sliding attention + MoE) | `laguna-s-2.1` → poolside/Laguna-S-2.1-NVFP4<br>`laguna-xs-2.1` → poolside/Laguna-XS-2.1-NVFP4 | MoE, Dense FFN | 377 |
| MiniMax-M2 | MiniMax-M2 (full attention + sigmoid MoE) | `minimax-m2-229b` → MiniMaxAI/MiniMax-M2.7 | MoE, Dense FFN | 376 |
| Step-3.7 | Step-3.7-Flash (full/sliding attention + sigmoid MoE) | `step3p7-flash` → stepfun-ai/Step-3.7-Flash-NVFP4 | MoE, Dense FFN, Vision encoder | 375 |
| LongCat | LongCat-Flash-Lite (MLA + MoE + n-gram embeddings) | `longcat-flash-lite` → meituan-longcat/LongCat-Flash-Lite | MLA, MoE, N-gram and memory embeddings, Dense FFN | 396 |
| NLLB | NLLB-200 (encoder-decoder translation) | `nllb-200-3.3b` → facebook/nllb-200-3.3B | Encoder-decoder translation | 28 |

## Components

| Component | Scope | Primary entry points | Launched from it (incl. other primaries) | Unique to it | Not launched | Measured |
|---|---|---|---|---|---|---|
| Attention (GQA/MHA: paged decode, split-K, prefill/flash) | every family | 105 | 337 | 180 | 37 | 20 |
| MLA (multi-head latent attention) | families listing it | 34 | 34 | 0 | 7 | 0 |
| Sparse / compressed attention (DSA, CSA/HCA, QSA) | families listing it | 43 | 61 | 44 | 0 | 3 |
| GDN (gated delta rule linear attention) | families listing it | 214 | 362 | 227 | 11 | 27 |
| KDA (Kimi delta attention, linear attention) | families listing it | 11 | 24 | 10 | 3 | 5 |
| Mamba2 (selective state-space scan) | families listing it | 6 | 63 | 7 | 1 | 8 |
| Causal conv1d (short convolution of GDN/KDA/Mamba2) | families listing it | 10 | 10 | 1 | 3 | 2 |
| MoE (routing, dispatch, expert GEMM/GEMV, combine) | families listing it | 198 | 296 | 203 | 46 | 22 |
| Dense FFN (gate/up/down projections of non-MoE layers) | families listing it | 0 | 88 | 27 | 0 | 12 |
| Projection GEMM/GEMV — BF16/F32 | every family | 27 | 27 | 6 | 1 | 4 |
| Projection GEMM/GEMV — FP8 (W8A16, W8A8, block-scaled) | every family | 87 | 87 | 21 | 1 | 6 |
| Projection GEMM/GEMV — NVFP4 W4A16 | every family | 63 | 63 | 17 | 10 | 7 |
| Projection GEMM/GEMV — W4A4 (FP4 activations) | every family | 22 | 22 | 11 | 0 | 2 |
| Projection GEMM/GEMV — integer / K-quant (Q2_0, Q2_K..Q6_K, INT8, MLX INT8) | every family | 23 | 23 | 6 | 31 | 0 |
| Normalization (RMSNorm, LayerNorm, L2, gated norms) | every family | 65 | 65 | 1 | 40 | 4 |
| Activations and elementwise (SiLU/GELU/ReLU², residual, gates, scale) | every family | 17 | 17 | 0 | 16 | 3 |
| Positional encoding (RoPE, YaRN, MRoPE) | every family | 14 | 14 | 0 | 0 | 1 |
| KV cache (write, quantize, TurboQuant rotation, slot metadata) | every family | 36 | 36 | 1 | 2 | 0 |
| Quantization and format conversion | every family | 52 | 52 | 8 | 7 | 4 |
| Embedding and LM head (lookup, overlays, softcap, scale) | every family | 11 | 18 | 7 | 6 | 0 |
| Sampling (argmax, top-p, feed-forward of the chosen token) | every family | 7 | 7 | 4 | 1 | 1 |
| Speculative decoding (MTP heads, DFlash drafter, verify helpers) | every family | 3 | 97 | 12 | 0 | 12 |
| Hyper-connections (mHC) | families listing it | 27 | 27 | 13 | 2 | 0 |
| N-gram and memory embeddings (Engram, PLE, n-gram tables) | families listing it | 5 | 16 | 6 | 0 | 2 |
| Vision encoder (ViT towers) | families listing it | 32 | 36 | 32 | 3 | 0 |
| Encoder-decoder translation (NLLB, self-contained kernel set) | families listing it | 26 | 29 | 24 | 28 | 1 |
| LoRA adapters (BGMV shrink/expand) | every family | 6 | 6 | 6 | 0 | 0 |
| Weight load and repack (one-time, not per token) | every family | 0 | 76 | 25 | 0 | 4 |
| Diagnostics and microtests (no serving path) | every family | 0 | 0 | 0 | 6 | 0 |

## Shared by all LLM architectures

Entry points used by **every one of the 14 decoder families** (every family with `shared_engine = true`; NLLB, the self-contained encoder-decoder, is excluded from the quorum and named when it also uses the kernel). 238 entry points qualify; 0 of them are used by all 16 families.

| Kernel (module::function) | File | Component · kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| argmax::`argmax_bf16` | [gb10/common/argmax_bf16.cu:14][f5] | Sampling · argmax / top-p | b200 b300 gb10 hop strix hip | all 14 decoder families + NLLB (31 ckpts) | [1 note][t5] | [2%][m5.argmax_bf16] (decode C=1 (R=4, MTP k=3)) |
| argmax::`argmax_{bf16_batch, bf16_batch_lp, fp32}` (3) | [gb10/common/argmax_bf16.cu:68][f5] | Sampling · argmax / top-p | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [3 notes][t5] | not measured |
| argmax_feed::`argmax_bf16_batch_feed`, `feed_resolve` | [gb10/common/argmax_feed.cu:44][f6] | Sampling · argmax / top-p | b200 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t6] | not measured |
| attn_prefill::`attn_prefill` | [gb10/common/attn_prefill.cu:78][f7] | Attention · prefill (flash) | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [2 notes][t7] | not measured |
| attn_prefill::`attn_prefill_64` | [gb10/common/attn_prefill.cu:562][f7] | Attention · prefill (flash) | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [2 notes][t7] | [27%][m7.attn_prefill_64] (prefill 32k (cold, 32772 tok)) |
| attn_prefill_fa128::`attn_prefill_{fa128, fa128_paged}` (2) | [gb10/common/attn_prefill_fa128.cu:356][f8] | Attention · prefill (flash) | b200 gb10 hop | all 14 decoder families (30 ckpts) | — | not measured |
| attn_prefill_h128::`attn_prefill_h128` | [gb10/common/attn_prefill_h128.cu:46][f10] | Attention · prefill (flash) | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [1 note][t10] | not measured |
| bf16_add::`bf16_add_inplace` | [gb10/common/bf16_add.cu:8][f12] | Activations and elementwise · activation / gate / residual | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t12] | not measured |
| gemm::`dense_gemm_bf16` | [gb10/common/dense_gemm_bf16.cu:26][f14] | Projection GEMM/GEMV — BF16/F32 · BF16/F32 GEMM/GEMV | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [1 note][t14] | [1–2%][m14.dense_gemm_bf16] (prefill 32k (cold, 32772 tok)) |
| gemm::`dense_gemm_bf16_f32out` | [gb10/common/dense_gemm_bf16.cu:85][f14] | Projection GEMM/GEMV — BF16/F32 · BF16/F32 GEMM/GEMV | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [1 note][t14] | not measured |
| gemm::`dense_gemm_bf16_pipelined` | [gb10/common/dense_gemm_bf16.cu:446][f14] | Projection GEMM/GEMV — BF16/F32 · BF16/F32 GEMM/GEMV | b200 b300 gb10 hop strix | all 14 decoder families + NLLB (31 ckpts) | [1 note][t14] | not measured |
| gemm_splitk::`dense_gemm_splitk_{partial, reduce}` (2) | [gb10/common/dense_gemm_splitk.cu:27][f15] | Projection GEMM/GEMV — BF16/F32 · BF16/F32 GEMM/GEMV | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t15] | not measured |
| gemm_tc::`dense_gemm_{tc, tc_scaled_acc}` (2) | [gb10/common/dense_gemm_tc.cu:185][f16] | Projection GEMM/GEMV — BF16/F32 · BF16/F32 GEMM/GEMV | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [2 notes][t16] | not measured |
| gemv::`dense_gemv_bf16` | [gb10/common/dense_gemv_bf16.cu:33][f17] | Projection GEMM/GEMV — BF16/F32 · BF16/F32 GEMM/GEMV | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [3 notes][t17] | [94–100%][m17.dense_gemv_bf16] (decode C=1 (R=4, MTP k=3)) |
| dense_gemv_bf16_batchm::`dense_gemv_bf16_batchm` | [gb10/common/dense_gemv_bf16_batchm.cu:226][f19] | Projection GEMM/GEMV — BF16/F32 · BF16/F32 GEMM/GEMV | b200 b300 gb10 hop | all 14 decoder families + GPT-OSS (31 ckpts) | [2 notes][t19] | [16–92%][m19.dense_gemv_bf16_batchm] (decode C=16 (R=32, MTP k=1)) |
| dense_gemv_bf16_tc::`dense_gemv_bf16_tc16` | [gb10/common/dense_gemv_bf16_tc.cu:251][f20] | Projection GEMM/GEMV — BF16/F32 · BF16/F32 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | [4 notes][t20] · [#1][pr1] | [86–95%][m20.dense_gemv_bf16_tc16] (decode C=16 (R=32, MTP k=1)) |
| dense_gemv_bf16_tc::`dense_gemv_bf16_{tc32, tc8}` (2) | [gb10/common/dense_gemv_bf16_tc.cu:250][f20] | Projection GEMM/GEMV — BF16/F32 · BF16/F32 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | [4 notes][t20] · [#1][pr1] | not measured |
| gemv_fp8w::`dense_gemv_fp8w` | [gb10/common/dense_gemv_fp8w.cu:131][f21] | Projection GEMM/GEMV — FP8 · FP8 GEMM/GEMV | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t21] | not measured |
| gemv_fp8w::`quantize_bf16_to_fp8` | [gb10/common/dense_gemv_fp8w.cu:65][f21] | Quantization and format conversion · activation quantize | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t21] | not measured |
| dense_gemv_fp8w_batch2::`dense_gemv_fp8w_batch2` | [gb10/common/dense_gemv_fp8w_batch2.cu:72][f22] | Projection GEMM/GEMV — FP8 · FP8 GEMM/GEMV | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t22] | not measured |
| dequant_fp8_blockscaled_bf16::`dequant_fp8_blockscaled_bf16` | [gb10/common/dequant_fp8_blockscaled_bf16.cu:89][f23] | Quantization and format conversion · dequant / repack / transpose | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [3 notes][t23] | not measured |
| dequant_gguf_bf16::`dequant_{q2_0_gn_to_bf16, q2_k_to_bf16, q3_k_to_bf16, q4_k_to_bf16, q6_k_to_bf16, q8_0_to_bf16}` (6) | [gb10/common/dequant_gguf_bf16.cu:43][f24] | Quantization and format conversion · dequant / repack / transpose | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t24] | not measured |
| dequant_nvfp4_bf16::`dequant_nvfp4_to_bf16` | [gb10/common/dequant_nvfp4_bf16.cu:50][f25] | Quantization and format conversion · dequant / repack / transpose | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t25] | not measured |
| dflash2::`dflash2_{conv2, selector_walk, topk16}` (3) | [gb10/common/dflash2.cu:31][f26] | Speculative decoding · DFlash drafter | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [3 notes][t26] | not measured |
| embed_from_argmax::`batched_embed`, `embed_from_argmax` | [gb10/common/embed_from_argmax.cu:17][f29] | Embedding and LM head · embedding / LM head | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t29] | not measured |
| fp8_gemm_blockscaled_pipe::`fp8_gemm_blockscaled_pipe_128x64` | [gb10/common/fp8_gemm_blockscaled_pipe.cu:63][f30] | Projection GEMM/GEMV — FP8 · FP8 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | — | not measured |
| fp8_gemm_t_blockscaled::`fp8_gemm_t_blockscaled` | [gb10/common/fp8_gemm_t_blockscaled.cu:113][f31] | Projection GEMM/GEMV — FP8 · FP8 GEMM/GEMV | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [3 notes][t31] | [14–49%][m31.fp8_gemm_t_blockscaled] (prefill 32k (cold, 32772 tok)) |
| fp8_gemv_rt::`fp8_gemv_rowscale_{batch16_rt2, batch8_rt2}` (2) | [gb10/common/fp8_gemv_rt.cu:156][f32] | Projection GEMM/GEMV — FP8 · FP8 GEMM/GEMV | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t32] | not measured |
| fp8_scale_transpose::`fp8_act_scale_to_kmajor` | [gb10/common/fp8_scale_transpose.cu:35][f33] | Quantization and format conversion · dequant / repack / transpose | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t33] | not measured |
| fused_k_norm_rope_cache::`fused_k_norm_rope_{cache_write_bf16, mrope_cache_write_bf16}` (2) | [gb10/common/fused_k_norm_rope_cache.cu:53][f34] | KV cache · cache write | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t34] | not measured |
| lora_bgmv::`lora_bgmv_{expand_fold, shrink}` (2) | [gb10/common/lora_bgmv.cu:50][f66] | LoRA adapters · BGMV shrink/expand | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t66] | not measured |
| metadata_fill::`fill_slots_from_block_table` | [gb10/common/metadata_fill.cu:5][f69] | KV cache · cache write | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t69] | not measured |
| moe_lora_gather_bgmv::`moe_lora_gather_bgmv_{expand_fold, shrink}` (2) | [gb10/common/moe_lora_gather_bgmv.cu:57][f83] | LoRA adapters · BGMV shrink/expand | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t83] | not measured |
| moe_lora_grouped_down::`moe_lora_grouped_down_{expand_fold, shrink}` (2) | [gb10/common/moe_lora_grouped_down.cu:67][f84] | LoRA adapters · BGMV shrink/expand | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t84] | not measured |
| paged_decode::`paged_decode_attn` | [gb10/common/paged_decode_attn.cu:343][f119] | Attention · paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t119] | [11–44%][m119.paged_decode_attn] (decode C=1 (R=2, MTP k=1)) |
| paged_decode_attn_bf16_gqa::`paged_decode_attn_bf16_gqa` | [gb10/common/paged_decode_attn_bf16_gqa.cu:46][f120] | Attention · paged decode | b200 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t120] | not measured |
| paged_decode_bf16k_turbo2v::`paged_decode_attn_bf16k_turbo2v` | [gb10/common/paged_decode_attn_bf16k_turbo2v.cu:85][f121] | Attention · paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t121] | not measured |
| paged_decode_bf16k_turbo2v_128::`paged_decode_attn_bf16k_turbo2v` | [gb10/common/paged_decode_attn_bf16k_turbo2v_128.cu:85][f122] | Attention · paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t122] | not measured |
| paged_decode_bf16k_turbo3v::`paged_decode_attn_bf16k_turbo3v` | [gb10/common/paged_decode_attn_bf16k_turbo3v.cu:91][f123] | Attention · paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t123] | not measured |
| paged_decode_bf16k_turbo3v_128::`paged_decode_attn_bf16k_turbo3v` | [gb10/common/paged_decode_attn_bf16k_turbo3v_128.cu:91][f124] | Attention · paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t124] | not measured |
| paged_decode_bf16k_turbo4v::`paged_decode_attn_bf16k_turbo4v` | [gb10/common/paged_decode_attn_bf16k_turbo4v.cu:84][f125] | Attention · paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t125] | not measured |
| paged_decode_bf16k_turbo4v_128::`paged_decode_attn_bf16k_turbo4v` | [gb10/common/paged_decode_attn_bf16k_turbo4v_128.cu:84][f126] | Attention · paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t126] | not measured |
| paged_decode_fp8::`paged_decode_attn_fp8` | [gb10/common/paged_decode_attn_fp8.cu:72][f127] | Attention · paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [3 notes][t127] | [45%][m127.paged_decode_attn_fp8] (decode C=16 (R=32, MTP k=1)) |
| paged_decode_fp8::`paged_decode_attn_{reduce_fp8, splitk_fp8}` (2) | [gb10/common/paged_decode_attn_fp8.cu:346][f127] | Attention · paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [4 notes][t127] | not measured |
| paged_decode_attn_fp8_gqa::`paged_decode_attn_fp8_gqa` | [gb10/common/paged_decode_attn_fp8_gqa.cu:112][f128] | Attention · paged decode | b200 gb10 hop | all 14 decoder families (30 ckpts) | [2 notes][t128] | not measured |
| paged_decode_fp8k_turbo2v::`paged_decode_attn_fp8k_turbo2v` | [gb10/common/paged_decode_attn_fp8k_turbo2v.cu:101][f129] | Attention · paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t129] | not measured |
| paged_decode_fp8k_turbo2v_128::`paged_decode_attn_fp8k_turbo2v` | [gb10/common/paged_decode_attn_fp8k_turbo2v_128.cu:101][f130] | Attention · paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t130] | not measured |
| paged_decode_fp8k_turbo3v::`paged_decode_attn_fp8k_turbo3v` | [gb10/common/paged_decode_attn_fp8k_turbo3v.cu:112][f131] | Attention · paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t131] | not measured |
| paged_decode_fp8k_turbo3v_128::`paged_decode_attn_fp8k_turbo3v` | [gb10/common/paged_decode_attn_fp8k_turbo3v_128.cu:112][f132] | Attention · paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t132] | not measured |
| paged_decode_fp8k_turbo4v::`paged_decode_attn_fp8k_turbo4v` | [gb10/common/paged_decode_attn_fp8k_turbo4v.cu:99][f133] | Attention · paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t133] | not measured |
| paged_decode_fp8k_turbo4v_128::`paged_decode_attn_fp8k_turbo4v` | [gb10/common/paged_decode_attn_fp8k_turbo4v_128.cu:99][f134] | Attention · paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t134] | not measured |
| paged_decode_turbo2::`paged_decode_attn_turbo2` | [gb10/common/paged_decode_attn_turbo2.cu:80][f136] | Attention · paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [2 notes][t136] | not measured |
| paged_decode_attn_turbo2_128::`paged_decode_attn_turbo2` | [gb10/common/paged_decode_attn_turbo2_128.cu:80][f137] | Attention · paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t137] | not measured |
| paged_decode_attn_turbo3::`paged_decode_attn_{splitk_nvfp4, turbo3}` (2) | [gb10/common/paged_decode_attn_turbo3.cu:110][f138] | Attention · paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [3 notes][t138] | not measured |
| paged_decode_attn_turbo3_128::`paged_decode_attn_{splitk_nvfp4, turbo3}` (2) | [gb10/common/paged_decode_attn_turbo3_128.cu:110][f139] | Attention · paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t139] | not measured |
| paged_decode_turbo3k_turbo8v::`paged_decode_attn_turbo3k_turbo8v` | [gb10/common/paged_decode_attn_turbo3k_turbo8v.cu:119][f140] | Attention · paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t140] | not measured |
| paged_decode_turbo3k_turbo8v_128::`paged_decode_attn_turbo3k_turbo8v` | [gb10/common/paged_decode_attn_turbo3k_turbo8v_128.cu:119][f141] | Attention · paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t141] | not measured |
| paged_decode_attn_turbo4::`paged_decode_attn_{splitk_nvfp4, turbo4}` (2) | [gb10/common/paged_decode_attn_turbo4.cu:95][f142] | Attention · paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t142] | not measured |
| paged_decode_attn_turbo4_128::`paged_decode_attn_{splitk_nvfp4, turbo4}` (2) | [gb10/common/paged_decode_attn_turbo4_128.cu:95][f143] | Attention · paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t143] | not measured |
| paged_decode_attn_turbo4_512::`paged_decode_attn_{splitk_nvfp4, turbo4}` (2) | [gb10/common/paged_decode_attn_turbo4_512.cu:119][f144] | Attention · paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [2 notes][t144] | not measured |
| paged_decode_turbo4k_turbo3v::`paged_decode_attn_turbo4k_turbo3v` | [gb10/common/paged_decode_attn_turbo4k_turbo3v.cu:122][f145] | Attention · paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t145] | not measured |
| paged_decode_turbo4k_turbo3v_128::`paged_decode_attn_turbo4k_turbo3v` | [gb10/common/paged_decode_attn_turbo4k_turbo3v_128.cu:122][f146] | Attention · paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t146] | not measured |
| paged_decode_turbo4k_turbo8v::`paged_decode_attn_turbo4k_turbo8v` | [gb10/common/paged_decode_attn_turbo4k_turbo8v.cu:114][f147] | Attention · paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t147] | not measured |
| paged_decode_turbo4k_turbo8v_128::`paged_decode_attn_turbo4k_turbo8v` | [gb10/common/paged_decode_attn_turbo4k_turbo8v_128.cu:114][f148] | Attention · paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t148] | not measured |
| paged_decode_attn_turbo8::`paged_decode_attn_{splitk_nvfp4, turbo8}` (2) | [gb10/common/paged_decode_attn_turbo8.cu:100][f149] | Attention · paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t149] | not measured |
| paged_decode_attn_turbo8_128::`paged_decode_attn_{splitk_nvfp4, turbo8}` (2) | [gb10/common/paged_decode_attn_turbo8_128.cu:98][f150] | Attention · paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [2 notes][t150] | not measured |
| paged_decode_attn_turbo8_512::`paged_decode_attn_{splitk_nvfp4, turbo8}` (2) | [gb10/common/paged_decode_attn_turbo8_512.cu:125][f151] | Attention · paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [2 notes][t151] | not measured |
| per_token_group_quant_fp8::`per_token_group_quant_fp8` | [gb10/common/per_token_group_quant_fp8.cu:39][f152] | Quantization and format conversion · activation quantize | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [2 notes][t152] | [81–83%][m152.per_token_group_quant_fp8] (prefill 32k (cold, 32772 tok)) |
| prefill_paged::`attn_prefill_{paged, paged_batched, paged_batched_64, paged_fp8, paged_fp8_64, paged_fp8_batched, paged_fp8_batched_64, paged_nvfp4, paged_nvfp4_64, paged_nvfp4_batched, paged_nvfp4_batched_64}` (11) | [gb10/common/prefill_paged_compute.cuh:162][f153] | Attention · prefill (flash) | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [8 notes][t153] | not measured |
| prefill_paged::`attn_prefill_paged_64` | [gb10/common/prefill_paged_compute.cuh:644][f153] | Attention · prefill (flash) | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [6 notes][t153] | [23–24%][m153.attn_prefill_paged_64] (prefill 32k (cold, 32772 tok)) |
| prefill_paged_indirect::`attn_prefill_paged_{indirect, turbo2, turbo3_64, turbo4, turbo4_64, turbo8_64}` (6) | [gb10/common/prefill_paged_compute.cuh:162][f153] | Attention · prefill (flash) | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [6 notes][t153] | not measured |
| attn_prefill_paged_512::`attn_prefill_paged_512` | [gb10/common/prefill_paged_compute_512.cuh:83][f154] | Attention · prefill (flash) | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [3 notes][t154] | not measured |
| prefill_paged_bf16k_turbo2v::`attn_prefill_paged_{bf16k_turbo2v_64, bf16k_turbo3v_64, bf16k_turbo4v_64, fp8k_turbo2v_64, fp8k_turbo3v_64, fp8k_turbo4v_64, turbo3k_turbo8v_64, turbo4k_turbo3v_64, turbo4k_turbo8v_64}` (9) | [gb10/common/prefill_paged_compute_asym.cuh:455][f155] | Attention · prefill (flash) | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [2 notes][t155] | not measured |
| q2_0_gemv_vec::`q2_0_gemv_vec` | [gb10/common/q2_0_gemv_vec.cu:80][f158] | Projection GEMM/GEMV — integer / K-quant · integer / K-quant GEMM/GEMV | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t158] | not measured |
| quant_rowwise_fp8::`quant_rowwise_fp8` | [gb10/common/quant_rowwise_fp8.cu:38][f159] | Quantization and format conversion · activation quantize | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t159] | not measured |
| quantize_nvfp4::`f32_to_bf16_trunc` | [gb10/common/quantize_bf16_to_nvfp4.cu:29][f161] | Quantization and format conversion · dtype conversion | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t161] | not measured |
| quantize_nvfp4::`nvfp4_global_absmax`, `quantize_bf16_to_nvfp4`, `quantize_bf16_to_nvfp4_mse` | [gb10/common/quantize_bf16_to_nvfp4.cu:133][f161] | Quantization and format conversion · activation quantize | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [3 notes][t161] · [#34][pr34] | not measured |
| reshape_and_cache::`bf16_absmax`, `reshape_and_cache_flash_fp8`, `reshape_and_cache_flash_nvfp4`, `reshape_and_cache_flash_v_only` | [gb10/common/reshape_and_cache.cu:30][f163] | KV cache · cache write | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [2 notes][t163] | not measured |
| reshape_and_cache::`reshape_and_cache_flash` | [gb10/common/reshape_and_cache.cu:67][f163] | KV cache · cache write | b200 b300 gb10 hop strix hip | all 14 decoder families + GPT-OSS (31 ckpts) | [1 note][t163] | not measured |
| reshape_and_cache_fused_k_fp8::`fused_k_norm_rope_cache_write_fp8_kv` | [gb10/common/reshape_and_cache_fused_k_fp8.cu:131][f164] | KV cache · cache write | b200 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t164] | not measured |
| reshape_and_cache_turbo::`reshape_and_cache_flash_{bf16k_turbo2v, bf16k_turbo3v, bf16k_turbo4v, fp8k_turbo2v, fp8k_turbo3v, fp8k_turbo4v, turbo2, turbo3, turbo3k_turbo8v, turbo4, turbo4k_turbo3v, turbo4k_turbo8v, turbo8}` (13) | [gb10/common/reshape_and_cache_turbo.cu:179][f165] | KV cache · cache write | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [2 notes][t165] | not measured |
| residual_add::`bf16_concat` | [gb10/common/residual_add.cu:142][f166] | Activations and elementwise · activation / gate / residual | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | — | [4%][m166.bf16_concat] (prefill 4k (cold, 4549 tok)) |
| residual_add::`bf16_residual_add` | [gb10/common/residual_add.cu:10][f166] | Activations and elementwise · activation / gate / residual | b200 b300 gb10 hop strix hip | all 14 decoder families + GPT-OSS (31 ckpts) | — | [97–100%][m166.bf16_residual_add] (prefill 4k (cold, 4103 tok)) |
| residual_add::`bf16_scaled_add`, `sigmoid_gate_mul`, `sigmoid_gate_mul_batched`, `sigmoid_gate_mul_head_broadcast`, `softplus_gate_mul_head_broadcast` | [gb10/common/residual_add.cu:60][f166] | Activations and elementwise · activation / gate / residual | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t166] | not measured |
| residual_add_rms_norm_exact::`residual_add_rms_norm_exact` | [gb10/common/residual_add_rms_norm_exact.cu:28][f167] | Normalization · normalization | b200 gb10 hop | all 14 decoder families (30 ckpts) | — | not measured |
| rms_norm_vanilla::`rms_norm_vanilla` | [gb10/common/rms_norm_vanilla.cu:38][f170] | Normalization · normalization | b200 b300 gb10 hop strix hip | all 14 decoder families + GPT-OSS (31 ckpts) | — | not measured |
| rms_norm_vanilla::`rms_norm_vanilla_warp_row` | [gb10/common/rms_norm_vanilla.cu:120][f170] | Normalization · normalization | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t170] | not measured |
| rope_mrope_interleaved::`rope_forward_mrope_interleaved` | [gb10/common/rope_mrope_interleaved.cu:34][f172] | Positional encoding · rotary | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t172] | [6%][m172.rope_forward_mrope_interleaved] (prefill 32k (cold, 32772 tok)) |
| rope_mrope_interleaved::`rope_forward_mrope_interleaved_k_only` | [gb10/common/rope_mrope_interleaved.cu:108][f172] | Positional encoding · rotary | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t172] | not measured |
| token_overlay::`embed_overlay_routed_bf16`, `embed_rowdiff_bf16`, `lmhead_overlay_routed_bf16`, `lmhead_overlay_routed_f32` | [gb10/common/token_overlay.cu:20][f178] | Embedding and LM head · embedding / LM head | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [2 notes][t178] | not measured |
| tq_plus_innerq_apply::`tq_plus_innerq_apply_{k, q}` (2) | [gb10/common/tq_plus_innerq_apply.cu:71][f179] | KV cache · TurboQuant rotation | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t179] | not measured |
| transpose_u8::`transpose_u8` | [gb10/common/transpose_u8.cu:15][f180] | Quantization and format conversion · dequant / repack / transpose | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | — | not measured |
| w4a16_fp8_ldmab::`fp8_fp8_gemm_ldmab` | [gb10/common/w4a16_fp8_ldmab.cu:65][f182] | Projection GEMM/GEMV — FP8 · FP8 GEMM/GEMV | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [2 notes][t182] | [11–43%][m182.fp8_fp8_gemm_ldmab] (prefill 32k (cold, 32772 tok)) |
| w4a16_fp8_ldmab::`fp8_predequant_nvfp4_t` | [gb10/common/w4a16_fp8_ldmab.cu:193][f182] | Quantization and format conversion · dequant / repack / transpose | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [2 notes][t182] | not measured |
| w4a16_gemv::`w4a16_{gemv, gemv_batch16, gemv_batch2, gemv_batch3, gemv_batch32, gemv_batch8, gemv_batch8_rt2, gemv_dual_batch2, gemv_dual_batch3, gemv_logits, gemv_qg, gemv_qg_batch2, gemv_qg_batch3}` (13) | [gb10/common/w4a16_gemv.cu:167][f184] | Projection GEMM/GEMV — NVFP4 W4A16 · NVFP4 W4A16 GEMM/GEMV | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [6 notes][t184] | not measured |
| w4a16_gemv::`w4a16_gemv_sw` | [gb10/common/w4a16_gemv.cu:233][f184] | Projection GEMM/GEMV — NVFP4 W4A16 · NVFP4 W4A16 GEMM/GEMV | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t184] | [96–99%][m184.w4a16_gemv_sw] (decode C=1 (R=4, MTP k=3)) |
| w4a16_gemv_fused::`w4a16_gemv_{dual, dual_sw}` (2) | [gb10/common/w4a16_gemv_fused.cu:144][f185] | Projection GEMM/GEMV — NVFP4 W4A16 · NVFP4 W4A16 GEMM/GEMV | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t185] | not measured |
| w4a16_gemv_tc::`w4a16_gemv_tc16` | [gb10/common/w4a16_gemv_tc.cu:256][f186] | Projection GEMM/GEMV — NVFP4 W4A16 · NVFP4 W4A16 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | [4 notes][t186] · [#1][pr1] | [86–92%][m186.w4a16_gemv_tc16] (decode C=16 (R=32, MTP k=1)) |
| w4a16_gemv_tc::`w4a16_gemv_tc8` | [gb10/common/w4a16_gemv_tc.cu:255][f186] | Projection GEMM/GEMV — NVFP4 W4A16 · NVFP4 W4A16 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | [4 notes][t186] · [#1][pr1] | [59–90%][m186.w4a16_gemv_tc8] (decode C=1 (R=4, MTP k=3)) |
| w4a16_tc_rows::`w4a16_tc_rows_{16, 32, 64}` (3) | [gb10/common/w4a16_tc_rows.cu:26][f187] | Projection GEMM/GEMV — NVFP4 W4A16 · NVFP4 W4A16 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | — | not measured |
| w4a4_gemv_mx::`w4a4_{gemv_mx16, gemv_mx16_nt2, gemv_mx16_ps, gemv_mx32, gemv_mx32_nt4, gemv_mx32_ps, gemv_mx64, gemv_mx64_nt2, gemv_mx8, quant_rows}` (10) | [gb10/common/w4a4_gemv_mx.cu:360][f188] | Projection GEMM/GEMV — W4A4 · W4A4 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | [23 notes][t188] · [#1][pr1] [#14][pr14] [#18][pr18] | not measured |
| w8a16_gemm::`w8a16_gemm` | [gb10/common/w8a16_gemm.cu:86][f189] | Projection GEMM/GEMV — FP8 · FP8 GEMM/GEMV | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [1 note][t189] | not measured |
| w8a16_gemm_pipe128::`w8a16_gemm_pipe128` | [gb10/common/w8a16_gemm_pipe128.cu:58][f190] | Projection GEMM/GEMV — FP8 · FP8 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | — | not measured |
| w8a16_gemm_pipelined::`w8a16_gemm_pipelined` | [gb10/common/w8a16_gemm_pipelined.cu:174][f191] | Projection GEMM/GEMV — FP8 · FP8 GEMM/GEMV | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [1 note][t191] | [23–24%][m191.w8a16_gemm_pipelined] (prefill 32k (cold, 32772 tok)) |
| w8a16_gemm_pipelined_m32::`w8a16_gemm_pipelined_m32` | [gb10/common/w8a16_gemm_pipelined_m32.cu:159][f192] | Projection GEMM/GEMV — FP8 · FP8 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | [7 notes][t192] · [#4][pr4] [#34][pr34] | [20–95%][m192.w8a16_gemm_pipelined_m32] (decode C=16 (R=32, MTP k=1)) |
| w8a16_gemm_pipelined_m32::`w8a16_gemm_pipelined_m64` | [gb10/common/w8a16_gemm_pipelined_m32.cu:318][f192] | Projection GEMM/GEMV — FP8 · FP8 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | [6 notes][t192] · [#4][pr4] [#34][pr34] | not measured |
| w8a16_gemm_t::`transpose_{block_scale, fp8}` (2) | [gb10/common/w8a16_gemm_t.cu:607][f193] | Quantization and format conversion · dequant / repack / transpose | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [1 note][t193] | not measured |
| w8a16_gemm_t::`w8a16_gemm_{t, t_pipelined}` (2) | [gb10/common/w8a16_gemm_t.cu:151][f193] | Projection GEMM/GEMV — FP8 · FP8 GEMM/GEMV | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [1 note][t193] | not measured |
| w8a16_gemm_t_m128::`w8a16_gemm_t_m128` | [gb10/common/w8a16_gemm_t_m128.cu:62][f194] | Projection GEMM/GEMV — FP8 · FP8 GEMM/GEMV | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t194] | [28%][m194.w8a16_gemm_t_m128] (prefill 4k (cold, 4549 tok)) |
| w8a16_gemv::`w8a16_gemv` | [gb10/common/w8a16_gemv.cu:110][f195] | Projection GEMM/GEMV — FP8 · FP8 GEMM/GEMV | b200 b300 gb10 strix hip | all 14 decoder families (30 ckpts) | [2 notes][t195] | not measured |
| w8a16_gemv_batch4::`w8a16_gemv_{batch16, batch16_strided, batch4, batch4_strided}` (4) | [gb10/common/w8a16_gemv_batch4.cu:234][f196] | Projection GEMM/GEMV — FP8 · FP8 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | [2 notes][t196] | not measured |
| w8a16_tc_rows::`w8a16_tc_rows_{16, 32, 64, 64c}` (4) | [gb10/common/w8a16_tc_rows.cu:38][f198] | Projection GEMM/GEMV — FP8 · FP8 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t198] | not measured |
| w8a8_act_quant::`w8a8_act_quant_{g128, row, silu_g128, silu_row}` (4) | [gb10/common/w8a8_act_quant.cu:168][f199] | Quantization and format conversion · activation quantize | b200 gb10 hop | all 14 decoder families (30 ckpts) | — | not measured |
| w8a8_gemv::`w8a8_gemv_{blk128_mb16, blk128_mb1_ku8, blk128_mb2, blk128_mb4, blk128_mb8, rowscale_mb16, rowscale_mb1_ku8, rowscale_mb2, rowscale_mb4, rowscale_mb8}` (10) | [gb10/common/w8a8_gemv.cu:171][f200] | Projection GEMM/GEMV — FP8 · FP8 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | — | not measured |
| wht_bf16::`wht_bf16_{inplace, inplace_inv}` (2) | [gb10/common/wht_bf16.cu:51][f201] | KV cache · TurboQuant rotation | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t201] | not measured |
| widen_block_scale_f32::`widen_block_scale_f32` | [gb10/common/widen_block_scale_f32.cu:21][f202] | Quantization and format conversion · dequant / repack / transpose | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t202] | not measured |

## Kernels by component

Every entry point with an engine call site, under each component that launches it: a full row under its primary component, and a name under every other component whose code launches it.

### Attention (GQA/MHA: paged decode, split-K, prefill/flash)

337 entry points: 105 primary here (full rows), 232 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| attn_prefill::`attn_prefill` | [gb10/common/attn_prefill.cu:78][f7] | prefill (flash) | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [2 notes][t7] | not measured |
| attn_prefill_512tc::`attn_prefill_512tc` | [gb10/common/attn_prefill.cu:78][f7] | prefill (flash) | gb10 | Gemma4 (2 ckpts) | [3 notes][t7] | not measured |
| attn_prefill::`attn_prefill_64` | [gb10/common/attn_prefill.cu:562][f7] | prefill (flash) | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [2 notes][t7] | [27%][m7.attn_prefill_64] (prefill 32k (cold, 32772 tok)) |
| attn_prefill_fa128::`attn_prefill_{fa128, fa128_paged}` (2) | [gb10/common/attn_prefill_fa128.cu:356][f8] | prefill (flash) | b200 gb10 hop | all 14 decoder families (30 ckpts) | — | not measured |
| attn_prefill_h128::`attn_prefill_h128` | [gb10/common/attn_prefill_h128.cu:46][f10] | prefill (flash) | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [1 note][t10] | not measured |
| paged_decode::`paged_decode_attn` | [gb10/common/paged_decode_attn.cu:343][f119] | paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t119] | [11–44%][m119.paged_decode_attn] (decode C=1 (R=2, MTP k=1)) |
| paged_decode::`paged_decode_attn_sink` | [gb10/common/paged_decode_attn.cu:363][f119] | paged decode | b200 b300 gb10 hop strix hip | GPT-OSS (1 ckpts) | [1 note][t119] | not measured |
| paged_decode_attn_bf16_gqa::`paged_decode_attn_bf16_gqa` | [gb10/common/paged_decode_attn_bf16_gqa.cu:46][f120] | paged decode | b200 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t120] | not measured |
| paged_decode_bf16k_turbo2v::`paged_decode_attn_bf16k_turbo2v` | [gb10/common/paged_decode_attn_bf16k_turbo2v.cu:85][f121] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t121] | not measured |
| paged_decode_bf16k_turbo2v_128::`paged_decode_attn_bf16k_turbo2v` | [gb10/common/paged_decode_attn_bf16k_turbo2v_128.cu:85][f122] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t122] | not measured |
| paged_decode_bf16k_turbo3v::`paged_decode_attn_bf16k_turbo3v` | [gb10/common/paged_decode_attn_bf16k_turbo3v.cu:91][f123] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t123] | not measured |
| paged_decode_bf16k_turbo3v_128::`paged_decode_attn_bf16k_turbo3v` | [gb10/common/paged_decode_attn_bf16k_turbo3v_128.cu:91][f124] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t124] | not measured |
| paged_decode_bf16k_turbo4v::`paged_decode_attn_bf16k_turbo4v` | [gb10/common/paged_decode_attn_bf16k_turbo4v.cu:84][f125] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t125] | not measured |
| paged_decode_bf16k_turbo4v_128::`paged_decode_attn_bf16k_turbo4v` | [gb10/common/paged_decode_attn_bf16k_turbo4v_128.cu:84][f126] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t126] | not measured |
| paged_decode_fp8::`paged_decode_attn_fp8` | [gb10/common/paged_decode_attn_fp8.cu:72][f127] | paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [3 notes][t127] | [45%][m127.paged_decode_attn_fp8] (decode C=16 (R=32, MTP k=1)) |
| paged_decode_fp8::`paged_decode_attn_{reduce_fp8, splitk_fp8}` (2) | [gb10/common/paged_decode_attn_fp8.cu:346][f127] | paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [4 notes][t127] | not measured |
| paged_decode_attn_fp8_gqa::`paged_decode_attn_fp8_gqa` | [gb10/common/paged_decode_attn_fp8_gqa.cu:112][f128] | paged decode | b200 gb10 hop | all 14 decoder families (30 ckpts) | [2 notes][t128] | not measured |
| paged_decode_fp8k_turbo2v::`paged_decode_attn_fp8k_turbo2v` | [gb10/common/paged_decode_attn_fp8k_turbo2v.cu:101][f129] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t129] | not measured |
| paged_decode_fp8k_turbo2v_128::`paged_decode_attn_fp8k_turbo2v` | [gb10/common/paged_decode_attn_fp8k_turbo2v_128.cu:101][f130] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t130] | not measured |
| paged_decode_fp8k_turbo3v::`paged_decode_attn_fp8k_turbo3v` | [gb10/common/paged_decode_attn_fp8k_turbo3v.cu:112][f131] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t131] | not measured |
| paged_decode_fp8k_turbo3v_128::`paged_decode_attn_fp8k_turbo3v` | [gb10/common/paged_decode_attn_fp8k_turbo3v_128.cu:112][f132] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t132] | not measured |
| paged_decode_fp8k_turbo4v::`paged_decode_attn_fp8k_turbo4v` | [gb10/common/paged_decode_attn_fp8k_turbo4v.cu:99][f133] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t133] | not measured |
| paged_decode_fp8k_turbo4v_128::`paged_decode_attn_fp8k_turbo4v` | [gb10/common/paged_decode_attn_fp8k_turbo4v_128.cu:99][f134] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t134] | not measured |
| paged_decode_nvfp4::`paged_decode_attn_{nvfp4, reduce_nvfp4, splitk_nvfp4}` (3) | [gb10/common/paged_decode_attn_nvfp4.cu:87][f135] | paged decode | b200 b300 gb10 hop strix hip | GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (28 ckpts) | [1 note][t135] | not measured |
| paged_decode_turbo2::`paged_decode_attn_turbo2` | [gb10/common/paged_decode_attn_turbo2.cu:80][f136] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [2 notes][t136] | not measured |
| paged_decode_attn_turbo2_128::`paged_decode_attn_turbo2` | [gb10/common/paged_decode_attn_turbo2_128.cu:80][f137] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t137] | not measured |
| paged_decode_attn_turbo3::`paged_decode_attn_{splitk_nvfp4, turbo3}` (2) | [gb10/common/paged_decode_attn_turbo3.cu:110][f138] | paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [3 notes][t138] | not measured |
| paged_decode_attn_turbo3_128::`paged_decode_attn_{splitk_nvfp4, turbo3}` (2) | [gb10/common/paged_decode_attn_turbo3_128.cu:110][f139] | paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t139] | not measured |
| paged_decode_turbo3k_turbo8v::`paged_decode_attn_turbo3k_turbo8v` | [gb10/common/paged_decode_attn_turbo3k_turbo8v.cu:119][f140] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t140] | not measured |
| paged_decode_turbo3k_turbo8v_128::`paged_decode_attn_turbo3k_turbo8v` | [gb10/common/paged_decode_attn_turbo3k_turbo8v_128.cu:119][f141] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t141] | not measured |
| paged_decode_attn_turbo4::`paged_decode_attn_{splitk_nvfp4, turbo4}` (2) | [gb10/common/paged_decode_attn_turbo4.cu:95][f142] | paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t142] | not measured |
| paged_decode_attn_turbo4_128::`paged_decode_attn_{splitk_nvfp4, turbo4}` (2) | [gb10/common/paged_decode_attn_turbo4_128.cu:95][f143] | paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t143] | not measured |
| paged_decode_attn_turbo4_512::`paged_decode_attn_{splitk_nvfp4, turbo4}` (2) | [gb10/common/paged_decode_attn_turbo4_512.cu:119][f144] | paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [2 notes][t144] | not measured |
| paged_decode_turbo4k_turbo3v::`paged_decode_attn_turbo4k_turbo3v` | [gb10/common/paged_decode_attn_turbo4k_turbo3v.cu:122][f145] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t145] | not measured |
| paged_decode_turbo4k_turbo3v_128::`paged_decode_attn_turbo4k_turbo3v` | [gb10/common/paged_decode_attn_turbo4k_turbo3v_128.cu:122][f146] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t146] | not measured |
| paged_decode_turbo4k_turbo8v::`paged_decode_attn_turbo4k_turbo8v` | [gb10/common/paged_decode_attn_turbo4k_turbo8v.cu:114][f147] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t147] | not measured |
| paged_decode_turbo4k_turbo8v_128::`paged_decode_attn_turbo4k_turbo8v` | [gb10/common/paged_decode_attn_turbo4k_turbo8v_128.cu:114][f148] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t148] | not measured |
| paged_decode_attn_turbo8::`paged_decode_attn_{splitk_nvfp4, turbo8}` (2) | [gb10/common/paged_decode_attn_turbo8.cu:100][f149] | paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t149] | not measured |
| paged_decode_attn_turbo8_128::`paged_decode_attn_{splitk_nvfp4, turbo8}` (2) | [gb10/common/paged_decode_attn_turbo8_128.cu:98][f150] | paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [2 notes][t150] | not measured |
| paged_decode_attn_turbo8_512::`paged_decode_attn_{splitk_nvfp4, turbo8}` (2) | [gb10/common/paged_decode_attn_turbo8_512.cu:125][f151] | paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [2 notes][t151] | not measured |
| prefill_paged::`attn_prefill_{paged, paged_batched, paged_batched_64, paged_fp8, paged_fp8_64, paged_fp8_batched, paged_fp8_batched_64, paged_nvfp4, paged_nvfp4_64, paged_nvfp4_batched, paged_nvfp4_batched_64}` (11) | [gb10/common/prefill_paged_compute.cuh:162][f153] | prefill (flash) | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [8 notes][t153] | not measured |
| prefill_paged::`attn_prefill_paged_64` | [gb10/common/prefill_paged_compute.cuh:644][f153] | prefill (flash) | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [6 notes][t153] | [23–24%][m153.attn_prefill_paged_64] (prefill 32k (cold, 32772 tok)) |
| prefill_paged_indirect::`attn_prefill_paged_{indirect, turbo2, turbo3_64, turbo4, turbo4_64, turbo8_64}` (6) | [gb10/common/prefill_paged_compute.cuh:162][f153] | prefill (flash) | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [6 notes][t153] | not measured |
| attn_prefill_paged_512::`attn_prefill_paged_512` | [gb10/common/prefill_paged_compute_512.cuh:83][f154] | prefill (flash) | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [3 notes][t154] | not measured |
| prefill_paged_bf16k_turbo2v::`attn_prefill_paged_{bf16k_turbo2v_64, bf16k_turbo3v_64, bf16k_turbo4v_64, fp8k_turbo2v_64, fp8k_turbo3v_64, fp8k_turbo4v_64, turbo3k_turbo8v_64, turbo4k_turbo3v_64, turbo4k_turbo8v_64}` (9) | [gb10/common/prefill_paged_compute_asym.cuh:455][f155] | prefill (flash) | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [2 notes][t155] | not measured |
| attn_prefill_512::`attn_prefill_512` | [gb10/deepseek-v4-flash/nvfp4/attn_prefill_512.cu:13][f203] | prefill (flash) | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [1 note][t203] | not measured |
| paged_decode_attn_512::`paged_decode_attn` | [gb10/deepseek-v4-flash/nvfp4/paged_decode_attn_512.cu:43][f220] | paged decode | b200 gb10 hop | DeepSeek-V4, Gemma4 (4 ckpts) | — | not measured |
| paged_decode_nvfp4::`paged_decode_attn_{nvfp4, reduce_nvfp4, splitk_nvfp4}` (3) | [gb10/deepseek-v4-flash/nvfp4/paged_decode_attn_nvfp4.cu:87][f223] | paged decode | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [1 note][t223] | not measured |
| attn_prefill_512::`attn_prefill_512` | [gb10/gemma-4-26b-a4b/nvfp4/attn_prefill_512.cu:13][f226] | prefill (flash) | gb10 | Gemma4 (2 ckpts) | [2 notes][t226] | not measured |
| paged_decode_attn_512::`paged_decode_attn` | [gb10/gemma-4-26b-a4b/nvfp4/paged_decode_attn_512.cu:43][f235] | paged decode | gb10 | Gemma4 (2 ckpts) | — | not measured |
| paged_decode_attn_fp8_512::`paged_decode_attn_{fp8, splitk_fp8}` (2) | [gb10/gemma-4-26b-a4b/nvfp4/paged_decode_attn_fp8_512.cu:46][f236] | paged decode | gb10 | Gemma4 (2 ckpts) | [1 note][t236] | not measured |
| attn_prefill_512::`attn_prefill_512` | [gb10/gemma-4-31b/nvfp4/attn_prefill_512.cu:12][f238] | prefill (flash) | gb10 | Gemma4 (2 ckpts) | [1 note][t238] | not measured |
| paged_decode_bf16_splitk_hopper::`paged_decode_attn_{reduce_bf16_hopper, splitk_bf16_hopper}` (2) | [hopper/common/paged_decode_bf16_splitk_hopper.cu:30][f297] | paged decode | hop | DeepSeek-V4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE (18 ckpts) | [2 notes][t297] | not measured |
| paged_decode_fp8_splitk_hopper::`paged_decode_attn_{reduce_fp8_hopper, splitk_fp8_hopper}` (2) | [hopper/common/paged_decode_fp8_splitk_hopper.cu:47][f298] | paged decode | hop | DeepSeek-V4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE (18 ckpts) | [3 notes][t298] | not measured |
| attention_decode::`attention_decode` | [metal/common/attention_decode.metal:30][f306] | paged decode | metal | Qwen-GDN (7 ckpts) | [1 note][t306] | not measured |
| attention_decode_bf16k_turbov::`attention_decode_bf16k_{turbo2v, turbo3v, turbo4v}` (3) | [metal/common/attention_decode_bf16k_turbov.metal:113][f307] | paged decode | metal | Qwen-GDN (7 ckpts) | [1 note][t307] | not measured |
| attention_decode_turbo2::`attention_decode_turbo2` | [metal/common/attention_decode_turbo2.metal:39][f308] | paged decode | metal | Qwen-GDN (7 ckpts) | [1 note][t308] | not measured |
| attention_decode_turbo3::`attention_decode_turbo3` | [metal/common/attention_decode_turbo3.metal:53][f309] | paged decode | metal | Qwen-GDN (7 ckpts) | [1 note][t309] | not measured |
| attention_decode_turbo4::`attention_decode_turbo4` | [metal/common/attention_decode_turbo4.metal:41][f310] | paged decode | metal | Qwen-GDN (7 ckpts) | [1 note][t310] | not measured |
| attention_decode_turbo8::`attention_decode_turbo8` | [metal/common/attention_decode_turbo8.metal:38][f311] | paged decode | metal | Qwen-GDN (7 ckpts) | [1 note][t311] | not measured |
| attn_prefill::`attn_{prefill, prefill_64}` (2) | [strix-hip/common/attn_prefill.cu:69][f347] | prefill (flash) | hip | Qwen-GDN, Qwen-GDN-MoE (13 ckpts) | [1 note][t347] | not measured |
| attn_prefill_h128::`attn_prefill_h128` | [strix-hip/common/attn_prefill_h128.cu:185][f349] | prefill (flash) | hip | Qwen-GDN, Qwen-GDN-MoE (13 ckpts) | [1 note][t349] | not measured |

Also launched here: [Activations and elementwise](#activations-and-elementwise-silu-gelu-relu-residual-gates-scale): `bf16_residual_add`, `sigmoid_gate_mul`, `sigmoid_gate_mul_batched`, `sigmoid_gate_mul_head_broadcast`, `softplus_gate_mul_head_broadcast`, `sigmoid_gate`; [Embedding and LM head](#embedding-and-lm-head-lookup-overlays-softcap-scale): `bf16_scale_inplace`, `bf16_scale_inplace`; [GDN](#gdn-gated-delta-rule-linear-attention): `deinterleave_qg`, `deinterleave_qg_{split, split_qnorm_mrope}` (2), `deinterleave_qg_split_qnorm`; [Hyper-connections](#hyper-connections-mhc): `hc_expand`, `hc_head`, `hc_post`, `hc_pre`, `hc_expand`, `hc_head`, `hc_post`, `hc_pre`; [KV cache](#kv-cache-write-quantize-turboquant-rotation-slot-metadata): `fused_k_norm_rope_{cache_write_bf16, mrope_cache_write_bf16}` (2), `reshape_and_cache_flash`, `reshape_and_cache_flash_{fp8, nvfp4, v_only}` (3), `fused_k_norm_rope_cache_write_fp8_kv`, `reshape_and_cache_flash_{bf16k_turbo2v, bf16k_turbo3v, bf16k_turbo4v, fp8k_turbo2v, fp8k_turbo3v, fp8k_turbo4v, turbo2, turbo3, turbo3k_turbo8v, turbo4, turbo4k_turbo3v, turbo4k_turbo8v, turbo8}` (13), `tq_plus_innerq_apply_{k, q}` (2), `wht_bf16_{inplace, inplace_inv}` (2), `wht_bf16_{inplace, inplace_inv}` (2); [MLA](#mla-multi-head-latent-attention): `grouped_gemm_mla`, `mla_{batched_gemv, cache_assemble, cache_assemble_batched, kv_assemble_batched, q_final_assemble_batched, q_rope_extract_batched, q_rope_scatter, q_rope_writeback, q_rope_writeback_batched}` (9), `mla_fused_prefill`, `mla_paged_decode_nvfp4`, `mla_paged_decode_fp8`, `mla_prefill_attn_320`, `paged_decode_attn_{fp8, splitk_fp8}` (2), `paged_decode_attn`, `mla_{batched_gemv, cache_assemble, cache_assemble_batched, kv_assemble_batched, q_final_assemble_batched, q_rope_extract_batched, q_rope_scatter, q_rope_writeback, q_rope_writeback_batched}` (9), `mla_fused_prefill`, `mla_prefill_attn_320`, `paged_decode_attn_{fp8, splitk_fp8}` (2), `paged_decode_attn`; [Normalization](#normalization-rmsnorm-layernorm-l2-gated-norms): `residual_add_rms_norm`, `residual_add_rms_norm_gatef32`, `residual_add_rms_norm_vanilla`, `rms_norm`, `rms_norm_residual_vanilla`, `rms_norm_strided`, `rms_norm_residual`, `rms_norm_vanilla`, `rms_norm_vanilla_warp_row`, `residual_add_rms_norm`, `residual_add_rms_norm_gatef32`, `rms_norm`, `rms_norm_residual`, `rms_norm_strided`, `residual_add_rms_norm`, `residual_add_rms_norm_gatef32`, `rms_norm`, `rms_norm_residual`, `rms_norm_strided`, `residual_add_rms_norm`, `residual_add_rms_norm_gatef32`, `rms_norm`, `rms_norm_residual`, `rms_norm_strided`, `residual_add_rms_norm`, `residual_add_rms_norm_gatef32`, `rms_norm`, `rms_norm_residual`, `rms_norm_strided`, `residual_add_rms_norm`, `residual_add_rms_norm_gatef32`, `rms_norm`, `rms_norm_residual`, `rms_norm_strided`; [Positional encoding](#positional-encoding-rope-yarn-mrope): `rope_{forward, forward_proportional, forward_strided, forward_yarn, forward_yarn_interleaved, forward_yarn_interleaved_inv, forward_yarn_scaled}` (7), `rope_forward_mrope_interleaved`, `rope_forward_mrope_interleaved_k_only`, `rope_{forward, forward_yarn}` (2); [Projection GEMM/GEMV — BF16/F32](#projection-gemm-gemv-bf16-f32): `dense_gemm_bf16`, `dense_gemm_bf16_pipelined`, `dense_gemm_splitk_{partial, reduce}` (2), `dense_gemm_tc`, `dense_gemv_bf16`, `dense_gemv_bf16_batchm`, `dense_gemm_{bf16, bf16_pipelined}` (2), `dense_gemm_tc`; [Projection GEMM/GEMV — FP8](#projection-gemm-gemv-fp8-w8a16-w8a8-block-scaled): `w8a16_gemv_{batch16, batch16_strided, batch4, batch4_strided}` (4), `fp8_gemm_t_blockscaled`, `w8a16_gemm`, `w8a16_gemm_pipelined`, `w8a16_gemm_pipelined_m32`, `w8a16_gemm_pipelined_m64`, `w8a16_gemm_{t, t_pipelined}` (2), `w8a16_gemm_t_m128`, `w8a16_gemv`, `w8a16_gemv_{batch16, batch16_strided, batch4, batch4_strided}` (4), `fp8_{fp8_gemm_t, fp8_gemm_t_m128, gemm_t, gemm_t_m128}` (4), `fp8_{fp8_gemm_t, fp8_gemm_t_m128, gemm_t, gemm_t_m128}` (4), `fp8_{fp8_gemm_t, fp8_gemm_t_m128, gemm_t}` (3), `fp8_gemm_t_m128`, `fp8_{fp8_gemm_t, fp8_gemm_t_m128, gemm_t, gemm_t_m128}` (4), `w8a16_gemm_{m16, m16_strided}` (2), `w8a16_gemv`, `w8a16_gemv_batch16_{ncol2, ncol2_strided, ncol4, ncol4_strided}` (4), `w8a16_gemm`, `w8a16_gemm_t`, `fp8_{fp8_gemm_t, fp8_gemm_t_m128, gemm_t, gemm_t_m128}` (4), `fp8_{fp8_gemm_t, fp8_gemm_t_m128, gemm_t, gemm_t_m128}` (4); [Projection GEMM/GEMV — NVFP4 W4A16](#projection-gemm-gemv-nvfp4-w4a16): `w4a16_gemm`, `w4a16_{gemv, gemv_batch2, gemv_batch3, gemv_dual_batch2, gemv_dual_batch3, gemv_qg, gemv_qg_batch2, gemv_qg_batch3}` (8), `w4a16_gemv_sw`, `w4a16_gemv_dual`, `w4a16_{gemm, gemm_t_m128}` (2), `w4a16_{gemm, gemm_t_m128}` (2), `w4a16_{gemm, gemm_t_m128_bf16}` (2), `w4a16_gemm_t_m128`, `w4a16_{gemm, gemm_t_m128}` (2), `w4a16_{gemm, gemm_t_m128}` (2), `w4a16_{gemm, gemm_t_m128}` (2); [Projection GEMM/GEMV — W4A4](#projection-gemm-gemv-w4a4-fp4-activations): `w4a4_gemm_mfast`; [Projection GEMM/GEMV — integer / K-quant](#projection-gemm-gemv-integer-k-quant-q2-0-q2-k-q6-k-int8-mlx-int8): `q2_0_gemv_vec`, `metrale_q2_0_mmq128_{nc, wc}` (2); [Quantization and format conversion](#quantization-and-format-conversion): `dequant_q2_0_gn_to_bf16`, `fp8_act_scale_to_kmajor`, `quantize_bf16_to_nvfp4`, `transpose_{block_scale, fp8}` (2), `bf16_to_fp8`, `predequant_nvfp4_to_fp8`, `bf16_to_fp8`, `predequant_nvfp4_to_fp8`, `metrale_q8_1_quantize_ds4_bf16`, `bf16_to_fp8`, `predequant_nvfp4_to_fp8`, `bf16_to_fp8`, `predequant_nvfp4_to_fp8`, `transpose_{block_scale, fp8}` (2), `bf16_to_fp8`, `predequant_nvfp4_to_fp8`, `bf16_to_fp8`, `predequant_nvfp4_to_fp8`; [Sparse / compressed attention](#sparse-compressed-attention-dsa-csa-hca-qsa): `csa_compress`, `prefill_attn_compressed`.

### MLA (multi-head latent attention)

34 entry points: 34 primary here (full rows), 0 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| grouped_gemm_mla::`grouped_gemm_mla` | [gb10/deepseek-v4-flash/nvfp4/grouped_gemm_mla.cu:35][f207] | MLA decode/prefill | b200 gb10 hop | DeepSeek-V4, LongCat, Mistral4 (4 ckpts) | [2 notes][t207] | not measured |
| mla_absorbed::`mla_{batched_gemv, cache_assemble, cache_assemble_batched, kv_assemble_batched, q_final_assemble_batched, q_rope_extract_batched, q_rope_scatter, q_rope_writeback, q_rope_writeback_batched}` (9) | [gb10/deepseek-v4-flash/nvfp4/mla_absorbed.cu:33][f211] | MLA decode/prefill | b200 gb10 hop | DeepSeek-V4, LongCat (3 ckpts) | [1 note][t211] | not measured |
| mla_fused_prefill::`mla_fused_prefill` | [gb10/deepseek-v4-flash/nvfp4/mla_fused_prefill.cu:21][f213] | MLA decode/prefill | b200 gb10 hop | DeepSeek-V4, LongCat (3 ckpts) | [1 note][t213] | not measured |
| mla_paged_decode::`mla_paged_decode_nvfp4` | [gb10/deepseek-v4-flash/nvfp4/mla_paged_decode.cu:78][f214] | MLA decode/prefill | b200 gb10 hop | DeepSeek-V4, LongCat (3 ckpts) | [1 note][t214] | not measured |
| mla_paged_decode_fp8::`mla_paged_decode_fp8` | [gb10/deepseek-v4-flash/nvfp4/mla_paged_decode_fp8.cu:38][f215] | MLA decode/prefill | b200 gb10 hop | DeepSeek-V4, LongCat (3 ckpts) | [1 note][t215] | not measured |
| mla_prefill_attn::`mla_prefill_attn_320` | [gb10/deepseek-v4-flash/nvfp4/mla_prefill_attn.cu:26][f216] | MLA decode/prefill | b200 gb10 hop | DeepSeek-V4, LongCat (3 ckpts) | [1 note][t216] | not measured |
| paged_decode_fp8_mla::`paged_decode_attn_{fp8, splitk_fp8}` (2) | [gb10/deepseek-v4-flash/nvfp4/paged_decode_attn_fp8_mla.cu:66][f221] | MLA decode/prefill | b200 gb10 hop | DeepSeek-V4, LongCat (3 ckpts) | [1 note][t221] | not measured |
| paged_decode_mla::`paged_decode_attn` | [gb10/deepseek-v4-flash/nvfp4/paged_decode_attn_mla.cu:59][f222] | MLA decode/prefill | b200 gb10 hop | DeepSeek-V4, LongCat (3 ckpts) | [1 note][t222] | not measured |
| glm5next_mla_latent_write::`glm5next_mla_latent_write_fp8` | [gb10/glm-5.3-flash/nvfp4/glm5next_mla_latent_write.cu:30][f243] | MLA decode/prefill | gb10 | GLM-5.3 (1 ckpts) | [1 note][t243] | not measured |
| mla_decode::`k3_mla_{maybe_rope_f32, sdpa_gate_f32}` (2) | [gb10/kimi-k3/bf16/mla_decode.cu:41][f247] | MLA decode/prefill | b200 b300 gb10 | Kimi-K3 (1 ckpts) | [1 note][t247] | not measured |
| mla_absorbed::`mla_{batched_gemv, cache_assemble, cache_assemble_batched, kv_assemble_batched, q_final_assemble_batched, q_rope_extract_batched, q_rope_scatter, q_rope_writeback, q_rope_writeback_batched}` (9) | [gb10/mistral-small-4/nvfp4/mla_absorbed.cu:33][f252] | MLA decode/prefill | gb10 | Mistral4 (1 ckpts) | [1 note][t252] | not measured |
| mla_fused_prefill::`mla_fused_prefill` | [gb10/mistral-small-4/nvfp4/mla_fused_prefill.cu:21][f253] | MLA decode/prefill | gb10 | Mistral4 (1 ckpts) | [1 note][t253] | not measured |
| mla_prefill_attn::`mla_prefill_attn_320` | [gb10/mistral-small-4/nvfp4/mla_prefill_attn.cu:24][f254] | MLA decode/prefill | gb10 | Mistral4 (1 ckpts) | [1 note][t254] | not measured |
| paged_decode_attn_fp8_mla::`paged_decode_attn_{fp8, splitk_fp8}` (2) | [gb10/mistral-small-4/nvfp4/paged_decode_attn_fp8_mla.cu:66][f255] | MLA decode/prefill | gb10 | Mistral4 (1 ckpts) | [1 note][t255] | not measured |
| paged_decode_mla::`paged_decode_attn` | [gb10/mistral-small-4/nvfp4/paged_decode_attn_mla.cu:59][f256] | MLA decode/prefill | gb10 | Mistral4 (1 ckpts) | [1 note][t256] | not measured |

### Sparse / compressed attention (DSA, CSA/HCA, QSA)

61 entry points: 43 primary here (full rows), 18 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| dsa_indexer::`dsa_{compact_pools, expand_selection, index_scores, indexer_store, kpool_compress, mla_masked_attn, topk_pools, topk_to_mask, write_geom}` (9) | [b300/common/dsa_indexer.cu:74][f2] | DSA indexer / sparse MLA | b300 | none — its callers' targets compile another copy | [1 note][t2] | not measured |
| dsa_indexer::`dsa_{compact_pools, expand_selection, index_scores, indexer_store, kpool_compress, mla_masked_attn, topk_pools, topk_to_mask, write_geom}` (9) | [gb10/common/dsa_indexer.cu:74][f27] | DSA indexer / sparse MLA | b200 gb10 hop | GLM-5.3 (1 ckpts) | [5 notes][t27] | not measured |
| attn_v41::`attn_v41_{act_quant_fp8, fp4_quant, gemm_f32, gemv_f32_staged, index_score, pool, ring_put, rmsnorm_bf16, rmsnorm_f32, rope, scale_bf16, scatter_cols, slice_cols, sparse_attn}` (14) | [gb10/deepseek-v4-flash/nvfp4/attn_v41.cu:100][f204] | CSA/HCA compressed attention | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [6 notes][t204] | not measured |
| csa_compress::`csa_compress` | [gb10/deepseek-v4-flash/nvfp4/csa_compress.cu:20][f205] | CSA/HCA compressed attention | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [1 note][t205] | not measured |
| prefill_attn_compressed::`prefill_attn_compressed` | [gb10/deepseek-v4-flash/nvfp4/prefill_attn_compressed.cu:23][f224] | CSA/HCA compressed attention | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [1 note][t224] | not measured |
| glm5next_dsa_mla_decode::`glm5next_dsa_mla_decode_fp8` | [gb10/glm-5.3-flash/nvfp4/glm5next_dsa_mla_decode.cu:94][f242] | DSA indexer / sparse MLA | gb10 | GLM-5.3 (1 ckpts) | [1 note][t242] | not measured |
| qsa_indexer::`qsa_{block_pool, gather, prefill_attn, qprep, qprep_rows, score, score_rows, score_rows_tc}` (8) | [gb10/qwen3.8-flash-next/nvfp4/qsa_indexer.cu:70][f288] | QSA sparse attention | gb10 | Qwen3.8-FN (1 ckpts) | [3 notes][t288] | not measured |

Also launched here: [Encoder-decoder translation](#encoder-decoder-translation-nllb-self-contained-kernel-set): `nllb_layernorm_bf16`, `nllb_layernorm_bf16`; [MLA](#mla-multi-head-latent-attention): `glm5next_mla_latent_write_fp8`; [MoE](#moe-routing-dispatch-expert-gemm-gemv-combine): `kquant_mmvq_q2_k_{groups_w, pair_w}` (2); [Normalization](#normalization-rmsnorm-layernorm-l2-gated-norms): `rms_norm_vanilla`; [Projection GEMM/GEMV — BF16/F32](#projection-gemm-gemv-bf16-f32): `dense_gemm_bf16`, `dense_gemm_bf16_f32out`, `dense_gemv_bf16`, `dense_gemv_bf16_fp32out`, `dense_gemv_bf16_batchm`, `dense_gemm_{bf16, bf16_f32out}` (2); [Projection GEMM/GEMV — integer / K-quant](#projection-gemm-gemv-integer-k-quant-q2-0-q2-k-q6-k-int8-mlx-int8): `kquant_mmvq_q2_k_w`, `metrale_q2_k_mmq128_nc`, `metrale_q2_k_mmq128_wc`; [Quantization and format conversion](#quantization-and-format-conversion): `kquant_q8_1_rows_bf16`, `metrale_q8_1_quantize_d2s6_bf16`.

### GDN (gated delta rule linear attention)

362 entry points: 214 primary here (full rows), 148 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| gated_delta_rule::`gated_delta_rule_{chunk2, chunk3, decode, decode_f32, decode_f32_conv_norm, decode_f32_norm, decode_f32_strided, decode_f32_strided_norm, prefill}` (9) | [gb10/common/gated_delta_rule.cu:78][f35] | delta-rule recurrence | b200 b300 gb10 hop | none — its callers' targets compile another copy | [6 notes][t35] | not measured |
| gated_delta_rule_carry::`gdn_carry_conv` | [gb10/common/gated_delta_rule_carry.cu:301][f36] | delta-rule recurrence | b200 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [5 notes][t36] · [#34][pr34] | [33%][m36.gdn_carry_conv] (decode C=16 (R=32, MTP k=1)) |
| gated_delta_rule_carry::`gdn_{carry_conv_f32, carry_conv_flush, carry_flush, carry_wy2, carry_wy3, carry_wy3_lazy, carry_wy4, carry_wy4_lazy, conv_chain_f32, conv_chain_f32_batched}` (10) | [gb10/common/gated_delta_rule_carry.cu:263][f36] | delta-rule recurrence | b200 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [5 notes][t36] · [#34][pr34] | not measured |
| gated_delta_rule_carry::`gdn_carry_wy2_lazy` | [gb10/common/gated_delta_rule_carry.cu:266][f36] | delta-rule recurrence | b200 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [5 notes][t36] · [#34][pr34] | [51%][m36.gdn_carry_wy2_lazy] (decode C=16 (R=32, MTP k=1)) |
| gated_delta_rule_fla::`gated_delta_rule_chunk_delta_h_{ksplit, pipe, tc_vblock, tma, vtile}` (5) | [gb10/common/gated_delta_rule_fla.cu:857][f37] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [10 notes][t37] | not measured |
| gated_delta_rule_fla::`gated_delta_rule_chunk_delta_h_vfused` | [gb10/common/gated_delta_rule_fla.cu:1085][f37] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [8 notes][t37] | [25–36%][m37.gated_delta_rule_chunk_delta_h_vfused] (prefill 32k (cold, 32772 tok)) |
| gated_delta_rule_fla::`gated_delta_rule_chunk_fwd_o` | [gb10/common/gated_delta_rule_fla.cu:2003][f37] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [8 notes][t37] | [32–33%][m37.gated_delta_rule_chunk_fwd_o] (prefill 32k (cold, 32772 tok)) |
| gated_delta_rule_fla::`gated_delta_rule_recompute_wu` | [gb10/common/gated_delta_rule_fla.cu:262][f37] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [11 notes][t37] | [34–40%][m37.gated_delta_rule_recompute_wu] (prefill 32k (cold, 32772 tok)) |
| gated_delta_rule_persistent::`gated_delta_rule_prefill_{persistent, persistent_batched, persistent_wy4, persistent_wy4_batched}` (4) | [gb10/common/gated_delta_rule_persistent.cu:54][f38] | delta-rule recurrence | b200 b300 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [3 notes][t38] | not measured |
| gated_delta_rule_regresident::`gated_delta_rule_prefill_regresident` | [gb10/common/gated_delta_rule_regresident.cu:46][f39] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t39] | not measured |
| gated_delta_rule_wy::`gated_delta_rule_wy2` | [gb10/common/gated_delta_rule_wy.cu:30][f40] | delta-rule recurrence | b200 b300 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [2 notes][t40] | [82%][m40.gated_delta_rule_wy2] (decode C=1 (R=2, MTP k=1)) |
| gated_delta_rule_wy2_resident::`gated_delta_rule_wy2_resident` | [gb10/common/gated_delta_rule_wy2_resident.cu:58][f41] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [2 notes][t41] | not measured |
| gated_delta_rule_wy2_resident_f16::`gated_delta_rule_wy2_resident_f16` | [gb10/common/gated_delta_rule_wy2_resident_f16.cu:65][f42] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [2 notes][t42] | [58%][m42.gated_delta_rule_wy2_resident_f16] (decode C=16 (R=32, MTP k=1)) |
| gated_delta_rule_wy3::`gated_delta_rule_wy3` | [gb10/common/gated_delta_rule_wy3.cu:18][f43] | delta-rule recurrence | b200 b300 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [2 notes][t43] | not measured |
| gated_delta_rule_wy3_f16::`gated_delta_rule_wy3_f16` | [gb10/common/gated_delta_rule_wy3_f16.cu:40][f44] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t44] | not measured |
| gated_delta_rule_wy3_resident::`gated_delta_rule_wy3_resident` | [gb10/common/gated_delta_rule_wy3_resident.cu:59][f45] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t45] | not measured |
| gated_delta_rule_wy3_resident_f16::`gated_delta_rule_wy3_resident_f16` | [gb10/common/gated_delta_rule_wy3_resident_f16.cu:48][f46] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t46] | not measured |
| gated_delta_rule_wy4::`gated_delta_rule_wy4` | [gb10/common/gated_delta_rule_wy4.cu:18][f47] | delta-rule recurrence | b200 b300 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [2 notes][t47] | [58%][m47.gated_delta_rule_wy4] (decode C=1 (R=4, MTP k=3)) |
| gated_delta_rule_wy4_f16::`gated_delta_rule_wy4_f16` | [gb10/common/gated_delta_rule_wy4_f16.cu:40][f48] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t48] | not measured |
| gated_delta_rule_wy4_woa::`gated_delta_rule_wy4_{flag_clear, fold, woa}` (3) | [gb10/common/gated_delta_rule_wy4_woa.cu:55][f49] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [2 notes][t49] | not measured |
| gated_delta_rule_wy64_prefill::`gated_delta_rule_prefill_{wy64, wy64_batched}` (2) | [gb10/common/gated_delta_rule_wy64_prefill.cu:41][f50] | delta-rule recurrence | b200 b300 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t50] | not measured |
| gated_delta_rule_wy_f16::`gated_delta_rule_wy2_f16` | [gb10/common/gated_delta_rule_wy_f16.cu:45][f51] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t51] | not measured |
| gated_delta_rule_wyn::`gated_delta_rule_{wy10, wy10_f16, wy10_f16_table, wy10_table, wy11, wy11_f16, wy11_f16_table, wy11_table, wy12, wy12_f16, wy12_f16_table, wy12_table, wy13, wy13_f16, wy13_f16_table, wy13_table, wy14, wy14_f16, wy14_f16_table, wy14_table, wy15, wy15_f16, wy15_f16_table, wy15_table, wy16, wy16_f16, wy16_f16_table, wy16_table, wy5, wy5_f16, wy5_f16_table, wy5_table, wy6, wy6_f16, wy6_f16_table, wy6_table, wy7, wy7_f16, wy7_f16_table, wy7_table, wy8, wy8_f16, wy8_f16_table, wy8_table, wy9, wy9_f16, wy9_f16_table, wy9_table}` (48) | [gb10/common/gated_delta_rule_wyn.cu:293][f52] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [2 notes][t52] | not measured |
| gdn_chunk_fwd_o_mma8::`gated_delta_rule_chunk_fwd_o_mma8` | [gb10/common/gdn_chunk_fwd_o_mma8.cu:96][f53] | delta-rule recurrence | b200 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | — | not measured |
| gdn_verify_fused_conv_kn::`gdn_verify_fused_conv_kn` | [gb10/common/gdn_verify_fused_conv_kn.cu:50][f54] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t54] | not measured |
| gdn_verify_fused_conv_kn::`gdn_verify_fused_conv_kn_batched` | [gb10/common/gdn_verify_fused_conv_kn.cu:157][f54] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t54] | [35%][m54.gdn_verify_fused_conv_kn_batched] (decode C=16 (R=32, MTP k=1)) |
| gdn_verify_fused_k2::`gdn_verify_fused_{conv_k2, norm_k2}` (2) | [gb10/common/gdn_verify_fused_k2.cu:60][f55] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t55] | not measured |
| ssm_ba_gates_hopper::`dense_gemm_ba_gates_prefill_hopper` | [gb10/common/ssm_ba_gates_hopper.cu:115][f173] | GDN pre/post-processing | b200 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [3 notes][t173] | not measured |
| ssm_h_dtype::`ssm_h_state_{f16_to_f32, f32_to_f16}` (2) | [gb10/common/ssm_h_dtype.cu:25][f175] | GDN pre/post-processing | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t175] | not measured |
| ssm_preprocess::`compute_gdn_gates`, `deinterleave_qg_split`, `deinterleave_qg_split_qnorm_mrope`, `deinterleave_qkvz`, `dense_gemv_ba_gates` | [gb10/common/ssm_preprocess.cu:35][f176] | GDN pre/post-processing | b200 b300 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | — | not measured |
| ssm_preprocess::`deinterleave_qg` | [gb10/common/ssm_preprocess.cu:90][f176] | GDN pre/post-processing | b200 b300 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | — | [3–50%][m176.deinterleave_qg] (decode C=1 (R=2, MTP k=1)) |
| ssm_preprocess::`deinterleave_qg_split_qnorm` | [gb10/common/ssm_preprocess.cu:190][f176] | GDN pre/post-processing | b200 b300 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | — | [88–90%][m176.deinterleave_qg_split_qnorm] (prefill 4k (cold, 4103 tok)) |
| ssm_preprocess::`dense_gemm_ba_gates_prefill` | [gb10/common/ssm_preprocess.cu:482][f176] | GDN pre/post-processing | b200 b300 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [3 notes][t176] | [8–43%][m176.dense_gemm_ba_gates_prefill] (prefill 32k (cold, 32772 tok)) |
| ssm_state_norm::`ssm_state_{clamp_norm_fused, clamp_norm_fused_f16, nonfinite_count}` (3) | [gb10/common/ssm_state_norm.cu:32][f177] | GDN pre/post-processing | b200 b300 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t177] | not measured |
| gated_delta_rule::`gated_delta_rule_{chunk2, chunk3, decode, decode_f32, decode_f32_conv_norm, decode_f32_norm, decode_f32_strided, decode_f32_strided_norm, prefill, prefill_split, prefill_split4, prefill_split4_batched}` (12) | [gb10/gemma-4-26b-a4b/nvfp4/gated_delta_rule.cu:24][f228] | delta-rule recurrence | gb10 | Qwen-GDN-MoE (6 ckpts) | [1 note][t228] | not measured |
| gated_delta_rule::`gated_delta_rule_{chunk2, chunk3, decode, decode_f32, decode_f32_conv_norm, decode_f32_norm, decode_f32_strided, decode_f32_strided_norm, prefill, prefill_split, prefill_split4, prefill_split4_batched}` (12) | [gb10/qwen3-next-80b-a3b/nvfp4/gated_delta_rule.cu:24][f263] | delta-rule recurrence | b200 gb10 hop | Qwen-GDN, Qwen-GDN-MoE (13 ckpts) | [1 note][t263] | not measured |
| gated_delta_rule::`gated_delta_rule_{chunk2, chunk3, decode, decode_f32, decode_f32_conv_norm, decode_f32_norm, decode_f32_strided, decode_f32_strided_norm, prefill, prefill_split, prefill_split4, prefill_split4_batched}` (12) | [gb10/qwen3.5-122b-a10b/nvfp4/gated_delta_rule.cu:24][f266] | delta-rule recurrence | gb10 | Qwen-GDN-MoE (6 ckpts) | [1 note][t266] | not measured |
| gated_delta_rule::`gated_delta_rule_{chunk2, chunk3, decode, decode_f16_norm, decode_f16_strided_norm_half, decode_f32, decode_f32_conv_norm, decode_f32_norm, decode_f32_strided, decode_f32_strided_norm, decode_f32_strided_norm_half, decode_f32_strided_norm_smem, prefill, prefill_split, prefill_split4, prefill_split4_batched}` (16) | [gb10/qwen3.6-27b/nvfp4/gated_delta_rule.cu:24][f267] | delta-rule recurrence | gb10 hop strix hip | Qwen-GDN (7 ckpts) | [6 notes][t267] | not measured |
| gated_delta_rule_snap::`gated_delta_rule_decode_f32_{norm_snap, strided_norm_snap}` (2) | [gb10/qwen3.6-27b/nvfp4/gated_delta_rule_snap.cu:66][f268] | delta-rule recurrence | gb10 hop | Qwen-GDN (7 ckpts) | [2 notes][t268] | not measured |
| gdn_exact_carry::`gdn_exact_{carry2, carry2_lazy, carry3, carry3_lazy, carry4, carry4_lazy, carry_flush, chain2, chain3, chain4, chain_f16_2, chain_f16_3, chain_f16_4}` (13) | [gb10/qwen3.6-27b/nvfp4/gdn_exact_carry.cu:246][f269] | delta-rule recurrence | gb10 hop | Qwen-GDN (7 ckpts) | — | not measured |
| gdn_verify_fused_conv_kn_f32::`gdn_verify_fused_conv_kn_f32` | [gb10/qwen3.6-27b/nvfp4/gdn_verify_fused_conv_kn_f32.cu:33][f270] | delta-rule recurrence | gb10 hop | Qwen-GDN (7 ckpts) | [1 note][t270] | not measured |
| gated_delta_rule::`gated_delta_rule_{chunk2, chunk3, decode, decode_f32, decode_f32_conv_norm, decode_f32_norm, decode_f32_strided, decode_f32_strided_norm, prefill, prefill_split, prefill_split4, prefill_split4_batched}` (12) | [gb10/qwen3.6-35b-a3b/nvfp4/gated_delta_rule.cu:63][f279] | delta-rule recurrence | b200 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [3 notes][t279] | not measured |
| gated_delta_rule_wy17::`gated_delta_rule_wy17` | [gb10/qwen3.6-35b-a3b/nvfp4/gated_delta_rule_wy17.cu:41][f280] | delta-rule recurrence | b200 gb10 hop strix | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [2 notes][t280] | not measured |
| gdn_exact_carry::`gdn_exact_{carry2, carry2_lazy, carry3, carry3_lazy, carry4, carry4_lazy, carry_flush, chain2, chain3, chain4}` (10) | [gb10/qwen3.6-35b-a3b/nvfp4/gdn_exact_carry.cu:209][f281] | delta-rule recurrence | b200 gb10 hop | Qwen-GDN-MoE (6 ckpts) | — | not measured |
| gated_delta_rule_chunk_tc::`gated_delta_rule_chunk_delta_h_tcfuse_x2` | [hopper/common/gated_delta_rule_chunk_tc.cu:409][f292] | delta-rule recurrence | hop | Qwen-GDN, Qwen-GDN-MoE (13 ckpts) | [5 notes][t292] | not measured |
| gdn_fwd_o_hopper::`gated_delta_rule_chunk_fwd_o_hopper` | [hopper/common/gdn_fwd_o_hopper.cu:94][f293] | delta-rule recurrence | hop | Qwen-GDN, Qwen-GDN-MoE (13 ckpts) | [3 notes][t293] | not measured |
| gdn_recompute_wu_hopper::`gated_delta_rule_recompute_wu_hopper` | [hopper/common/gdn_recompute_wu_hopper.cu:138][f294] | delta-rule recurrence | hop | Qwen-GDN, Qwen-GDN-MoE (13 ckpts) | [3 notes][t294] | not measured |
| gated_delta_rule_decode::`gated_delta_rule_decode` | [metal/common/gated_delta_rule_decode.metal:38][f321] | delta-rule recurrence | metal | Qwen-GDN (7 ckpts) | [1 note][t321] | not measured |
| gdn_helpers::`gdn_compute_gate` | [metal/common/gdn_helpers.metal:28][f322] | delta-rule recurrence | metal | Qwen-GDN (7 ckpts) | — | not measured |
| gdn_helpers::`sigmoid_bf16_to_f32` | [metal/common/gdn_helpers.metal:56][f322] | GDN helper | metal | Qwen-GDN (7 ckpts) | — | not measured |
| qwen35_qkv_split::`qwen35_qkv_split` | [metal/common/qwen35_qkv_split.metal:20][f339] | GDN helper | metal | Qwen-GDN (7 ckpts) | — | not measured |

Also launched here: [Activations and elementwise](#activations-and-elementwise-silu-gelu-relu-residual-gates-scale): `bf16_residual_add`, `sigmoid_gate`; [Attention](#attention-gqa-mha-paged-decode-split-k-prefill-flash): `attention_decode`, `attention_decode_bf16k_{turbo2v, turbo3v, turbo4v}` (3), `attention_decode_turbo2`, `attention_decode_turbo3`, `attention_decode_turbo4`, `attention_decode_turbo8`; [Causal conv1d](#causal-conv1d-short-convolution-of-gdn-kda-mamba2): `causal_conv1d_update`, `causal_conv1d_update_chunk2`, `causal_conv1d_update_l2norm`, `causal_conv1d_update_l2norm_{f32, f32_strided}` (2), `causal_conv1d_update_prefill`, `causal_conv1d_update_prefill_tp`, `causal_conv1d_update_l2norm`; [Hyper-connections](#hyper-connections-mhc): `hc_expand`, `hc_post`, `hc_pre`, `hc_expand`, `hc_post`, `hc_pre`; [KV cache](#kv-cache-write-quantize-turboquant-rotation-slot-metadata): `wht_bf16_{inplace, inplace_inv}` (2), `kv_cache_append`, `kv_cache_append_bf16k_{turbo2v, turbo3v, turbo4v}` (3), `kv_cache_append_turbo2`, `kv_cache_append_turbo3`, `kv_cache_append_turbo4`, `kv_cache_append_turbo8`, `wht_bf16_{inplace, inplace_inv}` (2); [Normalization](#normalization-rmsnorm-layernorm-l2-gated-norms): `gated_rms_norm`, `gated_rms_norm_f32_input`, `gated_rms_norm_f32_input_strided`, `residual_add_rms_norm_gatef32`, `gated_rms_norm_prefill`, `l2_norm_bf16`, `residual_add_rms_norm`, `rms_norm_residual`, `gated_rms_norm`, `gated_rms_norm_f32_input`, `gated_rms_norm_prefill`, `residual_add_rms_norm`, `residual_add_rms_norm_gatef32`, `rms_norm_residual`, `l2_norm_bf16`, `gated_rms_norm`, `gated_rms_norm_f32_input`, `gated_rms_norm_prefill`, `residual_add_rms_norm`, `residual_add_rms_norm_gatef32`, `rms_norm_residual`, `l2_norm_bf16`, `gated_rms_norm`, `gated_rms_norm_f32_input`, `gated_rms_norm_prefill`, `residual_add_rms_norm`, `residual_add_rms_norm_gatef32`, `rms_norm_residual`, `l2_norm_bf16`, `gated_rms_norm`, `gated_rms_norm_f32_input`, `gated_rms_norm_prefill`, `residual_add_rms_norm`, `residual_add_rms_norm_gatef32`, `rms_norm_residual`, `l2_norm_bf16`, `gated_rms_norm`, `gated_rms_norm_f32_input`, `gated_rms_norm_prefill`, `residual_add_rms_norm`, `residual_add_rms_norm_gatef32`, `rms_norm_residual`, `l2_norm_bf16`, `gated_rms_norm_{f32_input_sigmoid, prefill_sigmoid, sigmoid}` (3), `add_rms_norm`, `rms_norm`; [Positional encoding](#positional-encoding-rope-yarn-mrope): `rope_apply`; [Projection GEMM/GEMV — BF16/F32](#projection-gemm-gemv-bf16-f32): `dense_gemm_bf16`, `dense_gemm_bf16_pipelined`, `dense_gemv_bf16`, `dense_gemv_bf16_batch2`, `dense_gemm_{bf16, bf16_pipelined}` (2); [Projection GEMM/GEMV — FP8](#projection-gemm-gemv-fp8-w8a16-w8a8-block-scaled): `w8a16_gemv_{batch16, batch4}` (2), `fp8_gemm_t_blockscaled`, `w8a16_gemm`, `w8a16_gemm_pipelined`, `w8a16_gemm_pipelined_m32`, `w8a16_gemm_pipelined_m64`, `w8a16_gemm_t`, `w8a16_gemv`, `w8a16_gemv_{batch16, batch4}` (2), `fp8_gemm_{t, t_m128}` (2), `fp8_gemm_{t, t_m128}` (2), `fp8_gemm_t`, `fp8_gemm_t_m128`, `fp8_gemm_{t, t_m128}` (2), `w8a16_gemv`, `w8a16_gemm`, `w8a16_gemm_t`, `fp8_gemm_{t, t_m128}` (2), `fp8_gemm_{t, t_m128}` (2); [Projection GEMM/GEMV — NVFP4 W4A16](#projection-gemm-gemv-nvfp4-w4a16): `w4a16_gemm`, `w4a16_{gemv, gemv_batch16, gemv_batch2, gemv_batch3}` (4), `w4a16_gemv_qkvz`, `w4a16_gemv_sw`, `w4a16_{gemm, gemm_t_m128}` (2), `w4a16_{gemm, gemm_t_m128}` (2), `w4a16_gemm`, `w4a16_gemm_t_m128`, `w4a16_{gemm, gemm_t_m128}` (2), `w4a16_{gemm, gemm_t_m128}` (2), `w4a16_{gemm, gemm_t_m128}` (2); [Projection GEMM/GEMV — integer / K-quant](#projection-gemm-gemv-integer-k-quant-q2-0-q2-k-q6-k-int8-mlx-int8): `q2_0_gemv_vec`, `metrale_q2_0_mmq128_{nc, wc}` (2); [Quantization and format conversion](#quantization-and-format-conversion): `dequant_q2_0_gn_to_bf16`, `fp8_act_scale_to_kmajor`, `predequant_nvfp4_to_fp8`, `predequant_nvfp4_to_fp8`, `metrale_q8_1_quantize_ds4_bf16`, `predequant_nvfp4_to_fp8`, `predequant_nvfp4_to_fp8`, `predequant_nvfp4_to_fp8`, `predequant_nvfp4_to_fp8`.

### KDA (Kimi delta attention, linear attention)

24 entry points: 11 primary here (full rows), 13 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| kda_chunk::`kda_chunk_{prepare, scan}` (2) | [gb10/common/kda_chunk.cu:95][f62] | KDA op | b200 b300 gb10 hop | GLM-5.3 (1 ckpts) | [2 notes][t62] | not measured |
| kda_gate::`kda_gate_bf16` | [gb10/common/kda_gate.cu:74][f63] | KDA op | b200 b300 gb10 hop | GLM-5.3 (1 ckpts) | — | not measured |
| kda_layer_ops::`kda_{fill_f32, o_norm_gated_bf16, pack_qkv_bf16, sigmoid_bf16_f32, split_widen}` (5) | [gb10/common/kda_layer_ops.cu:50][f64] | KDA op | b200 b300 gb10 hop | GLM-5.3 (1 ckpts) | [1 note][t64] | not measured |
| kda_recurrent::`kda_recurrent_decode_{bf16, bf16_smem}` (2) | [gb10/common/kda_recurrent.cu:144][f65] | KDA op | b200 b300 gb10 hop | GLM-5.3 (1 ckpts) | [2 notes][t65] | not measured |
| kda_decode::`k3_kda_recurrent_step_f32` | [gb10/kimi-k3/bf16/kda_decode.cu:60][f246] | KDA op | b200 b300 gb10 | Kimi-K3 (1 ckpts) | [1 note][t246] | not measured |

Also launched here: [Causal conv1d](#causal-conv1d-short-convolution-of-gdn-kda-mamba2): `causal_conv1d_update_l2norm`, `causal_conv1d_update_prefill`, `k3_kda_conv_update_f32`; [Normalization](#normalization-rmsnorm-layernorm-l2-gated-norms): `l2_norm_bf16`, `l2_norm_bf16`, `l2_norm_bf16`, `l2_norm_bf16`, `l2_norm_bf16`, `l2_norm_bf16`; [Projection GEMM/GEMV — BF16/F32](#projection-gemm-gemv-bf16-f32): `dense_gemm_bf16`, `dense_gemv_bf16`, `dense_gemv_bf16_batchm`, `dense_gemm_bf16`.

### Mamba2 (selective state-space scan)

63 entry points: 6 primary here (full rows), 57 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| mamba2_ssd_chunk::`mamba2_ssd_{bmm, cumsum, scan}` (3) | [gb10/common/mamba2_ssd_chunk.cu:44][f67] | SSD / selective scan | b200 b300 gb10 hop | Nemotron-H (3 ckpts) | [2 notes][t67] | not measured |
| mamba2_ssm::`mamba2_ssm_{decode, prefill, prefill_persistent}` (3) | [gb10/common/mamba2_ssm_decode.cu:29][f68] | SSD / selective scan | b200 b300 gb10 hop strix hip | Nemotron-H (3 ckpts) | [2 notes][t68] | not measured |

Also launched here: [Activations and elementwise](#activations-and-elementwise-silu-gelu-relu-residual-gates-scale): `bf16_residual_add`; [Causal conv1d](#causal-conv1d-short-convolution-of-gdn-kda-mamba2): `causal_conv1d_update`, `causal_conv1d_update_prefill`, `causal_conv1d_update_prefill_tp`; [Normalization](#normalization-rmsnorm-layernorm-l2-gated-norms): `gated_rms_norm`, `rms_norm_residual`, `gated_rms_norm`, `rms_norm_residual`, `gated_rms_norm`, `rms_norm_residual`, `gated_rms_norm`, `rms_norm_residual`, `gated_rms_norm`, `rms_norm_residual`, `gated_rms_norm`, `rms_norm_residual`; [Projection GEMM/GEMV — BF16/F32](#projection-gemm-gemv-bf16-f32): `dense_gemm_bf16_pipelined`, `dense_gemv_bf16`, `dense_gemv_bf16`, `dense_gemm_bf16_pipelined`; [Projection GEMM/GEMV — FP8](#projection-gemm-gemv-fp8-w8a16-w8a8-block-scaled): `w8a16_gemm`, `w8a16_gemm_pipelined`, `w8a16_gemv`, `fp8_{fp8_gemm_t_m128_mfast, gemm_t_m128_mfast}` (2), `w8a16_gemv`, `w8a16_gemm`; [Projection GEMM/GEMV — NVFP4 W4A16](#projection-gemm-gemv-nvfp4-w4a16): `w4a16_{gemm, gemm_t}` (2), `w4a16_gemv`, `w4a16_gemv_sw`, `w4a16_{gemm, gemm_t, gemm_t_m128}` (3), `w4a16_{gemm, gemm_t, gemm_t_m128}` (3), `w4a16_{gemm, gemm_t}` (2), `w4a16_gemm_t_m128`, `w4a16_{gemm, gemm_t_m128}` (2), `w4a16_gemm_t`, `w4a16_{gemm, gemm_t, gemm_t_m128}` (3), `w4a16_{gemm, gemm_t, gemm_t_m128}` (3); [Projection GEMM/GEMV — W4A4](#projection-gemm-gemv-w4a4-fp4-activations): `w4a4_gemm_mfast`; [Quantization and format conversion](#quantization-and-format-conversion): `quantize_bf16_to_nvfp4`, `bf16_to_fp8`, `bf16_to_fp8`, `bf16_to_fp8`, `bf16_to_fp8`, `bf16_to_fp8`, `bf16_to_fp8`.

### Causal conv1d (short convolution of GDN/KDA/Mamba2)

10 entry points: 10 primary here (full rows), 0 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| causal_conv1d::`causal_conv1d_{prefill_state, update_l2norm_f32, update_l2norm_f32_strided}` (3) | [gb10/common/causal_conv1d.cu:413][f13] | causal conv1d | b200 b300 gb10 hop strix hip | GLM-5.3, Kimi-K3, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (19 ckpts) | [2 notes][t13] | not measured |
| causal_conv1d::`causal_conv1d_update` | [gb10/common/causal_conv1d.cu:96][f13] | causal conv1d | b200 b300 gb10 hop strix hip | Nemotron-H, Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (17 ckpts) | [1 note][t13] | not measured |
| causal_conv1d::`causal_conv1d_update_chunk2` | [gb10/common/causal_conv1d.cu:247][f13] | causal conv1d | b200 b300 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t13] | not measured |
| causal_conv1d::`causal_conv1d_update_l2norm` | [gb10/common/causal_conv1d.cu:327][f13] | causal conv1d | b200 b300 gb10 hop strix hip | GLM-5.3, Kimi-K3, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (19 ckpts) | [1 note][t13] | [33–54%][m13.causal_conv1d_update_l2norm] (decode C=1 (R=2, MTP k=1)) |
| causal_conv1d::`causal_conv1d_update_prefill` | [gb10/common/causal_conv1d.cu:184][f13] | causal conv1d | b200 b300 gb10 hop strix hip | GLM-5.3, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (18 ckpts) | [2 notes][t13] | not measured |
| causal_conv1d::`causal_conv1d_update_prefill_tp` | [gb10/common/causal_conv1d.cu:585][f13] | causal conv1d | b200 b300 gb10 hop strix hip | Nemotron-H, Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (17 ckpts) | [3 notes][t13] | [94–96%][m13.causal_conv1d_update_prefill_tp] (prefill 32k (cold, 32772 tok)) |
| kda_decode::`k3_kda_conv_update_f32` | [gb10/kimi-k3/bf16/kda_decode.cu:21][f246] | causal conv1d | b200 b300 gb10 | Kimi-K3 (1 ckpts) | [1 note][t246] | not measured |
| causal_conv1d_update_l2norm::`causal_conv1d_update_l2norm` | [metal/common/causal_conv1d_update_l2norm.metal:41][f316] | causal conv1d | metal | Qwen-GDN (7 ckpts) | [1 note][t316] | not measured |

### MoE (routing, dispatch, expert GEMM/GEMV, combine)

296 entry points: 198 primary here (full rows), 98 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| moe_shared_expert_fused::`moe_expert_{gate_up_shared, silu_down_shared}` (2) | [b300/common/moe_shared_expert_fused.cu:48][f3] | expert GEMM/GEMV | b300 | Kimi-K3 (1 ckpts) | [2 notes][t3] | not measured |
| gemm::`dense_gemm_bf16_router` | [gb10/common/dense_gemm_bf16.cu:204][f14] | routing / top-k | b200 b300 gb10 hop strix | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [3 notes][t14] | [6%][m14.dense_gemm_bf16_router] (prefill 32k (cold, 32772 tok)) |
| glm5next_ffn::`glm5next_moe_{combine, combine_indexed}` (2) | [gb10/common/glm5next_ffn.cu:213][f56] | dispatch / combine | b200 b300 gb10 hop | GLM-5.3 (1 ckpts) | [1 note][t56] | not measured |
| glm5next_ffn::`glm5next_router_topk` | [gb10/common/glm5next_ffn.cu:102][f56] | routing / top-k | b200 b300 gb10 hop | GLM-5.3 (1 ckpts) | [1 note][t56] | not measured |
| glm5next_ffn::`glm5next_swiglu_clamp` | [gb10/common/glm5next_ffn.cu:31][f56] | expert activation | b200 b300 gb10 hop | GLM-5.3 (1 ckpts) | [1 note][t56] | not measured |
| gpt_oss_expert_ops::`gpt_oss_{expert_reduce_bf16, selected_bias_bf16, swiglu_bf16}` (3) | [gb10/common/gpt_oss_expert_ops.cu:15][f58] | staged BF16 expert activation / reduction | b200 gb10 hop | GPT-OSS (1 ckpts) | — | not measured |
| gpt_oss_mxfp4_gemv::`gpt_oss_mxfp4_selected_bf16` | [gb10/common/gpt_oss_mxfp4_gemv.cu:54][f59] | row-major MXFP4 expert GEMV (correctness residual) | b200 gb10 hop | GPT-OSS (1 ckpts) | — | not measured |
| moe_bf16_grouped_gemm::`moe_bf16_grouped_gemm` | [gb10/common/moe_bf16_grouped_gemm.cu:86][f70] | expert GEMM/GEMV | b200 b300 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t70] | not measured |
| moe_bf16_grouped_tc::`moe_expert_{down_act_bf16_grouped_tc, gate_up_act_bf16_grouped_tc}` (2) | [gb10/common/moe_bf16_grouped_tc.cu:42][f71] | expert GEMM/GEMV | b200 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | — | not measured |
| moe_decode_atomic_c4::`moe_decode_atomic_c4_finalize` | [gb10/common/moe_decode_atomic_c4.cu:183][f72] | dispatch / combine | b200 b300 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t72] | not measured |
| moe_decode_atomic_c4::`moe_decode_atomic_c4_silu_down_accum` | [gb10/common/moe_decode_atomic_c4.cu:37][f72] | expert GEMM/GEMV | b200 b300 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t72] | not measured |
| moe_expert_gemv::`moe_expert_gemv` | [gb10/common/moe_expert_gemv.cu:57][f73] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t73] | not measured |
| moe_expert_gemv::`moe_weighted_sum_blend` | [gb10/common/moe_expert_gemv.cu:195][f73] | dispatch / combine | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t73] | not measured |
| moe_relu2_fused::`moe_expert_relu2_down_shared` | [gb10/common/moe_expert_relu2_down_shared.cu:53][f75] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | Nemotron-H (3 ckpts) | [1 note][t75] | not measured |
| moe_fp8_grouped_blend::`moe_weighted_sum_blend_fp8_grouped` | [gb10/common/moe_fp8_grouped_blend.cu:17][f76] | dispatch / combine | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t76] · [#4][pr4] | [16–77%][m76.moe_weighted_sum_blend_fp8_grouped] (decode C=1 (R=2, MTP k=1)) |
| moe_fp8_grouped_gemm::`moe_fp8_grouped_gemm` | [gb10/common/moe_fp8_grouped_gemm.cu:281][f77] | expert GEMM/GEMV | b200 b300 gb10 hop strix | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t77] | not measured |
| moe_fp8_grouped_sort::`moe_fp8_grouped_sort` | [gb10/common/moe_fp8_grouped_sort.cu:24][f78] | dispatch / combine | b200 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [3 notes][t78] · [#34][pr34] | [1%][m78.moe_fp8_grouped_sort] (decode C=1 (R=2, MTP k=1)) |
| moe_fp8_grouped_tc::`moe_expert_{down_act_fp8_grouped_tc, gate_up_act_fp8_grouped_tc}` (2) | [gb10/common/moe_fp8_grouped_tc.cu:60][f79] | expert GEMM/GEMV | b200 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t79] | not measured |
| moe_fp8_grouped_tc_w8a8::`moe_{act_quant_e4m3, expert_down_act_fp8_grouped_tc_w8a8, expert_gate_up_act_fp8_grouped_tc_w8a8}` (3) | [gb10/common/moe_fp8_grouped_tc_w8a8.cu:54][f80] | expert GEMM/GEMV | b200 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t80] | not measured |
| moe_gate_topk::`moe_gate_topk_fused` | [gb10/common/moe_gate_topk.cu:46][f81] | routing / top-k | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t81] | not measured |
| moe_hash_route::`moe_hash_{route, route_batched}` (2) | [gb10/common/moe_hash_route.cu:25][f82] | routing / top-k | b200 b300 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t82] | not measured |
| moe_nvfp4_grouped::`moe_expert_{down_act_nvfp4_grouped, gate_up_act_nvfp4_grouped}` (2) | [gb10/common/moe_nvfp4_grouped.cu:72][f85] | expert GEMM/GEMV | b200 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [5 notes][t85] · [#34][pr34] | not measured |
| moe_nvfp4_grouped_tc::`moe_expert_{down_act_nvfp4_grouped_tc, down_act_nvfp4_grouped_tc_lean, gate_up_act_nvfp4_grouped_tc, gate_up_act_nvfp4_grouped_tc_lean}` (4) | [gb10/common/moe_nvfp4_grouped_tc.cu:177][f86] | expert GEMM/GEMV | b200 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | — | not measured |
| moe::`moe_batched_blend` | [gb10/common/moe_permute.cu:126][f87] | dispatch / combine | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t87] | [97–100%][m87.moe_batched_blend] (prefill 32k (cold, 32772 tok)) |
| moe::`moe_build_tile_worklist` | [gb10/common/moe_permute.cu:276][f87] | dispatch / combine | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t87] | [0%][m87.moe_build_tile_worklist] (prefill 32k (cold, 32772 tok)) |
| moe::`moe_{permute_tokens, sort_by_expert}` (2) | [gb10/common/moe_permute.cu:20][f87] | dispatch / combine | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t87] | not measured |
| moe::`moe_unpermute_reduce_indexed` | [gb10/common/moe_permute.cu:95][f87] | dispatch / combine | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t87] | [87–88%][m87.moe_unpermute_reduce_indexed] (prefill 32k (cold, 32772 tok)) |
| moe_prefill::`moe_expert_{gate_up_shared_prefill, silu_down_shared_prefill}` (2) | [gb10/common/moe_prefill.cu:59][f88] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t88] | not measured |
| moe_prefill::`moe_weighted_sum_blend_prefill` | [gb10/common/moe_prefill.cu:360][f88] | dispatch / combine | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t88] | not measured |
| moe_router_gemm::`moe_router_gemm_bf16` | [gb10/common/moe_router_gemm.cu:32][f89] | routing / top-k | b200 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [3 notes][t89] · [#34][pr34] | not measured |
| moe_router_gemm_prefill::`moe_router_gemm_rt` | [gb10/common/moe_router_gemm_prefill.cu:24][f90] | routing / top-k | b200 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | — | not measured |
| moe_shared_expert_fused::`moe_expert_{gate_up_shared, silu_down_shared}` (2) | [gb10/common/moe_shared_expert_fused.cu:48][f91] | expert GEMM/GEMV | b200 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t91] | not measured |
| moe_fused_batch2::`moe_expert_{gate_up_shared_batch2, silu_down_shared_batch2}` (2) | [gb10/common/moe_shared_expert_fused_batch2.cu:292][f92] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t92] | not measured |
| moe_fused_batch2::`moe_weighted_sum_blend_batch2` | [gb10/common/moe_shared_expert_fused_batch2.cu:497][f92] | dispatch / combine | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t92] | not measured |
| moe_shared_expert_fused_batch2_t::`moe_expert_{gate_up_shared_batch2_t, silu_down_shared_batch2_t}` (2) | [gb10/common/moe_shared_expert_fused_batch2_t.cu:41][f93] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t93] | not measured |
| moe_fused_batch3::`moe_expert_{gate_up_shared_batch3, silu_down_shared_batch3}` (2) | [gb10/common/moe_shared_expert_fused_batch3.cu:50][f94] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t94] | not measured |
| moe_fused_batch3::`moe_weighted_sum_blend_batch3` | [gb10/common/moe_shared_expert_fused_batch3.cu:335][f94] | dispatch / combine | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t94] | not measured |
| moe_shared_expert_fused_batch3_t::`moe_expert_{gate_up_shared_batch3_t, silu_down_shared_batch3_t}` (2) | [gb10/common/moe_shared_expert_fused_batch3_t.cu:36][f95] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t95] | not measured |
| moe_shared_expert_fused_bf16::`moe_expert_{gate_up_shared_bf16, silu_down_shared_bf16}` (2) | [gb10/common/moe_shared_expert_fused_bf16.cu:26][f96] | expert GEMM/GEMV | b200 b300 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t96] | not measured |
| moe_shared_expert_fused_bf16_batch2::`moe_expert_{gate_up_shared_bf16_batch2, silu_down_shared_bf16_batch2}` (2) | [gb10/common/moe_shared_expert_fused_bf16_batch2.cu:39][f97] | expert GEMM/GEMV | b200 b300 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t97] | not measured |
| moe_shared_expert_fused_fp8::`moe_expert_gate_up_shared_fp8` | [gb10/common/moe_shared_expert_fused_fp8.cu:98][f98] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t98] | [85%][m98.moe_expert_gate_up_shared_fp8] (decode C=1 (R=2, MTP k=1)) |
| moe_shared_expert_fused_fp8::`moe_expert_silu_down_shared_fp8` | [gb10/common/moe_shared_expert_fused_fp8.cu:262][f98] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t98] | not measured |
| moe_shared_expert_fused_fp8_batch2::`moe_expert_{gate_up_shared_fp8_batch2, silu_down_shared_fp8_batch2}` (2) | [gb10/common/moe_shared_expert_fused_fp8_batch2.cu:97][f99] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t99] | not measured |
| moe_shared_expert_fused_fp8_batch2::`moe_weighted_sum_blend_fp8_batch2` | [gb10/common/moe_shared_expert_fused_fp8_batch2.cu:410][f99] | dispatch / combine | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t99] | not measured |
| moe_shared_expert_fused_fp8_batch2_t::`moe_expert_{gate_up_shared_fp8_batch2_t, silu_down_shared_fp8_batch2_t}` (2) | [gb10/common/moe_shared_expert_fused_fp8_batch2_t.cu:28][f100] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t100] | not measured |
| moe_shared_expert_fused_fp8_batch3::`moe_expert_{gate_up_shared_fp8_batch3, silu_down_shared_fp8_batch3}` (2) | [gb10/common/moe_shared_expert_fused_fp8_batch3.cu:97][f101] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t101] | not measured |
| moe_shared_expert_fused_fp8_batch3::`moe_weighted_sum_blend_fp8_batch3` | [gb10/common/moe_shared_expert_fused_fp8_batch3.cu:408][f101] | dispatch / combine | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t101] | not measured |
| moe_shared_expert_fused_fp8_batch3_t::`moe_expert_{gate_up_shared_fp8_batch3_t, silu_down_shared_fp8_batch3_t}` (2) | [gb10/common/moe_shared_expert_fused_fp8_batch3_t.cu:28][f102] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t102] | not measured |
| moe_shared_expert_fused_fp8_grouped::`moe_expert_down_act_fp8_grouped` | [gb10/common/moe_shared_expert_fused_fp8_grouped.cu:331][f103] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [12 notes][t103] · [#4][pr4] [#34][pr34] | [89–102%][m103.moe_expert_down_act_fp8_grouped] (decode C=16 (R=32, MTP k=1)) |
| moe_shared_expert_fused_fp8_grouped::`moe_expert_gate_up_act_fp8_grouped` | [gb10/common/moe_shared_expert_fused_fp8_grouped.cu:153][f103] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [11 notes][t103] · [#4][pr4] [#34][pr34] | [88–97%][m103.moe_expert_gate_up_act_fp8_grouped] (decode C=16 (R=32, MTP k=1)) |
| moe_shared_expert_fused_fp8_t::`moe_expert_{gate_up_shared_fp8_t, silu_down_shared_fp8_t}` (2) | [gb10/common/moe_shared_expert_fused_fp8_t.cu:34][f104] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t104] | not measured |
| moe_shared_expert_fused_t::`moe_expert_{gate_up_shared_t, gate_up_shared_t_e8m0, silu_down_shared_t, silu_down_shared_t_e8m0}` (4) | [gb10/common/moe_shared_expert_fused_t.cu:187][f105] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t105] | not measured |
| moe_silu_mul::`moe_silu_mul` | [gb10/common/moe_silu_mul.cu:38][f106] | expert activation | b200 b300 gb10 hop strix hip | GLM-5.3, Gemma4, Kimi-K3, Laguna, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN (19 ckpts) | — | not measured |
| moe_sorted::`moe_sorted_{gate_up, silu_down}` (2) | [gb10/common/moe_sorted_prefill.cu:54][f107] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t107] | not measured |
| moe_topk::`moe_topk_selected_bf16_rows` | [gb10/common/moe_topk.cu:497][f108] | routing / top-k | b200 b300 gb10 hop strix hip | GPT-OSS (1 ckpts) | [1 note][t108] | not measured |
| moe_topk::`moe_topk_{softmax, softmax_batched, softmax_f32}` (3) | [gb10/common/moe_topk.cu:205][f108] | routing / top-k | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t108] | not measured |
| moe_topk::`moe_topk_softmax_rows` | [gb10/common/moe_topk.cu:220][f108] | routing / top-k | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t108] · [#34][pr34] | [0%][m108.moe_topk_softmax_rows] (decode C=1 (R=2, MTP k=1)) |
| moe_topk_sig::`moe_topk_{sigmoid, sigmoid_batched}` (2) | [gb10/common/moe_topk_sigmoid.cu:22][f109] | routing / top-k | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t109] | not measured |
| moe_topk_softmax_bias::`moe_topk_softmax_{bias, bias_batched}` (2) | [gb10/common/moe_topk_softmax_bias.cu:189][f110] | routing / top-k | b200 b300 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t110] | not measured |
| moe_topk_softmax_bias::`moe_zero_expert_add` | [gb10/common/moe_topk_softmax_bias.cu:233][f110] | dispatch / combine | b200 b300 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t110] | not measured |
| moe_topk_sqrt::`moe_topk_{sqrtsoftplus, sqrtsoftplus_batched}` (2) | [gb10/common/moe_topk_sqrtsoftplus.cu:22][f111] | routing / top-k | b200 b300 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t111] | not measured |
| moe_transpose_batched::`moe_transpose_u8_batched` | [gb10/common/moe_transpose_batched.cu:21][f112] | dispatch / combine | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t112] | not measured |
| moe_unpermute_blend::`moe_unpermute_blend` | [gb10/common/moe_unpermute_blend.cu:17][f113] | dispatch / combine | b200 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | — | not measured |
| moe_w4a16::`moe_w4a16_grouped_gemm_{ptrtable, ptrtable_k32, ptrtable_t}` (3) | [gb10/common/moe_w4a16_grouped_gemm.cu:231][f114] | expert GEMM/GEMV | b200 gb10 hop | GLM-5.3, Gemma4, Nemotron-H (6 ckpts) | [4 notes][t114] | not measured |
| moe_w4a16::`moe_w4a16_grouped_gemm_ptrtable_{alkm_m16_k128, bt_m16_k128, bt_m16_n128_k128, k64, m16_k64}` (5) | [gb10/common/moe_w4a16_grouped_gemm.cu:945][f114] | expert GEMM/GEMV | b200 gb10 hop | GLM-5.3 (1 ckpts) | [4 notes][t114] | not measured |
| moe_w8a8_grouped_gemm::`moe_w8a8_grouped_gemm` | [gb10/common/moe_w8a8_grouped_gemm.cu:93][f115] | expert GEMM/GEMV | b200 b300 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t115] | not measured |
| moe_w8a8_grouped_gemm::`moe_w8a8_grouped_gemm_pm4` | [gb10/common/moe_w8a8_grouped_gemm.cu:396][f115] | expert GEMM/GEMV | b200 b300 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [4 notes][t115] | [18–32%][m115.moe_w8a8_grouped_gemm_pm4] (prefill 32k (cold, 32772 tok)) |
| moe_w8a8_grouped_gemm_e4m3::`moe_w8a8_{gateup_silu_e4m3_w1, gateup_silu_e4m3_w2, grouped_gemm_e4m3_dn, grouped_gemm_e4m3_gu}` (4) | [gb10/common/moe_w8a8_grouped_gemm_e4m3.cu:101][f116] | expert GEMM/GEMV | b200 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | — | not measured |
| nemotron_moe_prefill::`nemotron_moe_{relu2_down_prefill, up_prefill}` (2) | [gb10/common/nemotron_moe_prefill.cu:161][f117] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | Nemotron-H (3 ckpts) | — | not measured |
| nemotron_moe_prefill::`nemotron_moe_topk_sigmoid_batched` | [gb10/common/nemotron_moe_prefill.cu:53][f117] | routing / top-k | b200 b300 gb10 hop strix hip | Nemotron-H (3 ckpts) | [1 note][t117] | not measured |
| nemotron_moe_prefill::`nemotron_moe_weighted_sum_prefill` | [gb10/common/nemotron_moe_prefill.cu:444][f117] | dispatch / combine | b200 b300 gb10 hop strix hip | Nemotron-H (3 ckpts) | — | not measured |
| relu2::`moe_weighted_sum_scale` | [gb10/common/relu_squared.cu:57][f162] | dispatch / combine | b200 b300 gb10 hop strix hip | Nemotron-H (3 ckpts) | [1 note][t162] | not measured |
| w4a16_gemv::`glm5next_moe_row_union` | [gb10/common/w4a16_gemv.cu:2201][f184] | dispatch / combine | b200 b300 gb10 hop strix hip | GLM-5.3 (1 ckpts) | [2 notes][t184] | not measured |
| kquant_moe::`kquant_mmvq_{q2_k_experts_w2, q2_k_experts_w8, q2_k_groups_w, q2_k_pair_w, q3_k_experts_w2, q3_k_experts_w8}` (6) | [gb10/deepseek-v4-flash/nvfp4/kquant_moe.cu:321][f210] | expert GEMM/GEMV | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [7 notes][t210] | not measured |
| moe_silu_mul::`moe_silu_mul` | [gb10/deepseek-v4-flash/nvfp4/moe_silu_mul.cu:40][f217] | expert activation | b200 gb10 hop | DeepSeek-V4, LongCat (3 ckpts) | [1 note][t217] | not measured |
| moe_v41::`moe_v41_{accumulate, finish, gather_rows, scatter_add, slot_table_set, sum_rows, swiglu}` (7) | [gb10/deepseek-v4-flash/nvfp4/moe_v41.cu:16][f218] | dispatch / combine | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [1 note][t218] | not measured |
| moe_v41::`moe_v41_{route_select, router_gemv_f32out, router_gemv_f32out_products}` (3) | [gb10/deepseek-v4-flash/nvfp4/moe_v41.cu:145][f218] | routing / top-k | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [3 notes][t218] | not measured |
| moe_w4a16::`moe_{fp8_grouped_gemm_ptrtable_t, w4a16_fused_gate_up_t, w4a16_fused_gate_up_t_e8m0, w4a16_fused_gate_up_t_k64, w4a16_fused_gate_up_t_k64_e8m0, w4a16_grouped_gemm_ptrtable, w4a16_grouped_gemm_ptrtable_e8m0, w4a16_grouped_gemm_ptrtable_t, w4a16_grouped_gemm_ptrtable_t_e8m0, w4a16_grouped_gemm_ptrtable_t_k64, w4a16_grouped_gemm_ptrtable_t_k64_e8m0}` (11) | [gb10/deepseek-v4-flash/nvfp4/moe_w4a16_grouped_gemm.cu:188][f219] | expert GEMM/GEMV | b200 b300 gb10 hop | DeepSeek-V4, Kimi-K3, LongCat (4 ckpts) | [1 note][t219] | not measured |
| moe_shared_expert_fused::`moe_expert_{gate_up_shared, silu_down_shared}` (2) | [gb10/gemma-4-26b-a4b/nvfp4/moe_shared_expert_fused.cu:31][f231] | expert GEMM/GEMV | gb10 | Gemma4 (2 ckpts) | [1 note][t231] | not measured |
| moe_fused_batch2::`moe_expert_{gate_up_shared_batch2, silu_down_shared_batch2}` (2) | [gb10/gemma-4-26b-a4b/nvfp4/moe_shared_expert_fused_batch2.cu:35][f232] | expert GEMM/GEMV | gb10 | Gemma4 (2 ckpts) | [1 note][t232] | not measured |
| moe_fused_batch2::`moe_weighted_sum_blend_batch2` | [gb10/gemma-4-26b-a4b/nvfp4/moe_shared_expert_fused_batch2.cu:327][f232] | dispatch / combine | gb10 | Gemma4 (2 ckpts) | [1 note][t232] | not measured |
| moe_fused_batch3::`moe_expert_{gate_up_shared_batch3, silu_down_shared_batch3}` (2) | [gb10/gemma-4-26b-a4b/nvfp4/moe_shared_expert_fused_batch3.cu:33][f233] | expert GEMM/GEMV | gb10 | Gemma4 (2 ckpts) | [1 note][t233] | not measured |
| moe_fused_batch3::`moe_weighted_sum_blend_batch3` | [gb10/gemma-4-26b-a4b/nvfp4/moe_shared_expert_fused_batch3.cu:323][f233] | dispatch / combine | gb10 | Gemma4 (2 ckpts) | [1 note][t233] | not measured |
| moe_w4a16::`moe_{fp8_grouped_gemm_ptrtable_t, w4a16_fused_gate_up_t, w4a16_fused_gate_up_t_k64, w4a16_grouped_gemm_ptrtable, w4a16_grouped_gemm_ptrtable_t, w4a16_grouped_gemm_ptrtable_t_k64}` (6) | [gb10/gemma-4-26b-a4b/nvfp4/moe_w4a16_grouped_gemm.cu:34][f234] | expert GEMM/GEMV | b200 gb10 hop | Gemma4, Mistral4, Qwen-GDN-MoE, Qwen3-VL (10 ckpts) | [1 note][t234] | not measured |
| moe_w4a16::`moe_{fp8_grouped_gemm_ptrtable_t, w4a16_fused_gate_up_t, w4a16_fused_gate_up_t_k64, w4a16_fused_gate_up_t_k64_m128, w4a16_grouped_gemm_ptrtable, w4a16_grouped_gemm_ptrtable_t, w4a16_grouped_gemm_ptrtable_t_k64}` (7) | [gb10/minimax-m2-229b/nvfp4/moe_w4a16_grouped_gemm.cu:34][f248] | expert GEMM/GEMV | gb10 | Laguna, MiniMax-M2, Step-3.7 (4 ckpts) | [1 note][t248] | not measured |
| moe_w4a16::`moe_w4a16_grouped_gemm_{ptrtable, ptrtable_relu2, ptrtable_t}` (3) | [gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/moe_w4a16_grouped_gemm.cu:590][f258] | expert GEMM/GEMV | gb10 | Nemotron-H (3 ckpts) | [2 notes][t258] | not measured |
| moe_w4a4::`moe_w4a4_grouped_gemm_relu2` | [gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/moe_w4a4_grouped.cu:49][f259] | expert GEMM/GEMV | gb10 | Nemotron-H (3 ckpts) | [1 note][t259] | not measured |
| moe_w4a16::`moe_{fp8_grouped_gemm_ptrtable_t, w4a16_fused_gate_up_t, w4a16_fused_gate_up_t_k64, w4a16_grouped_gemm_ptrtable, w4a16_grouped_gemm_ptrtable_t, w4a16_grouped_gemm_ptrtable_t_k64}` (6) | [gb10/qwen3.6-27b/nvfp4/moe_w4a16_grouped_gemm.cu:135][f271] | expert GEMM/GEMV | gb10 hop strix | none — its callers' targets compile another copy | [1 note][t271] | not measured |
| moe_w4a16::`moe_{fp8_grouped_gemm_ptrtable_t, w4a16_down_t_k64_fp4, w4a16_fused_gate_up_t, w4a16_fused_gate_up_t_k64, w4a16_fused_gate_up_t_k64_fp4, w4a16_grouped_gemm_ptrtable, w4a16_grouped_gemm_ptrtable_k32, w4a16_grouped_gemm_ptrtable_m256, w4a16_grouped_gemm_ptrtable_t, w4a16_grouped_gemm_ptrtable_t_k64}` (10) | [gb10/qwen3.6-35b-a3b/nvfp4/moe_w4a16_grouped_gemm.cu:34][f282] | expert GEMM/GEMV | b200 gb10 hop strix | Qwen-GDN-MoE, Qwen3.8-FN (7 ckpts) | [9 notes][t282] | not measured |
| moe_silu_mul::`moe_silu_mul` | [gb10/step3p7-flash/nvfp4/moe_silu_mul.cu:31][f289] | expert activation | gb10 | Step-3.7 (1 ckpts) | [1 note][t289] | not measured |
| moe_bucket_builder::`bucket_builder` | [hopper/common/moe_bucket_builder.cu:5][f295] | dispatch / combine | hop | DeepSeek-V4, Nemotron-H, Qwen-GDN-MoE (11 ckpts) | [2 notes][t295] · [#25][pr25] | not measured |
| moe_w8a8_m16::`pm4_m16` | [hopper/common/moe_w8a8_m16.cu:106][f296] | expert GEMM/GEMV | hop | DeepSeek-V4, Nemotron-H, Qwen-GDN-MoE (11 ckpts) | [3 notes][t296] · [#25][pr25] | not measured |
| moe_fp8_grouped_gemm::`moe_fp8_grouped_gemm` | [strix-hip/common/moe_fp8_grouped_gemm.cu:249][f353] | expert GEMM/GEMV | hip | Qwen-GDN-MoE (6 ckpts) | [2 notes][t353] | not measured |
| moe_w4a16::`moe_{fp8_grouped_gemm_ptrtable_t, w4a16_fused_gate_up_t, w4a16_fused_gate_up_t_k64, w4a16_grouped_gemm_ptrtable, w4a16_grouped_gemm_ptrtable_t, w4a16_grouped_gemm_ptrtable_t_k64}` (6) | [strix-hip/qwen3.6-27b/nvfp4/moe_w4a16_grouped_gemm.cu:61][f356] | expert GEMM/GEMV | hip | Qwen-GDN-MoE (6 ckpts) | [4 notes][t356] | not measured |

Also launched here: [Activations and elementwise](#activations-and-elementwise-silu-gelu-relu-residual-gates-scale): `relu_squared_inplace`, `bf16_residual_add`, `gelu_mul`, `gelu`; [Normalization](#normalization-rmsnorm-layernorm-l2-gated-norms): `rms_norm`, `rms_norm_residual`, `rms_{norm, norm_residual}` (2), `rms_{norm, norm_residual}` (2), `rms_{norm, norm_residual}` (2), `rms_{norm, norm_residual}` (2), `rms_{norm, norm_residual}` (2); [Projection GEMM/GEMV — BF16/F32](#projection-gemm-gemv-bf16-f32): `dense_gemm_bf16`, `dense_gemm_bf16_f32out`, `dense_gemm_bf16_pipelined`, `dense_gemm_f32in_f32out`, `dense_gemv_bf16`, `dense_gemv_bf16_fp32out`, `dense_gemv_bf16_batchm`, `dense_gemm_{bf16, bf16_f32out, bf16_pipelined}` (3), `dense_gemm_f32in_f32out`; [Projection GEMM/GEMV — FP8](#projection-gemm-gemv-fp8-w8a16-w8a8-block-scaled): `fp8_gemm_t_blockscaled`, `w8a16_gemm`, `w8a16_gemm_pipelined`, `w8a16_gemv`, `fp8_gemm_t`, `fp8_gemm_{t, t_m128_mfast}` (2), `fp8_gemm_t`, `fp8_gemm_t`, `w8a16_gemv`, `w8a16_gemm`, `fp8_gemm_t`, `fp8_gemm_t`; [Projection GEMM/GEMV — NVFP4 W4A16](#projection-gemm-gemv-nvfp4-w4a16): `w4a16_{gemm, gemm_t}` (2), `w4a16_{gemv, gemv_batch2, gemv_batch3}` (3), `w4a16_gemv_sw`, `w4a16_gemv_sw_{moe, moe_batchm_m2, moe_batchm_m3, moe_batchm_m4, moe_batchm_m5, moe_batchm_m6, moe_batchm_m7, moe_batchm_m8}` (8), `w4a16_{gemm, gemm_t, gemm_t_m128}` (3), `w4a16_{gemm, gemm_t, gemm_t_m128}` (3), `w4a16_{gemm, gemm_t}` (2), `w4a16_gemm_t_m128`, `w4a16_{gemm, gemm_t_m128}` (2), `w4a16_gemm_t`, `w4a16_{gemm, gemm_t, gemm_t_m128}` (3), `w4a16_{gemm, gemm_t, gemm_t_m128}` (3); [Projection GEMM/GEMV — W4A4](#projection-gemm-gemv-w4a4-fp4-activations): `w4a4_gemm_mfast`; [Projection GEMM/GEMV — integer / K-quant](#projection-gemm-gemv-integer-k-quant-q2-0-q2-k-q6-k-int8-mlx-int8): `kquant_mmvq_q2_k_w`, `kquant_mmvq_q3_k_w`, `metrale_q2_k_mmq128_nc`, `metrale_q2_k_mmq128_wc`, `metrale_q3_k_mmq128_nc`, `metrale_q3_k_mmq128_wc`; [Quantization and format conversion](#quantization-and-format-conversion): `nvfp4_tc_lean_repack`, `silu_mul_quant_fp8`, `quantize_bf16_to_nvfp4`, `kquant_q8_1_rows_bf16`, `kquant_swiglu_q8_1_rows_bf16`, `metrale_q8_1_quantize_d2s6_bf16`, `metrale_q8_1_quantize_d4_bf16`, `bf16_to_fp8`, `predequant_nvfp4_to_fp8`, `bf16_to_fp8`, `predequant_nvfp4_to_fp8`, `bf16_to_fp8`, `predequant_nvfp4_to_fp8`, `bf16_to_fp8`, `predequant_nvfp4_to_fp8`, `bf16_to_fp8`, `predequant_nvfp4_to_fp8`, `bf16_to_fp8`, `predequant_nvfp4_to_fp8`.

### Dense FFN (gate/up/down projections of non-MoE layers)

88 entry points: 0 primary here (full rows), 88 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

Also launched here: [Activations and elementwise](#activations-and-elementwise-silu-gelu-relu-residual-gates-scale): `gelu_mul`, `metrale_nvfp4_silu_mul_quant`, `metrale_nvfp4_silu_mul_scaled`, `silu_mul_strided`, `gelu`; [MoE](#moe-routing-dispatch-expert-gemm-gemv-combine): `moe_silu_mul`, `moe_silu_mul`, `moe_silu_mul`; [Projection GEMM/GEMV — BF16/F32](#projection-gemm-gemv-bf16-f32): `k3_dense_{down_f32io, gate_up_situ_f32io}` (2), `dense_gemm_bf16`, `dense_gemm_tc`, `dense_gemv_bf16`, `dense_gemm_bf16`, `dense_gemv_bf16`, `dense_gemm_bf16`, `dense_gemm_tc`; [Projection GEMM/GEMV — FP8](#projection-gemm-gemv-fp8-w8a16-w8a8-block-scaled): `w8a16_gemv_{batch16, batch4}` (2), `fp8_gemm_t_blockscaled`, `w8a16_gemm`, `w8a16_gemm_pipelined`, `w8a16_gemm_t_m128`, `w8a16_gemv`, `w8a16_gemv_{batch16, batch4}` (2), `w8a16_gemv_{dual, silu_input}` (2), `w8a16_gemm_{m16, m16_n64}` (2), `w8a16_gemv`, `w8a16_gemv_{dual, silu_input}` (2), `w8a16_gemm`; [Projection GEMM/GEMV — NVFP4 W4A16](#projection-gemm-gemv-nvfp4-w4a16): `w4a16_gemm`, `w4a16_{gemv, gemv_batch2, gemv_batch3, gemv_dual_batch2, gemv_dual_batch3}` (5), `w4a16_gemv_sw`, `w4a16_gemv_{dual, dual_sw}` (2), `w4a16_gemv_silu_{input, input_sw}` (2), `w4a16_{gemm, gemm_t_m128}` (2), `w4a16_{gemm, gemm_t_m128}` (2), `w4a16_{gemm, gemm_t_m128_bf16, gemm_t_m128_bf16_v2}` (3), `w4a16_gemm_t_m128`, `w4a16_{gemm, gemm_t_m128}` (2), `w4a16_{gemm, gemm_t_m128}` (2), `w4a16_{gemm, gemm_t_m128}` (2); [Projection GEMM/GEMV — W4A4](#projection-gemm-gemv-w4a4-fp4-activations): `w4a4_gemm`, `metrale_nvfp4_mmq128_nc`, `metrale_nvfp4_{mmq128_wc, mmq16_nc, mmq16_wc, mmq32_wc, mmq64_nc, mmq64_wc}` (6), `metrale_nvfp4_mmq32_nc`, `w4a4_gemm`; [Projection GEMM/GEMV — integer / K-quant](#projection-gemm-gemv-integer-k-quant-q2-0-q2-k-q6-k-int8-mlx-int8): `q2_0_gemv_vec`, `q2_0_gemv_vec_batchm`, `metrale_q2_0_mmq128_{nc, wc}` (2), `metrale_q4k_mmq128_{nc, wc}` (2), `int8_gemm_faith2`, `int8_gemm_i32acc`, `requant_a_bf16_int8`, `requant_w_nvfp4_int8`; [Quantization and format conversion](#quantization-and-format-conversion): `dequant_q2_0_gn_to_bf16`, `dequant_nvfp4_to_bf16`, `fp8_act_scale_to_kmajor`, `quantize_bf16_to_nvfp4`, `metrale_nvfp4_quantize_bf16`, `metrale_nvfp4_repack`, `metrale_nvfp4_scale_bf16`, `metrale_q8_1_quantize_ds4_bf16`, `q4k_quantize`.

### Projection GEMM/GEMV — BF16/F32

27 entry points: 27 primary here (full rows), 0 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| dense_f32io::`k3_dense_{down_f32io, gate_up_situ_f32io}` (2) | [b200/kimi-k3/bf16/dense_f32io.cu:12][f1] | BF16/F32 GEMM/GEMV | b200 | Kimi-K3 (1 ckpts) | [1 note][t1] | not measured |
| gemm::`dense_gemm_bf16` | [gb10/common/dense_gemm_bf16.cu:26][f14] | BF16/F32 GEMM/GEMV | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [1 note][t14] | [1–2%][m14.dense_gemm_bf16] (prefill 32k (cold, 32772 tok)) |
| gemm::`dense_gemm_bf16_f32out` | [gb10/common/dense_gemm_bf16.cu:85][f14] | BF16/F32 GEMM/GEMV | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [1 note][t14] | not measured |
| gemm::`dense_gemm_bf16_pipelined` | [gb10/common/dense_gemm_bf16.cu:446][f14] | BF16/F32 GEMM/GEMV | b200 b300 gb10 hop strix | all 14 decoder families + NLLB (31 ckpts) | [1 note][t14] | not measured |
| gemm::`dense_gemm_f32in_f32out` | [gb10/common/dense_gemm_bf16.cu:131][f14] | BF16/F32 GEMM/GEMV | b200 b300 gb10 hop strix | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | — | not measured |
| gemm_splitk::`dense_gemm_splitk_{partial, reduce}` (2) | [gb10/common/dense_gemm_splitk.cu:27][f15] | BF16/F32 GEMM/GEMV | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t15] | not measured |
| gemm_tc::`dense_gemm_{tc, tc_scaled_acc}` (2) | [gb10/common/dense_gemm_tc.cu:185][f16] | BF16/F32 GEMM/GEMV | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [2 notes][t16] | not measured |
| gemv::`dense_gemv_bf16` | [gb10/common/dense_gemv_bf16.cu:33][f17] | BF16/F32 GEMM/GEMV | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [3 notes][t17] | [94–100%][m17.dense_gemv_bf16] (decode C=1 (R=4, MTP k=3)) |
| gemv::`dense_gemv_bf16_fp32out` | [gb10/common/dense_gemv_bf16.cu:120][f17] | BF16/F32 GEMM/GEMV | b200 b300 gb10 hop strix hip | GLM-5.3, GPT-OSS (2 ckpts) | [3 notes][t17] | not measured |
| dense_gemv_bf16_batch2::`dense_gemv_bf16_batch2` | [gb10/common/dense_gemv_bf16_batch2.cu:32][f18] | BF16/F32 GEMM/GEMV | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t18] | not measured |
| dense_gemv_bf16_batchm::`dense_gemv_bf16_batchm` | [gb10/common/dense_gemv_bf16_batchm.cu:226][f19] | BF16/F32 GEMM/GEMV | b200 b300 gb10 hop | all 14 decoder families + GPT-OSS (31 ckpts) | [2 notes][t19] | [16–92%][m19.dense_gemv_bf16_batchm] (decode C=16 (R=32, MTP k=1)) |
| dense_gemv_bf16_batchm::`dense_gemv_bf16_batchm_fp32out` | [gb10/common/dense_gemv_bf16_batchm.cu:233][f19] | BF16/F32 GEMM/GEMV | b200 b300 gb10 hop | GPT-OSS (1 ckpts) | [2 notes][t19] | not measured |
| dense_gemv_bf16_tc::`dense_gemv_bf16_tc16` | [gb10/common/dense_gemv_bf16_tc.cu:251][f20] | BF16/F32 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | [4 notes][t20] · [#1][pr1] | [86–95%][m20.dense_gemv_bf16_tc16] (decode C=16 (R=32, MTP k=1)) |
| dense_gemv_bf16_tc::`dense_gemv_bf16_{tc32, tc8}` (2) | [gb10/common/dense_gemv_bf16_tc.cu:250][f20] | BF16/F32 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | [4 notes][t20] · [#1][pr1] | not measured |
| dense_gemm_m16_bf16::`dense_gemm_m16_{bf16, bf16_n64}` (2) | [hopper/common/dense_gemm_m16_bf16.cu:339][f290] | BF16/F32 GEMM/GEMV | hop | DeepSeek-V4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE (18 ckpts) | [2 notes][t290] | not measured |
| dense_gemm_bf16::`dense_gemm_bf16` | [metal/common/dense_gemm_bf16.metal:23][f318] | BF16/F32 GEMM/GEMV | metal | Qwen-GDN (7 ckpts) | [1 note][t318] | not measured |
| dense_gemv_bf16::`dense_gemv_bf16` | [metal/common/dense_gemv_bf16.metal:27][f319] | BF16/F32 GEMM/GEMV | metal | Qwen-GDN (7 ckpts) | [1 note][t319] | not measured |
| gemm::`dense_gemm_{bf16, bf16_f32out, bf16_pipelined}` (3) | [strix-hip/common/dense_gemm_bf16.cu:26][f351] | BF16/F32 GEMM/GEMV | hip | Qwen-GDN, Qwen-GDN-MoE (13 ckpts) | [2 notes][t351] | not measured |
| gemm::`dense_gemm_f32in_f32out` | [strix-hip/common/dense_gemm_bf16.cu:129][f351] | BF16/F32 GEMM/GEMV | hip | Qwen-GDN-MoE (6 ckpts) | [1 note][t351] | not measured |
| gemm_tc::`dense_gemm_tc` | [strix-hip/common/dense_gemm_tc.cu:31][f352] | BF16/F32 GEMM/GEMV | hip | Qwen-GDN, Qwen-GDN-MoE (13 ckpts) | [1 note][t352] | not measured |

### Projection GEMM/GEMV — FP8 (W8A16, W8A8, block-scaled)

87 entry points: 87 primary here (full rows), 0 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| w8a16_gemv_batch4::`w8a16_gemv_{batch16, batch16_strided, batch4, batch4_strided}` (4) | [b300/common/w8a16_gemv_batch4.cu:213][f4] | FP8 GEMM/GEMV | b300 | Kimi-K3 (1 ckpts) | [1 note][t4] | not measured |
| gemv_fp8w::`dense_gemv_fp8w` | [gb10/common/dense_gemv_fp8w.cu:131][f21] | FP8 GEMM/GEMV | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t21] | not measured |
| dense_gemv_fp8w_batch2::`dense_gemv_fp8w_batch2` | [gb10/common/dense_gemv_fp8w_batch2.cu:72][f22] | FP8 GEMM/GEMV | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t22] | not measured |
| fp8_gemm_blockscaled_pipe::`fp8_gemm_blockscaled_pipe_128x64` | [gb10/common/fp8_gemm_blockscaled_pipe.cu:63][f30] | FP8 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | — | not measured |
| fp8_gemm_t_blockscaled::`fp8_gemm_t_blockscaled` | [gb10/common/fp8_gemm_t_blockscaled.cu:113][f31] | FP8 GEMM/GEMV | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [3 notes][t31] | [14–49%][m31.fp8_gemm_t_blockscaled] (prefill 32k (cold, 32772 tok)) |
| fp8_gemv_rt::`fp8_gemv_rowscale_{batch16_rt2, batch8_rt2}` (2) | [gb10/common/fp8_gemv_rt.cu:156][f32] | FP8 GEMM/GEMV | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t32] | not measured |
| w4a16_fp8_ldmab::`fp8_fp8_gemm_ldmab` | [gb10/common/w4a16_fp8_ldmab.cu:65][f182] | FP8 GEMM/GEMV | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [2 notes][t182] | [11–43%][m182.fp8_fp8_gemm_ldmab] (prefill 32k (cold, 32772 tok)) |
| w8a16_gemm::`w8a16_gemm` | [gb10/common/w8a16_gemm.cu:86][f189] | FP8 GEMM/GEMV | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [1 note][t189] | not measured |
| w8a16_gemm_pipe128::`w8a16_gemm_pipe128` | [gb10/common/w8a16_gemm_pipe128.cu:58][f190] | FP8 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | — | not measured |
| w8a16_gemm_pipelined::`w8a16_gemm_pipelined` | [gb10/common/w8a16_gemm_pipelined.cu:174][f191] | FP8 GEMM/GEMV | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [1 note][t191] | [23–24%][m191.w8a16_gemm_pipelined] (prefill 32k (cold, 32772 tok)) |
| w8a16_gemm_pipelined_m32::`w8a16_gemm_pipelined_m32` | [gb10/common/w8a16_gemm_pipelined_m32.cu:159][f192] | FP8 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | [7 notes][t192] · [#4][pr4] [#34][pr34] | [20–95%][m192.w8a16_gemm_pipelined_m32] (decode C=16 (R=32, MTP k=1)) |
| w8a16_gemm_pipelined_m32::`w8a16_gemm_pipelined_m64` | [gb10/common/w8a16_gemm_pipelined_m32.cu:318][f192] | FP8 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | [6 notes][t192] · [#4][pr4] [#34][pr34] | not measured |
| w8a16_gemm_t::`w8a16_gemm_{t, t_pipelined}` (2) | [gb10/common/w8a16_gemm_t.cu:151][f193] | FP8 GEMM/GEMV | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [1 note][t193] | not measured |
| w8a16_gemm_t_m128::`w8a16_gemm_t_m128` | [gb10/common/w8a16_gemm_t_m128.cu:62][f194] | FP8 GEMM/GEMV | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t194] | [28%][m194.w8a16_gemm_t_m128] (prefill 4k (cold, 4549 tok)) |
| w8a16_gemv::`w8a16_gemv` | [gb10/common/w8a16_gemv.cu:110][f195] | FP8 GEMM/GEMV | b200 b300 gb10 strix hip | all 14 decoder families (30 ckpts) | [2 notes][t195] | not measured |
| w8a16_gemv_batch4::`w8a16_gemv_{batch16, batch16_strided, batch4, batch4_strided}` (4) | [gb10/common/w8a16_gemv_batch4.cu:234][f196] | FP8 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | [2 notes][t196] | not measured |
| w8a16_gemv_fused::`w8a16_gemv_{dual, silu_input}` (2) | [gb10/common/w8a16_gemv_fused.cu:123][f197] | FP8 GEMM/GEMV | b200 b300 gb10 | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN, Step-3.7 (29 ckpts) | [1 note][t197] | not measured |
| w8a16_tc_rows::`w8a16_tc_rows_{16, 32, 64, 64c}` (4) | [gb10/common/w8a16_tc_rows.cu:38][f198] | FP8 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t198] | not measured |
| w8a8_gemv::`w8a8_gemv_{blk128_mb16, blk128_mb1_ku8, blk128_mb2, blk128_mb4, blk128_mb8, rowscale_mb16, rowscale_mb1_ku8, rowscale_mb2, rowscale_mb4, rowscale_mb8}` (10) | [gb10/common/w8a8_gemv.cu:171][f200] | FP8 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | — | not measured |
| w4a16::`fp8_{fp8_gemm_t, fp8_gemm_t_m128, gemm_t, gemm_t_m128}` (4) | [gb10/deepseek-v4-flash/nvfp4/w4a16_gemm.cu:382][f225] | FP8 GEMM/GEMV | b200 gb10 hop | DeepSeek-V4, Gemma4, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE, Qwen3-VL, Step-3.7 (27 ckpts) | [1 note][t225] | not measured |
| w4a16_v2::`w4a16_gemm_t_m128_v2` | [gb10/minimax-m2-229b/nvfp4/w4a16_gemm_v2.cu:72][f250] | FP8 GEMM/GEMV | gb10 | MiniMax-M2, Step-3.7 (2 ckpts) | [1 note][t250] | not measured |
| w4a16_v3::`w4a16_gemm_t_m128_v3` | [gb10/minimax-m2-229b/nvfp4/w4a16_gemm_v3.cu:73][f251] | FP8 GEMM/GEMV | gb10 | MiniMax-M2, Step-3.7 (2 ckpts) | [1 note][t251] | not measured |
| w4a16::`fp8_{fp8_gemm_t, fp8_gemm_t_m128, fp8_gemm_t_m128_mfast, gemm_t, gemm_t_m128, gemm_t_m128_mfast}` (6) | [gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/w4a16_gemm.cu:555][f261] | FP8 GEMM/GEMV | gb10 | Nemotron-H (3 ckpts) | [1 note][t261] | not measured |
| w4a16::`fp8_{fp8_gemm_t, fp8_gemm_t_m128, gemm_t, gemm_t_row_scaled, gemm_t_row_scaled_k64, gemm_t_row_scaled_m16, gemm_t_row_scaled_p4}` (7) | [gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu:820][f276] | FP8 GEMM/GEMV | gb10 hop strix | Qwen-GDN (7 ckpts) | [4 notes][t276] | not measured |
| w4a16::`fp8_gemm_t_m128` | [gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu:2745][f276] | FP8 GEMM/GEMV | gb10 hop strix | Qwen-GDN (7 ckpts) | [3 notes][t276] | [14–17%][m276.fp8_gemm_t_m128] (prefill 32k (cold, 32772 tok)) |
| w4a16_v2::`w4a16_gemm_t_m128_v2` | [gb10/qwen3.6-27b/nvfp4/w4a16_gemm_v2.cu:100][f277] | FP8 GEMM/GEMV | gb10 hop | Qwen-GDN (7 ckpts) | [1 note][t277] | not measured |
| w4a16::`fp8_{fp8_gemm_t, fp8_gemm_t_m128, gemm_t, gemm_t_m128, gemm_t_row_scaled, gemm_t_row_scaled_m16}` (6) | [gb10/qwen3.6-35b-a3b/nvfp4/w4a16_gemm.cu:630][f284] | FP8 GEMM/GEMV | b200 gb10 hop strix | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [2 notes][t284] · [#34][pr34] | not measured |
| w8a16_gemm_m16::`w8a16_gemm_{m16, m16_n64, m16_strided}` (3) | [hopper/common/w8a16_gemm_m16.cu:382][f300] | FP8 GEMM/GEMV | hop | DeepSeek-V4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE (18 ckpts) | [3 notes][t300] | not measured |
| w8a16_gemv::`w8a16_gemv` | [hopper/common/w8a16_gemv.cu:54][f301] | FP8 GEMM/GEMV | hop | DeepSeek-V4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE (18 ckpts) | [3 notes][t301] | not measured |
| w8a16_gemv_fused::`w8a16_gemv_{dual, silu_input}` (2) | [hopper/common/w8a16_gemv_fused.cu:66][f302] | FP8 GEMM/GEMV | hop | DeepSeek-V4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE (18 ckpts) | [2 notes][t302] | not measured |
| w8a16_gemv_ncol::`w8a16_gemv_batch16_{ncol2, ncol2_strided, ncol4, ncol4_strided}` (4) | [hopper/common/w8a16_gemv_ncol.cu:199][f303] | FP8 GEMM/GEMV | hop | DeepSeek-V4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE (18 ckpts) | [2 notes][t303] | not measured |
| w8a16_gemm::`w8a16_gemm` | [strix-hip/common/w8a16_gemm.cu:230][f354] | FP8 GEMM/GEMV | hip | Qwen-GDN, Qwen-GDN-MoE (13 ckpts) | [3 notes][t354] | not measured |
| w8a16_gemm_t::`w8a16_gemm_t` | [strix-hip/common/w8a16_gemm_t.cu:119][f355] | FP8 GEMM/GEMV | hip | Qwen-GDN, Qwen-GDN-MoE (13 ckpts) | [3 notes][t355] | not measured |
| w4a16::`fp8_{fp8_gemm_t, fp8_gemm_t_m128, gemm_t, gemm_t_m128}` (4) | [strix-hip/qwen3.6-27b/nvfp4/w4a16_gemm.cu:324][f357] | FP8 GEMM/GEMV | hip | Qwen-GDN (7 ckpts) | [1 note][t357] | not measured |
| w4a16::`fp8_{fp8_gemm_t, fp8_gemm_t_m128, gemm_t, gemm_t_m128}` (4) | [strix-hip/qwen3.6-35b-a3b/nvfp4/w4a16_gemm.cu:323][f358] | FP8 GEMM/GEMV | hip | Qwen-GDN-MoE (6 ckpts) | [2 notes][t358] | not measured |

### Projection GEMM/GEMV — NVFP4 W4A16

63 entry points: 63 primary here (full rows), 0 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| w4a16::`w4a16_{gemm, gemm_t}` (2) | [gb10/common/w4a16_gemm.cu:87][f183] | NVFP4 W4A16 GEMM/GEMV | b200 b300 gb10 | GLM-5.3, Kimi-K3 (2 ckpts) | [1 note][t183] | not measured |
| w4a16_gemv::`w4a16_{gemv, gemv_batch16, gemv_batch2, gemv_batch3, gemv_batch32, gemv_batch8, gemv_batch8_rt2, gemv_dual_batch2, gemv_dual_batch3, gemv_logits, gemv_qg, gemv_qg_batch2, gemv_qg_batch3}` (13) | [gb10/common/w4a16_gemv.cu:167][f184] | NVFP4 W4A16 GEMM/GEMV | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [6 notes][t184] | not measured |
| w4a16_gemv::`w4a16_gemv_qkvz` | [gb10/common/w4a16_gemv.cu:1557][f184] | NVFP4 W4A16 GEMM/GEMV | b200 b300 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t184] | not measured |
| w4a16_gemv::`w4a16_gemv_sw` | [gb10/common/w4a16_gemv.cu:233][f184] | NVFP4 W4A16 GEMM/GEMV | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t184] | [96–99%][m184.w4a16_gemv_sw] (decode C=1 (R=4, MTP k=3)) |
| w4a16_gemv::`w4a16_gemv_sw_{moe, moe_batchm_m2, moe_batchm_m3, moe_batchm_m4, moe_batchm_m5, moe_batchm_m6, moe_batchm_m7, moe_batchm_m8}` (8) | [gb10/common/w4a16_gemv.cu:300][f184] | NVFP4 W4A16 GEMM/GEMV | b200 b300 gb10 hop strix hip | GLM-5.3 (1 ckpts) | [1 note][t184] | not measured |
| w4a16_gemv_fused::`w4a16_gemv_{dual, dual_sw}` (2) | [gb10/common/w4a16_gemv_fused.cu:144][f185] | NVFP4 W4A16 GEMM/GEMV | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t185] | not measured |
| w4a16_gemv_fused::`w4a16_gemv_silu_{input, input_sw}` (2) | [gb10/common/w4a16_gemv_fused.cu:209][f185] | NVFP4 W4A16 GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN, Step-3.7 (29 ckpts) | [2 notes][t185] | not measured |
| w4a16_gemv_tc::`w4a16_gemv_tc16` | [gb10/common/w4a16_gemv_tc.cu:256][f186] | NVFP4 W4A16 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | [4 notes][t186] · [#1][pr1] | [86–92%][m186.w4a16_gemv_tc16] (decode C=16 (R=32, MTP k=1)) |
| w4a16_gemv_tc::`w4a16_gemv_tc8` | [gb10/common/w4a16_gemv_tc.cu:255][f186] | NVFP4 W4A16 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | [4 notes][t186] · [#1][pr1] | [59–90%][m186.w4a16_gemv_tc8] (decode C=1 (R=4, MTP k=3)) |
| w4a16_tc_rows::`w4a16_tc_rows_{16, 32, 64}` (3) | [gb10/common/w4a16_tc_rows.cu:26][f187] | NVFP4 W4A16 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | — | not measured |
| w4a16::`w4a16_{gemm, gemm_t, gemm_t_k64, gemm_t_m128}` (4) | [gb10/deepseek-v4-flash/nvfp4/w4a16_gemm.cu:32][f225] | NVFP4 W4A16 GEMM/GEMV | b200 gb10 hop | DeepSeek-V4, Gemma4, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE, Qwen3-VL, Step-3.7 (27 ckpts) | [1 note][t225] | not measured |
| w4a16::`w4a16_{gemm, gemm_t, gemm_t_k64, gemm_t_m128}` (4) | [gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/w4a16_gemm.cu:32][f261] | NVFP4 W4A16 GEMM/GEMV | gb10 | Nemotron-H (3 ckpts) | [1 note][t261] | not measured |
| w4a16::`w4a16_{gemm, gemm_t, gemm_t_k64, gemm_t_k64_p3, gemm_t_m128_bf16, gemm_t_m128_bf16_v2}` (6) | [gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu:151][f276] | NVFP4 W4A16 GEMM/GEMV | gb10 hop strix | Qwen-GDN (7 ckpts) | [6 notes][t276] | not measured |
| w4a16::`w4a16_gemm_t_k64_n64_p3` | [gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu:1648][f276] | NVFP4 W4A16 GEMM/GEMV | gb10 hop strix | Qwen-GDN (7 ckpts) | [3 notes][t276] | [54%][m276.w4a16_gemm_t_k64_n64_p3] (decode C=16 (R=32, MTP k=1)) |
| w4a16::`w4a16_gemm_t_m128` | [gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu:1860][f276] | NVFP4 W4A16 GEMM/GEMV | gb10 hop strix | Qwen-GDN (7 ckpts) | [2 notes][t276] | [28–32%][m276.w4a16_gemm_t_m128] (prefill 4k (cold, 4103 tok)) |
| w4a16::`w4a16_gemm_t_p3` | [gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu:585][f276] | NVFP4 W4A16 GEMM/GEMV | gb10 hop strix | Qwen-GDN (7 ckpts) | [3 notes][t276] | [8–78%][m276.w4a16_gemm_t_p3] (decode C=16 (R=32, MTP k=1)) |
| w4a16::`w4a16_{gemm, gemm_t_k64, gemm_t_m128}` (3) | [gb10/qwen3.6-35b-a3b/nvfp4/w4a16_gemm.cu:151][f284] | NVFP4 W4A16 GEMM/GEMV | b200 gb10 hop strix | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [4 notes][t284] · [#34][pr34] | not measured |
| w4a16::`w4a16_gemm_t` | [gb10/qwen3.6-35b-a3b/nvfp4/w4a16_gemm.cu:345][f284] | NVFP4 W4A16 GEMM/GEMV | b200 gb10 hop strix | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [6 notes][t284] · [#34][pr34] | [73–87%][m284.w4a16_gemm_t] (decode C=16 (R=32, MTP k=1)) |
| w4a16::`w4a16_{gemm, gemm_t, gemm_t_k64, gemm_t_m128}` (4) | [strix-hip/qwen3.6-27b/nvfp4/w4a16_gemm.cu:91][f357] | NVFP4 W4A16 GEMM/GEMV | hip | Qwen-GDN (7 ckpts) | [4 notes][t357] | not measured |
| w4a16::`w4a16_{gemm, gemm_t, gemm_t_k64, gemm_t_m128}` (4) | [strix-hip/qwen3.6-35b-a3b/nvfp4/w4a16_gemm.cu:91][f358] | NVFP4 W4A16 GEMM/GEMV | hip | Qwen-GDN-MoE (6 ckpts) | [4 notes][t358] | not measured |

### Projection GEMM/GEMV — W4A4 (FP4 activations)

22 entry points: 22 primary here (full rows), 0 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| w4a4_gemv_mx::`w4a4_{gemv_mx16, gemv_mx16_nt2, gemv_mx16_ps, gemv_mx32, gemv_mx32_nt4, gemv_mx32_ps, gemv_mx64, gemv_mx64_nt2, gemv_mx8, quant_rows}` (10) | [gb10/common/w4a4_gemv_mx.cu:360][f188] | W4A4 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | [23 notes][t188] · [#1][pr1] [#14][pr14] [#18][pr18] | not measured |
| w4a4::`w4a4_{gemm, gemm_mfast}` (2) | [gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/w4a4_gemm.cu:115][f262] | W4A4 GEMM/GEMV | gb10 | Nemotron-H (3 ckpts) | [1 note][t262] | not measured |
| nvfp4_mmq::`metrale_nvfp4_{gemm_pipe, mmq128_wc, mmq16_nc, mmq16_wc, mmq32_wc, mmq64_nc, mmq64_wc}` (7) | [gb10/qwen3.6-27b/nvfp4/nvfp4_mmq.cu:72][f272] | W4A4 GEMM/GEMV | gb10 hop | Qwen-GDN (7 ckpts) | [3 notes][t272] | not measured |
| nvfp4_mmq::`metrale_nvfp4_mmq128_nc` | [gb10/qwen3.6-27b/nvfp4/nvfp4_mmq.cu:67][f272] | W4A4 GEMM/GEMV | gb10 hop | Qwen-GDN (7 ckpts) | [3 notes][t272] | [17–18%][m272.metrale_nvfp4_mmq128_nc] (prefill 4k (cold, 4103 tok)) |
| nvfp4_mmq::`metrale_nvfp4_mmq32_nc` | [gb10/qwen3.6-27b/nvfp4/nvfp4_mmq.cu:97][f272] | W4A4 GEMM/GEMV | gb10 hop | Qwen-GDN (7 ckpts) | [3 notes][t272] | [82–87%][m272.metrale_nvfp4_mmq32_nc] (decode C=16 (R=32, MTP k=1)) |
| w4a4::`w4a4_gemm` | [gb10/qwen3.6-27b/nvfp4/w4a4_gemm.cu:50][f278] | W4A4 GEMM/GEMV | gb10 hop | Qwen-GDN (7 ckpts) | [2 notes][t278] | not measured |

### Projection GEMM/GEMV — integer / K-quant (Q2_0, Q2_K..Q6_K, INT8, MLX INT8)

23 entry points: 23 primary here (full rows), 0 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| q2_0_gemv_vec::`q2_0_gemv_vec` | [gb10/common/q2_0_gemv_vec.cu:80][f158] | integer / K-quant GEMM/GEMV | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t158] | not measured |
| q2_0_gemv_vec::`q2_0_gemv_vec_batchm` | [gb10/common/q2_0_gemv_vec.cu:162][f158] | integer / K-quant GEMM/GEMV | b200 b300 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN, Step-3.7 (29 ckpts) | [1 note][t158] | not measured |
| kquant_moe::`kquant_mmvq_q2_k_w`, `kquant_mmvq_q3_k_w`, `kquant_mmvq_q6_k_w`, `metrale_q2_k_mmq128_nc`, `metrale_q2_k_mmq128_wc`, `metrale_q3_k_mmq128_nc`, `metrale_q3_k_mmq128_wc` | [gb10/deepseek-v4-flash/nvfp4/kquant_moe.cu:76][f210] | integer / K-quant GEMM/GEMV | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [4 notes][t210] | not measured |
| q2_0_mmq::`metrale_q2_0_mmq128_{nc, wc}` (2) | [gb10/qwen3.6-27b/nvfp4/q2_0_mmq.cu:60][f273] | integer / K-quant GEMM/GEMV | gb10 hop | Qwen-GDN (7 ckpts) | [1 note][t273] | not measured |
| q4k_mmq::`metrale_q4k_mmq128_{nc, wc}` (2) | [gb10/qwen3.6-27b/nvfp4/q4k_mmq.cu:50][f274] | integer / K-quant GEMM/GEMV | gb10 hop | Qwen-GDN (7 ckpts) | [1 note][t274] | not measured |
| w4a16::`int8_gemm_faith2`, `int8_gemm_i32acc`, `requant_a_bf16_int8`, `requant_w_nvfp4_int8` | [gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu:4569][f276] | integer / K-quant GEMM/GEMV | gb10 hop strix | Qwen-GDN (7 ckpts) | [2 notes][t276] | not measured |
| mlx_int8_dequant::`mlx_int8_dequant` | [metal/common/mlx_int8_dequant.metal:21][f332] | integer / K-quant GEMM/GEMV | metal | Qwen-GDN (7 ckpts) | — | not measured |
| mlx_int8_gemm::`mlx_int8_gemm` | [metal/common/mlx_int8_gemm.metal:23][f333] | integer / K-quant GEMM/GEMV | metal | Qwen-GDN (7 ckpts) | [1 note][t333] | not measured |
| mlx_int8_gemv::`mlx_int8_gemv` | [metal/common/mlx_int8_gemv.metal:39][f334] | integer / K-quant GEMM/GEMV | metal | Qwen-GDN (7 ckpts) | [1 note][t334] | not measured |
| mlx_int8_gemv_gate_up::`mlx_int8_gemv_gate_up` | [metal/common/mlx_int8_gemv_gate_up.metal:41][f335] | integer / K-quant GEMM/GEMV | metal | Qwen-GDN (7 ckpts) | [1 note][t335] | not measured |
| mlx_int8_gemv_silu_gate::`mlx_int8_gemv_silu_{gate, gate_resid}` (2) | [metal/common/mlx_int8_gemv_silu_gate.metal:29][f336] | integer / K-quant GEMM/GEMV | metal | Qwen-GDN (7 ckpts) | [2 notes][t336] | not measured |

### Normalization (RMSNorm, LayerNorm, L2, gated norms)

65 entry points: 65 primary here (full rows), 0 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| residual_add_rms_norm_exact::`residual_add_rms_norm_exact` | [gb10/common/residual_add_rms_norm_exact.cu:28][f167] | normalization | b200 gb10 hop | all 14 decoder families (30 ckpts) | — | not measured |
| norm::`gated_rms_norm`, `gated_rms_norm_f32_input`, `gated_rms_norm_f32_input_strided`, `residual_add_rms_norm_gatef32`, `residual_add_rms_norm_vanilla`, `rms_norm`, `rms_norm_residual_vanilla`, `rms_norm_strided` | [gb10/common/rms_norm.cu:45][f168] | normalization | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Kimi-K3, Laguna, Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (20 ckpts) | [5 notes][t168] | not measured |
| norm::`gated_rms_norm_prefill` | [gb10/common/rms_norm.cu:1277][f168] | normalization | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Kimi-K3, Laguna, Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (20 ckpts) | [2 notes][t168] | [69–96%][m168.gated_rms_norm_prefill] (decode C=16 (R=32, MTP k=1)) |
| norm::`l2_norm_bf16` | [gb10/common/rms_norm.cu:1376][f168] | normalization | b200 b300 gb10 hop strix hip | GLM-5.3, Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (15 ckpts) | [2 notes][t168] | [81–90%][m168.l2_norm_bf16] (prefill 4k (cold, 4103 tok)) |
| norm::`residual_add_rms_norm` | [gb10/common/rms_norm.cu:382][f168] | normalization | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Kimi-K3, Laguna, Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (20 ckpts) | [3 notes][t168] | [6–80%][m168.residual_add_rms_norm] (decode C=1 (R=2, MTP k=1)) |
| norm::`rms_norm_residual` | [gb10/common/rms_norm.cu:259][f168] | normalization | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Kimi-K3, Laguna, Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (20 ckpts) | [2 notes][t168] | [4–90%][m168.rms_norm_residual] (decode C=1 (R=2, MTP k=1)) |
| rms_norm_vanilla::`rms_norm_vanilla` | [gb10/common/rms_norm_vanilla.cu:38][f170] | normalization | b200 b300 gb10 hop strix hip | all 14 decoder families + GPT-OSS (31 ckpts) | — | not measured |
| rms_norm_vanilla::`rms_norm_vanilla_warp_row` | [gb10/common/rms_norm_vanilla.cu:120][f170] | normalization | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t170] | not measured |
| norm::`gated_rms_norm`, `gated_rms_norm_f32_input`, `gated_rms_norm_prefill`, `residual_add_rms_norm`, `residual_add_rms_norm_gatef32`, `rms_norm`, `rms_norm_residual`, `rms_norm_strided` | [gb10/gemma-4-26b-a4b/nvfp4/rms_norm.cu:40][f237] | normalization | gb10 | Gemma4 (2 ckpts) | [1 note][t237] | not measured |
| norm::`l2_norm_bf16` | [gb10/gemma-4-26b-a4b/nvfp4/rms_norm.cu:775][f237] | normalization | gb10 | none — its callers' targets compile another copy | [1 note][t237] | not measured |
| norm::`gated_rms_norm`, `gated_rms_norm_f32_input`, `gated_rms_norm_prefill`, `residual_add_rms_norm`, `residual_add_rms_norm_gatef32`, `rms_norm`, `rms_norm_residual`, `rms_norm_strided` | [gb10/gemma-4-31b/nvfp4/rms_norm.cu:40][f241] | normalization | gb10 | Gemma4 (2 ckpts) | [1 note][t241] | not measured |
| norm::`l2_norm_bf16` | [gb10/gemma-4-31b/nvfp4/rms_norm.cu:796][f241] | normalization | gb10 | none — its callers' targets compile another copy | [1 note][t241] | not measured |
| norm::`gated_rms_norm`, `gated_rms_norm_f32_input`, `gated_rms_norm_prefill`, `residual_add_rms_norm`, `residual_add_rms_norm_gatef32`, `rms_norm`, `rms_norm_residual`, `rms_norm_strided` | [gb10/minimax-m2-229b/nvfp4/rms_norm.cu:42][f249] | normalization | b200 gb10 hop | LongCat, MiniMax-M2, Mistral4, Nemotron-H, Step-3.7 (7 ckpts) | [2 notes][t249] | not measured |
| norm::`l2_norm_bf16` | [gb10/minimax-m2-229b/nvfp4/rms_norm.cu:419][f249] | normalization | b200 gb10 hop | none — its callers' targets compile another copy | [1 note][t249] | not measured |
| norm::`gated_rms_norm`, `gated_rms_norm_f32_input`, `gated_rms_norm_prefill`, `residual_add_rms_norm`, `residual_add_rms_norm_gatef32`, `rms_norm`, `rms_norm_residual`, `rms_norm_strided` | [gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/rms_norm.cu:42][f260] | normalization | b200 gb10 hop | Nemotron-H (3 ckpts) | [2 notes][t260] | not measured |
| norm::`l2_norm_bf16` | [gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/rms_norm.cu:427][f260] | normalization | b200 gb10 hop | none — its callers' targets compile another copy | [1 note][t260] | not measured |
| norm::`gated_rms_norm`, `gated_rms_norm_f32_input`, `gated_rms_norm_prefill`, `residual_add_rms_norm`, `residual_add_rms_norm_gatef32`, `rms_norm`, `rms_norm_residual`, `rms_norm_strided` | [gb10/qwen3-vl-30b-a3b/nvfp4/rms_norm.cu:42][f264] | normalization | gb10 | Qwen3-VL (1 ckpts) | [1 note][t264] | not measured |
| norm::`l2_norm_bf16` | [gb10/qwen3-vl-30b-a3b/nvfp4/rms_norm.cu:378][f264] | normalization | gb10 | none — its callers' targets compile another copy | [1 note][t264] | not measured |
| gated_norm_sigmoid::`gated_rms_norm_{f32_input_sigmoid, prefill_sigmoid, sigmoid}` (3) | [gb10/qwen3.8-flash-next/nvfp4/gated_norm_sigmoid.cu:50][f285] | normalization | gb10 | Qwen3.8-FN (1 ckpts) | [1 note][t285] | not measured |
| add_rms_norm::`add_rms_norm` | [metal/common/add_rms_norm.metal:32][f304] | normalization | metal | Qwen-GDN (7 ckpts) | [1 note][t304] | not measured |
| rms_norm::`rms_norm` | [metal/common/rms_norm.metal:21][f340] | normalization | metal | Qwen-GDN (7 ckpts) | [1 note][t340] | not measured |

### Activations and elementwise (SiLU/GELU/ReLU², residual, gates, scale)

17 entry points: 17 primary here (full rows), 0 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| bf16_add::`bf16_add_inplace` | [gb10/common/bf16_add.cu:8][f12] | activation / gate / residual | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t12] | not measured |
| projection_bias::`projection_bias_bf16` | [gb10/common/projection_bias.cu:14][f156] | FP32 projection bias / BF16 store | b200 gb10 hop | GPT-OSS (1 ckpts) | — | not measured |
| relu2::`relu_squared_inplace` | [gb10/common/relu_squared.cu:26][f162] | activation / gate / residual | b200 b300 gb10 hop strix hip | Nemotron-H (3 ckpts) | — | not measured |
| residual_add::`bf16_concat` | [gb10/common/residual_add.cu:142][f166] | activation / gate / residual | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | — | [4%][m166.bf16_concat] (prefill 4k (cold, 4549 tok)) |
| residual_add::`bf16_residual_add` | [gb10/common/residual_add.cu:10][f166] | activation / gate / residual | b200 b300 gb10 hop strix hip | all 14 decoder families + GPT-OSS (31 ckpts) | — | [97–100%][m166.bf16_residual_add] (prefill 4k (cold, 4103 tok)) |
| residual_add::`bf16_scaled_add`, `sigmoid_gate_mul`, `sigmoid_gate_mul_batched`, `sigmoid_gate_mul_head_broadcast`, `softplus_gate_mul_head_broadcast` | [gb10/common/residual_add.cu:60][f166] | activation / gate / residual | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t166] | not measured |
| gelu::`gelu_mul` | [gb10/gemma-4-26b-a4b/nvfp4/gelu.cu:43][f229] | activation / gate / residual | gb10 | Gemma4 (2 ckpts) | [1 note][t229] | not measured |
| nvfp4_mmq::`metrale_nvfp4_silu_mul_quant` | [gb10/qwen3.6-27b/nvfp4/nvfp4_mmq.cu:242][f272] | activation / gate / residual | gb10 hop | Qwen-GDN (7 ckpts) | [4 notes][t272] | [57–87%][m272.metrale_nvfp4_silu_mul_quant] (decode C=16 (R=32, MTP k=1)) |
| nvfp4_mmq::`metrale_nvfp4_silu_mul_scaled` | [gb10/qwen3.6-27b/nvfp4/nvfp4_mmq.cu:221][f272] | activation / gate / residual | gb10 hop | Qwen-GDN (7 ckpts) | [3 notes][t272] | not measured |
| silu_mul_strided::`silu_mul_strided` | [hopper/common/silu_mul_strided.cu:44][f299] | activation / gate / residual | hop | DeepSeek-V4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE (18 ckpts) | [2 notes][t299] | not measured |
| bf16_add::`bf16_add` | [metal/common/bf16_add.metal:19][f314] | activation / gate / residual | metal | Qwen-GDN (7 ckpts) | — | not measured |
| gelu::`gelu` | [metal/common/gelu.metal:22][f323] | activation / gate / residual | metal | Qwen-GDN (7 ckpts) | — | not measured |
| sigmoid_gate::`sigmoid_gate` | [metal/common/sigmoid_gate.metal:16][f343] | activation / gate / residual | metal | Qwen-GDN (7 ckpts) | — | not measured |

### Positional encoding (RoPE, YaRN, MRoPE)

14 entry points: 14 primary here (full rows), 0 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| gpt_oss_rope::`gpt_oss_{rope_bf16, yarn_frequencies}` (2) | [gb10/common/gpt_oss_rope.cu:7][f60] | continuous YaRN / staged BF16 rotation | b200 gb10 hop | GPT-OSS (1 ckpts) | — | not measured |
| rope::`rope_{forward, forward_proportional, forward_strided, forward_yarn, forward_yarn_interleaved, forward_yarn_interleaved_inv, forward_yarn_scaled}` (7) | [gb10/common/rope.cu:29][f171] | rotary | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (29 ckpts) | [3 notes][t171] | not measured |
| rope_mrope_interleaved::`rope_forward_mrope_interleaved` | [gb10/common/rope_mrope_interleaved.cu:34][f172] | rotary | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t172] | [6%][m172.rope_forward_mrope_interleaved] (prefill 32k (cold, 32772 tok)) |
| rope_mrope_interleaved::`rope_forward_mrope_interleaved_k_only` | [gb10/common/rope_mrope_interleaved.cu:108][f172] | rotary | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t172] | not measured |
| rope::`rope_{forward, forward_yarn}` (2) | [gb10/mistral-small-4/nvfp4/rope.cu:29][f257] | rotary | gb10 | Mistral4 (1 ckpts) | [1 note][t257] | not measured |
| rope_apply::`rope_apply` | [metal/common/rope_apply.metal:33][f341] | rotary | metal | Qwen-GDN (7 ckpts) | — | not measured |

### KV cache (write, quantize, TurboQuant rotation, slot metadata)

36 entry points: 36 primary here (full rows), 0 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| fused_k_norm_rope_cache::`fused_k_norm_rope_{cache_write_bf16, mrope_cache_write_bf16}` (2) | [gb10/common/fused_k_norm_rope_cache.cu:53][f34] | cache write | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t34] | not measured |
| metadata_fill::`fill_slots_from_block_table` | [gb10/common/metadata_fill.cu:5][f69] | cache write | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t69] | not measured |
| reshape_and_cache::`bf16_absmax`, `reshape_and_cache_flash_fp8`, `reshape_and_cache_flash_nvfp4`, `reshape_and_cache_flash_v_only` | [gb10/common/reshape_and_cache.cu:30][f163] | cache write | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [2 notes][t163] | not measured |
| reshape_and_cache::`reshape_and_cache_flash` | [gb10/common/reshape_and_cache.cu:67][f163] | cache write | b200 b300 gb10 hop strix hip | all 14 decoder families + GPT-OSS (31 ckpts) | [1 note][t163] | not measured |
| reshape_and_cache_fused_k_fp8::`fused_k_norm_rope_cache_write_fp8_kv` | [gb10/common/reshape_and_cache_fused_k_fp8.cu:131][f164] | cache write | b200 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t164] | not measured |
| reshape_and_cache_turbo::`reshape_and_cache_flash_{bf16k_turbo2v, bf16k_turbo3v, bf16k_turbo4v, fp8k_turbo2v, fp8k_turbo3v, fp8k_turbo4v, turbo2, turbo3, turbo3k_turbo8v, turbo4, turbo4k_turbo3v, turbo4k_turbo8v, turbo8}` (13) | [gb10/common/reshape_and_cache_turbo.cu:179][f165] | cache write | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [2 notes][t165] | not measured |
| tq_plus_innerq_apply::`tq_plus_innerq_apply_{k, q}` (2) | [gb10/common/tq_plus_innerq_apply.cu:71][f179] | TurboQuant rotation | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t179] | not measured |
| wht_bf16::`wht_bf16_{inplace, inplace_inv}` (2) | [gb10/common/wht_bf16.cu:51][f201] | TurboQuant rotation | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t201] | not measured |
| kv_cache_append::`kv_cache_append` | [metal/common/kv_cache_append.metal:19][f324] | cache write | metal | Qwen-GDN (7 ckpts) | — | not measured |
| kv_cache_append_bf16k_turbov::`kv_cache_append_bf16k_{turbo2v, turbo3v, turbo4v}` (3) | [metal/common/kv_cache_append_bf16k_turbov.metal:129][f325] | cache write | metal | Qwen-GDN (7 ckpts) | [1 note][t325] | not measured |
| kv_cache_append_turbo2::`kv_cache_append_turbo2` | [metal/common/kv_cache_append_turbo2.metal:53][f326] | cache write | metal | Qwen-GDN (7 ckpts) | [1 note][t326] | not measured |
| kv_cache_append_turbo3::`kv_cache_append_turbo3` | [metal/common/kv_cache_append_turbo3.metal:65][f327] | cache write | metal | Qwen-GDN (7 ckpts) | [1 note][t327] | not measured |
| kv_cache_append_turbo4::`kv_cache_append_turbo4` | [metal/common/kv_cache_append_turbo4.metal:82][f328] | cache write | metal | Qwen-GDN (7 ckpts) | [1 note][t328] | not measured |
| kv_cache_append_turbo8::`kv_cache_append_turbo8` | [metal/common/kv_cache_append_turbo8.metal:47][f329] | cache write | metal | Qwen-GDN (7 ckpts) | [1 note][t329] | not measured |
| wht_bf16::`wht_bf16_{inplace, inplace_inv}` (2) | [metal/common/wht_bf16.metal:142][f346] | TurboQuant rotation | metal | Qwen-GDN (7 ckpts) | [1 note][t346] | not measured |

### Quantization and format conversion

52 entry points: 52 primary here (full rows), 0 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| gemv_fp8w::`quantize_bf16_to_fp8` | [gb10/common/dense_gemv_fp8w.cu:65][f21] | activation quantize | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t21] | not measured |
| dequant_fp8_blockscaled_bf16::`dequant_fp8_blockscaled_bf16` | [gb10/common/dequant_fp8_blockscaled_bf16.cu:89][f23] | dequant / repack / transpose | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [3 notes][t23] | not measured |
| dequant_gguf_bf16::`dequant_{q2_0_gn_to_bf16, q2_k_to_bf16, q3_k_to_bf16, q4_k_to_bf16, q6_k_to_bf16, q8_0_to_bf16}` (6) | [gb10/common/dequant_gguf_bf16.cu:43][f24] | dequant / repack / transpose | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t24] | not measured |
| dequant_nvfp4_bf16::`dequant_nvfp4_to_bf16` | [gb10/common/dequant_nvfp4_bf16.cu:50][f25] | dequant / repack / transpose | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t25] | not measured |
| fp8_scale_transpose::`fp8_act_scale_to_kmajor` | [gb10/common/fp8_scale_transpose.cu:35][f33] | dequant / repack / transpose | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t33] | not measured |
| moe_nvfp4_grouped_tc::`nvfp4_tc_lean_repack` | [gb10/common/moe_nvfp4_grouped_tc.cu:227][f86] | dequant / repack / transpose | b200 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | — | not measured |
| moe_silu_mul::`silu_mul_quant_fp8` | [gb10/common/moe_silu_mul.cu:107][f106] | activation quantize | b200 b300 gb10 hop strix hip | GLM-5.3, Gemma4, Kimi-K3, Laguna, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN (19 ckpts) | [1 note][t106] | [21–24%][m106.silu_mul_quant_fp8] (prefill 32k (cold, 32772 tok)) |
| per_token_group_quant_fp8::`per_token_group_quant_fp8` | [gb10/common/per_token_group_quant_fp8.cu:39][f152] | activation quantize | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [2 notes][t152] | [81–83%][m152.per_token_group_quant_fp8] (prefill 32k (cold, 32772 tok)) |
| quant_rowwise_fp8::`quant_rowwise_fp8` | [gb10/common/quant_rowwise_fp8.cu:38][f159] | activation quantize | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t159] | not measured |
| quantize_bf16_to_fp8_blockscaled::`quantize_bf16_to_fp8_blockscaled` | [gb10/common/quantize_bf16_to_fp8_blockscaled.cu:54][f160] | activation quantize | b200 b300 gb10 hop | LongCat (1 ckpts) | [1 note][t160] | not measured |
| quantize_nvfp4::`f32_to_bf16_trunc` | [gb10/common/quantize_bf16_to_nvfp4.cu:29][f161] | dtype conversion | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t161] | not measured |
| quantize_nvfp4::`nvfp4_global_absmax`, `quantize_bf16_to_nvfp4`, `quantize_bf16_to_nvfp4_mse` | [gb10/common/quantize_bf16_to_nvfp4.cu:133][f161] | activation quantize | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [3 notes][t161] · [#34][pr34] | not measured |
| transpose_u8::`transpose_u8` | [gb10/common/transpose_u8.cu:15][f180] | dequant / repack / transpose | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | — | not measured |
| w4a16_fp8_ldmab::`fp8_predequant_nvfp4_t` | [gb10/common/w4a16_fp8_ldmab.cu:193][f182] | dequant / repack / transpose | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [2 notes][t182] | not measured |
| w8a16_gemm_t::`transpose_{block_scale, fp8}` (2) | [gb10/common/w8a16_gemm_t.cu:607][f193] | dequant / repack / transpose | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [1 note][t193] | not measured |
| w8a8_act_quant::`w8a8_act_quant_{g128, row, silu_g128, silu_row}` (4) | [gb10/common/w8a8_act_quant.cu:168][f199] | activation quantize | b200 gb10 hop | all 14 decoder families (30 ckpts) | — | not measured |
| widen_block_scale_f32::`widen_block_scale_f32` | [gb10/common/widen_block_scale_f32.cu:21][f202] | dequant / repack / transpose | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t202] | not measured |
| kquant_moe::`kquant_q8_1_rows_bf16`, `kquant_swiglu_q8_1_rows_bf16`, `metrale_q8_1_quantize_d2s6_bf16`, `metrale_q8_1_quantize_d4_bf16` | [gb10/deepseek-v4-flash/nvfp4/kquant_moe.cu:82][f210] | activation quantize | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [4 notes][t210] | not measured |
| w4a16::`bf16_to_fp8` | [gb10/deepseek-v4-flash/nvfp4/w4a16_gemm.cu:540][f225] | activation quantize | b200 gb10 hop | DeepSeek-V4, Gemma4, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE, Qwen3-VL, Step-3.7 (27 ckpts) | [1 note][t225] | not measured |
| w4a16::`predequant_nvfp4_to_fp8` | [gb10/deepseek-v4-flash/nvfp4/w4a16_gemm.cu:501][f225] | dequant / repack / transpose | b200 gb10 hop | DeepSeek-V4, Gemma4, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE, Qwen3-VL, Step-3.7 (27 ckpts) | [1 note][t225] | not measured |
| w4a16::`bf16_to_fp8` | [gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/w4a16_gemm.cu:623][f261] | activation quantize | gb10 | Nemotron-H (3 ckpts) | [1 note][t261] | not measured |
| w4a16::`predequant_nvfp4_to_fp8` | [gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/w4a16_gemm.cu:584][f261] | dequant / repack / transpose | gb10 | Nemotron-H (3 ckpts) | [1 note][t261] | not measured |
| nvfp4_mmq::`metrale_nvfp4_quantize_bf16` | [gb10/qwen3.6-27b/nvfp4/nvfp4_mmq.cu:132][f272] | activation quantize | gb10 hop | Qwen-GDN (7 ckpts) | [3 notes][t272] | [28–54%][m272.metrale_nvfp4_quantize_bf16] (decode C=16 (R=32, MTP k=1)) |
| nvfp4_mmq::`metrale_nvfp4_repack` | [gb10/qwen3.6-27b/nvfp4/nvfp4_mmq.cu:142][f272] | dequant / repack / transpose | gb10 hop | Qwen-GDN (7 ckpts) | [3 notes][t272] | not measured |
| nvfp4_mmq::`metrale_nvfp4_scale_bf16` | [gb10/qwen3.6-27b/nvfp4/nvfp4_mmq.cu:214][f272] | dequant / repack / transpose | gb10 hop | Qwen-GDN (7 ckpts) | [3 notes][t272] | [99–103%][m272.metrale_nvfp4_scale_bf16] (prefill 32k (cold, 32772 tok)) |
| q4k_mmq::`metrale_q8_1_quantize_ds4_bf16` | [gb10/qwen3.6-27b/nvfp4/q4k_mmq.cu:68][f274] | activation quantize | gb10 hop | Qwen-GDN (7 ckpts) | [1 note][t274] | not measured |
| q4k_quantize::`q4k_quantize` | [gb10/qwen3.6-27b/nvfp4/q4k_quantize.cu:81][f275] | activation quantize | gb10 hop | Qwen-GDN (7 ckpts) | [1 note][t275] | not measured |
| w4a16::`bf16_to_fp8` | [gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu:974][f276] | activation quantize | gb10 hop strix | Qwen-GDN (7 ckpts) | [2 notes][t276] | not measured |
| w4a16::`predequant_nvfp4_to_fp8` | [gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu:932][f276] | dequant / repack / transpose | gb10 hop strix | Qwen-GDN (7 ckpts) | [2 notes][t276] | not measured |
| w4a16::`bf16_to_fp8` | [gb10/qwen3.6-35b-a3b/nvfp4/w4a16_gemm.cu:784][f284] | activation quantize | b200 gb10 hop strix | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [2 notes][t284] · [#34][pr34] | not measured |
| w4a16::`predequant_nvfp4_to_fp8` | [gb10/qwen3.6-35b-a3b/nvfp4/w4a16_gemm.cu:742][f284] | dequant / repack / transpose | b200 gb10 hop strix | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [2 notes][t284] · [#34][pr34] | not measured |
| fp8_act_quant_hopper::`per_token_group_quant_fp8_hopper` | [hopper/common/fp8_act_quant_hopper.cu:94][f291] | activation quantize | hop | DeepSeek-V4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE (18 ckpts) | [2 notes][t291] | not measured |
| w8a16_gemm_t::`transpose_{block_scale, fp8}` (2) | [strix-hip/common/w8a16_gemm_t.cu:198][f355] | dequant / repack / transpose | hip | Qwen-GDN, Qwen-GDN-MoE (13 ckpts) | [3 notes][t355] | not measured |
| w4a16::`bf16_to_fp8` | [strix-hip/qwen3.6-27b/nvfp4/w4a16_gemm.cu:451][f357] | activation quantize | hip | Qwen-GDN (7 ckpts) | [1 note][t357] | not measured |
| w4a16::`predequant_nvfp4_to_fp8` | [strix-hip/qwen3.6-27b/nvfp4/w4a16_gemm.cu:419][f357] | dequant / repack / transpose | hip | Qwen-GDN (7 ckpts) | [1 note][t357] | not measured |
| w4a16::`bf16_to_fp8` | [strix-hip/qwen3.6-35b-a3b/nvfp4/w4a16_gemm.cu:450][f358] | activation quantize | hip | Qwen-GDN-MoE (6 ckpts) | [2 notes][t358] | not measured |
| w4a16::`predequant_nvfp4_to_fp8` | [strix-hip/qwen3.6-35b-a3b/nvfp4/w4a16_gemm.cu:418][f358] | dequant / repack / transpose | hip | Qwen-GDN-MoE (6 ckpts) | [2 notes][t358] | not measured |

### Embedding and LM head (lookup, overlays, softcap, scale)

18 entry points: 11 primary here (full rows), 7 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| embed_from_argmax::`batched_embed`, `embed_from_argmax` | [gb10/common/embed_from_argmax.cu:17][f29] | embedding / LM head | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t29] | not measured |
| embed_from_argmax::`batched_embed_fp8` | [gb10/common/embed_from_argmax.cu:110][f29] | embedding / LM head | b200 b300 gb10 hop strix hip | LongCat (1 ckpts) | [2 notes][t29] | not measured |
| token_overlay::`embed_overlay_routed_bf16`, `embed_rowdiff_bf16`, `lmhead_overlay_routed_bf16`, `lmhead_overlay_routed_f32` | [gb10/common/token_overlay.cu:20][f178] | embedding / LM head | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [2 notes][t178] | not measured |
| embed_scale::`bf16_scale_inplace` | [gb10/gemma-4-26b-a4b/nvfp4/embed_scale.cu:12][f227] | embedding / LM head | gb10 | Gemma4 (2 ckpts) | — | not measured |
| logit_softcap::`logit_softcap_bf16` | [gb10/gemma-4-26b-a4b/nvfp4/logit_softcap.cu:12][f230] | embedding / LM head | gb10 | Gemma4 (2 ckpts) | [1 note][t230] | not measured |
| embed_scale::`bf16_scale_inplace` | [gb10/gemma-4-31b/nvfp4/embed_scale.cu:12][f239] | embedding / LM head | gb10 | Gemma4 (2 ckpts) | [1 note][t239] | not measured |
| logit_softcap::`logit_softcap_bf16` | [gb10/gemma-4-31b/nvfp4/logit_softcap.cu:12][f240] | embedding / LM head | gb10 | Gemma4 (2 ckpts) | [1 note][t240] | not measured |

Also launched here: [Projection GEMM/GEMV — FP8](#projection-gemm-gemv-fp8-w8a16-w8a8-block-scaled): `fp8_gemv_rowscale_{batch16_rt2, batch8_rt2}` (2); [Projection GEMM/GEMV — integer / K-quant](#projection-gemm-gemv-integer-k-quant-q2-0-q2-k-q6-k-int8-mlx-int8): `kquant_mmvq_q6_k_w`; [Quantization and format conversion](#quantization-and-format-conversion): `quantize_bf16_to_fp8`, `nvfp4_global_absmax`, `quantize_bf16_to_nvfp4`, `kquant_q8_1_rows_bf16`.

### Sampling (argmax, top-p, feed-forward of the chosen token)

7 entry points: 7 primary here (full rows), 0 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| argmax::`argmax_bf16` | [gb10/common/argmax_bf16.cu:14][f5] | argmax / top-p | b200 b300 gb10 hop strix hip | all 14 decoder families + NLLB (31 ckpts) | [1 note][t5] | [2%][m5.argmax_bf16] (decode C=1 (R=4, MTP k=3)) |
| argmax::`argmax_{bf16_batch, bf16_batch_lp, fp32}` (3) | [gb10/common/argmax_bf16.cu:68][f5] | argmax / top-p | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [3 notes][t5] | not measured |
| argmax_feed::`argmax_bf16_batch_feed`, `feed_resolve` | [gb10/common/argmax_feed.cu:44][f6] | argmax / top-p | b200 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t6] | not measured |
| argmax_bf16::`argmax_bf16` | [metal/common/argmax_bf16.metal:22][f305] | argmax / top-p | metal | Qwen-GDN (7 ckpts) | [1 note][t305] | not measured |

### Speculative decoding (MTP heads, DFlash drafter, verify helpers)

97 entry points: 3 primary here (full rows), 94 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| dflash2::`dflash2_{conv2, selector_walk, topk16}` (3) | [gb10/common/dflash2.cu:31][f26] | DFlash drafter | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [3 notes][t26] | not measured |

Also launched here: [Activations and elementwise](#activations-and-elementwise-silu-gelu-relu-residual-gates-scale): `bf16_concat`, `bf16_residual_add`, `sigmoid_gate_{mul, mul_batched}` (2); [Attention](#attention-gqa-mha-paged-decode-split-k-prefill-flash): `attn_prefill_h128`, `paged_decode_attn`, `paged_decode_attn_fp8`, `attn_prefill_{paged, paged_fp8}` (2), `attn_prefill_paged_indirect`, `attn_prefill_h128`; [Embedding and LM head](#embedding-and-lm-head-lookup-overlays-softcap-scale): `batched_embed`, `embed_from_argmax`; [GDN](#gdn-gated-delta-rule-linear-attention): `deinterleave_qg`; [Hyper-connections](#hyper-connections-mhc): `hc_expand`, `hc_head`, `hc_expand`, `hc_head`; [KV cache](#kv-cache-write-quantize-turboquant-rotation-slot-metadata): `fill_slots_from_block_table`, `reshape_and_cache_flash`, `reshape_and_cache_flash_fp8`; [MoE](#moe-routing-dispatch-expert-gemm-gemv-combine): `moe_expert_gemv`, `moe_weighted_sum_blend`, `moe_silu_mul`, `moe_topk_softmax`, `moe_silu_mul`, `moe_silu_mul`; [Normalization](#normalization-rmsnorm-layernorm-l2-gated-norms): `residual_add_rms_norm`, `rms_{norm, norm_strided}` (2), `rms_norm_residual`, `rms_norm_vanilla`, `residual_add_rms_norm`, `rms_norm`, `rms_norm_residual`, `rms_norm_strided`, `residual_add_rms_norm`, `rms_norm`, `rms_norm_residual`, `rms_norm_strided`, `residual_add_rms_norm`, `rms_norm`, `rms_norm_residual`, `rms_norm_strided`, `residual_add_rms_norm`, `rms_norm`, `rms_norm_residual`, `rms_norm_strided`, `residual_add_rms_norm`, `rms_norm`, `rms_norm_residual`, `rms_norm_strided`; [Positional encoding](#positional-encoding-rope-yarn-mrope): `rope_{forward, forward_strided, forward_yarn}` (3), `rope_{forward, forward_yarn}` (2); [Projection GEMM/GEMV — BF16/F32](#projection-gemm-gemv-bf16-f32): `dense_gemm_bf16`, `dense_gemm_bf16_pipelined`, `dense_gemv_bf16`, `dense_gemv_bf16_batchm`, `dense_gemm_bf16`, `dense_gemm_{bf16, bf16_pipelined}` (2); [Projection GEMM/GEMV — FP8](#projection-gemm-gemv-fp8-w8a16-w8a8-block-scaled): `dense_gemv_fp8w`, `fp8_gemv_rowscale_{batch16_rt2, batch8_rt2}` (2), `w8a16_gemv`, `fp8_gemm_t_row_{scaled, scaled_k64, scaled_m16, scaled_p4}` (4), `fp8_gemm_t_row_{scaled, scaled_m16}` (2), `w8a16_gemv`; [Projection GEMM/GEMV — NVFP4 W4A16](#projection-gemm-gemv-nvfp4-w4a16): `w4a16_gemm`, `w4a16_{gemv, gemv_batch16, gemv_batch32, gemv_qg}` (4), `w4a16_gemv_sw`, `w4a16_gemv_dual`, `w4a16_gemm`, `w4a16_gemm`, `w4a16_gemm`, `w4a16_gemm`, `w4a16_gemm`, `w4a16_gemm`; [Quantization and format conversion](#quantization-and-format-conversion): `quantize_bf16_to_fp8`, `nvfp4_global_absmax`, `quantize_bf16_to_nvfp4`; [Sampling](#sampling-argmax-top-p-feed-forward-of-the-chosen-token): `argmax_bf16`, `argmax_bf16_{batch, batch_lp}` (2).

### Hyper-connections (mHC)

27 entry points: 27 primary here (full rows), 0 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| glm5next_mhc::`glm5next_hc_{expand, finish, head, mix, mix_bf16, post, pre}` (7) | [gb10/common/glm5next_mhc.cu:64][f57] | hyper-connection mix | b200 b300 gb10 hop | GLM-5.3 (1 ckpts) | [4 notes][t57] | not measured |
| hc_v41::`hc_v41_{collapse, collapse_wide, finish_collapse, mixes_dot, mixes_finish, post_wide}` (6) | [gb10/deepseek-v4-flash/nvfp4/hc_v41.cu:118][f208] | hyper-connection mix | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [3 notes][t208] | not measured |
| hyper_connection::`hc_expand`, `hc_head`, `hc_post`, `hc_pre` | [gb10/deepseek-v4-flash/nvfp4/hyper_connection.cu:38][f209] | hyper-connection mix | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [2 notes][t209] | not measured |
| hyper_connection::`hc_expand`, `hc_head`, `hc_post`, `hc_pre`, `hc_pre_down`, `hc_pre_finish`, `hc_pre_mix`, `hc_pre_stage`, `hc_pre_stage_bf16`, `hc_silu_scale` | [gb10/qwen3.8-flash-next/nvfp4/hyper_connection.cu:96][f286] | hyper-connection mix | gb10 | Qwen3.8-FN (1 ckpts) | [2 notes][t286] | not measured |

### N-gram and memory embeddings (Engram, PLE, n-gram tables)

16 entry points: 5 primary here (full rows), 11 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| engram_v41::`engram_v41_{gate, wkv_q2k_gemv}` (2) | [gb10/deepseek-v4-flash/nvfp4/engram_v41.cu:43][f206] | memory embedding | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [3 notes][t206] | not measured |
| ple::`ple_{add_highway, conv, gate}` (3) | [gb10/qwen3.8-flash-next/nvfp4/ple.cu:90][f287] | memory embedding | gb10 | Qwen3.8-FN (1 ckpts) | [2 notes][t287] | not measured |

Also launched here: [Activations and elementwise](#activations-and-elementwise-silu-gelu-relu-residual-gates-scale): `bf16_scaled_add`; [Embedding and LM head](#embedding-and-lm-head-lookup-overlays-softcap-scale): `batched_embed`, `embed_from_argmax`, `batched_embed_fp8`; [Projection GEMM/GEMV — BF16/F32](#projection-gemm-gemv-bf16-f32): `dense_gemm_bf16`, `dense_gemm_bf16_pipelined`, `dense_gemv_bf16`, `dense_gemm_{bf16, bf16_pipelined}` (2); [Quantization and format conversion](#quantization-and-format-conversion): `quantize_bf16_to_fp8`, `dequant_q2_k_to_bf16`.

### Vision encoder (ViT towers)

36 entry points: 32 primary here (full rows), 4 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| glm_vit::`glm_vit_{add_bias, add_inplace, copy, f32_to_bf16, gelu_erf, im2col_2x2, layernorm, qknorm_rope_deint, rmsnorm, scatter_head, softmax_rows, swiglu_clamp}` (12) | [gb10/glm-5.3-flash/nvfp4/glm_vit.cu:42][f244] | ViT op | gb10 | GLM-5.3 (1 ckpts) | [2 notes][t244] | not measured |
| vision_encoder::`vision_{add_inplace, attention_rope, bf16_copy, f32_to_bf16, gelu, gemm_bias, layer_norm, spatial_merge}` (8) | [gb10/qwen3-vl-30b-a3b/nvfp4/vision_encoder.cu:23][f265] | ViT op | gb10 | Qwen-GDN-MoE, Qwen3-VL (7 ckpts) | [2 notes][t265] | not measured |
| vision_encoder::`vision_add_bias`, `vision_add_inplace`, `vision_attention_rope`, `vision_bf16_copy`, `vision_f32_to_bf16`, `vision_gelu`, `vision_gemm_bias`, `vision_layer_norm`, `vision_spatial_merge`, `vit_rope_deinterleave`, `vit_scatter_head`, `vit_softmax_rows` | [gb10/qwen3.6-35b-a3b/nvfp4/vision_encoder.cu:23][f283] | ViT op | b200 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [3 notes][t283] | not measured |

Also launched here: [Projection GEMM/GEMV — BF16/F32](#projection-gemm-gemv-bf16-f32): `dense_gemm_bf16_f32out`, `dense_gemm_bf16_pipelined`, `dense_gemm_bf16_{f32out, pipelined}` (2).

### Encoder-decoder translation (NLLB, self-contained kernel set)

29 entry points: 26 primary here (full rows), 3 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| nllb_encoder::`nllb_{add_bf16, add_row_bf16, attn_bdecode, attn_kv_bf16, beam_topk, bias_bf16, embed_bf16, gather_batched, gemv_bf16, layernorm_bf16, layernorm_oop_bf16, relu_bf16, scale_bf16, scatter_batched}` (14) | [gb10/common/nllb_encoder.cu:203][f118] | NLLB encoder/decoder op | b200 b300 gb10 hop | NLLB (1 ckpts) | [3 notes][t118] | not measured |
| nllb_encoder::`nllb_{add_bf16, add_row_bf16, attn_bdecode, attn_kv_bf16, bias_bf16, embed_bf16, gather_batched, gemv_bf16, layernorm_bf16, relu_bf16, scale_bf16, scatter_batched}` (12) | [metal/common/nllb_encoder.metal:281][f337] | NLLB encoder/decoder op | metal | NLLB (1 ckpts) | [1 note][t337] | not measured |

Also launched here: [Projection GEMM/GEMV — BF16/F32](#projection-gemm-gemv-bf16-f32): `dense_gemm_bf16_pipelined`, `dense_gemm_bf16_pipelined`; [Sampling](#sampling-argmax-top-p-feed-forward-of-the-chosen-token): `argmax_bf16`.

### LoRA adapters (BGMV shrink/expand)

6 entry points: 6 primary here (full rows), 0 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| lora_bgmv::`lora_bgmv_{expand_fold, shrink}` (2) | [gb10/common/lora_bgmv.cu:50][f66] | BGMV shrink/expand | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t66] | not measured |
| moe_lora_gather_bgmv::`moe_lora_gather_bgmv_{expand_fold, shrink}` (2) | [gb10/common/moe_lora_gather_bgmv.cu:57][f83] | BGMV shrink/expand | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t83] | not measured |
| moe_lora_grouped_down::`moe_lora_grouped_down_{expand_fold, shrink}` (2) | [gb10/common/moe_lora_grouped_down.cu:67][f84] | BGMV shrink/expand | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t84] | not measured |

### Weight load and repack (one-time, not per token)

76 entry points: 0 primary here (full rows), 76 of other components launched from this component's code (listed after the table; their full rows are under their primary component).

Also launched here: [Activations and elementwise](#activations-and-elementwise-silu-gelu-relu-residual-gates-scale): `bf16_add_inplace`, `projection_bias_bf16`, `bf16_residual_add`, `bf16_add`; [Attention](#attention-gqa-mha-paged-decode-split-k-prefill-flash): `paged_decode_attn_sink`; [Causal conv1d](#causal-conv1d-short-convolution-of-gdn-kda-mamba2): `k3_kda_conv_update_f32`; [GDN](#gdn-gated-delta-rule-linear-attention): `gated_delta_rule_decode`, `gated_delta_rule_decode`, `gated_delta_rule_decode`, `gated_delta_rule_decode`, `gated_delta_rule_decode`, `gated_delta_rule_decode`, `gated_delta_rule_decode`; [Hyper-connections](#hyper-connections-mhc): `hc_v41_{collapse, collapse_wide, finish_collapse, mixes_dot, mixes_finish, post_wide}` (6), `hc_expand`, `hc_post`, `hc_expand`, `hc_post`; [KDA](#kda-kimi-delta-attention-linear-attention): `k3_kda_recurrent_step_f32`; [KV cache](#kv-cache-write-quantize-turboquant-rotation-slot-metadata): `reshape_and_cache_flash`, `wht_bf16_inplace`, `wht_bf16_inplace`; [MLA](#mla-multi-head-latent-attention): `k3_mla_{maybe_rope_f32, sdpa_gate_f32}` (2); [MoE](#moe-routing-dispatch-expert-gemm-gemv-combine): `gpt_oss_{expert_reduce_bf16, selected_bias_bf16, swiglu_bf16}` (3), `gpt_oss_mxfp4_selected_bf16`, `moe_topk_selected_bf16_rows`; [Normalization](#normalization-rmsnorm-layernorm-l2-gated-norms): `rms_norm_vanilla`; [Positional encoding](#positional-encoding-rope-yarn-mrope): `gpt_oss_{rope_bf16, yarn_frequencies}` (2); [Projection GEMM/GEMV — BF16/F32](#projection-gemm-gemv-bf16-f32): `dense_gemm_bf16_pipelined`, `dense_gemv_bf16`, `dense_gemv_bf16_fp32out`, `dense_gemv_bf16_batchm`, `dense_gemv_bf16_batchm_fp32out`, `dense_gemv_bf16`, `dense_gemm_bf16_pipelined`; [Projection GEMM/GEMV — FP8](#projection-gemm-gemv-fp8-w8a16-w8a8-block-scaled): `w8a16_gemm`, `w8a16_gemm_pipelined`, `w8a16_gemv`, `w8a16_gemv`, `w8a16_gemm`; [Quantization and format conversion](#quantization-and-format-conversion): `dequant_fp8_blockscaled_bf16`, `dequant_{q2_0_gn_to_bf16, q2_k_to_bf16, q3_k_to_bf16, q4_k_to_bf16, q6_k_to_bf16, q8_0_to_bf16}` (6), `dequant_nvfp4_to_bf16`, `quantize_bf16_to_fp8_blockscaled`, `f32_to_bf16_trunc`, `nvfp4_global_absmax`, `quantize_bf16_to_nvfp4`, `quantize_bf16_to_nvfp4_mse`, `transpose_u8`, `widen_block_scale_f32`, `bf16_to_fp8`, `predequant_nvfp4_to_fp8`, `bf16_to_fp8`, `predequant_nvfp4_to_fp8`, `bf16_to_fp8`, `predequant_nvfp4_to_fp8`, `bf16_to_fp8`, `predequant_nvfp4_to_fp8`, `bf16_to_fp8`, `predequant_nvfp4_to_fp8`, `bf16_to_fp8`, `predequant_nvfp4_to_fp8`.

## Unique kernels by component

Entry points whose every engine call site belongs to one component.

### Unique to Attention (GQA/MHA: paged decode, split-K, prefill/flash)

180 entry points.

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| w8a16_gemv_batch4::`w8a16_gemv_{batch16_strided, batch4_strided}` (2) | [b300/common/w8a16_gemv_batch4.cu:255][f4] | FP8 GEMM/GEMV | b300 | Kimi-K3 (1 ckpts) | [1 note][t4] | not measured |
| attn_prefill::`attn_prefill` | [gb10/common/attn_prefill.cu:78][f7] | prefill (flash) | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [2 notes][t7] | not measured |
| attn_prefill_512tc::`attn_prefill_512tc` | [gb10/common/attn_prefill.cu:78][f7] | prefill (flash) | gb10 | Gemma4 (2 ckpts) | [3 notes][t7] | not measured |
| attn_prefill::`attn_prefill_64` | [gb10/common/attn_prefill.cu:562][f7] | prefill (flash) | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [2 notes][t7] | [27%][m7.attn_prefill_64] (prefill 32k (cold, 32772 tok)) |
| attn_prefill_fa128::`attn_prefill_{fa128, fa128_paged}` (2) | [gb10/common/attn_prefill_fa128.cu:356][f8] | prefill (flash) | b200 gb10 hop | all 14 decoder families (30 ckpts) | — | not measured |
| gemm_splitk::`dense_gemm_splitk_{partial, reduce}` (2) | [gb10/common/dense_gemm_splitk.cu:27][f15] | BF16/F32 GEMM/GEMV | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t15] | not measured |
| fused_k_norm_rope_cache::`fused_k_norm_rope_{cache_write_bf16, mrope_cache_write_bf16}` (2) | [gb10/common/fused_k_norm_rope_cache.cu:53][f34] | cache write | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t34] | not measured |
| paged_decode_attn_bf16_gqa::`paged_decode_attn_bf16_gqa` | [gb10/common/paged_decode_attn_bf16_gqa.cu:46][f120] | paged decode | b200 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t120] | not measured |
| paged_decode_bf16k_turbo2v::`paged_decode_attn_bf16k_turbo2v` | [gb10/common/paged_decode_attn_bf16k_turbo2v.cu:85][f121] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t121] | not measured |
| paged_decode_bf16k_turbo2v_128::`paged_decode_attn_bf16k_turbo2v` | [gb10/common/paged_decode_attn_bf16k_turbo2v_128.cu:85][f122] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t122] | not measured |
| paged_decode_bf16k_turbo3v::`paged_decode_attn_bf16k_turbo3v` | [gb10/common/paged_decode_attn_bf16k_turbo3v.cu:91][f123] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t123] | not measured |
| paged_decode_bf16k_turbo3v_128::`paged_decode_attn_bf16k_turbo3v` | [gb10/common/paged_decode_attn_bf16k_turbo3v_128.cu:91][f124] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t124] | not measured |
| paged_decode_bf16k_turbo4v::`paged_decode_attn_bf16k_turbo4v` | [gb10/common/paged_decode_attn_bf16k_turbo4v.cu:84][f125] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t125] | not measured |
| paged_decode_bf16k_turbo4v_128::`paged_decode_attn_bf16k_turbo4v` | [gb10/common/paged_decode_attn_bf16k_turbo4v_128.cu:84][f126] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t126] | not measured |
| paged_decode_fp8::`paged_decode_attn_{reduce_fp8, splitk_fp8}` (2) | [gb10/common/paged_decode_attn_fp8.cu:346][f127] | paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [4 notes][t127] | not measured |
| paged_decode_attn_fp8_gqa::`paged_decode_attn_fp8_gqa` | [gb10/common/paged_decode_attn_fp8_gqa.cu:112][f128] | paged decode | b200 gb10 hop | all 14 decoder families (30 ckpts) | [2 notes][t128] | not measured |
| paged_decode_fp8k_turbo2v::`paged_decode_attn_fp8k_turbo2v` | [gb10/common/paged_decode_attn_fp8k_turbo2v.cu:101][f129] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t129] | not measured |
| paged_decode_fp8k_turbo2v_128::`paged_decode_attn_fp8k_turbo2v` | [gb10/common/paged_decode_attn_fp8k_turbo2v_128.cu:101][f130] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t130] | not measured |
| paged_decode_fp8k_turbo3v::`paged_decode_attn_fp8k_turbo3v` | [gb10/common/paged_decode_attn_fp8k_turbo3v.cu:112][f131] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t131] | not measured |
| paged_decode_fp8k_turbo3v_128::`paged_decode_attn_fp8k_turbo3v` | [gb10/common/paged_decode_attn_fp8k_turbo3v_128.cu:112][f132] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t132] | not measured |
| paged_decode_fp8k_turbo4v::`paged_decode_attn_fp8k_turbo4v` | [gb10/common/paged_decode_attn_fp8k_turbo4v.cu:99][f133] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t133] | not measured |
| paged_decode_fp8k_turbo4v_128::`paged_decode_attn_fp8k_turbo4v` | [gb10/common/paged_decode_attn_fp8k_turbo4v_128.cu:99][f134] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t134] | not measured |
| paged_decode_nvfp4::`paged_decode_attn_{nvfp4, reduce_nvfp4, splitk_nvfp4}` (3) | [gb10/common/paged_decode_attn_nvfp4.cu:87][f135] | paged decode | b200 b300 gb10 hop strix hip | GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (28 ckpts) | [1 note][t135] | not measured |
| paged_decode_turbo2::`paged_decode_attn_turbo2` | [gb10/common/paged_decode_attn_turbo2.cu:80][f136] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [2 notes][t136] | not measured |
| paged_decode_attn_turbo2_128::`paged_decode_attn_turbo2` | [gb10/common/paged_decode_attn_turbo2_128.cu:80][f137] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t137] | not measured |
| paged_decode_attn_turbo3::`paged_decode_attn_{splitk_nvfp4, turbo3}` (2) | [gb10/common/paged_decode_attn_turbo3.cu:110][f138] | paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [3 notes][t138] | not measured |
| paged_decode_attn_turbo3_128::`paged_decode_attn_{splitk_nvfp4, turbo3}` (2) | [gb10/common/paged_decode_attn_turbo3_128.cu:110][f139] | paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t139] | not measured |
| paged_decode_turbo3k_turbo8v::`paged_decode_attn_turbo3k_turbo8v` | [gb10/common/paged_decode_attn_turbo3k_turbo8v.cu:119][f140] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t140] | not measured |
| paged_decode_turbo3k_turbo8v_128::`paged_decode_attn_turbo3k_turbo8v` | [gb10/common/paged_decode_attn_turbo3k_turbo8v_128.cu:119][f141] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t141] | not measured |
| paged_decode_attn_turbo4::`paged_decode_attn_{splitk_nvfp4, turbo4}` (2) | [gb10/common/paged_decode_attn_turbo4.cu:95][f142] | paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t142] | not measured |
| paged_decode_attn_turbo4_128::`paged_decode_attn_{splitk_nvfp4, turbo4}` (2) | [gb10/common/paged_decode_attn_turbo4_128.cu:95][f143] | paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t143] | not measured |
| paged_decode_attn_turbo4_512::`paged_decode_attn_{splitk_nvfp4, turbo4}` (2) | [gb10/common/paged_decode_attn_turbo4_512.cu:119][f144] | paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [2 notes][t144] | not measured |
| paged_decode_turbo4k_turbo3v::`paged_decode_attn_turbo4k_turbo3v` | [gb10/common/paged_decode_attn_turbo4k_turbo3v.cu:122][f145] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t145] | not measured |
| paged_decode_turbo4k_turbo3v_128::`paged_decode_attn_turbo4k_turbo3v` | [gb10/common/paged_decode_attn_turbo4k_turbo3v_128.cu:122][f146] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t146] | not measured |
| paged_decode_turbo4k_turbo8v::`paged_decode_attn_turbo4k_turbo8v` | [gb10/common/paged_decode_attn_turbo4k_turbo8v.cu:114][f147] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t147] | not measured |
| paged_decode_turbo4k_turbo8v_128::`paged_decode_attn_turbo4k_turbo8v` | [gb10/common/paged_decode_attn_turbo4k_turbo8v_128.cu:114][f148] | paged decode | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t148] | not measured |
| paged_decode_attn_turbo8::`paged_decode_attn_{splitk_nvfp4, turbo8}` (2) | [gb10/common/paged_decode_attn_turbo8.cu:100][f149] | paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t149] | not measured |
| paged_decode_attn_turbo8_128::`paged_decode_attn_{splitk_nvfp4, turbo8}` (2) | [gb10/common/paged_decode_attn_turbo8_128.cu:98][f150] | paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [2 notes][t150] | not measured |
| paged_decode_attn_turbo8_512::`paged_decode_attn_{splitk_nvfp4, turbo8}` (2) | [gb10/common/paged_decode_attn_turbo8_512.cu:125][f151] | paged decode | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [2 notes][t151] | not measured |
| prefill_paged::`attn_prefill_paged_64` | [gb10/common/prefill_paged_compute.cuh:644][f153] | prefill (flash) | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [6 notes][t153] | [23–24%][m153.attn_prefill_paged_64] (prefill 32k (cold, 32772 tok)) |
| attn_prefill_paged_batched::`attn_prefill_paged_{batched, batched_64, fp8_64, fp8_batched, fp8_batched_64, nvfp4, nvfp4_64, nvfp4_batched, nvfp4_batched_64}` (9) | [gb10/common/prefill_paged_compute.cuh:162][f153] | prefill (flash) | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [5 notes][t153] | not measured |
| prefill_paged_turbo2::`attn_prefill_paged_{turbo2, turbo3_64, turbo4, turbo4_64, turbo8_64}` (5) | [gb10/common/prefill_paged_compute.cuh:162][f153] | prefill (flash) | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [5 notes][t153] | not measured |
| attn_prefill_paged_512::`attn_prefill_paged_512` | [gb10/common/prefill_paged_compute_512.cuh:83][f154] | prefill (flash) | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [3 notes][t154] | not measured |
| prefill_paged_bf16k_turbo2v::`attn_prefill_paged_{bf16k_turbo2v_64, bf16k_turbo3v_64, bf16k_turbo4v_64, fp8k_turbo2v_64, fp8k_turbo3v_64, fp8k_turbo4v_64, turbo3k_turbo8v_64, turbo4k_turbo3v_64, turbo4k_turbo8v_64}` (9) | [gb10/common/prefill_paged_compute_asym.cuh:455][f155] | prefill (flash) | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [2 notes][t155] | not measured |
| reshape_and_cache::`reshape_and_cache_flash_{nvfp4, v_only}` (2) | [gb10/common/reshape_and_cache.cu:30][f163] | cache write | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [2 notes][t163] | not measured |
| reshape_and_cache_fused_k_fp8::`fused_k_norm_rope_cache_write_fp8_kv` | [gb10/common/reshape_and_cache_fused_k_fp8.cu:131][f164] | cache write | b200 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t164] | not measured |
| reshape_and_cache_turbo::`reshape_and_cache_flash_{bf16k_turbo2v, bf16k_turbo3v, bf16k_turbo4v, fp8k_turbo2v, fp8k_turbo3v, fp8k_turbo4v, turbo2, turbo3, turbo3k_turbo8v, turbo4, turbo4k_turbo3v, turbo4k_turbo8v, turbo8}` (13) | [gb10/common/reshape_and_cache_turbo.cu:179][f165] | cache write | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [2 notes][t165] | not measured |
| residual_add::`sigmoid_gate_mul_head_broadcast`, `softplus_gate_mul_head_broadcast` | [gb10/common/residual_add.cu:171][f166] | activation / gate / residual | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t166] | not measured |
| norm::`residual_add_rms_norm_vanilla`, `rms_norm_residual_vanilla` | [gb10/common/rms_norm.cu:335][f168] | normalization | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Kimi-K3, Laguna, Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (20 ckpts) | [2 notes][t168] | not measured |
| rms_norm_vanilla::`rms_norm_vanilla_warp_row` | [gb10/common/rms_norm_vanilla.cu:120][f170] | normalization | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t170] | not measured |
| rope::`rope_forward_{proportional, yarn_interleaved, yarn_interleaved_inv, yarn_scaled}` (4) | [gb10/common/rope.cu:265][f171] | rotary | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (29 ckpts) | [2 notes][t171] | not measured |
| rope_mrope_interleaved::`rope_forward_mrope_interleaved_k_only` | [gb10/common/rope_mrope_interleaved.cu:108][f172] | rotary | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t172] | not measured |
| ssm_preprocess::`deinterleave_qg_{split, split_qnorm_mrope}` (2) | [gb10/common/ssm_preprocess.cu:137][f176] | GDN pre/post-processing | b200 b300 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | — | not measured |
| ssm_preprocess::`deinterleave_qg_split_qnorm` | [gb10/common/ssm_preprocess.cu:190][f176] | GDN pre/post-processing | b200 b300 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | — | [88–90%][m176.deinterleave_qg_split_qnorm] (prefill 4k (cold, 4103 tok)) |
| tq_plus_innerq_apply::`tq_plus_innerq_apply_{k, q}` (2) | [gb10/common/tq_plus_innerq_apply.cu:71][f179] | TurboQuant rotation | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t179] | not measured |
| w8a16_gemm_t::`transpose_{block_scale, fp8}` (2) | [gb10/common/w8a16_gemm_t.cu:607][f193] | dequant / repack / transpose | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [1 note][t193] | not measured |
| w8a16_gemv_batch4::`w8a16_gemv_{batch16_strided, batch4_strided}` (2) | [gb10/common/w8a16_gemv_batch4.cu:276][f196] | FP8 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t196] | not measured |
| attn_prefill_512::`attn_prefill_512` | [gb10/deepseek-v4-flash/nvfp4/attn_prefill_512.cu:13][f203] | prefill (flash) | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [1 note][t203] | not measured |
| csa_compress::`csa_compress` | [gb10/deepseek-v4-flash/nvfp4/csa_compress.cu:20][f205] | CSA/HCA compressed attention | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [1 note][t205] | not measured |
| grouped_gemm_mla::`grouped_gemm_mla` | [gb10/deepseek-v4-flash/nvfp4/grouped_gemm_mla.cu:35][f207] | MLA decode/prefill | b200 gb10 hop | DeepSeek-V4, LongCat, Mistral4 (4 ckpts) | [2 notes][t207] | not measured |
| mla_absorbed::`mla_{batched_gemv, cache_assemble, cache_assemble_batched, kv_assemble_batched, q_final_assemble_batched, q_rope_extract_batched, q_rope_scatter, q_rope_writeback, q_rope_writeback_batched}` (9) | [gb10/deepseek-v4-flash/nvfp4/mla_absorbed.cu:33][f211] | MLA decode/prefill | b200 gb10 hop | DeepSeek-V4, LongCat (3 ckpts) | [1 note][t211] | not measured |
| mla_fused_prefill::`mla_fused_prefill` | [gb10/deepseek-v4-flash/nvfp4/mla_fused_prefill.cu:21][f213] | MLA decode/prefill | b200 gb10 hop | DeepSeek-V4, LongCat (3 ckpts) | [1 note][t213] | not measured |
| mla_paged_decode::`mla_paged_decode_nvfp4` | [gb10/deepseek-v4-flash/nvfp4/mla_paged_decode.cu:78][f214] | MLA decode/prefill | b200 gb10 hop | DeepSeek-V4, LongCat (3 ckpts) | [1 note][t214] | not measured |
| mla_paged_decode_fp8::`mla_paged_decode_fp8` | [gb10/deepseek-v4-flash/nvfp4/mla_paged_decode_fp8.cu:38][f215] | MLA decode/prefill | b200 gb10 hop | DeepSeek-V4, LongCat (3 ckpts) | [1 note][t215] | not measured |
| mla_prefill_attn::`mla_prefill_attn_320` | [gb10/deepseek-v4-flash/nvfp4/mla_prefill_attn.cu:26][f216] | MLA decode/prefill | b200 gb10 hop | DeepSeek-V4, LongCat (3 ckpts) | [1 note][t216] | not measured |
| paged_decode_attn_512::`paged_decode_attn` | [gb10/deepseek-v4-flash/nvfp4/paged_decode_attn_512.cu:43][f220] | paged decode | b200 gb10 hop | DeepSeek-V4, Gemma4 (4 ckpts) | — | not measured |
| paged_decode_fp8_mla::`paged_decode_attn_{fp8, splitk_fp8}` (2) | [gb10/deepseek-v4-flash/nvfp4/paged_decode_attn_fp8_mla.cu:66][f221] | MLA decode/prefill | b200 gb10 hop | DeepSeek-V4, LongCat (3 ckpts) | [1 note][t221] | not measured |
| paged_decode_nvfp4::`paged_decode_attn_{nvfp4, reduce_nvfp4, splitk_nvfp4}` (3) | [gb10/deepseek-v4-flash/nvfp4/paged_decode_attn_nvfp4.cu:87][f223] | paged decode | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [1 note][t223] | not measured |
| prefill_attn_compressed::`prefill_attn_compressed` | [gb10/deepseek-v4-flash/nvfp4/prefill_attn_compressed.cu:23][f224] | CSA/HCA compressed attention | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [1 note][t224] | not measured |
| w4a16::`fp8_fp8_gemm_{t, t_m128}` (2) | [gb10/deepseek-v4-flash/nvfp4/w4a16_gemm.cu:571][f225] | FP8 GEMM/GEMV | b200 gb10 hop | DeepSeek-V4, Gemma4, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE, Qwen3-VL, Step-3.7 (27 ckpts) | [1 note][t225] | not measured |
| attn_prefill_512::`attn_prefill_512` | [gb10/gemma-4-26b-a4b/nvfp4/attn_prefill_512.cu:13][f226] | prefill (flash) | gb10 | Gemma4 (2 ckpts) | [2 notes][t226] | not measured |
| paged_decode_attn_512::`paged_decode_attn` | [gb10/gemma-4-26b-a4b/nvfp4/paged_decode_attn_512.cu:43][f235] | paged decode | gb10 | Gemma4 (2 ckpts) | — | not measured |
| paged_decode_attn_fp8_512::`paged_decode_attn_{fp8, splitk_fp8}` (2) | [gb10/gemma-4-26b-a4b/nvfp4/paged_decode_attn_fp8_512.cu:46][f236] | paged decode | gb10 | Gemma4 (2 ckpts) | [1 note][t236] | not measured |
| attn_prefill_512::`attn_prefill_512` | [gb10/gemma-4-31b/nvfp4/attn_prefill_512.cu:12][f238] | prefill (flash) | gb10 | Gemma4 (2 ckpts) | [1 note][t238] | not measured |
| mla_absorbed::`mla_{batched_gemv, cache_assemble, cache_assemble_batched, kv_assemble_batched, q_final_assemble_batched, q_rope_extract_batched, q_rope_scatter, q_rope_writeback, q_rope_writeback_batched}` (9) | [gb10/mistral-small-4/nvfp4/mla_absorbed.cu:33][f252] | MLA decode/prefill | gb10 | Mistral4 (1 ckpts) | [1 note][t252] | not measured |
| mla_fused_prefill::`mla_fused_prefill` | [gb10/mistral-small-4/nvfp4/mla_fused_prefill.cu:21][f253] | MLA decode/prefill | gb10 | Mistral4 (1 ckpts) | [1 note][t253] | not measured |
| mla_prefill_attn::`mla_prefill_attn_320` | [gb10/mistral-small-4/nvfp4/mla_prefill_attn.cu:24][f254] | MLA decode/prefill | gb10 | Mistral4 (1 ckpts) | [1 note][t254] | not measured |
| paged_decode_attn_fp8_mla::`paged_decode_attn_{fp8, splitk_fp8}` (2) | [gb10/mistral-small-4/nvfp4/paged_decode_attn_fp8_mla.cu:66][f255] | MLA decode/prefill | gb10 | Mistral4 (1 ckpts) | [1 note][t255] | not measured |
| w4a16::`fp8_fp8_gemm_{t, t_m128}` (2) | [gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/w4a16_gemm.cu:654][f261] | FP8 GEMM/GEMV | gb10 | Nemotron-H (3 ckpts) | [1 note][t261] | not measured |
| w4a16::`fp8_fp8_gemm_{t, t_m128}` (2) | [gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu:1004][f276] | FP8 GEMM/GEMV | gb10 hop strix | Qwen-GDN (7 ckpts) | [2 notes][t276] | not measured |
| w4a16::`fp8_fp8_gemm_{t, t_m128}` (2) | [gb10/qwen3.6-35b-a3b/nvfp4/w4a16_gemm.cu:814][f284] | FP8 GEMM/GEMV | b200 gb10 hop strix | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [2 notes][t284] · [#34][pr34] | not measured |
| paged_decode_bf16_splitk_hopper::`paged_decode_attn_{reduce_bf16_hopper, splitk_bf16_hopper}` (2) | [hopper/common/paged_decode_bf16_splitk_hopper.cu:30][f297] | paged decode | hop | DeepSeek-V4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE (18 ckpts) | [2 notes][t297] | not measured |
| paged_decode_fp8_splitk_hopper::`paged_decode_attn_{reduce_fp8_hopper, splitk_fp8_hopper}` (2) | [hopper/common/paged_decode_fp8_splitk_hopper.cu:47][f298] | paged decode | hop | DeepSeek-V4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE (18 ckpts) | [3 notes][t298] | not measured |
| w8a16_gemm_m16::`w8a16_gemm_m16_strided` | [hopper/common/w8a16_gemm_m16.cu:426][f300] | FP8 GEMM/GEMV | hop | DeepSeek-V4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE (18 ckpts) | [3 notes][t300] | not measured |
| w8a16_gemv_ncol::`w8a16_gemv_batch16_{ncol2, ncol2_strided, ncol4, ncol4_strided}` (4) | [hopper/common/w8a16_gemv_ncol.cu:199][f303] | FP8 GEMM/GEMV | hop | DeepSeek-V4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE (18 ckpts) | [2 notes][t303] | not measured |
| attn_prefill::`attn_{prefill, prefill_64}` (2) | [strix-hip/common/attn_prefill.cu:69][f347] | prefill (flash) | hip | Qwen-GDN, Qwen-GDN-MoE (13 ckpts) | [1 note][t347] | not measured |
| w8a16_gemm_t::`transpose_{block_scale, fp8}` (2) | [strix-hip/common/w8a16_gemm_t.cu:198][f355] | dequant / repack / transpose | hip | Qwen-GDN, Qwen-GDN-MoE (13 ckpts) | [3 notes][t355] | not measured |
| w4a16::`fp8_fp8_gemm_{t, t_m128}` (2) | [strix-hip/qwen3.6-27b/nvfp4/w4a16_gemm.cu:475][f357] | FP8 GEMM/GEMV | hip | Qwen-GDN (7 ckpts) | [1 note][t357] | not measured |
| w4a16::`fp8_fp8_gemm_{t, t_m128}` (2) | [strix-hip/qwen3.6-35b-a3b/nvfp4/w4a16_gemm.cu:474][f358] | FP8 GEMM/GEMV | hip | Qwen-GDN-MoE (6 ckpts) | [2 notes][t358] | not measured |

### Unique to Sparse / compressed attention (DSA, CSA/HCA, QSA)

44 entry points.

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| dsa_indexer::`dsa_{compact_pools, expand_selection, index_scores, indexer_store, kpool_compress, mla_masked_attn, topk_pools, topk_to_mask, write_geom}` (9) | [b300/common/dsa_indexer.cu:74][f2] | DSA indexer / sparse MLA | b300 | none — its callers' targets compile another copy | [1 note][t2] | not measured |
| dsa_indexer::`dsa_{compact_pools, expand_selection, index_scores, indexer_store, kpool_compress, mla_masked_attn, topk_pools, topk_to_mask, write_geom}` (9) | [gb10/common/dsa_indexer.cu:74][f27] | DSA indexer / sparse MLA | b200 gb10 hop | GLM-5.3 (1 ckpts) | [5 notes][t27] | not measured |
| attn_v41::`attn_v41_{act_quant_fp8, fp4_quant, gemm_f32, gemv_f32_staged, index_score, pool, ring_put, rmsnorm_bf16, rmsnorm_f32, rope, scale_bf16, scatter_cols, slice_cols, sparse_attn}` (14) | [gb10/deepseek-v4-flash/nvfp4/attn_v41.cu:100][f204] | CSA/HCA compressed attention | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [6 notes][t204] | not measured |
| kquant_moe::`kquant_mmvq_q2_k_{groups_w, pair_w}` (2) | [gb10/deepseek-v4-flash/nvfp4/kquant_moe.cu:321][f210] | expert GEMM/GEMV | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [5 notes][t210] | not measured |
| glm5next_dsa_mla_decode::`glm5next_dsa_mla_decode_fp8` | [gb10/glm-5.3-flash/nvfp4/glm5next_dsa_mla_decode.cu:94][f242] | DSA indexer / sparse MLA | gb10 | GLM-5.3 (1 ckpts) | [1 note][t242] | not measured |
| glm5next_mla_latent_write::`glm5next_mla_latent_write_fp8` | [gb10/glm-5.3-flash/nvfp4/glm5next_mla_latent_write.cu:30][f243] | MLA decode/prefill | gb10 | GLM-5.3 (1 ckpts) | [1 note][t243] | not measured |
| qsa_indexer::`qsa_{block_pool, gather, prefill_attn, qprep, qprep_rows, score, score_rows, score_rows_tc}` (8) | [gb10/qwen3.8-flash-next/nvfp4/qsa_indexer.cu:70][f288] | QSA sparse attention | gb10 | Qwen3.8-FN (1 ckpts) | [3 notes][t288] | not measured |

### Unique to GDN (gated delta rule linear attention)

227 entry points.

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| causal_conv1d::`causal_conv1d_update_chunk2` | [gb10/common/causal_conv1d.cu:247][f13] | causal conv1d | b200 b300 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t13] | not measured |
| dense_gemv_bf16_batch2::`dense_gemv_bf16_batch2` | [gb10/common/dense_gemv_bf16_batch2.cu:32][f18] | BF16/F32 GEMM/GEMV | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t18] | not measured |
| gated_delta_rule::`gated_delta_rule_{chunk2, chunk3, decode_f32, decode_f32_conv_norm, decode_f32_norm, decode_f32_strided, decode_f32_strided_norm, prefill}` (8) | [gb10/common/gated_delta_rule.cu:233][f35] | delta-rule recurrence | b200 b300 gb10 hop | none — its callers' targets compile another copy | [6 notes][t35] | not measured |
| gated_delta_rule_carry::`gdn_carry_conv` | [gb10/common/gated_delta_rule_carry.cu:301][f36] | delta-rule recurrence | b200 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [5 notes][t36] · [#34][pr34] | [33%][m36.gdn_carry_conv] (decode C=16 (R=32, MTP k=1)) |
| gated_delta_rule_carry::`gdn_{carry_conv_f32, carry_conv_flush, carry_flush, carry_wy2, carry_wy3, carry_wy3_lazy, carry_wy4, carry_wy4_lazy, conv_chain_f32, conv_chain_f32_batched}` (10) | [gb10/common/gated_delta_rule_carry.cu:263][f36] | delta-rule recurrence | b200 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [5 notes][t36] · [#34][pr34] | not measured |
| gated_delta_rule_carry::`gdn_carry_wy2_lazy` | [gb10/common/gated_delta_rule_carry.cu:266][f36] | delta-rule recurrence | b200 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [5 notes][t36] · [#34][pr34] | [51%][m36.gdn_carry_wy2_lazy] (decode C=16 (R=32, MTP k=1)) |
| gated_delta_rule_fla::`gated_delta_rule_chunk_delta_h_{ksplit, pipe, tc_vblock, tma, vtile}` (5) | [gb10/common/gated_delta_rule_fla.cu:857][f37] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [10 notes][t37] | not measured |
| gated_delta_rule_fla::`gated_delta_rule_chunk_delta_h_vfused` | [gb10/common/gated_delta_rule_fla.cu:1085][f37] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [8 notes][t37] | [25–36%][m37.gated_delta_rule_chunk_delta_h_vfused] (prefill 32k (cold, 32772 tok)) |
| gated_delta_rule_fla::`gated_delta_rule_chunk_fwd_o` | [gb10/common/gated_delta_rule_fla.cu:2003][f37] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [8 notes][t37] | [32–33%][m37.gated_delta_rule_chunk_fwd_o] (prefill 32k (cold, 32772 tok)) |
| gated_delta_rule_fla::`gated_delta_rule_recompute_wu` | [gb10/common/gated_delta_rule_fla.cu:262][f37] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [11 notes][t37] | [34–40%][m37.gated_delta_rule_recompute_wu] (prefill 32k (cold, 32772 tok)) |
| gated_delta_rule_persistent::`gated_delta_rule_prefill_{persistent, persistent_batched, persistent_wy4, persistent_wy4_batched}` (4) | [gb10/common/gated_delta_rule_persistent.cu:54][f38] | delta-rule recurrence | b200 b300 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [3 notes][t38] | not measured |
| gated_delta_rule_regresident::`gated_delta_rule_prefill_regresident` | [gb10/common/gated_delta_rule_regresident.cu:46][f39] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t39] | not measured |
| gated_delta_rule_wy::`gated_delta_rule_wy2` | [gb10/common/gated_delta_rule_wy.cu:30][f40] | delta-rule recurrence | b200 b300 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [2 notes][t40] | [82%][m40.gated_delta_rule_wy2] (decode C=1 (R=2, MTP k=1)) |
| gated_delta_rule_wy2_resident::`gated_delta_rule_wy2_resident` | [gb10/common/gated_delta_rule_wy2_resident.cu:58][f41] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [2 notes][t41] | not measured |
| gated_delta_rule_wy2_resident_f16::`gated_delta_rule_wy2_resident_f16` | [gb10/common/gated_delta_rule_wy2_resident_f16.cu:65][f42] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [2 notes][t42] | [58%][m42.gated_delta_rule_wy2_resident_f16] (decode C=16 (R=32, MTP k=1)) |
| gated_delta_rule_wy3::`gated_delta_rule_wy3` | [gb10/common/gated_delta_rule_wy3.cu:18][f43] | delta-rule recurrence | b200 b300 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [2 notes][t43] | not measured |
| gated_delta_rule_wy3_f16::`gated_delta_rule_wy3_f16` | [gb10/common/gated_delta_rule_wy3_f16.cu:40][f44] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t44] | not measured |
| gated_delta_rule_wy3_resident::`gated_delta_rule_wy3_resident` | [gb10/common/gated_delta_rule_wy3_resident.cu:59][f45] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t45] | not measured |
| gated_delta_rule_wy3_resident_f16::`gated_delta_rule_wy3_resident_f16` | [gb10/common/gated_delta_rule_wy3_resident_f16.cu:48][f46] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t46] | not measured |
| gated_delta_rule_wy4::`gated_delta_rule_wy4` | [gb10/common/gated_delta_rule_wy4.cu:18][f47] | delta-rule recurrence | b200 b300 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [2 notes][t47] | [58%][m47.gated_delta_rule_wy4] (decode C=1 (R=4, MTP k=3)) |
| gated_delta_rule_wy4_f16::`gated_delta_rule_wy4_f16` | [gb10/common/gated_delta_rule_wy4_f16.cu:40][f48] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t48] | not measured |
| gated_delta_rule_wy4_woa::`gated_delta_rule_wy4_{flag_clear, fold, woa}` (3) | [gb10/common/gated_delta_rule_wy4_woa.cu:55][f49] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [2 notes][t49] | not measured |
| gated_delta_rule_wy64_prefill::`gated_delta_rule_prefill_{wy64, wy64_batched}` (2) | [gb10/common/gated_delta_rule_wy64_prefill.cu:41][f50] | delta-rule recurrence | b200 b300 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t50] | not measured |
| gated_delta_rule_wy_f16::`gated_delta_rule_wy2_f16` | [gb10/common/gated_delta_rule_wy_f16.cu:45][f51] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t51] | not measured |
| gated_delta_rule_wyn::`gated_delta_rule_{wy10, wy10_f16, wy10_f16_table, wy10_table, wy11, wy11_f16, wy11_f16_table, wy11_table, wy12, wy12_f16, wy12_f16_table, wy12_table, wy13, wy13_f16, wy13_f16_table, wy13_table, wy14, wy14_f16, wy14_f16_table, wy14_table, wy15, wy15_f16, wy15_f16_table, wy15_table, wy16, wy16_f16, wy16_f16_table, wy16_table, wy5, wy5_f16, wy5_f16_table, wy5_table, wy6, wy6_f16, wy6_f16_table, wy6_table, wy7, wy7_f16, wy7_f16_table, wy7_table, wy8, wy8_f16, wy8_f16_table, wy8_table, wy9, wy9_f16, wy9_f16_table, wy9_table}` (48) | [gb10/common/gated_delta_rule_wyn.cu:293][f52] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [2 notes][t52] | not measured |
| gdn_chunk_fwd_o_mma8::`gated_delta_rule_chunk_fwd_o_mma8` | [gb10/common/gdn_chunk_fwd_o_mma8.cu:96][f53] | delta-rule recurrence | b200 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | — | not measured |
| gdn_verify_fused_conv_kn::`gdn_verify_fused_conv_kn` | [gb10/common/gdn_verify_fused_conv_kn.cu:50][f54] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t54] | not measured |
| gdn_verify_fused_conv_kn::`gdn_verify_fused_conv_kn_batched` | [gb10/common/gdn_verify_fused_conv_kn.cu:157][f54] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t54] | [35%][m54.gdn_verify_fused_conv_kn_batched] (decode C=16 (R=32, MTP k=1)) |
| gdn_verify_fused_k2::`gdn_verify_fused_{conv_k2, norm_k2}` (2) | [gb10/common/gdn_verify_fused_k2.cu:60][f55] | delta-rule recurrence | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t55] | not measured |
| ssm_ba_gates_hopper::`dense_gemm_ba_gates_prefill_hopper` | [gb10/common/ssm_ba_gates_hopper.cu:115][f173] | GDN pre/post-processing | b200 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [3 notes][t173] | not measured |
| ssm_h_dtype::`ssm_h_state_{f16_to_f32, f32_to_f16}` (2) | [gb10/common/ssm_h_dtype.cu:25][f175] | GDN pre/post-processing | b200 b300 gb10 hop | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t175] | not measured |
| ssm_preprocess::`compute_gdn_gates`, `deinterleave_qkvz`, `dense_gemv_ba_gates` | [gb10/common/ssm_preprocess.cu:35][f176] | GDN pre/post-processing | b200 b300 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | — | not measured |
| ssm_preprocess::`dense_gemm_ba_gates_prefill` | [gb10/common/ssm_preprocess.cu:482][f176] | GDN pre/post-processing | b200 b300 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [3 notes][t176] | [8–43%][m176.dense_gemm_ba_gates_prefill] (prefill 32k (cold, 32772 tok)) |
| ssm_state_norm::`ssm_state_{clamp_norm_fused, clamp_norm_fused_f16, nonfinite_count}` (3) | [gb10/common/ssm_state_norm.cu:32][f177] | GDN pre/post-processing | b200 b300 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t177] | not measured |
| w4a16_gemv::`w4a16_gemv_qkvz` | [gb10/common/w4a16_gemv.cu:1557][f184] | NVFP4 W4A16 GEMM/GEMV | b200 b300 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [1 note][t184] | not measured |
| gated_delta_rule::`gated_delta_rule_{chunk2, chunk3, decode_f32, decode_f32_conv_norm, decode_f32_norm, decode_f32_strided, decode_f32_strided_norm, prefill, prefill_split, prefill_split4, prefill_split4_batched}` (11) | [gb10/gemma-4-26b-a4b/nvfp4/gated_delta_rule.cu:24][f228] | delta-rule recurrence | gb10 | Qwen-GDN-MoE (6 ckpts) | [1 note][t228] | not measured |
| gated_delta_rule::`gated_delta_rule_{chunk2, chunk3, decode_f32, decode_f32_conv_norm, decode_f32_norm, decode_f32_strided, decode_f32_strided_norm, prefill, prefill_split, prefill_split4, prefill_split4_batched}` (11) | [gb10/qwen3-next-80b-a3b/nvfp4/gated_delta_rule.cu:24][f263] | delta-rule recurrence | b200 gb10 hop | Qwen-GDN, Qwen-GDN-MoE (13 ckpts) | [1 note][t263] | not measured |
| gated_delta_rule::`gated_delta_rule_{chunk2, chunk3, decode_f32, decode_f32_conv_norm, decode_f32_norm, decode_f32_strided, decode_f32_strided_norm, prefill, prefill_split, prefill_split4, prefill_split4_batched}` (11) | [gb10/qwen3.5-122b-a10b/nvfp4/gated_delta_rule.cu:24][f266] | delta-rule recurrence | gb10 | Qwen-GDN-MoE (6 ckpts) | [1 note][t266] | not measured |
| gated_delta_rule::`gated_delta_rule_{chunk2, chunk3, decode_f16_norm, decode_f16_strided_norm_half, decode_f32, decode_f32_conv_norm, decode_f32_norm, decode_f32_strided, decode_f32_strided_norm, decode_f32_strided_norm_half, decode_f32_strided_norm_smem, prefill, prefill_split, prefill_split4, prefill_split4_batched}` (15) | [gb10/qwen3.6-27b/nvfp4/gated_delta_rule.cu:24][f267] | delta-rule recurrence | gb10 hop strix hip | Qwen-GDN (7 ckpts) | [6 notes][t267] | not measured |
| gated_delta_rule_snap::`gated_delta_rule_decode_f32_{norm_snap, strided_norm_snap}` (2) | [gb10/qwen3.6-27b/nvfp4/gated_delta_rule_snap.cu:66][f268] | delta-rule recurrence | gb10 hop | Qwen-GDN (7 ckpts) | [2 notes][t268] | not measured |
| gdn_exact_carry::`gdn_exact_{carry2, carry2_lazy, carry3, carry3_lazy, carry4, carry4_lazy, carry_flush, chain2, chain3, chain4, chain_f16_2, chain_f16_3, chain_f16_4}` (13) | [gb10/qwen3.6-27b/nvfp4/gdn_exact_carry.cu:246][f269] | delta-rule recurrence | gb10 hop | Qwen-GDN (7 ckpts) | — | not measured |
| gdn_verify_fused_conv_kn_f32::`gdn_verify_fused_conv_kn_f32` | [gb10/qwen3.6-27b/nvfp4/gdn_verify_fused_conv_kn_f32.cu:33][f270] | delta-rule recurrence | gb10 hop | Qwen-GDN (7 ckpts) | [1 note][t270] | not measured |
| gated_delta_rule::`gated_delta_rule_{chunk2, chunk3, decode_f32, decode_f32_conv_norm, decode_f32_norm, decode_f32_strided, decode_f32_strided_norm, prefill, prefill_split, prefill_split4, prefill_split4_batched}` (11) | [gb10/qwen3.6-35b-a3b/nvfp4/gated_delta_rule.cu:63][f279] | delta-rule recurrence | b200 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [2 notes][t279] | not measured |
| gated_delta_rule_wy17::`gated_delta_rule_wy17` | [gb10/qwen3.6-35b-a3b/nvfp4/gated_delta_rule_wy17.cu:41][f280] | delta-rule recurrence | b200 gb10 hop strix | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [2 notes][t280] | not measured |
| gdn_exact_carry::`gdn_exact_{carry2, carry2_lazy, carry3, carry3_lazy, carry4, carry4_lazy, carry_flush, chain2, chain3, chain4}` (10) | [gb10/qwen3.6-35b-a3b/nvfp4/gdn_exact_carry.cu:209][f281] | delta-rule recurrence | b200 gb10 hop | Qwen-GDN-MoE (6 ckpts) | — | not measured |
| gated_norm_sigmoid::`gated_rms_norm_{f32_input_sigmoid, prefill_sigmoid, sigmoid}` (3) | [gb10/qwen3.8-flash-next/nvfp4/gated_norm_sigmoid.cu:50][f285] | normalization | gb10 | Qwen3.8-FN (1 ckpts) | [1 note][t285] | not measured |
| gated_delta_rule_chunk_tc::`gated_delta_rule_chunk_delta_h_tcfuse_x2` | [hopper/common/gated_delta_rule_chunk_tc.cu:409][f292] | delta-rule recurrence | hop | Qwen-GDN, Qwen-GDN-MoE (13 ckpts) | [5 notes][t292] | not measured |
| gdn_fwd_o_hopper::`gated_delta_rule_chunk_fwd_o_hopper` | [hopper/common/gdn_fwd_o_hopper.cu:94][f293] | delta-rule recurrence | hop | Qwen-GDN, Qwen-GDN-MoE (13 ckpts) | [3 notes][t293] | not measured |
| gdn_recompute_wu_hopper::`gated_delta_rule_recompute_wu_hopper` | [hopper/common/gdn_recompute_wu_hopper.cu:138][f294] | delta-rule recurrence | hop | Qwen-GDN, Qwen-GDN-MoE (13 ckpts) | [3 notes][t294] | not measured |
| add_rms_norm::`add_rms_norm` | [metal/common/add_rms_norm.metal:32][f304] | normalization | metal | Qwen-GDN (7 ckpts) | [1 note][t304] | not measured |
| attention_decode::`attention_decode` | [metal/common/attention_decode.metal:30][f306] | paged decode | metal | Qwen-GDN (7 ckpts) | [1 note][t306] | not measured |
| attention_decode_bf16k_turbov::`attention_decode_bf16k_{turbo2v, turbo3v, turbo4v}` (3) | [metal/common/attention_decode_bf16k_turbov.metal:113][f307] | paged decode | metal | Qwen-GDN (7 ckpts) | [1 note][t307] | not measured |
| attention_decode_turbo2::`attention_decode_turbo2` | [metal/common/attention_decode_turbo2.metal:39][f308] | paged decode | metal | Qwen-GDN (7 ckpts) | [1 note][t308] | not measured |
| attention_decode_turbo3::`attention_decode_turbo3` | [metal/common/attention_decode_turbo3.metal:53][f309] | paged decode | metal | Qwen-GDN (7 ckpts) | [1 note][t309] | not measured |
| attention_decode_turbo4::`attention_decode_turbo4` | [metal/common/attention_decode_turbo4.metal:41][f310] | paged decode | metal | Qwen-GDN (7 ckpts) | [1 note][t310] | not measured |
| attention_decode_turbo8::`attention_decode_turbo8` | [metal/common/attention_decode_turbo8.metal:38][f311] | paged decode | metal | Qwen-GDN (7 ckpts) | [1 note][t311] | not measured |
| gdn_helpers::`gdn_compute_gate` | [metal/common/gdn_helpers.metal:28][f322] | delta-rule recurrence | metal | Qwen-GDN (7 ckpts) | — | not measured |
| gdn_helpers::`sigmoid_bf16_to_f32` | [metal/common/gdn_helpers.metal:56][f322] | GDN helper | metal | Qwen-GDN (7 ckpts) | — | not measured |
| kv_cache_append::`kv_cache_append` | [metal/common/kv_cache_append.metal:19][f324] | cache write | metal | Qwen-GDN (7 ckpts) | — | not measured |
| kv_cache_append_bf16k_turbov::`kv_cache_append_bf16k_{turbo2v, turbo3v, turbo4v}` (3) | [metal/common/kv_cache_append_bf16k_turbov.metal:129][f325] | cache write | metal | Qwen-GDN (7 ckpts) | [1 note][t325] | not measured |
| kv_cache_append_turbo2::`kv_cache_append_turbo2` | [metal/common/kv_cache_append_turbo2.metal:53][f326] | cache write | metal | Qwen-GDN (7 ckpts) | [1 note][t326] | not measured |
| kv_cache_append_turbo3::`kv_cache_append_turbo3` | [metal/common/kv_cache_append_turbo3.metal:65][f327] | cache write | metal | Qwen-GDN (7 ckpts) | [1 note][t327] | not measured |
| kv_cache_append_turbo4::`kv_cache_append_turbo4` | [metal/common/kv_cache_append_turbo4.metal:82][f328] | cache write | metal | Qwen-GDN (7 ckpts) | [1 note][t328] | not measured |
| kv_cache_append_turbo8::`kv_cache_append_turbo8` | [metal/common/kv_cache_append_turbo8.metal:47][f329] | cache write | metal | Qwen-GDN (7 ckpts) | [1 note][t329] | not measured |
| qwen35_qkv_split::`qwen35_qkv_split` | [metal/common/qwen35_qkv_split.metal:20][f339] | GDN helper | metal | Qwen-GDN (7 ckpts) | — | not measured |
| rope_apply::`rope_apply` | [metal/common/rope_apply.metal:33][f341] | rotary | metal | Qwen-GDN (7 ckpts) | — | not measured |

### Unique to KDA (Kimi delta attention, linear attention)

10 entry points.

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| kda_chunk::`kda_chunk_{prepare, scan}` (2) | [gb10/common/kda_chunk.cu:95][f62] | KDA op | b200 b300 gb10 hop | GLM-5.3 (1 ckpts) | [2 notes][t62] | not measured |
| kda_gate::`kda_gate_bf16` | [gb10/common/kda_gate.cu:74][f63] | KDA op | b200 b300 gb10 hop | GLM-5.3 (1 ckpts) | — | not measured |
| kda_layer_ops::`kda_{fill_f32, o_norm_gated_bf16, pack_qkv_bf16, sigmoid_bf16_f32, split_widen}` (5) | [gb10/common/kda_layer_ops.cu:50][f64] | KDA op | b200 b300 gb10 hop | GLM-5.3 (1 ckpts) | [1 note][t64] | not measured |
| kda_recurrent::`kda_recurrent_decode_{bf16, bf16_smem}` (2) | [gb10/common/kda_recurrent.cu:144][f65] | KDA op | b200 b300 gb10 hop | GLM-5.3 (1 ckpts) | [2 notes][t65] | not measured |

### Unique to Mamba2 (selective state-space scan)

7 entry points.

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| mamba2_ssd_chunk::`mamba2_ssd_{bmm, cumsum, scan}` (3) | [gb10/common/mamba2_ssd_chunk.cu:44][f67] | SSD / selective scan | b200 b300 gb10 hop | Nemotron-H (3 ckpts) | [2 notes][t67] | not measured |
| mamba2_ssm::`mamba2_ssm_{decode, prefill, prefill_persistent}` (3) | [gb10/common/mamba2_ssm_decode.cu:29][f68] | SSD / selective scan | b200 b300 gb10 hop strix hip | Nemotron-H (3 ckpts) | [2 notes][t68] | not measured |
| w4a16::`fp8_fp8_gemm_t_m128_mfast` | [gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/w4a16_gemm.cu:1678][f261] | FP8 GEMM/GEMV | gb10 | Nemotron-H (3 ckpts) | [1 note][t261] | not measured |

### Unique to Causal conv1d (short convolution of GDN/KDA/Mamba2)

1 entry points.

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| causal_conv1d::`causal_conv1d_prefill_state` | [gb10/common/causal_conv1d.cu:646][f13] | causal conv1d | b200 b300 gb10 hop strix hip | GLM-5.3, Kimi-K3, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (19 ckpts) | [1 note][t13] | not measured |

### Unique to MoE (routing, dispatch, expert GEMM/GEMV, combine)

203 entry points.

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| moe_shared_expert_fused::`moe_expert_{gate_up_shared, silu_down_shared}` (2) | [b300/common/moe_shared_expert_fused.cu:48][f3] | expert GEMM/GEMV | b300 | Kimi-K3 (1 ckpts) | [2 notes][t3] | not measured |
| gemm::`dense_gemm_bf16_router` | [gb10/common/dense_gemm_bf16.cu:204][f14] | routing / top-k | b200 b300 gb10 hop strix | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [3 notes][t14] | [6%][m14.dense_gemm_bf16_router] (prefill 32k (cold, 32772 tok)) |
| gemm::`dense_gemm_f32in_f32out` | [gb10/common/dense_gemm_bf16.cu:131][f14] | BF16/F32 GEMM/GEMV | b200 b300 gb10 hop strix | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | — | not measured |
| glm5next_ffn::`glm5next_moe_{combine, combine_indexed}` (2) | [gb10/common/glm5next_ffn.cu:213][f56] | dispatch / combine | b200 b300 gb10 hop | GLM-5.3 (1 ckpts) | [1 note][t56] | not measured |
| glm5next_ffn::`glm5next_router_topk` | [gb10/common/glm5next_ffn.cu:102][f56] | routing / top-k | b200 b300 gb10 hop | GLM-5.3 (1 ckpts) | [1 note][t56] | not measured |
| glm5next_ffn::`glm5next_swiglu_clamp` | [gb10/common/glm5next_ffn.cu:31][f56] | expert activation | b200 b300 gb10 hop | GLM-5.3 (1 ckpts) | [1 note][t56] | not measured |
| moe_bf16_grouped_gemm::`moe_bf16_grouped_gemm` | [gb10/common/moe_bf16_grouped_gemm.cu:86][f70] | expert GEMM/GEMV | b200 b300 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t70] | not measured |
| moe_bf16_grouped_tc::`moe_expert_{down_act_bf16_grouped_tc, gate_up_act_bf16_grouped_tc}` (2) | [gb10/common/moe_bf16_grouped_tc.cu:42][f71] | expert GEMM/GEMV | b200 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | — | not measured |
| moe_decode_atomic_c4::`moe_decode_atomic_c4_finalize` | [gb10/common/moe_decode_atomic_c4.cu:183][f72] | dispatch / combine | b200 b300 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t72] | not measured |
| moe_decode_atomic_c4::`moe_decode_atomic_c4_silu_down_accum` | [gb10/common/moe_decode_atomic_c4.cu:37][f72] | expert GEMM/GEMV | b200 b300 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t72] | not measured |
| moe_relu2_fused::`moe_expert_relu2_down_shared` | [gb10/common/moe_expert_relu2_down_shared.cu:53][f75] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | Nemotron-H (3 ckpts) | [1 note][t75] | not measured |
| moe_fp8_grouped_blend::`moe_weighted_sum_blend_fp8_grouped` | [gb10/common/moe_fp8_grouped_blend.cu:17][f76] | dispatch / combine | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t76] · [#4][pr4] | [16–77%][m76.moe_weighted_sum_blend_fp8_grouped] (decode C=1 (R=2, MTP k=1)) |
| moe_fp8_grouped_gemm::`moe_fp8_grouped_gemm` | [gb10/common/moe_fp8_grouped_gemm.cu:281][f77] | expert GEMM/GEMV | b200 b300 gb10 hop strix | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t77] | not measured |
| moe_fp8_grouped_sort::`moe_fp8_grouped_sort` | [gb10/common/moe_fp8_grouped_sort.cu:24][f78] | dispatch / combine | b200 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [3 notes][t78] · [#34][pr34] | [1%][m78.moe_fp8_grouped_sort] (decode C=1 (R=2, MTP k=1)) |
| moe_fp8_grouped_tc::`moe_expert_{down_act_fp8_grouped_tc, gate_up_act_fp8_grouped_tc}` (2) | [gb10/common/moe_fp8_grouped_tc.cu:60][f79] | expert GEMM/GEMV | b200 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t79] | not measured |
| moe_fp8_grouped_tc_w8a8::`moe_{act_quant_e4m3, expert_down_act_fp8_grouped_tc_w8a8, expert_gate_up_act_fp8_grouped_tc_w8a8}` (3) | [gb10/common/moe_fp8_grouped_tc_w8a8.cu:54][f80] | expert GEMM/GEMV | b200 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t80] | not measured |
| moe_gate_topk::`moe_gate_topk_fused` | [gb10/common/moe_gate_topk.cu:46][f81] | routing / top-k | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t81] | not measured |
| moe_hash_route::`moe_hash_{route, route_batched}` (2) | [gb10/common/moe_hash_route.cu:25][f82] | routing / top-k | b200 b300 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t82] | not measured |
| moe_nvfp4_grouped::`moe_expert_{down_act_nvfp4_grouped, gate_up_act_nvfp4_grouped}` (2) | [gb10/common/moe_nvfp4_grouped.cu:72][f85] | expert GEMM/GEMV | b200 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [5 notes][t85] · [#34][pr34] | not measured |
| moe_nvfp4_grouped_tc::`moe_expert_{down_act_nvfp4_grouped_tc, down_act_nvfp4_grouped_tc_lean, gate_up_act_nvfp4_grouped_tc, gate_up_act_nvfp4_grouped_tc_lean}` (4) | [gb10/common/moe_nvfp4_grouped_tc.cu:177][f86] | expert GEMM/GEMV | b200 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | — | not measured |
| moe_nvfp4_grouped_tc::`nvfp4_tc_lean_repack` | [gb10/common/moe_nvfp4_grouped_tc.cu:227][f86] | dequant / repack / transpose | b200 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | — | not measured |
| moe::`moe_batched_blend` | [gb10/common/moe_permute.cu:126][f87] | dispatch / combine | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t87] | [97–100%][m87.moe_batched_blend] (prefill 32k (cold, 32772 tok)) |
| moe::`moe_build_tile_worklist` | [gb10/common/moe_permute.cu:276][f87] | dispatch / combine | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t87] | [0%][m87.moe_build_tile_worklist] (prefill 32k (cold, 32772 tok)) |
| moe::`moe_{permute_tokens, sort_by_expert}` (2) | [gb10/common/moe_permute.cu:20][f87] | dispatch / combine | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t87] | not measured |
| moe::`moe_unpermute_reduce_indexed` | [gb10/common/moe_permute.cu:95][f87] | dispatch / combine | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t87] | [87–88%][m87.moe_unpermute_reduce_indexed] (prefill 32k (cold, 32772 tok)) |
| moe_prefill::`moe_expert_{gate_up_shared_prefill, silu_down_shared_prefill}` (2) | [gb10/common/moe_prefill.cu:59][f88] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t88] | not measured |
| moe_prefill::`moe_weighted_sum_blend_prefill` | [gb10/common/moe_prefill.cu:360][f88] | dispatch / combine | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t88] | not measured |
| moe_router_gemm::`moe_router_gemm_bf16` | [gb10/common/moe_router_gemm.cu:32][f89] | routing / top-k | b200 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [3 notes][t89] · [#34][pr34] | not measured |
| moe_router_gemm_prefill::`moe_router_gemm_rt` | [gb10/common/moe_router_gemm_prefill.cu:24][f90] | routing / top-k | b200 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | — | not measured |
| moe_shared_expert_fused::`moe_expert_{gate_up_shared, silu_down_shared}` (2) | [gb10/common/moe_shared_expert_fused.cu:48][f91] | expert GEMM/GEMV | b200 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t91] | not measured |
| moe_fused_batch2::`moe_expert_{gate_up_shared_batch2, silu_down_shared_batch2}` (2) | [gb10/common/moe_shared_expert_fused_batch2.cu:292][f92] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t92] | not measured |
| moe_fused_batch2::`moe_weighted_sum_blend_batch2` | [gb10/common/moe_shared_expert_fused_batch2.cu:497][f92] | dispatch / combine | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t92] | not measured |
| moe_shared_expert_fused_batch2_t::`moe_expert_{gate_up_shared_batch2_t, silu_down_shared_batch2_t}` (2) | [gb10/common/moe_shared_expert_fused_batch2_t.cu:41][f93] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t93] | not measured |
| moe_fused_batch3::`moe_expert_{gate_up_shared_batch3, silu_down_shared_batch3}` (2) | [gb10/common/moe_shared_expert_fused_batch3.cu:50][f94] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t94] | not measured |
| moe_fused_batch3::`moe_weighted_sum_blend_batch3` | [gb10/common/moe_shared_expert_fused_batch3.cu:335][f94] | dispatch / combine | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t94] | not measured |
| moe_shared_expert_fused_batch3_t::`moe_expert_{gate_up_shared_batch3_t, silu_down_shared_batch3_t}` (2) | [gb10/common/moe_shared_expert_fused_batch3_t.cu:36][f95] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t95] | not measured |
| moe_shared_expert_fused_bf16::`moe_expert_{gate_up_shared_bf16, silu_down_shared_bf16}` (2) | [gb10/common/moe_shared_expert_fused_bf16.cu:26][f96] | expert GEMM/GEMV | b200 b300 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t96] | not measured |
| moe_shared_expert_fused_bf16_batch2::`moe_expert_{gate_up_shared_bf16_batch2, silu_down_shared_bf16_batch2}` (2) | [gb10/common/moe_shared_expert_fused_bf16_batch2.cu:39][f97] | expert GEMM/GEMV | b200 b300 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t97] | not measured |
| moe_shared_expert_fused_fp8::`moe_expert_gate_up_shared_fp8` | [gb10/common/moe_shared_expert_fused_fp8.cu:98][f98] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t98] | [85%][m98.moe_expert_gate_up_shared_fp8] (decode C=1 (R=2, MTP k=1)) |
| moe_shared_expert_fused_fp8::`moe_expert_silu_down_shared_fp8` | [gb10/common/moe_shared_expert_fused_fp8.cu:262][f98] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t98] | not measured |
| moe_shared_expert_fused_fp8_batch2::`moe_expert_{gate_up_shared_fp8_batch2, silu_down_shared_fp8_batch2}` (2) | [gb10/common/moe_shared_expert_fused_fp8_batch2.cu:97][f99] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t99] | not measured |
| moe_shared_expert_fused_fp8_batch2::`moe_weighted_sum_blend_fp8_batch2` | [gb10/common/moe_shared_expert_fused_fp8_batch2.cu:410][f99] | dispatch / combine | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t99] | not measured |
| moe_shared_expert_fused_fp8_batch2_t::`moe_expert_{gate_up_shared_fp8_batch2_t, silu_down_shared_fp8_batch2_t}` (2) | [gb10/common/moe_shared_expert_fused_fp8_batch2_t.cu:28][f100] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t100] | not measured |
| moe_shared_expert_fused_fp8_batch3::`moe_expert_{gate_up_shared_fp8_batch3, silu_down_shared_fp8_batch3}` (2) | [gb10/common/moe_shared_expert_fused_fp8_batch3.cu:97][f101] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t101] | not measured |
| moe_shared_expert_fused_fp8_batch3::`moe_weighted_sum_blend_fp8_batch3` | [gb10/common/moe_shared_expert_fused_fp8_batch3.cu:408][f101] | dispatch / combine | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t101] | not measured |
| moe_shared_expert_fused_fp8_batch3_t::`moe_expert_{gate_up_shared_fp8_batch3_t, silu_down_shared_fp8_batch3_t}` (2) | [gb10/common/moe_shared_expert_fused_fp8_batch3_t.cu:28][f102] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t102] | not measured |
| moe_shared_expert_fused_fp8_grouped::`moe_expert_down_act_fp8_grouped` | [gb10/common/moe_shared_expert_fused_fp8_grouped.cu:331][f103] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [12 notes][t103] · [#4][pr4] [#34][pr34] | [89–102%][m103.moe_expert_down_act_fp8_grouped] (decode C=16 (R=32, MTP k=1)) |
| moe_shared_expert_fused_fp8_grouped::`moe_expert_gate_up_act_fp8_grouped` | [gb10/common/moe_shared_expert_fused_fp8_grouped.cu:153][f103] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [11 notes][t103] · [#4][pr4] [#34][pr34] | [88–97%][m103.moe_expert_gate_up_act_fp8_grouped] (decode C=16 (R=32, MTP k=1)) |
| moe_shared_expert_fused_fp8_t::`moe_expert_{gate_up_shared_fp8_t, silu_down_shared_fp8_t}` (2) | [gb10/common/moe_shared_expert_fused_fp8_t.cu:34][f104] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t104] | not measured |
| moe_shared_expert_fused_t::`moe_expert_{gate_up_shared_t, gate_up_shared_t_e8m0, silu_down_shared_t, silu_down_shared_t_e8m0}` (4) | [gb10/common/moe_shared_expert_fused_t.cu:187][f105] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t105] | not measured |
| moe_silu_mul::`silu_mul_quant_fp8` | [gb10/common/moe_silu_mul.cu:107][f106] | activation quantize | b200 b300 gb10 hop strix hip | GLM-5.3, Gemma4, Kimi-K3, Laguna, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN (19 ckpts) | [1 note][t106] | [21–24%][m106.silu_mul_quant_fp8] (prefill 32k (cold, 32772 tok)) |
| moe_sorted::`moe_sorted_{gate_up, silu_down}` (2) | [gb10/common/moe_sorted_prefill.cu:54][f107] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t107] | not measured |
| moe_topk::`moe_topk_softmax_{batched, f32}` (2) | [gb10/common/moe_topk.cu:242][f108] | routing / top-k | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t108] | not measured |
| moe_topk::`moe_topk_softmax_rows` | [gb10/common/moe_topk.cu:220][f108] | routing / top-k | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t108] · [#34][pr34] | [0%][m108.moe_topk_softmax_rows] (decode C=1 (R=2, MTP k=1)) |
| moe_topk_sig::`moe_topk_{sigmoid, sigmoid_batched}` (2) | [gb10/common/moe_topk_sigmoid.cu:22][f109] | routing / top-k | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t109] | not measured |
| moe_topk_softmax_bias::`moe_topk_softmax_{bias, bias_batched}` (2) | [gb10/common/moe_topk_softmax_bias.cu:189][f110] | routing / top-k | b200 b300 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t110] | not measured |
| moe_topk_softmax_bias::`moe_zero_expert_add` | [gb10/common/moe_topk_softmax_bias.cu:233][f110] | dispatch / combine | b200 b300 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t110] | not measured |
| moe_topk_sqrt::`moe_topk_{sqrtsoftplus, sqrtsoftplus_batched}` (2) | [gb10/common/moe_topk_sqrtsoftplus.cu:22][f111] | routing / top-k | b200 b300 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t111] | not measured |
| moe_transpose_batched::`moe_transpose_u8_batched` | [gb10/common/moe_transpose_batched.cu:21][f112] | dispatch / combine | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [1 note][t112] | not measured |
| moe_unpermute_blend::`moe_unpermute_blend` | [gb10/common/moe_unpermute_blend.cu:17][f113] | dispatch / combine | b200 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | — | not measured |
| moe_w4a16::`moe_w4a16_grouped_gemm_{ptrtable, ptrtable_k32, ptrtable_t}` (3) | [gb10/common/moe_w4a16_grouped_gemm.cu:231][f114] | expert GEMM/GEMV | b200 gb10 hop | GLM-5.3, Gemma4, Nemotron-H (6 ckpts) | [4 notes][t114] | not measured |
| moe_w4a16::`moe_w4a16_grouped_gemm_ptrtable_{alkm_m16_k128, bt_m16_k128, bt_m16_n128_k128, k64, m16_k64}` (5) | [gb10/common/moe_w4a16_grouped_gemm.cu:945][f114] | expert GEMM/GEMV | b200 gb10 hop | GLM-5.3 (1 ckpts) | [4 notes][t114] | not measured |
| moe_w8a8_grouped_gemm::`moe_w8a8_grouped_gemm` | [gb10/common/moe_w8a8_grouped_gemm.cu:93][f115] | expert GEMM/GEMV | b200 b300 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [2 notes][t115] | not measured |
| moe_w8a8_grouped_gemm::`moe_w8a8_grouped_gemm_pm4` | [gb10/common/moe_w8a8_grouped_gemm.cu:396][f115] | expert GEMM/GEMV | b200 b300 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | [4 notes][t115] | [18–32%][m115.moe_w8a8_grouped_gemm_pm4] (prefill 32k (cold, 32772 tok)) |
| moe_w8a8_grouped_gemm_e4m3::`moe_w8a8_{gateup_silu_e4m3_w1, gateup_silu_e4m3_w2, grouped_gemm_e4m3_dn, grouped_gemm_e4m3_gu}` (4) | [gb10/common/moe_w8a8_grouped_gemm_e4m3.cu:101][f116] | expert GEMM/GEMV | b200 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN-MoE, Qwen3-VL, Qwen3.8-FN, Step-3.7 (23 ckpts) | — | not measured |
| nemotron_moe_prefill::`nemotron_moe_{relu2_down_prefill, up_prefill}` (2) | [gb10/common/nemotron_moe_prefill.cu:161][f117] | expert GEMM/GEMV | b200 b300 gb10 hop strix hip | Nemotron-H (3 ckpts) | — | not measured |
| nemotron_moe_prefill::`nemotron_moe_topk_sigmoid_batched` | [gb10/common/nemotron_moe_prefill.cu:53][f117] | routing / top-k | b200 b300 gb10 hop strix hip | Nemotron-H (3 ckpts) | [1 note][t117] | not measured |
| nemotron_moe_prefill::`nemotron_moe_weighted_sum_prefill` | [gb10/common/nemotron_moe_prefill.cu:444][f117] | dispatch / combine | b200 b300 gb10 hop strix hip | Nemotron-H (3 ckpts) | — | not measured |
| relu2::`moe_weighted_sum_scale` | [gb10/common/relu_squared.cu:57][f162] | dispatch / combine | b200 b300 gb10 hop strix hip | Nemotron-H (3 ckpts) | [1 note][t162] | not measured |
| relu2::`relu_squared_inplace` | [gb10/common/relu_squared.cu:26][f162] | activation / gate / residual | b200 b300 gb10 hop strix hip | Nemotron-H (3 ckpts) | — | not measured |
| w4a16_gemv::`glm5next_moe_row_union` | [gb10/common/w4a16_gemv.cu:2201][f184] | dispatch / combine | b200 b300 gb10 hop strix hip | GLM-5.3 (1 ckpts) | [2 notes][t184] | not measured |
| w4a16_gemv::`w4a16_gemv_sw_{moe, moe_batchm_m2, moe_batchm_m3, moe_batchm_m4, moe_batchm_m5, moe_batchm_m6, moe_batchm_m7, moe_batchm_m8}` (8) | [gb10/common/w4a16_gemv.cu:300][f184] | NVFP4 W4A16 GEMM/GEMV | b200 b300 gb10 hop strix hip | GLM-5.3 (1 ckpts) | [1 note][t184] | not measured |
| kquant_moe::`kquant_mmvq_{q2_k_experts_w2, q2_k_experts_w8, q3_k_experts_w2, q3_k_experts_w8}` (4) | [gb10/deepseek-v4-flash/nvfp4/kquant_moe.cu:397][f210] | expert GEMM/GEMV | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [6 notes][t210] | not measured |
| kquant_moe::`kquant_mmvq_q3_k_w`, `metrale_q3_k_mmq128_nc`, `metrale_q3_k_mmq128_wc` | [gb10/deepseek-v4-flash/nvfp4/kquant_moe.cu:78][f210] | integer / K-quant GEMM/GEMV | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [4 notes][t210] | not measured |
| kquant_moe::`kquant_swiglu_q8_1_rows_bf16`, `metrale_q8_1_quantize_d4_bf16` | [gb10/deepseek-v4-flash/nvfp4/kquant_moe.cu:86][f210] | activation quantize | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [4 notes][t210] | not measured |
| moe_v41::`moe_v41_{accumulate, finish, gather_rows, scatter_add, slot_table_set, sum_rows, swiglu}` (7) | [gb10/deepseek-v4-flash/nvfp4/moe_v41.cu:16][f218] | dispatch / combine | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [1 note][t218] | not measured |
| moe_v41::`moe_v41_{route_select, router_gemv_f32out, router_gemv_f32out_products}` (3) | [gb10/deepseek-v4-flash/nvfp4/moe_v41.cu:145][f218] | routing / top-k | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [3 notes][t218] | not measured |
| moe_w4a16::`moe_{fp8_grouped_gemm_ptrtable_t, w4a16_fused_gate_up_t, w4a16_fused_gate_up_t_e8m0, w4a16_fused_gate_up_t_k64, w4a16_fused_gate_up_t_k64_e8m0, w4a16_grouped_gemm_ptrtable, w4a16_grouped_gemm_ptrtable_e8m0, w4a16_grouped_gemm_ptrtable_t, w4a16_grouped_gemm_ptrtable_t_e8m0, w4a16_grouped_gemm_ptrtable_t_k64, w4a16_grouped_gemm_ptrtable_t_k64_e8m0}` (11) | [gb10/deepseek-v4-flash/nvfp4/moe_w4a16_grouped_gemm.cu:188][f219] | expert GEMM/GEMV | b200 b300 gb10 hop | DeepSeek-V4, Kimi-K3, LongCat (4 ckpts) | [1 note][t219] | not measured |
| moe_shared_expert_fused::`moe_expert_{gate_up_shared, silu_down_shared}` (2) | [gb10/gemma-4-26b-a4b/nvfp4/moe_shared_expert_fused.cu:31][f231] | expert GEMM/GEMV | gb10 | Gemma4 (2 ckpts) | [1 note][t231] | not measured |
| moe_fused_batch2::`moe_expert_{gate_up_shared_batch2, silu_down_shared_batch2}` (2) | [gb10/gemma-4-26b-a4b/nvfp4/moe_shared_expert_fused_batch2.cu:35][f232] | expert GEMM/GEMV | gb10 | Gemma4 (2 ckpts) | [1 note][t232] | not measured |
| moe_fused_batch2::`moe_weighted_sum_blend_batch2` | [gb10/gemma-4-26b-a4b/nvfp4/moe_shared_expert_fused_batch2.cu:327][f232] | dispatch / combine | gb10 | Gemma4 (2 ckpts) | [1 note][t232] | not measured |
| moe_fused_batch3::`moe_expert_{gate_up_shared_batch3, silu_down_shared_batch3}` (2) | [gb10/gemma-4-26b-a4b/nvfp4/moe_shared_expert_fused_batch3.cu:33][f233] | expert GEMM/GEMV | gb10 | Gemma4 (2 ckpts) | [1 note][t233] | not measured |
| moe_fused_batch3::`moe_weighted_sum_blend_batch3` | [gb10/gemma-4-26b-a4b/nvfp4/moe_shared_expert_fused_batch3.cu:323][f233] | dispatch / combine | gb10 | Gemma4 (2 ckpts) | [1 note][t233] | not measured |
| moe_w4a16::`moe_{fp8_grouped_gemm_ptrtable_t, w4a16_fused_gate_up_t, w4a16_fused_gate_up_t_k64, w4a16_grouped_gemm_ptrtable, w4a16_grouped_gemm_ptrtable_t, w4a16_grouped_gemm_ptrtable_t_k64}` (6) | [gb10/gemma-4-26b-a4b/nvfp4/moe_w4a16_grouped_gemm.cu:34][f234] | expert GEMM/GEMV | b200 gb10 hop | Gemma4, Mistral4, Qwen-GDN-MoE, Qwen3-VL (10 ckpts) | [1 note][t234] | not measured |
| moe_w4a16::`moe_{fp8_grouped_gemm_ptrtable_t, w4a16_fused_gate_up_t, w4a16_fused_gate_up_t_k64, w4a16_fused_gate_up_t_k64_m128, w4a16_grouped_gemm_ptrtable, w4a16_grouped_gemm_ptrtable_t, w4a16_grouped_gemm_ptrtable_t_k64}` (7) | [gb10/minimax-m2-229b/nvfp4/moe_w4a16_grouped_gemm.cu:34][f248] | expert GEMM/GEMV | gb10 | Laguna, MiniMax-M2, Step-3.7 (4 ckpts) | [1 note][t248] | not measured |
| moe_w4a16::`moe_w4a16_grouped_gemm_{ptrtable, ptrtable_relu2, ptrtable_t}` (3) | [gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/moe_w4a16_grouped_gemm.cu:590][f258] | expert GEMM/GEMV | gb10 | Nemotron-H (3 ckpts) | [2 notes][t258] | not measured |
| moe_w4a4::`moe_w4a4_grouped_gemm_relu2` | [gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/moe_w4a4_grouped.cu:49][f259] | expert GEMM/GEMV | gb10 | Nemotron-H (3 ckpts) | [1 note][t259] | not measured |
| moe_w4a16::`moe_{fp8_grouped_gemm_ptrtable_t, w4a16_fused_gate_up_t, w4a16_fused_gate_up_t_k64, w4a16_grouped_gemm_ptrtable, w4a16_grouped_gemm_ptrtable_t, w4a16_grouped_gemm_ptrtable_t_k64}` (6) | [gb10/qwen3.6-27b/nvfp4/moe_w4a16_grouped_gemm.cu:135][f271] | expert GEMM/GEMV | gb10 hop strix | none — its callers' targets compile another copy | [1 note][t271] | not measured |
| moe_w4a16::`moe_{fp8_grouped_gemm_ptrtable_t, w4a16_down_t_k64_fp4, w4a16_fused_gate_up_t, w4a16_fused_gate_up_t_k64, w4a16_fused_gate_up_t_k64_fp4, w4a16_grouped_gemm_ptrtable, w4a16_grouped_gemm_ptrtable_k32, w4a16_grouped_gemm_ptrtable_m256, w4a16_grouped_gemm_ptrtable_t, w4a16_grouped_gemm_ptrtable_t_k64}` (10) | [gb10/qwen3.6-35b-a3b/nvfp4/moe_w4a16_grouped_gemm.cu:34][f282] | expert GEMM/GEMV | b200 gb10 hop strix | Qwen-GDN-MoE, Qwen3.8-FN (7 ckpts) | [9 notes][t282] | not measured |
| moe_bucket_builder::`bucket_builder` | [hopper/common/moe_bucket_builder.cu:5][f295] | dispatch / combine | hop | DeepSeek-V4, Nemotron-H, Qwen-GDN-MoE (11 ckpts) | [2 notes][t295] · [#25][pr25] | not measured |
| moe_w8a8_m16::`pm4_m16` | [hopper/common/moe_w8a8_m16.cu:106][f296] | expert GEMM/GEMV | hop | DeepSeek-V4, Nemotron-H, Qwen-GDN-MoE (11 ckpts) | [3 notes][t296] · [#25][pr25] | not measured |
| gemm::`dense_gemm_f32in_f32out` | [strix-hip/common/dense_gemm_bf16.cu:129][f351] | BF16/F32 GEMM/GEMV | hip | Qwen-GDN-MoE (6 ckpts) | [1 note][t351] | not measured |
| moe_fp8_grouped_gemm::`moe_fp8_grouped_gemm` | [strix-hip/common/moe_fp8_grouped_gemm.cu:249][f353] | expert GEMM/GEMV | hip | Qwen-GDN-MoE (6 ckpts) | [2 notes][t353] | not measured |
| moe_w4a16::`moe_{fp8_grouped_gemm_ptrtable_t, w4a16_fused_gate_up_t, w4a16_fused_gate_up_t_k64, w4a16_grouped_gemm_ptrtable, w4a16_grouped_gemm_ptrtable_t, w4a16_grouped_gemm_ptrtable_t_k64}` (6) | [strix-hip/qwen3.6-27b/nvfp4/moe_w4a16_grouped_gemm.cu:61][f356] | expert GEMM/GEMV | hip | Qwen-GDN-MoE (6 ckpts) | [4 notes][t356] | not measured |

### Unique to Dense FFN (gate/up/down projections of non-MoE layers)

27 entry points.

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| dense_f32io::`k3_dense_{down_f32io, gate_up_situ_f32io}` (2) | [b200/kimi-k3/bf16/dense_f32io.cu:12][f1] | BF16/F32 GEMM/GEMV | b200 | Kimi-K3 (1 ckpts) | [1 note][t1] | not measured |
| q2_0_gemv_vec::`q2_0_gemv_vec_batchm` | [gb10/common/q2_0_gemv_vec.cu:162][f158] | integer / K-quant GEMM/GEMV | b200 b300 gb10 hop | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN, Step-3.7 (29 ckpts) | [1 note][t158] | not measured |
| w4a16_gemv_fused::`w4a16_gemv_silu_{input, input_sw}` (2) | [gb10/common/w4a16_gemv_fused.cu:209][f185] | NVFP4 W4A16 GEMM/GEMV | b200 b300 gb10 hop strix hip | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN, Step-3.7 (29 ckpts) | [2 notes][t185] | not measured |
| w8a16_gemv_fused::`w8a16_gemv_{dual, silu_input}` (2) | [gb10/common/w8a16_gemv_fused.cu:123][f197] | FP8 GEMM/GEMV | b200 b300 gb10 | DeepSeek-V4, GLM-5.3, Gemma4, Kimi-K3, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN, Step-3.7 (29 ckpts) | [1 note][t197] | not measured |
| w4a4::`w4a4_gemm` | [gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/w4a4_gemm.cu:115][f262] | W4A4 GEMM/GEMV | gb10 | Nemotron-H (3 ckpts) | — | not measured |
| nvfp4_mmq::`metrale_nvfp4_mmq128_nc` | [gb10/qwen3.6-27b/nvfp4/nvfp4_mmq.cu:67][f272] | W4A4 GEMM/GEMV | gb10 hop | Qwen-GDN (7 ckpts) | [3 notes][t272] | [17–18%][m272.metrale_nvfp4_mmq128_nc] (prefill 4k (cold, 4103 tok)) |
| nvfp4_mmq::`metrale_nvfp4_{mmq128_wc, mmq16_wc, mmq32_wc, mmq64_wc}` (4) | [gb10/qwen3.6-27b/nvfp4/nvfp4_mmq.cu:72][f272] | W4A4 GEMM/GEMV | gb10 hop | Qwen-GDN (7 ckpts) | [3 notes][t272] | not measured |
| nvfp4_mmq::`metrale_nvfp4_repack` | [gb10/qwen3.6-27b/nvfp4/nvfp4_mmq.cu:142][f272] | dequant / repack / transpose | gb10 hop | Qwen-GDN (7 ckpts) | [3 notes][t272] | not measured |
| nvfp4_mmq::`metrale_nvfp4_silu_mul_scaled` | [gb10/qwen3.6-27b/nvfp4/nvfp4_mmq.cu:221][f272] | activation / gate / residual | gb10 hop | Qwen-GDN (7 ckpts) | [3 notes][t272] | not measured |
| q4k_mmq::`metrale_q4k_mmq128_{nc, wc}` (2) | [gb10/qwen3.6-27b/nvfp4/q4k_mmq.cu:50][f274] | integer / K-quant GEMM/GEMV | gb10 hop | Qwen-GDN (7 ckpts) | [1 note][t274] | not measured |
| q4k_quantize::`q4k_quantize` | [gb10/qwen3.6-27b/nvfp4/q4k_quantize.cu:81][f275] | activation quantize | gb10 hop | Qwen-GDN (7 ckpts) | [1 note][t275] | not measured |
| w4a16::`int8_gemm_faith2`, `int8_gemm_i32acc`, `requant_a_bf16_int8`, `requant_w_nvfp4_int8` | [gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu:4569][f276] | integer / K-quant GEMM/GEMV | gb10 hop strix | Qwen-GDN (7 ckpts) | [2 notes][t276] | not measured |
| w4a4::`w4a4_gemm` | [gb10/qwen3.6-27b/nvfp4/w4a4_gemm.cu:50][f278] | W4A4 GEMM/GEMV | gb10 hop | Qwen-GDN (7 ckpts) | [2 notes][t278] | not measured |
| silu_mul_strided::`silu_mul_strided` | [hopper/common/silu_mul_strided.cu:44][f299] | activation / gate / residual | hop | DeepSeek-V4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE (18 ckpts) | [2 notes][t299] | not measured |
| w8a16_gemm_m16::`w8a16_gemm_m16_n64` | [hopper/common/w8a16_gemm_m16.cu:407][f300] | FP8 GEMM/GEMV | hop | DeepSeek-V4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE (18 ckpts) | [3 notes][t300] | not measured |
| w8a16_gemv_fused::`w8a16_gemv_{dual, silu_input}` (2) | [hopper/common/w8a16_gemv_fused.cu:66][f302] | FP8 GEMM/GEMV | hop | DeepSeek-V4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE (18 ckpts) | [2 notes][t302] | not measured |

### Unique to Projection GEMM/GEMV — BF16/F32

6 entry points.

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| gemm_tc::`dense_gemm_tc_scaled_acc` | [gb10/common/dense_gemm_tc.cu:197][f16] | BF16/F32 GEMM/GEMV | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [2 notes][t16] | not measured |
| dense_gemv_bf16_tc::`dense_gemv_bf16_tc16` | [gb10/common/dense_gemv_bf16_tc.cu:251][f20] | BF16/F32 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | [4 notes][t20] · [#1][pr1] | [86–95%][m20.dense_gemv_bf16_tc16] (decode C=16 (R=32, MTP k=1)) |
| dense_gemv_bf16_tc::`dense_gemv_bf16_{tc32, tc8}` (2) | [gb10/common/dense_gemv_bf16_tc.cu:250][f20] | BF16/F32 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | [4 notes][t20] · [#1][pr1] | not measured |
| dense_gemm_m16_bf16::`dense_gemm_m16_{bf16, bf16_n64}` (2) | [hopper/common/dense_gemm_m16_bf16.cu:339][f290] | BF16/F32 GEMM/GEMV | hop | DeepSeek-V4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE (18 ckpts) | [2 notes][t290] | not measured |

### Unique to Projection GEMM/GEMV — FP8 (W8A16, W8A8, block-scaled)

21 entry points.

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| dense_gemv_fp8w_batch2::`dense_gemv_fp8w_batch2` | [gb10/common/dense_gemv_fp8w_batch2.cu:72][f22] | FP8 GEMM/GEMV | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t22] | not measured |
| fp8_gemm_blockscaled_pipe::`fp8_gemm_blockscaled_pipe_128x64` | [gb10/common/fp8_gemm_blockscaled_pipe.cu:63][f30] | FP8 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | — | not measured |
| w4a16_fp8_ldmab::`fp8_fp8_gemm_ldmab` | [gb10/common/w4a16_fp8_ldmab.cu:65][f182] | FP8 GEMM/GEMV | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [2 notes][t182] | [11–43%][m182.fp8_fp8_gemm_ldmab] (prefill 32k (cold, 32772 tok)) |
| w8a16_gemm_pipe128::`w8a16_gemm_pipe128` | [gb10/common/w8a16_gemm_pipe128.cu:58][f190] | FP8 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | — | not measured |
| w8a16_tc_rows::`w8a16_tc_rows_{16, 32, 64, 64c}` (4) | [gb10/common/w8a16_tc_rows.cu:38][f198] | FP8 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t198] | not measured |
| w8a8_gemv::`w8a8_gemv_{blk128_mb16, blk128_mb1_ku8, blk128_mb2, blk128_mb4, blk128_mb8, rowscale_mb16, rowscale_mb1_ku8, rowscale_mb2, rowscale_mb4, rowscale_mb8}` (10) | [gb10/common/w8a8_gemv.cu:171][f200] | FP8 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | — | not measured |
| w4a16_v2::`w4a16_gemm_t_m128_v2` | [gb10/minimax-m2-229b/nvfp4/w4a16_gemm_v2.cu:72][f250] | FP8 GEMM/GEMV | gb10 | MiniMax-M2, Step-3.7 (2 ckpts) | [1 note][t250] | not measured |
| w4a16_v3::`w4a16_gemm_t_m128_v3` | [gb10/minimax-m2-229b/nvfp4/w4a16_gemm_v3.cu:73][f251] | FP8 GEMM/GEMV | gb10 | MiniMax-M2, Step-3.7 (2 ckpts) | [1 note][t251] | not measured |
| w4a16_v2::`w4a16_gemm_t_m128_v2` | [gb10/qwen3.6-27b/nvfp4/w4a16_gemm_v2.cu:100][f277] | FP8 GEMM/GEMV | gb10 hop | Qwen-GDN (7 ckpts) | [1 note][t277] | not measured |

### Unique to Projection GEMM/GEMV — NVFP4 W4A16

17 entry points.

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| w4a16_gemv::`w4a16_gemv_{batch8, batch8_rt2, logits}` (3) | [gb10/common/w4a16_gemv.cu:363][f184] | NVFP4 W4A16 GEMM/GEMV | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [4 notes][t184] | not measured |
| w4a16_gemv_tc::`w4a16_gemv_tc16` | [gb10/common/w4a16_gemv_tc.cu:256][f186] | NVFP4 W4A16 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | [4 notes][t186] · [#1][pr1] | [86–92%][m186.w4a16_gemv_tc16] (decode C=16 (R=32, MTP k=1)) |
| w4a16_gemv_tc::`w4a16_gemv_tc8` | [gb10/common/w4a16_gemv_tc.cu:255][f186] | NVFP4 W4A16 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | [4 notes][t186] · [#1][pr1] | [59–90%][m186.w4a16_gemv_tc8] (decode C=1 (R=4, MTP k=3)) |
| w4a16_tc_rows::`w4a16_tc_rows_{16, 32, 64}` (3) | [gb10/common/w4a16_tc_rows.cu:26][f187] | NVFP4 W4A16 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | — | not measured |
| w4a16::`w4a16_gemm_t_k64` | [gb10/deepseek-v4-flash/nvfp4/w4a16_gemm.cu:695][f225] | NVFP4 W4A16 GEMM/GEMV | b200 gb10 hop | DeepSeek-V4, Gemma4, Laguna, LongCat, MiniMax-M2, Mistral4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE, Qwen3-VL, Step-3.7 (27 ckpts) | [1 note][t225] | not measured |
| w4a16::`w4a16_gemm_t_k64` | [gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/w4a16_gemm.cu:778][f261] | NVFP4 W4A16 GEMM/GEMV | gb10 | Nemotron-H (3 ckpts) | [1 note][t261] | not measured |
| w4a16::`w4a16_gemm_t_{k64, k64_p3}` (2) | [gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu:1121][f276] | NVFP4 W4A16 GEMM/GEMV | gb10 hop strix | Qwen-GDN (7 ckpts) | [3 notes][t276] | not measured |
| w4a16::`w4a16_gemm_t_k64_n64_p3` | [gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu:1648][f276] | NVFP4 W4A16 GEMM/GEMV | gb10 hop strix | Qwen-GDN (7 ckpts) | [3 notes][t276] | [54%][m276.w4a16_gemm_t_k64_n64_p3] (decode C=16 (R=32, MTP k=1)) |
| w4a16::`w4a16_gemm_t_p3` | [gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu:585][f276] | NVFP4 W4A16 GEMM/GEMV | gb10 hop strix | Qwen-GDN (7 ckpts) | [3 notes][t276] | [8–78%][m276.w4a16_gemm_t_p3] (decode C=16 (R=32, MTP k=1)) |
| w4a16::`w4a16_gemm_t_k64` | [gb10/qwen3.6-35b-a3b/nvfp4/w4a16_gemm.cu:931][f284] | NVFP4 W4A16 GEMM/GEMV | b200 gb10 hop strix | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [3 notes][t284] · [#34][pr34] | not measured |
| w4a16::`w4a16_gemm_t_k64` | [strix-hip/qwen3.6-27b/nvfp4/w4a16_gemm.cu:570][f357] | NVFP4 W4A16 GEMM/GEMV | hip | Qwen-GDN (7 ckpts) | [2 notes][t357] | not measured |
| w4a16::`w4a16_gemm_t_k64` | [strix-hip/qwen3.6-35b-a3b/nvfp4/w4a16_gemm.cu:569][f358] | NVFP4 W4A16 GEMM/GEMV | hip | Qwen-GDN-MoE (6 ckpts) | [3 notes][t358] | not measured |

### Unique to Projection GEMM/GEMV — W4A4 (FP4 activations)

11 entry points.

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| w4a4_gemv_mx::`w4a4_{gemv_mx16, gemv_mx16_nt2, gemv_mx16_ps, gemv_mx32, gemv_mx32_nt4, gemv_mx32_ps, gemv_mx64, gemv_mx64_nt2, gemv_mx8, quant_rows}` (10) | [gb10/common/w4a4_gemv_mx.cu:360][f188] | W4A4 GEMM/GEMV | b200 gb10 hop | all 14 decoder families (30 ckpts) | [23 notes][t188] · [#1][pr1] [#14][pr14] [#18][pr18] | not measured |
| nvfp4_mmq::`metrale_nvfp4_gemm_pipe` | [gb10/qwen3.6-27b/nvfp4/nvfp4_mmq.cu:356][f272] | W4A4 GEMM/GEMV | gb10 hop | Qwen-GDN (7 ckpts) | [3 notes][t272] | not measured |

### Unique to Projection GEMM/GEMV — integer / K-quant (Q2_0, Q2_K..Q6_K, INT8, MLX INT8)

6 entry points.

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| mlx_int8_dequant::`mlx_int8_dequant` | [metal/common/mlx_int8_dequant.metal:21][f332] | integer / K-quant GEMM/GEMV | metal | Qwen-GDN (7 ckpts) | — | not measured |
| mlx_int8_gemm::`mlx_int8_gemm` | [metal/common/mlx_int8_gemm.metal:23][f333] | integer / K-quant GEMM/GEMV | metal | Qwen-GDN (7 ckpts) | [1 note][t333] | not measured |
| mlx_int8_gemv::`mlx_int8_gemv` | [metal/common/mlx_int8_gemv.metal:39][f334] | integer / K-quant GEMM/GEMV | metal | Qwen-GDN (7 ckpts) | [1 note][t334] | not measured |
| mlx_int8_gemv_gate_up::`mlx_int8_gemv_gate_up` | [metal/common/mlx_int8_gemv_gate_up.metal:41][f335] | integer / K-quant GEMM/GEMV | metal | Qwen-GDN (7 ckpts) | [1 note][t335] | not measured |
| mlx_int8_gemv_silu_gate::`mlx_int8_gemv_silu_{gate, gate_resid}` (2) | [metal/common/mlx_int8_gemv_silu_gate.metal:29][f336] | integer / K-quant GEMM/GEMV | metal | Qwen-GDN (7 ckpts) | [2 notes][t336] | not measured |

### Unique to Normalization (RMSNorm, LayerNorm, L2, gated norms)

1 entry points.

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| residual_add_rms_norm_exact::`residual_add_rms_norm_exact` | [gb10/common/residual_add_rms_norm_exact.cu:28][f167] | normalization | b200 gb10 hop | all 14 decoder families (30 ckpts) | — | not measured |

### Unique to KV cache (write, quantize, TurboQuant rotation, slot metadata)

1 entry points.

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| reshape_and_cache::`bf16_absmax` | [gb10/common/reshape_and_cache.cu:443][f163] | cache write | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t163] | not measured |

### Unique to Quantization and format conversion

8 entry points.

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| per_token_group_quant_fp8::`per_token_group_quant_fp8` | [gb10/common/per_token_group_quant_fp8.cu:39][f152] | activation quantize | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [2 notes][t152] | [81–83%][m152.per_token_group_quant_fp8] (prefill 32k (cold, 32772 tok)) |
| quant_rowwise_fp8::`quant_rowwise_fp8` | [gb10/common/quant_rowwise_fp8.cu:38][f159] | activation quantize | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t159] | not measured |
| w4a16_fp8_ldmab::`fp8_predequant_nvfp4_t` | [gb10/common/w4a16_fp8_ldmab.cu:193][f182] | dequant / repack / transpose | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [2 notes][t182] | not measured |
| w8a8_act_quant::`w8a8_act_quant_{g128, row, silu_g128, silu_row}` (4) | [gb10/common/w8a8_act_quant.cu:168][f199] | activation quantize | b200 gb10 hop | all 14 decoder families (30 ckpts) | — | not measured |
| fp8_act_quant_hopper::`per_token_group_quant_fp8_hopper` | [hopper/common/fp8_act_quant_hopper.cu:94][f291] | activation quantize | hop | DeepSeek-V4, Nemotron-H, Qwen-GDN, Qwen-GDN-MoE (18 ckpts) | [2 notes][t291] | not measured |

### Unique to Embedding and LM head (lookup, overlays, softcap, scale)

7 entry points.

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| token_overlay::`embed_overlay_routed_bf16`, `embed_rowdiff_bf16`, `lmhead_overlay_routed_bf16`, `lmhead_overlay_routed_f32` | [gb10/common/token_overlay.cu:20][f178] | embedding / LM head | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [2 notes][t178] | not measured |
| kquant_moe::`kquant_mmvq_q6_k_w` | [gb10/deepseek-v4-flash/nvfp4/kquant_moe.cu:356][f210] | integer / K-quant GEMM/GEMV | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [4 notes][t210] | not measured |
| logit_softcap::`logit_softcap_bf16` | [gb10/gemma-4-26b-a4b/nvfp4/logit_softcap.cu:12][f230] | embedding / LM head | gb10 | Gemma4 (2 ckpts) | [1 note][t230] | not measured |
| logit_softcap::`logit_softcap_bf16` | [gb10/gemma-4-31b/nvfp4/logit_softcap.cu:12][f240] | embedding / LM head | gb10 | Gemma4 (2 ckpts) | [1 note][t240] | not measured |

### Unique to Sampling (argmax, top-p, feed-forward of the chosen token)

4 entry points.

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| argmax::`argmax_fp32` | [gb10/common/argmax_bf16.cu:194][f5] | argmax / top-p | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [2 notes][t5] | not measured |
| argmax_feed::`argmax_bf16_batch_feed`, `feed_resolve` | [gb10/common/argmax_feed.cu:44][f6] | argmax / top-p | b200 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t6] | not measured |
| argmax_bf16::`argmax_bf16` | [metal/common/argmax_bf16.metal:22][f305] | argmax / top-p | metal | Qwen-GDN (7 ckpts) | [1 note][t305] | not measured |

### Unique to Speculative decoding (MTP heads, DFlash drafter, verify helpers)

12 entry points.

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| attn_prefill_h128::`attn_prefill_h128` | [gb10/common/attn_prefill_h128.cu:46][f10] | prefill (flash) | b200 b300 gb10 hop strix | all 14 decoder families (30 ckpts) | [1 note][t10] | not measured |
| dflash2::`dflash2_{conv2, selector_walk, topk16}` (3) | [gb10/common/dflash2.cu:31][f26] | DFlash drafter | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [3 notes][t26] | not measured |
| prefill_paged_indirect::`attn_prefill_paged_indirect` | [gb10/common/prefill_paged_compute.cuh:162][f153] | prefill (flash) | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [6 notes][t153] | not measured |
| w4a16::`fp8_gemm_t_row_{scaled, scaled_k64, scaled_m16, scaled_p4}` (4) | [gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu:6843][f276] | FP8 GEMM/GEMV | gb10 hop strix | Qwen-GDN (7 ckpts) | [4 notes][t276] | not measured |
| w4a16::`fp8_gemm_t_row_{scaled, scaled_m16}` (2) | [gb10/qwen3.6-35b-a3b/nvfp4/w4a16_gemm.cu:1748][f284] | FP8 GEMM/GEMV | b200 gb10 hop strix | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [2 notes][t284] · [#34][pr34] | not measured |
| attn_prefill_h128::`attn_prefill_h128` | [strix-hip/common/attn_prefill_h128.cu:185][f349] | prefill (flash) | hip | Qwen-GDN, Qwen-GDN-MoE (13 ckpts) | [1 note][t349] | not measured |

### Unique to Hyper-connections (mHC)

13 entry points.

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| glm5next_mhc::`glm5next_hc_{expand, finish, head, mix, mix_bf16, post, pre}` (7) | [gb10/common/glm5next_mhc.cu:64][f57] | hyper-connection mix | b200 b300 gb10 hop | GLM-5.3 (1 ckpts) | [4 notes][t57] | not measured |
| hyper_connection::`hc_pre_down`, `hc_pre_finish`, `hc_pre_mix`, `hc_pre_stage`, `hc_pre_stage_bf16`, `hc_silu_scale` | [gb10/qwen3.8-flash-next/nvfp4/hyper_connection.cu:329][f286] | hyper-connection mix | gb10 | Qwen3.8-FN (1 ckpts) | [2 notes][t286] | not measured |

### Unique to N-gram and memory embeddings (Engram, PLE, n-gram tables)

6 entry points.

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| embed_from_argmax::`batched_embed_fp8` | [gb10/common/embed_from_argmax.cu:110][f29] | embedding / LM head | b200 b300 gb10 hop strix hip | LongCat (1 ckpts) | [2 notes][t29] | not measured |
| engram_v41::`engram_v41_{gate, wkv_q2k_gemv}` (2) | [gb10/deepseek-v4-flash/nvfp4/engram_v41.cu:43][f206] | memory embedding | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [3 notes][t206] | not measured |
| ple::`ple_{add_highway, conv, gate}` (3) | [gb10/qwen3.8-flash-next/nvfp4/ple.cu:90][f287] | memory embedding | gb10 | Qwen3.8-FN (1 ckpts) | [2 notes][t287] | not measured |

### Unique to Vision encoder (ViT towers)

32 entry points.

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| glm_vit::`glm_vit_{add_bias, add_inplace, copy, f32_to_bf16, gelu_erf, im2col_2x2, layernorm, qknorm_rope_deint, rmsnorm, scatter_head, softmax_rows, swiglu_clamp}` (12) | [gb10/glm-5.3-flash/nvfp4/glm_vit.cu:42][f244] | ViT op | gb10 | GLM-5.3 (1 ckpts) | [2 notes][t244] | not measured |
| vision_encoder::`vision_{add_inplace, attention_rope, bf16_copy, f32_to_bf16, gelu, gemm_bias, layer_norm, spatial_merge}` (8) | [gb10/qwen3-vl-30b-a3b/nvfp4/vision_encoder.cu:23][f265] | ViT op | gb10 | Qwen-GDN-MoE, Qwen3-VL (7 ckpts) | [2 notes][t265] | not measured |
| vision_encoder::`vision_add_bias`, `vision_add_inplace`, `vision_attention_rope`, `vision_bf16_copy`, `vision_f32_to_bf16`, `vision_gelu`, `vision_gemm_bias`, `vision_layer_norm`, `vision_spatial_merge`, `vit_rope_deinterleave`, `vit_scatter_head`, `vit_softmax_rows` | [gb10/qwen3.6-35b-a3b/nvfp4/vision_encoder.cu:23][f283] | ViT op | b200 gb10 hop strix hip | Qwen-GDN, Qwen-GDN-MoE, Qwen3.8-FN (14 ckpts) | [3 notes][t283] | not measured |

### Unique to Encoder-decoder translation (NLLB, self-contained kernel set)

24 entry points.

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| nllb_encoder::`nllb_{add_bf16, add_row_bf16, attn_bdecode, attn_kv_bf16, beam_topk, bias_bf16, embed_bf16, gather_batched, gemv_bf16, layernorm_oop_bf16, relu_bf16, scale_bf16, scatter_batched}` (13) | [gb10/common/nllb_encoder.cu:203][f118] | NLLB encoder/decoder op | b200 b300 gb10 hop | NLLB (1 ckpts) | [3 notes][t118] | not measured |
| nllb_encoder::`nllb_{add_bf16, add_row_bf16, attn_bdecode, attn_kv_bf16, bias_bf16, embed_bf16, gather_batched, gemv_bf16, relu_bf16, scale_bf16, scatter_batched}` (11) | [metal/common/nllb_encoder.metal:281][f337] | NLLB encoder/decoder op | metal | NLLB (1 ckpts) | [1 note][t337] | not measured |

### Unique to LoRA adapters (BGMV shrink/expand)

6 entry points.

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| lora_bgmv::`lora_bgmv_{expand_fold, shrink}` (2) | [gb10/common/lora_bgmv.cu:50][f66] | BGMV shrink/expand | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t66] | not measured |
| moe_lora_gather_bgmv::`moe_lora_gather_bgmv_{expand_fold, shrink}` (2) | [gb10/common/moe_lora_gather_bgmv.cu:57][f83] | BGMV shrink/expand | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t83] | not measured |
| moe_lora_grouped_down::`moe_lora_grouped_down_{expand_fold, shrink}` (2) | [gb10/common/moe_lora_grouped_down.cu:67][f84] | BGMV shrink/expand | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t84] | not measured |

### Unique to Weight load and repack (one-time, not per token)

25 entry points.

| Kernel (module::function) | File | Kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| dense_gemv_bf16_batchm::`dense_gemv_bf16_batchm_fp32out` | [gb10/common/dense_gemv_bf16_batchm.cu:233][f19] | BF16/F32 GEMM/GEMV | b200 b300 gb10 hop | GPT-OSS (1 ckpts) | [2 notes][t19] | not measured |
| dequant_gguf_bf16::`dequant_{q3_k_to_bf16, q4_k_to_bf16, q6_k_to_bf16, q8_0_to_bf16}` (4) | [gb10/common/dequant_gguf_bf16.cu:43][f24] | dequant / repack / transpose | b200 b300 gb10 hop | all 14 decoder families (30 ckpts) | [1 note][t24] | not measured |
| gpt_oss_expert_ops::`gpt_oss_{expert_reduce_bf16, selected_bias_bf16, swiglu_bf16}` (3) | [gb10/common/gpt_oss_expert_ops.cu:15][f58] | staged BF16 expert activation / reduction | b200 gb10 hop | GPT-OSS (1 ckpts) | — | not measured |
| gpt_oss_mxfp4_gemv::`gpt_oss_mxfp4_selected_bf16` | [gb10/common/gpt_oss_mxfp4_gemv.cu:54][f59] | row-major MXFP4 expert GEMV (correctness residual) | b200 gb10 hop | GPT-OSS (1 ckpts) | — | not measured |
| gpt_oss_rope::`gpt_oss_{rope_bf16, yarn_frequencies}` (2) | [gb10/common/gpt_oss_rope.cu:7][f60] | continuous YaRN / staged BF16 rotation | b200 gb10 hop | GPT-OSS (1 ckpts) | — | not measured |
| moe_topk::`moe_topk_selected_bf16_rows` | [gb10/common/moe_topk.cu:497][f108] | routing / top-k | b200 b300 gb10 hop strix hip | GPT-OSS (1 ckpts) | [1 note][t108] | not measured |
| paged_decode::`paged_decode_attn_sink` | [gb10/common/paged_decode_attn.cu:363][f119] | paged decode | b200 b300 gb10 hop strix hip | GPT-OSS (1 ckpts) | [1 note][t119] | not measured |
| projection_bias::`projection_bias_bf16` | [gb10/common/projection_bias.cu:14][f156] | FP32 projection bias / BF16 store | b200 gb10 hop | GPT-OSS (1 ckpts) | — | not measured |
| quantize_bf16_to_fp8_blockscaled::`quantize_bf16_to_fp8_blockscaled` | [gb10/common/quantize_bf16_to_fp8_blockscaled.cu:54][f160] | activation quantize | b200 b300 gb10 hop | LongCat (1 ckpts) | [1 note][t160] | not measured |
| quantize_nvfp4::`f32_to_bf16_trunc` | [gb10/common/quantize_bf16_to_nvfp4.cu:29][f161] | dtype conversion | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t161] | not measured |
| quantize_nvfp4::`quantize_bf16_to_nvfp4_mse` | [gb10/common/quantize_bf16_to_nvfp4.cu:265][f161] | activation quantize | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [3 notes][t161] · [#34][pr34] | not measured |
| transpose_u8::`transpose_u8` | [gb10/common/transpose_u8.cu:15][f180] | dequant / repack / transpose | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | — | not measured |
| widen_block_scale_f32::`widen_block_scale_f32` | [gb10/common/widen_block_scale_f32.cu:21][f202] | dequant / repack / transpose | b200 b300 gb10 hop strix hip | all 14 decoder families (30 ckpts) | [1 note][t202] | not measured |
| hc_v41::`hc_v41_{collapse, collapse_wide, finish_collapse, mixes_dot, mixes_finish, post_wide}` (6) | [gb10/deepseek-v4-flash/nvfp4/hc_v41.cu:118][f208] | hyper-connection mix | b200 gb10 hop | DeepSeek-V4 (2 ckpts) | [3 notes][t208] | not measured |

## Compiled but not launched

No engine call site names these entry points: they are reached only from tests or `examples/` (microbenchmarks, the Metal Qwen3.5 driver), or not at all. They cost build time and are candidates for removal or for wiring up.

| Kernel (module::function) | File | Component · kind | HW | LLMs | Trade-offs · PRs | % of floor |
|---|---|---|---|---|---|---|
| prefill_fp8kv::`attn_prefill_fp8kv_64` | [gb10/common/attn_prefill_fp8kv.cu:94][f9] | Attention · prefill (flash) | b200 b300 gb10 hop strix | — | [1 note][t9] | not measured |
| attn_prefill_h128::`attn_prefill_h128_64` | [gb10/common/attn_prefill_h128.cu:509][f10] | Attention · prefill (flash) | b200 b300 gb10 hop strix | — | [1 note][t10] | not measured |
| attn_prefill_v47::`attn_prefill_v47` | [gb10/common/attn_prefill_v47.cu:30][f11] | Attention · prefill (flash) | b200 b300 gb10 hop strix | — | [1 note][t11] | not measured |
| causal_conv1d::`causal_conv1d_{fwd, update_f32}` (2) | [gb10/common/causal_conv1d.cu:30][f13] | Causal conv1d · causal conv1d | b200 b300 gb10 hop strix hip | — | [1 note][t13] | not measured |
| gemm::`fused_silu_mul` | [gb10/common/dense_gemm_bf16.cu:590][f14] | Activations and elementwise · activation / gate / residual | b200 b300 gb10 hop strix | — | — | not measured |
| e2m1::`e2m1_quantize` | [gb10/common/e2m1_branchless.cu:48][f28] | Quantization and format conversion · activation quantize | b200 b300 gb10 hop strix hip | — | [1 note][t28] | not measured |
| embed_from_argmax::`batched_embed_f32`, `embed_from_argmax_f32` | [gb10/common/embed_from_argmax.cu:57][f29] | Embedding and LM head · embedding / LM head | b200 b300 gb10 hop strix hip | — | [1 note][t29] | not measured |
| fused_k_norm_rope_cache::`fused_k_norm_rope_cache_write_fp8` | [gb10/common/fused_k_norm_rope_cache.cu:252][f34] | KV cache · cache write | b200 b300 gb10 hop | — | [1 note][t34] | not measured |
| gated_delta_rule_fla::`gated_delta_rule_chunk_delta_{h, h_dvsplit, h_ksplit_vblock2, h_ksplit_vblock4, h_ksplit_vblock8, h_tc}` (6) | [gb10/common/gated_delta_rule_fla.cu:532][f37] | GDN · delta-rule recurrence | b200 b300 gb10 hop | — | [10 notes][t37] | not measured |
| gated_delta_rule_persistent::`gated_delta_rule_prefill_persistent_{multihead, regtile}` (2) | [gb10/common/gated_delta_rule_persistent.cu:498][f38] | GDN · delta-rule recurrence | b200 b300 gb10 hop strix hip | — | [3 notes][t38] | not measured |
| glm5next_ffn::`glm5next_swiglu_clamp_f32out` | [gb10/common/glm5next_ffn.cu:50][f56] | MoE · expert activation | b200 b300 gb10 hop | — | [1 note][t56] | not measured |
| glm5next_mhc::`glm5next_hc_post_ref` | [gb10/common/glm5next_mhc.cu:497][f57] | Hyper-connections · hyper-connection mix | b200 b300 gb10 hop | — | [2 notes][t57] | not measured |
| gpt_oss_mxfp4_gemv::`gpt_oss_mxfp4_gemv_bf16` | [gb10/common/gpt_oss_mxfp4_gemv.cu:43][f59] | MoE · row-major MXFP4 expert GEMV (correctness residual) | b200 gb10 hop | — | — | not measured |
| gpt_oss_staged_attention::`gpt_oss_staged_attention_bf16` | [gb10/common/gpt_oss_staged_attention.cu:14][f61] | Attention · staged BF16 sink attention (correctness residual) | b200 gb10 hop | — | — | not measured |
| kda_gate::`kda_gate_f32` | [gb10/common/kda_gate.cu:101][f63] | KDA · KDA op | b200 b300 gb10 hop | — | — | not measured |
| kda_layer_ops::`kda_o_norm_gated_f32` | [gb10/common/kda_layer_ops.cu:66][f64] | KDA · KDA op | b200 b300 gb10 hop | — | — | not measured |
| kda_recurrent::`kda_recurrent_decode_f32` | [gb10/common/kda_recurrent.cu:125][f65] | KDA · KDA op | b200 b300 gb10 hop | — | [1 note][t65] | not measured |
| moe_expert_gemv::`moe_weighted_sum` | [gb10/common/moe_expert_gemv.cu:164][f73] | MoE · dispatch / combine | b200 b300 gb10 hop strix hip | — | [2 notes][t73] | not measured |
| moe_expert_gemv_fused::`moe_expert_gemv_{gate_up, gate_up_2x, silu_down, silu_down_2x, silu_down_wide}` (5) | [gb10/common/moe_expert_gemv_fused.cu:53][f74] | MoE · expert GEMM/GEMV | b200 b300 gb10 hop strix hip | — | [1 note][t74] | not measured |
| moe::`moe_{count_experts, unpermute_reduce}` (2) | [gb10/common/moe_permute.cu:45][f87] | MoE · dispatch / combine | b200 b300 gb10 hop strix hip | — | [1 note][t87] | not measured |
| moe_w4a16::`moe_w4a16_grouped_{gemm, gemm_ptrtable_al_k64, gemm_ptrtable_al_m16_k128, gemm_ptrtable_al_m16_k32, gemm_ptrtable_al_m16_k64, gemm_ptrtable_al_m16_n128_k64, gemm_ptrtable_alkm_k128, gemm_ptrtable_alkm_k64, gemm_ptrtable_alkm_m16_k256, gemm_ptrtable_alkm_m16_k32, gemm_ptrtable_alkm_m16_k64, gemm_ptrtable_alkm_m16_n128_k128, gemm_ptrtable_alkm_m16_n128_k64, gemm_ptrtable_bt_k128, gemm_ptrtable_bt_m16_k256, gemm_ptrtable_bt_m16_k64, gemm_ptrtable_k128, gemm_ptrtable_km_k64, gemm_ptrtable_km_m16_k128, gemm_ptrtable_km_m16_k32, gemm_ptrtable_km_m16_k64, gemm_ptrtable_km_m16_n128_k128, gemm_ptrtable_km_m16_n128_k64, gemm_ptrtable_m16_k128, gemm_ptrtable_m16_k256, gemm_ptrtable_m16_k32, gemm_ptrtable_m16_n128_k128, gemm_ptrtable_m16_n128_k32, gemm_ptrtable_m16_n128_k64}` (29) | [gb10/common/moe_w4a16_grouped_gemm.cu:37][f114] | MoE · expert GEMM/GEMV | b200 gb10 hop | — | [4 notes][t114] | not measured |
| moe_w4a16::`moe_w4a16_grouped_stream_probe` | [gb10/common/moe_w4a16_grouped_gemm.cu:978][f114] | Diagnostics and microtests · microtest / smoke | b200 gb10 hop | — | [5 notes][t114] | not measured |
| nllb_encoder::`nllb_{add_inplace, argmax_batched, attention, attn_kv, embed, layernorm, linear, relu_inplace, scale_inplace}` (9) | [gb10/common/nllb_encoder.cu:18][f118] | Encoder-decoder translation · NLLB encoder/decoder op | b200 b300 gb10 hop | — | [1 note][t118] | not measured |
| paged_decode::`paged_decode_attn_{reduce, splitk}` (2) | [gb10/common/paged_decode_attn.cu:388][f119] | Attention · paged decode | b200 b300 gb10 hop strix hip | — | [2 notes][t119] | not measured |
| paged_decode_attn_turbo3::`paged_decode_attn_reduce_nvfp4` | [gb10/common/paged_decode_attn_turbo3.cu:561][f138] | Attention · paged decode | b200 b300 gb10 hop strix hip | — | [2 notes][t138] | not measured |
| paged_decode_attn_turbo3_128::`paged_decode_attn_reduce_nvfp4` | [gb10/common/paged_decode_attn_turbo3_128.cu:551][f139] | Attention · paged decode | b200 b300 gb10 hop strix hip | — | [1 note][t139] | not measured |
| paged_decode_attn_turbo4::`paged_decode_attn_reduce_nvfp4` | [gb10/common/paged_decode_attn_turbo4.cu:536][f142] | Attention · paged decode | b200 b300 gb10 hop strix hip | — | [1 note][t142] | not measured |
| paged_decode_attn_turbo4_128::`paged_decode_attn_reduce_nvfp4` | [gb10/common/paged_decode_attn_turbo4_128.cu:536][f143] | Attention · paged decode | b200 b300 gb10 hop strix hip | — | [1 note][t143] | not measured |
| paged_decode_attn_turbo4_512::`paged_decode_attn_reduce_nvfp4` | [gb10/common/paged_decode_attn_turbo4_512.cu:560][f144] | Attention · paged decode | b200 b300 gb10 hop strix hip | — | [2 notes][t144] | not measured |
| paged_decode_attn_turbo8::`paged_decode_attn_reduce_nvfp4` | [gb10/common/paged_decode_attn_turbo8.cu:543][f149] | Attention · paged decode | b200 b300 gb10 hop strix hip | — | [1 note][t149] | not measured |
| paged_decode_attn_turbo8_128::`paged_decode_attn_reduce_nvfp4` | [gb10/common/paged_decode_attn_turbo8_128.cu:543][f150] | Attention · paged decode | b200 b300 gb10 hop strix hip | — | [1 note][t150] | not measured |
| paged_decode_attn_turbo8_512::`paged_decode_attn_reduce_nvfp4` | [gb10/common/paged_decode_attn_turbo8_512.cu:566][f151] | Attention · paged decode | b200 b300 gb10 hop strix hip | — | [2 notes][t151] | not measured |
| prefill_paged_indirect::`attn_prefill_paged_{indirect_64, turbo2_64, turbo3, turbo8}` (4) | [gb10/common/prefill_paged_compute.cuh:162][f153] | Attention · prefill (flash) | b200 b300 gb10 hop | — | [5 notes][t153] | not measured |
| prefill_paged_bf16k_turbo2v::`attn_prefill_paged_{bf16k_turbo2v, bf16k_turbo3v, bf16k_turbo4v, fp8k_turbo2v, fp8k_turbo3v, fp8k_turbo4v, turbo3k_turbo8v, turbo4k_turbo3v, turbo4k_turbo8v}` (9) | [gb10/common/prefill_paged_compute_asym.cuh:99][f155] | Attention · prefill (flash) | b200 b300 gb10 hop | — | [2 notes][t155] | not measured |
| q2_0_gemv::`q2_0_{gemv, gemv_batchm}` (2) | [gb10/common/q2_0_gemv.cu:46][f157] | Projection GEMM/GEMV — integer / K-quant · integer / K-quant GEMM/GEMV | b200 b300 gb10 hop | — | [1 note][t157] | not measured |
| relu2::`bias_add_bf16_f32`, `relu_squared` | [gb10/common/relu_squared.cu:12][f162] | Activations and elementwise · activation / gate / residual | b200 b300 gb10 hop strix hip | — | — | not measured |
| relu2::`convert_f32_to_bf16` | [gb10/common/relu_squared.cu:80][f162] | Quantization and format conversion · dtype conversion | b200 b300 gb10 hop strix hip | — | — | not measured |
| reshape_and_cache::`bf16_absmax_per_head` | [gb10/common/reshape_and_cache.cu:395][f163] | KV cache · cache write | b200 b300 gb10 hop strix hip | — | [1 note][t163] | not measured |
| residual_add::`bf16_sigmoid_blend`, `bf16_sigmoid_blend_device`, `silu_mul_separate` | [gb10/common/residual_add.cu:41][f166] | Activations and elementwise · activation / gate / residual | b200 b300 gb10 hop strix hip | — | [1 note][t166] | not measured |
| residual_add::`bf16_to_f32` | [gb10/common/residual_add.cu:25][f166] | Quantization and format conversion · dtype conversion | b200 b300 gb10 hop strix hip | — | — | not measured |
| norm::`f32_residual_add` | [gb10/common/rms_norm.cu:987][f168] | Activations and elementwise · activation / gate / residual | b200 b300 gb10 hop strix hip | — | [2 notes][t168] | not measured |
| norm::`residual_add_rms_norm_f32`, `residual_add_rms_norm_f32_abs`, `rms_norm_f32`, `rms_norm_f32_in_abs`, `rms_norm_residual_f32`, `rms_norm_residual_f32_abs` | [gb10/common/rms_norm.cu:199][f168] | Normalization · normalization | b200 b300 gb10 hop strix hip | — | [2 notes][t168] | not measured |
| rms_norm_act_quant::`rms_norm_quant_{fp8_g128, fp8_row, nvfp4}` (3) | [gb10/common/rms_norm_act_quant.cu:98][f169] | Normalization · normalization | b200 gb10 hop | — | — | not measured |
| ssm_ba_gates_tiled::`dense_gemm_ba_gates_prefill_tiled` | [gb10/common/ssm_ba_gates_tiled.cu:23][f174] | Projection GEMM/GEMV — BF16/F32 · BF16/F32 GEMM/GEMV | b200 gb10 hop | — | — | not measured |
| vector_add::`vector_add` | [gb10/common/vector_add.cu:6][f181] | Activations and elementwise · activation / gate / residual | b200 b300 gb10 hop strix hip | — | — | not measured |
| w4a16::`w4a16_dequant` | [gb10/common/w4a16_gemm.cu:292][f183] | Quantization and format conversion · dequant / repack / transpose | b200 b300 gb10 | — | [1 note][t183] | not measured |
| w4a16_gemv::`w4a16_gemv_{batch4, batch5, batch6, batch7, batch8_pf, batch8_pf2, batch8_pf3, batch8_pf_free, batch8_rt4}` (9) | [gb10/common/w4a16_gemv.cu:720][f184] | Projection GEMM/GEMV — NVFP4 W4A16 · NVFP4 W4A16 GEMM/GEMV | b200 b300 gb10 hop strix hip | — | [3 notes][t184] | not measured |
| w8a16_gemm::`w8a16_dequant` | [gb10/common/w8a16_gemm.cu:224][f189] | Quantization and format conversion · dequant / repack / transpose | b200 b300 gb10 hop strix | — | [1 note][t189] | not measured |
| hc_v41::`hc_v41_mixes` | [gb10/deepseek-v4-flash/nvfp4/hc_v41.cu:46][f208] | Hyper-connections · hyper-connection mix | b200 gb10 hop | — | [2 notes][t208] | not measured |
| kquant_moe::`kquant_mmvq_{q2_k, q3_k}` (2) | [gb10/deepseek-v4-flash/nvfp4/kquant_moe.cu:210][f210] | Projection GEMM/GEMV — integer / K-quant · integer / K-quant GEMM/GEMV | b200 gb10 hop | — | [4 notes][t210] | not measured |
| kquant_moe::`kquant_mmvq_{q2_k_experts, q2_k_experts_w, q3_k_experts, q3_k_experts_w}` (4) | [gb10/deepseek-v4-flash/nvfp4/kquant_moe.cu:231][f210] | MoE · expert GEMM/GEMV | b200 gb10 hop | — | [4 notes][t210] | not measured |
| mla_cache_assemble_fp8::`mla_cache_assemble_fp8_batched` | [gb10/deepseek-v4-flash/nvfp4/mla_cache_assemble_fp8.cu:32][f212] | MLA · MLA decode/prefill | b200 gb10 hop | — | [1 note][t212] | not measured |
| moe_v41::`moe_v41_router_gemv_f32out_staged` | [gb10/deepseek-v4-flash/nvfp4/moe_v41.cu:171][f218] | MoE · routing / top-k | b200 gb10 hop | — | [1 note][t218] | not measured |
| paged_decode_attn_512::`paged_decode_attn_{reduce, splitk}` (2) | [gb10/deepseek-v4-flash/nvfp4/paged_decode_attn_512.cu:305][f220] | Attention · paged decode | b200 gb10 hop | — | [1 note][t220] | not measured |
| paged_decode_fp8_mla::`paged_decode_attn_reduce_fp8` | [gb10/deepseek-v4-flash/nvfp4/paged_decode_attn_fp8_mla.cu:507][f221] | MLA · MLA decode/prefill | b200 gb10 hop | — | [1 note][t221] | not measured |
| paged_decode_mla::`paged_decode_attn_{reduce, splitk}` (2) | [gb10/deepseek-v4-flash/nvfp4/paged_decode_attn_mla.cu:316][f222] | MLA · MLA decode/prefill | b200 gb10 hop | — | [1 note][t222] | not measured |
| embed_scale::`f32_scale_inplace` | [gb10/gemma-4-26b-a4b/nvfp4/embed_scale.cu:25][f227] | Embedding and LM head · embedding / LM head | gb10 | — | — | not measured |
| gelu::`gelu_tanh` | [gb10/gemma-4-26b-a4b/nvfp4/gelu.cu:21][f229] | Activations and elementwise · activation / gate / residual | gb10 | — | [1 note][t229] | not measured |
| paged_decode_attn_512::`paged_decode_attn_{reduce, splitk}` (2) | [gb10/gemma-4-26b-a4b/nvfp4/paged_decode_attn_512.cu:305][f235] | Attention · paged decode | gb10 | — | [1 note][t235] | not measured |
| paged_decode_attn_fp8_512::`paged_decode_attn_reduce_fp8` | [gb10/gemma-4-26b-a4b/nvfp4/paged_decode_attn_fp8_512.cu:487][f236] | Attention · paged decode | gb10 | — | [1 note][t236] | not measured |
| norm::`f32_residual_add` | [gb10/gemma-4-26b-a4b/nvfp4/rms_norm.cu:471][f237] | Activations and elementwise · activation / gate / residual | gb10 | — | [1 note][t237] | not measured |
| norm::`residual_add_rms_norm_f32`, `residual_add_rms_norm_f32_abs`, `rms_norm_f32`, `rms_norm_f32_in_abs`, `rms_norm_residual_f32`, `rms_norm_residual_f32_abs` | [gb10/gemma-4-26b-a4b/nvfp4/rms_norm.cu:282][f237] | Normalization · normalization | gb10 | — | [1 note][t237] | not measured |
| embed_scale::`f32_scale_inplace` | [gb10/gemma-4-31b/nvfp4/embed_scale.cu:28][f239] | Embedding and LM head · embedding / LM head | gb10 | — | [1 note][t239] | not measured |
| logit_softcap::`logit_softcap_fp32` | [gb10/gemma-4-31b/nvfp4/logit_softcap.cu:31][f240] | Embedding and LM head · embedding / LM head | gb10 | — | [1 note][t240] | not measured |
| norm::`f32_residual_add` | [gb10/gemma-4-31b/nvfp4/rms_norm.cu:487][f241] | Activations and elementwise · activation / gate / residual | gb10 | — | [1 note][t241] | not measured |
| norm::`residual_add_rms_norm_f32`, `residual_add_rms_norm_f32_abs`, `rms_norm_f32`, `rms_norm_f32_in_abs`, `rms_norm_residual_f32`, `rms_norm_residual_f32_abs` | [gb10/gemma-4-31b/nvfp4/rms_norm.cu:294][f241] | Normalization · normalization | gb10 | — | [1 note][t241] | not measured |
| fp4_mma_microtest::`fp4_microtest_{mma, pack}` (2) | [gb10/holo-3.1-0.8b/nvfp4/fp4_mma_microtest.cu:87][f245] | Diagnostics and microtests · microtest / smoke | gb10 | — | [1 note][t245] | not measured |
| norm::`f32_residual_add` | [gb10/minimax-m2-229b/nvfp4/rms_norm.cu:494][f249] | Activations and elementwise · activation / gate / residual | b200 gb10 hop | — | [1 note][t249] | not measured |
| norm::`residual_add_rms_norm_f32`, `residual_add_rms_norm_f32_abs`, `rms_norm_f32`, `rms_norm_f32_in_abs`, `rms_norm_residual_f32`, `rms_norm_residual_f32_abs` | [gb10/minimax-m2-229b/nvfp4/rms_norm.cu:684][f249] | Normalization · normalization | b200 gb10 hop | — | [1 note][t249] | not measured |
| paged_decode_attn_fp8_mla::`paged_decode_attn_reduce_fp8` | [gb10/mistral-small-4/nvfp4/paged_decode_attn_fp8_mla.cu:507][f255] | MLA · MLA decode/prefill | gb10 | — | [1 note][t255] | not measured |
| paged_decode_mla::`paged_decode_attn_{reduce, splitk}` (2) | [gb10/mistral-small-4/nvfp4/paged_decode_attn_mla.cu:316][f256] | MLA · MLA decode/prefill | gb10 | — | [1 note][t256] | not measured |
| moe_w4a16::`moe_w4a16_grouped_gemm` | [gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/moe_w4a16_grouped_gemm.cu:91][f258] | MoE · expert GEMM/GEMV | gb10 | — | — | not measured |
| norm::`f32_residual_add` | [gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/rms_norm.cu:502][f260] | Activations and elementwise · activation / gate / residual | b200 gb10 hop | — | [1 note][t260] | not measured |
| norm::`residual_add_rms_norm_f32`, `residual_add_rms_norm_f32_abs`, `rms_norm_f32`, `rms_norm_f32_in_abs`, `rms_norm_residual_f32`, `rms_norm_residual_f32_abs` | [gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/rms_norm.cu:692][f260] | Normalization · normalization | b200 gb10 hop | — | [1 note][t260] | not measured |
| w4a16::`fp8_gemm_t_mfast` | [gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/w4a16_gemm.cu:565][f261] | Projection GEMM/GEMV — FP8 · FP8 GEMM/GEMV | gb10 | — | [2 notes][t261] | not measured |
| norm::`f32_residual_add` | [gb10/qwen3-vl-30b-a3b/nvfp4/rms_norm.cu:531][f264] | Activations and elementwise · activation / gate / residual | gb10 | — | [1 note][t264] | not measured |
| norm::`residual_add_rms_norm_f32`, `residual_add_rms_norm_f32_abs`, `rms_norm_f32`, `rms_norm_f32_in_abs`, `rms_norm_residual_f32`, `rms_norm_residual_f32_abs` | [gb10/qwen3-vl-30b-a3b/nvfp4/rms_norm.cu:644][f264] | Normalization · normalization | gb10 | — | [1 note][t264] | not measured |
| vision_encoder::`vision_gemm_bias_nn` | [gb10/qwen3-vl-30b-a3b/nvfp4/vision_encoder.cu:45][f265] | Vision encoder · ViT op | gb10 | — | [1 note][t265] | not measured |
| q4k_mmq::`metrale_q8_1_quantize_ds4` | [gb10/qwen3.6-27b/nvfp4/q4k_mmq.cu:63][f274] | Quantization and format conversion · activation quantize | gb10 hop | — | [1 note][t274] | not measured |
| w4a16::`int8_gemm_8w`, `int8_gemm_8w3`, `int8_gemm_8w_ilp`, `int8_gemm_8w_ldm`, `int8_gemm_8w_ldmab`, `int8_gemm_8w_pipe`, `int8_gemm_faith`, `int8_gemm_faith10`, `int8_gemm_faith3`, `int8_gemm_faith4`, `int8_gemm_faith5`, `int8_gemm_faith6`, `int8_gemm_faith7`, `int8_gemm_faith8`, `int8_gemm_faith9`, `int8_gemm_mmq`, `int8_gemm_mmq2`, `int8_gemm_mmqf`, `int8_gemm_mmqf2`, `int8_gemm_mmqf3`, `int8_gemm_padA`, `int8_gemm_splitk`, `int8_gemm_t_m128`, `int8_gemm_t_m128_k64`, `int8_gemm_t_m64`, `int8_splitk_reduce`, `requant_a_bf16_int8_il` | [gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu:3043][f276] | Projection GEMM/GEMV — integer / K-quant · integer / K-quant GEMM/GEMV | gb10 hop strix | — | [4 notes][t276] | not measured |
| w4a16::`w4a16_gemm_t_m64_bf16` | [gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu:2584][f276] | Projection GEMM/GEMV — NVFP4 W4A16 · NVFP4 W4A16 GEMM/GEMV | gb10 hop strix | — | [2 notes][t276] | not measured |
| vision_encoder::`vision_gemm_bias_nn` | [gb10/qwen3.6-35b-a3b/nvfp4/vision_encoder.cu:45][f283] | Vision encoder · ViT op | b200 gb10 hop strix hip | — | [1 note][t283] | not measured |
| gated_delta_rule_chunk_tc::`gated_delta_rule_chunk_delta_h_tcfuse` | [hopper/common/gated_delta_rule_chunk_tc.cu:408][f292] | GDN · delta-rule recurrence | hop | — | [3 notes][t292] | not measured |
| attention_full::`attention_full` | [metal/common/attention_full.metal:22][f312] | Attention · prefill (flash) | metal | — | [1 note][t312] | not measured |
| attention_prefill::`attention_prefill` | [metal/common/attention_prefill.metal:27][f313] | Attention · prefill (flash) | metal | — | [1 note][t313] | not measured |
| causal_conv1d_decode::`causal_conv1d_decode` | [metal/common/causal_conv1d_decode.metal:28][f315] | Causal conv1d · causal conv1d | metal | — | [1 note][t315] | not measured |
| conv3d_patch_embed::`conv3d_patch_embed` | [metal/common/conv3d_patch_embed.metal:24][f317] | Vision encoder · ViT op | metal | — | [1 note][t317] | not measured |
| embed_lookup::`embed_lookup` | [metal/common/embed_lookup.metal:17][f320] | Embedding and LM head · embedding / LM head | metal | — | [1 note][t320] | not measured |
| gdn_helpers::`bf16_mul`, `silu_apply` | [metal/common/gdn_helpers.metal:79][f322] | GDN · GDN helper | metal | — | — | not measured |
| layer_norm::`layer_norm` | [metal/common/layer_norm.metal:27][f330] | Normalization · normalization | metal | — | [1 note][t330] | not measured |
| lora_bgmv::`lora_bgmv_{expand_fold_stub, shrink_stub}` (2) | [metal/common/lora_bgmv.metal:12][f331] | Diagnostics and microtests · microtest / smoke | metal | — | [1 note][t331] | not measured |
| nllb_encoder::`nllb_{add_inplace, add_position_bf16, argmax_batched, argmax_bf16_rows, attention, attn_kv, attn_kv_batched_bf16, cache_write_bf16, embed, gemv_batched_bf16, gemv_bf16_no_bias, layernorm, linear, linear_bf16, linear_no_bias, linear_no_bias_bf16, relu_inplace, scale_inplace, topk_lse_bf16}` (19) | [metal/common/nllb_encoder.metal:13][f337] | Encoder-decoder translation · NLLB encoder/decoder op | metal | — | [2 notes][t337] | not measured |
| noop_smoke::`noop_smoke` | [metal/common/noop_smoke.metal:11][f338] | Diagnostics and microtests · microtest / smoke | metal | — | — | not measured |
| selective_scan_decode::`selective_scan_decode` | [metal/common/selective_scan_decode.metal:40][f342] | Mamba2 · SSD / selective scan | metal | — | [1 note][t342] | not measured |
| silu_gate::`silu_gate` | [metal/common/silu_gate.metal:20][f344] | Activations and elementwise · activation / gate / residual | metal | — | — | not measured |
| softmax_topp::`softmax_topp` | [metal/common/softmax_topp.metal:32][f345] | Sampling · argmax / top-p | metal | — | [1 note][t345] | not measured |
| prefill_fp8kv::`attn_prefill_fp8kv_64` | [strix-hip/common/attn_prefill_fp8kv.cu:63][f348] | Attention · prefill (flash) | hip | — | [1 note][t348] | not measured |
| attn_prefill_h128::`attn_prefill_h128_64` | [strix-hip/common/attn_prefill_h128.cu:207][f349] | Attention · prefill (flash) | hip | — | [1 note][t349] | not measured |
| attn_prefill_v47::`attn_prefill_v47` | [strix-hip/common/attn_prefill_v47.cu:46][f350] | Attention · prefill (flash) | hip | — | [1 note][t350] | not measured |
| gemm::`fused_silu_mul` | [strix-hip/common/dense_gemm_bf16.cu:322][f351] | Activations and elementwise · activation / gate / residual | hip | — | [1 note][t351] | not measured |
| moe_fp8_grouped_gemm::`moe_fp8_grouped_gemm_v2` | [strix-hip/common/moe_fp8_grouped_gemm.cu:394][f353] | MoE · expert GEMM/GEMV | hip | — | [3 notes][t353] | not measured |
| w8a16_gemm::`w8a16_dequant` | [strix-hip/common/w8a16_gemm.cu:296][f354] | Quantization and format conversion · dequant / repack / transpose | hip | — | [2 notes][t354] | not measured |

## Measurements

199 rows over 61 entry points; every row, with its shape, time, floor, source and notes, is in [`docs/kernel-perf/MEASUREMENTS.md`](docs/kernel-perf/MEASUREMENTS.md). Per regime (median over the regime's rows, unweighted):

| Hardware | Model | Regime | Rows | Entry points | Median % of floor |
|---|---|---|---|---|---|
| gb10 | Qwen/Qwen3.6-35B-A3B-FP8 | decode C=1 (R=2, MTP k=1) | 23 | 19 | 55% |
| gb10 | Qwen/Qwen3.6-35B-A3B-FP8 | decode C=16 (R=32, MTP k=1) | 15 | 10 | 73% |
| gb10 | Qwen/Qwen3.6-35B-A3B-FP8 | prefill 32k (cold, 32772 tok) | 33 | 24 | 25% |
| gb10 | Qwen/Qwen3.6-35B-A3B-FP8 | prefill 4k (cold, 4549 tok) | 35 | 27 | 33% |
| gb10 | unsloth/Qwen3.8-27B-NVFP4 | decode C=1 (R=4, MTP k=3) | 19 | 10 | 89% |
| gb10 | unsloth/Qwen3.8-27B-NVFP4 | decode C=16 (R=32, MTP k=1) | 20 | 13 | 68% |
| gb10 | unsloth/Qwen3.8-27B-NVFP4 | prefill 32k (cold, 32772 tok) | 26 | 20 | 29% |
| gb10 | unsloth/Qwen3.8-27B-NVFP4 | prefill 4k (cold, 4103 tok) | 28 | 21 | 36% |

[f1]: kernels/b200/kimi-k3/bf16/dense_f32io.cu
[f2]: kernels/b300/common/dsa_indexer.cu
[f3]: kernels/b300/common/moe_shared_expert_fused.cu
[f4]: kernels/b300/common/w8a16_gemv_batch4.cu
[f5]: kernels/gb10/common/argmax_bf16.cu
[f6]: kernels/gb10/common/argmax_feed.cu
[f7]: kernels/gb10/common/attn_prefill.cu
[f8]: kernels/gb10/common/attn_prefill_fa128.cu
[f9]: kernels/gb10/common/attn_prefill_fp8kv.cu
[f10]: kernels/gb10/common/attn_prefill_h128.cu
[f11]: kernels/gb10/common/attn_prefill_v47.cu
[f12]: kernels/gb10/common/bf16_add.cu
[f13]: kernels/gb10/common/causal_conv1d.cu
[f14]: kernels/gb10/common/dense_gemm_bf16.cu
[f15]: kernels/gb10/common/dense_gemm_splitk.cu
[f16]: kernels/gb10/common/dense_gemm_tc.cu
[f17]: kernels/gb10/common/dense_gemv_bf16.cu
[f18]: kernels/gb10/common/dense_gemv_bf16_batch2.cu
[f19]: kernels/gb10/common/dense_gemv_bf16_batchm.cu
[f20]: kernels/gb10/common/dense_gemv_bf16_tc.cu
[f21]: kernels/gb10/common/dense_gemv_fp8w.cu
[f22]: kernels/gb10/common/dense_gemv_fp8w_batch2.cu
[f23]: kernels/gb10/common/dequant_fp8_blockscaled_bf16.cu
[f24]: kernels/gb10/common/dequant_gguf_bf16.cu
[f25]: kernels/gb10/common/dequant_nvfp4_bf16.cu
[f26]: kernels/gb10/common/dflash2.cu
[f27]: kernels/gb10/common/dsa_indexer.cu
[f28]: kernels/gb10/common/e2m1_branchless.cu
[f29]: kernels/gb10/common/embed_from_argmax.cu
[f30]: kernels/gb10/common/fp8_gemm_blockscaled_pipe.cu
[f31]: kernels/gb10/common/fp8_gemm_t_blockscaled.cu
[f32]: kernels/gb10/common/fp8_gemv_rt.cu
[f33]: kernels/gb10/common/fp8_scale_transpose.cu
[f34]: kernels/gb10/common/fused_k_norm_rope_cache.cu
[f35]: kernels/gb10/common/gated_delta_rule.cu
[f36]: kernels/gb10/common/gated_delta_rule_carry.cu
[f37]: kernels/gb10/common/gated_delta_rule_fla.cu
[f38]: kernels/gb10/common/gated_delta_rule_persistent.cu
[f39]: kernels/gb10/common/gated_delta_rule_regresident.cu
[f40]: kernels/gb10/common/gated_delta_rule_wy.cu
[f41]: kernels/gb10/common/gated_delta_rule_wy2_resident.cu
[f42]: kernels/gb10/common/gated_delta_rule_wy2_resident_f16.cu
[f43]: kernels/gb10/common/gated_delta_rule_wy3.cu
[f44]: kernels/gb10/common/gated_delta_rule_wy3_f16.cu
[f45]: kernels/gb10/common/gated_delta_rule_wy3_resident.cu
[f46]: kernels/gb10/common/gated_delta_rule_wy3_resident_f16.cu
[f47]: kernels/gb10/common/gated_delta_rule_wy4.cu
[f48]: kernels/gb10/common/gated_delta_rule_wy4_f16.cu
[f49]: kernels/gb10/common/gated_delta_rule_wy4_woa.cu
[f50]: kernels/gb10/common/gated_delta_rule_wy64_prefill.cu
[f51]: kernels/gb10/common/gated_delta_rule_wy_f16.cu
[f52]: kernels/gb10/common/gated_delta_rule_wyn.cu
[f53]: kernels/gb10/common/gdn_chunk_fwd_o_mma8.cu
[f54]: kernels/gb10/common/gdn_verify_fused_conv_kn.cu
[f55]: kernels/gb10/common/gdn_verify_fused_k2.cu
[f56]: kernels/gb10/common/glm5next_ffn.cu
[f57]: kernels/gb10/common/glm5next_mhc.cu
[f58]: kernels/gb10/common/gpt_oss_expert_ops.cu
[f59]: kernels/gb10/common/gpt_oss_mxfp4_gemv.cu
[f60]: kernels/gb10/common/gpt_oss_rope.cu
[f61]: kernels/gb10/common/gpt_oss_staged_attention.cu
[f62]: kernels/gb10/common/kda_chunk.cu
[f63]: kernels/gb10/common/kda_gate.cu
[f64]: kernels/gb10/common/kda_layer_ops.cu
[f65]: kernels/gb10/common/kda_recurrent.cu
[f66]: kernels/gb10/common/lora_bgmv.cu
[f67]: kernels/gb10/common/mamba2_ssd_chunk.cu
[f68]: kernels/gb10/common/mamba2_ssm_decode.cu
[f69]: kernels/gb10/common/metadata_fill.cu
[f70]: kernels/gb10/common/moe_bf16_grouped_gemm.cu
[f71]: kernels/gb10/common/moe_bf16_grouped_tc.cu
[f72]: kernels/gb10/common/moe_decode_atomic_c4.cu
[f73]: kernels/gb10/common/moe_expert_gemv.cu
[f74]: kernels/gb10/common/moe_expert_gemv_fused.cu
[f75]: kernels/gb10/common/moe_expert_relu2_down_shared.cu
[f76]: kernels/gb10/common/moe_fp8_grouped_blend.cu
[f77]: kernels/gb10/common/moe_fp8_grouped_gemm.cu
[f78]: kernels/gb10/common/moe_fp8_grouped_sort.cu
[f79]: kernels/gb10/common/moe_fp8_grouped_tc.cu
[f80]: kernels/gb10/common/moe_fp8_grouped_tc_w8a8.cu
[f81]: kernels/gb10/common/moe_gate_topk.cu
[f82]: kernels/gb10/common/moe_hash_route.cu
[f83]: kernels/gb10/common/moe_lora_gather_bgmv.cu
[f84]: kernels/gb10/common/moe_lora_grouped_down.cu
[f85]: kernels/gb10/common/moe_nvfp4_grouped.cu
[f86]: kernels/gb10/common/moe_nvfp4_grouped_tc.cu
[f87]: kernels/gb10/common/moe_permute.cu
[f88]: kernels/gb10/common/moe_prefill.cu
[f89]: kernels/gb10/common/moe_router_gemm.cu
[f90]: kernels/gb10/common/moe_router_gemm_prefill.cu
[f91]: kernels/gb10/common/moe_shared_expert_fused.cu
[f92]: kernels/gb10/common/moe_shared_expert_fused_batch2.cu
[f93]: kernels/gb10/common/moe_shared_expert_fused_batch2_t.cu
[f94]: kernels/gb10/common/moe_shared_expert_fused_batch3.cu
[f95]: kernels/gb10/common/moe_shared_expert_fused_batch3_t.cu
[f96]: kernels/gb10/common/moe_shared_expert_fused_bf16.cu
[f97]: kernels/gb10/common/moe_shared_expert_fused_bf16_batch2.cu
[f98]: kernels/gb10/common/moe_shared_expert_fused_fp8.cu
[f99]: kernels/gb10/common/moe_shared_expert_fused_fp8_batch2.cu
[f100]: kernels/gb10/common/moe_shared_expert_fused_fp8_batch2_t.cu
[f101]: kernels/gb10/common/moe_shared_expert_fused_fp8_batch3.cu
[f102]: kernels/gb10/common/moe_shared_expert_fused_fp8_batch3_t.cu
[f103]: kernels/gb10/common/moe_shared_expert_fused_fp8_grouped.cu
[f104]: kernels/gb10/common/moe_shared_expert_fused_fp8_t.cu
[f105]: kernels/gb10/common/moe_shared_expert_fused_t.cu
[f106]: kernels/gb10/common/moe_silu_mul.cu
[f107]: kernels/gb10/common/moe_sorted_prefill.cu
[f108]: kernels/gb10/common/moe_topk.cu
[f109]: kernels/gb10/common/moe_topk_sigmoid.cu
[f110]: kernels/gb10/common/moe_topk_softmax_bias.cu
[f111]: kernels/gb10/common/moe_topk_sqrtsoftplus.cu
[f112]: kernels/gb10/common/moe_transpose_batched.cu
[f113]: kernels/gb10/common/moe_unpermute_blend.cu
[f114]: kernels/gb10/common/moe_w4a16_grouped_gemm.cu
[f115]: kernels/gb10/common/moe_w8a8_grouped_gemm.cu
[f116]: kernels/gb10/common/moe_w8a8_grouped_gemm_e4m3.cu
[f117]: kernels/gb10/common/nemotron_moe_prefill.cu
[f118]: kernels/gb10/common/nllb_encoder.cu
[f119]: kernels/gb10/common/paged_decode_attn.cu
[f120]: kernels/gb10/common/paged_decode_attn_bf16_gqa.cu
[f121]: kernels/gb10/common/paged_decode_attn_bf16k_turbo2v.cu
[f122]: kernels/gb10/common/paged_decode_attn_bf16k_turbo2v_128.cu
[f123]: kernels/gb10/common/paged_decode_attn_bf16k_turbo3v.cu
[f124]: kernels/gb10/common/paged_decode_attn_bf16k_turbo3v_128.cu
[f125]: kernels/gb10/common/paged_decode_attn_bf16k_turbo4v.cu
[f126]: kernels/gb10/common/paged_decode_attn_bf16k_turbo4v_128.cu
[f127]: kernels/gb10/common/paged_decode_attn_fp8.cu
[f128]: kernels/gb10/common/paged_decode_attn_fp8_gqa.cu
[f129]: kernels/gb10/common/paged_decode_attn_fp8k_turbo2v.cu
[f130]: kernels/gb10/common/paged_decode_attn_fp8k_turbo2v_128.cu
[f131]: kernels/gb10/common/paged_decode_attn_fp8k_turbo3v.cu
[f132]: kernels/gb10/common/paged_decode_attn_fp8k_turbo3v_128.cu
[f133]: kernels/gb10/common/paged_decode_attn_fp8k_turbo4v.cu
[f134]: kernels/gb10/common/paged_decode_attn_fp8k_turbo4v_128.cu
[f135]: kernels/gb10/common/paged_decode_attn_nvfp4.cu
[f136]: kernels/gb10/common/paged_decode_attn_turbo2.cu
[f137]: kernels/gb10/common/paged_decode_attn_turbo2_128.cu
[f138]: kernels/gb10/common/paged_decode_attn_turbo3.cu
[f139]: kernels/gb10/common/paged_decode_attn_turbo3_128.cu
[f140]: kernels/gb10/common/paged_decode_attn_turbo3k_turbo8v.cu
[f141]: kernels/gb10/common/paged_decode_attn_turbo3k_turbo8v_128.cu
[f142]: kernels/gb10/common/paged_decode_attn_turbo4.cu
[f143]: kernels/gb10/common/paged_decode_attn_turbo4_128.cu
[f144]: kernels/gb10/common/paged_decode_attn_turbo4_512.cu
[f145]: kernels/gb10/common/paged_decode_attn_turbo4k_turbo3v.cu
[f146]: kernels/gb10/common/paged_decode_attn_turbo4k_turbo3v_128.cu
[f147]: kernels/gb10/common/paged_decode_attn_turbo4k_turbo8v.cu
[f148]: kernels/gb10/common/paged_decode_attn_turbo4k_turbo8v_128.cu
[f149]: kernels/gb10/common/paged_decode_attn_turbo8.cu
[f150]: kernels/gb10/common/paged_decode_attn_turbo8_128.cu
[f151]: kernels/gb10/common/paged_decode_attn_turbo8_512.cu
[f152]: kernels/gb10/common/per_token_group_quant_fp8.cu
[f153]: kernels/gb10/common/prefill_paged_compute.cuh
[f154]: kernels/gb10/common/prefill_paged_compute_512.cuh
[f155]: kernels/gb10/common/prefill_paged_compute_asym.cuh
[f156]: kernels/gb10/common/projection_bias.cu
[f157]: kernels/gb10/common/q2_0_gemv.cu
[f158]: kernels/gb10/common/q2_0_gemv_vec.cu
[f159]: kernels/gb10/common/quant_rowwise_fp8.cu
[f160]: kernels/gb10/common/quantize_bf16_to_fp8_blockscaled.cu
[f161]: kernels/gb10/common/quantize_bf16_to_nvfp4.cu
[f162]: kernels/gb10/common/relu_squared.cu
[f163]: kernels/gb10/common/reshape_and_cache.cu
[f164]: kernels/gb10/common/reshape_and_cache_fused_k_fp8.cu
[f165]: kernels/gb10/common/reshape_and_cache_turbo.cu
[f166]: kernels/gb10/common/residual_add.cu
[f167]: kernels/gb10/common/residual_add_rms_norm_exact.cu
[f168]: kernels/gb10/common/rms_norm.cu
[f169]: kernels/gb10/common/rms_norm_act_quant.cu
[f170]: kernels/gb10/common/rms_norm_vanilla.cu
[f171]: kernels/gb10/common/rope.cu
[f172]: kernels/gb10/common/rope_mrope_interleaved.cu
[f173]: kernels/gb10/common/ssm_ba_gates_hopper.cu
[f174]: kernels/gb10/common/ssm_ba_gates_tiled.cu
[f175]: kernels/gb10/common/ssm_h_dtype.cu
[f176]: kernels/gb10/common/ssm_preprocess.cu
[f177]: kernels/gb10/common/ssm_state_norm.cu
[f178]: kernels/gb10/common/token_overlay.cu
[f179]: kernels/gb10/common/tq_plus_innerq_apply.cu
[f180]: kernels/gb10/common/transpose_u8.cu
[f181]: kernels/gb10/common/vector_add.cu
[f182]: kernels/gb10/common/w4a16_fp8_ldmab.cu
[f183]: kernels/gb10/common/w4a16_gemm.cu
[f184]: kernels/gb10/common/w4a16_gemv.cu
[f185]: kernels/gb10/common/w4a16_gemv_fused.cu
[f186]: kernels/gb10/common/w4a16_gemv_tc.cu
[f187]: kernels/gb10/common/w4a16_tc_rows.cu
[f188]: kernels/gb10/common/w4a4_gemv_mx.cu
[f189]: kernels/gb10/common/w8a16_gemm.cu
[f190]: kernels/gb10/common/w8a16_gemm_pipe128.cu
[f191]: kernels/gb10/common/w8a16_gemm_pipelined.cu
[f192]: kernels/gb10/common/w8a16_gemm_pipelined_m32.cu
[f193]: kernels/gb10/common/w8a16_gemm_t.cu
[f194]: kernels/gb10/common/w8a16_gemm_t_m128.cu
[f195]: kernels/gb10/common/w8a16_gemv.cu
[f196]: kernels/gb10/common/w8a16_gemv_batch4.cu
[f197]: kernels/gb10/common/w8a16_gemv_fused.cu
[f198]: kernels/gb10/common/w8a16_tc_rows.cu
[f199]: kernels/gb10/common/w8a8_act_quant.cu
[f200]: kernels/gb10/common/w8a8_gemv.cu
[f201]: kernels/gb10/common/wht_bf16.cu
[f202]: kernels/gb10/common/widen_block_scale_f32.cu
[f203]: kernels/gb10/deepseek-v4-flash/nvfp4/attn_prefill_512.cu
[f204]: kernels/gb10/deepseek-v4-flash/nvfp4/attn_v41.cu
[f205]: kernels/gb10/deepseek-v4-flash/nvfp4/csa_compress.cu
[f206]: kernels/gb10/deepseek-v4-flash/nvfp4/engram_v41.cu
[f207]: kernels/gb10/deepseek-v4-flash/nvfp4/grouped_gemm_mla.cu
[f208]: kernels/gb10/deepseek-v4-flash/nvfp4/hc_v41.cu
[f209]: kernels/gb10/deepseek-v4-flash/nvfp4/hyper_connection.cu
[f210]: kernels/gb10/deepseek-v4-flash/nvfp4/kquant_moe.cu
[f211]: kernels/gb10/deepseek-v4-flash/nvfp4/mla_absorbed.cu
[f212]: kernels/gb10/deepseek-v4-flash/nvfp4/mla_cache_assemble_fp8.cu
[f213]: kernels/gb10/deepseek-v4-flash/nvfp4/mla_fused_prefill.cu
[f214]: kernels/gb10/deepseek-v4-flash/nvfp4/mla_paged_decode.cu
[f215]: kernels/gb10/deepseek-v4-flash/nvfp4/mla_paged_decode_fp8.cu
[f216]: kernels/gb10/deepseek-v4-flash/nvfp4/mla_prefill_attn.cu
[f217]: kernels/gb10/deepseek-v4-flash/nvfp4/moe_silu_mul.cu
[f218]: kernels/gb10/deepseek-v4-flash/nvfp4/moe_v41.cu
[f219]: kernels/gb10/deepseek-v4-flash/nvfp4/moe_w4a16_grouped_gemm.cu
[f220]: kernels/gb10/deepseek-v4-flash/nvfp4/paged_decode_attn_512.cu
[f221]: kernels/gb10/deepseek-v4-flash/nvfp4/paged_decode_attn_fp8_mla.cu
[f222]: kernels/gb10/deepseek-v4-flash/nvfp4/paged_decode_attn_mla.cu
[f223]: kernels/gb10/deepseek-v4-flash/nvfp4/paged_decode_attn_nvfp4.cu
[f224]: kernels/gb10/deepseek-v4-flash/nvfp4/prefill_attn_compressed.cu
[f225]: kernels/gb10/deepseek-v4-flash/nvfp4/w4a16_gemm.cu
[f226]: kernels/gb10/gemma-4-26b-a4b/nvfp4/attn_prefill_512.cu
[f227]: kernels/gb10/gemma-4-26b-a4b/nvfp4/embed_scale.cu
[f228]: kernels/gb10/gemma-4-26b-a4b/nvfp4/gated_delta_rule.cu
[f229]: kernels/gb10/gemma-4-26b-a4b/nvfp4/gelu.cu
[f230]: kernels/gb10/gemma-4-26b-a4b/nvfp4/logit_softcap.cu
[f231]: kernels/gb10/gemma-4-26b-a4b/nvfp4/moe_shared_expert_fused.cu
[f232]: kernels/gb10/gemma-4-26b-a4b/nvfp4/moe_shared_expert_fused_batch2.cu
[f233]: kernels/gb10/gemma-4-26b-a4b/nvfp4/moe_shared_expert_fused_batch3.cu
[f234]: kernels/gb10/gemma-4-26b-a4b/nvfp4/moe_w4a16_grouped_gemm.cu
[f235]: kernels/gb10/gemma-4-26b-a4b/nvfp4/paged_decode_attn_512.cu
[f236]: kernels/gb10/gemma-4-26b-a4b/nvfp4/paged_decode_attn_fp8_512.cu
[f237]: kernels/gb10/gemma-4-26b-a4b/nvfp4/rms_norm.cu
[f238]: kernels/gb10/gemma-4-31b/nvfp4/attn_prefill_512.cu
[f239]: kernels/gb10/gemma-4-31b/nvfp4/embed_scale.cu
[f240]: kernels/gb10/gemma-4-31b/nvfp4/logit_softcap.cu
[f241]: kernels/gb10/gemma-4-31b/nvfp4/rms_norm.cu
[f242]: kernels/gb10/glm-5.3-flash/nvfp4/glm5next_dsa_mla_decode.cu
[f243]: kernels/gb10/glm-5.3-flash/nvfp4/glm5next_mla_latent_write.cu
[f244]: kernels/gb10/glm-5.3-flash/nvfp4/glm_vit.cu
[f245]: kernels/gb10/holo-3.1-0.8b/nvfp4/fp4_mma_microtest.cu
[f246]: kernels/gb10/kimi-k3/bf16/kda_decode.cu
[f247]: kernels/gb10/kimi-k3/bf16/mla_decode.cu
[f248]: kernels/gb10/minimax-m2-229b/nvfp4/moe_w4a16_grouped_gemm.cu
[f249]: kernels/gb10/minimax-m2-229b/nvfp4/rms_norm.cu
[f250]: kernels/gb10/minimax-m2-229b/nvfp4/w4a16_gemm_v2.cu
[f251]: kernels/gb10/minimax-m2-229b/nvfp4/w4a16_gemm_v3.cu
[f252]: kernels/gb10/mistral-small-4/nvfp4/mla_absorbed.cu
[f253]: kernels/gb10/mistral-small-4/nvfp4/mla_fused_prefill.cu
[f254]: kernels/gb10/mistral-small-4/nvfp4/mla_prefill_attn.cu
[f255]: kernels/gb10/mistral-small-4/nvfp4/paged_decode_attn_fp8_mla.cu
[f256]: kernels/gb10/mistral-small-4/nvfp4/paged_decode_attn_mla.cu
[f257]: kernels/gb10/mistral-small-4/nvfp4/rope.cu
[f258]: kernels/gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/moe_w4a16_grouped_gemm.cu
[f259]: kernels/gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/moe_w4a4_grouped.cu
[f260]: kernels/gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/rms_norm.cu
[f261]: kernels/gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/w4a16_gemm.cu
[f262]: kernels/gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/w4a4_gemm.cu
[f263]: kernels/gb10/qwen3-next-80b-a3b/nvfp4/gated_delta_rule.cu
[f264]: kernels/gb10/qwen3-vl-30b-a3b/nvfp4/rms_norm.cu
[f265]: kernels/gb10/qwen3-vl-30b-a3b/nvfp4/vision_encoder.cu
[f266]: kernels/gb10/qwen3.5-122b-a10b/nvfp4/gated_delta_rule.cu
[f267]: kernels/gb10/qwen3.6-27b/nvfp4/gated_delta_rule.cu
[f268]: kernels/gb10/qwen3.6-27b/nvfp4/gated_delta_rule_snap.cu
[f269]: kernels/gb10/qwen3.6-27b/nvfp4/gdn_exact_carry.cu
[f270]: kernels/gb10/qwen3.6-27b/nvfp4/gdn_verify_fused_conv_kn_f32.cu
[f271]: kernels/gb10/qwen3.6-27b/nvfp4/moe_w4a16_grouped_gemm.cu
[f272]: kernels/gb10/qwen3.6-27b/nvfp4/nvfp4_mmq.cu
[f273]: kernels/gb10/qwen3.6-27b/nvfp4/q2_0_mmq.cu
[f274]: kernels/gb10/qwen3.6-27b/nvfp4/q4k_mmq.cu
[f275]: kernels/gb10/qwen3.6-27b/nvfp4/q4k_quantize.cu
[f276]: kernels/gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu
[f277]: kernels/gb10/qwen3.6-27b/nvfp4/w4a16_gemm_v2.cu
[f278]: kernels/gb10/qwen3.6-27b/nvfp4/w4a4_gemm.cu
[f279]: kernels/gb10/qwen3.6-35b-a3b/nvfp4/gated_delta_rule.cu
[f280]: kernels/gb10/qwen3.6-35b-a3b/nvfp4/gated_delta_rule_wy17.cu
[f281]: kernels/gb10/qwen3.6-35b-a3b/nvfp4/gdn_exact_carry.cu
[f282]: kernels/gb10/qwen3.6-35b-a3b/nvfp4/moe_w4a16_grouped_gemm.cu
[f283]: kernels/gb10/qwen3.6-35b-a3b/nvfp4/vision_encoder.cu
[f284]: kernels/gb10/qwen3.6-35b-a3b/nvfp4/w4a16_gemm.cu
[f285]: kernels/gb10/qwen3.8-flash-next/nvfp4/gated_norm_sigmoid.cu
[f286]: kernels/gb10/qwen3.8-flash-next/nvfp4/hyper_connection.cu
[f287]: kernels/gb10/qwen3.8-flash-next/nvfp4/ple.cu
[f288]: kernels/gb10/qwen3.8-flash-next/nvfp4/qsa_indexer.cu
[f289]: kernels/gb10/step3p7-flash/nvfp4/moe_silu_mul.cu
[f290]: kernels/hopper/common/dense_gemm_m16_bf16.cu
[f291]: kernels/hopper/common/fp8_act_quant_hopper.cu
[f292]: kernels/hopper/common/gated_delta_rule_chunk_tc.cu
[f293]: kernels/hopper/common/gdn_fwd_o_hopper.cu
[f294]: kernels/hopper/common/gdn_recompute_wu_hopper.cu
[f295]: kernels/hopper/common/moe_bucket_builder.cu
[f296]: kernels/hopper/common/moe_w8a8_m16.cu
[f297]: kernels/hopper/common/paged_decode_bf16_splitk_hopper.cu
[f298]: kernels/hopper/common/paged_decode_fp8_splitk_hopper.cu
[f299]: kernels/hopper/common/silu_mul_strided.cu
[f300]: kernels/hopper/common/w8a16_gemm_m16.cu
[f301]: kernels/hopper/common/w8a16_gemv.cu
[f302]: kernels/hopper/common/w8a16_gemv_fused.cu
[f303]: kernels/hopper/common/w8a16_gemv_ncol.cu
[f304]: kernels/metal/common/add_rms_norm.metal
[f305]: kernels/metal/common/argmax_bf16.metal
[f306]: kernels/metal/common/attention_decode.metal
[f307]: kernels/metal/common/attention_decode_bf16k_turbov.metal
[f308]: kernels/metal/common/attention_decode_turbo2.metal
[f309]: kernels/metal/common/attention_decode_turbo3.metal
[f310]: kernels/metal/common/attention_decode_turbo4.metal
[f311]: kernels/metal/common/attention_decode_turbo8.metal
[f312]: kernels/metal/common/attention_full.metal
[f313]: kernels/metal/common/attention_prefill.metal
[f314]: kernels/metal/common/bf16_add.metal
[f315]: kernels/metal/common/causal_conv1d_decode.metal
[f316]: kernels/metal/common/causal_conv1d_update_l2norm.metal
[f317]: kernels/metal/common/conv3d_patch_embed.metal
[f318]: kernels/metal/common/dense_gemm_bf16.metal
[f319]: kernels/metal/common/dense_gemv_bf16.metal
[f320]: kernels/metal/common/embed_lookup.metal
[f321]: kernels/metal/common/gated_delta_rule_decode.metal
[f322]: kernels/metal/common/gdn_helpers.metal
[f323]: kernels/metal/common/gelu.metal
[f324]: kernels/metal/common/kv_cache_append.metal
[f325]: kernels/metal/common/kv_cache_append_bf16k_turbov.metal
[f326]: kernels/metal/common/kv_cache_append_turbo2.metal
[f327]: kernels/metal/common/kv_cache_append_turbo3.metal
[f328]: kernels/metal/common/kv_cache_append_turbo4.metal
[f329]: kernels/metal/common/kv_cache_append_turbo8.metal
[f330]: kernels/metal/common/layer_norm.metal
[f331]: kernels/metal/common/lora_bgmv.metal
[f332]: kernels/metal/common/mlx_int8_dequant.metal
[f333]: kernels/metal/common/mlx_int8_gemm.metal
[f334]: kernels/metal/common/mlx_int8_gemv.metal
[f335]: kernels/metal/common/mlx_int8_gemv_gate_up.metal
[f336]: kernels/metal/common/mlx_int8_gemv_silu_gate.metal
[f337]: kernels/metal/common/nllb_encoder.metal
[f338]: kernels/metal/common/noop_smoke.metal
[f339]: kernels/metal/common/qwen35_qkv_split.metal
[f340]: kernels/metal/common/rms_norm.metal
[f341]: kernels/metal/common/rope_apply.metal
[f342]: kernels/metal/common/selective_scan_decode.metal
[f343]: kernels/metal/common/sigmoid_gate.metal
[f344]: kernels/metal/common/silu_gate.metal
[f345]: kernels/metal/common/softmax_topp.metal
[f346]: kernels/metal/common/wht_bf16.metal
[f347]: kernels/strix-hip/common/attn_prefill.cu
[f348]: kernels/strix-hip/common/attn_prefill_fp8kv.cu
[f349]: kernels/strix-hip/common/attn_prefill_h128.cu
[f350]: kernels/strix-hip/common/attn_prefill_v47.cu
[f351]: kernels/strix-hip/common/dense_gemm_bf16.cu
[f352]: kernels/strix-hip/common/dense_gemm_tc.cu
[f353]: kernels/strix-hip/common/moe_fp8_grouped_gemm.cu
[f354]: kernels/strix-hip/common/w8a16_gemm.cu
[f355]: kernels/strix-hip/common/w8a16_gemm_t.cu
[f356]: kernels/strix-hip/qwen3.6-27b/nvfp4/moe_w4a16_grouped_gemm.cu
[f357]: kernels/strix-hip/qwen3.6-27b/nvfp4/w4a16_gemm.cu
[f358]: kernels/strix-hip/qwen3.6-35b-a3b/nvfp4/w4a16_gemm.cu
[m5.argmax_bf16]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-argmax-bf16-cu-argmax-bf16
[m166.bf16_concat]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-residual-add-cu-bf16-concat
[m168.l2_norm_bf16]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-rms-norm-cu-l2-norm-bf16
[m284.w4a16_gemm_t]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-qwen3-6-35b-a3b-nvfp4-w4a16-gemm-cu-w4a16-gemm-t
[m184.w4a16_gemv_sw]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-w4a16-gemv-cu-w4a16-gemv-sw
[m36.gdn_carry_conv]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-gated-delta-rule-carry-cu-gdn-carry-conv
[m7.attn_prefill_64]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-attn-prefill-cu-attn-prefill-64
[m14.dense_gemm_bf16]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-dense-gemm-bf16-cu-dense-gemm-bf16
[m17.dense_gemv_bf16]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-dense-gemv-bf16-cu-dense-gemv-bf16
[m186.w4a16_gemv_tc8]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-w4a16-gemv-tc-cu-w4a16-gemv-tc8
[m176.deinterleave_qg]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-ssm-preprocess-cu-deinterleave-qg
[m186.w4a16_gemv_tc16]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-w4a16-gemv-tc-cu-w4a16-gemv-tc16
[m276.fp8_gemm_t_m128]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-qwen3-6-27b-nvfp4-w4a16-gemm-cu-fp8-gemm-t-m128
[m276.w4a16_gemm_t_p3]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-qwen3-6-27b-nvfp4-w4a16-gemm-cu-w4a16-gemm-t-p3
[m87.moe_batched_blend]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-moe-permute-cu-moe-batched-blend
[m119.paged_decode_attn]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-paged-decode-attn-cu-paged-decode-attn
[m166.bf16_residual_add]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-residual-add-cu-bf16-residual-add
[m168.rms_norm_residual]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-rms-norm-cu-rms-norm-residual
[m194.w8a16_gemm_t_m128]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-w8a16-gemm-t-m128-cu-w8a16-gemm-t-m128
[m276.w4a16_gemm_t_m128]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-qwen3-6-27b-nvfp4-w4a16-gemm-cu-w4a16-gemm-t-m128
[m36.gdn_carry_wy2_lazy]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-gated-delta-rule-carry-cu-gdn-carry-wy2-lazy
[m106.silu_mul_quant_fp8]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-moe-silu-mul-cu-silu-mul-quant-fp8
[m182.fp8_fp8_gemm_ldmab]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-w4a16-fp8-ldmab-cu-fp8-fp8-gemm-ldmab
[m20.dense_gemv_bf16_tc16]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-dense-gemv-bf16-tc-cu-dense-gemv-bf16-tc16
[m40.gated_delta_rule_wy2]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-gated-delta-rule-wy-cu-gated-delta-rule-wy2
[m47.gated_delta_rule_wy4]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-gated-delta-rule-wy4-cu-gated-delta-rule-wy4
[m78.moe_fp8_grouped_sort]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-moe-fp8-grouped-sort-cu-moe-fp8-grouped-sort
[m191.w8a16_gemm_pipelined]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-w8a16-gemm-pipelined-cu-w8a16-gemm-pipelined
[m108.moe_topk_softmax_rows]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-moe-topk-cu-moe-topk-softmax-rows
[m127.paged_decode_attn_fp8]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-paged-decode-attn-fp8-cu-paged-decode-attn-fp8
[m14.dense_gemm_bf16_router]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-dense-gemm-bf16-cu-dense-gemm-bf16-router
[m153.attn_prefill_paged_64]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-prefill-paged-compute-cuh-attn-prefill-paged-64
[m168.residual_add_rms_norm]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-rms-norm-cu-residual-add-rms-norm
[m19.dense_gemv_bf16_batchm]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-dense-gemv-bf16-batchm-cu-dense-gemv-bf16-batchm
[m31.fp8_gemm_t_blockscaled]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-fp8-gemm-t-blockscaled-cu-fp8-gemm-t-blockscaled
[m168.gated_rms_norm_prefill]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-rms-norm-cu-gated-rms-norm-prefill
[m272.metrale_nvfp4_mmq32_nc]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-qwen3-6-27b-nvfp4-nvfp4-mmq-cu-metrale-nvfp4-mmq32-nc
[m87.moe_build_tile_worklist]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-moe-permute-cu-moe-build-tile-worklist
[m272.metrale_nvfp4_mmq128_nc]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-qwen3-6-27b-nvfp4-nvfp4-mmq-cu-metrale-nvfp4-mmq128-nc
[m276.w4a16_gemm_t_k64_n64_p3]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-qwen3-6-27b-nvfp4-w4a16-gemm-cu-w4a16-gemm-t-k64-n64-p3
[m192.w8a16_gemm_pipelined_m32]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-w8a16-gemm-pipelined-m32-cu-w8a16-gemm-pipelined-m32
[m272.metrale_nvfp4_scale_bf16]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-qwen3-6-27b-nvfp4-nvfp4-mmq-cu-metrale-nvfp4-scale-bf16
[m115.moe_w8a8_grouped_gemm_pm4]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-moe-w8a8-grouped-gemm-cu-moe-w8a8-grouped-gemm-pm4
[m152.per_token_group_quant_fp8]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-per-token-group-quant-fp8-cu-per-token-group-quant-fp8
[m13.causal_conv1d_update_l2norm]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-causal-conv1d-cu-causal-conv1d-update-l2norm
[m176.deinterleave_qg_split_qnorm]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-ssm-preprocess-cu-deinterleave-qg-split-qnorm
[m176.dense_gemm_ba_gates_prefill]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-ssm-preprocess-cu-dense-gemm-ba-gates-prefill
[m272.metrale_nvfp4_quantize_bf16]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-qwen3-6-27b-nvfp4-nvfp4-mmq-cu-metrale-nvfp4-quantize-bf16
[m37.gated_delta_rule_chunk_fwd_o]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-gated-delta-rule-fla-cu-gated-delta-rule-chunk-fwd-o
[m87.moe_unpermute_reduce_indexed]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-moe-permute-cu-moe-unpermute-reduce-indexed
[m272.metrale_nvfp4_silu_mul_quant]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-qwen3-6-27b-nvfp4-nvfp4-mmq-cu-metrale-nvfp4-silu-mul-quant
[m37.gated_delta_rule_recompute_wu]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-gated-delta-rule-fla-cu-gated-delta-rule-recompute-wu
[m98.moe_expert_gate_up_shared_fp8]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-moe-shared-expert-fused-fp8-cu-moe-expert-gate-up-shared-fp8
[m13.causal_conv1d_update_prefill_tp]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-causal-conv1d-cu-causal-conv1d-update-prefill-tp
[m172.rope_forward_mrope_interleaved]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-rope-mrope-interleaved-cu-rope-forward-mrope-interleaved
[m103.moe_expert_down_act_fp8_grouped]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-moe-shared-expert-fused-fp8-grouped-cu-moe-expert-down-act-fp8-grouped
[m54.gdn_verify_fused_conv_kn_batched]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-gdn-verify-fused-conv-kn-cu-gdn-verify-fused-conv-kn-batched
[m42.gated_delta_rule_wy2_resident_f16]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-gated-delta-rule-wy2-resident-f16-cu-gated-delta-rule-wy2-resident-f16
[m76.moe_weighted_sum_blend_fp8_grouped]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-moe-fp8-grouped-blend-cu-moe-weighted-sum-blend-fp8-grouped
[m103.moe_expert_gate_up_act_fp8_grouped]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-moe-shared-expert-fused-fp8-grouped-cu-moe-expert-gate-up-act-fp8-grouped
[m37.gated_delta_rule_chunk_delta_h_vfused]: docs/kernel-perf/MEASUREMENTS.md#m-kernels-gb10-common-gated-delta-rule-fla-cu-gated-delta-rule-chunk-delta-h-vfused
[pr1]: https://github.com/Metrale/metrale-inference/pull/1
[pr4]: https://github.com/Metrale/metrale-inference/pull/4
[pr14]: https://github.com/Metrale/metrale-inference/pull/14
[pr18]: https://github.com/Metrale/metrale-inference/pull/18
[pr25]: https://github.com/Metrale/metrale-inference/pull/25
[pr34]: https://github.com/Metrale/metrale-inference/pull/34
[t1]: docs/kernel-perf/TRADEOFFS.md#to-kernels-b200-kimi-k3-bf16-dense-f32io-cu
[t2]: docs/kernel-perf/TRADEOFFS.md#to-kernels-b300-common-dsa-indexer-cu
[t3]: docs/kernel-perf/TRADEOFFS.md#to-kernels-b300-common-moe-shared-expert-fused-cu
[t4]: docs/kernel-perf/TRADEOFFS.md#to-kernels-b300-common-w8a16-gemv-batch4-cu
[t5]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-argmax-bf16-cu
[t6]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-argmax-feed-cu
[t7]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-attn-prefill-cu
[t9]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-attn-prefill-fp8kv-cu
[t10]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-attn-prefill-h128-cu
[t11]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-attn-prefill-v47-cu
[t12]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-bf16-add-cu
[t13]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-causal-conv1d-cu
[t14]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-dense-gemm-bf16-cu
[t15]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-dense-gemm-splitk-cu
[t16]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-dense-gemm-tc-cu
[t17]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-dense-gemv-bf16-cu
[t18]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-dense-gemv-bf16-batch2-cu
[t19]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-dense-gemv-bf16-batchm-cu
[t20]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-dense-gemv-bf16-tc-cu
[t21]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-dense-gemv-fp8w-cu
[t22]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-dense-gemv-fp8w-batch2-cu
[t23]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-dequant-fp8-blockscaled-bf16-cu
[t24]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-dequant-gguf-bf16-cu
[t25]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-dequant-nvfp4-bf16-cu
[t26]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-dflash2-cu
[t27]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-dsa-indexer-cu
[t28]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-e2m1-branchless-cu
[t29]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-embed-from-argmax-cu
[t31]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-fp8-gemm-t-blockscaled-cu
[t32]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-fp8-gemv-rt-cu
[t33]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-fp8-scale-transpose-cu
[t34]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-fused-k-norm-rope-cache-cu
[t35]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-gated-delta-rule-cu
[t36]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-gated-delta-rule-carry-cu
[t37]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-gated-delta-rule-fla-cu
[t38]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-gated-delta-rule-persistent-cu
[t39]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-gated-delta-rule-regresident-cu
[t40]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-gated-delta-rule-wy-cu
[t41]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-gated-delta-rule-wy2-resident-cu
[t42]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-gated-delta-rule-wy2-resident-f16-cu
[t43]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-gated-delta-rule-wy3-cu
[t44]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-gated-delta-rule-wy3-f16-cu
[t45]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-gated-delta-rule-wy3-resident-cu
[t46]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-gated-delta-rule-wy3-resident-f16-cu
[t47]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-gated-delta-rule-wy4-cu
[t48]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-gated-delta-rule-wy4-f16-cu
[t49]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-gated-delta-rule-wy4-woa-cu
[t50]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-gated-delta-rule-wy64-prefill-cu
[t51]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-gated-delta-rule-wy-f16-cu
[t52]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-gated-delta-rule-wyn-cu
[t54]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-gdn-verify-fused-conv-kn-cu
[t55]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-gdn-verify-fused-k2-cu
[t56]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-glm5next-ffn-cu
[t57]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-glm5next-mhc-cu
[t62]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-kda-chunk-cu
[t64]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-kda-layer-ops-cu
[t65]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-kda-recurrent-cu
[t66]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-lora-bgmv-cu
[t67]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-mamba2-ssd-chunk-cu
[t68]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-mamba2-ssm-decode-cu
[t69]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-metadata-fill-cu
[t70]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-bf16-grouped-gemm-cu
[t72]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-decode-atomic-c4-cu
[t73]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-expert-gemv-cu
[t74]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-expert-gemv-fused-cu
[t75]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-expert-relu2-down-shared-cu
[t76]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-fp8-grouped-blend-cu
[t77]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-fp8-grouped-gemm-cu
[t78]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-fp8-grouped-sort-cu
[t79]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-fp8-grouped-tc-cu
[t80]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-fp8-grouped-tc-w8a8-cu
[t81]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-gate-topk-cu
[t82]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-hash-route-cu
[t83]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-lora-gather-bgmv-cu
[t84]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-lora-grouped-down-cu
[t85]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-nvfp4-grouped-cu
[t87]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-permute-cu
[t88]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-prefill-cu
[t89]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-router-gemm-cu
[t91]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-shared-expert-fused-cu
[t92]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-shared-expert-fused-batch2-cu
[t93]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-shared-expert-fused-batch2-t-cu
[t94]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-shared-expert-fused-batch3-cu
[t95]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-shared-expert-fused-batch3-t-cu
[t96]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-shared-expert-fused-bf16-cu
[t97]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-shared-expert-fused-bf16-batch2-cu
[t98]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-shared-expert-fused-fp8-cu
[t99]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-shared-expert-fused-fp8-batch2-cu
[t100]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-shared-expert-fused-fp8-batch2-t-cu
[t101]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-shared-expert-fused-fp8-batch3-cu
[t102]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-shared-expert-fused-fp8-batch3-t-cu
[t103]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-shared-expert-fused-fp8-grouped-cu
[t104]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-shared-expert-fused-fp8-t-cu
[t105]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-shared-expert-fused-t-cu
[t106]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-silu-mul-cu
[t107]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-sorted-prefill-cu
[t108]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-topk-cu
[t109]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-topk-sigmoid-cu
[t110]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-topk-softmax-bias-cu
[t111]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-topk-sqrtsoftplus-cu
[t112]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-transpose-batched-cu
[t114]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-w4a16-grouped-gemm-cu
[t115]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-moe-w8a8-grouped-gemm-cu
[t117]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-nemotron-moe-prefill-cu
[t118]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-nllb-encoder-cu
[t119]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-cu
[t120]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-bf16-gqa-cu
[t121]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-bf16k-turbo2v-cu
[t122]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-bf16k-turbo2v-128-cu
[t123]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-bf16k-turbo3v-cu
[t124]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-bf16k-turbo3v-128-cu
[t125]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-bf16k-turbo4v-cu
[t126]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-bf16k-turbo4v-128-cu
[t127]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-fp8-cu
[t128]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-fp8-gqa-cu
[t129]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-fp8k-turbo2v-cu
[t130]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-fp8k-turbo2v-128-cu
[t131]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-fp8k-turbo3v-cu
[t132]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-fp8k-turbo3v-128-cu
[t133]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-fp8k-turbo4v-cu
[t134]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-fp8k-turbo4v-128-cu
[t135]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-nvfp4-cu
[t136]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-turbo2-cu
[t137]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-turbo2-128-cu
[t138]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-turbo3-cu
[t139]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-turbo3-128-cu
[t140]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-turbo3k-turbo8v-cu
[t141]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-turbo3k-turbo8v-128-cu
[t142]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-turbo4-cu
[t143]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-turbo4-128-cu
[t144]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-turbo4-512-cu
[t145]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-turbo4k-turbo3v-cu
[t146]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-turbo4k-turbo3v-128-cu
[t147]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-turbo4k-turbo8v-cu
[t148]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-turbo4k-turbo8v-128-cu
[t149]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-turbo8-cu
[t150]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-turbo8-128-cu
[t151]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-paged-decode-attn-turbo8-512-cu
[t152]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-per-token-group-quant-fp8-cu
[t153]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-prefill-paged-compute-cuh
[t154]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-prefill-paged-compute-512-cuh
[t155]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-prefill-paged-compute-asym-cuh
[t157]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-q2-0-gemv-cu
[t158]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-q2-0-gemv-vec-cu
[t159]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-quant-rowwise-fp8-cu
[t160]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-quantize-bf16-to-fp8-blockscaled-cu
[t161]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-quantize-bf16-to-nvfp4-cu
[t162]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-relu-squared-cu
[t163]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-reshape-and-cache-cu
[t164]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-reshape-and-cache-fused-k-fp8-cu
[t165]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-reshape-and-cache-turbo-cu
[t166]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-residual-add-cu
[t168]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-rms-norm-cu
[t170]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-rms-norm-vanilla-cu
[t171]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-rope-cu
[t172]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-rope-mrope-interleaved-cu
[t173]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-ssm-ba-gates-hopper-cu
[t175]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-ssm-h-dtype-cu
[t176]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-ssm-preprocess-cu
[t177]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-ssm-state-norm-cu
[t178]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-token-overlay-cu
[t179]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-tq-plus-innerq-apply-cu
[t182]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-w4a16-fp8-ldmab-cu
[t183]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-w4a16-gemm-cu
[t184]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-w4a16-gemv-cu
[t185]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-w4a16-gemv-fused-cu
[t186]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-w4a16-gemv-tc-cu
[t188]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-w4a4-gemv-mx-cu
[t189]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-w8a16-gemm-cu
[t191]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-w8a16-gemm-pipelined-cu
[t192]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-w8a16-gemm-pipelined-m32-cu
[t193]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-w8a16-gemm-t-cu
[t194]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-w8a16-gemm-t-m128-cu
[t195]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-w8a16-gemv-cu
[t196]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-w8a16-gemv-batch4-cu
[t197]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-w8a16-gemv-fused-cu
[t198]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-w8a16-tc-rows-cu
[t201]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-wht-bf16-cu
[t202]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-common-widen-block-scale-f32-cu
[t203]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-deepseek-v4-flash-nvfp4-attn-prefill-512-cu
[t204]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-deepseek-v4-flash-nvfp4-attn-v41-cu
[t205]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-deepseek-v4-flash-nvfp4-csa-compress-cu
[t206]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-deepseek-v4-flash-nvfp4-engram-v41-cu
[t207]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-deepseek-v4-flash-nvfp4-grouped-gemm-mla-cu
[t208]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-deepseek-v4-flash-nvfp4-hc-v41-cu
[t209]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-deepseek-v4-flash-nvfp4-hyper-connection-cu
[t210]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-deepseek-v4-flash-nvfp4-kquant-moe-cu
[t211]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-deepseek-v4-flash-nvfp4-mla-absorbed-cu
[t212]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-deepseek-v4-flash-nvfp4-mla-cache-assemble-fp8-cu
[t213]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-deepseek-v4-flash-nvfp4-mla-fused-prefill-cu
[t214]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-deepseek-v4-flash-nvfp4-mla-paged-decode-cu
[t215]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-deepseek-v4-flash-nvfp4-mla-paged-decode-fp8-cu
[t216]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-deepseek-v4-flash-nvfp4-mla-prefill-attn-cu
[t217]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-deepseek-v4-flash-nvfp4-moe-silu-mul-cu
[t218]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-deepseek-v4-flash-nvfp4-moe-v41-cu
[t219]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-deepseek-v4-flash-nvfp4-moe-w4a16-grouped-gemm-cu
[t220]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-deepseek-v4-flash-nvfp4-paged-decode-attn-512-cu
[t221]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-deepseek-v4-flash-nvfp4-paged-decode-attn-fp8-mla-cu
[t222]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-deepseek-v4-flash-nvfp4-paged-decode-attn-mla-cu
[t223]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-deepseek-v4-flash-nvfp4-paged-decode-attn-nvfp4-cu
[t224]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-deepseek-v4-flash-nvfp4-prefill-attn-compressed-cu
[t225]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-deepseek-v4-flash-nvfp4-w4a16-gemm-cu
[t226]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-gemma-4-26b-a4b-nvfp4-attn-prefill-512-cu
[t228]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-gemma-4-26b-a4b-nvfp4-gated-delta-rule-cu
[t229]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-gemma-4-26b-a4b-nvfp4-gelu-cu
[t230]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-gemma-4-26b-a4b-nvfp4-logit-softcap-cu
[t231]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-gemma-4-26b-a4b-nvfp4-moe-shared-expert-fused-cu
[t232]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-gemma-4-26b-a4b-nvfp4-moe-shared-expert-fused-batch2-cu
[t233]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-gemma-4-26b-a4b-nvfp4-moe-shared-expert-fused-batch3-cu
[t234]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-gemma-4-26b-a4b-nvfp4-moe-w4a16-grouped-gemm-cu
[t235]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-gemma-4-26b-a4b-nvfp4-paged-decode-attn-512-cu
[t236]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-gemma-4-26b-a4b-nvfp4-paged-decode-attn-fp8-512-cu
[t237]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-gemma-4-26b-a4b-nvfp4-rms-norm-cu
[t238]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-gemma-4-31b-nvfp4-attn-prefill-512-cu
[t239]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-gemma-4-31b-nvfp4-embed-scale-cu
[t240]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-gemma-4-31b-nvfp4-logit-softcap-cu
[t241]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-gemma-4-31b-nvfp4-rms-norm-cu
[t242]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-glm-5-3-flash-nvfp4-glm5next-dsa-mla-decode-cu
[t243]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-glm-5-3-flash-nvfp4-glm5next-mla-latent-write-cu
[t244]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-glm-5-3-flash-nvfp4-glm-vit-cu
[t245]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-holo-3-1-0-8b-nvfp4-fp4-mma-microtest-cu
[t246]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-kimi-k3-bf16-kda-decode-cu
[t247]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-kimi-k3-bf16-mla-decode-cu
[t248]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-minimax-m2-229b-nvfp4-moe-w4a16-grouped-gemm-cu
[t249]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-minimax-m2-229b-nvfp4-rms-norm-cu
[t250]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-minimax-m2-229b-nvfp4-w4a16-gemm-v2-cu
[t251]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-minimax-m2-229b-nvfp4-w4a16-gemm-v3-cu
[t252]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-mistral-small-4-nvfp4-mla-absorbed-cu
[t253]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-mistral-small-4-nvfp4-mla-fused-prefill-cu
[t254]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-mistral-small-4-nvfp4-mla-prefill-attn-cu
[t255]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-mistral-small-4-nvfp4-paged-decode-attn-fp8-mla-cu
[t256]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-mistral-small-4-nvfp4-paged-decode-attn-mla-cu
[t257]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-mistral-small-4-nvfp4-rope-cu
[t258]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-nemotron-labs-3-puzzle-75b-a9b-nvfp4-moe-w4a16-grouped-gemm-cu
[t259]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-nemotron-labs-3-puzzle-75b-a9b-nvfp4-moe-w4a4-grouped-cu
[t260]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-nemotron-labs-3-puzzle-75b-a9b-nvfp4-rms-norm-cu
[t261]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-nemotron-labs-3-puzzle-75b-a9b-nvfp4-w4a16-gemm-cu
[t262]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-nemotron-labs-3-puzzle-75b-a9b-nvfp4-w4a4-gemm-cu
[t263]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-qwen3-next-80b-a3b-nvfp4-gated-delta-rule-cu
[t264]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-qwen3-vl-30b-a3b-nvfp4-rms-norm-cu
[t265]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-qwen3-vl-30b-a3b-nvfp4-vision-encoder-cu
[t266]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-qwen3-5-122b-a10b-nvfp4-gated-delta-rule-cu
[t267]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-qwen3-6-27b-nvfp4-gated-delta-rule-cu
[t268]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-qwen3-6-27b-nvfp4-gated-delta-rule-snap-cu
[t270]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-qwen3-6-27b-nvfp4-gdn-verify-fused-conv-kn-f32-cu
[t271]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-qwen3-6-27b-nvfp4-moe-w4a16-grouped-gemm-cu
[t272]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-qwen3-6-27b-nvfp4-nvfp4-mmq-cu
[t273]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-qwen3-6-27b-nvfp4-q2-0-mmq-cu
[t274]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-qwen3-6-27b-nvfp4-q4k-mmq-cu
[t275]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-qwen3-6-27b-nvfp4-q4k-quantize-cu
[t276]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-qwen3-6-27b-nvfp4-w4a16-gemm-cu
[t277]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-qwen3-6-27b-nvfp4-w4a16-gemm-v2-cu
[t278]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-qwen3-6-27b-nvfp4-w4a4-gemm-cu
[t279]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-qwen3-6-35b-a3b-nvfp4-gated-delta-rule-cu
[t280]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-qwen3-6-35b-a3b-nvfp4-gated-delta-rule-wy17-cu
[t282]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-qwen3-6-35b-a3b-nvfp4-moe-w4a16-grouped-gemm-cu
[t283]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-qwen3-6-35b-a3b-nvfp4-vision-encoder-cu
[t284]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-qwen3-6-35b-a3b-nvfp4-w4a16-gemm-cu
[t285]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-qwen3-8-flash-next-nvfp4-gated-norm-sigmoid-cu
[t286]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-qwen3-8-flash-next-nvfp4-hyper-connection-cu
[t287]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-qwen3-8-flash-next-nvfp4-ple-cu
[t288]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-qwen3-8-flash-next-nvfp4-qsa-indexer-cu
[t289]: docs/kernel-perf/TRADEOFFS.md#to-kernels-gb10-step3p7-flash-nvfp4-moe-silu-mul-cu
[t290]: docs/kernel-perf/TRADEOFFS.md#to-kernels-hopper-common-dense-gemm-m16-bf16-cu
[t291]: docs/kernel-perf/TRADEOFFS.md#to-kernels-hopper-common-fp8-act-quant-hopper-cu
[t292]: docs/kernel-perf/TRADEOFFS.md#to-kernels-hopper-common-gated-delta-rule-chunk-tc-cu
[t293]: docs/kernel-perf/TRADEOFFS.md#to-kernels-hopper-common-gdn-fwd-o-hopper-cu
[t294]: docs/kernel-perf/TRADEOFFS.md#to-kernels-hopper-common-gdn-recompute-wu-hopper-cu
[t295]: docs/kernel-perf/TRADEOFFS.md#to-kernels-hopper-common-moe-bucket-builder-cu
[t296]: docs/kernel-perf/TRADEOFFS.md#to-kernels-hopper-common-moe-w8a8-m16-cu
[t297]: docs/kernel-perf/TRADEOFFS.md#to-kernels-hopper-common-paged-decode-bf16-splitk-hopper-cu
[t298]: docs/kernel-perf/TRADEOFFS.md#to-kernels-hopper-common-paged-decode-fp8-splitk-hopper-cu
[t299]: docs/kernel-perf/TRADEOFFS.md#to-kernels-hopper-common-silu-mul-strided-cu
[t300]: docs/kernel-perf/TRADEOFFS.md#to-kernels-hopper-common-w8a16-gemm-m16-cu
[t301]: docs/kernel-perf/TRADEOFFS.md#to-kernels-hopper-common-w8a16-gemv-cu
[t302]: docs/kernel-perf/TRADEOFFS.md#to-kernels-hopper-common-w8a16-gemv-fused-cu
[t303]: docs/kernel-perf/TRADEOFFS.md#to-kernels-hopper-common-w8a16-gemv-ncol-cu
[t304]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-add-rms-norm-metal
[t305]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-argmax-bf16-metal
[t306]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-attention-decode-metal
[t307]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-attention-decode-bf16k-turbov-metal
[t308]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-attention-decode-turbo2-metal
[t309]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-attention-decode-turbo3-metal
[t310]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-attention-decode-turbo4-metal
[t311]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-attention-decode-turbo8-metal
[t312]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-attention-full-metal
[t313]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-attention-prefill-metal
[t315]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-causal-conv1d-decode-metal
[t316]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-causal-conv1d-update-l2norm-metal
[t317]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-conv3d-patch-embed-metal
[t318]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-dense-gemm-bf16-metal
[t319]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-dense-gemv-bf16-metal
[t320]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-embed-lookup-metal
[t321]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-gated-delta-rule-decode-metal
[t325]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-kv-cache-append-bf16k-turbov-metal
[t326]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-kv-cache-append-turbo2-metal
[t327]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-kv-cache-append-turbo3-metal
[t328]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-kv-cache-append-turbo4-metal
[t329]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-kv-cache-append-turbo8-metal
[t330]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-layer-norm-metal
[t331]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-lora-bgmv-metal
[t333]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-mlx-int8-gemm-metal
[t334]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-mlx-int8-gemv-metal
[t335]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-mlx-int8-gemv-gate-up-metal
[t336]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-mlx-int8-gemv-silu-gate-metal
[t337]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-nllb-encoder-metal
[t340]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-rms-norm-metal
[t342]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-selective-scan-decode-metal
[t345]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-softmax-topp-metal
[t346]: docs/kernel-perf/TRADEOFFS.md#to-kernels-metal-common-wht-bf16-metal
[t347]: docs/kernel-perf/TRADEOFFS.md#to-kernels-strix-hip-common-attn-prefill-cu
[t348]: docs/kernel-perf/TRADEOFFS.md#to-kernels-strix-hip-common-attn-prefill-fp8kv-cu
[t349]: docs/kernel-perf/TRADEOFFS.md#to-kernels-strix-hip-common-attn-prefill-h128-cu
[t350]: docs/kernel-perf/TRADEOFFS.md#to-kernels-strix-hip-common-attn-prefill-v47-cu
[t351]: docs/kernel-perf/TRADEOFFS.md#to-kernels-strix-hip-common-dense-gemm-bf16-cu
[t352]: docs/kernel-perf/TRADEOFFS.md#to-kernels-strix-hip-common-dense-gemm-tc-cu
[t353]: docs/kernel-perf/TRADEOFFS.md#to-kernels-strix-hip-common-moe-fp8-grouped-gemm-cu
[t354]: docs/kernel-perf/TRADEOFFS.md#to-kernels-strix-hip-common-w8a16-gemm-cu
[t355]: docs/kernel-perf/TRADEOFFS.md#to-kernels-strix-hip-common-w8a16-gemm-t-cu
[t356]: docs/kernel-perf/TRADEOFFS.md#to-kernels-strix-hip-qwen3-6-27b-nvfp4-moe-w4a16-grouped-gemm-cu
[t357]: docs/kernel-perf/TRADEOFFS.md#to-kernels-strix-hip-qwen3-6-27b-nvfp4-w4a16-gemm-cu
[t358]: docs/kernel-perf/TRADEOFFS.md#to-kernels-strix-hip-qwen3-6-35b-a3b-nvfp4-w4a16-gemm-cu

<!-- kernel_perf.py: END GENERATED -->
