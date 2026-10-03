# The circuit over the whole model lifecycle: design

Status: REVIEWED 2026-09-29, approved with the changes recorded in section 0. Section 15
(2026-10-03, PROPOSED) plans the milestone that makes the circuit run the engine and takes
precedence over sections 1 and 10 where they disagree.
Owner: metrale-circuit (branch `feat/circuit`).

Owner directive (2026-09-29): "Please make sure the circuit covers the whole lifecycle,
including prefill. Ideally, the entire model can be built from this architectural method."

Path prefixes used below: `ml/` = `crates/model-layers/src`, `me/` = `crates/model-engine/src`,
`ma/` = `crates/model-arch/src`, `mw/` = `crates/model-weights/src`, `cfg/` =
`crates/config/src`, `sv/` = `crates/server/src`, `k/` = `kernels/gb10/common`. Line numbers are
feat/circuit at 67ebc194. Three surveys of the legacy loader, state/config and prefill code are
behind this document; their findings are cited inline.

## 0. Review decisions (2026-09-29)

1. **Architecture package:** approved.
2. **IR extensions:** approved (the `t`/`s` row symbols, typed state edges with lifetimes,
   declared outputs, state ops as nodes). Addition: the state kinds cover Mamba2 too (section
   3.6).
3. **Prefill model:** approved:
   - one program per (mode, bucket), with the bucket ladder derived from rule boundaries and
     pinned by the goldens;
   - lengths as runtime arguments;
   - eager by default, with capture per bucket only after measurement;
   - unmodelled route keys fall back to legacy and are disclosed on the record.
4. **Load plan:** approved, including not reproducing the orphans, measuring and reporting the
   resident-memory drop, and checking legacy-vs-legacy requant repeatability first.
5. **Parity bar:** approved, on one condition. The legacy-nondeterminism row exclusion is
   temporary: M6 step 0 localises the cause, and the cause is fixed, or proven benign with
   evidence, before the M3 flip of any family. The exclusion rate stays pinned in the
   instrument.
6. **Milestones:** approved, with these changes:
   - **The M8 pilot is Nemotron-3.5-Lightning** (`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`),
     built with the circuit. Llama 3.x / Qwen3-dense follow as the second new family.
   - **Nemotron-H is a first-class second family in M5-M7** (sections 3.6, 5.5, 6.1).
   - **The HF reference tool (7.2) moves to right after M5.** Its first use is the Nano 30B RoPE
     check: HF's NemotronHAttention applies no RoPE, while the engine applies it. That fix is a
     standalone PR to main, not part of the circuit.
7. **Section 8 corrected:** the dense high-ISL gate is green.

Order: M1 step 2 timed leg; merge feat/circuit-venn (done, 6f9b8ba0); the W8A8 declared leg;
M5 (incl. Mamba2); the HF reference tool, then the Nano RoPE proof and fix PR; M6a. MoE M1 waits
for perf/moe-wide.

## 1. Where the circuit stands (M0-M1)

2026-10-03: out of date. Section 15.1 has the state at c92e43109: the batched verify and the
n-row draft also run as programs, the prefill nondeterminism is fixed, and the default mixed
step still runs legacy decode rows under `--forward circuit`.

The circuit already drives these, byte-identical to legacy, eager and graphed:

| Mode | Rows | Evidence |
|---|---|---|
| decode | 1 | `met circuit diff`, 4 prompts x 64 steps |
| multi_seq | 2..128, padded ladder | `met circuit diff --batch`, every width |
| verify (MTP) | K = 2, 3, 4 | `met circuit diff --verify 2,3,4`, 32 steps |
| draft (MTP head) | 1 | `met circuit diff --verify 2,3,4 --mtp`, every propose byte-identical |

Dense Qwen3.8-27B only; MoE waits for perf/moe-wide. Everything else still runs legacy code:
load, config, state allocation, prefill (single, chunked, varlen-batched, co-dispatched),
prefix-cache restore, the drafter prefill and catch-up, the batched verify and propose, and the
state operations between steps (checkpoint, rollback, commit, fold, the decode ring).

Two M1 lessons shape this design:

- **The draft program's logits were first placed in its arena.** Legacy leaves the draft logits
  in the model's logits buffer, where host code (the harness, draft confidence, shadow top-k)
  reads them. The circuit only knew that the head's `token` was an output. The fix declares
  `logits` an output of `mtp_out`. General rule (section 3.4): every buffer read outside the
  program is a declared output, and the executor binds each by name or refuses to build.
- **Legacy is not always a deterministic reference.** Legacy prefill gives run-to-run different
  last-position logits for 1-4 rows in 128, and the first MTP run of
  a process drafts from different drafter-KV history than later runs. Prefill parity therefore
  needs a legacy-repeat precondition (section 7).

## 2. End state: the architecture package

A model is built from one package: data only, when every op it uses already has kernels.

| File | Content | Status |
|---|---|---|
| `kernels/circuits/<arch>.toml` (+ `blocks/*.toml`) | Blocks, layout, draft head, dims | exists (decode-side) |
| `kernels/circuits/<arch>.config.toml` | config.json -> `ArchShape` mapping (section 6) | new, M7 |
| `kernels/circuits/<arch>.load.toml` | Tensor patterns -> transform chains -> bindings (section 5) | new, M7 |
| state section of `<arch>.toml` | Typed state edges (section 3.3) | new, M5 |
| `kernels/<hw>/common/FUSIONS.toml` | Rules per mode, incl. prefill modes | exists; grows in M6 |
| `kernels/gb10/common/KERNEL_FAMILIES.toml` | Kernel families, parameter spaces, envelopes (section 11) | on feat/circuit-venn |
| `kernels/circuits/INSTANCES.toml` | Recipe -> arch, dims, policy, plan matrix | exists |
| MODEL.toml | Serving metadata ids (tokenizer, chat template, parsers, stop tokens), referenced by id, not re-implemented | unchanged |

The pure crate turns the package into plans. It does no I/O: every file, tensor and device
effect crosses a trait the caller implements (section 12).

## 3. IR extensions

### 3.1 Modes

`Mode` today: `decode`, `multi_seq`, `verify`, `draft`. Added:

| Mode | Rows | What a plan of it does |
|---|---|---|
| `load` | none | The load plan: one node per bound weight, a chain of transform nodes, the device buffer it ends in. Not a launch program; compiled into a load schedule (section 5). |
| `prefill` | T tokens of one sequence, whole prompt, T in a bucket | Every layer at T rows; KV write for T tokens; GDN chunked scan and conv prefill; lm_head on the last row only; the state edges at the end. |
| `prefill_chunk` | T tokens of one sequence at offset P | As `prefill`, but reads the prior chunks' KV and GDN state, and writes the carry. The chunk loop is the host driver; each chunk is one program run. |
| `prefill_batch` | S sequences, sum T tokens (varlen) | Varlen metadata (cu_seqlens); per-sequence state rows. |
| `prefix_restore` | S sequences | State ops only: restore KV block tables and the GDN/conv snapshot into live slots. No layer compute. |
| `mixed` | S_p prefill sequences + S_d decode rows | One program that runs a prefill chunk and a decode step together (legacy co-dispatch). Last of the M6 steps. |
| `state` | per op | Checkpoint, rollback, commit-accepted, fold-accepted, decode-ring save/restore, zero, free. Each a small program over state edges (section 3.5). |

The fuser needs nothing new for the new modes. Rules already carry `modes` and `rows` ranges,
and prefill rules select by T bucket the way multi_seq rules select by padded width.

### 3.2 Rows become two symbols

Edges today are `n x dim`. Prefill needs tokens and sequences apart: `t x dim` for activations,
`s x dim` for per-sequence rows (the last-token hidden, the lm_head input), `1` for scalars.
The planner sizes buffers at the bucket's upper bound `t_max`, so one workspace covers a bucket
and nothing is reallocated after boot. The lm_head input is a `gather_rows` node (the last row
of each sequence, `s x hidden`): legacy does exactly that and never runs the head on all T.

### 3.3 State edges

Today state is implicit. KV pools reach emitters through bindings; GDN state through
`StepEnv::gdn_state`; snapshot slots through `GdnState::{h_steps, conv_steps}`. M5 makes every
piece of state a declared edge:

```toml
[[state]]
id = "kv"                 # per attention layer
kind = "paged_kv"
format = "{kv_dtype}"     # per-layer dtype vector from kv_high_precision_layers
shape = "blocks x block_size x kv_heads x head_dim"
lifetime = "model"        # pool; blocks are per sequence, ref-counted
per = "attention_layer"

[[state]]
id = "gdn_h"
kind = "recurrent"
format = "{ssm_h_dtype}"  # f32 | f16 | f16-pool
shape = "slots x lin_v_heads x lin_k_dim x lin_v_dim"
lifetime = "sequence"     # one pool slot per sequence, plus the padding dummy
per = "linear_attention_layer"

[[state]]
id = "gdn_h_steps"
kind = "checkpoint"
of = "gdn_h"
count = "num_drafts"      # K-1 intermediates, tiered per slot by the MTP ladder
lifetime = "verify"
```

Every kind in the state survey gets a row: paged KV (target and draft), GDN h and conv, verify
intermediates, Marconi prefix snapshots with their last-token hidden, the decode rollback ring,
the write-on-accept stash, the GDN carry buffers, the WY pointer tables, the MTP hidden save and
verify stashes, the drafter prefill capture.

Lifetimes: `model` (allocated once), `sequence` (claimed per sequence), `step`, `verify` (valid
from a verify to its commit), `snapshot` (Marconi / ring slot, owned by the prefix cache or the
scheduler). The planner checks:

- a `step` edge never survives its program;
- a `verify` edge is written only by the verify program's `state_snapshot` and read only by
  `commit_accepted` / `rollback`;
- a `snapshot` edge is written and read only by state ops.

**Sizing becomes one function.** Today the same arithmetic is written twice: in preflight
(`sv/main_modules/serve_phases/preflight.rs:40-297`) and again at allocation (`me/model/
ssm_pool.rs`, `ml/ssm_reserve.rs`, `me/factory/build/kv_budget.rs`), kept equal only by
comments. The circuit workspace is not in the reserve at all. M5 derives both from the state
edges: preflight asks the state plan for bytes, the allocator asks it for the same plan. The
refusals stay. Three facts that are process globals today become explicit plan inputs, not
reads of `OnceLock`s or env: the rollback mode, the decode-ring depth, the MTP sequence cap.

### 3.4 Declared outputs

A block's `outputs` list becomes the complete set of edges read outside the program, each
naming the model buffer it binds to (`logits`, `tokens`, `hidden_last`, `draft_hidden`). The
executor refuses a program with an unbound output, or an external buffer that is not a
declared output. This is the generalisation of the M1 draft-logits fix.

### 3.5 State ops as nodes

`StateSnapshot` exists (the verify conv snapshot). Added, each an op with explicit state-edge
inputs and outputs, so the legacy copies become circuit nodes, not host code:

| Op | Legacy site | Kernel or copy |
|---|---|---|
| `state_restore` | `commit_accepted_prefix_dispatch` (`me/model/trait_impl/async_chkpt.rs:204-330`), rollback (`verify_a_ssm.rs:89-120`) | D2D copy, pitched 2-D per pool family |
| `state_fold_accepted` | `gdn_fold_accepted_dispatch` (`me/model/trait_impl/gdn_woa.rs:90`) | existing fold kernel |
| `state_zero` | `zero_slot`, `reset_slot` (`me/model/ssm_pool_slots.rs`) | memset |
| `prefix_snapshot_save` / `_restore` | Marconi `save` / `restore` (`me/model/ssm_snapshot.rs:117-365`) | copy (+ widen f16 to f32) |
| `ring_save` / `ring_restore` | `ssm_snapshot_decode.rs:42-122` | copy |
| `kv_trim` | lowering `seq_len` | host metadata only |

The secondary stream and its event stay executor mechanics: a state program declares whether it
may overlap the next compute program, and the executor places it.

### 3.6 Nemotron-H: Mamba2 state (review decision 2)

A survey of the legacy Nemotron-H code is behind this section; its findings are cited inline.

**Mamba2 state kinds:**

| State | Format | Shape per slot and layer | Legacy |
|---|---|---|---|
| `mamba_h` | f32 | `mamba_heads x mamba_head_dim x ssm_state` (state fastest); 2 MiB per layer for Nano and Lightning | the shared `SsmStatePool` slot (`me/model/ssm_pool.rs:190`); 'M' layers are `LayerType::LinearAttention` (`cfg/dispatch.rs:237`) |
| `mamba_conv` | f32 | `(mamba_heads*mamba_head_dim + 2*ssm_groups*ssm_state) x conv_kernel`, oldest first | same pool; sized by `ssm_conv_state_bytes` (`cfg/methods.rs:253-272`) |
| `mamba_h_steps`, `mamba_conv_steps` | as above | K-1 / K per verify slot | **absent in legacy** (below) |

**Shared state machinery.** Nemotron's state kinds reuse GDN's machinery (pool slots, the padding
dummy, Marconi snapshots, the decode ring). Snapshots are opaque byte blobs, so they are already
correct for Mamba2. The state schema makes that sharing explicit: a recurrent state is `(kind,
shape expression, format)`, and the pool, snapshot and ring ops are generic over it. Legacy has
one failure of exactly that genericity. The verify checkpoint and rollback size the conv copy
with the GatedDeltaNet formula (`me/model/trait_impl/verify_a_ssm.rs:55-62,133-140`,
`async_chkpt.rs:58,118,292`), which evaluates to **0 bytes** for Nemotron. One sizing function
per state edge removes that class of bug.

**Speculative decoding over Mamba2 does not exist in legacy:**

- the MTP weights are never loaded (`ma/weight_loader/nemotron.rs:414-421`);
- nothing writes Mamba2 per-row intermediates;
- the conv copy is 0 bytes (above);
- no guard refuses a proposer on this arch.

Lightning's MTP pilot needs three new pieces:

1. a multi-row Mamba2 update that writes the per-row h and conv snapshots (the `state_snapshot`
   nodes already in `nemotron_h.toml`);
2. the rollback and commit state ops over those edges;
3. a refusal of any proposer on a Mamba2 model until (1) and (2) have parity.

Because HF does not model MTP (`_keys_to_ignore_on_load_unexpected = ["mtp.*"]`), the draft
head's reference is checked per op against the published reference code for the MTP head, and
end to end by acceptance rate and the accuracy gates.

**Legacy numerics that differ from HF**, each a declared policy in the package, never silent:

| Legacy behaviour | Where |
|---|---|
| Mamba2 decode clamps every h element to +-200 | `k/mamba2_ssm_decode.cu:116`; the prefill kernels do not clamp |
| Every 64 decode tokens, each head's state norm is rescaled to at most 200 | `me/model/trait_impl/decode_a3.rs:30-38` (the circuit executor already refuses it) |
| dt is clamped to [1e-9, 1e9] | HF clamps dt >= `time_step_min` in its torch path and uses `time_step_limit = (0, inf)` in its kernel path |
| The SSD chunk size is 64 | the config says 128. The SSD kernel does not fit shared memory at `ssm_state` 128 (`ml/ops/ssm_ssd.rs:95-119`), so Nano and Lightning prefill with the sequential Mamba2 kernels |
| Router logits are BF16 | HF computes them in FP32 |

The Lightning package states which of these it keeps. Its reference is HF, not legacy (it is a
new family), so the default is HF's numerics. A clamp kept for stability is an opt-in policy
with a measured accuracy result.

## 4. Prefill: dynamic lengths, chunk programs, graph policy

### 4.1 What legacy prefill is (survey summary)

- **Entry points.** Seven feed the scheduler (`me/traits/model/forward.rs`):
  - single-pass `prefill` (`me/model/trait_impl/prefill_a.rs:42`);
  - chunked `prefill_chunk` (`prefill_b.rs:66-318`);
  - two-phase (`prefill_c.rs:42`): per GDN layer, a phase over all chunks, then one full-range
    recurrence;
  - multi-stream batched (`prefill_b/batch.rs:38`), with a kernel-batched varlen path
    (`batch_kernel.rs:60-476`);
  - fused mixed prefill+decode (`decode_b.rs:43`);
  - the tail split;
  - the drafter prefill.
- **Lengths are host scalars.** Every launch bakes its token count and computes its grid from
  it. Only positions, slots, block tables, `seq_len` and `cu_seqlens` live on the device,
  uploaded once per chunk from pinned staging into `scratch()`
  (`prefill_b/upload_meta.rs:43-179`).
- **Prefill is never captured.** Every prefill `ForwardContext` sets `graph_capture: false`.
  Prefill also allocates per-sequence page metadata lazily (`me/model/impl_a2.rs:97-107`) and
  host-syncs mid-pass (`prefill_b.rs:233`).
- **Routes switch on the row count at many boundaries:**
  - FLA chunk count (64-token chunks), and the regresident / WY4 / persistent / split4 arms;
  - NVFP4 MMQ tile (≤16, ≤32, ≤64, else 128);
  - w4a16 small-M (m ≤ 64);
  - W8A8 (≤ 256 rows);
  - the BA-gates Hopper twin (< 96 tokens);
  - MoE grouped path (> 64 rows);
  - the attention route: contiguous on a single stream's first chunk, paged afterwards.
- **A prefix-cache restore changes the route.** It sets `gdn_exact_replay`, which forces
  WY4/regresident instead of FLA (`qwen3_ssm/trait_prefill_recur.rs:126-130`).
- **lm_head runs on the last row only** (`prefill_a.rs:473-479`, `finalize_last.rs:122-126`).

### 4.2 The circuit's prefill model

1. **One program per (mode, bucket).** A bucket is a token-count range `[t_lo, t_hi]`. The
   bucket ladder is derived, not chosen: it is the union of every row-range boundary of every
   rule that can apply in the mode. So a plan is exact for every T in its bucket: no rule
   changes inside a bucket. For today's rules that gives roughly
   1-16, 17-32, 33-64, 65-95, 96-128, 129-256, 257-4096 and 4097-`max_prefill_tokens`. The
   golden-plan test pins the ladder, so a new rule boundary that splits a bucket shows up as a
   plan diff.
2. **Lengths stay runtime arguments, as in legacy.** `StepEnv` gains `tokens`, `seqs`, the
   chunk offset and the host copy of `cu_seqlens`. Emitters compute grids from them at launch,
   exactly as legacy does. Compiled launch closures are bucket-specific, but not T-specific.
3. **Device metadata** keeps legacy's layout and upload: one pinned staging pack, one H2D per
   chunk, the same `scratch()` offsets. It becomes a declared `metadata` edge of the mode, so
   the upload is a node (a copy), not code in a dispatch function.
4. **Route facts that are not row counts become plan keys:**
   - first chunk vs a later chunk (contiguous vs paged attention);
   - after-restore (FLA forbidden);
   - single stream vs varlen batch.

   Each key is a policy input to `fuse`, exactly like the recipe settings today. A key the
   circuit does not model (e.g. two-phase) keeps the legacy path under `--forward circuit`,
   and the fallback is reported on the record, never silent.
5. **The chunk loop, prefix lookup, snapshot scheduling and wave planning stay host code** in
   the driver (the scheduler-facing `Model` methods). The circuit supplies one program run per
   chunk, plus the state ops (`prefix_snapshot_save`, `state_zero`, ...) as programs of their
   own.

### 4.3 CUDA-graph policy

**Prefill stays eager.** Three reasons:

- Legacy never captures it, so eager is the byte-parity reference.
- Grids depend on T, so capture would need either one graph per exact T or kernels that read T
  from the device and mask up to `t_hi`. The second changes kernels and needs the stability
  gate.
- The launch share is small. About 15 launches per layer × 64 layers ≈ 1k launches, a few
  milliseconds against a TTFT of 100 ms to seconds.

Capture is allowed later only per bucket and only where measured launch overhead is a
significant share of that bucket's TTFT (small-T chunks of the mixed step are the candidate). It
also needs the host syncs and lazy allocations removed from the path first. A captured bucket is
a plan attribute with its own parity run (graphed vs eager must be byte-identical, as in M1).

### 4.4 Modes in order of delivery

1. `prefill`, first chunk, single stream (dense).
2. `prefill_chunk` (paged attention, GDN state carry, KV floor).
3. The MTP drafter prefill: a draft-head block at T rows, no attention, KV write only
   (`ml/layers/mtp_head/prefill.rs:41`).
4. `prefix_restore` + after-restore plans.
5. `prefill_batch` (varlen), then the kernel-batched variant.
6. `mixed`: last, because legacy's fused path skips prefix lookup and snapshot save
   (`decode_b.rs:200-202`), and codispatch is known to cost warm TTFT.
7. MoE prefill (sorted grouped GEMM, FP8 pointer tables) once perf/moe-wide has landed.

Two-phase prefill (layer-major over chunks) is a schedule-order change. It is modelled only if
measured TTFT says it matters for a gated instrument.

## 5. Load plan

### 5.1 What the legacy loaders do (survey summary)

Loader choice: `qwen3_5` with no experts goes to `Qwen35DenseWeightLoader`; `qwen3_5_moe` and
`qwen3_6_moe` go to `Qwen35WeightLoader` (`me/factory.rs:64-75`). About 5.7k lines are
family-specific, plus about 12k of shared load-path code.

- **Formats are chosen by tensor names and dtypes, not by the declared plan.** `quantized_any`
  (`ml/weight_map/nvfp4_detect.rs:185-293`) resolves a per-key variant. `DeclaredPrecisionPlan`
  (`cfg/precision_plan.rs:191-208`) only stamps the activation (A4 / Wide), gates the W8A8
  install and picks the lm_head.
- Allocation is one `cuMemAlloc` per tensor, with no weight arena. H2D copies are synchronous,
  and the loaders make many small D2H syncs (scale2 reads, absmax).
- Defaults (`declared`, experts `fp8`, MTP `bf16`, TP=1) produce:
  - **Unsloth 27B:**
    - attention and GDN: FP8 per-channel -> BF16 -> NVFP4 (Wide), plus a W8A8 per-row install;
    - FFN layers 0-55: zero-copy compressed-tensors NVFP4 (A4), plus twins and the MMQ repack,
      with the twins then freed;
    - FFN layers 56-63: FP8 -> NVFP4 (Wide), plus W8A8;
    - lm_head: native FP8 + W8A8, plus a draft-only NVFP4 head when speculating.
  - **35B FP8:**
    - attention and GDN: native FP8;
    - routed experts: native FP8 pointer tables;
    - shared expert: FP8 -> NVFP4;
    - router and lm_head: BF16.

### 5.2 Transform vocabulary

Each binding is `source tensors -> [transform...] -> device buffer(s)`. The vocabulary covers
every transform the survey found in the two families:

| Transform | Parameters | Legacy site (examples) |
|---|---|---|
| `store` | none (zero-copy store pointer) | norms, embed, conv1d (`ma/weight_loader/qwen35_dense.rs:331`) |
| `cast` | from, to, rounding (`rne`, `trunc`) | F16->BF16 host RNE (`mw/weights.rs:126`); F32->BF16 trunc (`ml/weight_map/model_a.rs:220-245`); BF16->F32 widen for `A_log`/`dt_bias` (`:250-285`) |
| `dequant` | from (`fp8`, `nvfp4`), scale layout (per-row, block r x c, tensor) | `dequant_fp8_blockscaled_bf16` (`ml/weight_map/quant_helpers.rs:37-246`) |
| `quant` | to `nvfp4`, algorithm (`absmax`, `mse`), global-scale rule | `quantize_to_nvfp4` (`ml/weight_map/loaders_fp8.rs:147-272`); MSE for NVFP4 expert tiers (`ma/weight_loader/qwen35/load_layers.rs:89-94`) |
| `concat_rows` | inputs, requires-equal-scale2 | GDN `[qkv\|z]` (`ma/.../qwen35_dense/gdn_dequant.rs:200-209`); MoE FP8 concat with scale rows |
| `interleave` | pattern | GDN `in_proj_b`/`in_proj_a` into `[2*nv, K]` per key group (`ml/weight_map/fp8_lut.rs:384-417`) |
| `transpose` | element (`u8` NVFP4 pairs, `fp8`), scale transpose | NVFP4 twins (`ml/weight_map/quantized/transpose.rs:30-119`); FP8 prefill twins |
| `scale_widen` | block scale -> F32 grid, BF16 `[N,1]` -> F32 | native FP8 overlays (`ml/weight_map/loaders_fp8.rs:26-138`); W8A8 install (`ma/.../w8a8_install.rs:52-186`) |
| `scale_fold` | `scale2 = 1/global_scale` | compressed-tensors NVFP4 (`ml/weight_map/quant_helpers.rs:324`) |
| `repack` | target (`block_nvfp4` MMQ, `q4k`, `int8`) | FFN MMQ (`ml/dense_ffn_load.rs:98-212`) |
| `swizzle` | `cutlass_sfb` | expert scales under the CUTLASS grouped path (`ml/moe/helpers_a.rs:380-461`) |
| `slice` | axis, range (expert e of a fused tensor, TP shard) | fused experts (`ml/weight_map/ssm_qwen35.rs:128-227`) |
| `ptr_table` | element kinds (addresses + per-expert scale2) | expert pointer tables (`ml/moe/ptr_table_build.rs:19-134`) |
| `rotate` | Hadamard (TQ+) | `ma/.../tq_plus_weight_rotation.rs:22-55` |
| `tie` | source binding | lm_head <- embed when absent |

Selectors (the flags and policies that choose an arm) become explicit plan inputs, resolved once:
`--weight-quantization`, `--expert-quantization`, `--mtp-quantization`, `--lm-head-dtype`,
`KernelCaps`, and the dense-FP8 route plan. A lever that is env-only today
(`METRALE_DENSE_FP8`, `METRALE_NO_GDN_FP8_PREFILL`, `METRALE_FP8_ROWWISE`, the MoE `HOLO_*` set, ...)
is either a declared plan input or refused while the circuit loader is selected. There is no
silent third state.

The plan also declares each binding's derived copies (twins, MMQ, FP8 prefill copies) with the
mode that needs them. So "the prefill GEMM reads the transposed twin" is a checked edge, not a
convention, and a copy nothing reads is a plan error.

### 5.3 What the plan will not reproduce

The survey found accidental behaviour a plan must not copy:

- **Orphaned buffers:**
  - the BF16 intermediates of the compressed-tensors attention arms
    (`ma/.../qwen35_dense/attn_arms.rs:143`);
  - the first B/A interleave (`gdn_dequant.rs:97`);
  - the predequantised `out_proj_fp8` that the FP8 cast overwrites (`:398-405`);
  - the shared expert's double FP8 scale load (`ml/weight_map/ssm_qwen35.rs:337-354`).
- **Store pointers freed while they stay in the map** (`nvfp4_detect.rs:274`,
  `ssm_qwen35.rs:271-278`).

Parity therefore compares the **bound weights**, not the allocation list. Resident memory then
drops by the orphans, which is measured and reported, not hidden.

### 5.4 Load parity instrument

No weight hash exists today. The instrument is `met circuit diff --load`:

1. Load legacy, and walk every layer's `CircuitBindings` slot map (already complete for the
   decode modes: `ml/circuit_exec/bindings.rs`).
2. For each bound weight, hash the pointee bytes. For a pointer table, hash each pointee in
   table order, not the table (its addresses differ run to run).
3. Load via the plan and repeat.
4. Report per slot: equal, or differing bytes.
5. A detection control flips one byte of one weight after load.

**Open question: NVFP4 requant may be nondeterministic.** Absmax uses an atomic
(`nvfp4_global_absmax`), so legacy-vs-legacy equality is the first check. If legacy-repeat
differs, requantisation joins the prefill nondeterminism finding and is fixed before load parity
is claimed.

### 5.5 Nemotron-H load transforms

The vocabulary in 5.2 covers Nemotron-H with one addition, `input_scale` as a bound tensor.

- **Mamba2 in_proj / out_proj:** F8_E4M3 with a scalar `weight_scale` and an `input_scale [1]`.
  - Legacy widens the scalar over a 128-block grid (`scale_widen`,
    `ml/weight_map/loaders_fp8.rs:114-130`) and **never reads `input_scale`**. It runs W8A16.
  - The circuit's declared plan is W8A8 with a static activation scale (`act_quant` nodes bound
    to `input_scale`). That is a numerics change relative to legacy, which is why this family's
    reference is HF, not legacy.
- **Experts (routed and shared):** NVFP4 `U8` weights with FP8 `weight_scale` and `weight_scale_2`,
  loaded as stored (`store` + `scale_fold`). They are ungated: `up_proj` and `down_proj` only.
- **Attention and the MTP head:** BF16 (`store`).
- **conv1d:** the weight is stored as is; the bias gets `cast` BF16->F32.
- **`A_log`, `D`, `dt_bias`:** `cast` BF16->F32. Legacy recomputes `-exp(A_log)` in every
  kernel; a precompute is a later fusion rule, not a load change.
- **Norms: plain weights (`x * w`, not `x * (1 + w)`).**
  - Legacy gets this right only by accident: the Nemotron kernel tree compiles another model's
    `rms_norm.cu`, and `ships_vanilla_norm_weights` does not list `nemotron_h`
    (`ml/lib.rs:57-60`).
  - The circuit states it per node (`params = { weight_form = "plain" }`). The `rms_norm`
    emitter refuses a node whose weight form it does not implement.
- **Router:** the gate is F32 in the checkpoint; legacy casts it to BF16 (`ml/weight_map/
  nemotron.rs:228-231`). The package keeps F32 (HF's numerics). `e_score_correction_bias` is F32.
- **MTP head:** two modules. `mtp.layers.0` holds enorm, hnorm, `eh_proj`, the norm and the
  attention; `mtp.layers.1` holds the MoE and `final_layernorm`. The binding schema needs
  per-block module paths (a block-level `module` key), replacing today's single `draft_module`.
  `nemotron_h.toml` works around this with explicit bindings.

## 6. Config mapping

Today `cfg/dispatch.rs` maps `model_type` to a parser, `text_config` is required for the qwen3_5
family, and `ModelConfig` accepts any keys. The survey found three kinds of drift that a
declarative mapping must refuse, not reproduce:

- **Silent overrides.** `norm_topk_prob` is forced on (`cfg/dispatch.rs:159`). `attn_gated`
  is derived from `model_type`, and `attn_output_gate` is never read (`:161`). `model_type` is
  rewritten (`qwen3_5_moe` -> `qwen3_6_moe`, `holo3_1_moe`: `:182-203`).
- **Math-changing keys that are ignored:**
  - `rope_scaling` / `rope_type` on this path;
  - `output_gate_type` for the GDN norm (the checkpoint says `swish`, the code hard-codes SiLU);
  - `mamba_ssm_dtype`;
  - `mtp_use_dedicated_embeddings`.
- **Facts taken from elsewhere:**
  - MTP presence comes from the weights (`mtp.fc.weight`), not from `mtp_num_hidden_layers`;
  - the norm style `(1+w)` is keyed on a `model_type` list (`ml/lib.rs:47-61`);
  - MODEL.toml `mtp_layers = 0` contradicts the checkpoint.

The schema (`<arch>.config.toml`):

```toml
[source]
nest = "text_config"                 # required for this arch
[[key]]
hf = "linear_num_value_heads"
dim = "lin_v_heads"                  # into ArchShape.dims
required = true
[[key]]
hf = "norm_topk_prob"
param = "routing.renormalize"
[[key]]
hf = "rope_scaling"
refuse_unless = ["null"]             # math-changing, not modelled yet
[[key]]
hf = "output_gate_type"
param = "gdn.norm_gate"
values = { swish = "silu" }          # the mapping states the equivalence it relies on
[unknown]
policy = "refuse_math"               # listed ignorable keys pass; any other unknown key refuses
ignorable = ["architectures", "torch_dtype", "transformers_version", "..."]
```

- **Every key is classified:** a dim, a param, refused-unless, or ignorable. An unclassified
  key in a checkpoint refuses the build, and the message names the key.
- **Every `ArchShape` dim is sourced**, so the dims in `INSTANCES.toml` become a check of the
  mapping, not a second source.
- **Legacy parity:** the mapping produces an `ArchShape` plus params. A test compares them field
  by field with `parse_config` for every cached checkpoint of the family.
- **Absent keys take a stated default.** An absent key takes the default of the Hugging Face
  class for the arch, and the mapping cites that default. Example: `norm_topk_prob` is absent
  from both Qwen checkpoints (checked in the local snapshots), so the forced-on legacy value is
  right only if the transformers default is true. The mapping states that and cites the source,
  instead of relying on it silently.
- **Divergences are recorded, not copied.** Where legacy diverges from the checkpoint (the drift
  kinds above), the mapping follows the checkpoint only if the two are equivalent for it. Two
  equivalences hold today: `output_gate_type = "swish"` is SiLU, and `mamba_ssm_dtype =
  "float32"` matches the engine's f32 h state. Any non-equivalent value is refused and listed in
  the package.

### 6.1 Nemotron-H config mapping

Legacy cannot parse the Lightning config:

- **`moe_latent_size: null` is a parse error.** The field is a plain `usize` with
  `#[serde(default)]` (`cfg/model_config.rs:168-169`), and serde rejects an explicit null.
- **The layer schedule is missing.** Lightning declares `layers_block_type`, and the parser
  reads that key only for `nemotron_h_puzzle` (`cfg/dispatch.rs:212,245,384-398`). The layers
  stay empty, and the loader builds zero layers.
- **`hybrid_override_pattern` panics on any character other than M/E/`*`** (`cfg/dispatch.rs:233-243`).

The mapping for `nemotron_h`:

- **Layer schedule:** `layers_block_type` (a list of `mamba`/`moe`/`attention`) or
  `hybrid_override_pattern` (a string). Exactly one must be present; each is mapped by a table,
  and an unknown value is refused.
- **Dims:**
  - `mamba_num_heads` -> `mamba_heads`, `mamba_head_dim`, `ssm_state_size` -> `ssm_state`,
    `n_groups` -> `ssm_groups`, `conv_kernel`;
  - `n_routed_experts` -> `experts`, `num_experts_per_tok` -> `top_k`, `moe_intermediate_size`,
    `moe_shared_expert_intermediate_size` -> `shared_inter`;
  - `head_dim`, `num_attention_heads`, `num_key_value_heads`.
  - Checks legacy lacks: heads divisible by `n_groups` (the kernels assume it), and
    `d_inner == heads * head_dim` (`expand` is informational).
- **Epsilon:** `layer_norm_epsilon` (legacy reads only `norm_eps`, else a silent 1e-6; the HF
  default is 1e-5). A config carrying both with different values is refused.
- **Params:**
  - `routed_scaling_factor`, `norm_topk_prob`;
  - `n_group` / `topk_group`: refused unless 1;
  - `time_step_limit` / `time_step_min` (the dt clamp policy);
  - `mamba_ssm_cache_dtype` (must match the h state format);
  - `use_conv_bias` (must be true);
  - `mamba_proj_bias` / `attention_bias` / `mlp_bias`: refused unless false;
  - `mlp_hidden_act` / `mamba_hidden_act`: must match the circuit's `relu2` / SiLU;
  - `num_nextn_predict_layers` and `mtp_layers_block_type` (the draft head's block list).
- **Attention: NoPE.** `rope_theta` and `partial_rotary_factor` appear in the config, but the
  `nemotron_h` reference applies no rotary embedding. The mapping classifies both as ignorable
  **for this arch**, with that reason, and the circuit has no `rope` node. Legacy reads them and
  applies RoPE (the bug in 7.2 below; the config test `cfg/tests/nemotron.rs:60` asserts
  `rotary_dim() == 128`).
- **Ignorable with a reason:** `chunk_size` (a kernel tiling choice, not math),
  `rescale_prenorm_residual` (initialisation only), `residual_in_fp32 = false` (it matches the
  BF16 residual).

### 6.2 Interface for the hardware axis (2026-09-30)

The model axis is one pure entry point in `metrale-circuit`, module `checkpoint`, re-exported at
the crate root:

```rust
pub struct QuantMetadata<'a> {
    /// `hf_quant_config.json` (ModelOpt), verbatim; read only when config.json has no
    /// `quantization_config` (the serve rule).
    pub hf_quant_config: Option<&'a str>,
}

/// Every edge at the checkpoint's DECLARED formats.
pub fn instantiate_from_checkpoint(config_json: &str, quant: QuantMetadata<'_>)
    -> Result<Circuit, CheckpointError>;

/// Arch, shape, params, KV-cache format and circuit, with the edge formats `serve` gives.
pub fn resolve_checkpoint(config_json: &str, quant: QuantMetadata<'_>, serve: &ServePrecision)
    -> Result<ResolvedCheckpoint, CheckpointError>;

/// The config mapping alone (arch, shape, params), without instantiating.
pub fn map_checkpoint(config_json: &str) -> Result<MappedConfig, CheckpointError>;

pub enum ServePrecision {
    Declared,
    Policy { tier: String, caps: Vec<String>, engine: Vec<(String, LinearFormats)> },
}

pub struct ResolvedCheckpoint {
    pub arch: String,        // qwen3_5 | qwen3_6_moe | nemotron_h | dense_gqa
    pub model_type: String,
    pub shape: ArchShape,
    pub params: BTreeMap<String, String>,   // math params as JSON text: rope.*, rms_norm_eps, ...
    pub kv_cache: Option<Format>,           // declared KV-cache format; None = 16-bit
    pub circuit: Circuit,
}
```

`CheckpointError` names why a checkpoint has no circuit:

- `UnknownModelType`: no map serves the `model_type`.
- `Map(UnmappedKey | Refused | MissingKey)`: the config map rejects the config.
- `Quant`: a declared format the circuit has no edge or weight format for, or malformed quant
  metadata.
- `Circuit`: the circuit does not instantiate.

The circuit, block and config-map TOMLs are embedded, so the caller passes file text only.
Fusing is unchanged: `fuse(&resolved.circuit, &rules, &available, &policy, mode, rows)`.

**Which checkpoints resolve.** The coverage, over every checkpoint cached on the three boxes plus
the G1 and Nemotron-H ones fetched from the Hub (`crates/circuit/tests/checkpoints.rs`):

- Served:
  - the dense Qwen3.5 / 3.6 / 3.8 hybrids (`qwen3_5`);
  - the Qwen3.5 / 3.6 MoE hybrids (`qwen3_6_moe`);
  - Nemotron-3 Nano, Nemotron-3 Super (latent MoE) and Nemotron-3.5 Lightning (`nemotron_h`);
  - Llama 3.1, Qwen3 dense, Qwen2.5 and Mistral Small (`dense_gqa`).
  - Holo-3.1, which has no MTP layer, so its circuit has no draft head (`draft_when = "mtp"`).
- Refused, with the reason named:
  - The DFlash drafts: they have unmapped keys.
  - Every other model type.

**Two mechanisms the config maps rely on:**

- **Switch dims.** An integer dim that is 0 or 1, read by a node's `when` (absent node = identity)
  or `params_when`, and by a circuit's `draft_when`.
- **Layout and draft variants.** `[layout.when]` replaces a layer kind's blocks while a switch
  holds, and `draft_variant` replaces the draft list. A draft entry `template@module` reuses a
  main-stack template at another module, e.g. the Nemotron-H draft MoE is `moe@mtp.layers.1`,
  or `moe_latent@...` under `moe_latent`.
- **Quantizer insertion.** A projection whose declared activation is quantized gets its
  `act_quant` node from the precision. A static per-tensor scale is bound to that projection's
  `input_scale` and is not shared.

## 7. Parity instruments per phase

### 7.1 Legacy byte parity (existing families)

Every instrument follows the M1 pattern:

- runs, in order: reference, legacy-repeat, circuit-reference, circuit, then a detection
  control under the last forward;
- the verdict **fails** when the control shows no difference;
- the verdict **fails** when the reference did not exercise the case (e.g. no partial accept).

| Phase | Instrument | Compared |
|---|---|---|
| config | unit test over every cached checkpoint of the family | `ArchShape` + params field by field vs `parse_config` |
| load | `met circuit diff --load` | pointee hash of every bound weight per slot (section 5.4) |
| state | unit test + boot log | `StatePlan` bytes per family vs the legacy allocation ledger; preflight reserve vs allocation (must be equal, not "close") |
| prefill | `met circuit diff --prefill` | last-row logits, then per layer the GDN h and conv state and the sequence's KV blocks, then the draft KV rows |
| prefill_chunk | same, `--max-prefill-tokens` forced small | as above, across many chunk boundaries, plus the later decode logits |
| prefill_batch | same, `--batch` | per row, as above |
| prefix_restore | same, a primed re-send | restored state bytes, then logits |
| mixed | same, with decode rows in flight | decode rows' logits, then the prefill rows' state |
| state ops | covered by the verify/draft diff (commit/rollback) and a restore diff | state bytes after the op |

- **Prompt lengths straddle every bucket boundary.** The prefill instruments run T = t_lo and
  t_hi of every bucket, so each plan runs at both ends of its range.
- **Legacy-repeat precondition.** Legacy prefill is not always repeatable (section 0):
  - a row whose legacy-repeat differs from the reference is left out of the circuit comparison
    and reported, as `--batch` does today;
  - a phase passes only if at most the documented rate of rows is left out, and the
    instrument pins that rate;
  - the exclusion is temporary: the cause is fixed, or proven benign with evidence, before
    any family's M3 flip (review decision 5).
- **Step 0 of M6 localises that nondeterminism.** The prefill survey found every affected row in
  the recorded data has a prompt longer than 64 tokens. That is where the FLA chunk count, the
  MMQ tile and the small-M route all switch. Existing levers A/B each suspect cheaply
  (`METRALE_GDN_PIPE=0`, `METRALE_NO_GDN_FWD_O_MMA8`, `METRALE_NO_GDN_FLA=1`,
  `METRALE_FFN_SMALLM=0`). The fix itself is separate work (owner order: record, do not fix
  now), but knowing the cause decides whether the exclusion rule above is sound.

### 7.2 New models: Hugging Face tolerance parity

There is no legacy path to match for a new model. The reference is Hugging Face transformers on
the same prompts:

- **The script is generic.** One reference script, `tools/circuit/hf_reference.py`, is keyed by
  the package, never by a family. It writes last-position logits per prompt, and optionally
  per-op hidden rows via forward hooks for localisation. The repo has only family-specific
  scripts today (the FP8 drift and qwen4_exp scripts under `bench/`).
- **Tolerance is declared per format in the package.** For example: BF16 weights need top-1
  agreement on every position and max |Δlogit| under a stated bound; quantized formats (NVFP4,
  FP8) are bounded by KL and top-k overlap. The reference runs at the checkpoint's dequantized
  precision, so the tolerance measures only the engine's numerics, not the quantization itself.
  The bounds are set from the existing families first: the circuit-built Qwen models must pass
  their own HF check before a new model is held to it.
- **Then the accuracy gates.** The model must pass the accuracy gates of its class: BFCL, and
  agentic-webserver for precision-lowering choices (the W4A4 accuracy bar).
- **First use: the Nemotron-3-Nano NoPE bug (a standalone PR to main).**
  - **The bug.** Legacy builds Nemotron-H attention with `Qwen3AttentionLayer::new_ungated`
    (`ma/weight_loader/nemotron.rs:336-351`) and applies full RoPE (`rotary_dim` 128, theta
    10000) in decode, batched decode, verify and both prefill paths. HF's `NemotronHAttention`
    applies none. No config value can switch it off: `rotary_dim = 0` hits
    `assert!(rotary_dim > 0)`.
  - **The proof.** HF logits vs the engine on Nano, on the same prompts: all six attention
    layers are wrong from the first attention layer on, so the divergence shows up in the
    last-position logits and in per-layer hidden rows via the hooks.
  - **The fix.** A `no_rope` flag on the layer, set by the Nemotron loader. It skips only the
    RoPE step in `attention_forward_rope`, `ms_phase_rope`, `cache_skip_rope` and
    `prefill_paged_rope_cache_write`, keeping their KV writes. It excludes the fused FP8 KV
    writer, and updates the config test.
  - **The regression tests.** A mock-backend test asserts no rope launch in any attention path.
    A loader test asserts every Nemotron attention layer has `no_rope`. The HF parity re-run
    shows the logits within tolerance.


## 8. Keeping the TTFT and high-ISL gates green

The instruments:

- **Cold and warm TTFT:** 256, 1024 and 4096 tokens × 12. The gate fails at median +3% or p90
  +5% vs the same-box baseline (`crates/bench/src/benchmarks/ttft.rs:284-310`).
- **High-ISL:** a 32k prompt, cold and warm, with ceilings pinned to vLLM's same-box one-shot
  TTFT (`kernels/gb10/qwen3.8-27b/BENCH.toml:2280-2337`;
  `kernels/gb10/qwen3.6-35b-a3b/BENCH.toml:486-540`).

**Green means inside each gate's own bounds, as for every other gate.** The dense 27B passes
high-isl-ttft-cold today: the banked records on origin/main are
`.benchmarks/high-isl-ttft-cold/2026-09-28-c4e6fde56c.json` (median 17282 ms, PASS) and
`2026-09-29-e55829e49b.json` (16799 ms, PASS), both under the 24.1 s vLLM ceiling. Campaigns
#39 and #48 certified all 17 gates. TODO for a later PR: the BENCH.toml note text
(`kernels/gb10/qwen3.8-27b/BENCH.toml:2246-2262`) still describes the 2026-09-27 Phase 1
measurement (45.4 s) and needs refreshing.

The rules:

1. **Legacy stays the default** until a family's prefill modes pass parity and a same-box A/B:
   legacy vs circuit, interleaved, 2 reps, on every TTFT instrument, with a same-night control.
   The circuit must sit inside the gate's own bounds. Byte-identical plans run the same kernels,
   so any difference is host overhead (closures, metadata packing), measured as enqueue time.
2. **No new host sync and no new allocation** on the prefill path. The executor's prefill
   workspace is sized at boot from the `StatePlan` budget (section 3.3). Before the flip, both
   the legacy arena and the circuit workspace are resident; their sum is measured against the
   KV budget, and prefill edges bind to arena buffers where a second copy would not fit.
3. **Warm TTFT needs `prefix_restore`.** Until that mode exists, a warm request under
   `--forward circuit` restores with the legacy path. The record says so.
4. **The mixed step stays opt-in**, as the codispatch lever is today (it costs about 4.8%
   median warm TTFT). 2026-10-03: wrong. The fused mixed step is the default scheduler path
   whenever a chunk continues with decodes active; only codispatch is opt-in (section 15.4).
5. **Prefill fusions (M2-style) are measured on these instruments first.** A fusion that helps
   decode but costs TTFT is gated per mode: rules have `modes`, so it can apply to decode and
   not to prefill.

## 9. Certification cost

- **Any `crates/**` change** outside the bench-driver exclusions invalidates all 17 required
  gates (`crates/bench/src/gate/coverage.rs:33-41, 817-895`). That is 15 plain gates plus the
  BFCL shards: 27 units on three nodes, about 2 h 15 min of wall time and about 4.5-5 GPU-hours
  (`AGENTS.md:86-90`).
- **A change only under `kernels/`** is excused by the closure hash. So is a package TOML edit
  whose closure hash does not change a gate's served code; most package edits will change it.
- **Plan: one campaign per landing stack, not per milestone step.**
  - Campaign 1: M1 remainder (dense verify + draft, W8A8 declared leg) with M5 (state schema).
  - Campaign 2: M6 dense prefill modes (prefill, chunk, drafter prefill, restore).
  - Campaign 3: M6 batch and mixed, plus M1 MoE and MoE prefill (after perf/moe-wide).
  - Campaign 4: M7 (load plan, config mapping) with M8's pilot family.
  - Campaign 5: the M3 flip-and-delete.

  Each lands behind `--forward legacy`, so gate numbers should not move. A campaign that moves
  one is a finding before it is a merge.
- **Parity and A/B legs run outside campaigns,** on a box no campaign holds.

## 10. Milestones

2026-10-03: section 15.8 replaces this table's order for M5, M6 and M3. M7 and M8 follow the
flip.

Estimates come from measured pace. M1 took about 1.5 working days for multi_seq and about one
more for verify plus the draft head. Each estimate below separates work time from box queueing
and campaign pacing.

| # | Milestone | Acceptance | Work estimate |
|---|---|---|---|
| M1 rest | MoE decode / multi_seq / verify (after perf/moe-wide); the W8A8 declared leg | byte parity eager + graphed; tok/s and J/tok within noise | 1.5 days |
| M5 | State schema: state edges, lifetimes, state ops as nodes, one sizing function; GDN and Mamba2 kinds | `StatePlan` bytes == legacy allocation for the two Qwen families and Nemotron-3-Nano (the supported Nemotron-H checkpoint); preflight == allocation; verify/draft parity unchanged, with the commit/rollback copies as nodes | 1.5 days |
| HF tool | `tools/circuit/hf_reference.py` (7.2), keyed by the package; first use: HF logits parity on Nemotron-3-Nano-30B to prove the NoPE attention bug, then a standalone fix PR to main with a regression test | the bug shown by HF parity, and fixed | 1 day |
| M6a | Dense `prefill` + `prefill_chunk` + drafter prefill | prefill parity at every bucket's ends, plus the decode after it; TTFT A/B inside the gate bounds | 2-3 days |
| M6b | `prefix_restore` + after-restore plans; `prefill_batch` | restore parity; warm TTFT A/B | 1.5 days |
| M6c | `mixed`; MoE prefill | as above, both families | 2 days |
| M7 | Load plan + config mapping; `met circuit diff --load` | every bound weight's pointee hash equal (or the requant nondeterminism documented and fixed first); config field-by-field equality | 3 days |
| M8 | Pilot: Nemotron-3.5-Lightning built only from its package (Mamba2, ReLU² MoE with sigmoid routing, NoPE attention, FP8 per-tensor static-scale W8A8, MTP over Mamba2 rollback); then Llama 3.x / Qwen3-dense (G1) | HF tolerance parity (7.2), then the accuracy gates; certifies | 4-6 days |
| M3 | Flip `--forward circuit` for the circuit-built families; delete their legacy forward, loader and dispatch code | full campaign green; code deleted | 1 day + campaign |
| skill | `/new-model` adopted as the standard method (a draft is on feat/circuit-venn) | the M8 pilot followed it end to end | with M8 |

## 11. The Venn method

Owner direction 2026-09-29: every new-model build starts from a kernel Venn diagram against the
closest supported models. The shared set is maximised by parameterising differences safely.
Where parameterising is not advised, the kernel is split into composable pieces that fusion rules
recombine. Kernels outside the diagram are wired, measured and optimised one at a time, then
fused.

### 11.1 The tool (from feat/circuit-venn, merged into feat/circuit at 6f9b8ba0)

`met circuit venn --target <recipe|checkpoint|arch|checkpoint dir> --against <arch>[,<arch>]
[--mode ...] [--rows ...]` instantiates both circuits and classifies every target node, per mode
and row count:

| Class | Meaning |
|---|---|
| **Shared** | Same kernel family at the same compile-time and policy point, with a microbench record at that point and row count. |
| **Shared, unmeasured** | Same point, no record. It is never called "optimised". |
| **Parameterisation opportunity** | Same family; a compile-time parameter differs (e.g. head_dim, group size, activation). The report names each parameter and whether the kernel takes it at run time or compile time. |
| **Policy variant** | Same structure, only policies differ (sigmoid vs softmax routing, ReLU² vs SiLU·mul, static vs dynamic activation scale). |
| **Novel** | No family exists. |

- **Ranking.** Nodes are ranked by estimated share of step time: a roofline from edge shapes
  and formats, KV and recurrent-state traffic, and distinct experts. Measured profiles replace
  the estimate when a serve runs.
- **Flags come first.** The report opens with the layer kinds that cannot batch rows, costed as
  the added time of the per-row fallback.
- **Output.** The report is `kernels/circuits/venn/<target>-vs-<arch>.md`. Its `--check` mode
  fails on a stale report.

### 11.2 Kernel manifests: parameter space and evidence envelope

`kernels/gb10/common/KERNEL_FAMILIES.toml` declares, for each kernel family:

- its ops and constraints;
- its parameters, each classed **runtime** (strides, counts, eps, scale pointers: anything that
  does not size registers, smem or unrolling), **compile-time** (head_dim, group size, tiles) or
  **policy** (format, scale granularity, activation quantizer, epilogue);
- its instantiated points and how each is realised (a template instance, a macro instance, or a
  file copy);
- its **evidence envelope**: the `docs/kernel-perf/measurements.toml` rows per point and row
  count.

"Optimised" is claimed only inside the envelope; shared code outside it is reported as "shared,
unmeasured here". Tests fail on drift from the sources, on a FUSIONS.toml kernel that belongs to
no family, and on a citation that no longer holds.

### 11.3 Parameterisation rules

- **Compile-time instantiation from the model union.** Compile-time parameters are instantiated
  from the union of MODEL.toml values (e.g. every head_dim a target declares):
  - `build.rs` emits the explicit instantiations, with lookup names that encode the point
    (e.g. `paged_decode_attn_bf16_hd256`);
  - the closure hash already covers MODEL.toml;
  - the kernel-lookup audit enumerates the instantiations, so a lookup of a point that was never
    instantiated fails at build time, not at boot.
- **Policy templates** (the WxAy engine pattern) cover:
  - formats;
  - scale granularity: per-tensor, per-channel, block r x c, group g;
  - activation quantization: dynamic per-token or group, static per-tensor;
  - the activation epilogue: SiLU·mul, ReLU², GELU.
- **Stability gate for parameterising an existing optimised kernel.** All three must hold:
  1. byte-identical output at every existing point (the circuit's parity harness per mode, plus
     a kernel microtest);
  2. a microbench with no regression, within noise, at every existing point;
  3. the existing models' gates unchanged.

  New points then get their own microbench and tuning; the tile choice may itself become a
  parameter.
- **Split instead of parameterising** when:
  - the parameter changes the algorithm (e.g. a different tiling at head_dim 512);
  - the instantiation count or compile time explodes;
  - an existing point would regress.

  The pieces become separate kernels, and `bit_identical` fusion rules recombine them. The
  fuser's microtest requirement proves the recombination.

### 11.4 Kernels outside the diagram

1. Wire them into the arch TOML with reference implementations.
2. Check correctness against Hugging Face logits (section 7.2).
3. Microbench each against its roofline, in step-share order, and optimise one at a time.
4. Add fusion rules.

### 11.5 First targets and first application

- **Attention head_dim.** Today it is duplicated per head_dim, in about 18 `*_128.cu` /
  `*_512.cu` copies (e.g. `attn_prefill_h128.cu`, `paged_decode_attn_*_{128,512}.cu`). It
  becomes a template parameter, instantiated from MODEL.toml head_dims.
- **BF16 projections at many rows.** `dense_gemv_bf16` becomes a W16A16 policy of the WxAy
  engine, so the row-tier machinery (1..256, tensor core) applies to BF16 too.
- **FP8 scale.** A per-tensor weight-scale policy and a static activation-scale quantizer, beside
  the per-row / block-128 weight policies and the dynamic per-token / g128 quantizers.
- **MoE experts.** The tensor-core grouped expert kernel becomes format- and
  activation-parameterised: FP8 or NVFP4 (W4A16) weights, gated SiLU·mul or ungated ReLU².
- **Routing.** A top-k scoring policy: softmax, or sigmoid with a correction bias, plus the
  routed scaling factor.

First application, after the MoE and dense energy wrap-up: Nemotron-3.5-Lightning vs
Qwen3.6-35B-A3B, plus the dense 27B for the W4A16 GEMV and head. The first report is already
generated (`kernels/circuits/venn/nemotron-3.5-lightning-vs-qwen3.6-35b-a3b.md`).
It surfaces all five targets above, and its top flags are Lightning's MoE and Mamba2 layers,
which run one row at a time in multi-sequence decode.

The Venn method and the lifecycle package meet in one place: a Venn report's Novel and Policy
rows are exactly the ops the new package's circuit names that have no rule yet. The package is
complete when the Venn report for it has no Novel row without a rule and a parity-checked
reference kernel.

## 12. Strict SBIO

The circuit crate stays pure: no file reads, no GPU calls, no environment reads, no clocks. The
plans it returns are data. Every effect crosses a trait the caller implements:

| Trait (in `crates/circuit`) | What the caller supplies | Implemented by |
|---|---|---|
| `PackageSource` | the package TOML text by name | server CLI (files), tests (strings) |
| `CheckpointIndex` | tensor names, dtypes, shapes, the quantization config (no bytes) | model-weights (safetensors header, GGUF) |
| `ConfigSource` | the parsed config.json as a JSON value | server |
| `KernelInventory` | which (module, func) exist, and the KernelCaps | kernels crate probe (`AvailableKernels` today) |
| `StateBudget` | free device bytes, gpu_memory_utilization | server preflight |

The crate returns plans: `FusionPlan` (exists), `BufferPlan` (exists), `LoadPlan`, `StatePlan`,
`PrefillSchedule` (bucket ladder -> plan per bucket). The executor side (model-layers /
model-engine) turns plans into effects: the loader runs the `LoadPlan` transforms through
existing `weight_map` helpers, the allocator runs the `StatePlan`, the emitters run programs.
The pure crate never sees a `DevicePtr`.

Consequences:

- Every rule of the plan is unit-testable on the host with fixtures, the way the golden plans
  are today. The mocks are fixture implementations of the traits.
- The tensor bytes never enter the crate. The transform chain is checked against the index
  (dtypes, shapes), and the bytes are the loader's concern.
- Sizing (section 3.3) is a pure function of the `StatePlan` and a `StateBudget`, so preflight
  and allocation can no longer disagree.

## 13. Out of scope, and why

| Item | Why |
|---|---|
| Tokenizer, chat template, tool and reasoning parsers, stop tokens | Serving metadata. The package references the existing implementations by id; re-implementing them adds risk and no kernel reuse. |
| The scheduler (batching, admission, prefix-cache policy, spec-decode policy) | It decides *what* runs; the circuit decides *how*. The interface is the program per mode plus the state ops. |
| Auto-generating heavy kernels (GEMM, attention, GDN scan) | Heavy ops stay opaque nodes, per the M0 design. Fusion happens at their edges. |
| Multi-node TP/EP | `EpReduce` exists as an op, but no parity reference runs on a single box. It follows once a single-box family is circuit-built. |
| Vision towers and multimodal prefill | They are not on either target's gate path. A later package section. |
| GGUF-only transforms (norm -1, `A_log`) | Not used by the safetensors families. Vocabulary entries can be added when a GGUF family is circuit-built. |
| The legacy env levers not in any recipe | Refused under `--forward circuit` rather than modelled. Each is either promoted to a plan input by an owner decision or deleted with the legacy path in M3. |
| Fixing the legacy prefill nondeterminism | Recorded as its own finding. Prefill parity works around it (section 7.1) until it is fixed separately. |

## 14. Risks

| Risk | Consequence | Mitigation |
|---|---|---|
| Legacy prefill is nondeterministic for some prompts > 64 tokens | Prefill parity cannot be claimed on the affected rows | 2026-10-03: fixed on main (#67, the conv1d prefill state race; 35B evidence). M6 step 0 repeats it on the dense 27B and removes the exclusion rule if 0 rows differ (15.4) |
| NVFP4 requantization at load may be nondeterministic (atomic absmax) | Load parity is undefined for requantized weights | Legacy-vs-legacy hash first; fix or document before M7 acceptance |
| Two live workspaces before the flip (legacy arena + circuit prefill workspace; the 27B FFN's gate/up and activation edges alone are 0.86 GB at 8192 tokens) | KV budget shrinks, or boot refuses | Size from the `StatePlan`, bind prefill edges to arena buffers where needed, and measure the KV blocks lost; after M3 only one exists |
| Plan count grows: modes × buckets × route keys | Boot compile time, golden-plan volume | Programs are small `Vec<Launch>`s (hundreds of closures each) compiled at boot per bucket; golden plans render one plan per bucket |
| Env-only levers that change prefill routes (dozens: FLA, MMQ, small-M, two-phase, `HOLO_*`, ...) | A silent route difference between forwards | Every lever is classified in the policy table: modelled (a plan input), refused under `--forward circuit`, or ignorable with a reason. Nothing is unclassified (the M1 `MULTI_SEQ_ENV_SWITCHES` pattern) |
| Host-side hazards the survey found in legacy prefill: the pinned staging buffer reused across streams without a sync (`prefill_b/batch_kernel.rs:156-351`); `prefill_stream` and the default stream sharing one arena | A circuit copy of legacy inherits them, or parity chases them | Recorded in the nondeterminism finding. The circuit's metadata upload is a node on the program's stream, with no cross-stream reuse; parity then reveals whether legacy's hazard is live |
| MoE row order within an expert after the sort is nondeterministic by design | MoE prefill parity only if the grouped sums are order-independent (claimed, not shown) | Legacy-repeat on the MoE prefill first; if it differs, the MoE prefill bar is the HF tolerance plus the gates, stated as an exception |
| The loader rewrite touches every weight | A wrong transform ships a subtly wrong model | Pointee-hash parity per bound weight, plus a detection control; legacy loader kept until M3 |
| The mapping refuses keys legacy ignored silently | A checkpoint that legacy serves no longer boots under the circuit | Intended: each refusal is a real divergence. Listed per family; an owner decision per key |
| Mamba2 has no multi-row kernels (every Mamba2 and Nemotron-MoE launch is one row; verify and multi-sequence decode loop per row) | Lightning's MTP and C>1 decode are slow until written; the Venn report ranks them as the top two flags | New multi-row Mamba2 update + snapshot kernel, and a ReLU² grouped expert path via the TC-kernel parameterisation (11.5), each behind the stability gate |
| Nemotron-H legacy bugs found by the survey (NoPE, the 0-byte conv rollback, unparseable Lightning config, a latent read/write race in `causal_conv1d_update_prefill_tp`) | Legacy is not a valid reference for this family | HF is the reference (7.2); the NoPE fix goes to main now; the others are recorded with file:line and fixed where they sit on a gated path |
| Certification cost if landings are not batched | 5+ campaigns at ~5 GPU-hours each | Stacks per section 9; kernel-only package edits ride the closure-hash exemption where it applies |

## 15. The circuit runs the engine: plan for M5, M6, the boot check, certification and the flip (2026-10-03)

Status: PROPOSED 2026-10-03, for review before any M5/M6 code. Line numbers are origin/main at
c92e43109. This section takes precedence over sections 1 and 10 where they disagree.

Owner directive (2026-10-03), condensed:

1. Prefill under the circuit (M5, then M6) for the dense 27B and the MoE 35B: the chunk loop,
   prefix-cache restore (KV and SSM snapshot, including #76's single-pass in-pass snapshots),
   batched multi-sequence prefill and the mixed step. Byte-identical prefill logits and final
   state bytes vs legacy on every path, eager and graphed. Then the TTFT cold/warm and high-ISL
   instruments within noise of legacy.
2. A boot-time plan check. Every kernel the serve launches is the plan's, with #85's precision
   pipelines and #78's tensor-core policy. A mismatch refuses to boot.
3. Full-path certification: `--forward circuit` on every gate, at the same bounds as legacy.
4. The flip. `--forward circuit` becomes the default, and the legacy decode, verify, draft and
   prefill loops are deleted for the circuit-built families. The PR counts the deleted lines.
   No deprecation window is recommended.

The memory model (#82) is the single source for state sizing, and #85's pipelines are
authoritative.

### 15.1 Where the circuit stands at c92e43109 (corrects section 1)

| Family | Under `--forward circuit` |
|---|---|
| Dense `qwen3_5` (two instances: the recipe and `-declared`) | decode (1 row); multi_seq 2..128; verify K = 2, 3, 4; **verify_batch** per row table; **draft** at 1 row and at the batched-propose widths. Byte parity eager + graphed. Draft programs run eagerly (no capture in the MTP paths). |
| MoE `qwen3_6_moe` | Refused at boot: the MoE FFN is not bound (`ml/layers/mod.rs:232`). Golden plans exist for decode, multi_seq, verify and draft n1. The perf/moe-wide blocker has landed as #51. |
| Nemotron-H | Refused (`mamba_num_heads > 0`, `me/model/trait_impl/impl_circuit.rs:60-63`). |

Paths that still run legacy under `--forward circuit`. None is disclosed today:
`ForwardDisclosure` carries only the forward, the plan digest and the launches per step.

1. **Every prefill entry**: single-pass, chunked, two-phase, batched/varlen, prefix restore and
   the drafter prefill. `Mode` has no prefill variant (`crates/circuit/src/rules.rs:21-33`).
2. **The fused mixed step.** `mixed_forward_dispatch` runs `layer.decode_multi_seq` for the
   decode rows and `layer.prefill` for the chunk, per layer (`me/model/trait_impl/
   decode_b.rs:405-432`). The scheduler takes it **by default** whenever a chunk continues
   while decodes are active (`run_standard.rs:98`; not EP, no spec step this tick). So every
   concurrency gate runs legacy decode rows today, even under `--forward circuit`.
3. **Grammar-masked draft steps** (`ml/layers/mtp_head/forward.rs:61`).
4. **State ops between steps**: checkpoint, rollback, commit, fold, the ring. They are host
   copies, although `crates/circuit/src/state_ops.rs` already models four of them as nodes.
5. **Boot refusals** (`impl_circuit.rs:48-76`): TP/EP, DFlash hidden capture, LoRA, KV
   high-speed swap, profiling, Mamba-2, latent attention, vanilla norm weights. An FP16 h state
   is refused only at the first step (`impl_circuit_run.rs:36-39`), not at boot.

**The prefill nondeterminism (section 0, decision 5) is fixed on main.** It was a race in
`causal_conv1d_update_prefill_tp`; the state write moved to `causal_conv1d_prefill_state`
(`ml/layers/ops/ssm_mamba.rs:199-205`, #67 at 2f66da5dd). The evidence is from the 35B only:
5/0/12 rows of 88 differed before the fix and 0/0/0 after. M6 step 0 repeats it on the dense
27B (15.4).

### 15.2 What "runs the engine" requires: the coverage matrix

The flip is per family, so every recipe of the family must boot and run under the circuit, not
only the gate recipes. The matrix has these two axes:

- **rows:** the recipes;
- **columns:** every path the scheduler can take for that recipe's serve arguments (decode,
  multi_seq, verify, verify_batch, draft, each prefill mode, restore, mixed, grammar draft, state
  ops).

Each cell is one of: a program with parity evidence; refused at boot with a reason; or retired by
an owner decision. The flip needs no empty cell.

| Family | Gate recipes (subject of a required gate) | Other recipes of the family |
|---|---|---|
| Dense 27B (`qwen3_5`) | `qwen3.8-27b-nvfp4-unsloth` (decode-floor, high-ISL cold/warm), `-unsloth-bfcl` (bfcl-subset, kat-equality, scheduler-equivalence), `-throughput` (concurrency-sweep with `prefill_codispatch = "true"`, default-tier-boot), `-dflash2` (concurrency-sweep-dflash2), `qwen3.6-27b-nvfp4-unsloth` (video-fidelity) | `qwen3.8-27b-nvfp4-latency`, `qwen3.6-27b-{nvfp4, fp8, fp8-mtp, nvfp4-prefill-record}` |
| MoE 35B (`qwen3_6_moe`) | `qwen3.6-35b-a3b-fp8-bf16head` (ttft cold/warm, high-ISL-moe cold/warm, bfcl-echolp, agentic-webserver, ssm-poisoning, vision-fidelity), `-fp8-nvfp4head` (concurrency-sweep-moe) | `-fp8-mtp`, `-fp8-nvfp4head-experts-{nvfp4, nvfp4-gate-up}`, `-nvfp4` (the NVIDIA NVFP4 checkpoint) |

Only two instances exist today (both dense). Most new instances are data: an `INSTANCES.toml`
row plus golden plans, because the recipes differ in policy, not in blocks. Three cells are
real work.

**Owner decisions needed (D-A to D-C):**

- **D-A. DFlash2.** `concurrency-sweep-dflash2` is a required gate, and the circuit refuses
  DFlash. Two options:
  - *Model it* (M6e below): the target forward gains declared hidden-capture outputs at the
    capture layers (section 3.4), and the DFlash drafter becomes a draft program. About 2-3
    days.
  - *Keep it on legacy*: `--forward circuit` refuses `--dflash`, the gate stays a legacy gate,
    and the verify and capture code it uses is not deleted.

  Recommendation: model it. Otherwise the flip leaves one required gate on a path nobody
  certifies under the new forward.
- **D-B. Unmodelled features on the circuit families.** TP/EP, LoRA, KV high-speed swap and
  `--profile`. No recipe of either family uses TP/EP, LoRA or swap (checked in
  `recipes/qwen3.6`, `recipes/qwen3.8`). Recommendation:
  - after the flip, refuse these four at boot for the circuit families;
  - profiling is replaced by the launch trace (15.5);
  - EP returns as an M-later item (`EpReduce` exists as an op).
- **D-C. Deletion scope.** See 15.7: the honest count for a two-family flip is small.

The vision and video gates need no new tower work. The vision encoder is not one of the four
loops and stays as it is. Only the text prefill that consumes its rows runs under the circuit,
through the `embeds_injected` route key (15.4).

### 15.3 M5: state under the plan, with #82 as the single source

Base: #82 (`crates/circuit/src/memory/`, `met circuit memory`, stacked on #85). M5 adds no
second state schema. It adopts #82's kinds and lifetimes and makes the allocators and the
executor consume them.

1. **Kinds and lifetimes.** #82's `state_kinds.rs` (prefix and ring snapshots, the carry stash
   and table, verify tables, accept stash, hidden capture and the rest) and its `Snapshot`
   lifetime are adopted as they are. M5 adds:
   - the `Step` lifetime;
   - the planner checks from section 3.3: a step edge never outlives its program, a verify edge
     is written only by `state_snapshot` and read only by commit/rollback, and a snapshot edge
     is touched only by state ops.
2. **Every allocator asks the plan.** Today these still allocate from their own arithmetic (the
   KV budget in `factory/build/kv_budget.rs`, the rest at their own sites):

   | Allocation | Moves to |
   |---|---|
   | KV pool | memory-model term |
   | Marconi pool | memory-model term |
   | decode ring | memory-model term |
   | carry stash | memory-model term |
   | h prefill staging | memory-model term |
   | circuit workspace, including the prefill buckets at `t_max` | memory-model term |

   The SSM pool already does this (`PoolPlan`, `ml/ssm_reserve/pool_plan.rs:186-290`).
   `ReservePlan` reads the same terms (#82's `MEMORY-DESIGN.md` §3 wiring). The circuit
   workspace enters the pre-load reserve; it is absent today.
3. **Globals become plan inputs**: the rollback mode, the decode-ring depth and the MTP sequence
   cap.
4. **State ops run as programs.** The executor runs these as state programs:
   - from `state_ops.rs`: VerifyCheckpoint, VerifyRollback, CommitAccepted, SlotZero;
   - added: fold-accepted, prefix snapshot save/restore, ring save/restore.

   The legacy host copies stay reachable under `--forward legacy` until the flip.

**Acceptance:**

- On every instance of 15.2, plus Nano, the boot allocation digests equal legacy byte for byte
  (#82's ledger test, tightened from percentages to 0 bytes).
- Preflight equals allocation exactly.
- `met circuit diff --verify 2,3,4 --mtp` and `--verify-batch` stay byte-identical, eager and
  graphed, with commit/rollback running as state programs.

**Size:** one PR, about 1.5 days of work.

### 15.4 M6: prefill under the circuit

**Execution model.** This refines section 4.2; everything there stands.

- **Layer segments.** A program gains layer segments (`Program.segments`, the launch range of
  each layer). Segments are what the host drivers compose. Three compositions cover every legacy
  prefill shape with no new plans:
  - *chunk-major*: the chunk loop;
  - *interleaved*: the mixed step. For layer ℓ it runs the multi_seq program's segment ℓ, then
    the prefill program's segment ℓ, which is exactly legacy's per-layer order;
  - *layer-major*: two-phase, if it is kept.

  A fusion that crosses a segment boundary is refused in a composable mode. Legacy fuses none.
- **Route keys.** These become plan keys, each a policy input to `fuse`:
  - `chunk` (first or later; contiguous vs paged attention);
  - `after_restore` (exact replay: WY4/regresident, never FLA);
  - `tail_cut` (#76: a runtime cut row; the GDN and conv nodes emit a `Snapshot`-lifetime
    state output at the cut, and rows past the replay point take the row-invariant tier);
  - `varlen` (S sequences);
  - `all_rows_logits` (prompt logprobs);
  - `embeds_injected` (vision/video rows and mrope positions replace the token embedding).
- **Lengths are runtime arguments** (section 4.2). Device metadata is a node: one H2D per chunk
  from pinned staging, in legacy's layout. The batched path gets per-stream staging, so the
  cross-stream reuse hazard in the risk table is not copied.
- **Host code that stays**: the chunk loop, prefix lookup, snapshot scheduling, wave planning
  and admission. They move into one driver that calls programs. The per-layer legacy calls do
  not survive in it.
- **Graphs.** Prefill is eager in both forwards (section 4.3); legacy never captures it. The
  "eager and graphed" bar therefore means two things:
  - each prefill instrument continues into decode, verify and draft steps run graphed, and those
    must stay byte-identical;
  - the mixed step's decode rows are compared under the same graph mode legacy uses.

  Owner: please confirm this reading.

**Step 0** (untimed, about 1 h of box time, no code):

- **(a)** Legacy-repeat over bucket-edge prompts on the dense 27B. If 0 rows differ, as
  expected after #67, the row-exclusion rule is removed from every instrument and the bar is
  strict.
- **(b)** Which arm a 32k prompt takes on each model. Two-phase falls back to chunks when the
  prompt exceeds `max_batch_tokens` or the GDN buffer (`me/model/trait_impl/prefill_c.rs:54-90`),
  and which one runs is not known without a serve log.
- **(c)** Classify every prefill env lever (MMQ, FLA, small-M, two-phase, tail-split, Marconi,
  `HOLO_*`, ...) as modelled, refused or ignorable. The executor's policy table
  (`ml/circuit_exec/policy.rs:140-170`) covers decode switches only. An unclassified lever
  fails a test.

**The milestones, dense first:**

| Step | Content | Parity instrument |
|---|---|---|
| M6a | `prefill` + `prefill_chunk` + the drafter prefill; the segment type | `met circuit diff --prefill` (new). Prompt lengths are t_lo and t_hi of every bucket. It compares the last-row logits, then each layer's GDN h and conv bytes, the sequence's KV blocks and the drafter's KV rows. Then 32 graphed decode/verify steps. `--prefill-chunk` repeats it with a small `max_prefill_tokens`, across many chunk boundaries |
| M6b | `prefix_restore` + `after_restore` + `tail_cut` (#76) | primed re-sends: the restored state bytes, the in-pass snapshot bytes vs legacy's, then the logits |
| M6c | `prefill_batch` (varlen, kernel-batched, codispatch), then `mixed` | per row, as above; the mixed step with decode rows in flight compares the decode rows' logits and the chunk's state |
| M6d | Two-phase, decided by step 0(b) (below) | as M6a, at 32k |

**Two-phase, if it is on a gated path:** measure legacy two-phase against legacy chunked at 32k
first (timed, 2 interleaved reps). If chunked sits inside the high-ISL gate's bounds, two-phase
is deleted at the flip without being modelled, which saves about 1.5 days. Otherwise it is
modelled as a layer-major composition with its GDN phase split.

**Corrections to section 8:**

- **Rule 4 is wrong.** The fused mixed step is the default scheduler path, not opt-in, so M6c
  is required for the flip. Only `--prefill-codispatch` is opt-in, and `concurrency-sweep`
  pins it on.
- **The dense model has no 256/1024/4096 TTFT gate.** The TTFT A/B below runs those lengths on
  both models anyway.

**TTFT A/B** (timed, in a campaign gap, per model):

- legacy vs circuit, interleaved, 2 reps, with a same-night control;
- instruments: cold and warm TTFT at 256, 1024 and 4096 tokens × 12, and the 32k high-ISL cold
  and warm;
- pass: inside each gate's own bounds against the legacy leg (median +3%, p90 +5%).

The plans are byte-identical, so a difference is host enqueue time, and that is reported per
chunk. Before the flip both the legacy prefill arena and the circuit workspace are resident.
From M6c on, `--forward circuit` stops allocating the legacy prefill arena, because every
prefill path is then a program. The KV blocks are compared against legacy on the same boot.

**MoE (M6-MoE), after the dense pattern is set:**

1. **MoE M1.** Bind the MoE FFN in decode, multi_seq, verify, verify_batch and draft:
   - the router, top-k, the sorted grouped GEMM and the FP8 pointer tables;
   - the W8A8 decode twin vs W8A16;
   - the shared expert.

   `--expert-quantization` and `--activation-quantization adaptive,moe:bf16` are plan inputs.
   Parity is eager + graphed, as in M1.
2. **MoE prefill.** The M6a-c modes over the grouped path (> 64 rows), with legacy-repeat first.
   If the grouped sums turn out order-dependent, the MoE prefill bar becomes HF tolerance plus
   the gates, stated as an exception (risk table). They are not shown order-independent today.
3. **Instances** for every 35B recipe in 15.2.

**Parallelism proposal (for approval).** Once M6a has merged into the stack branch, a second
workstream takes M6-MoE. By then the segment type, the prefill `StepEnv`, the state programs and
the `--prefill` instrument exist. The two touch different emitters (the MoE FFN vs dense
attention/GDN). They share only `INSTANCES.toml` and the golden plans, which merge as unions.

**M6e (only if D-A = model):** DFlash2. The target gains hidden-capture outputs, and the
drafter's propose becomes a draft program. Parity runs through `met circuit diff
--verify-batch` under `--dflash`.

### 15.5 Step 2: the boot plan check

In `CircuitExec::build`, right after each `fuse` (`ml/circuit_exec/mod.rs:189-235`):

1. **`pipeline::check_plan` (#85)** against the pipeline the checkpoint and the settings
   require. This is the first time it runs at serve time, which closes #85's caveat.
2. **`tc_policy::enforce` (#78)** against `[tensor_core_policy]`.
3. **Live digest == golden digest** for every (mode, rows or bucket, route key) of the
   instance. The live policy comes from levers, the golden plans from the INSTANCES policy,
   and today they can differ silently. A difference refuses boot and names the differing keys.
4. **Coverage.** Every path the boot configuration can reach (spec method, prefix caching,
   codispatch, varlen, chunk size, grammar, prompt logprobs, vision) maps to a compiled
   program. Until the flip a path without a program is disclosed. From the flip on, it refuses
   boot.
5. **Disclosure.** `/forward` reports programs run and fallbacks taken, per path. The bench
   record carries both, and `check_record` refuses a circuit record with a non-zero fallback
   count. The FP16-h refusal moves to boot.
6. **Launch audit (parity leg, not serve time).** Run the parity suite with
   `metrale_telemetry::launch_trace` on (`crates/gpu-runtime/src/gpu.rs:110-125`). The set of
   kernels the serve launched must be a subset of the union of the plans' kernels. A kernel
   outside the plans fails the leg and names its call site.

### 15.6 Step 3: full-path certification

A gate record must carry exactly its BENCH entry's serve overrides (`check_record`). So a
separate circuit leg would need a second set of entries. The cheaper route is that **the flip
PR itself is the full-path certification**. With the circuit as the default:

- every gate runs it at unchanged bounds;
- the record proves it (`forward` and `plan_digest` are verified live, `bench_lease.rs:189,337`);
- the new record check proves it ran with 0 fallbacks.

Before that campaign is armed, at the final sha:

- the full parity suite on an untimed box;
- one untimed smoke per gate recipe under the circuit (boot, 0 fallbacks, the launch audit).

### 15.7 Step 4: the flip and what it can delete

**The flip:**

- the forward becomes a property of the family: an instance with `golden = true` runs the
  circuit, with no override;
- `--forward` is removed;
- the D-B features are refused for the circuit families.

No deprecation window. Legacy byte parity is the only thing lost. Two things replace it:

- a *reference fixture set* recorded at the flip sha: per parity prompt, the hashes of the
  last-row logits and the state bytes, so a later circuit change can still be compared to
  legacy's bytes offline;
- the golden plans.

**Measured non-test line counts at c92e43109:**

| Code | Lines | Who else runs it |
|---|---|---|
| model-engine `trait_impl` loops: prefill 7,362, verify/MTP 5,653, decode 3,646, mixed 699 | 17,360 | every `TransformerModel` family |
| `Qwen3SsmLayer` (GDN) | 14,647 | qwen3_next, qwen4_exp, qwen3_5_moe, holo3_1_moe |
| `Qwen3AttentionLayer` | 23,637 | about 12 loaders (Nemotron, Gemma4, Mistral, MiniMax, DeepSeek-V4, ...) |
| MoE layer, dense FFN, `MtpHead`, `ops/prefill*` | 24,546 | shared |
| loaders `qwen35_dense`, `qwen35` | 6,007 | the two families (plus holo3_1_moe for `qwen35`); deletable only with M7's load plan |

**What a two-family flip can delete** is the circuit families' legacy arms, plus the code only
those families reach. That is small: low thousands of lines, not the roughly 75k of shared
loops, which stay while any other family runs them. The flip PR measures it exactly:

1. delete the arms;
2. iterate the compiler's dead-code lint to a fixpoint;
3. report `git diff --numstat` per crate.

**Owner decision D-C:**

- **(a)** Flip the two families and delete what they alone reach (recommended now).
- **(b)** Then move the Qwen-hybrid siblings that share `blocks/qwen3_hybrid.toml` (qwen3_5_moe,
  holo3_1_moe, qwen3_next, qwen4_exp). That is mostly instance data plus a parity leg per family.
  It makes `Qwen3SsmLayer`'s legacy forward and the GDN branches of the loops deletable, about
  15-20k lines. Recommended as the next milestone.
- **(c)** Retire families that have no recipe in use.
- **(d)** Delete the loops outright. This needs every `TransformerModel` family circuit-built
  (M7/M8 and the `/new-model` method), so it is out of this milestone.

### 15.8 PR boundaries, stacks and campaigns

Any `crates/**` change re-opens all 17 required gates (27 units, about 2 h 15 min and 5
GPU-hours). Everything before the flip lands behind the legacy default, so its campaigns certify
that legacy did not move.

| Stack | PRs (bottom first) | Campaign |
|---|---|---|
| S1 dense | 1. M5 state (on #82) · 2. M6a prefill + chunk + drafter + `--prefill` instrument + prefill lever table · 3. M6b restore + tail cut · 4. M6c batch + mixed (segments) · 5. boot plan check (disclosure mode) + every dense instance | C1, legacy default |
| S2 MoE (+ DFlash) | 6. MoE M1 · 7. MoE prefill + every 35B instance · 8. M6e DFlash (if D-A) | C2, legacy default; **merged into C1** if S2 is ready when S1 is |
| S3 flip | 9. flip (default circuit, coverage refusal on, `--forward` removed, record check) · 10. delete + the line count | C3, circuit default: the full-path certification (15.6) |

Two or three campaigns in all, against section 9's five. Splitting C3 into a flip campaign and
a delete campaign costs one more campaign and buys a single bisect point. It is not recommended:
the deletion removes only code the flip made unreachable, and the coverage counters and the
launch audit prove that before arming.

Box use:

- parity legs: untimed, on a box no campaign holds, after checking its lock;
- TTFT A/Bs and the two-phase measurement: campaign gaps only.

### 15.9 Estimates

The estimates come from M1's measured pace (multi_seq about 1.5 days; verify plus draft about 1
day). They are work time only; box queueing and campaigns come on top.

| Item | Work |
|---|---|
| M5 | 1-1.5 days |
| M6 step 0 | 1 h of box time |
| M6a | 2 days |
| M6b | 1.5 days |
| M6c | 2 days |
| M6d | 0 if deleted, 1.5 days if modelled |
| boot check + dense instances | 1-1.5 days |
| MoE M1 + prefill | 3.5 days (parallel workstream) |
| M6e | 2-3 days |
| flip + delete | 1 day |

With the parallel workstream, the dense critical path is about 9-10 working days plus C1 and C3.
M7 (load plan) and M8 (the Lightning pilot) follow the flip; neither is on its critical path.
