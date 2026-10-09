---
name: new-model
description: The standard method for adding support for a new model or architecture to the Metrale Engine. Describe the model as an architecture circuit, run `met circuit venn` against the closest supported model(s), maximize the shared kernel set by SAFE parameterization (head_dim, scale policy, weight format, activation, routing), split and let the auto-fuser recombine where parameterizing is not advised, then prove correctness, microbench and optimize the kernels outside the diagram one by one, fuse, and beat vLLM on tok/s AND J/tok at every concurrency from C1 to C128. Use whenever asked to support, onboard, port or add a model, checkpoint or architecture (e.g. "support nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4", "add Llama 3.x", "zero-day support for X"), or to compare a new model's kernels against an existing one.
---

# /new-model: build model support from the architecture circuit

**The standard.** Every new model is built the same way:
1. Describe it as an architecture circuit, which is data.
2. Share as many optimized kernels as possible with models we already support, by
   parameterizing the kernels, not copying them.
3. Build and optimize only what is truly new.
4. Prove the model correct.
5. Beat vLLM on speed *and* energy at every concurrency from 1 to 128.

Generalization is the point: shared code at a shape nobody measured is not shared
optimization, and a per-model copy of a kernel is a parameterization nobody did yet.

**Composes with `/new-hardware`.** When the model is to run on a hardware class the engine does
not serve well yet (a new GPU or kernel class), run this skill inside
`.claude/skills/new-hardware/SKILL.md`: that skill opens the Hardware Beachhead Campaign, adds
the hardware axis to the Venn (every plan and Venn here with `--hardware <device>`), starts from
a mock checkpoint (`met ml-utils mockify`, `met serve --mock`), and owns the vLLM improvement
loop. Steps 1-9 here are its model axis.

Read before starting, and apply throughout:
- `AGENTS.md`: the core directives and the "Big Three" invariants (SSOT, PCND, SBIO), the
  500-line cap, the local checks, and the certification rules for perf paths.
- `CONTRIBUTING.md`: the PR loop and its gates.
- `KERNEL-PERF.md`: the roofline floors, the GB10 peaks and how a kernel's "% of floor" is
  measured (`docs/kernel-perf/measurements.toml`).
- `KERNEL_ARCH_ROADMAP.md`: which architectures are supported, which have kernels, which are
  optimized.
- `book/src/architecture/circuit-compiler.md`, `book/src/architecture/lkb.md` and
  `book/src/architecture/lab.md`: the vocabulary this method uses. A new model is a parameter
  point of the Latent Architecture Blueprint plus a named architecture residual (refusals). A
  new kernel is a point of a family in the Latent Kernel Blueprint, or a named LKB residual
  entry with its evidence. The math behind it is in `book/src/appendix/lkb-math.md`.
- Record the LAB and LKB metrics at the start and the exit of the work: architecture coverage,
  models expressible as parameter points, the LAB residual, LKB coverage (with its measured
  part), the LKB residual, and what was promoted.

## Where things live

| Thing | Path |
|---|---|
| Circuit crate (pure logic: IR, TOML loader, fuser, planner, digest, display, Venn) | `crates/circuit/` |
| Venn logic (families manifest, classification, roofline, report) | `crates/circuit/src/venn/` |
| `met circuit show / display / diff / venn` | `crates/server/src/cli/circuit*.rs` |
| Architecture circuits and the shared block library | `kernels/circuits/<arch>.toml`, `kernels/circuits/blocks/` |
| Checkpoint shape instances | `kernels/circuits/INSTANCES.toml` |
| Precision tables and checkpoint plan fixtures | `kernels/circuits/precision/`, `kernels/circuits/checkpoints/` |
| Kernel families: ops, parameters, instantiated points, evidence | `kernels/<hw>/common/KERNEL_FAMILIES.toml` |
| Fusion rules (per hardware) | `kernels/<hw>/common/FUSIONS.toml` |
| Golden plans and routing citations | `kernels/circuits/plans/`, `kernels/circuits/ROUTING-AUDIT.md` |
| Venn reports (output of step 3) | `kernels/circuits/venn/<target>-vs-<arch>.md` |
| Per-model engine config | `kernels/<hw>/<model>/MODEL.toml` (+ `BENCH.toml`, `<quant>/KERNEL.toml`) |
| Microbench records (the evidence) | `docs/kernel-perf/measurements.toml` |
| Executor | `crates/model-layers/src/circuit_exec/` |
| Kernel sources | `kernels/<hw>/common/*.cu` (per-target overrides in `kernels/<hw>/<model>/<quant>/`) |

A worked example of the whole method's first half is checked in:
`kernels/circuits/nemotron_h.toml` and
`kernels/circuits/venn/nemotron-3.5-lightning-vs-qwen3.6-35b-a3b.md`.

## Step 1: read the checkpoint (never guess)

1. Fetch `config.json` and the quantization metadata: `hf_quant_config.json` and/or
   `config.json`'s `quantization_config`
   (`curl -sfL https://huggingface.co/<repo>/resolve/main/<file>`). Large JSON goes to a
   scratch file and is parsed with a script, never printed whole.
2. Record:
   - `model_type` and architectures;
   - the layer mix (`layers_block_type` / `hybrid_override_pattern` / `layer_types` /
     interval rules);
   - every dim;
   - the attention variant: GQA/MLA/sliding window; head_dim; positional encoding (rope type,
     scaling, partial factor, or none); q/k norm; output gate;
   - the mixer types: GDN / Mamba2 / KDA;
   - the MoE: expert count, top-k, routing (softmax/sigmoid, correction bias, grouping,
     scaling factor, renormalization), shared experts (gated or not), activation (SiLU·mul,
     ReLU², GELU);
   - the MTP / next-n layers and their module layout;
   - vocab and tied embeddings.
3. Read the reference modelling code (transformers) for anything the config does not say:
   norm weight form (`w` vs `1 + w`), gate-before-norm vs norm-before-gate, which ops a layer
   really runs. The circuit follows the reference, not the engine's closest existing path.
4. **Declared precision, per module group:** weight format and granularity (NVFP4 g16, FP8
   per-tensor, per-channel or block r×c), activation format, and static vs dynamic scales.
   Also the KV-cache quantization and scale tensors, and the excluded (BF16) modules.
   - The engine default is the declared precision (`--weight-quantization declared`).
   - Running above it silently is forbidden. Going below it is an opt-in flag only,
     disclosed on records.
5. Check whether a sibling is already supported and diff its `config.json` against the
   target's (e.g. Lightning is Nemotron-3 Nano plus an MTP head, a new quantization mix and
   the `layers_block_type` config format).

## Step 2: choose the comparison set

- Pick the supported model(s) with the largest op overlap. More than one is normal: a MoE for
  the experts and routing, a dense model for the W4A16 GEMV and the head.
- Prefer models with microbench evidence on the hardware. An optimized kernel is one with
  records at that parameter point.
- A compared model must be a golden instance in `kernels/circuits/INSTANCES.toml` (its
  `FUSIONS.toml` rules cover every node), so the Venn sees the kernels it really runs.

## Step 3: write the circuit, then build the kernel Venn diagram

**3a. Describe the target as data.**
1. `kernels/circuits/<arch>.toml`: block templates per layer kind, the layout, the head and
   the draft head. Reuse `kernels/circuits/blocks/` via `include` where the math is the same.
   Every op comes from the closed vocabulary (`crates/circuit/src/ir.rs`); extend it only
   with ops the model needs, and prefer a node parameter (`params = { scoring = ... }`) when
   the difference is a policy of an existing op rather than a new op.
2. A precision table (`kernels/circuits/precision/<name>.toml`) or a checkpoint plan fixture
   that states the declared formats; explicit `act_quant` nodes where activations are
   quantized.
3. An `INSTANCES.toml` entry (`golden = false` until rules cover it) with the shape from
   `config.json` and every policy setting stated.

**3b. Run the Venn.**

```
met circuit venn --target <recipe | checkpoint id | arch | checkpoint dir> \
  --against <recipe>[,<recipe>] [--mode decode,multi_seq,verify,draft] \
  [--rows 1,16,128] [--verify-rows 2] --out kernels/circuits/venn/<target>-vs-<arch>.md
```

- Given a checkpoint directory, it first checks the instance against that checkpoint: layer
  kinds, and the declared precision of every bound module. A disagreement is an error.
- It refuses to run while `KERNEL_FAMILIES.toml` has drifted from the kernel sources, or
  while a compared plan runs a kernel no family lists.
- `--check` verifies the checked-in report is current; `crates/circuit/tests/venn.rs` runs
  the same check in `cargo test`.

For every target node, in every mode and representative row count (C1, C16, C128; the MTP
verify K), the report classifies it against the kernel families:

| Class | Meaning | Action |
|---|---|---|
| **Shared** | A family a compared model runs for this op, at the same compile-time and policy point, with a microbench record at that point and row count | Reuse |
| **Shared, unmeasured** | The same without the record, or a point the family already realises by an instantiation or a runtime branch | Reuse, then microbench it (step 7) |
| **Param. opportunity** | Same family; declared parameters differ and one is compile-time; the target point is missing or exists only as a file copy | Parameterize (step 4) |
| **Policy variant** | The same, where every differing parameter is a policy | A policy template (step 4), or split (step 5) |
| **Novel** | No family implements the op with these formats at this row count | Build (step 7) |

- A difference in a **runtime** parameter (strides, counts, `top_k`) never makes an
  opportunity.
- **Evidence counts only at its own point and row count**: a record at head_dim 256 says
  nothing about 128, and a record at M = 32 nothing about M = 16.
- Each row also lists the other families that could run the node as opportunities (e.g. a
  BF16 projection that a dense GEMV runs today is also a W16A16 policy of the WxAy engine).
- Rows are ranked by estimated step share: a roofline per node,
  `max(bytes / DRAM bandwidth, FLOPs / peak)`, with the peaks of `KERNEL-PERF.md`, the KV
  cache and recurrent states counted, and routed experts at the expected number of distinct
  experts. Replace it with measured profiles (nsys) as soon as a serve runs.
- **The top of the report flags every layer kind that cannot batch rows**: a layer whose
  circuit ops, or whose legacy implementation, loops per sequence above C1 (facts about
  legacy code live in `KERNEL_FAMILIES.toml` `[[legacy_path]]`, each citing the line that
  proves it). That fallback is always a top-ranked gap.

**3c. Keep the manifest honest.** When the report is wrong, fix the data, not the report:
- a kernel family is missing or too coarse: add or split a `[[family]]` (its `kernels` must
  cover every `FUSIONS.toml` kernel; a kernel belongs to one family);
- a family varies something it does not declare: add the `[[family.param]]` with its kind
  (`runtime` never sizes registers, smem or unrolling; `compile` does; `policy` is a
  plug-in), and its instantiated `[[family.point]]`s with `how` = `instantiation`, `copy`
  or `branch`;
- a microbench exists: add `[[family.evidence]]` at its exact point and row counts, keyed
  to its `docs/kernel-perf/measurements.toml` row (or a `microbench` note until it is
  recorded there);
- points that can be rediscovered from the sources (per-point copies, macro instantiations)
  get a `[[family.discover]]` rule, so drift fails the tests.
- every family states where its arithmetic runs: `compute = "tensor_core"` with its `mma`
  atom and format, `"cuda_core"`, or `"memory"`; `kernel_compute."<kernel>"` names a kernel
  that runs elsewhere (a quantizer beside an MMA tile). A matmul-class op the class's
  `HARDWARE.toml [tensor_core_policy]` covers must plan onto a tensor-core kernel, or the
  plan (and the matrix `--check`) fails until an exemption lists it with its reason; a new
  model's CUDA-core matmul is tensor-core backlog, never a silent default.
- every family declares the numeric pipeline each op runs at (`pipeline.<op> = { in, <steps>,
  out }`, per point or `kernel_pipeline."<kernel>"` where they differ; the steps per op are
  `crates/circuit/src/pipeline/vocab.rs`), read from the kernel sources: the activation and
  weight precision at the multiply, accumulation, where scales apply, element-wise compute,
  cache and state dtypes, and what a fused kernel hands on between ops. Every plan checks each
  node's declared pipeline against what its circuit formats and policy require; a kernel that
  runs another precision is refused unless its FUSIONS.toml rule states the departure
  (`holds`, `steps`). `met circuit precision --checkpoint <id|dir> --node '<glob>'` lists what
  a new model's nodes need before any kernel exists.

The report is checked in with the architecture package and reviewed before any kernel
work.

### Step 3b: plan it on every target device

`met circuit plan --checkpoint <hf-id|dir> --hardware <device> --precision declared` plans the
checkpoint (its own config.json, at its declared formats) on any device of
`kernels/DEVICES.toml` (h100-sxm, h200-sxm, b200, gb300, gb10, ...): the fused plan the
device's kernel class compiles, the gap report against that class ranked by the device's
roofline, decode (C1/C16/C128) and prefill (4k/32k) roofline estimates, and the memory fit
with the TP degree it needs. Add `--allow-network` for a checkpoint that is not cached,
`--format plan --mode multi_seq --rows 16` for one plan as text, and
`met circuit display --recipe <r> --hardware <device>` to draw it.

- A layer whose declared format the device cannot run natively is never upcast: Hopper runs
  NVFP4 through the exact E2M1->E4M3 conversion on the FP8 MMA, and the report says so per
  layer. A format no MMA of the device can run is "no path".
- A class plans with its own `common/FUSIONS.toml` over the rules it inherits (`inherits`,
  override by id, `remove`), and its kernels are what its build compiles: a kernel in a source
  guard needs the instruction the guard stands for (`[[guard]]` in DEVICES.toml).
- Only the class's own microbench records make a kernel "Shared"; on every other class it is
  "Shared, unmeasured".
- Add the model to the roadmap matrix (`MATRIX_MODELS` in `crates/server/src/cli/circuit_hw.rs`,
  config fixture under `crates/circuit/tests/fixtures/checkpoints/`), then
  `met circuit plan --matrix kernels/circuits/plans/hw` (the `--check` test holds the reports).

## Step 4: parameterize safely (maximize the intersection)

**Choosing the mechanism:**
- **Runtime argument**: when the parameter does not size registers, shared memory or
  unrolling (strides, counts, eps, scale pointers, flags that do not change the inner loop).
- **Compile-time template parameter**: when it sizes registers, smem or unrolling (head_dim,
  group size, tile shape).
  - Instantiate only the points present in the union of MODEL.toml values; build.rs emits
    the explicit instantiations, and the lookup names encode the point (e.g. `..._hd128`).
  - MODEL.toml is in the certification closure hash; the kernel-lookup audit must enumerate
    the instantiations.
  - Never instantiate every possible value.
- **Policy template, WxAy style** (`kernels/gb10/common/wxay_engine.cuh`: a `Policy` with
  load_w / prep_w / mma / store epilogue), for:
  - weight formats (BF16/W16, FP8, NVFP4, MXFP4);
  - scale layout (per-tensor, per-row, block r×c, group g);
  - activation quantization (dynamic per-token, dynamic group, static per-tensor);
  - activation epilogue (SiLU·mul, ReLU², GELU);
  - routing scoring (softmax, sigmoid + correction bias, scaling factor).

**The stability gate.** Mandatory when parameterizing a kernel an existing model already
uses. All three must hold:
1. **Byte-identical output at every existing point**, from a microtest comparing old and new
   on adversarial inputs over the full row range.
2. **No microbench regression at any existing point**: interleaved reps, within the noise
   band.
3. **The existing models' gates are unchanged**: greedy byte-identity on their recipes, plus
   their certified gates at the next campaign.

Only then does the new point get its own microbench and tuning. The tile choice may itself
become a parameter.

**Worked examples** (Nemotron-3.5-Lightning vs Qwen3.6-35B-A3B and Qwen3.8-27B, from the
checked-in report):
- **Attention head_dim** (Param. opportunity): paged decode attention is a copy per head_dim
  and per KV dtype (`paged_decode_attn_*_128.cu`, `*_512.cu`, `-DHDIM=128` targets). Make
  head_dim a template parameter instantiated from MODEL.toml head_dims, and the KV dtype a
  load policy.
- **BF16 projections at many rows** (Policy variant): the WxAy engine takes W16A16 as a
  policy, so its 1-128-row tensor-core tiers apply to BF16 too.
- **FP8 per-tensor scales** (Policy variant): a per-tensor weight-scale policy (a degenerate
  per-row scale) and a static activation-scale quantizer policy (read the calibrated scale,
  no amax) beside the per-row / block-128 and dynamic per-token / g128 policies.
- **The tensor-core grouped MoE expert kernel** (Policy variant): parameterize its format
  (FP8 or NVFP4 W4A16) and epilogue (gated SiLU·mul or ungated ReLU²), so a new MoE inherits
  the optimized expert path instead of a CUDA-core or one-row fallback.
- **Routing** (Policy variant): one scoring policy; top-k and the scaling factor stay
  runtime.

## Step 5: split when parameterizing is not advised

Split instead of parameterizing when:
- the parameter changes the algorithm (e.g. a different tiling strategy at head_dim 512);
- the instantiation count or compile time explodes;
- an existing point would regress.

How to split:
1. Break the op into composable kernels (e.g. separate the activation quantizer from the
   GEMV; the scoring from the top-k).
2. Build the target from the pieces.
3. Let the auto-fuser recombine them with `bit_identical` fusion rules, each with a
   microtest.

## Step 6: complete the architecture package

1. **Circuit** (from step 3), now with every mode the model runs: decode, multi-sequence,
   verify, draft, prefill, state operations.
2. **Config mapping**: a declarative map from `config.json` to shape. Keys that change the
   math and are not mapped are refused.
3. **Bindings**: weight name patterns, then transforms, then device weights, driven by the
   declared precision plan and `KernelCaps`.
4. **State schema**: KV, SSM/conv state, MTP KV, prefix snapshots, rollback checkpoints.
5. **MODEL.toml**: an exact-match `[[model_types]]` pin (e.g. `hidden_size`), the sampling
   defaults, and serving metadata by id (tokenizer, chat template, tool/reasoning parser).
   Serving metadata is referenced, never re-implemented.
6. **Rules and checks**: `FUSIONS.toml` rules for the new nodes (each citing its dispatch
   site in `ROUTING-AUDIT.md`), `met circuit show` / `display` renders, golden plans with
   the `--check` test, and `golden = true` once the rules cover the model.

## Step 7: correctness first, then optimize the kernels outside the diagram one by one

1. **Reference kernels first.**
   - A family with a legacy path: byte parity with `met circuit diff` (eager and graphed,
     every padded width, a detection control, a legacy-repeat control).
   - A new family: logits parity against Hugging Face transformers on fixed prompts, with
     the tolerance declared per format.
2. **Optimize.** For each Novel or Shared-unmeasured kernel, in step-share order:
   1. Microbench it against its roofline (`KERNEL-PERF.md`).
   2. Profile it (nsys for which kernel, ncu for why).
   3. Optimize, and re-prove parity after every change.
   4. Record the microbench in `docs/kernel-perf/measurements.toml` and as evidence at that
      parameter point in `KERNEL_FAMILIES.toml`.

   Never call a kernel optimized without that record.
3. **Accuracy at the declared precision**: the full BFCL draw plus agentic-webserver, each
   against a same-box, same-night control. A numerics change needs this evidence before it
   becomes a default.

## Step 8: fuse

- Add fusion rules to `FUSIONS.toml`:
  - `bit_identical` names a microtest and applies by default;
  - `differs` needs an opt-in lever disclosed on records;
  - `reference` is only for grandfathered existing defaults.
- Regenerate the golden plans and measure launches per step (nsys), tok/s and J/tok.
- Keep only the fusions that win.

## Step 9: beat vLLM, then certify

1. **vLLM baseline**: same box, same instrument (the concurrency ladder at C1-C128 with NVML
   energy; no request timeout at C >= 64), running vLLM's published configuration. A vLLM
   stall is a publishable result.
2. **Win on both axes**: tok/s AND J/tok at every rung from C1 to C128. If a lever buys
   tok/s but costs J/tok, say so.
3. **Recipe and BENCH.toml entries**: measure-then-declare bounds; pin the precision tier
   explicitly (`weight_quantization`).
4. **Certify and merge** as `AGENTS.md` and `CONTRIBUTING.md` describe: the certification
   campaign for perf paths, then `/stamp` and `/seal`, then the merge queue.

## Rules that always apply

- **Numerics**: bit-identical by default. The canonical-tier / row-invariance policy applies
  to MoE paths. Keep high-sensitivity intermediates (e.g. the SiLU product) near FP32.
- **Evidence**: "optimized" means a microbench record at that parameter point and row count;
  unknown means unmeasured, never optimized.
- **Shared machines**: never build or run anything during someone else's timed measurement;
  GPU memory utilization at most 0.85 on GB10; start servers detached (`setsid -f`) and stop
  them by PID.
- **Before handing back**: `cargo fmt`, clippy `-D warnings` (with the no-CUDA environment of
  `CONTRIBUTING.md` where needed), tests, `cargo doc`, the 500-line cap, SPDX headers, dated
  comments and typos (`AGENTS.md`, "Local checks before a PR").
- **Verification**: verify a delegated result with a control before building on it.
