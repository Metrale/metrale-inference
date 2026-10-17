# GLM Campaign: GLM-5.3 Flash on 3×GB10 with DFlash2

<!-- 2026-10-08: The campaign target file. It follows targets/TEMPLATE.md of the new-hardware kit
     (the H100 beachhead PR): every claim cites the tree at the branch base; "unverified" says so. -->

- **Campaign PR:** the PR titled "GLM Campaign: GLM-5.3 Flash on 3×GB10 with DFlash2".
- **Branch / base:** `glm/glm53-flash-3node` on `main` at `3e954ac14`.
- **Devices:** three GB10 boxes (`gb10` in `kernels/DEVICES.toml`), one GPU each, joined by a
  RoCE triangle (one cable per pair of boxes, no switch). **Kernel class:** `kernels/gb10/`.
- **Model:** `nvidia/GLM-5.3-Flash-NVFP4` (ModelOpt 0.47; `model_type = glm5_next`;
  320B total / 18B active) with the DFlash2 drafter `incoai/GLM-5.3-Flash-DFlash2`.
- **Status:** audit and baseline done; no Metrale serve has run on this topology yet.
- **Done means:** Metrale serves this model on the three boxes with every serving setting of the
  reference vLLM deployment (below), passes bit parity (Tiers 1-2) and the accuracy bar
  (Tier 3) at or above vLLM, and is faster than vLLM on tok/s at every concurrency C1-C16 on
  the same boxes. TTFT, J/tok (sum of the three boxes' NVML energy counters over one window)
  and C > 16 are measured and reported.
- **Ledger:** `ledger/gb10x3.toml` (milestones) and `ledger/levers-glm53.toml` (the lever
  journal, same schema as the kit's `levers.toml`, folded into it once the kit merges).
- **Loop budget:** 10 calendar days or 60 three-box device-hours of loop measurement, whichever
  comes first; then escalate.

## 1. The reference deployment (what we must match)

A patched vLLM nightly (`0.30.1rc1.dev193+gddd6fbca1`), one rank per box, TP=3 over Ray.

| Setting | vLLM value | Metrale today |
|---|---|---|
| parallelism | TP=3 (heads padded 64→66, expert intermediate 2048→2304 at load) | TP=3 refused: `serve_phases/topology.rs:89-121`, `glm5next_kda/tp.rs:129-141`, `glm5next_mlp/mod.rs:292-310`; EP=3/TP=1 passes the checks, never run |
| context | `--max-model-len 262144` | `--max-seq-len`; indexer cache flat per sequence at max-seq-len (`glm5next_dsa/mod.rs:176-183`): ~1.5 GB per sequence at 256k |
| KV | fp8_e4m3, 18 GiB pin, block 3456 | latent cache always FP8, `kv_scale` 1.0 hardcoded (`weight_loader/glm5_next_load/loader.rs:182`); the checkpoint's `kv_cache_scheme` is not read |
| batch | `--max-num-seqs 64`, 16384 batched tokens, long-prefill threshold 2304 | multi-sequence decode vetoed (`glm5next_layer/mod.rs:255-262`), so every sequence decodes alone |
| prefix caching | on, 64-token match unit | off for `glm5_next` (`config/src/kv_completeness.rs:18-30`) |
| speculation | DFlash2, 7 drafts, per-batch-size draft table, adaptive-K scheduler, draft truncation | DFlash generic; no GLM wiring; no multi-rank DFlash (`verify_dflash_step.rs:8`); batched verify refused (`glm5next_layer/mod.rs:268`) |
| tool calls | GLM-4.7 XML format, fail-closed + repair of nested/abandoned calls, keep-alives | `poolside_v1` parses the format (`tool_defaults.toml:21`); no fail-closed/repair equivalent |
| reasoning | glm45 parser, `<think>` always parsed | falls back to the `qwen` parser (`serve_phases/tokenizer_runtime.rs:58-84`) |
| chat template | checkpoint template + "thinking off means low effort"; default `reasoning_effort: low` | `./jinja-templates/glm5_next.jinja` override only, no file flag; `--default-chat-template-kwargs` exists |
| images | up to 16 per prompt, encoder data-parallel | GLM ViT wired behind `METRALE_GLM_VISION=1` (`config/src/parsers/glm5_next/parse.rs:179-185`); no per-prompt cap |
| precision as executed | routed experts W4A4 (CUTLASS) at prefill, W4A16 for decode batches of up to 8 tokens; **dense BF16 layers re-quantized below the checkpoint**: FP8 per-channel/per-token, NVFP4 W4A16 for KDA projections, attention `o_proj` and shared experts, FP8 `lm_head` | W4A16 experts (`*.input_scale` skipped: `serve_phases/weights.rs:392-404`); dense layers BF16 as declared |

The vLLM precision row is the campaign's main parity question; see section 4.

## 2. Audit: the gap list

Each item cites the tree; estimates are agent working hours, CPU-side unless marked GPU.

1. **TP=3 for attention and dense layers** (64 heads, 2048 shared-expert width, neither
   divisible by 3). Parameterize the head and column split per rank (22/21/21 heads,
   uneven column ranges) instead of padding weights; the slowest rank holds 22 heads either
   way, and no zero weights occupy memory. Touches the topology checks, the KDA/DSA/MLP TP
   plans and the loader slices. *8-12 h.*
2. **Experts at 3 ranks**: EP=3 (96 experts per rank, already legal) vs TP=3 on experts
   (2048/3 needs padding, +12.5 % bytes). At C1 the busiest EP rank holds about 4 of the 8
   routed experts against 2.67 balanced; TP padded reads 3.0. Decide by measurement; start
   with world = TP = EP = 3 (the existing overlapping-group shape). *4 h + GPU.*
3. **A fast 3-rank all-reduce.** Decode does two reductions per layer (90 per step); NCCL's
   small-message latency (tens of microseconds each) is milliseconds per step at C1. The
   custom path exists only for world 2 (`crates/comm/src/nccl_backend.rs:352-362`). *8-16 h.*
4. **Batched multi-sequence decode** for KDA, DSA (sparse MLA + indexer), mHC and the MoE: the
   per-sequence fallback is the top gap above C1. *16-24 h + GPU.*
5. **DFlash2 for GLM**:
   - the tap: the engine copies `hidden_states()` (the mHC pre-mix output y), while the
     drafter was trained on each target layer's completed output averaged over the four mHC
     streams;
   - the drafter config: `rope_theta` sits inside `rope_parameters` and is read as 1e7
     instead of 1e4 (`weight_loader/dflash_loader/config.rs:33-41`); the trained 2048
     sliding window is not read (`--dflash-window-size` defaults to 4096);
   - multi-rank verify: the worker protocol has no γ-width verify command (opcodes reserved,
     `decode_checkpoint/plan.rs:60-65`), and capture runs only on rank 0;
   - batched verify for the GLM layer, with KDA state rollback at width γ+1.
   *24-32 h + GPU.*
6. **KDA state at 64 sequences.** One FP32 state per sequence is ~142 MB before TP split
   (34 layers × 64 heads × 128²); per-draft-position intermediates multiply it by γ+1.
   Keep one state per sequence and replay accepted tokens after verify (exact), as the vLLM
   deployment does. *8-12 h + GPU.*
7. **Context to 256k at 64 sequences**: page the indexer cache like the latent cache; read
   `kv_cache_scheme`; size the pool. *8-12 h.*
8. **Prefix caching for the hybrid** (KDA state snapshots + DSA indexer state), the gate in
   `kv_completeness.rs`. *8-12 h + GPU.*
9. **Precision as declared**: W4A4 NVFP4 for the routed experts and layers 0-2 (the
   checkpoint declares `input_activations` FP4 g16); the GLM loader and kernels ignore the
   declared plan today. *8-12 h + GPU.*
10. **Serving surface**: a GLM tool parser that fails closed and repairs nested and abandoned
    calls; a GLM reasoning parser; a chat-template file flag (or a checked-in
    `glm5_next.jinja`); `fp8_e4m3` accepted as a spelling; a per-prompt image cap; vision on
    by default for this checkpoint. *12-16 h.*
11. **Do not load the MTP layer** (13.8 GiB of BF16 experts in layer 45) when DFlash2 is the
    speculator. *2 h.*
12. **A circuit and a Venn**: no GLM circuit exists in `kernels/circuits/`; the IR has no MLA,
    sparse-indexer, KDA-gate or mHC ops (`crates/circuit/src/ir.rs:104-200`). *8 h.*
13. **A 3-node launcher and recipe**, NCCL over both RoCE ports of the triangle, a scripted
    restore of the reference deployment at the end of every window. *4 h.*

## 3. What transfers

| Work | Transfers? | Why |
|---|---|---|
| NVFP4 grouped MoE kernels (gb10) | as code | measured on the Qwen3.6/3.8 MoE shapes, not at 4096×2304 nor 96 experts per rank |
| DFlash2 drafter head and kernels | as code | the drafter is the same 5-layer DFlash2 family as Qwen3.8's; θ and window differ |
| GDN verify rollback (`gated_delta_rule_wy17`) | partly | KDA's gate is per-channel, GDN's per-head; the rollback pattern transfers, the kernel does not |
| world-2 custom all-reduce | no | partner-exchange only |
| GLM KDA/DSA/mHC kernels | yes | single-sequence today; batching is new work |

## 4. Layout and precision decisions

**Layout: world = 3, TP = 3 for attention and dense layers, experts EP = 3 first.** The model
is 190 GiB; it fits only across all three boxes (~59 GiB of weights per rank once the unused
MTP layer is skipped, plus ~3.3 GiB replicated: the vision tower and the drafter). The per-token bytes decide the rest: dense layers
are 14.8 GiB of BF16 and are read every step, routed experts 4.4 GiB at top-8. Replicating
attention (EP only) reads all 14.8 GiB on every rank, a floor of ~60 ms per step; TP=3 cuts it
to ~5 GiB. There is no pipeline parallelism in the engine, and at C ≤ 16 PP would only add
bubbles to a bandwidth-bound decode.

**Precision: declared by default.** The checkpoint declares BF16 for attention, shared experts
and `lm_head`; the reference deployment runs them below that (FP8 and NVFP4). Our default keeps
the declared formats; a below-declared dense tier is an opt-in flag, disclosed on every record,
with the accuracy bar. At declared precision the C1 bandwidth floor is ~24 ms per verify step
per rank; the reference deployment's measured verify steps are 30-49 ms, so a declared-precision
win at C1 is possible but not certain. **Owner decision (2026-10-08): declared first, matched as a
fallback.** The campaign aims to win at the declared BF16 dense formats. A rung that cannot be won
at declared precision may use a matched below-declared tier (FP8 / NVFP4 W4A16 dense, as the
reference deployment runs), shipped as an opt-in flag, default off, disclosed in the parity record
and on every number it produces, behind the accuracy bar; it then counts toward the C1-C16 bar.

## 5. Checklist

- [ ] Gap 13: launcher + restore script; boot the existing path at world 3 (EP=3, TP=1) on
      real weights; fabric check; Tier-1 probes. (window 1)
- [ ] Gaps 1-3: TP=3 split, 3-rank all-reduce; parity vs the EP=3 run. (window 2)
- [ ] Gaps 4, 6, 7, 9: batched decode, KDA replay, paged indexer, declared W4A4. (windows 3-4)
- [ ] Gap 5: DFlash2 for GLM; acceptance measured first (cheap decisive number). (window 5)
- [ ] Gaps 8, 10, 11: prefix caching, serving surface, MTP skip.
- [ ] PARITY-O.R.A.C.L.E on both resolved configs; then the improvement loop.
- [ ] Accuracy bar: BFCL (same draw), agentic-webserver with a control, the long-context probes.

## 6. Log

| Date | Step | Result | Evidence |
|---|---|---|---|
| 2026-10-08 | audit | gap list above | this file |
| 2026-10-08 | vLLM baseline | ladder C1-C16 47.5 / 65.7 / 87.7 / 118.8 / 133.8 / 158.5 tok/s; BFCL n=995 85.53 (normalized 85.54); agentic-webserver 10/10, 7.96 s/turn; long context 8/8 | PR comments |
| 2026-10-08 | precision | owner: declared first, matched tier as an opt-in fallback | section 4 |
