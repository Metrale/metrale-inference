# Laguna XS 2.1 INT4 on strix-hip (gfx1151)

Target: `poolside/Laguna-XS-2.1-INT4` @ `4b7e28abdc0a8b121def816b89d631750bc53c92` on the
native HIP target `strix-hip` (gfx1151). Not the SCALE-based `strix` target.

Status, October 8, 2026: serves on a Radeon 8060S (gfx1151, native Ubuntu 24.04, ROCm
7.2.1). The target compiles with hipcc, the packed-int kernels match their host emulation
bit for bit on the device, the packed-int MoE layer runs every MoE layer, the boot kernel
gate passes, and next-token predictions agree with the checkpoint's own modeling code (see
"GPU gates on gfx1151"). Not certified; one first C1 measurement only. The NVFP4 GB10
target of the same model is unchanged (see "Unchanged targets").

## Checkpoint facts

Read from `config.json` and the five safetensors headers at the pinned revision
(HTTP range requests; 23,889,444,000 bytes of tensors).

- `quantization_config`: compressed-tensors, `format` and both groups `pack-quantized`,
  `quantization_status` `compressed`, no sparsity.
  - `group_0`: routed experts of layers 1-30 (`re:.*layers\.([1-9]|[12]\d|30)\..*(w[1-3]|gate_proj|up_proj|down_proj)$`),
    INT4, symmetric, `strategy` group, `group_size` 128, static, no `actorder`,
    no `zp_dtype`, no `scale_dtype`, `input_activations` null.
  - `group_1`: the same projections of layers 31-39, **INT8**, otherwise identical.
  - `ignore`: `lm_head`, layer 0's dense `gate_proj`/`up_proj`/`down_proj`, every
    `self_attn.{q,k,v,o,g}_proj`, every router `mlp.gate`, every
    `mlp.shared_expert.{gate,up,down}_proj`. The shared expert matches group_0's pattern
    and is ignored: the ignore list wins.
  - `kv_cache_scheme`: FP8 float, per tensor, static; `self_attn.k_scale` / `v_scale`
    are BF16 `[1]` per layer.
  - `transform_config.R1`: a deterministic Hadamard of size 128 (`head_dim` 128), applied
    at `weight_output` of `embed_tokens`, `o_proj`, `down_proj` and inverted at
    `weight_input` of `q/k/v/g_proj`, `gate_proj`, `up_proj`, the router and `lm_head`.
- Expert tensors: `weight_packed` I32 and `weight_scale` BF16, no zero-point or shape
  tensors. INT4 gate/up `[512, 256]` + `[512, 16]`, down `[2048, 64]` + `[2048, 4]`;
  INT8 gate/up `[512, 512]` + `[512, 16]`, down `[2048, 128]` + `[2048, 4]`.

## Packed layout, and how it was established

compressed-tensors `pack_to_int32` adds `2^(bits-1)` to the signed code and ORs field `i`
of each group of `32 / bits` consecutive K values into bits `[bits*i, bits*(i+1))` of a
32-bit word. So: words little-endian, fields least significant first, offset binary
(`q = u - 8` for INT4, `q = u - 128` for INT8), one scale per 128 K values.

Checked against the real tensors, not only the format source
(`scripts/laguna/int4_layout_probe.py`, about 6 MB of range requests). Expert 0 gate_proj
was decoded under all four (field order x sign) conventions and correlated with the BF16
release (`poolside/Laguna-XS-2.1` @ `c5f36269`) after the declared transform:

| Decode | Layer 1 (INT4) | Layer 35 (INT8) |
| --- | --- | --- |
| LSB-first, offset binary | +0.99394 | +0.99997 |
| LSB-first, two's complement | -0.59605 | -0.58868 |
| MSB-first, offset binary | +0.01231 | -0.00040 |
| MSB-first, two's complement | -0.00776 | +0.00074 |

The reference that matches is `((W_bf16 * g_bf16) @ H) / g_int4`: block-diagonal
normalized Sylvester Hadamard `H` over K in blocks of 128, the BF16 release's
`post_attention_layernorm` weight folded in, and the INT4 checkpoint's own (different,
0.875 to 1.14) norm weight divided out. Without the rotation the correlation is about 0;
without the norm terms it drops to 0.973 (layer 1) and 0.997 (layer 35). Consequence: the
transforms are fused into the stored weights and norms, and the runtime runs the
checkpoint's tensors as written. The admission therefore accepts weight-located transforms
and refuses any online location.

## Admission (crates/config)

`precision_plan::packed_int::admit_packed_int` accepts only symmetric, static, group-128,
weight-only INT4/INT8 `pack-quantized` groups stored compressed, and refuses, naming the
field: asymmetric weights, zero points, activation ordering (`group`, `weight`, `true`),
group sizes other than 128 (or none), strategies other than `group`, widths other than 4
and 8, dynamic weight scales, block scales, a `scale_dtype` override, quantized input or
output activations, formats other than `pack-quantized` (including one inherited from the
block), mixed integer and float groups, any status but `compressed`, sparsity, and
transforms at any location but `weight_input` / `weight_output`.

The Laguna parser additionally requires that only routed experts are integer, every MoE
layer's experts share one admitted scheme (`routed_expert_scheme`), and attention, the head
gate, the dense layer, the router, the shared expert and `lm_head` are unquantized. The
quantization label derives `INT4` from an integer group_0, so the server labels the
checkpoint `int4` and pairs it with the `int4` kernel target. Float checkpoints take none
of these paths.

## Kernel target `kernels/strix-hip/laguna-xs-2.1/int4`

- `MODEL.toml`: `[[model_types]] laguna / 2048`, `supported_quants = ["int4"]` (the
  per-model precision allowlist: `nvfp4`, `bf16`, `fp8`, `mxfp4` refuse
  the common-only fallback), sampling and behavior mirrored from the GB10 target, and 78
  `[expected_absent]` declarations from the first gfx1151 boot (see "Lookups").
- `KERNEL.toml`: `-DHDIM=128` as the GB10 Laguna target, `packed_int_gemv.cu`,
  gb10/common `dense_gemv_bf16_batchm.cu` (the attention head gate's decode GEMV), the Qwen
  strix-hip `w4a16_gemm.cu` fork (model init requires `w4a16::w4a16_gemm`; it is never
  launched for INT4 weights), and fail-closed shadows with no entry points for the
  strix-hip/common no-op MoE stubs (`moe_w4a16_grouped_gemm*`, `moe_fp8_grouped_gemm_v2`),
  `moe_fp8_grouped_gemm` and gb10's `gated_delta_rule`.
- `packed_int_gemv.cu` + `packed_int_dequant.cuh`: `packed_int{4,8}_gemv_g128` (dense
  decode, `y[M, N]`) and `moe_packed_int{4,8}_gemv_ptrtable_g128` (grouped experts: slot
  `s` uses `expert_ids[s]`, activation row `s / x_row_div`, per-expert word and scale
  pointer tables; an out-of-range id or null pointer writes 0), and `moe_packed_int_combine`
  (slot-order weighted sum plus the shared expert, the bits of `moe_unpermute_blend` with an
  identity permutation and no shared-expert gate). Block 256 = 8 wave32
  waves, one output column per wave, codes dequantized in registers, fp32 products and sums
  with no FMA, xor-shuffle tree 16/8/4/2/1 reduction, BF16 round-to-nearest-even store. The
  word-at-a-time inner product is the unit a later `V_DOT4_I32_IU8` path replaces (TODO in
  the header). The lookup pair is pinned by
  `metrale_model_layers::quant_format::packed_int::packed_int_gemv_kernels`.

Host evidence for the kernels (no GPU):

- `scripts/laguna/packed_int_gemv_host_check.cpp` runs the header's decode and lane
  arithmetic for 32 lanes in the kernel's reduction order: literal known answers pass; the
  MSB-first and two's-complement controls fail them (23 and 29 mismatches); random INT4/INT8
  `[64, 2048]` and `[64, 512]` and the real layer-1 gate/down (INT4) and layer-35 gate (INT8)
  tensors match a double reference within one BF16 ulp (worst 0.48 of tolerance); grouped
  slot mapping and padding are exact.
- A device-only compile of `packed_int_gemv.cu` for `gfx1151` with Homebrew LLVM and a
  minimal stand-in for `hip_runtime.h` (no ROCm) succeeds: wave32, 20 to 28 VGPRs, no
  float FMA emitted. This proved syntax and codegen against the LLVM AMDGPU backend only;
  the hipcc compile with the repo's compat headers is GPU gate 1 below.
- `metrale_model_layers::quant_format::packed_int` is the Rust CPU reference: literal-word
  known answers, wrong-order and wrong-sign controls that must fail, checkpoint tensor
  shapes, and a symmetric min-max round trip.

## Dispatch and loader

- `metrale_model_weights`: `WeightDtype::Int32` keeps `weight_packed` words as stored
  (every loader refused I32 before).
- `metrale_model_layers::layers::packed_int_moe::PackedIntMoeLayer`
  (`FfnComponent::PackedIntMoe`): per pass of n rows, the BF16 router GEMM
  (`dense_gemm_bf16`), `moe_topk_sigmoid_batched` (selection on sigmoid plus the correction
  bias, unbiased weights normalized and scaled by 2.5), the BF16 shared expert, the grouped
  gate and up GEMVs over per-expert pointer tables in slot order, SiLU times up, the grouped
  down GEMV and `moe_packed_int_combine` into `moe_output`. Every row count (decode, k2/k3,
  prefill, batched) runs this one path. It never builds the NVFP4 `MoeLayer`.
- Laguna loader (`weight_loader/laguna/packed_int.rs`): a MoE layer whose expert 0
  `gate_proj.weight_packed` is I32 takes this path with the scheme the admitted precision
  plan declares (INT4 g128 in layers 1 to 30, INT8 g128 in 31 to 39); every tensor's dtype
  and shape is checked before upload. NVFP4 checkpoints store U8 there and keep their path.
- Mock-backend tests pin the launch order, grids and arguments, the pointer tables, the
  INT8 kernel choice and the refusals (denied kernels, missing or null experts, rows beyond
  the arena with no launch, malformed tensors, undeclared I32 words, the NVFP4 control).

## Lookups on Laguna's load path

`crates/kernels/tests/strix_hip_laguna_int4.rs` scans every literal kernel lookup in the
Laguna loader, the attention, dense-FFN and MoE constructors, the model-level kernels and
head, and the packed-int lookup, and classifies each against this target
(`-- --nocapture laguna_lookup_inventory` prints the table). Written before the GPU work
(279 lookups, 149 resolving); the current counts are 284 lookups: 151 resolve, 65 are
`[expected_absent]`, and 68 stay classified gaps (Fp8WeightOnly 15, Nvfp4Only 15,
OtherModelFeature 14, HipMissing 18, HopperOnly 4, GgufOnly 2). The first table:

| Class | Lookups | Required | What |
| --- | --- | --- | --- |
| Resolves | 149 | 103 | strix-hip/common attention (paged decode/prefill BF16, FP8, NVFP4 KV), norms, RoPE, BF16 GEMV/GEMM, sigmoid top-k, MoE permute/sort, shared-expert fusions, argmax, packed_int_gemv |
| Nvfp4Only | 35 | 5 | `moe_w4a16` grouped/fused NVFP4, NVFP4 MMQ, W4A4, NVFP4 dequant/requant |
| Fp8WeightOnly | 27 | 1 | E4M3 W8A16 GEMV/GEMM tiers, block-scaled and W8A8 MoE, FP8 lm_head |
| TurboKvOnly | 18 | 0 | turbo and rotated KV prefill/decode |
| OtherModelFeature | 16 | 2 | hash, softmax-bias, sqrt-softplus routing, softcap, embed scale, GELU, SSM, MRoPE, BF16 experts, atomic C4 decode |
| GgufOnly | 9 | 0 | Q2_0, Q4_K |
| HopperOnly | 4 | 0 | sm_90 split-K decode |
| HipMissing | 21 | 0 | device token feed, M16 BF16 GEMM, multi-row BF16 GEMV, router GEMMs, BF16 shared-expert fusion, fused K-norm/RoPE/cache writes (BF16, FP8 KV), GQA decode, strided SiLU, unpermute blend, BF16 prefill twins |

The eight required unresolved lookups are the NVFP4 `MoeLayer` constructor's
`moe_w4a16` (and its FP8 twin) entries and two feature-gated ones (GELU, embed scale). So
the existing NVFP4 MoE layer cannot be constructed on this target, by design: the INT4
expert path must be a separate dispatch that never constructs it. Every HipMissing lookup
is a probe with a resolved fallback today. The test pins this classification
(`tests/strix_hip_laguna_int4/gaps.rs`): a lookup that starts resolving or disappears
fails it, and no required lookup may be HipMissing.

The first `met serve --check-kernels` on gfx1151 with the real weights made 180 lookups and
left 79 unresolved. `dense_gemv_bf16_batchm` is on the serving path (the head gate launches
it every decode step), so it is now built. The other 78 are `[expected_absent]`, each with
the reason its dispatch cannot run for this checkpoint: the layer-0 dense FFN and every
attention projection are BF16 with no NVFP4, FP8 or Q2 weights, KV is FP8 or BF16 (never
turbo), the head is BF16, and there are no SSM layers or LoRA overlays. Of the 78, the 65
that the static scan sees moved out of `GPU_GATE_GAPS`; the other 13, whose entry names
pass through closures or layers/mod.rs helpers, are pinned by site
(`COMPUTED_NAME_LOOKUPS`). Boots with FP8 and BF16
KV then pass (180 lookups, 0 unresolved, 78 declared); a build with one declaration
removed fails the gate naming it.

## Unchanged targets

`scripts/lib/kernel_layout.py dump` before (`91cc4ae`) and after: all 58 existing
targets resolve identically (sources, layers, shadows, configs, module names); the only
difference is the added `strix-hip/laguna-xs-2.1/int4`. The INT4 work changed no
`kernels/gb10` file. One `kernels/strix-hip/common` file changed, as a bug fix: `dense_gemm_tc.cu` loaded only rows
0..7 of its 16-row A tile (one 128-thread pass over 256 elements), so every M > 8 read
uninitialized shared memory; every strix-hip target that compiles it gets the fix. The Laguna routing test is now per hardware:
gb10 routes XS and S as before; strix-hip routes XS to this target and claims no S.

## GPU gates on gfx1151

Host: Radeon 8060S (gfx1151), Ubuntu 24.04.4, kernel 7.0.0, ROCm 7.2.1, rustc 1.93.1.
Build: `METRALE_TARGET_HW=strix-hip METRALE_TARGET_MODEL=laguna-xs-2.1
METRALE_TARGET_QUANT=int4 CUDARC_CUDA_VERSION=13000 METRALE_NO_RDMA=1 cargo build --release
-p metrale-server --no-default-features --features cuda --bin met`. Weights checked against
the Hub LFS sha256 of every safetensors shard at the pinned revision. Serve flags for gates 5
to 7: `--max-batch-size 1 --kv-cache-dtype fp8|bf16 --lm-head-dtype bf16 --swap-space-gb 0
--gpu-memory-utilization 0.85 --max-seq-len 4096 --activation-quantization adaptive
--forward legacy`.

1. hipcc compile: all 92 kernels of the target compiled the first time, with no source
   change (93 with the head gate GEMV).
2. `packed_int_gemv` on the device (the built code object, loaded with `hipModuleLoad`), 140
   checks: dense INT4/INT8 at M 1 and 3, random `[512, 2048]` and `[2048, 512]`; the real
   expert tensors of layers 1 and 30 (INT4) and 31 and 35 (INT8); grouped launches with
   pointer tables, `x_row_div` 8 and 1, and padding slots (id -1, id E, null pointer); the
   combine. Device output equals the host emulation bit for bit everywhere, is within one
   BF16 ulp of an f64 reference (worst 0.496 of tolerance), and padding writes 0. Controls
   detected: MSB-first decode, complemented weight words, the wrong expert, an unrounded
   routed sum. First run under WSL2 on the same machine, then reproduced bit for bit
   on native Ubuntu.
3. Dispatch and loader: above.
4. Boot kernel gate: above.
5. Parity. Reference: the checkpoint's own `modeling_laguna.py` (transformers 5.19, torch
   2.14 CPU, BF16, eager attention, BF16 KV) over the same checkpoint, experts dequantized as
   compressed-tensors does (`q * scale` rounded to BF16) and every other tensor as stored.
   The server's `/v1/completions` prompt logprobs (top 5, echo) and 32-token greedy
   continuations are compared on the same token ids.

   | Prompt set | KV | Next-token agreement | Greedy, tokens before first divergence |
   | --- | --- | --- | --- |
   | 7 short (20 to 57 tokens) | BF16 | 242/273 (88.6%) | 138/196 |
   | 7 short | FP8 | 245/273 (89.7%) | 130/196 |
   | 2 long (704, 1,523 tokens) | BF16 | 2,187/2,227 (98.2%) | 64/64 |
   | 2 long | FP8 | 2,188/2,227 (98.3%) | 64/64 |

   The five plain-text short prompts agree at 158 of 165 positions; the two chat prompts
   carry most short-prompt disagreements, mostly inside the system prompt where both
   implementations are near-uniform (reference top-2 margins of 0 to 0.375 nats at all but
   one position). Every greedy divergence is at a reference margin of 1.125 or less. On the
   long prompts the median absolute logprob difference of the reference's top-1 token is
   0.0002 (p90 0.019); a few positions inside the repeated "courier" split of the recall
   text disagree with large margins in both directions, where the two implementations each
   predict tokens a correct reading would not. Layer parity (last row of the residual
   stream entering each layer, BF16 KV): exact at layer 0, 0.4 to 1.5% relative L2 after
   layer 0 (attention block 0.5 to 1.2%, dense FFN 1.0 to 6.5%, two prompts), growing to at
   most 27% in the middle and late layers (cosine 0.964 or higher).
6. Serve smoke (chat API, FP8 KV, the model's template): factual answer, arithmetic
   (48 x 52 = 2496), an `is_prime` function that passes executed asserts, a `get_weather`
   tool call with `{"city":"Paris"}` and a correct answer from the tool result, and a
   streamed answer: 6 of 6.
7. First measurement on gfx1151, C1, not a certification: 704 prompt tokens and 256 output
   tokens (FP8 KV, max batch 1, one warmup): first token 2,758 ms (255 tok/s prefill),
   decode 24.2 tok/s.

Found on the way: `dense_gemm_tc` (above) turned every prompt of 10 or more tokens into
NaN through the BF16 dense FFN of layer 0.

## Open

- Concurrency, soak, and the arithmetic, streaming and tool suites at C2 and above.
- Speed: grouped GEMVs one column per wave, no `V_DOT4_I32_IU8`, decode and prefill
  through the same row path; throughput, memory and energy with build identity.
- `/v1/completions` with a text prompt counted 703 tokens for a text the checkpoint
  tokenizer encodes as 704 with its BOS, and that run's output degenerated into a
  repeated line; gate 7 used the 704 token ids. Not investigated.
- `--lm-head-dtype fp8` and LoRA overlays are not qualified on this target.
