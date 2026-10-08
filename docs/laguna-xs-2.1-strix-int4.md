# Laguna XS 2.1 INT4 on strix-hip (gfx1151): host-side phase

Target: `poolside/Laguna-XS-2.1-INT4` @ `4b7e28abdc0a8b121def816b89d631750bc53c92` on the
native HIP target `strix-hip` (gfx1151). Not the SCALE-based `strix` target.

Status, October 7, 2026: host-side only. The config admission, the weight layout and its
CPU reference, the kernel target and the HIP kernel sources exist and are tested on the
host. Nothing has been compiled by hipcc, launched, or compared on a GPU, and no Rust
dispatch runs the packed-int experts yet. The NVFP4 GB10 target of the same model is
unchanged (see "Unchanged targets").

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
  allowlist ported from the GPT-OSS packaging work: `nvfp4`, `bf16`, `fp8`, `mxfp4` refuse
  the common-only fallback), sampling and behavior mirrored from the GB10 target. No
  `[expected_absent]` yet (see "Lookups").
- `KERNEL.toml`: `-DHDIM=128` as the GB10 Laguna target, `packed_int_gemv.cu`, the Qwen
  strix-hip `w4a16_gemm.cu` fork (model init requires `w4a16::w4a16_gemm`; it is never
  launched for INT4 weights), and fail-closed shadows with no entry points for the
  strix-hip/common no-op MoE stubs (`moe_w4a16_grouped_gemm*`, `moe_fp8_grouped_gemm_v2`),
  `moe_fp8_grouped_gemm` and gb10's `gated_delta_rule`.
- `packed_int_gemv.cu` + `packed_int_dequant.cuh`: `packed_int{4,8}_gemv_g128` (dense
  decode, `y[M, N]`) and `moe_packed_int{4,8}_gemv_ptrtable_g128` (grouped experts: slot
  `s` uses `expert_ids[s]`, activation row `s / x_row_div`, per-expert word and scale
  pointer tables; an out-of-range id or null pointer writes 0). Block 256 = 8 wave32
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
  float FMA emitted. This proves syntax and codegen against the LLVM AMDGPU backend only;
  the real hipcc compile with the repo's compat headers is still a GPU-gate item.
- `metrale_model_layers::quant_format::packed_int` is the Rust CPU reference: literal-word
  known answers, wrong-order and wrong-sign controls that must fail, checkpoint tensor
  shapes, and a symmetric min-max round trip.

## Lookups on Laguna's load path

`crates/kernels/tests/strix_hip_laguna_int4.rs` scans every literal kernel lookup in the
Laguna loader, the attention, dense-FFN and MoE constructors, the model-level kernels and
head, and the packed-int lookup, and classifies each against this target
(`-- --nocapture laguna_lookup_inventory` prints the table): 279 lookups, 149 resolve
(103 of them required), 130 do not.

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

The six required unresolved lookups are the NVFP4 `MoeLayer` constructor's
`moe_w4a16` (and its FP8 twin) entries and two feature-gated ones (GELU, embed scale). So
the existing NVFP4 MoE layer cannot be constructed on this target, by design: the INT4
expert path must be a separate dispatch that never constructs it. Every HipMissing lookup
is a probe with a resolved fallback today. The test pins this classification
(`tests/strix_hip_laguna_int4/gaps.rs`): a lookup that starts resolving or disappears
fails it, and no required lookup may be HipMissing. The boot gate's
`[expected_absent]` is written from the first `met serve --check-kernels` on gfx1151, not
from this static table.

## Unchanged targets

`scripts/lib/kernel_layout.py dump` before (`91cc4ae`) and after: all 58 existing
targets resolve identically (sources, layers, shadows, configs, module names); the only
difference is the added `strix-hip/laguna-xs-2.1/int4`. No `kernels/gb10` or
`kernels/strix-hip/common` file changed. The Laguna routing test is now per hardware:
gb10 routes XS and S as before; strix-hip routes XS to this target and claims no S.

## Remaining GPU gates (in order)

1. hipcc compile of the whole target (`METRALE_TARGET_HW=strix-hip
   METRALE_TARGET_MODEL=laguna-xs-2.1 METRALE_TARGET_QUANT=int4`), including the first
   strix-hip compile of the shared attention and prefill sources at HDIM 128.
2. `packed_int_gemv` launch parity: device bytes against the host emulation (expected
   bit-identical) and the CPU reference, INT4 and INT8, dense and grouped, padding slots.
3. The INT4 expert dispatch and loader branch (upload `weight_packed` / `weight_scale` as
   stored, pointer tables per layer, router and shared expert on the BF16 path), then
   `met serve --check-kernels` and the `[expected_absent]` list from its output.
4. Layer and logits parity against a pinned reference on fixed prompts, with the
   tolerance declared for W4A16/W8A16.
5. Serve: sequential and concurrent arithmetic oracles, tools, streaming, soak; then
   throughput, latency, memory and energy with exact build identity; then the
   `V_DOT4_I32_IU8` and multi-column optimizations against measured roofline.
