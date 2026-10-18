# CITATIONS

Prior art the engine builds on. **Rule:** when code implements a technique
from a paper, it cites the arXiv id in a comment beside the implementation
(`// arXiv:XXXX.XXXXX — what is taken`), and the same id gets a row here in
the same commit. A row marked *planned* has no code yet; the commit that lands
the code changes it to the file that implements it.

## Kernel and serving techniques

### Low-precision GEMM and MoE

| Paper | arXiv | What we take | Where |
|---|---|---|---|
| DeepSeek-V3 Technical Report — DeepSeek-AI | [2412.19437](https://arxiv.org/abs/2412.19437) | Tensor-core FP8 accumulation is not full FP32; promote partial sums to FP32 registers at a fixed K interval. We gate every new low-precision MMA on an accumulator-precision microtest. | `kernels/DEVICES.toml`; W4A8 grouped MoE (*planned*) |
| Four Over Six: More Accurate NVFP4 Quantization with Adaptive Block Scaling — Cook, Guo, Xiao et al. | [2512.02010](https://arxiv.org/abs/2512.02010) | Per-block choice between two FP4 scale targets during activation quantization. | NVFP4 activation quant (*planned*) |
| Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization — Egiazarian, Castro, Kuznedelev et al. | [2509.23202](https://arxiv.org/abs/2509.23202) | Rotation rule: Hadamard rotation helps MXFP4 (power-of-two block scales) and hurts NVFP4. Decides where rotation is allowed. | FP4 weight/activation paths (*planned*) |
| MonoMoE: An Efficient Fused Mega-kernel for Quantized MoE Decoding — Gong, Budhathoki, Kim et al. | [2609.04244](https://arxiv.org/abs/2609.04244) | Weight-major fused MoE decode. We replace its atomic reduction with a fixed-order reduction so results stay batch-invariant. | Fused MoE decode (*planned*) |
| Flash Communication: Reducing Tensor Parallelization Bottleneck for Fast Large Language Model Inference — Li, Zhang, Ye et al. | [2412.04964](https://arxiv.org/abs/2412.04964) | One-shot all-reduce shape for small tensor-parallel messages. We use the shape only, unquantized and with a fixed rank order. | Tensor-parallel all-reduce (*planned*) |

Two related pieces have no paper:
- **FP4→E4M3 exact widening.** It is derived here, not taken from a paper. An E2M1 code `n` maps to the E4M3 byte `((n & 8) << 4) | ((n & 7) << 2)`, which equals the FP4 value × 2⁻⁶, exactly for all 16 codes (subnormals included). Adding 6 to the E8M0 block exponent restores the scale. The kernel that uses it carries the derivation in its comment.
- **Block-scaled MMA (`mma.sync … kind::mxf8f6f4.block_scale`).** Reference: NVIDIA PTX ISA, warp-level matrix instructions. Element and scale formats: OCP Microscaling Formats (MX) Specification v1.0.

### Attention, sparse attention and linear attention

| Paper | arXiv | What we take | Where |
|---|---|---|---|
| Gated Delta Networks: Improving Mamba2 with Delta Rule — Yang, Kautz, Hatamizadeh | [2412.06464](https://arxiv.org/abs/2412.06464) | The gated delta-rule recurrence and its chunked form. | Gated DeltaNet kernels (*planned* rework) |
| Parallelizing Linear Transformers with the Delta Rule over Sequence Length — Yang, Wang, Zhang et al. | [2406.06484](https://arxiv.org/abs/2406.06484) | WY representation of products of Householder-like transitions, used for the chunk-parallel form. | Gated DeltaNet prefill (*planned*) |
| Tiled Flash Linear Attention: More Efficient Linear RNN and xLSTM Kernels — Beck, Pöppel, Lippe et al. | [2503.14376](https://arxiv.org/abs/2503.14376) | Two-level tiling along the sequence. Chunk states combine exactly through `S_in(r+1) = P_r · S_in(r) + S_loc(r)`, so the three spine kernels fuse and parallelize over the sequence. | Gated DeltaNet prefill (*planned*) |
| Fast and Stable Triangular Inversion for Delta-Rule Linear Transformers — Sobczyk, Gottardo, Matzoros et al. | [2605.21325](https://arxiv.org/abs/2605.21325) | Stability guard for the triangular solve: no whole-chunk Neumann series. | Gated DeltaNet prefill (*planned*) |
| Technical Report: AI-Assisted Gated DeltaNet Optimization on NVIDIA Blackwell — Shin, Jang, Maeng et al. | [2607.16831](https://arxiv.org/abs/2607.16831) | Decode-step recipe for Gated DeltaNet on Blackwell. | Gated DeltaNet decode (*planned*) |
| DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models — DeepSeek-AI | [2512.02556](https://arxiv.org/abs/2512.02556) | DeepSeek Sparse Attention: lightning indexer plus top-k token selection. | DSA indexer and sparse-attend kernels |
| LiteTopK: Exploiting the Curse of Dimensionality for a Fused Indexer-TopK Kernel in Long-Context Sparse Attention — Yin, Gao, Yin et al. | [2607.11976](https://arxiv.org/abs/2607.11976) | Indexer scoring fused with an exact threshold-based top-k in one kernel. | Sparse-attention indexer (*planned*) |
| SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training — Zhang, Wei, Zhang et al. | [2505.11594](https://arxiv.org/abs/2505.11594) | Microscaling FP4 attention on SM120-class tensor cores, with a per-block quantization layout for Q·Kᵀ and P·V. | FP4 attention (*planned*) |
| FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling — Zadouri, Hoehnerbach, Shah et al. | [2603.05451](https://arxiv.org/abs/2603.05451) | Conditional online-softmax rescaling: rescale the output only when the running max grows past a threshold. (Its polynomial exp emulation is not adopted.) | Attention decode/prefill (*planned*) |

### Speculative decoding on MoE models

| Paper | arXiv | What we take | Where |
|---|---|---|---|
| CAST: Cost-Aware Speculative Trees from One-Pass Block Drafters — Lee, Eo | [2610.00321](https://arxiv.org/abs/2610.00321) | Best-first verify-tree growth under a cost model: add a node only while its expected acceptance pays for the extra verify cost at the current batch width. | Verify-tree budget (*planned*) |
| Less Experts, Faster Decoding: Cost-Aware Speculative Decoding for Mixture-of-Experts — Xie, Liu, Huang et al. | [2607.12696](https://arxiv.org/abs/2607.12696) | The verify cost of a MoE layer grows with the union of routed experts across draft rows, not with the row count. This is the expert-union term of our cost model. | Verify-tree budget (*planned*) |
| Making Every Verified Token Count: Adaptive Verification for MoE Speculative Decoding — Pan, Tao, Wang et al. | [2605.00342](https://arxiv.org/abs/2605.00342) | Lossless truncation of the verify set when the expert union stops paying for itself. | Verify-tree budget (*planned*) |

Evaluated and **lossy**, so adoptable only behind an explicit flag that is off by default:

| Paper | arXiv | Status |
|---|---|---|
| Opportunistic Expert Activation: Batch-Aware Expert Routing for Faster Decode Without Retraining — Oncescu, Wu, Chung et al. | [2511.02237](https://arxiv.org/abs/2511.02237) | not adopted |
| MoE-Spec: Expert Budgeting for Efficient Speculative Decoding — McDanel, Li, Surineni et al. | [2602.16052](https://arxiv.org/abs/2602.16052) | `crates/server/src/moe_quality.rs` |

### Model architecture components

| Paper | arXiv | What we take | Where |
|---|---|---|---|
| Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models — Cheng, Tian, Zeng et al. | [2601.07372](https://arxiv.org/abs/2601.07372) | Engram conditional-memory lookup (hashed n-gram embedding rows). | Engram row gather (*planned*) |
| mHC: Manifold-Constrained Hyper-Connections — Xie, Wei, Cao et al. | [2512.24880](https://arxiv.org/abs/2512.24880) | Manifold-constrained hyper-connection residual streams: per-site mix, pre-reduce and post-expand around each sublayer. | `deepseek_v41` circuit (`kernels/circuits/deepseek_v41.toml`) |

### Already cited in code

| Paper | arXiv | Where |
|---|---|---|
| Top-nσ: Not All Logits Are You Need — Tang, Liu, Xu et al. | [2411.07641](https://arxiv.org/abs/2411.07641) | `crates/server/src/cli/serve_args/scheduling.rs` |
| Marconi: Prefix Caching for the Era of Hybrid LLMs — Pan, Wang, Jia et al. | [2411.19379](https://arxiv.org/abs/2411.19379) | `crates/cache/src/radix_tree/snapshot.rs` |
| Earley-Driven Dynamic Pruning for Efficient Structured Decoding — Sun, Wei, Tian et al. | [2506.01151](https://arxiv.org/abs/2506.01151) | `crates/grammar/src/earley/prune.rs` |
| WGRAMMAR: Leverage Prior Knowledge to Accelerate Structured Decoding — Wang, Liu, Ren et al. | [2507.16768](https://arxiv.org/abs/2507.16768) | `crates/grammar/src/compiler/decompose.rs` |
| From Score Distributions to Balance: Plug-and-Play Mixture-of-Experts Routing — Shahout, Cai, Du et al. | [2510.03293](https://arxiv.org/abs/2510.03293) | `crates/server/src/moe_quality.rs` |
| Scaling Embeddings Outperforms Scaling Experts in Language Models — Liu, Zhang, Wang et al. | [2601.21204](https://arxiv.org/abs/2601.21204) | `crates/model-layers/src/layers/ngram_embed.rs` |
| DFlash: Block Diffusion for Flash Speculative Decoding — Chen, Liang, Liu | [2602.06036](https://arxiv.org/abs/2602.06036) | `crates/model-arch/src/dflash_head.rs` |
| ROM: Real-time Overthinking Mitigation via Streaming Detection and Intervention — Wang, Liu, Pei et al. | [2603.22016](https://arxiv.org/abs/2603.22016) | `crates/kernels/src/lib.rs` |
| Closing the Loop on LLM Agent Output with Token-Level Runtime Control (title shortened) — Cruz | [2603.27905](https://arxiv.org/abs/2603.27905) | `crates/kernels/src/lib.rs` |
| TSCG: Deterministic Tool-Schema Compilation for Agentic LLM Deployments — Sakizli | [2605.04107](https://arxiv.org/abs/2605.04107) | `crates/server/src/tscg/mod.rs` |
| D-cut: Adaptive Verification Depth Pruning for Batched Speculative Decoding — Liu, Shen, Cen et al. | [2607.14647](https://arxiv.org/abs/2607.14647) | `crates/server/src/scheduler/mtp_dcut.rs` |
| Spec Sheets Are Not Kernels: An ISA- and Source-Level Audit of INT8 Availability on NVIDIA Blackwell Ultra — Chen | [2608.11693](https://arxiv.org/abs/2608.11693) | `kernels/DEVICES.toml` |

## TurboQuant+ KV cache

Prior art that Metrale Engine's TurboQuant+ KV-cache work (the `turbo*` KV
dtypes, [`docs/turboquant-plus.md`](docs/turboquant-plus.md)) builds on, and
what was ported from each.

### Prior art (in order it should be cited)

#### 1. Google — TurboQuant (canonical)
- **TurboQuant: Online Vector Quantization with Near-optimal Distortion
  Rate** — Amir Zandieh, Majid Daliri, Majid Hadian, Vahab Mirrokni.
  arXiv:[2504.19874](https://arxiv.org/abs/2504.19874) (April 2025, 25
  pages).
- Establishes that random rotation of input vectors induces a
  concentrated Beta distribution on coordinates, after which the same
  scalar Lloyd-Max quantizer can be applied per coordinate. For inner
  product estimation specifically, the paper proposes a two-stage
  approach: an MSE-optimal quantizer followed by a 1-bit Quantized JL
  transform on the residual to remove bias. Information-theoretic
  near-optimal distortion at all bit widths; the paper reports absolute
  quality neutrality at 3.5 bits per channel and marginal degradation
  at 2.5 bpc on KV cache quantization.
- Metrale Engine's `wht_bf16.cu` implemented plain WHT (no random sign mask) —
  strictly weaker than the canonical Randomized Hadamard form.

#### 2. `TheTom/turboquant_plus` — TurboQuant+ umbrella research repo
- Primary research dumping ground for the TQ+ work that spans multiple
  downstream inference engines (Metrale Engine, llama.cpp, vLLM, etc.).
  [https://github.com/TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus)
- ~15 papers under `docs/papers/` plus reference quant/dequant
  implementations and bench harnesses. Where the per-feature designs
  cited below (matched-norm L2, sparse V, asymmetric K/V, InnerQ,
  layer-aware V) are documented and benchmarked.

#### 3. `TheTom/llama-cpp-turboquant` — TurboQuant+ engine reference
- First public llama.cpp fork implementing a complete TurboQuant KV
  cache (`--kv-cache-dtype turbo3/turbo4/turbo8`).
  [https://github.com/TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)
- Source: `ggml/src/ggml-cuda/turbo-wht.cu`, `turbo-quant.cuh`,
  `turbo-innerq.cu`.
- The sign arrays vendored into `kernels/gb10/common/tq_plus_signs.cuh`
  are byte-identical to `TURBO_WHT_SIGNS1`/`TURBO_WHT_SIGNS2` in
  `turbo-quant.cuh` (seed=42 Rademacher draws).
- The CLI surface `turbo3 / turbo4 / turbo8` that Metrale Engine adopted matches
  this fork's prior public CLI.

#### 4. TurboQuant+ paper set — per-feature designs
Pieces relevant to this branch's port sequence, all from
`TheTom/turboquant_plus/docs/papers/`:

- `asymmetric-kv-compression.md` — K is bandwidth-critical, V can tolerate
  harder quant. Motivates `KvCacheDtype::TurboKV { k: Turbo4, v: Turbo3 }`.
- `layer-aware-v-compression.md` (LA-V7) — boundary V layers (first/last N)
  must stay higher precision; intermediate layers compress aggressively.
- `sparse-v-dequant.md` — at decode, only dequant V rows where attention
  weight > threshold (skip < threshold → 30-50% V-dequant cost savings).
- `weight-compression-tq4.md` — TQ4_1S 4-bit weight quantization with N(0,1)
  Lloyd-Max + WHT rotation per weight tile. Distinct codebook from KV cache
  (which uses N(0, 1/d) since post-WHT the per-coordinate variance is 1/d).
- `triattention-v3.md` — long-context attention eviction policy.
- `moe-v-compression-frontier.md` — V compression for MoE models specifically.

### What this branch changes vs upstream Metrale Engine (`87b7bb3`)

#### Two-sided Rademacher signs in WHT (commit `3822f8a`)

| File | Change |
|---|---|
| `kernels/gb10/common/tq_plus_signs.cuh` | new — `TQP_SIGNS1_128 / TQP_SIGNS2_128` + apply helpers |
| `kernels/gb10/common/wht_bf16.cu` | `TQ_PLUS_SIGNS`-gated: signs1 before / signs2 after butterfly for hd=128; new `wht_bf16_inplace_inv` kernel for post-attention un-rotation |
| `qwen3_attention/types.rs` | adds `wht_bf16_k_inv` handle |
| `qwen3_attention/init.rs` | loads inverse kernel |
| `qwen3_attention/decode/attention_forward.rs:478` | routes iWHT to `wht_bf16_k_inv` |

With `TQ_PLUS_SIGNS` undefined: kernels are byte-equivalent to upstream Metrale Engine
(A/B baseline). With it defined: hd=128 forward + inverse rotations carry the
canonical two-sided sign masks; attention dot product preserved since
`(S2·H·S1)·(S1·H·S2)^T = I`.

#### N(0, 1/d) codebook replacement + amax-loop removal

| File | Change |
|---|---|
| `kernels/gb10/common/reshape_and_cache_turbo.cu` | TURBO4 + TURBO3 codebooks + bounds + MAX defines swapped to N(0,1/128); turbo4/turbo3/turbo8 write paths drop the per-group amax compute and write a unit FP8/BF16 scale for byte-layout compat |
| `kernels/gb10/common/paged_decode_attn_turbo4{,_128,_512}.cu` | 16-level codebook → N(0,1/128) |
| `kernels/gb10/common/paged_decode_attn_turbo3{,_128}.cu` | 8-level codebook → N(0,1/128) |
| `kernels/gb10/common/paged_decode_attn_turbo8{,_128,_512}.cu` | No codebook change: turbo8 has no codebook. It stores one FP8 E4M3 byte per element and one BF16 scale per 16-element group (`crates/cache/src/kv_cache.rs` block sizing, `reshape_and_cache_flash_turbo8`). This row previously said "16-level codebook", which was wrong. |
| `crates/cache/src/kv_dequant.rs` | `TURBO4_LUT`, `TURBO3_LUT` → N(0,1/128) |

Behavior: post-WHT values feed the matching N(0,1/d) Lloyd-Max codebook
directly — no per-group amax loop on the write path, decoder still multiplies
by per-group scale (now always 1.0) for byte-layout backward compat. A
follow-up commit will drop the scale section entirely to reclaim ~11% KV
storage (4.5 → 4.0 bits/elem for turbo4).

#### Sparse V dequant (attention-gated row skip, both paths)

All 9 `paged_decode_attn_turbo{2,3,4,8}*.cu` kernels: wraps the per-position V
load + dequant in `if (exp_new > TQ_PLUS_SPARSE_V_THRESHOLD)` on the remainder
loop AND `if (exp_factors[b] > TQ_PLUS_SPARSE_V_THRESHOLD)` per row on the
batched (BC=4) path. Below threshold (default 1e-3) the V vector is left at
zero, saving the V-data + V-scale loads + dequant. Defined as a `#define` so
workloads can override; ref policy in
`turboquant_plus/docs/papers/sparse-v-dequant.md`.

The batched-path version requires moving the `v_vals[BC][VEC_BF16]` load to
AFTER the softmax `exp_factors[]` computation so the per-row exp magnitude
gates the load. Port of Tom's `de44bfe60` (+22% decode at 32K M5 in
`llama-cpp-turboquant`).

#### LA-V7 boundary-V protection

Verified: Metrale Engine's `--kv-high-precision-layers` already implements first-N +
last-N → BF16 in `build_layer_kv_dtypes`. Default `"auto"` (=2) covers Tom's
LA-V7 primitive. Per-layer dtype vector flows through scheduler/dispatch. No
port work needed.

#### InnerQ — per-channel Q/K equalization machinery

| File | Change |
|---|---|
| `kernels/gb10/common/tq_plus_innerq.cuh` | new — namespace + state declarations + `apply_innerq_scale[_inv]_128` + `accumulate_innerq_calibration_128` helpers |
| `kernels/gb10/common/tq_plus_innerq_apply.cu` | new — `d_innerq_scale[128]`, `d_innerq_scale_inv[128]`, sq-accum, active/calibrating flags. Host controllers: `turbo_innerq_start_calibration(target, strength)` and `turbo_innerq_finalize(group_size, strength)` |

Default state: identity scales (1.0), active=0. Currently a stand-alone
infrastructure drop — integration with `wht_bf16_inplace` requires that
kernel to take a per-tensor `scale_inv` pointer arg (Q gets `scale_inv`, K
gets `scale` after WHT, V gets nothing). Tom's reference design in
`ggml-cuda/turbo-wht.cu` templates on `direction` and passes `scale_inv`
nullable through op_params. Metrale Engine's kernel signature can grow a 4th `kind`
parameter (Q=0, K=1, V=2, output=3) once Rust callers are updated to thread
the tensor kind in. Calibration controller exported as `extern "C"` so a
Rust shim can drive it from a CLI flag like `--innerq-tokens=N`.

W_k pre-multiplication by per-output-channel `scale` at calibration finalize
(per Tom's `turbo_innerq_publish` hook) is part of the weight-pre-rotation
work tracked separately.

#### Weight pre-rotation helper (Q/K/V/O at model load)

| File | Change |
|---|---|
| `crates/model-arch/src/weight_loader/qwen35/load_layers/tq_plus_weight_rotation.rs` | new — `apply_canonical_rotation_inplace(gpu, weight, outer, n_heads, head_dim, stream)` reuses `wht_bf16_inplace` kernel with grid=outer×n_heads so every contiguous head_dim chunk of a `[outer, n_heads*head_dim]` weight matrix is rotated independently; `weight_rotation_enabled()` reads `TQ_PLUS_WEIGHT_ROTATION=1` env var. |
| `crates/model-arch/src/weight_loader/qwen35/load_layers.rs` | adds `mod tq_plus_weight_rotation;` |

Integration site (not yet wired): in `attention_arms::load_bf16_then_nvfp4`,
between `shard_dense_bf16` and `quantize_to_nvfp4`, call
`apply_canonical_rotation_inplace` on the bf16 sharded weight for each of
Q/K/V/O. After all 4 are rotated and quantized, the runtime
`wht_bf16_inplace` calls in `write_kv_cache.rs` and
`attention_forward.rs:398/478` become no-ops (or should be skipped via the
same env var). Per-token savings: 4 kernel launches × 40 layers = 160
launches per token.

FP8-native weight path: a future pass needs to dequant→rotate→requant; not
covered by this helper which operates on bf16.

#### Asymmetric K/V dtype scaffold

| File | Change |
|---|---|
| `crates/cache/src/kv_cache.rs` | adds 3 asym variants: `Turbo4KTurbo3V`, `Turbo4KTurbo8V`, `Turbo3KTurbo8V` to `KvCacheDtype`. New `kv_pair()` returns the (K, V) symmetric pair; `is_asymmetric()` predicate. Display + FromStr cover all three (parse names `turbo4k_turbo3v`, `turbo4k_turbo8v`, `turbo3k_turbo8v` plus aliases `turbo4k3v` etc.) |
| 9 files in `crates/model-layers/...` + `crates/server/...` | exhaustive `match` arms extended to route asym variants to their K-side existing kernels (compiles green). 16 match arms covered programmatically; remaining 1 site (`paged_attn_batched.rs`) absorbs via existing `(dtype, _) =>` catch-all. |

Stub-routing behavior: each asym variant currently dispatches the SAME write
and decode kernels as its K-side symmetric counterpart, so storage-byte and
runtime behavior match the K-side dtype only — V side asym savings are NOT
realized until proper asym kernels exist. Tom's
`turboquant_plus/docs/papers/asymmetric-kv-compression.md` predicts ~14%
bandwidth saving at decode for `Turbo4KTurbo3V` once the V-side gets its
own 3-bit decode path.

Remaining work for true asymmetric:
- New write kernels per combo (`reshape_and_cache_flash_turbo4k_turbo3v` etc) that pack K with one layout and V with another in a single launch
- New decode kernels (`paged_decode_attn_turbo4k_turbo3v`) that mix K-side and V-side dequant logic
- Storage byte accounting for asym layouts in `kv_cache.rs` block-stride helpers

### Trellis-coded KV cache (TCQ)

The 128-group rotated codecs `turbo3_tcq`, `turbo2_tcq`, `turbo1_tcq` and
`turbo4_n128` ([`docs/kv-cache-codecs.md`](docs/kv-cache-codecs.md)) are
derived from `spiritbuun/buun-llama-cpp` (MIT, pinned at `34ef0ef7c6ea`;
licence in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) §7). Its
codebooks are copied verbatim and credit spiritbuun, as upstream asks. The
design is described in "Closing the Gap: Trellis-Coded Quantization for KV
Cache at 2-3 Bits" (spiritbuun, 2026; Hugging Face dataset
`spiritbuun/turboquant-tcq-kv-cache`, not on arXiv).

| Paper | arXiv | What we take | Where |
|---|---|---|---|
| QTIP: Quantization with Trellises and Incoherence Processing — Tseng, Sun, Hou, De Sa | [2406.11235](https://arxiv.org/abs/2406.11235) | The bitshift trellis (a state is the last L bits of the stream, so any element decodes from two bytes) and its Viterbi encoder. | `crates/kv-tq128/src/trellis.rs`, `kernels/gb10/common/reshape_and_cache_tcq.cu` |
| Optimal Quantization for Matrix Multiplication — Ordentlich, Polyanskiy | [2410.13780](https://arxiv.org/abs/2410.13780) | The product-aware distortion that upstream trained the TCQ codebooks against. We copy the trained books and do not retrain. | `crates/kv-tq128/src/codebooks/` |
| TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate — Zandieh, Daliri, Hadian, Mirrokni | [2504.19874](https://arxiv.org/abs/2504.19874) | Normalize, then randomized Hadamard rotation, then quantize the coordinates against a fixed codebook, with a norm-preserving stored scale. This covers all four codecs. `turbo4_n128` is the plain scalar form: upstream's 16-level codebook with one fp16 norm per 128 values. | `crates/kv-tq128/src/rotation.rs`, `crates/kv-tq128/src/group.rs`, `kernels/gb10/common/tq128_common.cuh`, `kernels/gb10/common/reshape_and_cache_turbo4_n128.cu`, `kernels/gb10/common/paged_decode_attn_tq128.cu` |

### Outstanding (not yet ported)

- 256/512 sign arrays (Rademacher seed=42, derived from same random draws)
- Block-layout refactor to drop the per-group scale section (~11% KV bandwidth saving)
- Pre-rotation of Q/K/V/O projection weights at model load (folds runtime WHT into projection gemm — saves 4 kernel launches × 40 layers per token)
- Asymmetric K/V dtype (`Turbo4K + Turbo3V` etc) — Rust enum + CLI + new asym decode kernels
- InnerQ per-channel scale_inv (Q/K equalization with calibration phase)

### License note

Metrale Engine is MIT OR Apache-2.0. The TurboQuant+ work cites the prior-art
chain (1) → (2) → (3) above; the licence of the code it draws on is recorded in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) §7.
