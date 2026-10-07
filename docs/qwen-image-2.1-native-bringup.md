# Qwen Image 2.1 integration

This is the cumulative integration plan for `Qwen/Qwen-Image-2.1`.
Add implementation, regressions and qualification evidence to this model branch
and draft PR. Status: checkpoint and verified backup complete; two external-reference images
generated and visually inspected. Native generation remains unimplemented. Image API routes currently return unsupported responses.

## Inputs and use boundary

- [Official model repository](https://github.com/QwenLM/Qwen-Image-2.1).
- [Checkpoint and component descriptions](https://huggingface.co/Qwen/Qwen-Image-2.1).
- [Model license](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE).

The reviewed license permits research/evaluation and requires separate permission
for commercial use. Verify the license at the chosen immutable revision before
retrieval or distribution. Commercial product availability is blocked pending
appropriate permission. Do not treat the engine source license as the weight license.

## Implementation and evidence gates

- [x] Pin the full checkpoint revision, tokenizer/text encoder, VAE, scheduler and
      image processing components in the [component manifest](model-manifests/qwen-image-2.1.json).
      Local active files and the complete backup have now passed SHA-256 checks.
- [x] SHA-256 verify a complete TrueNAS backup and BACKUP-MANIFEST.json after
      download, retaining local active components. A transformer-only backup is incomplete.
- [x] Pin and execute the official reference pipeline to establish expected image
      behavior and resource needs. Label reference results explicitly.
- [x] Describe native component boundaries and the transformer block as
      [residual architecture data](../kernels/circuits/residuals/qwen_image21.json).
      Candidate primitive reuse is recorded; executable circuit lowering remains open.
- [ ] Implement native loading and generation with intermediate numerical parity;
      keep reference-backend and native-engine qualification separate.
- [ ] Implement image generation/editing API validation, bounded job queues,
      cancellation, errors and result retrieval. Never return placeholder success.
- [ ] Validate text rendering, reference-image editing, RGBA and advertised size
      limits with saved prompts, seeds, components, outputs and human image review.
- [ ] Exercise invalid dimensions, missing components, overload, cancellation,
      disconnected clients and memory pressure with known-bad controls.
- [ ] Expose per-image latency, peak memory, generation settings and measured
      cost per image; do not reuse text-token economics for image jobs.
- [ ] Complete native stability and required certification before marking support.

Start with one device and explicit memory limits. Do not interrupt the GPT-OSS or
Laguna campaigns to stage this model. Multi-device operation and product UI
availability require separate measured evidence. This PR remains draft while
these gates are open; the reference observations below are not native benchmarks.

## Pinned component inventory (2026-10-06)

The manifest pins checkpoint `d26bb61231c349cf6b7896fa83353113880e1ba3`
and every serving file, plus the README and license. Selected download size is
33,131,615,131 bytes (30.86 GiB); weights account for 33,115,613,408 bytes.
This is disk size, not a measured GPU memory requirement. The repository's QR
asset and Git attributes are excluded from that serving inventory. The complete
backup includes both: 28 files, 33,134,949,561 bytes, with a verified manifest.

| Component | Pinned identity | Download bytes |
| --- | --- | ---: |
| Joint text/image encoder | Qwen3VLForConditionalGeneration | 17,534,409,013 |
| Visual transformer | QwenImage21Transformer2DModel | 14,230,315,061 |
| RGBA autoencoder | AutoencoderKLQwenImage21 | 1,350,991,591 |
| Tokenizer and preprocessing | Qwen3VLProcessor | 15,884,996 |
| Flow scheduler | FlowMatchEulerDiscreteScheduler | 485 |

The visual transformer has 32 layers, 32 attention heads of width 128 and
64 input/output channels. Its 7B description excludes the joint encoder and VAE.
The encoder includes 36 text layers (hidden width 4096, 32 query/8 KV heads)
and a 27-layer vision tower. The VAE has four input/output channels and 64 latent
channels. Preserve the checkpoint's latent normalization arrays and scheduler
configuration; a generic RGB VAE or text-only encoder is not interchangeable.

The [pinned license](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/d26bb61231c349cf6b7896fa83353113880e1ba3/LICENSE)
allows research/evaluation only without a separate commercial license. Permission
for commercial use has not been verified. Preserve the agreement and required
Notice when distributing materials. This does not block inspecting metadata or
planning research qualification; it does block claiming commercial readiness.

## Retrieval and reference verification

1. Retrieve only manifest paths from the immutable checkpoint revision. Reject
   missing files, unexpected component substitutions, wrong sizes or SHA-256
   mismatches. Check both safetensors indexes: every referenced shard must exist,
   and every indexed tensor must be found in its assigned shard header. All local file hashes were verified before reference loading; adversarial
   shard-index integrity controls remain to be implemented.
2. Copy the complete selected set to TrueNAS and independently hash destination
   bytes. Write `BACKUP-MANIFEST.json` only after all checks pass, with repository,
   revision, each file's size/hash and completion time. Retain active local weights.
   Reserve space for the entire checkpoint and temporary download files on both
   destinations; a transformer-only archive cannot restore this pipeline.
3. Install the manifest's exact Diffusers and Transformers source revisions in
   an isolated reference environment, then freeze the resolved dependency lock.
   These source pins ran successfully in the isolated environment described below;
   broader reference qualification remains pending. The model index's `0.37.0.dev0` is provenance, not a
   reproducible dependency pin. Record Torch/CUDA, driver and binary identities.
4. The pinned official pipeline performs joint text/image conditioning and
   optional prefix KV reuse; it samples without classifier-free guidance by
   default. Capture actual settings explicitly. First compare cache on/off at a
   fixed seed, prompt, resolution, dtype and step count, saving intermediate
   embeddings, denoising latents and decoded output for later native parity.
5. Keep first-run load time separate from warm generation. Measure complete
   request latency and encoder, denoising and VAE stages; report images/minute,
   peak memory and joules/image at fixed quality settings. Start at one request
   and qualify bounded concurrent queues before increasing GPU parallelism.
   A reference result is not a native Metrale pass.

Integrity controls must include a missing encoder shard, a corrupted VAE byte,
an index pointing to the wrong shard and a mismatched processor config, each
rejected alongside an unchanged valid checkpoint. Numerical controls need a
deliberately wrong latent scale or scheduler shift to demonstrate that comparison
detects errors. Image review covers RGB, actual alpha-channel transparency,
reference-image editing, typography and repeatability; merely writing a PNG does
not establish correctness. Set numerical tolerances before evaluating results,
and record any precision-dependent differences rather than relaxing them to pass.

## Reference observations (2026-10-06)

Single NVIDIA GB10, Torch 2.13.0+cu130, CUDA 13.0, BF16, memory fraction 0.85.
Diffusers `c6df88a511a98740646ee55577b590c9852650ce` and Transformers
`14e738b5d0cc69aa27a95dde272aea41fde44f2f`; offline local checkpoint loading.
Initial pipeline load took 210.11 seconds, excluded from generation below.
Both cases used 40 steps and guidance scale 1.0. These are individual samples,
not repeated throughput or native-engine measurements.

| Case | Seed | Size | Generation seconds | Peak allocated / reserved bytes | Visual inspection |
| --- | ---: | --- | ---: | --- | --- |
| Red cube left, blue sphere right | 426 | 512×512 | 12.77 | 34,276,423,168 / 34,661,728,256 | Correct colors and positions; red object has polygonal/beveled shape rather than a strict cube. Prompt fidelity incomplete. |
| METRALE / INFERENCE GPU poster | 427 | 1024×1024 | 53.47 | 39,566,055,424 / 41,204,842,496 | Both words legible and correctly spelled; green chip centered. |

Output SHA-256: geometry `5c037c4d2c7ada8a73e29f8178802c539152a976ed1ed98d77ca1315d83cb141`;
poster `44e568efbdfc0368351ee7d3e73513097cc6d0e20a6e4ce3d82447b8e2994b04`.
RGBA file mode alone does not prove useful alpha transparency. Editing, alpha,
repeatability, size limits, component parity, native API and native performance
remain unqualified. The geometry miss is retained as a quality counterexample.

## Native architecture boundary (2026-10-07)

The residual graph is deliberately outside `INSTANCES.toml`: it is not a golden
instance or an executable model registration. Its transformer block explicitly
names every unlowered operation. Tests verify topological closure and require the
closed circuit parser to refuse those residual names. Existing `qk_norm`,
`silu_mul` and `residual_add` operations are semantic reuse candidates; projection
bindings, dtype behavior and image-row performance still need parity evidence.
No measured kernel coverage or completed Venn comparison is claimed. The present
decoder IR cannot faithfully represent this pipeline, so encoding it as ordinary
paged causal attention would hide the architecture residual.

The native pure visibility contract in
`crates/circuit/src/image_attention.rs` implements the pinned reference predicate:
same sample, valid key, and either a causal position or the same image block.
It validates sample/image contiguity and answers a pair in constant time after
layout construction without allocating a quadratic mask. A hand-written mask
control catches a causal-only substitution, future-text leakage, cross-sample
leakage and confusing padded queries with padded keys. This is a CPU semantic
primitive, not a GPU kernel or a serving path.

Remaining lowering must preserve these distinctions:

- **Conditioning:** joint Qwen3-VL text/vision embeddings, image-slot placement,
  zero-centered text RMSNorm, GELU-tanh projection and a sinusoidal timestep
  embedding. Ordinary text LM logits are not the conditioning output.
- **Denoiser:** 32 single-stream blocks with non-affine LayerNorm, shared scale
  and tanh-gate modulation, Q/K RMSNorm, centered three-axis RoPE and image-block
  causal attention. Prefix tokens use timestep zero; target tokens use the actual
  timestep. The final adaptive LayerNorm applies scale without a shift.
- **State:** prefix KV becomes immutable across denoising steps; each request owns
  its cache and invalidates it when conditioning changes. This differs from token
  decode's growing paged cache and needs explicit lifecycle declarations.
- **Image reconstruction:** four-channel VAE with residual spatial/temporal
  convolution and upsampling, exact latent means/stds and Euler flow scheduling.
  These need new circuit vocabulary/lowering before a complete native plan exists.

Reference semantics are from the manifest-pinned Diffusers transformer, pipeline
and VAE source files, with their upstream file hashes.

## Visual transformer config and bindings (2026-10-07)

`crates/model-weights/src/qwen_image21` now validates the exact pinned visual
transformer config and binds all 297 tensors: nine global weights and nine per
block across 32 blocks. Typed bindings retain borrowed storage, including separate
FFN gate/up projections; they do not copy or repack weights. A component-local
`WeightStore` adapter is available. Missing or unknown config fields, unsupported
math settings, missing/extra tensor names, wrong shapes and non-BF16 weights fail
before any binding is returned. All normative behavior remains unregistered.

The fixture under `crates/model-weights/tests/fixtures/qwen-image21` was captured
from the downloaded checkpoint's two safetensors headers using CPU reads only.
Its provenance records each source shard's size, header SHA-256, declared upstream
file SHA-256 and fixture hashes. All 297 names and shard assignments matched the
pinned upstream index. The norm weights are BF16 storage; this does not permit
lower-precision accumulation where the reference performs normalization in FP32.
Tests use actual header metadata with missing, extra, transposed-shape and dtype
mutations, and verify returned storage pointer identity. They do not allocate or
claim execution of checkpoint tensors on a GPU.

Next connect these validated bindings to native visual-transformer operators and
compare intermediate values at fixed inputs. Encoder/VAE bindings, scheduler,
state lifecycle, numerical parity, generation and serving registration remain
open; this component binder alone does not establish native image support.

## Native modulation primitive (2026-10-07)

`kernels/gb10/common/image_modulation.cu` implements scale-only modulation and
`tanh`-gated residual updates. The typed layout in `model-layers` builds the
sample-major row map: target tokens select their sample's timestep row, while
prefix tokens select the trailing timestep-zero row. Components use explicit
strides/offsets so the same operators cover both block halves and the final
scale-only norm epilogue. Normalization and projection are separate operations.

The implementation preserves the pinned eager BF16 boundaries: round `1+scale`
before multiplying normalized input; round `tanh(gate)` and its product with the
branch before adding the residual. Ordinary residual-add and sigmoid/SiLU kernels
do not realize this selection/precision policy. This is a named residual family
point, not a promoted circuit lowering rule, registered model or optimized kernel.
No existing kernel point was changed. LAB/LKB coverage remains unqualified; only
the isolated component now has numerical evidence.

The bounded `scripts/qwen_image21/modulation_parity.py` loads the exact manifest-
pinned Diffusers source (SHA-256 checked) and calls its `_modulate` method, followed
by its eager gated residual expression. On GB10, all 360,452 output elements across
six cases matched exact BF16 bits: width 4096 with two samples and mixed prefix/
target tokens, odd tail width 257, and all 65,280 finite BF16 gate encodings at
width 65,280. Both block halves were tested. These are synthetic component inputs,
not checkpoint execution or an image-quality result. The separate normalization
used to prepare inputs is reference code, not a native normalization claim.

Known-bad controls detected 31,748 differing values when scale rounding was
omitted, 25,401 when product rounding was omitted, 32,690 for wrong prefix row
selection, and 161,395 for substituting sigmoid gating. Rust tests cover the
row-selection contract, invalid dimensions/masks/components, null pointers and
exact launch geometry/arguments. No throughput claim is made. Native LayerNorm,
conditioning, attention, projections, complete denoising and image generation
remain open, as do the existing license and full-model qualification gates.

CHKI reports the new common source reaches GB10, B200 and Hopper. The change is
benign for existing dispatch: it adds two uniquely named, unregistered entry
points and modifies no existing kernel or target lookup. It uses ordinary BF16
conversion and CUDA elementwise arithmetic with no GB10-specific instruction.
Numerical evidence covers GB10 only; B200/Hopper execution remains unmeasured.

### LayerNorm reuse refusal

The unchanged `nllb_layernorm_oop_bf16` kernel was evaluated with explicit unit
scale/zero bias, width 4096 and epsilon `1e-6`. Random, constant, large-offset and
small-variance cases matched exactly. The tiny-input case differed at one value
and the large-magnitude case at two: three of 1,597,440 outputs differed by one
BF16 ULP, with maximum absolute error `4.76837158203125e-7`. The exact-bit gate
failed and the candidate was **not adopted**. The saved rejection receipt and
`scripts/qwen_image21/norm_reuse_parity.py` preserve this result. Wrong RMSNorm and
wrong-epsilon controls detected 1,508,329 and 532,685 differences respectively.

The pinned Torch build's LayerNorm uses a vectorized Welford reduction; the
existing native NLLB kernel uses a two-pass sum/variance reduction. Any new
precision policy must retain the old kernel's behavior and be separately tested.
This is a lowering gap, not permission to widen the criterion after seeing data.

### Diagnostic native block prelude

`model-arch::qwen_image21::DiagnosticImagePrelude` now composes the unchanged
LayerNorm candidate, exact native scale modulation and existing tensor-core BF16
GEMMs for raw Q/K/V projections. It borrows projection weights, owns/reclaims one
scratch allocation and immutable uploaded row selection, and refuses invalid
weights, missing kernels or null input pointers. This separate diagnostic type
is not a `TransformerLayer` implementation or model-factory entry.

A bounded CUDA test used the real pinned block-0 projection weights with fixed
BF16 inputs (seed 2123, two samples × three tokens, width 4096). All 73,728 raw
Q/K/V output values matched the reference exactly, including this case's norm
and modulation intermediates. Transposed-weight controls detected differences.
The receipt records weight-content and native-source hashes; raw output pairs
were retained. The known three LayerNorm adversarial mismatches remain unchanged,
so this finite test does not qualify the general norm policy or complete block.
Rust tests verify the five-launch composition, storage lifetime and refusal paths.

The next boundaries are Q/K normalization, three-axis RoPE and image-block
attention. The pinned Diffusers `RMSNorm` rounds normalized Q/K activation to
BF16 **before** multiplying its BF16 weight. Existing `rms_norm_vanilla` multiplies
the weight in FP32 before the final cast and cannot be silently substituted.
The additional reference normalization source/hash is now included in the
component manifest. Encoder/VAE/scheduler and image serving remain unimplemented.

### Staged Q/K normalization and three-axis rotation

The diagnostic prelude now continues through per-head Q/K normalization. It
reuses unchanged `rms_norm_vanilla` with explicit unit weights, then separately
multiplies the rounded BF16 row by its real BF16 norm weights. The saved real
Q/K projections and random, tiny, large, constant and small-variance inputs
matched exactly for 296,192 BF16 outputs. Fusing the weight before the rounding
boundary caused 53,006 differences; the wrong epsilon caused 49,429. No existing
RMSNorm behavior was modified.

`ImageRopeLayout` implements the pinned frame/height/width geometry and rejects
unsupported multi-frame shapes, noncontiguous blocks, unaccounted image tokens
and positions outside the pinned table range. Odd grids retain the reference's
asymmetric centred coordinates. Text advances all axes; an image holds the frame
position and advances subsequent text by its maximum spatial dimension. The
native table export example runs the production Rust geometry/math for replay.

The native complex-pair CUDA transform follows the `[16,56,56]` axis partition
and shared sequence frequencies. Initial split multiply/add failed one tiny
near-cancellation value; the failure persisted with identical reference
frequencies, isolating arithmetic contraction. Explicit FMA follows the pinned
one CUDA contraction candidate and passed all 344,064 BF16 outputs of
that corpus. The rejected receipt is retained. Conjugated-frequency and split-
half pairing controls detect 173,070 and 222,430 differences respectively.
Native Rust tables still differ from the reference at 4/14 FP32 frequency values
in those two layouts (maximum absolute error `1.4901161193847656e-8`); this did not
change the final BF16 results in the passing cases. This is bounded evidence,
not a claim of bit-identical tables across platforms or all positions.

The prelude rejects Q/K normalization before projection, repeated normalization,
rotation before normalization and a rotary sequence of the wrong length. Four
Rust tests cover composition, ownership, geometry and refusal paths. The earlier
LayerNorm discrepancy remains open. Full image-block attention, complete block
residual/FFN execution and native generation are still unqualified.

A fresh seed (2126) plus a 266-token layout **failed** that exact gate: 132 of
6,881,280 BF16 outputs differed; using identical reference cis bits still left
63 differences. The longer layout has 752 FP32 frequency differences (maximum
`4.76837158203125e-7`). Thus both complex contraction and native frequency
generation remain diagnostic candidates. The independent failure receipt is
retained alongside the earlier bounded pass; no tolerance was widened.

An identical-operand contraction probe (fresh seed 2127; 266 tokens; random,
tiny and large inputs) evaluated all four fused-product choices and split
operations. The corrected imaginary component `fma(y, cos, x*sin)`, paired
with real `fma(x, cos, -y*sin)`, matched all 6,537,216 BF16 outputs; the previous
imaginary order differed at 47 values. Inputs, cis bits, reference and every
variant output were saved. The explicit order was corrected. This isolates
arithmetic only: the native frequency-table gate remains open, and the earlier
independent rejection is preserved.

Reuse accounting: LAB coverage and model registration are unchanged. Q/K norm
reuses the existing RMSNorm family plus one named BF16 storage-boundary residual;
complex image RoPE is a second unregistered residual. Kernel inventory now
contains 1,398 entry points. CHKI reaches GB10, B200 and Hopper; the change is
additive, leaves existing dispatch unchanged and is measured only on GB10.
Scoped integration tests, Clippy, rustdoc, formatting and inventory checks pass.
Broad host `--tests` Clippy remains blocked by existing CUDA-gated tests and
macOS dependency-stub lint errors; it is not reported as passing.

Final corrected production-kernel replay, fresh seed 2128: all 6,881,280 BF16
outputs match when given the identical reference cis table. Native Rust cis
tables still produce 55 differences in the 266-token layout (24 random, 11
tiny, 20 large), so the overall exact rotary gate remains **failed**. The saved
real Q/K and shorter mixed-image cases are exact. Every input, reference/native
cis table, source snapshot and output pair is retained for this replay. This
separates the remaining frequency-generation issue from complex multiplication.

### Diagnostic image-block attention composition

The prelude now continues through native block-causal attention and the real
block-0 output projection. Shared circuit visibility lowers to stable valid-key
gather indices and one visible prefix per query, in linear host storage/time.
The unchanged `batched_embed` kernel gathers BF16 K/V rows; unchanged
`nllb_attn_kv_bf16` attends each query to its packed sample-local prefix. Image
blocks include future keys within the same block; text stays causal. Padding
removes keys, not queries; fully masked query rows remain exactly zero.

This implementation deliberately launches once per query and is a **slow
diagnostic**, not a production image attention path. It allocates bounded linear
scratch, refuses malformed metadata, wrong geometry/backend and stage order,
and reuses the existing output GEMM without modifying any shared CUDA source.

The exact SDPA gate **failed**: saved real-projection inputs differ at
2,450/24,576 BF16 outputs; constructed random/high-logit/uniform cases differ at
11,131/172,032. The same-input output GEMM is exact for all 196,608 outputs;
composed output differences therefore inherit the attention discrepancy. All
raw inputs, source snapshots and output pairs are retained. No threshold was
widened and no full-block or generation qualification is inferred.

Visibility controls remain useful independently of arithmetic precision:
changing padded values or another sample's values changes zero protected outputs;
fully masked rows have zero nonzero values. Incorrect causal-only, dense and
query-padding masks differ at 19,888, 138,907 and 36,861 values respectively.
The output-weight transpose control differs at 184,170 values. The shared CPU
planner also matches the independent visibility oracle across 128 validity
patterns and two samples. Six prelude tests and four circuit tests pass.

The earlier LayerNorm and CPU cis mismatches remain open. Remaining block work
includes gated residuals, the second norm/modulation and FFN composition, then
full-transformer/encoder/VAE execution. LAB model registration remains unchanged;
this increment adds no CUDA entry points and reuses existing kernel families.

### Complete native diagnostic block and visual stack

`DiagnosticImageBlock` composes the actual block operations: the existing prelude
and output projection, first tanh-gated residual, second non-affine norm/scale,
real gate/up/down FFN projections, staged BF16 SiLU product, and second gated
residual. The new SiLU entry rounds the eager activation before multiplication;
existing fused MoE policies are unchanged. Construction validates all borrowed
weights, and the full block rejects rotary/image visibility disagreement before
launch. Fixed scratch is owned/reclaimed; output may feed the next block.

Actual block-0 weights were replayed with seeds 2130 and 2131. All same-input
stages were exact except attention (1,972 and 1,771 differences respectively).
Complete-block comparison to the pinned Diffusers block remains **failed**:
9,475/24,576 differences and relative L2 `0.0030184336937963963` for seed 2130;
9,916/24,576 and `0.0031924771509037842` for independent seed 2131. This is retained
error characterization, not an exact pass or a new acceptance threshold.

Staged SiLU matches 73,728 actual projected values and all 65,280 finite BF16 gate
encodings in each run. The independent run includes 24 matching infinite output
pairs from finite-input overflow, explicitly counted separately from finite error
metrics (zero nonfinite bit differences). Omitting the activation rounding causes
19,341 differences in the independent actual-projection case. An early summary
metric produced NaN for matching infinities; the retained initial evidence is
unchanged, and the fresh run separates nonfinite classifications without changing
outputs or the exact gate.

The standalone `qwen_image21_block_native` Rust example now runs the actual
`DiagnosticImageBlock` on CUDA, with strict safetensors dtype/shape/offset checks,
PTX SHA verification and an 85% device-memory guard. Its 24,576 block-0 output
values are bit-identical to the separate native ctypes composition (SHA-256
`ca81695885c36db3e2dd788791f3078fc5a05d04856023f542a9292eba8b9a5c`). A 32-block
native visual stack also completed with finite outputs at every layer and
per-layer tensor/output hashes. This is a fixed two-sample × three-token fixture:
~2.95ms block-0 and 94.12ms summed block forwards exclude weight loading/hashing
and are **not** image-generation or production-speed claims.

Independent CPU FP64 same-input attention characterization gives 66/196,608
native differences from rounded FP64 versus 13,520 for pinned SDPA. Native counts
by real/random/high-logit/uniform case are 1/4/1/60; SDPA counts are
2,450/10,844/226/0. Uniform values expose reduction/tie-rounding differences; the
native result is not uniformly more accurate. The original exact SDPA gate stays
failed. Any future accuracy admission must use a separately predeclared numerical
contract and held-out controls, not widen the retained gate after observing it.

The visual stack still excludes input/time/text projections, final adaptive norm,
encoder, VAE and denoising/scheduling. Neither this diagnostic nor the external
reference images establish native image support. Kernel/LAB registration remains
unchanged; the added staged-SiLU entry is an unregistered residual. Existing
LayerNorm, CPU cis and attention precision gaps remain documented.

The pinned 32-block reference replay completed with verified matching input and
weight hashes. All outputs are finite, but the exact stack gate remains failed:
final layer 31 differs at 18,062/24,576 values, relative L2
`0.01287850703204185` (maximum cumulative relative L2 over the stack
`0.012908305023796477`). Replaying that final reference block on the identical
native layer-30 input yields only 247 differences and relative L2
`0.00020246686431468674`, separating local error from accumulated drift. Full
per-layer and same-input receipts are retained; no admission threshold changed.
