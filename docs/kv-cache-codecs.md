# 128-group rotated KV cache codecs

This page covers four KV cache formats that share one structure: `turbo3_tcq`, `turbo2_tcq`,
`turbo1_tcq` and `turbo4_n128`. Each cuts a head row into groups of 128 elements. A group
gets one fp16 norm and a randomized Hadamard rotation, and its rotated coordinates are
coded against a fixed codebook.

The three `_tcq` codecs code the coordinates along a bitshift trellis (trellis-coded
quantization). `turbo4_n128` codes each coordinate on its own. The format, the codebooks and
the V decode scales come from
[spiritbuun/buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp) (MIT, pinned at
`34ef0ef7c6ea`; credit spiritbuun). The design is written up in "Closing the Gap:
Trellis-Coded Quantization for KV Cache at 2-3 Bits" (spiritbuun, 2026). Citations are in
[`CITATIONS.md`](../CITATIONS.md) under "Trellis-coded KV cache (TCQ)".

The engine's older `turbo2/3/4/8` dtypes are a different family. They use 16-element groups,
amax-selected codes and a per-16 FP8 or BF16 scale, as described in
[`turboquant-plus.md`](turboquant-plus.md). `turbo4_n128` is a separate format and does not
change `turbo4`.

The bit-exact reference is the pure Rust crate `crates/kv-tq128` (`metrale-kv-tq128`,
`no_std`). The GPU kernels are tested against it byte for byte.

## Block formats

One block encodes one 128-element group. The first two bytes are the corrected norm as IEEE
fp16, little-endian. The index payload follows, zero-padded to an even block length.

| codec | index per element | payload | block | bits per value |
|---|---|---|---|---|
| `turbo3_tcq` | 9-bit trellis state (3 new bits per step) | 49 B | 52 B | 3.25 |
| `turbo2_tcq` | 8-bit trellis state (2 new bits per step) | 33 B | 36 B | 2.25 |
| `turbo1_tcq` | 8-bit trellis state (1 new bit per step) | 17 B | 20 B | 1.25 |
| `turbo4_n128` | 4-bit level, two per byte, even element low | 64 B | 66 B | 4.125 |

A head of `head_dim` elements (128, 256 or 512) is stored as `head_dim / 128` consecutive
blocks. Groups are independent: any group of any token decodes from its own block alone.

## Encode

For each group `x` of 128 values:

1. `norm = sqrt(sum x_i^2)`.
2. `x = x * (1 / norm)`. When `norm <= 1e-10`, the factor is 0 instead.
3. Rotate: `x * signs1`, then the 128-point Walsh-Hadamard transform, then
   `* (1/sqrt(128)) * signs2`.
4. Quantize the rotated coordinates against the side's codebook:
   - a minimum-squared-error Viterbi path for a `_tcq` codec, as described in the next section;
   - for `turbo4_n128`, the nearest of 16 levels. A value equal to a decision midpoint takes the
     upper level.
5. `recon = |codebook[index]|` over the group. Store `norm / recon` as fp16, or `norm` itself
   when `recon <= 1e-10`. A decoded group therefore keeps the input's norm (within fp16
   rounding) rather than shrinking toward the codebook's centre.

## Decode

Decode multiplies each element's codebook value by the stored norm and the side's decode
scale. This is `y_t = codebook[index_t] * (fp16(norm) * decode_scale)`, with the product in
parentheses formed first. The result is in the rotated domain.

Attention never needs to un-rotate K. Rotation is orthogonal, so `q · k = R q · R k`. The
decode kernel rotates the query once per group, takes the scores against the rotated keys,
and accumulates the rotated values. It un-rotates only the final output row.

Chunked prefill over a cached prefix works the same way: its kernel rotates each query
group, reads the rotated cache, and un-rotates the output row. Dequantization to bf16 runs
the inverse rotation, which is the same three steps with the sign tables swapped.

## The trellis

A `_tcq` codec stores one state per element, but most of each state is shared with its
neighbours:

- A state is the last `L` bits of a bitstream.
- Each step shifts `k` new bits in at the top: `next = (prev >> k) | (symbol << (L - k))`.
- The value a state reconstructs is `codebook[state]`, so the codebook has `2^L` entries.

| codec | k | L | states |
|---|---|---|---|
| `turbo3_tcq` | 3 | 9 | 512 |
| `turbo2_tcq` | 2 | 8 | 256 |
| `turbo1_tcq` | 1 | 8 | 256 |

The stream (LSB first) starts with the low `L - k` bits of the first state. Each state's top
`k` bits (its symbol) follow in order. State `t` is then the `L`-bit window at bit `k * t`,
which is `(qs[b] | qs[b + 1] << 8) >> (k t mod 8)` with `b = k t / 8`. That is a two-byte
read, so any element decodes without touching the others (QTIP, arXiv:2406.11235).

Encoding picks the state sequence of minimum total squared error with a Viterbi search over
all `2^L` states and 128 steps. The initial state is free. The tie rules are fixed so that
the reference and the kernel agree on every input:

- A state's predecessors are `((s & (2^(L-k) - 1)) << k) | p`. The scan over `p` keeps the
  first minimum.
- The path ends in the lowest-numbered state among the minimum costs.
- Costs accumulate as `best + d * d` with `d = x - codebook[s]`, with no fused multiply-add.

The codebooks were trained by upstream against a product-aware distortion (Ordentlich and
Polyanskiy, arXiv:2410.13780) and are copied verbatim. The tests verify Viterbi optimality by
exhaustive search on short sequences.

## K and V are coded differently

- The `_tcq` codecs carry a separate codebook per side. K's book serves the attention
  logits and V's book serves the weighted sum.
- V decodes with a factor slightly above 1:
  - 1.26 for `turbo1_tcq`
  - 1.06 for `turbo2_tcq`
  - 1.02 for `turbo3_tcq`

  The trellis search shrinks reconstructions toward the codebook's dense centre, and these
  factors undo that shrinkage. Upstream tuned them on output KL divergence; they are
  `TURBO_TCQ_ALPHA_V_T{1,2,3}`.
- K decodes at 1.0, since K errors enter softmax through a sum where a uniform scale would only
  sharpen or flatten the distribution.
- `turbo4_n128` uses one book and a decode scale of 1.0 for both sides, as upstream's scalar
  turbo4 does.
- The encode side stores the norm with factor 1.0 everywhere, matching upstream's default
  deployment. Upstream's optional encode-time V factor is not ported.

The encode and dequant kernels run once per side and take that side's codebook as an
argument, so an asymmetric pair (for example `turbo3_tcq` K with `turbo2_tcq` V) needs no new
encoder. The decode attention kernel is a template over the (K codec, V codec) pair and takes
separate K and V block strides, so such a pair needs one more instantiation of it, not new
code. Only the four symmetric pairs are instantiated today.

## Rotation for head_dim 256 and 512

The reference and the kernels rotate a wide head **128 elements at a time, with the same
128-point sign tables for every group**, instead of using one 256- or 512-point transform
over the whole head. The reasons:

- **The codebooks assume it.** Upstream trained every book on coordinates of a unit-norm
  128-vector after this exact 128-point transform, where coordinates are close to
  N(0, 1/128). A 512-point transform with one norm per head would give coordinates near
  N(0, 1/512) per unit head norm. The books would then need retraining, or a per-group
  renormalization that is the 128-group scheme again.
- **Format compatibility.** buun's format is per-128 groups. Matching it keeps caches and
  codebooks interchangeable with the reference implementation.
- **One kernel path.** A warp holds one 128-group (four elements per lane). The same
  shuffle-based transform, reductions and Viterbi serve every head size, and a wider head
  only loops over more groups. Groups stay independently addressable, which the decode
  kernel and partial dequant rely on.
- **Cost.** A whole-head transform spreads each output over all 512 inputs. It needs the full
  head resident per warp, or a cross-warp exchange, for no gain the codebooks can use.

The sign tables are the engine's existing `TQP_SIGNS{1,2}_128` (seed-42 Rademacher draws,
equal entry for entry to buun's). A test parses the kernel header and compares. A
whole-head variant would be a separate codec ID, added only if it measures better on output
KL.

## Bit-exactness contract

The GPU encoder and dequant kernels produce the same bytes as `metrale-kv-tq128`. The decode
attention kernel agrees with attention over the reference's f32 dequant to within a stated
tolerance, because its softmax runs in a different order. To keep the two in step:

- Every 128-term sum runs in the kernel's order. Lane `l` holds elements `4l..4l+3` and adds
  them in order, then the warp reduces with an xor butterfly over offsets 16, 8, 4, 2, 1. The
  reference simulates the 32 lanes.
- The Walsh-Hadamard butterflies run with `h = 1, 2, ..., 64`, giving `a + b` and `a - b`. In
  the kernel the first two stages are in-lane and the rest are lane shuffles; the arithmetic
  is identical.
- The encoder and dequant kernels spell every operation as an explicit round-to-nearest
  intrinsic (`__fadd_rn`, `__fmul_rn`, `__fsqrt_rn`, `__fdiv_rn`), so their bytes do not depend
  on contraction or fast-math flags. The reference implements a correctly rounded `sqrt` and
  round-to-nearest-even fp16 and bf16 conversions in software, so it needs no `std`.
- The fp16 norm is widened to a 32-bit integer before its bytes are stored. CUDA 13.0 ptxas
  folds a byte store of `__half_as_ushort(__float2half_rn(x)) & 0xFF` into a float-to-u8
  conversion (`F2I.U8.F16.TRUNC`), which writes the truncated value instead of the low byte.
  The GPU gate caught it as a one-byte mismatch in every block.
- The GPU gate is `crates/gpu-examples/examples/tq128_microtest/`. It compares encode and
  dequant bytes for all four codecs at head_dim 128, 256 and 512. It checks decode attention
  and causal chunked-prefill attention against f64 attention over the reference's f32
  dequant, to a stated tolerance, with negative controls that must miss it.

## Status

- **Present:** the CPU reference, together with:
  - CUDA encode kernels (`reshape_and_cache_tcq{1,2,3}`, `reshape_and_cache_turbo4_n128`),
  - decode attention kernels (`paged_decode_attn_tcq{1,2,3}`, `paged_decode_attn_turbo4_n128`),
  - causal chunked-prefill attention kernels in the same module
    (`paged_prefill_attn_tcq{1,2,3}`, `paged_prefill_attn_turbo4_n128`),
  - a bf16 dequant kernel (`tq128_dequant_bf16_*`).
- **Selectable:** each codec is a `kv_codec_registry!` row of the `Trellis` launch family, so
  `--kv-cache-dtype turbo3_tcq` (and `turbo2_tcq`, `turbo1_tcq`, `turbo4_n128`) is accepted
  and planned like any other codec. The attention layer builds its trellis state (device
  codebooks, write midpoints, the prefill kernel) once at construction; the chunked-prefill
  kernel is required at boot, since no dense prefill can read these records.
- **Refused, with a coded error:** a sliding attention window and a split K/V pair. A single
  stream's first prefill chunk attends its own bf16 K/V directly (as for every codec); later
  chunks and decode read the cache.
- **Not implemented:** kernel-batched multi-sequence prefill (two or more streams' later
  chunks in one forward). As for every codec other than bf16, fp8 and nvfp4, the batched
  attention match has no arm for these caches and fails that forward with an uncoded error;
  the batch eligibility check does not yet route them per-stream.
