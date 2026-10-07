# GPT-OSS packed tensor-core diagnostic

Updated 2026-10-07. This is an explicit diagnostic path, with **no serving admission or numerical qualification**. The accepted serving expert path remains unchanged.

The checkpoint is [openai/gpt-oss-20b at 6cee5e81ee83917806bbde320786a8fb61efebee](https://huggingface.co/openai/gpt-oss-20b/tree/6cee5e81ee83917806bbde320786a8fb61efebee). The comparison uses pinned Transformers 4.55.0 BF16 eager execution on GB10. Packed E2M1/g32 weights remain in their native layout. The diagnostic decodes BF16 tiles inside the existing grouped tensor-core family; it does not materialize the entire checkpoint as BF16.

## Scope and controls

`PrefillScratch::new_packed_tc_diagnostic` explicitly selects the alternate expert reduction policy for full 128-token chunks. Smaller chunks and tails keep the accepted expert path. A validated expert plan supplies offsets, gate/down gather rows, and an inverse permutation; existing BF16 gather restores slot-major outputs before the existing bias, activation and reduction stages. Extra scratch allocation is explicit and checked against remaining device memory. No CLI flag or factory default selects this path.

The default-false family policy preserves existing E8M0/NVFP4 entries. All 11 old PTX entry bodies remained instruction-identical after normalization of compiler labels and template symbols. Constructed tests passed 119 comparisons, including all 4,096 packed decoder combinations, mixed expert counts, gather permutations, independent exact operands, negative controls, and old-entry fractional output identity. Selected learned projection/bias tests matched the pinned B32 reference across 7,223,040 values at M1/16/64/128. These bounded results do not establish whole-model equivalence.

## Full-model qualification failed

The unchanged 251-token teacher-forced fixture crosses sliding-window and page boundaries. All outputs were finite. The scalar and 64-token fallback arms reproduced their retained baseline bytes exactly. The 128-token tensor-core arm failed the strict native hidden/cache byte gate; raw failures were retained.

| Expert policy | Attention policy | Matching next-token IDs /251 | Exact layer-0 positions /251 | Mean logit normalized RMS error |
|---|---|---:|---:|---:|
| Accepted | FP32 online | 244 | 0 | 0.0233855 |
| Packed tensor core | FP32 online | 232 | 0 | 0.0250988 |
| Accepted | Staged BF16 | 243 | 170 | 0.0265144 |
| Packed tensor core | Staged BF16 | 236 | 193 | 0.0239959 |

The staged-attention factorial reproduced the previous staged baseline hidden/logit SHA-256 values before comparison. Improving a local stage or average error did not improve the strict whole-model next-token gate. Neither tensor-core arm is promoted. The existing accepted path also retains its original 244/251 qualification limitation; no threshold or tie policy was changed.

Warm projection-only screening suggested an M128 opportunity, but excluded plan construction and output reorder. It is not a whole-request speed result. Further work must identify the first differing routing decision and reproduce the responsible operation on identical operands before altering production numerical policy.

Targeted reference hooks at positions 1, 12, 29, 35, 47, 48, 52 and 70 reproduced the original complete reference hidden/logit hashes. In the combined tensor-core/staged-attention arm, all eight layer-0 router logits and selected IDs match the reference exactly. Positions 29 and 70 nevertheless have 236 and 253 differing layer-0 hidden values. Positions 29, 35 and 52 retain the reference selected-expert sets through all 24 layers despite their final next-token mismatches. Thus routing crossings explain only part of the divergence; equal router logits alone do not prove equal full router inputs. Bounded expert-stage captures and identical-input replay remain necessary.

Identical-operand follow-up narrowed the first-layer differences further. At positions 29 and 70, a single upstream normalized-input value differs; passing that same native input through the pinned reference expert operator changes 76 and 214 gate-after-bias values. The selected tensor-core expert stages reproduce the reference on identical inputs. The native O projection differs from reference `F.linear` by 1–3 BF16 values at several sampled positions. At position 12, this changed O input explains the normalized-output difference. At positions 29 and 70, however, reference normalization on the native O-plus-embedding input reproduces the original reference norm while native normalization differs by one value. This isolates a same-input normalization arithmetic difference before MoE. Subsequent normalization experiments below distinguish reduction policy without changing the accepted path.


## Normalization policy experiments were not promoted

The installed pinned Torch source uses a vectorized FP32 mean reduction and multiplication by a pre-rounded `1/N`. The accepted native norm uses a different reduction tree and FP32 division by `N`. Exact rational analysis also separates ideal mathematical RMS normalization from the pinned staged FP32 policy: at one captured midpoint the native result matches the ideal rounding, while at another the reference does. Ideal rounding is therefore not a universal substitute for the pinned execution contract.

Two private PTX overrides changed only the norm module. Reciprocal scaling alone reproduced the sampled normalization outputs, but reduced the original whole-model next-token agreement. A second override reproduced the pinned reduction tree, mean, reciprocal square root and output bits across 39 identical-input cases, including fresh constructed inputs and zero. Its whole-model results also failed to improve qualification:

| Norm policy | Attention | Accepted experts /251 | Packed tensor-core experts /251 |
|---|---|---:|---:|
| Existing | FP32 online | 244 | 232 |
| Reciprocal scaling only | FP32 online | 238 | 237 |
| Pinned reduction tree | FP32 online | 237 | 236 |
| Existing | Staged BF16 | 243 | 236 |
| Reciprocal scaling only | Staged BF16 | 233 | 237 |
| Pinned reduction tree | Staged BF16 | 236 | 234 |

All runs used the same 251 input IDs, checkpoint, reference traces and frozen native executable. The module manifests verified every unchanged PTX hash. Neither the reference nor acceptance criterion was regenerated. The reduction-tree source SHA-256 is `d5adb4266a54586a047f33453cd95f30a904dcc7a158084588cfaea71a196f13`; the complete factorial comparison receipt is `680c21c9cceb796f105685758153ea52e0b3c062a1faea374b3a4ee9881fe45d`. These failures show why a same-input primitive improvement cannot establish whole-model numerical equivalence. Both overrides remain private diagnostics; shared normalization and serving defaults are unchanged.

A subsequent identical-input O-projection probe preserved FP32 tensor-core accumulators and added bias before the single BF16 store. It matched Torch batched M8/M16 `F.linear` exactly across 23,040/46,080 outputs, but differed from the original sequential M1 reference at 55 outputs across eight rows; the accepted projection differed at eight. Batch geometry therefore changes the reference arithmetic policy and must remain explicit. Rounding the product to BF16 before adding bias failed 7,386 M8 outputs and a constructed exact cancellation control. This epilogue boundary is necessary, but does not establish sequential-reference equivalence. No tensor-core O substitution was promoted.

## Private API experiment: useful speed potential, still unqualified

A separately frozen, uncommitted admission overlay exercised this alternate policy through the real HTTP server. It required native experimental admission, explicit chunk size 128, C1, BF16 KV and the existing 0.85 memory limit; additional tensor-core scratch was reserved before sizing the KV pool. The checked-in server has no such admission flag. A dispatcher marker verified full-128 tensor-core execution, while tails and decode used the accepted implementation.

The corrected `gpt_oss_mxfp4_mma` module loaded on GB10 and reproduced all nine original tensor-core hidden/cache/logit hashes, including their strict numerical failures. The first attempt hit the existing memory guard and was retained. A fresh run followed release of page-cache advice only for completed owned build artifacts; no guard, weights or numerical policy changed.

Separate API checks passed 12 text cases, nine blocking-tool cases and five streamed-tool cases. Exact prompt-boundary checks at 128, 129, 255 and 256 tokens returned the required answer. Text prompts ranged from 363 to 931 tokens, and tool call/result prompts had 157/207 tokens. These checks exercise actual tensor-core dispatch; short fallback-only successes were not counted as tensor-core evidence.

An idle-host A→B then B→A latency campaign compared accepted wide reuse with the private alternate policy. Each arm used three measured repetitions after warmup and unchanged output graders. Prompt/generated counts were identical in every arm: 225/15, 231/63 and 448/29. Geometric total request latency was **19.40% and 19.22% lower** for the diagnostic. First-generated latency ratios were 1.28–1.65×; decode remained approximately 40.3–40.7 tokens/s. This measures speed potential, not numerical equivalence or certification.

The candidate was based on source `6d75fdc` plus a hash-recorded private admission overlay and module packaging `7b8a998`. The accepted executable SHA-256 was `e52eef8199a0afbb84dd5ea04b0891c5780b93b81bbaa12e08d22e26f1f82fbc`. Complete source identities, raw requests/responses, server logs and opposite-order receipts are retained. The original 251-token qualification failure remains authoritative, and no alternate numerical path is promoted.

## Reference batch self-consistency does not waive qualification

A pinned-reference self-consistency diagnostic completed on the unchanged 251-token corpus/checkpoint. The scalar rerun reproduced the original complete hidden/logit SHA-256 values exactly. Changing only input batching to 128+123 tokens or a single 251-token batch produced **238/251** and **239/251** matching next-token IDs versus scalar. Pinned Transformers4.55 eager BF16 math/cache settings and model files were unchanged.

All native policies were then compared to all three reference geometries, without substituting a favorable oracle:

| Native policy | Scalar reference | 128+123 reference | Full251 reference |
|---|---:|---:|---:|
| Accepted experts, FP32 attention |244|241|242|
| Private packed TC, FP32 attention |232|241|240|
| Accepted experts, staged BF16 attention |243|240|243|
| Private packed TC, staged BF16 attention |236|239|240|

Counts are matching next IDs /251. Native TC uses a full128 TC chunk followed by a123-token accepted-expert tail, while the reference batches all operations; these execution policies are explicitly distinct. Complete next-ID vectors, per-token/per-layer NRMS, first differing coordinates and source/output hashes are retained. TC does not become a numerical winner: its logit mean NRMS versus the batched references is ~0.0267–0.0300, compared with accepted FP32 ~0.0237–0.0241.

This identifies real reference batching variability, **not a relaxed acceptance threshold**. The original scalar-reference244/251 limitation and all failed TC/norm gates remain unchanged; no default or serving promotion. Checkpoint: [openai/gpt-oss-20b@6cee5e81ee83917806bbde320786a8fb61efebee](https://huggingface.co/openai/gpt-oss-20b/tree/6cee5e81ee83917806bbde320786a8fb61efebee).
