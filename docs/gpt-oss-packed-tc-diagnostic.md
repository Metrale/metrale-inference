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
