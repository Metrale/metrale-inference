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
