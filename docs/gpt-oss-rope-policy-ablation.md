# GPT-OSS rotary arithmetic policy diagnostic

The private FP32 rotary variant is **not promoted**. It matches a bounded
compiled-reference generation prefix, but agreement with the original pinned
sequential Transformers reference falls from 244/251 to 233/251 next-token IDs.
The accepted implementation and all original qualification failures remain.

Checkpoint: [openai/gpt-oss-20b at
6cee5e81ee83917806bbde320786a8fb61efebee](https://huggingface.co/openai/gpt-oss-20b/tree/6cee5e81ee83917806bbde320786a8fb61efebee).
The preceding [coding investigation](gpt-oss-coding-comparison.md) records the
reference image, actual compiled-wrapper captures and remaining task failures.

## Same-operand evidence

At position 149, the captured reference embedding and initial normalized input
match native exactly. Its executed rotary kernel retains FP32 cosine/sine,
products and sum before the final BF16 store. Native follows the pinned
Transformers policy with BF16 cosine/sine and intermediate products.

An independent exact-rational operation-boundary oracle, supplied the same
captured reference QKV and FP32 cosine/sine row, finds:

| Policy | Differences from 4,608 captured reference Q/K values |
|---|---:|
| BF16 staged products | 1,935 |
| FP32 products and sum | 0 |
| Either tested single-FMA contraction direction | 0 |

This case does not distinguish FMA contraction. Exact cancellation zeros are
canonicalized by the oracle and sign-only differences reported separately;
there were none here. FTZ is not modeled. Controls include 65,279 finite BF16
round trips, signed midpoint cases, 4,000 independently checked bounded FP32
sum/product pairs and four constructed policy counterexamples.

The staged policy on those reference operands matches actual native Q/K at
4,606/4,608 coordinates. Projection differences remain separate: one V output
already differs before rotary arithmetic, and exact dot-plus-bias rounding
agrees with native for that coordinate. Matching a reference bit pattern is
therefore not a universal repair criterion.

## Private full-path ablation

Only the private rotary PTX module changed. Native frequency generation,
checkpoint, attention, experts and all other module bytes stayed fixed. Source
`de70fa3b7e169dc0de79f7f2d0ff6b4b03f7dc8c661d33b885185301ecc7f00c`
compiled with `nvcc -ptx -arch=compute_121 --fmad=false` to PTX
`5468894e6e4b00895c73ef4ee3b76d8e927d66cb3acbd43ff19dcff6e73ccb6f`.
The initial planning record included `-O3`; the executed manifest records the
actual command above. This is not a performance experiment.

Frozen diagnostic binary
`f99471d75df455172f2a2000eede577539bbd3a683a9ef09cea996fc18ead928`
ran the original 251-token corpus and the 150-token topology prefix on GB10,
retaining the 15% memory reserve. All baseline module hashes were checked before
replacement, and required symbols resolved through the actual GPU backend.

| Reference on the fixed 251-token corpus | Accepted native matches | Private FP32 rotary matches |
|---|---:|---:|
| Original sequential Transformers | 244 | 233 |
| Transformers chunks 128 + 123 | 241 | 235 |
| Transformers full 251-token prefill | 242 | 235 |

The fixed topology next-token decision changes from 23665 to 23483, matching
the compiled reference. A separate actual 32-token generation produces exactly
the same 32 IDs as that reference. Both stop at the diagnostic length limit;
this is an analysis prefix, not a completed answer or a coding-task pass.
The exact 1,024-token coding request was not run. A separately prepared larger
consumer was CPU-built but never executed and was removed from the accepted
source stage; existing harness guards were not changed.

Raw traces, all three reference comparisons and the rejected decision remain
private. Comparison receipt SHA-256:
`9bbfd991c1f8c291b3eedf8422fc38c0031c690762b1154fd75c3a6f7b92c1a8`.
Generation receipt SHA-256:
`bd90f9212bf2cae881ad1330dacd59051cd8700fea78f61817d7c9fe6b586f60`.
No production precision policy, serving default or qualification gate changed.
