# GPT-OSS cooperative expert-weight reuse

2026-10-07. Experimental C1 work for [the pinned GPT-OSS-20B checkpoint](https://huggingface.co/openai/gpt-oss-20b/tree/6cee5e81ee83917806bbde320786a8fb61efebee). Numerical reference qualification and benchmark certification remain open.

The accepted 16-token path reuses each decoded expert weight across four independently accumulated token rows. The wider entry preserves that arithmetic and moves plan validation to cooperating threads within each CTA. Every thread reaches both barriers before row-tail return. Invalid counts and entries are checked before addressing weights or inputs; malformed plans poison affected expert outputs. Complete, unique coverage remains a required host typed-plan invariant.

The original narrow entry and its public 16-token limit remain unchanged. Explicit serving capacities 64 and 128 use the new entry; scalar prefill remains the default, and the existing chunk opt-in still defaults to 16. The selected capacity drives allocation, chunk boundaries and the allocator-derived reserve subtracted before KV sizing. C1, BF16 KV, stream-bound teardown and refusal of prefix adoption, disk swapping and graph capture remain enforced.

## Correctness evidence

- 168 constructed gate/down projection comparisons against the accepted token-grid implementation: row counts 35/5760/2880, input widths 96/2880, and token counts 1/2/3/4/5/15/16/17/31/63/64/65/127/128. Every output bit matches.
- Reversed valid plans match; omitted/duplicate/wrong-expert host plans refuse. Raw device count overflow, duplicate entry, out-of-range entry and wrong-expert controls poison exactly the affected outputs. Wrong slot stride and wrong score-token stride are detected. Independent BF16 product/ascending-expert reduction controls pass.
- All 24 layers over the 251-token Harmony fixture and following decode match scalar hidden-state and KV bytes at widths 16/31/64/127/128. Future KV storage begins as NaN; no future reads leak nonfinite values. Stream mismatch, released scratch, invalid extents and allocator-failure poisoning controls pass.
- GB10 production-equivalent resource compilation: 40 registers, no local stack or spills; the cooperative entry uses four bytes of shared storage. SM90 and SM100a compilation passes. Cross-hardware reach is 50 targets across GB10, Hopper and B200; these compile checks do not certify those devices.

The earlier larger token-grid serving experiment was rejected for latency regressions. Its raw results remain preserved. This change evaluates weight reuse across the larger chunk instead of simply increasing token-grid launch dimensions. Kernel-only comparisons against token-grid are not end-to-end gains against the accepted 16-token serving path.

## Serving and repeated timing

Both capacities passed 7 lifecycle cases (including cancellation and reuse), 12 text-quality cases, 9 blocking-tool cases and 5 tool-stream cases. Independent grader controls and prior failures remain retained. The serving binary is `f1153c9c4dc5270680b8554418724b4e1097d5d37a0b1fda8edc9c21341891d9`, built from `553e9eae516660f9b130afda6488f7e5e3385ed0` plus the recorded source overlay. The MXFP4 module PTX is `4332cde9015839212603056f3921e70b041eeaa3f9c74c5ea088379e373a3c25`; source is `07e94ed1647e5087c7658783753e3652af338db062ca3976ad14be3a1cf024f7`. Documentation and inventory regeneration followed the frozen build without changing runtime sources.

The same release binary ran capacities 16→64→128, then 128→64→16 on an idle GB10. Each arm included warmup followed by three repetitions per prompt. No profiler or other build/compute jobs ran during these six arms. Prompt/completion counts remained 81/20, 87/63 and 304/24, with exact expected text and clean termination throughout. Predeclared gates required no individual total-latency or first-generated-token regression above 2%, plus a geometric total-latency speedup above 1.02.

| Capacity | First-order total-latency reduction | Reverse-order reduction |
|---|---:|---:|
| 64 | 10.49% | 10.24% |
| 128 | 12.03% | 11.85% |

Reductions are geometric across the three case medians, relative to capacity16 in the same session. Both orders pass the complete gate. First-session capacity128 medians:

| Case | Capacity16 total | Capacity128 total | First generated | First visible | Decode tokens/s |
|---|---:|---:|---:|---:|---:|
| Arithmetic | 1.0692s | 0.9378s | 0.4591s | 0.9047s | 40.58 |
| Counting | 2.1915s | 2.0284s | 0.4894s | 0.8876s | 40.59 |
| Retrieval | 2.6710s | 2.2395s | 1.6586s | 2.1574s | 40.33 |

First-generated timing is the server boundary, first-visible timing is the first client-visible SSE text, and total is client request completion. Hidden reasoning explains why the first two boundaries differ. These are bounded native-to-native C1 improvements, not a claim of competitive speed, energy superiority, reference-logit equivalence or certification. The separate optimized-reference gap remains open. Select this route explicitly with `--experimental-gpt-oss --experimental-gpt-oss-chunk-prefill --experimental-gpt-oss-chunk-tokens 128`; omitted capacity retains16.
