# GPT expert block scheduling screen (2026-10-07)

The private expert-last block scheduling candidate is **rejected for promotion**. Its two opposite-order request sessions reduced geometric latency by **1.996527% and 1.675704%**, below the predeclared **2% in both sessions** requirement. The accepted runtime and serving defaults remain unchanged.

Model: [openai/gpt-oss-20b at 6cee5e81ee83917806bbde320786a8fb61efebee](https://huggingface.co/openai/gpt-oss-20b/tree/6cee5e81ee83917806bbde320786a8fb61efebee). Tests used Spark1/GB10, C1, native packed MXFP4, BF16 KV/head, localhost, memory utilization 0.85 and context 2048. This is a bounded diagnostic, not certification or a claim of competitive performance.

The only kernel change remapped the cooperative expert/group grid so each expert's token groups are adjacent in the logical block ordering. Per-token FMA, reduction, BF16 stores, legacy small shapes and launch dimensions were unchanged. Both kernels used 40 registers, zero stack/spills and four bytes shared memory. A warm kernel screen on eight saved router distributions passed with 2.3277%/2.4978% reductions; that did not establish a request-level win or prove an L2-cache mechanism.

Correctness passed 295 primitive records, actual-distribution output comparisons, all twelve full251 hidden/cache hashes including following decode, text quality (12), blocking tools (9), streamed tools (5), cancellation/reuse lifecycle (7), and prompt boundaries 128/129/255/256. All 120 ladder requests retained exact input IDs, output counts/text and actual first sampled IDs correlated separately from complete server logs. The API itself exposes text/counts, not generated IDs.

Each request session used five measured requests after one warmup at each prompt length, with exactly one generated token. Timing is HTTP prompt processing plus head, sampling and protocol, not isolated prefill. The primary statistic uses lengths 256/512/1024. Secondary chat workloads retained counts 81/20, 87/63 and 304/24 and passed the declared no-regression controls.

| Prompt tokens | Session 1 baseline/candidate ms | Session 2 baseline/candidate ms |
|---|---:|---:|
| 64 | 340.18 / 344.99 | 343.99 / 342.35 |
| 128 | 622.49 / 619.88 | 629.74 / 618.08 |
| 256 | 1224.18 / 1203.54 | 1223.96 / 1204.91 |
| 512 | 2423.26 / 2370.83 | 2415.57 / 2373.86 |
| 1024 | 4841.73 / 4738.13 | 4828.88 / 4744.68 |

Frozen identities:

- Accepted release: `829dbf3a9b048dbc4370cc126d5c6f0d250e77f7e47500869b67c81d2313086f`.
- Candidate release: `9653c7c424dcf0bf95a88b8612cc399e37bf995e1d60b59dd57b6ebb8318c2ef`.
- Candidate kernel: `b0b4b473aa8ff4a8da42b90239af3debf7fbf9d870736e4807d4579b0e105316` (base 8b2f8ec plus this single source override).
- Comparison receipt SHA-256: `02cf779989c99ab8cc49f3bc03df19d88fec6ca51467ca7dcc8787485c3d669f`.
- Actual sampled-ID control SHA-256: `1dc25ead6e7c27a4f0dc65163eff5d1c0f9ba53bee521267ac82cc4fb8cf075c`.

Private raw requests, logs, source/module identities, failed decision and complete numerical controls are retained. Original numerical qualification remains open; this experiment neither fixes nor relaxes that gate.
