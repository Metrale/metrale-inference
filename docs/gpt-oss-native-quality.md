# GPT-OSS native text API quality checkpoint

2026-10-06: twelve bounded native API cases passed, with eleven offline grader
controls. This expands the earlier smoke tests; it is not broad model, numerical,
performance or production qualification.

Tested [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) at
[6cee5e81ee83917806bbde320786a8fb61efebee](https://huggingface.co/openai/gpt-oss-20b/tree/6cee5e81ee83917806bbde320786a8fb61efebee),
original MXFP4 weights, native Metrale C1, BF16 KV/head, memory utilization 0.85,
localhost port 18842, maximum sequence length 2048, temperature 0 and low reasoning effort.
The existing API binary is SHA-256
`11873e8c8d6ad21b8519e277c22167cbf24143388e2935f4192df55d44f45267`.
Its full source-overlay receipt is retained from the tool-streaming campaign.

| Case | Required result | Result |
|---|---|---|
| Negative arithmetic | `17 - 29` → `-12` | Pass |
| Multiplication | `13 * 17` → `221` | Pass |
| Current identifier extraction | `LM-731`, not the old record | Pass |
| Unicode copy | `café 日本 😀` | Pass |
| Multi-turn update | Latest identifier `WEST-09` | Pass |
| System output constraint | Only `ORBIT` | Pass |
| Instruction/data distinction | Extract `ZX-204` despite an instruction inside the record | Pass |
| Empty-set reasoning | Only `NONE` | Pass |
| JSON sorting | Bare array `[1,2,7,9]` | Pass |
| JSON extraction | Exact keys and integer count | Pass |
| Early-context retrieval | `NORTH-83` from a 643-token prompt | Pass |
| Reuse after the longer request | Only `ORBIT` | Pass |

Every response additionally passed model identity, assistant role, stop reason,
usage accounting and hidden-analysis/framing checks. JSON grading rejects duplicate
keys, Markdown fences, extra keys and type substitutions such as a boolean for an
integer. Known-bad controls were exercised before live requests. Full response
bytes, requests, expected values, launch command, server log and source identity
are retained privately. Response JSONL SHA-256:
`664dc79a6a92986f0bc7a91a2726d7b31a45fd0bbfc622e034c69adef8c34db2`.
Probe source SHA-256:
`25b7eaf3c30d4df2a175c7502c677451e4e55334813efdbadb44188347d8c1a4`.

The sorting prompt explicitly prohibited Markdown; the earlier less-specific
sorting fixture that returned fences remains a historical failure. This run
neither repairs that result nor establishes JSON-schema constrained decoding.
The 643-token check is bounded retrieval, not long-context qualification. The
instruction/data example is one fixture, not prompt-injection robustness.

A concurrent CPU build makes these client timings unsuitable for performance
claims. The owned server was stopped after the run and GPU memory released.
Original numerical parity misses, broader quality, concurrency, efficient prefill,
performance and certification remain open.
