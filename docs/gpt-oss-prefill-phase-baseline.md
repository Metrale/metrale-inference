# GPT-OSS prompt-length baseline

On October 7, 2026, the accepted native binary remained substantially slower
than the pinned optimized reference as prompt length increased. This controlled
C1 workload uses identical token arrays and exactly one generated token. It
measures warm HTTP completion latency through the terminal event, including
prefill, final projection, sampling and protocol overhead. It does not isolate
pure prefill or prove a first-generated-token timestamp, and is not a quality
test.

| Prompt tokens | Native session 1 / 2 (ms) | Reference session 1 / 2 (ms) | Native/reference ratio, sessions 1 / 2 |
|---:|---:|---:|---:|
| 64 | 340.65 / 343.87 | 60.03 / 61.77 | 5.67 / 5.57 |
| 128 | 623.35 / 628.72 | 65.39 / 67.74 | 9.53 / 9.28 |
| 256 | 1227.09 / 1222.16 | 74.21 / 75.82 | 16.53 / 16.12 |
| 512 | 2427.44 / 2411.25 | 94.53 / 94.72 | 25.68 / 25.46 |
| 1024 | 4851.82 / 4828.35 | 141.16 / 143.21 | 34.37 / 33.71 |

Each cell is the median of five measured requests after one warmup per length.
Engine order was native, reference, reference, native. The two reference sessions
shared one process; native restarted between its sessions. Length order
alternated within each session. Loading and compilation were excluded, no
profiler ran, and no competing host/GPU job ran during measurement. All 120 raw
responses, including 20 warmups, passed exact prompt-ID/count admission, one
completion token, length termination and exactly one DONE. Native usage explicitly
reports zero cached tokens; reference usage omits that field, so its cache
boundary is the recorded `--no-enable-prefix-caching` setting.
Generated token identities may differ between engines; fixed counts do not imply
identical internal routing or numerical policies.

Both used the original [openai/gpt-oss-20b checkpoint at
6cee5e81ee83917806bbde320786a8fb61efebee](https://huggingface.co/openai/gpt-oss-20b/tree/6cee5e81ee83917806bbde320786a8fb61efebee),
MXFP4 weights, BF16 KV, C1/TP1, context 2048, memory utilization 0.85 and disabled
prefix caching on the same DGX Spark. Native used the accepted explicit
128-token chunk path and existing raw `/v1/completions` admission, without
changing Harmony chat guards. Temperature was zero and minimum/maximum output
counts were both one. Operational prose supplied the pinned-tokenizer prefix
arrays; these were not chat-template prompts.

Native binary SHA-256:
`829dbf3a9b048dbc4370cc126d5c6f0d250e77f7e47500869b67c81d2313086f`.
The reference used the unmodified official NVIDIA image
`nvcr.io/nvidia/vllm@sha256:fa68ef92f906e1b3770621625c5af539d15297fea15dacfc1466b853a567c5b6`,
with its optimized graph/attention execution. No diagnostic precision policy or
capture hook was enabled. Earlier comparisons of older native binaries remain
separate in the [reference record](gpt-oss-competitive-reference.md).

Controlled evidence package `gpt/prefill-phase-sprint` retains exact commands,
server identities, source evidence, all requests/SSE responses and an independent
admission/median analysis in `comparison.json`. Fixture SHA-256:
`857996de4dcd1c308163b94507b9e45cf64ca81499dedcbfa0a77465ec6a6373`.
The gap grows with prompt length; this measurement does not establish which
operation causes it.
Any candidate must retain numerical, lifecycle and task-quality gates and be
measured against a contemporary native control in both orders.
