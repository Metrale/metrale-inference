# GPT-OSS optimized reference comparison

The measured native chunk-prefill baseline is not yet competitive with the
optimized reference on these bounded warm C1 workloads. This is a diagnostic
comparison, not model certification or a general throughput claim. It compares
the explicit chunk implementation at `c5dd000`, before later token-grid changes;
subsequent candidate results belong in the [native performance record](gpt-oss-native-performance.md).

Both use the same DGX Spark, original MXFP4 checkpoint and BF16 KV cache, C1/TP1,
maximum context512, memory utilization0.85 and disabled prefix caching. Actual
prompt-token arrays match exactly at81/87/304 tokens. The reference keeps its
mature graph and FlashInfer/Marlin execution; native does not use CUDA graphs.
Equal checkpoint format does not imply identical intermediate rounding.

| Workload | Native visible / total | Reference visible / total | Generated tokens native / reference |
|---|---:|---:|---:|
| Arithmetic |1.512 /1.539s|0.448 /0.469s|20 /20|
| Count 1–16 |1.536 /2.671s|0.407 /1.351s|63 /63|
| Retrieval |4.423 /4.500s|0.403 /0.465s|24 /19|

These are medians of six measured requests per case, from two sessions with
warmups. All18 reference requests pass the bounded final-answer/framing checks.
Visible latency means first nonempty user-visible SSE content, excluding hidden
analysis. Native is3.28×/1.98× slower in total latency for the first two workloads.
Retrieval has unequal output work: its task-latency ratio is not fixed-work
throughput. A separate exact-token completion diagnostic produced24 tokens in
0.569s, with an unexplained early generated-token divergence from the chat path.
Its raw first-token arrival includes analysis and must not be called visible TTFT.

## Frozen reference identity

- Official NVIDIA vLLM26.09 arm64 image:
  `sha256:fa68ef92f906e1b3770621625c5af539d15297fea15dacfc1466b853a567c5b6`.
- Installed vLLM `0.29.0+5013de39.nv26.9.69442229`,
  Torch `2.14.0a0+b2c75dd062.nv26.9.68203377`, Transformers `5.16.1`.
  Actual package receipts take precedence over advertised release versions.
- Bundled user-space forward-compatibility driver615.71.09 over unchanged host
  driver580.173.02; the initial tiny CUDA qualification passed.
- First chat request failed during separate Harmony vocabulary setup, before
  inference. The official vocabulary was provisioned with its upstream hash;
  the failed request and setup logs remain retained.

Tested model: [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b), pinned
to [6cee5e81ee83917806bbde320786a8fb61efebee](https://huggingface.co/openai/gpt-oss-20b/tree/6cee5e81ee83917806bbde320786a8fb61efebee).
Controlled evidence is in the session's `gpt-competitive-reference` package:
`comparison.json`, `artifact-hashes.json`, and raw commands, package metadata,
prompt arrays, API outputs and logs under `remote/`.

Limitations remain explicit: separate sequential engine sessions rather than
randomized interleaved restarts; no energy, concurrency or certification result;
no assumption of identical generated traces; no production-readiness conclusion.
The earlier dequantized BF16 eager reference was a different, slower execution
path and cannot establish competitive inference.
