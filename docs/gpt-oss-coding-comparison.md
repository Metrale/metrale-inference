# GPT-OSS bounded coding comparison, October 7, 2026

The accepted native path passed fewer authored coding tasks than the pinned
reference in this comparison. This is an unresolved quality counterexample,
not a general coding benchmark or a certification result.

Checkpoint: [openai/gpt-oss-20b at 6cee5e81ee83917806bbde320786a8fb61efebee](https://huggingface.co/openai/gpt-oss-20b/tree/6cee5e81ee83917806bbde320786a8fb61efebee).
Four tasks cover interval merging, retry-delay repair, dependency ordering and
idempotent ingestion repair. Their 31 authored semantic cases were repeated
three times; repeated cases are not independent coverage.

| Observation | Native | Pinned vLLM |
| --- | ---: | ---: |
| Complete task attempts passing every semantic case | 6/12 | 9/12 |
| Individual semantic checks, including repeats | 75/93 | 84/93 |
| Strict requested code-only format plus semantics | 0/12 | 0/12 |

Every response contained one unambiguous Python code fence. These remain strict
format failures. A separate semantic score uses only the code inside that single
fence. Hidden reasoning was never used as candidate code. Both paths failed the
retry-delay task in every repeat. Native dependency-order answers incremented
indegree on dependencies and traversed dependencies as successors, reversing the
required dependency direction; the reference constructed dependency-to-dependent
edges and passed that task. This failure is independent of the fence formatting.

Actual request JSON was identical: temperature zero, explicit low reasoning
effort, 1,024 generated tokens including analysis, and unchanged user messages.
Both used C1, context capacity 2,048, BF16 activations/KV and the original MXFP4
checkpoint with memory utilization capped at 0.85. Prompt token counts matched
146/224/147/213 across the four tasks, but actual rendered token IDs were not
verified at capture. This is a same-message API comparison; model-versus-runtime
attribution remains unresolved. No request was truncated or omitted from the
score. Raw responses preserve finish reasons, usage and reasoning metadata.

Native binary SHA-256:
`e52eef8199a0afbb84dd5ea04b0891c5780b93b81bbaa12e08d22e26f1f82fbc`.
Its source was `6be4eaadaf3b0f3d1c18c66416f1d6cf01da5036` plus the accepted
inactive-group source overlay
`a48f910c4f3ed86acead23e1ed4bd404849fc6feadbefc2051a7087ddc28beb5`;
chunk capacity was 128. The alternate packed tensor-core diagnostic was not used.
The official reference image was
`nvcr.io/nvidia/vllm@sha256:fa68ef92f906e1b3770621625c5af539d15297fea15dacfc1466b853a567c5b6`,
with native vLLM MXFP4/Marlin execution, prefix caching disabled and seed zero.

Seven local adapter controls passed before requests. Generated code executed
only in the existing pinned isolated Docker grader: no network, no host mounts,
read-only filesystem, unprivileged user, bounded memory/CPU/wall time/output.
No generated code ran directly on the host. These correctness runs overlapped
unrelated CPU compilation, so their elapsed times are not performance evidence.

Private evidence package: `gpt/coding-comparison`, retaining collectors, immutable
requests, engine identities, raw HTTP responses, candidate hashes and per-case
Docker results. Native/reference collection hashes respectively:
`2d6f89c6d41095d1b8334981c150f2193abdf48bc953e4d7e55caa5e009b61f9` and
`8d957f0ad6babb0cfc45a5f15953960607606f46a71dcdb3529272dfc299f40e`.
Original numerical qualification limits remain unchanged.
