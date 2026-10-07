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
instrumented at capture. A subsequent offline replay used the installed pinned
vLLM `OnlineRenderer._make_request_with_harmony` and the native Harmony tokenizer
oracle on the captured requests and date (2026-10-07): all four complete token
arrays matched exactly. This narrows template uncertainty without proving the
live GPU input buffers or identifying the cause of different generated answers;
model-versus-runtime attribution remains unresolved. No request was truncated or omitted from the
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

Offline replay evidence is `offline-id-comparison.json` in the same private
package, with raw IDs, renderer source hashes and the native oracle binary hash.
The CPU-only reference container used no GPU, model weights or network. Original
semantic and formatting scores are unchanged.

## Private tensor-core diagnostic follow-up

The original private packed tensor-core path was also run on the exact same
12 request JSON objects, with the same 2,048 context and generation policy.
Its full-128 gate/down dispatch marker was observed. It passed **6/12 complete
task attempts and 78/93 repeated semantic checks**, with **0/12 strict format**.
It still failed retry-delay and dependency-order tasks in every repeat, although
generated code differed from the accepted native path. The three additional
individual checks do not repair either complete task or establish broader quality.

This used binary
`ad473b4507a1201150d56fb9d9cc5e50ad2ccb8eb3fdf97ba64d1f3e379bd0d3`,
source base `6d75fdc6ee4574b295b76410486625e15bd095ad` plus the frozen private
CLI/runtime overlay in its identity receipt. The separate rejected sparse-gate
variant was not used. Private collection hash:
`2f0751ece08c0d5bff5daa54674e9b96c08a5ed5f9979ca318842586a5fb07ad`.
The original failed 251-token numerical comparison remains failed; this diagnostic
has not been promoted or admitted as supported. `comparison-three-arms.json`
retains all original scores and per-case results. No performance claim derives
from these correctness runs.

## Actual-token first-divergence diagnostic

A separate 32-token greedy diagnostic used the verified 147-token dependency-order
prompt directly. Actual generated IDs first differed at index 3, after three
shared Harmony header tokens: native chose `23665` (`Implement`), while the
reference chose `23483` (`Need`). Native BF16 logits were 26.875 and 26.75,
respectively, a 0.125 margin. The reference exposed equal top log-probabilities
for those two tokens (`-0.9627416133880615`). Equality at the API's reported
precision does not establish an exact internal-logit tie or its tie-breaking rule.

Both engines then received the identical 150-token common prefix independently.
Each repeated its own choice. The complete native 201,088-logit row was byte-identical
between generation and replay (SHA-256
`e29ce03a48146b00ea1cbb7cd6aadd9e36977adfea5de70ee41bcda6105526ec`).
Reference top-two reported log-probabilities remained equal, but other reported
values changed between decode and replay, so reference dispatch equivalence is
not assumed. This locates an early ranking difference; it does not establish that
this particular decision caused the later reversed-dependency implementation.

Native used a freshly rebuilt standalone example from the frozen accepted source,
explicitly loading all 13 SHA-verified accepted PTX modules; skipped target builds
were not used as executable kernels. Its binary SHA-256 is
`f99471d75df455172f2a2000eede577539bbd3a683a9ef09cea996fc18ead928` and module
manifest SHA-256 is `2545623a0e89c6a8214a806e092e972f0db42c084cda4ce6e18b0f37b9382b6a`.
On this prompt, scalar and chunk widths 16/31/64/127/128 produced identical hidden
states and KV caches, including a following decode. The standalone generation
prompt trace also matched that scalar trace exactly. Nine local actual-ID
admission/comparison controls passed before requests.

The reference used the same pinned official image and actual `token_ids` from
raw completions with log-probability capture. No IDs were reconstructed from text.
These diagnostic routes differ from the original chat captures and provide no
performance claim or proof that their complete trajectories reproduce those
captures. Evidence is retained privately in `gpt/topology-first-divergence`;
all original semantic scores and failed numerical gates remain unchanged.

A subsequent same-input head microprobe reproduced the four retained native logits
exactly, then compared the selected BF16 head rows against an independent exact
rational dot-product oracle. All four correctly rounded BF16 results agreed,
including the two competing tokens. Torch 2.13.0+cu130 `F.linear` on those identical
native inputs also agreed. This is a separate comparator environment from the
NVIDIA image's Torch 2.14.0a0 build, and four-row projection can select a different
backend from the full vocabulary. Thus no native head arithmetic defect was found
in these four dots; actual reference incoming hidden values and head execution
remain to be captured. Learned operands remain private on the device. The original
six-control finite-bounded oracle and raw receipt are preserved; a follow-up
13-control suite additionally verifies overflow refusal, subnormal ties and
nonfinite-input refusal. No production arithmetic changed.

An isolated hook in the pinned optimized reference subsequently captured its actual
final-normalized input and complete head output for that same 150-token prefix.
The unchanged original head implementation remained behind an explicit armed
wrapper. An unarmed 32-token control reproduced all prior actual IDs, and the
armed one-token control repeated the prior decision. These bounded controls do
not prove universally transparent instrumentation or support timing claims.

The observed reference input, weights and logits were BF16, using
`UnquantizedEmbeddingMethod` on Torch `2.14.0a0+b2c75dd062.nv26.09`.
Its incoming normalized vector differed from native in **2,725/2,880 BF16 values**
(maximum absolute difference 1.25). Both competing reference logits were actually
26.625. All four selected reference head rows matched independent exact rational
dot-product rounding on that captured reference input, just as the native rows
matched on their own input. This localizes these four output differences upstream
of the final head; it does not identify the responsible transformer operation or
explain the entire coding failure. Original scores and qualification gates remain.
The source-bound hook, actual shape/dtype checks, control responses and filtered
comparison are in `gpt/topology-first-divergence/reference-head-hook`; learned
vectors and head rows remain private on the device.

A further compiled-graph diagnostic localized the last matching boundary to the
first layer's initial RMSNorm. For logical position 149 of the same 150-token
prefix, the embedding and initial normalized vector matched native in all 2,880
BF16 values. By the first layer's post-attention output projection, 2,642 values
differed (maximum absolute difference 0.078125); the post-attention norm differed
in 2,114 values and router logits in 24 of 32. This identifies an interval spanning
QKV projection, positional encoding, attention and output projection; it does not
assign the cause to any one of them or explain the entire coding failure.

The reference executed 25 piecewise CUDA graphs with 152 padded tokens. A private
loader hook retained bounded graph buffers and read logical row 149 after replay.
The active capture cohort was selected by its unique exact embedding match;
inactive buffers were retained. Every post-attention norm also matched its
separately captured router input. The hook preserved the original 32 generated IDs,
the one-token decision, all 201,088 final BF16 logits and the final normalized
vector byte for byte. Native snapshots independently preserved the full 150-row
hidden/logit traces. Source insertion controls removed only snapshot calls to
recover the original wrapper AST; embedded kernel source was unchanged. Editing
the loader changed compilation-cache identity, so these are explicit diagnostic
controls, not a claim of universally transparent instrumentation. Two earlier
unsuccessful wrapper-hook attempts remain recorded. Private evidence is under
`gpt/topology-first-divergence/reference-layer-hook4`; original scores, numerical
gates and production arithmetic remain unchanged.
