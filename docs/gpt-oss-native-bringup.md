# GPT-OSS-20B native bring-up

Status: native eager C1 prototype executes the packed checkpoint and bounded
Harmony generation. Default factory admission remains disabled; explicit experimental
C1 admission passes bounded Linux blocking and streaming API checks. Full-model numerical
differences remain unresolved; no broad correctness, performance or tool-use
qualification is claimed. Verified backups are complete. Owner: investor-mvp #43.

## Reproducible inputs

- Checkpoint: `openai/gpt-oss-20b`, revision
  `6cee5e81ee83917806bbde320786a8fb61efebee`.
- Config copied from the verified model backup on Spark 1 into
  `crates/circuit/tests/fixtures/checkpoints/openai--gpt-oss-20b/config.json`.
- SHA-256: `3a2a26ded679375b7928ddeca59764df7cea83220c1961035f6d6e232659e9ce`.
- [Pinned config](https://huggingface.co/openai/gpt-oss-20b/blob/6cee5e81ee83917806bbde320786a8fb61efebee/config.json).
- [Reference implementation, Transformers v4.55.0](https://github.com/huggingface/transformers/blob/v4.55.0/src/transformers/models/gpt_oss/modeling_gpt_oss.py).

The fixture's provenance file records these identifiers without machine-specific
paths or credentials. Any reference execution must also pin its installed
Transformers/kernel versions; a tag URL alone is not an execution receipt.

## Architecture contract

24 layers alternate sliding-window (128) and full causal GQA, each followed by
MoE. Hidden width is 2880; 64 query heads and 8 KV heads use head dimension 64.
The MoE has 32 experts, selects 4, and uses width 2880. Vocabulary is 201088;
the embedding and output head are untied. YaRN factor32 uses original length4096,
base150000, beta32/1, and `truncate=false`.

Q/K/V/O projections and router are biased. Attention has a learned per-head sink
in the softmax denominator. Norm scales multiply before the FP32 result is cast.
Router selection is top-k logits followed by softmax on the selected logits.
The interleaved expert projection splits even gate / odd up channels. With
`g=min(gate,7)` and `u=clamp(up,-7,7)`, activation is
`g*sigmoid(1.702*g)*(u+1)`. Expert down bias is added before the routing-weighted
sum. There are no shared experts, Q/K norm, or attention output gate.

## What this change establishes

The circuit format vocabulary distinguishes `mxfp4/g32` (E2M1 values and one
E8M0 byte per32 values, no global scale) from NVFP4. Byte accounting includes
packed values and E8M0 scales and rejects partial groups/overflow. This is a
storage description, not proof of executable lowering. MXFP4 activation edges
and unimplemented circuit numeric pipelines are refused explicitly. Hardware
reports do not claim NVFP4 emulation or a BF16 native path for these weights.

The real checkpoint remains refused by `resolve_checkpoint` with `gpt_oss` named
in the error. A regression test makes accidental generic-model dispatch visible.
A pure architecture package now represents these semantics explicitly, including
sliding/full layer kinds and MXFP4 precision. It is not registered as a runnable
checkpoint: marked lowering gaps cannot match generic fusion rules. No executable
golden instance or complete native lowering is claimed.

## Explicit architecture and kernel residuals

| Area | Implemented and observed | Still open |
| --- | --- | --- |
| MXFP4 storage | Strict precision/config, 459 bindings, packed expert GEMV and native full forward | Executable circuit lowering and broader numerical qualification |
| Attention | Parameterized online FP32 sink kernel; bounded staged-BF16 diagnostic kernel | Full-model parity, prefill/batching, optimization |
| Bias and activation | Projection bias, expert bias, asymmetric interleaved activation and weighted reduction | Broader checkpoint mixtures and full-model acceptance |
| YaRN | Continuous half-split kernel with large-position CUDA comparisons | Full-model long-context qualification |
| Routing/norm | Selected-logit BF16 scores and plain norm; same-operand probes | Accumulated numerical differences and expert decision sensitivity |
| Harmony | Native blocking and incremental SSE checks, including bounded disconnect recovery | Tools, broader API parity and quality |

Initial comparison candidates are `dense_gqa` for attention structure and golden
`qwen3_6_moe` instances for shared MoE operations. `dense_gqa` itself has executor
residuals; it is not a substitute for a verified runtime comparator. Laguna and
Gemma loaders exist but have no architecture circuit package in this checkout.
Run the new-model skill's Venn only once the target faithfully expresses these
policies; do not publish misleading full-coverage counts before then.

## Next acceptance gates

1. Resolve the observed full-model numerical differences using frozen traces and
   identical-operand probes. Preserve failed candidates and original baselines.
2. Complete executable circuit lowering and explicit native target/factory wiring
   without substituting NVFP4 for MXFP4 or omitting semantic fields.
3. Expand native generation checks to reasoning, tool use, streaming, cancellation,
   longer contexts and concurrent serving; one arithmetic answer is insufficient.
4. Measure sustained prefill/decode, throughput and energy against a pinned
   same-device baseline, then perform required certification on a frozen build.

The new-model skill's LAB/LKB start state for this target is an explicit residual:
a strict pure architecture mapping exists, but there is no registered executable
checkpoint, no executed golden plan, and no new measured kernel points. The
architecture and storage descriptions do not establish native execution.
Existing architecture coverage and performance claims are unchanged.

## Harmony foundation

`crates/server/src/harmony/` now provides a pure, bounded token-event decoder.
It preserves message-end, assistant-turn-end and tool-handoff as distinct typed
values, keeps channel and recipient separate from body text, supports either
header ordering and a prompt-seeded partial header, and rejects malformed or
truncated streams. Errors poison the stream. Tests cover these boundaries,
including Unicode chunk splits and undeclared recipients. The checkpoint tool
header `commentary json` preserves a separate JSON content type; duplicates,
unknown types and JSON metadata without a tool recipient are refused.

Blocking text and incremental UTF-8 streaming pass bounded live API validation.
Streaming retains sampled terminal-token evidence and hides analysis. JSON/tool
schema handling and broader API parity remain open. No native tool-use result
is claimed by these framing tests.

### Checkpoint token identity adapter

`harmony::adapter::TokenMap` reads supplied tokenizer JSON without performing I/O.
It derives IDs from metadata, validates unique vocabulary/added-token identities,
and requires the six exact special-token declarations used by the pinned template:

| Token | Pinned ID | Decoder event |
| --- | --- | --- |
| `<|start|>` | 200006 | Start |
| `<|channel|>` | 200005 | Channel |
| `<|message|>` | 200008 | Separator |
| `<|end|>` | 200007 | Message end |
| `<|return|>` | 200002 | Assistant turn end |
| `<|call|>` | 200012 | Tool handoff |

These are the checkpoint's actual spellings, not aliases inferred from another
Harmony release. Runtime dispatch uses token identity only. Ordinary text that
spells a delimiter stays text. Padding, reserved tokens, unsupported specials and
IDs absent from the tokenizer are errors, not successful completion. The
classifier must precede skip-special-token decoding and scheduler EOS filtering.
`harmony::stream::{ByteTokenizer, Stream}` now translates ordinary token IDs
through the checkpoint ByteLevel byte alphabet, retains incomplete UTF-8 across
tokens, and feeds complete Unicode into the framing decoder. Invalid bytes or
incomplete sequences at a framing boundary/EOF are explicit errors. U+FFFD is
preserved when genuinely encoded, not treated as an incomplete-token heuristic.
This strict adapter is necessary because tokenizers 0.23 DecodeStream uses lossy
decoding and exposes no final flush/pending-byte check. ID classification always
precedes byte decoding; decoded delimiter spellings never become control events.
The adapter now feeds blocking and experimental streaming API composition;
framing tests alone do not establish working HTTP inference.

`fixtures/gpt-oss-byte-vocab.json` is a compact decoder fixture: the original
first 256 byte-vocabulary entries, added tokens and decoder metadata, with no
merges or pre/post processing. It supports byte-level decoder tests, not production
encoding. Tests compare valid Unicode against the real tokenizers ByteLevel
decoder and demonstrate its lossy behavior on an incomplete byte sequence as a
known-bad control. The strict adapter rejects that sequence rather than reporting
a successful completion.

The fixture at `crates/server/src/harmony/fixtures/gpt-oss-token-metadata.json`
contains all 21 added-token records and only the two boundary ordinary vocabulary
entries, selected without modification from the pinned tokenizer. It is a compact
metadata test fixture, not a usable tokenizer. The complete source tokenizer's
SHA-256 is `0614fe83cadab421296e664e1f48f4261fa8fef6e03e63bb75c20f38e37d07d3`.
An independent adapter run against that complete file classified 199,998 ordinary
IDs and six framing IDs; across the 201,088 model logits, 1,084 unsupported or
unassigned IDs were refused. This validates metadata handling only, not inference.

## Packed expert bindings and architecture package

`metrale_circuit::gpt_oss::architecture` constructs the pinned 24-layer graph with
48 KV states, biased projections, denominator-only sinks, YaRN policy and the
interleaved clipped expert activation. Required math fields are checked; missing
and changed policies are refused. An `unlowered` node cannot match generic fusion
rules. Three focused controls validate the graph, config mutations and refusal.

`PackedMxfp4Experts` binds exact U8 rank4 blocks and rank3 E8M0 scales, validates
expert extents and pointer arithmetic, and exposes typed per-expert views without
copying or transcoding. Five controls cover checkpoint dimensions, actual byte
readback through the mock GPU backend, malformed layouts and overflow. This is
host binding evidence, not GPU arithmetic. The loader still needs to use these
views in an actual model with matching kernels, biases and scale policy.

## Runtime config parsing

The runtime now parses the pinned GPT-OSS-20B config with explicit typed policies
for selected-logit routing, sink attention, interleaved asymmetric SwiGLU, biased
expert reduction, FP32 normalization and nontruncated YaRN. Missing, changed and
unknown math keys fail closed; conflicting expert-count aliases are refused.
The precision plan identifies MXFP4 experts and preserves the excluded BF16 head.
All 193 config tests passed, including the new policy and mutation controls.
The default model factory still refuses GPT-OSS. A separate explicit experimental
policy now admits the native loader; see the serving-admission section below.
Parsing a config alone is not loading or serving the model.

## Complete checkpoint binding

`GptOssCheckpoint::bind` now validates and borrows all 459 tensors: embedding and
untied head, final norm, and every layer's norms, biased attention projections,
sinks, biased router, packed experts and expert biases. It retains the typed
config policy and exact BF16/U8 storage. Missing or extra tensors, wrong shapes
and dtypes, null pointers and overflowing extents are errors. No implicit casts
or quantization substitutions occur.

The fixture matches the captured pinned safetensors headers. Four host tests
exercise valid binding, each missing tensor, every shape/dtype mutation and
boundary controls, using inert addresses. These tests do not read learned values
or execute a GPU. Native loader/layer construction and the standalone forward
path are now implemented and exercised below; experimental API validation remains open.

## Projection bias GPU primitive

The `projection_bias_bf16` CUDA epilogue consumes FP32 projection accumulators,
adds the checkpoint's BF16 bias in FP32, and rounds once to BF16. It is intended
for Q/K/V/O/router projections after the existing `dense_gemv_bf16_fp32out`.
Expert `bmm` plus bias has a different intermediate rounding contract and is
not assigned this primitive. The Rust launcher rejects empty or overflowing
geometry, null or misaligned addresses, address overflow and overlapping output.

The standalone CUDA harness at
`crates/model-layers/tests/cuda/projection_bias.cu` passed on GB10 with CUDA 13,
`--fmad=false -arch=sm_121`: all 777 outputs matched an independent integer-bit
round-to-nearest-even oracle. The corpus includes odd column counts, row bias
broadcasting, tail threads, signed zero, subnormals, cancellation, infinities and
NaNs. A known-bad early accumulator rounding control differs on 180 elements.
Optional JSON output retains operand and result bits for independent replay.
Two host launcher tests also passed; they establish ABI/refusal behavior only.

This is a named LKB residual, `projection_bias_before_cast`, with primitive GPU
parity. The common source is inherited by GB10, Hopper and B200; the addition
changes no existing kernel entry point or caller. There is no GPT kernel target,
forward-path linkage, full `F.linear` parity or measured performance yet.

## Denominator-only sink attention primitive

The existing BF16 paged decode attention body is parameterized by sink policy.
The original entry point and ABI retain the no-sink policy; a separate entry
accepts BF16 per-query-head sink logits. It adds sink mass once after the global
warp merge, rescales stably for large logits, and contributes no value vector.
Negative infinity is exactly a no-op. Nonempty sequences with NaN or positive
infinite sinks produce NaNs; empty sequences produce zero. Sliding-window and
GQA/cache indexing stay shared. The new Rust launcher currently requires HD64.

A standalone CUDA harness compared the old source with the parameterized legacy
entry on GB10: bit identity passed at HD64/128/192/256/512, C1/C16/C128, lengths
0/1/7/128/129 and full/window128 attention. HD64 alone compared 5,939,200 values.
Sink output passed an independent dense oracle with absolute tolerance 0.04;
constructed cases include per-head sinks, KV-head-dependent values, nonfinite
sinks, large positive logits and empty/window-boundary cases. HD64 detects 43,980
wrong outputs when sink mass is added once per warp and 80,182 when omitted.
Raw C1 operands, layout and output/control bits can be emitted for offline replay.
Two host tests cover the wrapper's argument layout and geometry refusals.

These tested cases support the stated FP32 online-softmax contract, not equivalence to
Transformers' staged BF16 eager attention. Later sections record native composition;
full-reference intermediate parity, API validation and performance remain open. The shared file
also reaches Hopper, B200, B300 and Strix variants; their compile/performance gates
and existing-model certification remain required before merge.

## Packed expert GEMV primitive

The row-major MXFP4 correctness residual consumes the validated packed expert view
and emits BF16 before separate bias, activation and routing operations. It
preserves low-nibble/even-column order, group32 exponent addressing and the pinned
unpack behavior. No conversion to the existing transposed NVFP4 format occurs.
The Rust wrapper checks pointer geometry/aliasing and has two passing host tests.

On GB10, all 4,096 nibble/scale pairs passed the independent unpack oracle.
Constructed 35-row K96/K2880 cases and a BF16-before-bias control passed.
Actual layer0/expert0 gate/up and down slices (35 rows each, K2880) matched
independent double-sum/BF16 results and Torch CUDA BF16 bmm exactly. The declared
actual-slice gate allowed at most one BF16 ULP; no gate was widened. Wrong nibble
order, group16 addressing and transposed layout controls were detected.

Torch CPU BF16 bmm differed by up to eight/four ULP on these gate/down slices;
that discrepancy is retained. CPU FP32 then BF16 matched. Reference backend and
precision must therefore stay explicit. The actual checkpoint scale census
covered 48 tensors / 597,196,800 bytes, all between 115 and 136, with no 0/255.
Constructed boundary tests still cover those bytes' pinned unpack behavior.

This GEMV measurement is primitive evidence, not a full-model qualification.
Later sections record its composition with YaRN, routing, bias, activation and
reduction into native forward execution. No throughput or energy improvement is
claimed. Circuit routing remains explicitly unlowered; factory/API integration
and full-model qualification are open.

## Staged arithmetic and routing (October 6 continuation)

Continuous half-split YaRN now matches the pinned Transformers v4.55 CUDA
reference exactly for all 32 frequencies and 36,864 BF16 Q/K values at positions
0, 1, 127, 128, 4095, 4096, 8192 and 131071. Truncating correction bounds changes
6,777 outputs; adjacent-pair rotation changes 25,263. Both errors are detected.

Expert post-bmm bias reuses `nllb_bias_bf16`; the new asymmetric interleaved
SwiGLU and selected-expert reduction preserve BF16 operation boundaries. Exact
CUDA reference comparison passed 777 bias outputs, 65,795 activation outputs
(including all BF16 gate encodings), and 777 reduction outputs. Known-bad
fused rounding, symmetric clamping and unrounded weighted products fail.
Sparse reduction assumes finite unselected expert outputs; it does not mimic
dense NaN-times-zero contamination.

The shared top-k kernel now has a distinct selected-logit BF16 policy returning
dense 32-expert scores and four IDs. Existing callers retain their FP32 policy
and ABI. On 515 constructed CUDA rows, legacy outputs are bit-identical and the
new selected sets and BF16 scores exactly match Torch CUDA. The declared score
gate was one BF16 ULP; observed maximum was zero. A wrong full-expert softmax
without selected renormalization differs at 489 positions. Finite logits are
required. Ties use lower expert ID; no universal reference tie-order claim is
made. Two host admission/ABI tests pass.

The primitives are now composed in an explicit eager C1 layer/loader and standalone
full-forward example. Default factory admission remains disabled. On GB10, the native packed
checkpoint completed all 24 layers for diagnostic token IDs `[1,2,3,4]`, producing
finite logits and saved per-layer BF16 traces. Checkpoint load took 31.805 seconds;
the four passes took 36–44 milliseconds each including trace copies. These are
short debug-harness observations, not serving throughput or speed qualification.
The pinned Transformers 4.55.0 eager reference agrees on all four next-token IDs
(`[326,1981,4,5]`), but the traces are not bit-identical. Per-position maximum
absolute logit differences are 0.0625, 0.3046875, 0.203125 and 0.21875; RMS
differences are 0.01797, 0.05319, 0.03588 and 0.06437. Differences start in
layer 0. This comparison is diagnostic, with no full-model acceptance inferred.
The reference explicitly dequantizes MXFP4 to BF16; native weights stay packed.
Attention precision and longer-context behavior remain under investigation.

Follow-up review fixes now choose
the first tied maximum and reject missing, rewound or changed KV prefixes; a
failed execution poisons its state. Duplicate physical cache blocks are refused.
Linux tests cover these admissions and
argmax behavior. Original sequential teacher-forced traces are retained. Quality,
Harmony serving, concurrency and certification remain open.


## Native Harmony generation and longer-context diagnostics

The bounded greedy example now accepts an explicit maximum generation length and
stop-token IDs. A pinned-tokenizer, low-reasoning arithmetic prompt produced 16
native tokens: an analysis message followed by final answer `4`, terminated by
`<|return|>` (200002). The stop token is recorded and not fed back into the cache.
This is actual packed-checkpoint native execution, but one elementary answer does
not establish broader reasoning, tool use or API compatibility.

A 251-token teacher-forced Harmony fixture completed with finite outputs across
block and sliding-window boundaries. Against the pinned eager BF16 reference,
244 of 251 next-token choices match. Differences occur at positions 47, 48, 58,
164, 180, 231 and 249; three reference maxima are tied. Numerical differences
remain material: position 215 has logit normalized RMS 0.2694 and maximum absolute
difference 7.787. Positions 49 and 215 first show a large layer-level divergence
at layer 9. Boundary positions 127–130 do not show an abrupt error increase.
These observations are diagnostic, not a correctness acceptance.

The original pinned BF16 reference is preserved. A separately labeled FP32
attention ablation and selected intermediate traces are being used to locate
the differences. No tolerance has been widened and no serving-speed or quality
qualification is inferred from the debug harness timings.


## Attention precision investigation and rejected full-model candidate

On identical native layer-9 Q/K/V operands at positions 49 and 215, the online
attention output differs from pinned BF16 eager attention in 2,124 and 2,230
values. It differs from FP32 eager attention in zero and one value respectively.
The same-input norm, projections and routing are otherwise nearly bit-exact.

An explicit staged-BF16 attention residual reproduces QK rounding, scaling,
max subtraction, softmax probability rounding and AV output rounding. Its exact
GPU gate passes ten captured/constructed/boundary cases, with missing-sink,
window and precision controls. The correctness-only policy is limited to HD64,
finite Q/K/V and at most 4096 tokens. Masked nonfinite operands are outside its
reference-equivalence claim. Existing online attention is unchanged.

A separate, source-hash-bound PTX package tested this policy through all 251
tokens. Full-model agreement worsened: 243/251 next-token matches versus 244,
mean normalized logit RMS 0.02651 versus 0.02339, and maximum 0.27724 versus
0.26940. The candidate is **not promoted**. Exact sampled primitive agreement
does not establish full-model agreement; expert decisions and accumulated
rounding remain under investigation. All original, ablation and candidate
traces are retained. Diagnostic snapshot reruns reproduced their baseline
logits/layer traces byte-for-byte, with KV export layout controls passing.

## Routing and independent arithmetic checks

On identical captured logits, Torch and native selected the same experts in all
144 sampled routing rows. Six same-input router projection replays were also
bit-exact. Observed routing changes arise from different upstream hidden states
crossing narrow score gaps; these checks do not establish a selector defect.

Full-row selected-expert replay reproduced all eight captured native outputs.
The four experts at staged position 215/layer 6 matched the BF16 reference. At
position 49/layer 9, expert 18 differed in one gate projection value, propagating
into 131 expert-output values. B4 and B32 reference geometries agreed, and the
isolated bias, activation and down-projection checks matched.

An independent rational dot-product oracle resolves that differing gate row:
the exact sum is `-3350529/16777216`. Native returns the correctly rounded BF16
value `-0.2001953125` (bits 48717); the CUDA reference returns `-0.19921875`
(bits 48716). The emulated native FP32 accumulation equals the exact sum for
this row. Reference internal accumulation is not exposed. The exact-reference
gate remains recorded as failed, but changing this correct native result to
match the reference is not justified. This one-row finding neither qualifies
the full model nor resolves its remaining output differences.

## Additional bounded native generation checks

A three-case greedy smoke run used the frozen diagnostic binary and original
online-attention package. Multiplication (`17 * 6` → `102`) and identifier
extraction (`ZX-204`) passed exact output checks. Numeric sorting returned the
correct sequence but surrounded the requested JSON array with Markdown fences;
that case fails the strict format check. All three outputs contained a Harmony
final channel and terminated with return token 200002. The aggregate is **2/3**,
not a clean pass. The pinned eager BF16 reference produces the same three final
outputs, including the fenced-JSON failure. Multiplication and sorting have
identical complete generated token sequences; extraction differs only in its
analysis wording. This small comparison does not resolve longer-context
numerical differences or establish broad quality. Load and trace I/O remain in
the native harness timings, which are not serving performance measurements.


## Explicit experimental serving admission

An optional `MODEL.model.supported_quants` list restricts the new GPT target to
MXFP4. Explicit incompatible selections fail; wildcard model builds skip only
declared-incompatible combinations. Existing model manifests without an allowlist
retain their previous resolution. Rust/Python resolver agreement and structure
checks pass, and the HD64 leaf compiles to PTX on Spark1.

`--experimental-gpt-oss` explicitly selects the experimental loader policy;
default factory calls still refuse it. Admission requires C1, BF16 KV/head,
single-device execution, memory utilization in `(0, 0.85]`, and no prefix reuse,
swap, speculative decoding, LoRA or batched prefill. Circuit lowering remains
unsupported. Layer capabilities independently refuse graphs and multi-sequence
execution. Factory/CLI controls and a Metal-feature server check pass. The full
Linux target/server build succeeds. Bounded HTTP/SSE and disconnect-recovery
checks pass as described below. This admission is not certification.


## Server integration boundaries

Actual startup exposed three gaps that standalone forward tests could not:
missing MXFP4 quant-format preflight, a tokenizer cap conflicting with physical
checkpoint vocabulary, and an unreachable NVFP4 head-kernel probe under explicit
BF16 head policy. Their failed startup logs are retained. The format validator
now reuses packed tensor contracts; its legacy NVFP4 mapping is explicitly absent.
The parsed policy retains 201,088 physical embedding/head rows while sampling,
projection output and logits allocation use the 200,019-token logical cap.
Invalid input IDs are refused before embedding/state mutation. The unused head
probe is policy-gated rather than hidden from the kernel audit.

Blocking Harmony returns only validated final text. API usage categorizes
ordinary token IDs in analysis bodies as reasoning tokens, excluding framing
headers/delimiters. Total completion tokens remain all sampled output IDs;
subtracting reasoning therefore still includes protocol overhead, not just
visible-text tokens. This is an engine accounting convention, not provider
billing parity. Harmony now disables the unrelated generic scheduler thinking
budget; explicit unsupported budget/loop requests are refused. Checkpoint
reasoning-effort hints remain independent. This newer policy fix is not part of
the frozen SSE binary below.

## First native HTTP lifecycle result

A frozen Linux build passed all eight bounded HTTP checks: multiplication,
identifier extraction, early-identifier recall across a 420-token prompt,
the same short response before and after refused requests, and explicit HTTP400
refusals for streaming, schema and tools in that blocking-only build. Successful
responses contain final text without analysis or framing, stop normally, and
conserve reported total usage. Eleven independent grader controls include wrong
answers, framing/analysis leakage, truncation, invalid usage and unrelated errors.

The 420-token case crosses both block and sliding-window boundaries. These
sequential checks exercise state reuse but do not qualify concurrency, long
contexts generally or sustained stability. The frozen build excludes the later
reasoning-accounting and streaming changes. Its binary SHA-256 is
`d854034d586b0d10a8a100a7aa5251a9d57a6e45529a132f9e1ecea7d7367f60`.
Client times were 2.125–2.482 seconds for the short successful requests and
9.499 seconds for the 420-token request; this debug-build smoke is not a
performance or energy certification. Broader quality and optimization remain open.


## Native streaming lifecycle result

A separate frozen streaming build passed seven live cases and sixteen grader
controls. Literal Unicode text arrived in four content chunks over 79 ms, with
stop 106 ms after first visible content. Unsupported thinking budgets, disabling
thinking, custom loops and raw-token options returned explicit HTTP400 errors.
Disconnecting after the first visible content ended generation at 24 of the
256-token limit; the retained lifecycle log records client departure and the next
request successfully returned ORBIT. This is bounded cancellation evidence, not
a sustained concurrency qualification.

The binary SHA-256 is
`ee3f61515173f4a576c0c59ca02a8b74db163506229d5842bede56178bebae0c`.
Its captured source overlays and admission patch are preserved separately.
Reported analysis-body usage was 5/19 completion IDs for the Unicode case and
7/19 for ORBIT. Client first-visible latency is separate from server TTFT, which
measures the first generated token and can include hidden analysis.

A frozen release build at `43598c8` subsequently passed nine diagnostic response
checks (three repetitions each). Median decode was 37.3–37.8 tokens/second.
Prefill was roughly 47 tokens/second; median first-generated TTFT for 81-, 87-
and 304-token prompts was 1.740, 1.868 and 6.499 seconds. Corresponding client
first-visible medians were 2.223, 2.295 and 7.036 seconds. These bounded results
exclude certification and energy qualification; profiling and optimization
remain in progress.

### Blocking tool roundtrip qualification (2026-10-07)

The controlled localhost C1 gate passed **9/9 cases**, backed by **18 independent
probe-grader controls**. Both Chat Completions and Anthropic Messages emitted an
exact `lookup_part({"part_id":"A-42"})` call and answered `7` after a fixed local
stock-result fixture. No requested external tool was executed. Wrong result IDs,
duplicate argument keys, unsupported schema keywords, streaming tools and
Responses tools were refused. The first run failed closed on checkpoint token
200003 (`<|constrain|>`); the corrected strict format-header parser passed the
second run. No reserved-token wildcard was added.

Tested debug binary SHA-256:
`c32201291c7729c9b1ffdf675fcd67c94b6ef9ba7030178ed00c13d7ad679ae8`.
The tested source archive, exact overlay/file hashes, requests/responses, failed
first run and private controlled diagnostic are retained with the integration
receipts. This qualifies the narrow blocking protocol path, not general tool
quality or optimized throughput. Allowed schemas are explicitly bounded;
unsupported JSON Schema features fail admission. Calls require an exact declared
recipient, unique JSON keys, valid typed arguments and a completed handoff.
Analysis-channel handoffs remain unsupported. API analysis-token accounting is
separate from scheduler thought budgets and provider billing conventions.

### Chat Completions tool SSE gate (2026-10-07)

The subsequent Chat Completions tool-stream gate passed **5/5 cases** with **16
independent grader controls**: declared tool call, Unicode arguments (`café 日本
😀`), truncated generation producing an error with no tool call, disconnect
before a call and a valid subsequent request. Separate live checks confirmed
Anthropic and Responses tool-stream requests remain HTTP 400. CPU Harmony tests
passed 37 cases; scoped server Clippy passed.

Tool arguments are intentionally buffered. Only after the handoff token, strict
JSON parsing and schema validation succeed does the adapter emit a call-start
delta, a complete UTF-8 argument delta and `finish_reason: tool_calls`. Final text
continues to stream incrementally. There is no incremental tool-argument latency
claim and no external tool execution. The cancellation trace recorded receiver
closure, two generated tokens before stopping, then successful request reuse;
cancellation does not preempt an already running prefill kernel.

The first gate exposed invalid raw-text SSE error data on truncation. That failure
and raw response were retained; the corrected adapter reuses the shared JSON error
envelope and rate-limit refund handler. Tested debug binary SHA-256:
`11873e8c8d6ad21b8519e277c22167cbf24143388e2935f4192df55d44f45267`.
Exact tested source hashes, event timing, usage and failure receipts are retained
privately. This is a bounded protocol/cancellation qualification, not broader
model-quality, throughput, billing or multi-request concurrency qualification.
