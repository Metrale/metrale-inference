# GPT-OSS-20B native bring-up

Status: architecture foundation only. No native serving, correctness, performance,
or tool-use pass is claimed. The downloaded checkpoint and backup are separate
from engine qualification. Owner: investor-mvp issue #43.

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

| Residual | Existing reusable work | Required evidence / change |
| --- | --- | --- |
| MXFP4 precision declaration | Native rank2 E8M0 pointer helper in model-layers | Preserve scale encoding through declared precision; never infer MXFP4 from group32 alone. Validate packed expert block views and shard binding. |
| Attention sinks at HD64 | DeepSeek HD512 sink attention | Generic GQA decode/prefill, sliding/full and batching; sink affects denominator only. |
| Biased attention and experts | dense_gqa circuit describes QKV bias | O/router/expert bias bindings and lowering; weighted down bias included. |
| Nonstandard gated activation | Shared SiLU families | Explicit alpha, clamp and offset policy plus exact interleaved layout; no plain-SiLU substitution. |
| YaRN | Existing rope/table utilities | Half-split positions, continuous correction boundaries with truncate=false, scale parity. |
| Routing and norm precision | Shared routing/RMS kernels | Selected-logit softmax and FP32 scale-before-cast reference parity. |
| Harmony serving | Existing request/streaming/parser framework | Separate channel/recipient boundaries, tool handoff, turn completion, round trips and cancellation. |

Initial comparison candidates are `dense_gqa` for attention structure and golden
`qwen3_6_moe` instances for shared MoE operations. `dense_gqa` itself has executor
residuals; it is not a substitute for a verified runtime comparator. Laguna and
Gemma loaders exist but have no architecture circuit package in this checkout.
Run the new-model skill's Venn only once the target faithfully expresses these
policies; do not publish misleading full-coverage counts before then.

## Next acceptance gates

1. Extend circuit/config mapping and numeric pipeline vocabulary without silently
   dropping any listed semantic field; declare all remaining lowering gaps.
2. Test primitive parity: sink logits and zero-value mass, window128/129,
   large/negative gate inputs, biased expert mixtures, MXFP4 extreme codes/scales,
   and YaRN beyond4096. Test existing parameter points for regressions.
3. Load exact checkpoint, compare layer outputs and logits against a pinned
   reference, with numerical tolerances stated before execution.
4. Validate native text generation and Harmony API/tool behavior before a soak.
5. Record native runtime/commit/model identity and real integration outcomes;
   download completion and reference-engine output do not pass this gate.

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

This decoder is deliberately not connected to serving yet. Next steps are
preserving developer messages during prompt rendering, scheduler termination
metadata, shared API IR conversion,
JSON/tool-schema validation, and blocking/streaming/Anthropic parity. No native
tool-use result is claimed by these framing tests.

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
It remains a tested foundation, not wired into serving or scheduler termination.

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
The model factory still refuses GPT-OSS until native weight assembly and matching
forward kernels exist. Parsing a config is not loading or serving the model.

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
or execute a GPU. ModelWeightLoader construction, matching kernels and the actual
forward path are still absent; the factory remains intentionally unsupported.

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
Transformers' staged BF16 eager attention. Native loader/layer/factory linkage,
full-reference intermediate parity and performance remain open. The shared file
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

This is an unoptimized GEMV primitive, not the complete routed MoE or native
model. No throughput or energy improvement is claimed. Attention reference
precision, YaRN, selected-logit routing, expert bias/activation/reduction and the
full loader/layer/factory/serving integration remain open. Routing weights in the
circuit retain the reference BF16 output precision and remain explicitly unlowered.

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
full-forward example; the factory remains unregistered. On GB10, the native packed
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
