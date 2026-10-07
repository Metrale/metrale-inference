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
