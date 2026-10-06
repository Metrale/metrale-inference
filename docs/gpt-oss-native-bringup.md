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
No golden instance or complete circuit is added: the remaining semantics below
cannot yet be represented and lowered faithfully by the existing package.

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
no registered circuit mapping, no executed golden plan, and no new measured kernel
points. This foundation removes only the storage-format representation gap.
Existing architecture coverage and performance claims are unchanged.

## Harmony foundation

`crates/server/src/harmony/` now provides a pure, bounded token-event decoder.
It preserves message-end, assistant-turn-end and tool-handoff as distinct typed
values, keeps channel and recipient separate from body text, supports either
header ordering and a prompt-seeded partial header, and rejects malformed or
truncated streams. Errors poison the stream. Seven standalone tests cover these
boundaries, including Unicode chunk splits and undeclared recipients.

This decoder is deliberately not connected to serving yet. Next steps are
checkpoint-specific special-token mapping, preserving developer messages during
prompt rendering, scheduler termination metadata, shared API IR conversion,
JSON/tool-schema validation, and blocking/streaming/Anthropic parity. No native
tool-use result is claimed by these framing tests.
