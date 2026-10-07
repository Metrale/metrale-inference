# GPT expert-plan validation amortization proposal (2026-10-07)

**Design only: no implementation, GPU run, speed claim, or admission change.** The next candidate would validate each immutable expert plan once on device and reuse that result for gate/up and down projections. It must preserve existing refusal and poisoning behavior. It must not replace validation with a caller-controlled “trusted” flag.

The accepted release remains `829dbf3a9b048dbc4370cc126d5c6f0d250e77f7e47500869b67c81d2313086f`, with source arithmetic from the [accepted fixed16 build](gpt-oss-fixed16-projection.md), recorded in commit `91d703f517b8151e0e0f5dffd1a7d59c1a247412`. The exact model is [openai/gpt-oss-20b@6cee5e81ee83917806bbde320786a8fb61efebee](https://huggingface.co/openai/gpt-oss-20b/tree/6cee5e81ee83917806bbde320786a8fb61efebee). Baseline expert-kernel source SHA-256 is `a48f910c4f3ed86acead23e1ed4bd404849fc6feadbefc2051a7087ddc28beb5`. Rejected [device-plan](gpt-oss-device-plan-screen.md) and [scheduling/unroll](gpt-oss-expert-axis-screen.md) experiments are separate; none is a new baseline.

## Current contract

`runtime/expert_plan.rs::ExpertTokenPlan` accepts 1–128 token rows with four distinct expert IDs per token, each below 32. It constructs 32 lists with stride `tokens + 1`: a count followed by encoded entries `token * 4 + slot`. Counts cannot exceed the token count. Validation requires every encoded entry to be in range, globally unique, associated with its listed expert, and present exactly once across all lists. Shared experts across different tokens are valid. Missing slots are refused by this host validation.

`runtime/prefill.rs` synchronously reads router IDs, constructs this typed plan, and uploads its bytes before expert launches. This proposal retains that path and its synchronous refusal boundary. It does not reuse the rejected deferred-ID-validation policy.

For token counts above 16, `gpt_oss_mxfp4_reuse_body<true>` repeats per-expert checks in every active row/group CTA: bounded count, bounded entry before ID access, expert-ID agreement, and no duplicate entry within that list. All threads reach the shared-flag barriers before row-tail exits. Valid inactive groups return uniformly. Invalid counts still reach the poison path. On a bad expert list, lane zero of group zero writes NaNs to that expert's selected output slots for each valid output row; other experts retain their existing behavior. The raw CUDA path does **not** prove complete slot coverage: a missing entry can leave output unwritten, which is why the host coverage contract must remain.

The token-16 noncooperative path, smaller token-grid tails, decode, packed-TC diagnostic, bias stages and ascending-expert reduction remain outside the first candidate. FMA column order, warp reductions and BF16 store boundaries must stay unchanged.

## Proposed status ownership and ordering

Use a private prepared-plan handle produced only by the plan-upload/validation wrapper, bound to the same scratch allocation, immutable ID/plan buffers, token count, stream and generation. Consumers must not accept an arbitrary status pointer or caller-supplied validity assertion. Checked wrappers must validate status extent/alignment and non-aliasing with plan, IDs, inputs and outputs. The handle remains logically live through both projections; the next chunk may overwrite these buffers only after both consumers have been queued on the same bound stream.

A provisional status record is four `u32` words per expert: generation low/high, validated count, and verdict (`0` pending/uninitialized, `1` valid, `2` invalid; all other values refused). A checked host `u64` generation starts at one and advances for every plan upload. Zero is reserved for uninitialized status; overflow refuses further preparation rather than wrapping. Scratch status is initialized to zero on its bound work stream before first use. A separate validator launch must overwrite **all 32 records**, including empty and invalid experts, for each generation. There is no early return that leaves an old verdict live.

The sequence is:

1. Keep the existing host ID/coverage checks and upload the immutable plan.
2. Launch one bounded validator CTA per expert on the scratch's bound stream. Check count before list reads, entry before ID reads, expert agreement and duplicates; publish a complete status record for empty, valid or invalid lists.
3. Launch gate/up with the prepared-plan handle, then existing bias/activation, then down with the same handle. Same-stream kernel completion orders status writes before either consumer; no new D2H read or inter-stream event is proposed.
4. Each consumer verifies generation, verdict and validated count against the bounded current plan count before addressing weights or inputs. Unknown/pending/stale verdicts fail closed into the existing poison path. These checks must precede the current valid-inactive-group early return, including when the current count is zero. Preserve group-zero/lane-zero poison ownership and row-tail bounds before any output address. Retain cheap bounds and expert checks for the selected entries before their input addresses are formed. Amortize the complete-list duplicate scan and its repeated CTA barriers, not all memory-safety checks.

This is an ownership/ordering contract, not a cryptographic claim about arbitrary external device writes. A matching generation alone cannot prove that another writer did not mutate entries after validation. The private handle, same-stream sequencing, exclusive scratch use and refusal of cross-stream reuse are therefore required, not optional. Dropping the host handle does not complete asynchronous GPU work: it must not permit another stream to reuse the buffers or free them before bound-stream drain. Same-stream queued overwrite is allowed only after both consumers are queued. A validator launch error prevents consumer dispatch; normal state poisoning and bound-stream teardown remain in force. A raw malformed-plan diagnostic must retain per-expert NaN behavior without claiming that NaNs alone constitute a host-level `Result` error.

## Allocation and integration

The proposed GPU status allocation is exactly `32 * 16 = 512` bytes per layer, already a multiple of the scratch allocator's 16-byte alignment. The host generation is not a GPU allocation. Add status to `PrefillScratch::sizes`; `required_bytes` must remain the allocator's single source of truth. The factory's pre-KV scratch reserve must therefore increase by `24 * 512 = 12,288` bytes for C1 chunk admission, with checked arithmetic and the existing configured/actual-free memory limits. A scalar path with no chunk scratch pays no reserve.

At 8,192 maximum pages, illustrative scratch totals would be:

| Chunk capacity | Current bytes/layer | Proposed bytes/layer |
|---|---:|---:|
| 16 | 2,837,120 | 2,837,632 |
| 64 | 11,348,096 | 11,348,608 |
| 128 | 22,696,064 | 22,696,576 |

These are layout calculations, not allocations performed by this note. Test actual allocation-ledger equality at multiple capacities and non-default page sizes. Missing kernels, allocation failure, generation overflow, cancellation and release must preserve the current failure/drain contract. Do not silently fall back after a failed validator launch.

Integration points are `prefill.rs` after typed-plan upload, `prefill_experts.rs` for both consumers, `prefill_scratch.rs` for storage/stream lifetime, and the checked model-layer wrapper. Parameterize the existing reuse family and retain original entries unchanged; add a separately named diagnostic family point plus a validator entry before considering serving selection. Update target declarations, taxonomy/inventory, source-closure/build dependencies and cross-hardware checks if implementation reaches that stage. Preserve C1, BF16 KV/head, explicit chunk admission, no prefix adoption/swap/graphs/TP and the 0.85 memory limit.

## Required controls and acceptance

Before GPU timing, test rows 1/2/3/4/5/15/16/17/31/64/127/128; hot and balanced experts; empty experts; repeated experts across tokens; distinct gate/down inputs; shuffled token slots; reversed expert lists; N35 row tails; nonzero chunk positions; and legacy small-shape refusal/unchanged behavior. Compare valid outputs bitwise to the accepted kernel and retain the constructed rational-dot controls.

Malformed controls must cover oversized count, out-of-range entry, wrong expert, duplicate list entry, duplicate expert within a token, invalid router ID, and omitted slot. The omitted-slot control must fail at the retained host coverage check; do not label an unchanged raw-kernel omission as detected. Initialize output sentinels so unwritten rows are visible. Verify affected-expert poison output and unaffected-expert equality for raw device-invalid lists.

Freshness controls must include skipped validator, previous-generation status, uninitialized status, pending/invalid verdict (including count zero while IDs still select that expert), changed count/token geometry, changed plan after preparation, wrong scratch and cross-stream use. The supported wrapper must refuse mutation/rebinding while its handle is live; an explicitly corrupted raw buffer is a separate negative test, not a supported API. Exercise repeated chunks, generation overflow, validator launch failure before consumer launches, missing symbols, allocation rollback, cancellation and retry after poisoned state.

Predeclare the performance campaign before execution. First screen validator **plus both projection consumers**, including status initialization/upload and launch costs, against the original accepted path on the same saved router distributions. Require at least 2% geometric latency reduction in both alternating-order cohorts and no case over 2% regression; retain register/stack/spill receipts. A passing micro screen only admits further testing.

Then require all twelve full251 hidden/cache hashes and following decode to match the accepted baseline, unchanged original numerical qualification reporting, and existing text/tool/SSE/cancellation/boundary gates. Run isolated opposite-order whole-request sessions with exact raw input IDs at 64/128/256/512/1024 tokens and exactly one generated token, one warmup plus five measured repetitions per length. Require at least 2% geometric total-latency reduction over 256/512/1024 in **both** sessions, no length above 2% regression, identical counts/text and separately verified actual sampled IDs. Existing 81/20, 87/63 and 304/24 chat cases must retain quality/counts and no median total/TTFT regression above 2%. HTTP measurements include head, sampling and protocol; they are not pure prefill. No default change or competitive/certification claim follows from a primitive result.

## Why investigate, without a speed prediction

For saved layer-zero 128-token routing, the existing gate projection's source loops imply 204,480 active validation CTAs and 43,336,800 duplicate comparisons, versus 4,823 comparisons if each expert list were scanned once. Down projection gives 102,240 CTAs and 21,668,400 comparisons for the same lists. These are **static source-operation estimates**, not measured GPU instructions, memory traffic, occupancy or saved time. Compiler transformations, cache behavior, barriers and the extra validator launch can invalidate a speed prediction. The recent failed scheduling and unroll screens demonstrate why the complete measured gates remain necessary.

The private estimate binds router-ID SHA-256 `244b84502ab0f1f7f1d6c48f87120adc70475d202c60d834fc1ccd5b1f97aac3`. No model weights or private routing payloads are published here. No GPU work was performed for this design note.
