# GPT-OSS device-plan screen — 2026-10-07

The device-plan candidate was rejected. It preserved the accepted arithmetic and
all tested outputs, but increased long-prompt latency in both execution orders.
The accepted release and host plan path remain unchanged.

The checkpoint is
[openai/gpt-oss-20b@6cee5e81ee83917806bbde320786a8fb61efebee](https://huggingface.co/openai/gpt-oss-20b/tree/6cee5e81ee83917806bbde320786a8fb61efebee).
Both arms used Spark1/GB10, C1, BF16 KV/head, packed MXFP4 weights, localhost,
context 2048 and memory utilization 0.85. Prefix reuse, swapping and graphs stayed
disabled. This is a bounded optimization experiment, not certification.

The private candidate built the deterministic token/slot expert plan on device.
It checked a sticky error flag before each layer's prefill returned, replacing
per-chunk ID readbacks. For a single 1024-token layer call at capacity 128, that
reduces the routing readback boundaries from eight to one; it does not eliminate
all model readbacks. Logged scheduler chunk lengths confirmed whole 1024-token
calls. Invalid IDs or duplicate selections poisoned the plan and caused an error
before the layer returned. The additional 16 bytes per layer were included in the
same allocation-derived reserve used before KV sizing.

Validation passed 36 constructed plan cases, including both expert input layouts,
invalid and duplicate IDs, and sticky failures. All 12 hidden/KV files from the
251-token fixture plus following decode matched the immutable accepted files
bytewise across scalar and chunk widths 16/31/64/127/128. Eleven Linux runtime
tests passed, including an injected second-chunk failure, one completion readback,
poisoned state, retry refusal, and bound-stream cleanup. Two earlier test-harness
failures were retained: aliased mock kernel handles and a wrong mock copy counter.
The existing text, blocking-tool, tool-streaming, lifecycle and chunk-boundary API
gates also passed.

Unprofiled A→B and B→A sessions used identical raw prompt IDs, exactly one output
token, one warmup and five measured repetitions per length. These HTTP latencies
include prompt processing, head projection, sampling and protocol; they are not
pure prefill kernel durations.

| Prompt tokens | Session 1 accepted / candidate ms | Session 2 accepted / candidate ms |
| --- | ---: | ---: |
| 64 | 339.63 / 361.90 | 345.85 / 364.91 |
| 128 | 622.41 / 651.22 | 629.80 / 652.10 |
| 256 | 1225.03 / 1250.85 | 1225.36 / 1244.56 |
| 512 | 2422.32 / 2449.74 | 2418.12 / 2437.32 |
| 1024 | 4845.21 / 4881.30 | 4830.48 / 4850.94 |

The predeclared primary geometric reduction over 256/512/1024 tokens was
**−1.326% and −0.927%**, failing the required 2% improvement in both sessions.
The shorter cases regressed too. Raw API counts/text matched; actual generated
IDs are not exposed by that API. A separate audit correlated all 120 existing
scheduler first-token log records with request times, order and exact chunk
lengths, establishing identical sampled first-token IDs across these native arms.
The original overbroad plan wording and its clarification were preserved.

The accepted binary is
`829dbf3a9b048dbc4370cc126d5c6f0d250e77f7e47500869b67c81d2313086f`;
the rejected binary is
`52a4d4bf2d8b3ce749a47449945809115490fc456b7c0eaa125534672f7e5233`.
Its source was `a065310` plus the archived device-plan overlay and separately
identified test-only changes. The comparison receipt SHA256 is
`e6bfe4cb1dbb879e064bcf2e44cd5ed9d18d4fda4c2d820e4db8cfb983e154cb`.
Private evidence retains source, commands, linked PTX identity, raw requests,
logs, tests and the rejected patch under `gpt/device-plan-private`.

The original 244/251 sequential-reference agreement remains an open numerical
qualification limitation. This experiment neither changes that reference nor
promotes any alternate tensor-core, normalization or rotary policy.

## Final-chunk fence placement control

A separate private candidate checked the same sticky routing-error flag immediately after the final chunk's plan builder and **before** its expert projections, instead of waiting until the layer's expert work finished. Earlier builders remain ordered before this check on the same stream. The purpose was to test whether fence placement, rather than fence count alone, explained the first candidate's regression. It did not meet the unchanged performance gate and is also **rejected**.

Thirteen runtime tests pass, including errors injected at the first and final builders across two chunks, sticky refusal before the final expert launches, poisoned-state retry at the correct next position, all-valid completion, and bound-stream cleanup. Faults execute two expert projections from the first chunk; the valid case executes four. An initial immutable-borrow compile failure is retained, then fixed by using the already exclusively borrowed mutable scratch.

All twelve full251 hidden/cache hashes match the accepted baseline. Text quality (12), blocking tools (9), streamed tools (5), lifecycle (7), and 128/129/255/256 boundaries pass. The same opposite-order raw-ID and secondary chat campaigns were repeated without concurrent work. All 120 actual first sampled IDs also match across arms, independently correlated from server logs.

The primary geometric reductions were **−0.188295% and −0.130396%** (small regressions), below the predeclared positive 2% requirement in both sessions. Moving the fence is not accepted as a speed improvement. The measurements include head/sampling/protocol and do not independently isolate host planning cost.

| Prompt tokens | Session 1 baseline/candidate ms | Session 2 baseline/candidate ms |
|---|---:|---:|
| 64 | 340.16 / 339.98 | 340.57 / 340.73 |
| 128 | 622.90 / 623.37 | 623.11 / 623.28 |
| 256 | 1225.37 / 1227.37 | 1225.44 / 1227.32 |
| 512 | 2423.77 / 2426.94 | 2424.54 / 2427.43 |
| 1024 | 4841.76 / 4854.91 | 4850.80 / 4856.57 |

Candidate release: `0fa94b01cbc394c7b2d53dbe3682ce36911a3430041805ed273a96d320fcb6d3`; accepted release and exact model pin are unchanged from above. The plan-builder PTX is unchanged from the first device-plan diagnostic; only the host control boundary and its tests changed.

- `comparison.json` SHA-256: `943dceef2f3f114a4c816c8cf107bab89b77b313ac489ec488a4423a0daec623`.
- `source-identity.json` SHA-256: `c4ee95b03dde952c8e008e211c3d9fef0d8e99cc7a6fd77799e9b8ee3928b1ae`.
- `sampled-id-log-control.json` SHA-256: `a53b7e2ebb8ae26371bb90c316689a63020470f76dcc8c946466ed002a581671`.
- `full251-comparison.json` SHA-256: `a47788a6a6c26d4afd3c1f1c9c29e335352fe3665a8f2c60411d88fd11373a88`.

All private source overlays, raw requests, logs and failures remain retained. No runtime, kernel, admission or default change is promoted.
