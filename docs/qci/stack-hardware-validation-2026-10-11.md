# QCI stack hardware validation — 2026-10-11

Frozen runtime: `99b2613072a69a8708048347f1b74d9cc7880ed7`, PR142 after PR145/143 merge-down. This is bounded regression evidence, not model certification or throughput qualification.

## Hardware and checkpoints

Native Linux Strix (`tom@strix.local`): gfx1151, ROCm7.2.1/HIP7.2.53211. Checkpoints: GPT-OSS20B MXFP4 pin `6cee5e81ee83917806bbde320786a8fb61efebee`; Laguna XS2.1 INT4 pin `4b7e28abdc0a8b121def816b89d631750bc53c92`. GB10 component tests: Spark2, sm121, CUDA13.0.88. Existing Spark services were left running.

## Results on the frozen tree

- Both native HIP release server builds passed; Laguna kernel check resolved180 lookups with0 unresolved (78 explicitly absent). GPT serial and chunk128 kernel checks passed.
- Dense union: CUDA and HIP each passed108 BF16 cases comparing original143 legacy/M4, merged legacy/M4 and scalar;13 FP32 cases comparing original142, merged and scalar; independent CPU int64 dyadic oracle had0 mismatches. Wrong-row and omitted-grid-Y controls failed as intended.
- The older dense harness passed65 FP32/BF16 cases, split/direct grids and launch refusals on both backends, with0 mismatches. Early-rounding and truncation controls detected bad results.
- CUDA resource check: legacy/M4/FP32 static shared memory16896/4224/16896 bytes;48/38/56 registers; no spills.
- Laguna:9 retained prompt fixtures (including704/1523-token prompts), BF16 and FP8 KV: all18 candidate/older-binary response comparisons exactly match returned logprob maps, greedy logprob maps, greedy text and finish reasons. Reference argmax tallies differ when tied scores arrive in a different JSON-map order; that tally is not a stable equality oracle. Existing reference numerical gaps remain.
- GPT-OSS: all251 retained next-token predictions match the earlier WSL receipt. Native chunk capacities16/64/128 match serial hidden and KV bytes and following decode, with0 nonfinite values. The earlier receipt comparison is across operating systems and only checks next-token IDs.
- GPT-OSS API smoke:6/6 serial and6/6 chunk128 (arithmetic, extraction, streaming, exact fixture tool call, tool response and repeat). Wrong-answer/raw-Harmony/wrong-tool-argument grader controls were rejected.
- Laguna API smoke:5/6 on both the frozen candidate and retained older binary. The tool response emits correct content `7`, but reports `length` after budget exhaustion. This predates the merge. A local scheduler fix narrows the post-thinking EOS guard to required tool calls; optional sticky `tool_request` remains available to the prose watchdog. The real scheduler regression fails before the fix on native HIP and macOS Metal, then passes locally with3 other JSON EOS controls. Five tool grammar controls,7 first-token policy tests,3 thinking-EOS tests and2 post-think tests also pass (including required, incomplete and parallel calls). Native HIP EOS tests also pass4/4; rebuilt release server completed successfully. Final native API replay passes6/6; tool-result content is `7` with `finish_reason: stop`. It is saved as `smoke-laguna-fixed/` separately from the frozen-tree evidence. The fix is included in this change; its source hashes and separate native binary identity are recorded in the receipt.

## Reproducibility repairs

The checked-in dense harness referenced an absent frozen include and leaked its `MAX_M` macro into the new template. The local repair restores the exact historical blob from `9484665ee9a7b2445358163ec0e1e33f8c16d166` (SHA256 `dbe5bc83ba5f02d74483dce95004549c936184bf48296a967cfed57139b15604`) and undefines that macro. Production kernels are unchanged. Canceled/wrong-head intermediate family runs do not count; only `family_union.final.run.log` and final receipt count on CUDA.

## Receipts and limits

Compact results, identities and controls: [JSON receipt](stack-hardware-validation-2026-10-11.json).

Local evidence: `/Users/tom/dev/.metrale-work/qci-stack-validation-20261011/` (CUDA `receipt.json`, native `receipts/`, replay scripts). Remote evidence: `/home/tom/qci-stack-validation-20261011/evidence` and `/hip`. Sources were staged with `git archive`; binaries and module manifests are hashed. Full model traces stay on Strix. No concurrency campaign, broad quality suite, performance certification, Gemma/Nemotron full-checkpoint test or Qwen Image test was run. Strix hardware metadata still declares48 SMs while device reports20 (existing issue155). These checks address the merge kernel and available checkpoints, not every stack model. No benchmark certification records were created.

## Pre-push checks

Workspace formatting, changed-file typos, diff whitespace and the repository license engine passed (3,820 files checked, 0 invalid). macOS workspace Clippy with Metal/no-default features was blocked by storage integration tests importing CUDA-only modules; focused server Clippy was blocked by existing Metal GPU-runtime lints (`too_many_arguments` and non-Send/Sync `Arc`). Those files are unchanged. These lint attempts are not green CI claims. Native release builds and the focused scheduler/API validations above passed.
