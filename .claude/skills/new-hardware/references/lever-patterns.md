# Lever patterns: what to check first, and what not to do

Mined from `ledger/levers.toml` (24 entries as of 2026-10-05: 8 kept, 4 discarded, 1 marginal,
2 parity-broken, 9 failed — the failed and discarded entries are as load-bearing as the kept
ones). **Read this file BEFORE choosing a lever** (`references/improvement-loop.md` step c);
**append a `levers.toml` entry AFTER every verdict**, kept or not. A pattern below with a low
evidence count is a hypothesis, not a rule — treat it as a thing to check, not a thing to assume.

## Patterns: symptom -> root cause to check first -> lever type to try

### P1. Inherited constants underfill the new class or the new model
**Evidence: 3** (`hopper-dense27b-l1-tc-rows-band`, `hopper-dense27b-util-ceiling-0.85-inherited`,
`gb10-dense-27b-recipe-max-batch-size-inherited`).
IF a tile band, a util ceiling, a batch cap, or any other hardware-sized constant was carried
over from a PARENT class or an earlier, smaller model, THEN check whether its own rationale
(SM count, memory architecture, concurrency ladder) still holds on the new class/model FIRST,
before profiling anything else — it is usually the single biggest lever available on day one.
Try a **data-row** or **parameterization** fix (derive the constant from `DEVICES.toml`/the
recipe's own shape), not a new kernel.
- **Speed payoff:** large when it hits (+74..+207% tok/s at the affected rungs in the strongest
  case) but zero at rungs the constant does not touch — always state it per rung, not as one number.
- **Energy payoff:** large and in the SAME direction as speed here (-23..-51% J/tok) — this
  family of fix tends not to trade one axis for the other, unlike most levers below.

### P2. A path runs silently below the declared precision
**Evidence: 4** (`gb10-moe-nvfp4-moe-prefill-tc`, `hopper-dense27b-w8a8-attention-and-prefill-fixed`,
`hopper-dense27b-nvfp4-tier-below-declared-precision`, `gb10-moe-nvfp4-fp8-kv-declared` as a
negative case). IF Tier 2 (or a manual read of the kernel's own arithmetic) shows a format
BELOW what the checkpoint declares, or below what the comparison engine executes, THEN the
parity oracle or a precision audit finds it, not a speed profile — a below-declared path can be
fast and simultaneously wrong, so audit precision on prefill AND decode explicitly as a standard
bring-up step, not only where a speed gap already points.
- **Speed payoff:** mixed. Raising precision to match declared sometimes also speeds things up
  (prefill moved to the same arithmetic as the already-faster decode kernel, -10..-13% TTFT) and
  sometimes costs nothing either way (attention fix: tok/s unaffected) — **never assume** a
  precision-correctness fix is a performance lever; time it.
- **Energy payoff:** the one measured counter-example (FP8 KV, a precision-RAISING lever) made
  energy WORSE (+1.5% J/tok at wide rungs) despite being the more-correct choice — precision and
  cost are independent axes. Judge both, every time, as `references/speed-and-energy.md` says.

### P3. Phase-mixed profiling hides which phase actually costs time
**Evidence: 1** (`gb10-moe-nvfp4-exact-wave-prefill`), reinforced by its own precursor failure
below (P6). IF TTFT is behind the comparison engine but decode tok/s is fine, THEN split the
prefill burst from the steady decode state BEFORE profiling either — a burst-only profile found
that per-stream attention/GDN projections, not the expert FFN, were the next TTFT lever (34% of
the C16 burst window), which a mixed-window profile would not have isolated.
- **Speed payoff:** winning the burst split itself won tok/s at every rung measured (+0.9..+6.8%)
  with TTFT p99 roughly halved from C4 up — this is a phase-level lever, so its tok/s gain grows
  with concurrency (more of the step is burst-shaped at wider C).
- **Energy payoff:** same direction as speed (-0.6..-6.5% J/tok); the one lever that tried to go
  FURTHER inside the same burst (batching the already-small per-stream projections, P-marginal
  below) found the remaining share too small to matter — profile the phase SHARE before building
  a fusion for it, not just its plausibility.

### P4. A kernel win measured in isolation doesn't show at the rung level
**Evidence: 1 direct (`gb10-moe-nvfp4-lever1b-batched-gdn-projections`, marginal), 1 adjacent
(`gb10-gdn-spine-five-dead-ends`, all 5 discarded)**. IF a microbenchmark shows a kernel-level
win, THEN measure the kernel's SHARE of the rung's step before deciding the rung-level payoff is
worth a PR — a correctly-identified, correctly-implemented, bit-identical fusion still landed
inside the noise band because the share it targeted (launch overhead on an already-small piece)
was small at the shapes that matter in production.
- **Speed/energy payoff:** can be legitimately ~0 even when the underlying idea is sound; this is
  the pattern that justifies the `marginal` verdict as distinct from `discarded` — the idea may
  still apply at a different, wider concurrency than was screened.

### P5. Speed and energy diverge — judge both axes, separately, every time
**Evidence: 2** (`gb10-moe-nvfp4-fp8-kv-declared`: 0% tok/s, +1.5% J/tok = discarded;
`gb10-moe-nvfp4-moe-prefill-tc`: TTFT win, bits change = accuracy-gated, not a blanket win).
IF a lever's tok/s delta is flat or small, THEN check J/tok independently before calling it a
non-event — a flat-speed lever can still be a real energy loss (or, in other campaigns, a real
energy win), and folding either into a default without recording both deltas violates the
standing rule in `references/speed-and-energy.md`.
- **Payoff of APPLYING this pattern:** it is what correctly caught the FP8 KV discard — tok/s
  alone would have called it a no-op; J/tok alone (without also checking tok/s) would not have
  told you the lever cost something it never bought back.

## ANTI-PATTERNS: what NOT to do

**DON'T** trust a lever's A/B delta (zero OR nonzero) without a route proof that it actually
armed at the measured shape (N=2: `gb10-moe-nvfp4-no-w4a16-tc-null-lever`,
`hopper-dense27b-w8a8-attention-first-cut-gated-off`). A near-zero delta has two causes — the
path truly does not matter, or the lever never dispatched — and only an nsys/ncu route check or
an explicit kernel-absence assertion tells them apart. **Instead:** profile the route BEFORE
trusting the number, every time a lever targets a specific phase (prefill has its own stream on
this engine) or a specific kernel family.

**DON'T** trust an identity/parity check that only compares two things to EACH OTHER without
first asserting both sides are non-empty or non-trivial (N=1: `gb10-vacuous-mockify-identical`,
same shape of bug as the broader "a passing test may not have run" class of defect). **Instead:**
assert a nonzero file/byte/row count on both sides before reporting a verdict.

**DON'T** assume a lever proven bit-identical at concurrency 1 is safe at the concurrency it is
meant to help (N=1: `gb10-moe-nvfp4-first-varlen-codispatch-gate-fail`, 2/19 then 15/19 transcript
diffs at width 16 after a clean width-1 pass). **Instead:** run the identity gate at the WIDEST
concurrency the lever can reach before any timed A/B is scheduled.

**DON'T** carry over a class's own serving constant (util ceiling, precision tier, row-tile band)
to a new class just because it was the certified choice there (N=3, see P1 and
`hopper-dense27b-nvfp4-tier-below-declared-precision`). **Instead:** re-derive each constant from
the new class's own hardware facts or re-run the parity oracle on the proposed baseline before
measuring a single data point.

**DON'T** share one build directory across worktrees that edit kernels, and don't trust a staged
A/B binary's identity by filename alone (N=2: `gb10-moe-nvfp4-stale-ptx-shared-target-dir`,
`gb10-moe-nvfp4-ab-restarted-stale-prehead-binary`). **Instead:** one `CARGO_TARGET_DIR` per
worktree with kernel edits, and stamp + verify the commit sha of every A/B binary immediately
before launch.

**DON'T** time a gate from a debug build, and don't trust a floating `:latest` image tag for a
comparison-engine baseline (N=2: `gb10-debug-build-false-hang`, `gb10-vllm-latest-tag-drift`).
**Instead:** assert the build profile before a timed run; pin the comparison engine by exact
version AND digest, recorded on every baseline.

**DON'T** size a buffer or arena unconditionally for an opt-in flag (N=1:
`gb10-moe-nvfp4-121-arena-growth-defect`). **Instead:** gate the SIZER call itself, not only the
feature's own call site — run the full memory-ledger suite, not just the new flag's own test.

**DON'T** build full correctness machinery (state checkpoint/rollback, a ported fused kernel)
before checking the cheap number that decides whether it can ever pay (N=2:
`gb10-trtllm-ngram-specdecode-dead-end`: 73% draft rejection measured only after full rollback
machinery existed; `gb10-flashinfer-fused-moe-garbage-dead-end`: silent wrong output found only
by reading output quality after four compile fixes). **Instead:** measure the cheap, decisive
number first (draft acceptance rate; a known-good-output check right after first compile on a
new target) before investing in the expensive machinery around it.

**DON'T** measure a recurrent/load-bound kernel only at a narrow, GPU-underfilling shape (N=1,
`gb10-gdn-spine-five-dead-ends`: a 1.60x win at nt=1 became 0.92x at nt=64). **Instead:** measure
at the shape that fills the device (nt=16/64 rule), since occupancy effects only show there.

## Gates that catch failures early

Ordered by how early each fires (cheapest/earliest first) and mapped to the failure mode it is
the standing defense against:

**Six of these are now one command, not just a table row.** `met bench preflight` (PR
`feat/bench-preflight`, 2026-10-05) makes the first six gates below AUTOMATIC: it refuses to let a
timed run start rather than merely documenting that it should have been refused. Run it before
EVERY timed A/B, ladder or benchmark (`references/improvement-loop.md`'s "One iteration" now says
so explicitly). Its `Unavailable` status on the recipe-explicitness row is itself loud, not a
silent pass — it names PR #124 as the still-missing real check.

| Gate | Fires at | Catches | `met bench preflight` check id |
|---|---|---|---|
| Build-profile assertion (refuse a timed run from a debug build) | before any measurement | measurement-methodology-error (debug-timed-as-release) | `release_build` |
| Nonzero-content assertion on both sides of an identity/digest check | inside the check itself | measurement-methodology-error (vacuous pass) | `comparison_non_vacuous` (`--compare DIR_A DIR_B`) |
| Commit-sha stamp + verify on every A/B binary, immediately before launch | before a timed run starts | measurement-methodology-error (stale/pre-head binary) | `binary_head` (`--expect-head <sha>`) + `binary_clean` |
| Per-worktree `CARGO_TARGET_DIR` | at build time | measurement-methodology-error (stale PTX from a shared build cache) | `kernel_freshness` (recomputes the kernel closure hash from the tree and compares it to the binary's own baked attestation — catches the shared-`CARGO_TARGET_DIR` case directly, not just the per-worktree hygiene rule) |
| Pin comparison engine by exact version + digest, recorded on the baseline | at baseline-measurement time | config-drift (a floating `vllm:latest` tag) | `vllm_image_pinned` (`--vllm-image name@sha256:...`) |
| Recipe-explicitness checker (every CLI-settable flag named, derived from the CLI's own manifest) | at recipe resolution | inherited-constant-underfill (a silent CLI default) | `recipe_explicit` (`--recipe <id>`) — **`Unavailable` until PR #124 lands**; not yet a real check, reported as such |
| Route proof (nsys/ncu kernel-presence check, or an explicit route assertion in code) | before trusting any A/B delta | silent-gate-bypass | not yet automated — do this by hand |
| Identity gate run at the WIDEST concurrency the lever reaches | before the timed A/B | numerics-path-mismatch (width-dependent divergence) | not yet automated — do this by hand |
| PARITY-O.R.A.C.L.E on the proposed baseline configuration | before iteration 0 counts | below-declared-precision / config asymmetry | not yet automated — do this by hand |
| Full memory-ledger suite (not just the new feature's own test) | before merge | memory-waste (unconditional buffer growth) | not yet automated — do this by hand |
| Cheap decisive pre-check (draft-acceptance rate; known-good-output check right after first compile on a new target) | before building expensive correctness machinery | wasted effort on a dead end whose fate the cheap number already decided | not yet automated — do this by hand |

`met bench preflight` also prints an environment record (every `METRALE_*` var set, and any
mismatch against the recipe's declared `env:`) on every run, so a lever's arming is proven, not
assumed (`a-lever-lives-in-three-places` / prove-the-lever-moved).

## Meta-metrics: how each pattern shortens TTBP / TTPV

- **P1 (inherited constants)** shortens **TTPV-speed** the most directly: it is usually the
  first and largest lever available, often closing most of the gap to the exit criterion before
  any new kernel is written. Checking it FIRST (before profiling) turns a multi-iteration search
  into a single iteration.
- **P2 (below-declared precision)** shortens **TTBP**, not TTPV: finding it during the Tier-2
  audit (before the loop starts) avoids measuring a performance ladder against the wrong
  baseline and then having to redo it once the precision call is made.
- **P3 (phase split)** shortens **TTPV-speed and TTPV-energy together** for TTFT-bound rungs: it
  turns "the comparison engine wins TTFT" from an open-ended profiling problem into a two-step
  one (split the window, then rank gaps inside the burst only).
- **P4 (measure the share)** shortens the LOOP ITSELF by preventing stall-limit burns on
  marginal ideas: checking the profiled share before building a fusion avoids spending a full
  iteration (implementation + proof + timed A/B) on something the share already ruled out.
- **P5 (judge both axes)** does not shorten TTBP/TTPV directly but prevents a FALSE TTPV-energy
  declaration — catching this after the exit criterion is declared costs an entire re-campaign.
- **The gates table** is the highest-leverage meta-pattern: every gate in it was written AFTER a
  real time loss (0.5-6 hours each in `levers.toml`); a new campaign that installs all of them on
  day one should not re-pay any of those costs.

## How to update

1. **After every loop-iteration verdict** (`references/improvement-loop.md` step f), append a
   `[[lever]]` entry to `ledger/levers.toml` — kept, discarded, marginal, regressed,
   parity-broken, or failed. A non-kept entry needs `failure_mode`, `caught_by`,
   `time_lost_hours` and `would_catch_earlier` filled in; these are what make the entry useful to
   a future campaign, not optional color.
2. **After every campaign** (exit criterion met, or escalation), re-read `levers.toml` in full
   and revise this file: update each pattern's evidence count, re-check whether its stated
   payoff still holds, retire a pattern whose evidence turned out to be one-off, and add a new
   pattern once two or more entries independently support it. Do not add a pattern from a single
   entry — state it as a hypothesis in the entry's own `notes` field instead, and promote it here
   once a second entry confirms it.
3. Keep this file's two kinds of evidence separate: a KEPT lever's payoff numbers, and a
   non-kept lever's failure mode + the gate that would have caught it sooner. Both are first-class.
