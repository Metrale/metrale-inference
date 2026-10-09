---
name: new-hardware
description: The standard method for bringing up a new HARDWARE + MODEL combination on the Metrale Engine (a new GPU class such as H100/H200, B200, a consumer Blackwell board or a non-NVIDIA device, together with the models to serve on it). It runs the hardware-class infrastructure (kernels/<class>/HARDWARE.toml inheritance and policies, DEVICES.toml, the class's FUSIONS/KERNEL_FAMILIES overlays, `met circuit plan --hardware`, golden plans, PTX gates, `--check-kernels`) and the /new-model paradigm (circuit, two-axis kernel Venn, safe parameterization, split-and-fuse, prove, microbench, optimize) in concert, starting from a mock (rehearsal) checkpoint, then iterates against a same-box vLLM baseline until Metrale wins tok/s AND J/tok at every concurrency C1-C128. Standing objective: reduce code by parameterization. Use when asked to "bring up", "port to", "support" or "start a Hardware Beachhead Campaign" for a GPU or accelerator, or when a session is handed such a campaign PR.
---

# /new-hardware: bring up a new hardware + model combination

> **BIT PARITY FIRST, THEN PERFORMANCE.** Iterating on performance is useless without bit
> parity, on the mock or on the real model. The improvement loop may not start until parity is
> achieved and recorded, and every iteration keeps it: a lever that breaks parity is discarded,
> or kept only behind a flag with the accuracy bar. Parity is defined in three tiers
> (`references/bit-parity.md`): Tier 1 self-consistency on the target, bit-exact; Tier 2
> correctness against a reference (bit-exact where the arithmetic order is the same, a declared
> tolerance and a transcript match rate where it cannot be); Tier 3 the accuracy bar.
>
> **SPEED AND ENERGY ARE SEPARATE METRICS.** Finishing sooner does not mean using less energy:
> `J/tok = P̄ / (tok/s)`, so a faster engine is cheaper only if its average power rises less
> than its speed. Measure and report both (tok/s, TTFT, and J/tok from the energy counter),
> record victory per objective, and never fold a lever that trades one for the other into a
> default (`references/speed-and-energy.md`).

**The standard.** A new class is brought up the same way every time:
1. Open a **Hardware Beachhead Campaign** (below): one PR with the target's context, its
   checklist and the code-free preparation, so any session can pick the work up cold.
2. Describe the class as **data** (HARDWARE.toml, DEVICES.toml, its policies and overlays),
   never as per-class copies of code.
3. Run the **two-axis Venn**: the new class against its nearest supported class, and each
   model against its nearest supported model. Every op lands in one of five classes.
4. **Rehearse on a mock** first, then box calibration, kernel checks, golden-plan measurement,
   the class's tensor-core policy and measured limits, then full weights. **Reach bit parity**
   (Tier 1 on the mock, Tiers 1 and 2 on the real model) and record it.
5. Measure a **same-box vLLM baseline** behind the PARITY-O.R.A.C.L.E, then run the
   **improvement loop**, keeping parity, until Metrale beats it on tok/s and J/tok at every
   rung.
6. Pass the accuracy bar (Tier 3), certify, then rotate to the next model and repeat.

**Measure the bring-up itself.** Every campaign records, per hardware + model combination, in
`.claude/skills/new-hardware/ledger/<class>.toml`: its start, **TTBP** (time to bit parity, for
the mock and for the real model), **TTPV** (time to performance/energy victory, split into TTPV-speed and TTPV-energy), the iteration
count, the levers kept and discarded, and the lines added and removed by parameterization.
Update it at every milestone; read every previous ledger before starting a new campaign.

**The campaign workflow.** One beachhead covers many models, one at a time:

```
hardware -> first model (the best-scoped one) -> improvement loop until it wins -> certification
         -> next model, same process -> ... until every model of the campaign is certified
```

Pick the first model by least work: the one whose formats the class runs natively or by an
existing fallback, with the fewest "novel" ops on the Venn. The target file names the order.

**The meta-goal: the Latent Kernel Blueprint** (`references/lkb.md`). Parameterize hardware
and kernel differences until an invariant kernel basis emerges (algorithm vs schedule), so
agents work against it instead of writing kernels. Its registry is `KERNEL_FAMILIES.toml`; a
pattern is promoted into it once two hardware/model points use it byte-identically with no
regression; everything else is named residual. **A shrinking residual means the LKB is
converging.** Every campaign records LKB coverage, residual, parameterization yield, TTBP,
TTPV and zero-day readiness in its ledger, and its exit report states "Promoted into the LKB:
...", "Residual delta: ..." (the single-class bucket separately), "Promoted into the LAB: ..."
and "LAB residual delta: ...".

**Standing objective: less code.** A new class is the moment duplication is cheapest to remove,
because the second point of every parameter is now in view. Every bring-up PR states a
**code-deleted tally** (lines and files removed, lines added) and the parameterizations behind
it. "We copied the gb10 kernel and changed two constants" is a parameterization nobody did yet.

Read before starting, and apply throughout:
- `.claude/skills/new-model/SKILL.md` in full. This skill composes with it: the model axis is
  that skill's steps 1-9, run with `--hardware <device>`.
- `AGENTS.md`, `CONTRIBUTING.md` (local checks, the 500-line cap, certification rules).
- `docs/HARDWARE.md` (adding a target, the Hopper and B200 sections, the PTX gate).
- `KERNEL-PERF.md` and `docs/perf/PERFORMANCE_NUMBERS_STANDARD.md`.
- The campaign's target file: `.claude/skills/new-hardware/targets/<class>.md`.

Detail lives in the references; read each when its step comes up:

| Reference | Step |
|---|---|
| `references/bit-parity.md` | the three parity tiers, when the loop may start, TTBP and TTPV |
| `references/speed-and-energy.md` | why speed does not imply energy, the arithmetic, energy accounting, trade-off levers |
| `references/two-axis-venn.md` | the Venn on both axes, the five classes, ranking |
| `references/lkb.md` | the Latent Kernel Blueprint: families, parameter kinds (incl. numerics), evidence, promotion, residual, convergence metrics |
| `references/lkb-math.md` | the LKB as engineering rules: parity = same reduction tree, composed error budgets, Para families, fusion gain as laxity, an e-graph fuser experiment |
| `references/parameterization.md` | the standing objective, hardware facts as data, the stability gate, the tally |
| `references/bring-up-order.md` | the ordered bring-up, fastest first, with commands |
| `references/improvement-loop.md` | the loop after the baseline, its exit and stop conditions |
| `references/lever-patterns.md` | recurring (symptom -> root cause -> lever) rules mined from the lever journal, incl. anti-patterns and the gates that catch failures early; read BEFORE choosing a lever |
| `references/measurement-discipline.md` | the rules every number must follow |
| `.claude/agents/flag-parity-oracle.md` | PARITY-O.R.A.C.L.E, the blocking config-equivalence review |
| `ledger/<class>.toml` | the bring-up ledger: TTBP, TTPV, iterations, levers, lines added and removed |
| `ledger/levers.toml` | the lever journal: one entry per optimization attempt, kept OR discarded OR failed, with its symptom, root cause and evidence |

## The Hardware Beachhead Campaign

A bring-up for a new class starts by opening a PR titled exactly
`Hardware Beachhead Campaign: <class>` (e.g. `Hardware Beachhead Campaign: H100`). It is a
strong starting point plus directions, written so ANY agent can check it out, understand the
context and move forward. It holds:
- `.claude/skills/new-hardware/targets/<class>.md`, made from `targets/TEMPLATE.md`: the class
  as the tree has it today, an audit of the gaps (each claim verified against the tree, with
  file and line), the wins from other classes that do and do not transfer, the first models,
  and an ordered first-day checklist;
- the code-free preparation that needs no device: CLI and planner changes, a tensor-core
  policy for what the class already plans, PTX-gate fixes, golden plans;
- a PR body for a cold-start agent: what it is, the current state, the first steps, where the
  context lives, and what "done" means.

It is **not merged** until work on the real device proves it; it is the branch the device
session checks out. To start the next one (B200, a consumer Blackwell, a non-NVIDIA class):
copy `targets/TEMPLATE.md` to `targets/<class>.md`, fill every section from the tree (never
from memory), open the PR with that title, and link it from the target file's header.

## Where things live

| Thing | Path |
|---|---|
| Device registry (SKUs: SMs, memory, bandwidth, peaks, native MMA, guards) | `kernels/DEVICES.toml` (schema `crates/circuit/src/hardware/device.rs`) |
| Kernel class: arch, `inherits`, `[build] extra_nvcc_flags`, `[defaults]`, `[tensor_core_policy]`, `[benchmarks.limits]`, `[memory]` | `kernels/<class>/HARDWARE.toml` |
| Class overlays of rules, families and build flags | `kernels/<class>/common/{FUSIONS,KERNEL_FAMILIES,KERNEL}.toml` (override by id, `remove`, `[shadow]`) |
| Per-model, per-class config and expected absences | `kernels/<class>/<model>/MODEL.toml` (`[expected_absent.*]`) |
| Class resolution, availability, planning, tensor-core policy | `crates/circuit/src/hardware/` (`class.rs`, `avail.rs`, `plan.rs`, `tc_policy*.rs`) |
| Golden plans per device | `kernels/circuits/plans/hw/<model>--<device>.md`, `MATRIX.md` |
| Serving defaults baked per class | `crates/kernels/build_defaults.rs`, `crates/model-layers/src/layers/ops/target_defaults.rs` |
| Benchmark limits reader | `crates/bench/src/hardware/limits.rs` |
| PTX gates | `scripts/hopper_ptx_gate.sh --hw <class>` (any class), CI `kernel-compile.yml` |
| Mock (rehearsal) checkpoints | `met ml-utils {inspect,mockify,extrapolate}`, `met serve --mock` |
| vLLM baseline harness and manifests | `bench/ladder38/harness_w55_conc_ladder.py`, `bench/ladder38/power_window.py`, `bench/baselines/<model>/published.json` |

## Step 1: open the campaign and read the class as the tree has it

1. Branch, copy `targets/TEMPLATE.md`, open the PR (title above).
2. Read the class's `HARDWARE.toml` top to bottom, its `inherits` chain, its `common/` overlays
   and every `kernels/<class>/<model>/MODEL.toml`. Record which kernels are the class's own,
   which are inherited, which are `[expected_absent]` and why.
3. `met circuit plan --matrix kernels/circuits/plans/hw --check` must pass before you change
   anything, and the class's golden plans are your first audit input: their "Tensor-core
   policy", "Declared formats on this device", gap reports and "Rule kernels this device cannot
   run" sections say what the planner thinks the class runs.
4. **Check that the planner sees the class's own kernels: this comes first.** A kernel the
   class compiles that no `FUSIONS.toml` rule names is invisible to every plan, policy and
   Venn: the reports then describe the parent class's routing, not what this class runs.
   List those kernels, then model them (rules in the class's `common/FUSIONS.toml`, families
   in its `common/KERNEL_FAMILIES.toml`, each rule citing its dispatch site) before trusting
   any hardware-axis report.

## Step 2: the two-axis Venn

Run both axes; `references/two-axis-venn.md` has the procedure.
- **Hardware axis**: `met circuit plan --checkpoint <id> --hardware <device> --precision
  declared` for every model the campaign targets, and `met circuit venn --hardware <device>`
  against the nearest class's supported models.
- **Model axis**: `/new-model` step 3 for any model the class's nearest class does not serve.
- Rank every op by step share at C1, C16 and C128 and classify it: **shared-measured**,
  **shared-unmeasured**, **parameterize**, **policy variant**, **novel**. On a new class,
  "shared-measured" needs a microbench record ON THIS CLASS; everything inherited starts as
  shared-unmeasured.

## Step 3: the bring-up, fastest first

`references/bring-up-order.md` has every command. The order:
1. **Mock**: the SAME serve command as the real model, with `--mock <spec>` added (or a
   `met ml-utils mockify` directory). Its route is identical to the real model's and its speed
   ranks levers correctly, so iterate on it for speed; confirm every kept lever's magnitude and
   every energy number on the real model by dropping `--mock` (mock energy is not trusted).
   The measured fidelity verdict is in `references/bring-up-order.md` step 1.
2. **Bit parity on the mock** (Tier 1); record it and `ttbp_mock` in the ledger.
3. **Box calibration** (`met benchmark calibrate`, where the branch has it).
4. **`met serve --check-kernels`**: re-harvest `[expected_absent]` on the real device.
5. **Golden-plan measurement**: nsys per-kernel times of the planned groups; the class's first
   `docs/kernel-perf/measurements.toml` rows and `KERNEL_FAMILIES.toml` evidence.
6. **`[tensor_core_policy]`** for the class: required ops, and every CUDA-core matmul an
   explicit exemption with its kind.
7. **`[benchmarks.limits]`**: measured on the device, then declared. Certification refuses a
   class without them; never copy another class's numbers.
8. **Full weights**, then **bit parity on the real model** (Tiers 1 and 2); record it and
   `ttbp_real` in the ledger. The loop does not start before this.
9. **Same-box vLLM baseline** on the checked-in harness, recording the exact vLLM version and
   image digest. The PARITY-O.R.A.C.L.E (`.claude/agents/flag-parity-oracle.md`) must return
   PARITY-PASS on both sides' resolved configs before the baseline is recorded.
10. **Improvement loop** (`references/improvement-loop.md`), keeping parity, until the exit
    criterion holds; record `ttpv` in the ledger. **Run `met bench preflight` before EVERY timed
    run** (lever-patterns.md's gates table, now automatic rather than merely documented). **Read
    `references/lever-patterns.md` BEFORE choosing a lever each iteration; append a
    `ledger/levers.toml` entry AFTER every verdict** (kept, discarded, marginal, regressed,
    parity-broken or failed) — a non-kept entry is as valuable as a kept one and needs its
    failure mode, what caught it, and what would have caught it sooner.
11. **Accuracy bar** (Tier 3): BFCL with N, sample pct and the draw's SHA; agentic-webserver
    with a same-night control.
12. **Certification**; then the next model, from step 1.

## Step 4: parameterize as you go

`references/parameterization.md`. In short:
- hardware facts are data (SM count, smem, bandwidth, tensor-core formats, cluster/TMA/wgmma
  availability), read from DEVICES.toml or probed, never forked into per-class copies;
- a template or policy parameter instead of a duplicated kernel;
- every parameterization passes the **stability gate**: byte-identical at every existing point,
  no microbench regression, existing gates unchanged;
- every PR states its code-deleted tally.

## Step 5: beat vLLM, then certify

The improvement loop runs after the baseline, keeping bit parity, until Metrale is faster on
tok/s at EVERY rung C1-C128 AND cheaper on J/tok at every rung but at most one mid-ladder rung
lost by a small margin; an edge rung (C1/C2 or C64/C128) lost, or two rungs lost, keeps it going.
It is bounded: it also ends at the stall limit (three iterations without improving the worst
rung) or at the target file's loop budget, and then escalates with which objective is won,
which is not, the worst rung's profile and the best levers left. Every comparison against vLLM,
and every A/B whose arms differ in any setting beyond the lever, gets a PARITY-O.R.A.C.L.E
ruling first. Then: recipes that set every variable explicitly, measure-then-declare bounds,
certification as `AGENTS.md` describes.

## Rules that always apply

- **Precision follows the checkpoint**: the default is the declared format per layer. A format
  the device cannot run natively takes an exact path (e.g. E2M1 to E4M3 on an FP8 MMA) and the
  plan says so; lowering precision is a flagged opt-in with an accuracy bar, disclosed on
  records.
- **Same-box numbers only.** A delta between two boxes is a box effect until a same-box A/B
  says otherwise.
- **Own build directory per worktree** (`CARGO_TARGET_DIR`): a shared one ships stale PTX.
- **Shared machines**: never build or run during someone else's timed measurement; respect the
  class's memory `util_ceiling`; start servers detached and stop them by PID.
- **Public repository**: no host names, private paths, addresses or internal names in anything
  committed; "the reference box" is enough.
- **Before handing back**: `cargo fmt`, clippy `-D warnings`, tests, `cargo doc`, the 500-line
  cap, SPDX headers, dated comments, and the code-deleted tally in the PR body.
