---
name: new-hardware
description: The standard method for bringing up a new HARDWARE + MODEL combination on the Metrale Engine (a new GPU class such as H100/H200, B200, a consumer Blackwell board or a non-NVIDIA device, together with the models to serve on it). It runs the hardware-class infrastructure (kernels/<class>/HARDWARE.toml inheritance and policies, DEVICES.toml, the class's FUSIONS/KERNEL_FAMILIES overlays, `met circuit plan --hardware`, golden plans, PTX gates, `--check-kernels`) and the /new-model paradigm (circuit, two-axis kernel Venn, safe parameterization, split-and-fuse, prove, microbench, optimize) in concert, starting from a mock (rehearsal) checkpoint, then iterates against a same-box vLLM baseline until Metrale wins tok/s AND J/tok at every concurrency C1-C128. Standing objective: reduce code by parameterization. Use when asked to "bring up", "port to", "support" or "start a Hardware Beachhead Campaign" for a GPU or accelerator, or when a session is handed such a campaign PR.
---

# /new-hardware: bring up a new hardware + model combination

**The standard.** A new class is brought up the same way every time:
1. Open a **Hardware Beachhead Campaign** (below): one PR with the target's context, its
   checklist and the code-free preparation, so any session can pick the work up cold.
2. Describe the class as **data** (HARDWARE.toml, DEVICES.toml, its policies and overlays),
   never as per-class copies of code.
3. Run the **two-axis Venn**: the new class against its nearest supported class, and each
   model against its nearest supported model. Every op lands in one of five classes.
4. **Rehearse on a mock** first, then box calibration, kernel checks, golden-plan measurement,
   the class's tensor-core policy and measured limits, then full weights.
5. Measure a **same-box vLLM baseline** behind the PARITY-O.R.A.C.L.E, then run the
   **improvement loop** until Metrale beats it on tok/s and J/tok at every rung.
6. Certify, then rotate to the next model and repeat.

**The campaign workflow.** One beachhead covers many models, one at a time:

```
hardware -> first model (the best-scoped one) -> improvement loop until it wins -> certification
         -> next model, same process -> ... until every model of the campaign is certified
```

Pick the first model by least work: the one whose formats the class runs natively or by an
existing fallback, with the fewest "novel" ops on the Venn. The target file names the order.

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
| `references/two-axis-venn.md` | the Venn on both axes, the five classes, ranking |
| `references/parameterization.md` | the standing objective, hardware facts as data, the stability gate, the tally |
| `references/bring-up-order.md` | the ordered bring-up, fastest first, with commands |
| `references/improvement-loop.md` | the loop after the baseline, its exit and stop conditions |
| `references/measurement-discipline.md` | the rules every number must follow |
| `.claude/agents/flag-parity-oracle.md` | PARITY-O.R.A.C.L.E, the blocking config-equivalence review |

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
   `met ml-utils mockify` directory). Iterate on the mock for speed (and, once its fidelity
   verdict allows, energy), then switch to the real model by dropping `--mock`. What the loop
   may trust from a mock, and what each iteration must confirm on real weights, is the
   fidelity verdict in `references/bring-up-order.md` step 1.
2. **Box calibration** (`met benchmark calibrate`, where the branch has it).
3. **`met serve --check-kernels`**: re-harvest `[expected_absent]` on the real device.
4. **Golden-plan measurement**: nsys per-kernel times of the planned groups; the class's first
   `docs/kernel-perf/measurements.toml` rows and `KERNEL_FAMILIES.toml` evidence.
5. **`[tensor_core_policy]`** for the class: required ops, and every CUDA-core matmul an
   explicit exemption with its kind.
6. **`[benchmarks.limits]`**: measured on the device, then declared. Certification refuses a
   class without them; never copy another class's numbers.
7. **Full weights**.
8. **Same-box vLLM baseline** on the checked-in harness, recording the exact vLLM version and
   image digest. The PARITY-O.R.A.C.L.E (`.claude/agents/flag-parity-oracle.md`) must return
   PARITY-PASS on both sides' resolved configs before the baseline is recorded.
9. **Accuracy bar**: BFCL with N, sample pct and the draw's SHA; agentic-webserver with a
   same-night control.
10. **Improvement loop** (`references/improvement-loop.md`) until the exit criterion holds.
11. **Certification**.

## Step 4: parameterize as you go

`references/parameterization.md`. In short:
- hardware facts are data (SM count, smem, bandwidth, tensor-core formats, cluster/TMA/wgmma
  availability), read from DEVICES.toml or probed, never forked into per-class copies;
- a template or policy parameter instead of a duplicated kernel;
- every parameterization passes the **stability gate**: byte-identical at every existing point,
  no microbench regression, existing gates unchanged;
- every PR states its code-deleted tally.

## Step 5: beat vLLM, then certify

The improvement loop runs after the baseline and repeats until Metrale is faster on tok/s at
EVERY rung C1-C128 AND cheaper on J/tok at every rung but at most one mid-ladder rung lost by a
small margin. An edge rung (C1/C2 or C64/C128) lost, or two rungs lost, keeps it going. It
stops and reports, with the profile, after three iterations without progress. Every
iteration's comparison against vLLM, and every A/B whose arms differ in any serve or bench
setting beyond the lever, gets a PARITY-O.R.A.C.L.E ruling first. Then: recipes that set every variable explicitly,
measure-then-declare bounds, certification as `AGENTS.md` describes.

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
