# Hardware Beachhead Campaign: <CLASS>

<!-- Copy this file to targets/<class>.md, fill every section FROM THE TREE (cite file:line),
     never from memory, and open a PR titled exactly "Hardware Beachhead Campaign: <CLASS>".
     Delete these comments. A claim you did not verify is written as "unverified: <why>". -->

- **Campaign PR:** <link, once open>
- **Branch / base:** <branch> on <base branch at commit>
- **Devices** (`kernels/DEVICES.toml` ids): <ids>  ·  **Kernel class:** `kernels/<class>/`
- **Status:** <not started | code-free prep done | on device: step N of the checklist>
- **Done means:** bit parity is recorded for every model below (`references/bit-parity.md`,
  Tiers 1-3), the exit criterion of `references/improvement-loop.md` holds on a same-box
  scoreboard for each, `[benchmarks.limits]` and `[tensor_core_policy]` are measured and
  declared, and the class's gates are certified.
- **Ledger:** `ledger/<class>.toml`, one `[[combo]]` per model (copy the schema from an existing
  ledger); read the other ledgers first.
- **LKB baseline** (`references/lkb.md`): coverage <C1/C16/C128 %> (measured <%>), residual
  <count> kernels / <lines> lines; the exit report states "Promoted into the LKB" and
  "Residual delta".
- **Loop budget per model:** <calendar days and device-hours>; the loop escalates when it runs out.

## 1. The class as the tree has it today

| Fact | Value | Source |
|---|---|---|
| arch / compute capability | | `kernels/<class>/HARDWARE.toml [hardware]` |
| inherits | | same |
| SMs, shared memory, L2, memory, bandwidth | | `kernels/DEVICES.toml` (per device) |
| tensor-core instructions (native MMA formats) | | `kernels/DEVICES.toml` `native_mma`, `mma_family` |
| formats with no native MMA, and how the plan runs them | | the golden plan's "Declared formats on this device" |
| `[build] extra_nvcc_flags` and what each guard compiles out | | HARDWARE.toml, `[[guard]]` in DEVICES.toml |
| `[defaults]` that differ from the parent | | HARDWARE.toml |
| the class's own kernels (additions and `[shadow]` replacements) | | `kernels/<class>/common/`, its KERNEL.toml |
| `[expected_absent]` per model, and where it was harvested | | `kernels/<class>/<model>/MODEL.toml` |
| `[tensor_core_policy]` | present / absent | HARDWARE.toml |
| `[benchmarks.limits]` | present / absent | HARDWARE.toml |
| CI coverage (PTX gate job, golden plans) | | `.github/workflows/kernel-compile.yml`, `kernels/circuits/plans/hw/` |

## 2. Audit: what is missing or wrong

Numbered, each with its evidence (file:line or command output) and its consequence. Typical
entries: plans that are 100 % roofline projection; ops with no rule on the class; the class's
own kernels that no FUSIONS.toml rule names (invisible to plans); CUDA-core matmuls the class
has a tensor-core path for; serving defaults copied from the parent and never measured here;
missing `[benchmarks.limits]` (certification refuses the class); CLI views that cannot target
the class.

## 3. What transfers from other classes, and what does not

| Work | Transfers? | Why (instruction, format, measured only elsewhere) |
|---|---|---|

A kernel that compiles for the class transfers as **code**; its measured win does not transfer
until it is measured here.

## 4. Models to bring up first

| Model (checkpoint id) | Declared formats | Why first | Golden plan |
|---|---|---|---|

## 5. Parameterization opportunities seen from here

The duplicated constants, per-class copies and per-point kernel copies this class makes visible
(`references/parameterization.md`), each with the files involved and the expected code deleted.

## 6. First-day checklist (in order)

- [ ] Check out the campaign PR; build with a private `CARGO_TARGET_DIR` and `METRALE_TARGET_HW=<class>`.
- [ ] PTX gate for the class (`scripts/hopper_ptx_gate.sh --hw <class> --model all --strict`).
- [ ] Mock the first model (its serve command plus `--mock`); boot, chat, C1/C16 decode; route fidelity vs the real model.
- [ ] Bit parity on the mock (Tier 1); ledger `ttbp_mock`.
- [ ] `met serve --check-kernels` per model; re-harvest `[expected_absent]`.
- [ ] nsys the mock at C1/C16/C128; compare planned groups with profiled kernels.
- [ ] Measure and declare `[benchmarks.limits]`; then box calibration where available.
- [ ] Full weights; `--check-kernels` again; memory model vs the boot ledger.
- [ ] Bit parity on the real model (Tiers 1 and 2); ledger `ttbp_real`. The loop waits for this.
- [ ] vLLM baseline (PARITY-O.R.A.C.L.E first; record version and image digest).
- [ ] Improvement loop, keeping parity, until the exit criterion; ledger `ttpv`.
- [ ] Accuracy bar (Tier 3); certification; next model.

## 7. Log

| Date | Step | Result | Evidence |
|---|---|---|---|
