# The bring-up order, fastest first

Each step is cheaper than the next and catches a class of failure the next one would hit more
expensively. Do not skip ahead: a full-weight serve that fails on a kernel the mock would have
exercised costs an hour of download and load.

Every command below runs from the repository root. `<device>` is a `kernels/DEVICES.toml` id
(`h100-sxm`, `h200-sxm`, `b200`, ...), `<class>` its `class` (`hopper`, `b200`, ...).

## 0. Build for the class, in its own build directory

```
export CARGO_TARGET_DIR=<a directory only this worktree uses>
METRALE_TARGET_HW=<class> cargo build --release -p metrale-server --bin met
```

- A build directory shared with another worktree tracks the other worktree's kernel source
  paths: a `.cu` edit here is not rebuilt and the binary runs stale PTX with no error. One
  worktree, one `CARGO_TARGET_DIR`. Prove a kernel change reached the binary (PTX timestamp,
  or the route in the serve log, or nsys) before reading any A/B.
- Before the device: `scripts/hopper_ptx_gate.sh --hw <class> --model all --strict` (despite
  its name it takes any class) answers "does every kernel assemble for this arch" with nvcc
  alone. It runs its self-test first; a gate whose failure path never ran is not evidence.
- Never measure with a debug build, and never derive a gate value from one.

## 1. Mock (rehearsal) checkpoint

A mock keeps the architecture exactly (every dim, format, scale layout, layer kind and
per-layer declared precision) with fewer layers and synthetic, deterministic weights. It lets
kernels be iterated and timed on the new device in minutes, before the full weights are
downloaded, and it runs in vLLM too.

```
met ml-utils inspect  --checkpoint <id> --spec mock.toml      # the plan: kept layers, bytes, signatures
met ml-utils mockify  --checkpoint <id> --spec mock.toml --out <dir>   # a real HF checkpoint
met serve <id> <every serve flag explicitly> --mock mock.toml          # synthesizes at load, reads no weights
met serve <id> <the same flags>                                        # the real model: drop --mock
met ml-utils extrapolate --point <mock-a>=<value> --point <mock-b>=<value> --scaling rate
```

**The workflow.** Keep one serve command per model. Iterate a lever on the mock (add
`--mock`), then confirm it on the real model with the identical command minus `--mock`. The mock
is only useful if it takes the same kernel path as the real model: before the first iteration,
check route fidelity (the kernels the serve selects, from `--check-kernels`, the serve log and an
nsys kernel list at C1/C16, are the same set apart from the layer count). If they differ
anywhere, the mock optimizes the wrong thing: stop and find out why.

- `mock.toml` states every key (no defaults): `layers.per_signature` (one count per layer
  signature, e.g. `[1, 1]` for a checkpoint whose FFN format changes part-way), and `experts`,
  `vocab`, `mtp`, `vision`, `capacity`, `routing`, `values`, `speculative`. Keep all experts and
  the full vocab for any timing claim.
- What a mock is good for on a new class: boot, kernel availability, the planned routes, the
  per-kernel timing of the kept layers (nsys), the decode and prefill step shape, and a vLLM
  load check of the same architecture.
- **Fidelity verdict: pending.** It is being measured on the GB10 reference box (route,
  per-rung speed, extrapolation, a known lever's relative speed and energy delta on mock vs
  real weights); the results follow in the campaign PR and replace this paragraph. Until then:
  - **timing** is usable for direction (an earlier fidelity round on the reference box put C1
    tok/s and cold 8k TTFT within a few percent; wide-rung extrapolation missed by more);
  - **energy is not trusted.** Synthetic weights switch more bits than trained ones, so a mock
    drew more power (67-74 W against 52-62 W for the full model at C16-C64 on the reference
    box) and extrapolated J/tok missed by 10-60 %. Never quote a mock's J/tok, and confirm
    every energy claim on real weights.
- A mock never certifies anything: `GET /forward` discloses its digest, gate records carry it
  and fail, and `met benchmark run` refuses a gate or accuracy run against a mock server.
  Accuracy is measured on the full model, at the end.

## 1b. Bit parity on the mock

Tier 1 of `references/bit-parity.md` on the mock: eager vs graphed, run-to-run determinism,
batch-row invariance where the class promises it, circuit vs legacy where the arch runs on the
circuit, and mock-byte determinism (mockify twice, same digest). Record it in the target file's
log and set `ttbp_mock_at` / `ttbp_mock_h` in `ledger/<class>.toml`. No timing taken before this
point is evidence of anything.

## 2. Box calibration

Where the branch carries it, `met benchmark calibrate --hardware <class> --checkpoint <id>`
records the box's decode bandwidth, C1 energy, idle power and 32k prefill/restore, so later
records on this box can be compared with each other and with other boxes of the class. It
drives gate runs, so it needs the class's `[benchmarks.limits]` (step 6) to exist first; on a
brand-new class, measure the limits by hand, declare them, then calibrate.

## 3. Kernel checks on the device

```
met serve <id> --check-kernels <serve flags>
```

prints which kernels loaded, which are absent, and whether each absence is declared in the
model's `[expected_absent.*]` with its reason. The class's MODEL.toml copies may have been
harvested on another class: re-harvest them here, and turn every undeclared absence into either
a fix or a declared, reasoned entry. A kernel compiled out by a `[build] extra_nvcc_flags`
guard (e.g. `-DMETRALE_NO_WARP_BLOCKSCALE_MMA`) must be absent; one that is absent without a
guard is a build defect.

## 4. Golden-plan measurement

```
met circuit plan --checkpoint <id> --hardware <device> --precision declared
met circuit show --recipe <recipe> --hardware <device> --mode multi_seq --rows 16
met circuit display --recipe <recipe> --hardware <device>
```

- The class's golden plans (`kernels/circuits/plans/hw/<model>--<device>.md`) start as roofline
  projections: "shared-unmeasured" everywhere, because no record exists on the class.
- Profile a serve (mock first, then full) with nsys at C1, C16 and C128; map each planned group
  to its kernels; write the class's rows into `docs/kernel-perf/measurements.toml`
  (`hardware = "<class>"`) and the evidence into the class's `KERNEL_FAMILIES.toml` overlay.
  Only then is anything "shared-measured" on this class.
- A planned group whose kernels never appear in the profile means the planner and the runtime
  disagree: fix the rule (and its `ROUTING-AUDIT.md` citation) before trusting the plan.

## 5. The class's tensor-core policy

`kernels/<class>/HARDWARE.toml [tensor_core_policy]` (schema `crates/circuit/src/hardware/
tc_policy.rs`; gb10's is the worked example). It is NOT inherited.
- `require`: the matmul-class ops (`linear`, `lm_head`, `router`, `expert_gate_up`,
  `expert_down`, `paged_attention`) at the modes and minimum rows the class must run on tensor
  cores.
- `exempt`: every covered op the class's plans run off tensor cores today, with its kernels and
  its kind: `measured` (a CUDA-core kernel wins there; cite the evidence), `shape` (no MMA tile
  to fill), or `backlog` (a known gap). The backlog entries ARE the class's tensor-core work list.
- Regenerate the golden plans; `met circuit plan --matrix ... --check` then fails on any
  tensor-core regression in CI.

## 6. Measured `[benchmarks.limits]`

`[benchmarks.limits.{thermal,memory,timing,equivalence}]` decide when a box of the class is
parked for heat, how much free memory a self-started gate needs, how long a serve may take to
boot, and when two boxes count as one. `met benchmark certify` and every self-serving gate run
refuse a class without them (`crates/bench/src/hardware/limits.rs`). **Measure them on the
device** (die and chassis temperature under a sustained C128 load and at idle, boot time of
the largest recipe, free memory with the serve up, clock and memory spread across two boxes)
and declare them with the measurement beside each value. Never copy another class's numbers,
and never invent them before the device exists. The file is a closure input: changing a limit
re-opens every gate of the class.

## 7. Full weights

Pin the checkpoint revision. Stage the weights before the device clock starts when the device
is rented. Boot with the same flags the mock used, re-run `--check-kernels`, and check the
memory model: `met circuit memory --hardware <device> --recipe <recipe>` against the serve's
boot ledger.

## 7b. Bit parity on the real model

Tiers 1 and 2 of `references/bit-parity.md` on the real model: the Tier 1 checks again, logits
against the reference implementation within each format's declared tolerance, greedy
transcripts against the certified reference box with a match rate and a same-box control, and
byte identity with the reference box for every op whose arithmetic order is the same on both
classes. Record it and set `ttbp_real_at` / `ttbp_real_h` in the ledger. **The improvement loop
does not start before this.**

## 8. Same-box vLLM baseline

- The checked-in harness (`bench/ladder38/harness_w55_conc_ladder.py`, energy with
  `bench/ladder38/power_window.py`) against vLLM's published configuration for the checkpoint;
  the manifest shape is `bench/baselines/<model>/published.json` (box, harness sha, engine
  build, parity note per series).
- Record the vLLM version AND the image digest (`docker image inspect`), the launch command,
  and the effective engine args from its log.
- **PARITY-O.R.A.C.L.E first** (`.claude/agents/flag-parity-oracle.md`): no vLLM number is
  recorded until both sides' resolved configs are ruled effectively equivalent.
- C1-C128, one fresh serve per rung at high concurrency, a watchdog, no request timeout at
  C >= 64 on either side. A vLLM stall or crash at a rung is a result: record it with evidence,
  then a one-change mitigation ladder.

## 9. The improvement loop

`references/improvement-loop.md`, keeping bit parity at every iteration, until its exit
criterion holds (set `ttpv_at` / `ttpv_h` in the ledger) or its stop condition fires. It runs
after step 7b and before step 10: the accuracy bar is taken on the configuration that will be
certified.

## 10. Accuracy bar (Tier 3)

Before certification, and before any precision-changing lever is used in a published number,
at the declared precision, on the full model:
- BFCL: record N, the category sample pct and the SHA-256 of the ordered sample ids beside
  every score. A score without its draw is not comparable to anything.
- agentic-webserver with a same-night control, because its pass rate is noisy.
- A precision-lowering lever is a flag, default off, and needs this bar before it is used in a
  published number.

## 11. Certification

Recipes that set every serve variable explicitly (an inherited default once throttled a
published measurement to 8 sequences); BENCH.toml bounds measure-then-declare (an unmeasured
variant cannot bootstrap a gate); a measurement-definition change rides its own PR; then the
certification campaign as `AGENTS.md` describes.
