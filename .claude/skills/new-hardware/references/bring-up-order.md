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
- **Run `met bench preflight` before EVERY timed run from here on** (step 8's baseline, every
  iteration of step 9's loop, step 11's certification) — `--expect-head <sha>` catches a stale
  pre-head binary, and the kernel-freshness check recomputes this class's kernel closure hash
  from the working tree and compares it to the binary's own baked attestation, which is exactly
  the shared-`CARGO_TARGET_DIR` failure this section already warns about, now refused
  automatically instead of merely hoped for.

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
- **Fidelity verdict** (measured 2026-10-05 on the GB10 reference box: the dense
  Qwen3.8-27B NVFP4, the same serve command with and without `--mock`, mock `[1, 1]` = 8 of 64
  layers; n = 3 interleaved fresh serves per arm; die at or below 55 °C before every serve):
  - **Route: identical.** The same 52 kernels in the same 119 launch configurations (name,
    block, grid, shared memory) at C1 and C16 decode under nsys, and the same `--check-kernels`
    report (325 lookups, same kernel-set hash). The mock iterates on the kernels the real model
    runs. Check this once per model and after any routing change.
  - **Speed: trusted for direction and ranking, scaled.** A per-layer lever
    (`--ssm-batched-recurrent off`) moved the real model -8.1 % (C16) and -11.1 % (C32) tok/s
    and the mock -4.9 % / -8.0 % (-5.4 % / -7.2 % on a value-statistics mock), with C1 unmoved
    on both (the control). Relative deltas do not transfer 1:1, because fixed costs (head,
    embedding, host) are a larger share of the mock's step: compare absolute per-step deltas
    times the layer ratio instead, which came to 0.75-0.87 of the real one. A null lever
    (`METRALE_NO_W4A16_TC=1`, inert in this configuration) read within ±2 % on both. Absolute
    numbers extrapolated from mocks `[1, 1]`, `[2, 1]`, `[1, 2]` (`met ml-utils extrapolate`)
    landed within +2.3..+4.2 % of the real tok/s at C16/C32, +3.3..+5.8 % at C1, and within
    ±1.8 % for cold and warm 1k TTFT.
  - **Energy: not trusted, absolute or relative.** Mock power matched the real model at
    C16/C32 (0.99x) but read 16 % low at C1, where the real model streams 8x the weight bytes;
    extrapolated J/tok was within -1.8 % (C16) and -4.7 % (C32) but -15 % at C1. The same lever
    moved real J/tok +2.3 % / +3.2 % and the mock +1.8 % / -0.2 % (+0.6 % / +0.7 % on the
    value-statistics mock): one sign wrong, magnitudes 0-1x, all inside the 1.5-3 % serve
    spread. The value-statistics mock (`values.mode = "stats"`, on this branch) changed
    nothing measurable for the dense model, so this gap is the mock's step mix, not its
    weight values. (An earlier round on the MoE models found mocks drawing 67-74 W against
    52-62 W at C16-C64 and J/tok missing by 10-60 %: there, values and routing matter.)
  - **So, each loop iteration:** use the mock to find and rank levers and to confirm the
    route; confirm every kept lever's speed magnitude, and every energy number, on the real
    model by dropping `--mock`. Never quote a mock's J/tok.
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
  and the effective engine args from its log. `met bench preflight --vllm-image <ref>@sha256:...`
  refuses a bare tag (`:latest` or any tag with no `@sha256:...`) and verifies the local image's
  digest matches before the baseline leg runs — the standing defense against a `:latest` tag
  moving versions under a baseline without anyone noticing.
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
