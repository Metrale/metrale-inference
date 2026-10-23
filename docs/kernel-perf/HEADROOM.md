# Headroom: how far each kernel is from its ceiling

`met circuit headroom` answers, for one step of a plan on a hardware class:

- how far each **launch site** is from its floor;
- what the step and tok/s would be if that site alone reached its floor, or a stated fraction
  of it (Amdahl's law), and the cumulative curve when the top N sites all do;
- where the rest of the time goes: kernels outside the plan, overlap, and the **residual**
  (time no kernel runs: host work, syncs, waits, launch gaps). This is the LKB residual that
  `book/src/appendix/lkb-math.md` section 5 lists as not yet computed.

The code is `crates/circuit/src/hardware/headroom*.rs` (pure, `no_std`). The CLI is
`crates/server/src/cli/circuit_headroom.rs`. The capture exporter is
`scripts/headroom_trace.py`.

## Usage

```bash
# Plan only: floors, bounds and the step ceiling. Every site reads "unmeasured" with its
# projection.
met circuit headroom --recipe glm-5.3/glm-5.3-flash-nvfp4-tp3-512k --hardware gb10 \
  --mode verify --rows 2 --also draft:1

# With a capture: measured time per site, upside, residual.
met circuit headroom --recipe glm-5.3/glm-5.3-flash-nvfp4-tp3-512k --hardware gb10 \
  --mode verify --rows 2 --also draft:1 \
  --trace docs/kernel-perf/traces/glm-5.3-flash-nvfp4-tp3--gb10--f13-p1k.toml \
  --target-pct 80 --format md|json|svg-roofline|svg-waterfall [--out <rel> [--check]]

# Straight from an nsys SQLite export (runs scripts/headroom_trace.py):
met circuit headroom ... --trace run.sqlite --step-kernel screen_head_scan \
  --attribution scripts/headroom/glm-5.3-flash-nvfp4-tp3--gb10.toml
```

The flags:

- `--also <mode>:<rows>` adds another forward to the step, e.g. the MTP draft head.
- `--trace` can be repeated to pool captures of one configuration (see Statistics).
- `--kv-tokens`, `--tokens-per-step` and `--launch-floor-us` state inputs the capture may also
  state. If both give a value and the values differ, the command fails.
- With no stated KV length, the class's `[roofline] context_tokens` assumption is used.
- With no stated tokens per step, the main forward's rows are used (for a verify, this assumes
  every draft is accepted).
- With no launch floor, no site is classed launch-bound. The report says which of these
  applied.

The checked-in reports are regenerated and compared by `cargo test -p met circuit_headroom`. To
rewrite them, run `-- --ignored regenerate_headroom`.

## Where the reports live, and why not in the hw reports

The reports are in `kernels/circuits/plans/headroom/<model>--<device>--<capture>.{md,json,roofline.svg,waterfall.svg}`.
The captures they read are in `docs/kernel-perf/traces/`.

They are kept out of `kernels/circuits/plans/hw/*.md` because of SSOT:

- A hw report is a function of the tree alone (the plan, families and roofline). It is
  regenerated whenever the tree changes, and its `--check` test holds all 60 reports to that.
- A headroom report is a function of the tree **and of one capture** (a fingerprinted
  measurement of one run). Putting measured numbers into the hw reports would make every hw
  report depend on capture files. It would also duplicate the hw gap table's per-node estimate
  in a second form, and every new capture would invalidate reports that do not use it.

So the hw reports keep the plan-only estimate. The headroom reports join a plan with one
capture, and they name both. A plan-only headroom (`--trace` omitted) needs no file: it is a
command. Its floors are the per-launch-site form of the numbers the hw report already prints.

## The model

**Launch site.** One plan group template across its layers. It has the same nodes and the same
kernels, and any rule may have formed it: rules split by layer kind that launch the same kernels
for the same nodes count as one site. The site name follows the hw report's "Fused plan" table,
for example `kda [q,k,v,f_a,b,g_a]`.

**Floor.** The floor is `max(bytes / DRAM bandwidth, Σ FLOPs / peak of each node's MMA class)`.
The constants come from the class's `[roofline]` (measured achievable on GB10). The cost model
is `venn::roofline::node_cost`, the same one the hw reports use.

- A single-launch site's floor drops the edges the plan fuses inside it. Their write and read
  are what the fusion saves; this is the "laxity" of lkb-math.md section 5.
- A multi-kernel site counts every edge, because its kernels hand intermediates through memory.
- `node sum` is the hw report's per-node estimate of the same nodes. `node sum - floor` is the
  fusion's modelled gain.
- With a launch floor stated, a site whose roofline floor is below `launches × launch floor` is
  **launch-bound**, and that product becomes its floor.

**Measured.** A site's time comes only from a capture's kernels that were attributed to it.

- `total` is Σ launch durations per step, as `[[m]]` rows define `time_us`. It includes PDL and
  fork waits.
- `exclusive` is the part of the total that no earlier-launched kernel overlaps.
- A site with no capture is **unmeasured**. It shows its projection (its floor) and enters no
  measured sum, pct-of-floor or upside.

**Budget.** Measured step = Σ sites' exclusive time + Σ outside-the-plan kernels' exclusive time
+ residual. The identity is exact, and the tests hold it. The overlap (Σ durations − Σ
exclusive) is reported separately and is not added.

**Upside.** If a site falls from `t` to `floor / p`, it saves `min(t - floor / p, exclusive)`.
This assumes the time it already overlapped stays overlapped. It is therefore an **upper bound**:
the most the step can gain from that site alone. Exclusive times are disjoint, so savings add
along the cumulative curve. tok/s = tokens per step / step.

**Step ceiling.** The ceiling is Σ site floors, which includes the unmeasured sites'
projections. Kernels outside the plan have no floor, so the ceiling is the plan's ceiling and
not the capture's.

**Attribution.** The capture says which site a kernel belongs to (`site`, from its launch shape).
The core accepts the attribution when the kernel is:

- one of the site's kernels;
- in the same family as one of them;
- in a family that implements one of the site's ops;
- running at a placeholder site.

Every case except the first is listed under "Plan vs capture". Anything else is **refused**, and
every mismatch is listed. A kernel without a site is matched by name if exactly one launch site
runs it. Otherwise it is reported "outside the plan" (unknown or ambiguous), and its time stays
in the budget.

**Energy.** The energy columns (µJ/launch, % of the energy floor, energy upside) are filled only
from captured `energy_uj` and an energy roofline (pJ/byte, pJ/FLOP). Neither exists yet: see the
kernel-energy design. Until then the columns read `–`, never 0.

## Reading a report: the caveats it states

- **Precision or lowering drift.** The floor is the plan's lowering at the plan's formats. Where
  the serve lowers differently, the site's pct-of-floor measures the lowering, not the kernel.
  Examples on GLM:
  - DSA `q_b` and `o` absorb `kv_b_proj`, so the serve reads 2× the plan's bytes there and none
    at `attend`.
  - The INT8 screen LM head and the FP8 draft head run where the plan has BF16 heads.

  The attribution map leaves precision mismatches outside the plan and notes lowering moves.
  A pct-of-floor above 100 always means work moved out of the site.
- **Tensor parallelism.** The circuit plans the whole model. A one-rank capture states the rank's
  shard dims (`[plan] dims`), and the plan is made at those dims.
- **Routed experts.** The floor uses uniform routing (`E(1-(1-k/E)^T)` distinct experts), not
  the union the capture actually swept.
- **Single capture.** A report from one capture is labelled preliminary. It carries that
  capture's fingerprint (measurement discipline, rules 1 and 5). Its intervals cover
  step-to-step variation inside that capture, not run-to-run variation (see Statistics).

## Statistics: the intervals and what they cover

Every measured site time, its pct-of-floor and upside, and the step, tok/s and residual carry
a 95% interval, computed in `crates/circuit/src/hardware/headroom_ci.rs`:

- **Per-step series.** The exporter writes `step_us` and, per kernel, `step_total_us` and
  `step_exclusive_us` (exclusive time split across the steps it covers). `Trace::check` refuses
  series whose sums disagree with the totals.
- **Moving-block bootstrap** (Künsch 1989). Steps are autocorrelated, so steps are resampled in
  blocks of `ceil(n^(1/3))` consecutive steps (Hall, Horowitz, Jing 1995), with 1000 replicates,
  seed `0x68656164726f6f6d` ("headroom", SplitMix64) and a percentile interval. Each replicate
  resamples the **same step indices for every series**, so the joint quantities (upside,
  residual, tok/s at the target) keep their covariance.
- **Delta-method cross-check.** pct-of-floor and the residual also get a normal interval from
  the Newey-West (HAC) variance of the mean, with lag = block. The report prints both side by
  side; a large disagreement means the bootstrap is not to be trusted at this n.
- **Point estimates are window means**, because the step budget is additive only in means. The
  JSON also carries each site's per-step **median** total and its interval (`median_total_us`,
  `median_total_ci`), which is robust to a slow step.
- **Ranking.** Sites are ranked by the point upside. Where neighbouring upside intervals overlap,
  the rank is flagged `≈` (tied): the order between them is not established.
- **Pooling.** `--trace` can be repeated. Captures must agree on the step kernel, the plan dims,
  the KV length and tokens per step, or they are refused ("would mix configurations"). Each
  capture is one segment, and bootstrap blocks never cross segments. The header then says
  "n=k captures pooled". Run-to-run variance needs k ≥ 2 independent serves.
- **No series, no interval.** A capture without series is labelled "preliminary, n=1 (no CI)".
  Its interval cells read `–`, and no interval is ever invented.

**Consolidation (one source of truth).** `crates/circuit/src/hardware/stats.rs` is interim. The
shared crate `metrale-stats` (branch `kernel-energy`, a placeholder as of 2026-10-22) is to hold
BCa and block bootstrap, HAC ESS, mixed effects, TOST and Holm/BH. `stats.rs` uses the names it
is to export: `Interval`, `SplitMix64`, `median`, `block_len`, `block_resample`,
`percentile_ci`, `hac_variance_of_mean`. The merge has five steps:

1. Move these functions and their tests into `metrale-stats`, or map them onto its equivalents.
2. Add the dependency to metrale-circuit (both are no_std).
3. Change the `use super::stats::…` line in `headroom_ci.rs`.
4. Delete `stats.rs`.
5. Check that the golden headroom reports are byte-identical. With the same seed and RNG they
   must be; if a different RNG is adopted, regenerate the reports deliberately.

## Tools that complement it, per platform, and how they feed `--trace`

The contract is the trace TOML (`crates/circuit/src/hardware/headroom_trace.rs`). It holds:

- per kernel: `kernel`, `calls`, `total_us`, `exclusive_us`, and optionally `site`, `label`,
  `note` and `energy_uj`;
- for the window: `steps` and `window_us`;
- the fingerprint `source`.

Any profiler whose per-launch start and end times you can read can produce it.

| platform | timeline (feeds `--trace`) | counters / roofline (validates the byte and FLOP model) |
|---|---|---|
| GB10 (DGX Spark) | `nsys launch --trace cuda,nvtx --cuda-graph-trace=node` on the serve, `nsys start/stop` around a mid-decode window, `nsys export --type sqlite`, then `scripts/headroom_trace.py` (or `--trace x.sqlite`) | **Never attach ncu to a serve on GB10: it froze dgx1 and needed a power cycle.** Run `ncu --set roofline` only on a standalone microbenchmark of the one kernel (crates/gpu-examples), with no serve running |
| H100 / B200 / GB300 | the same nsys flow | `ncu --set roofline --section SpeedOfLight_RooflineChart -k regex:<kernel>` on a short serve or microbench. Compare its dram bytes and FLOPs with the site's `bytes` / `flops` columns |
| gfx1151 (Strix Halo) | `rocprofv3 --kernel-trace --output-format csv` (start/end per dispatch). Convert to the trace TOML; there is no converter yet | `rocprof-compute` (Omniperf) roofline on the kernel |
| Apple (Metal) | Instruments `Metal System Trace` (`xctrace record --template 'Metal System Trace'`, `xctrace export`). Convert per encoder to the trace TOML; there is no converter yet | Xcode's Metal debugger GPU counters (limiter and utilization per encoder) |

On every platform, the step is the window between two launches of a kernel that runs once per
step (`--step-kernel`). Exclusive time is computed from start and end times, so a timeline tool
is enough. Counter tools validate the floors; they do not feed the budget.

## What the Blueprint Store needs next

- A reader for `schemas/headroom.v1.json` (its own schema; the kbp card and graph schemas are
  unchanged).
- A join from a headroom site to the graph's groups by `(run, site)`. The graph names groups by
  plan group, and a headroom site is a group template, so the join is many-to-one.
- A capture registry: where trace TOMLs are published, and which plan and recipe each one
  belongs to.
- The energy columns, when the kernel-energy rows exist. The JSON already carries them as
  `null`.
