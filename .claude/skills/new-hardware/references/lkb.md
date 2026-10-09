# The Latent Kernel Blueprint (LKB)

**The meta-goal.** Parameterize hardware and kernel differences until an invariant kernel basis
emerges: the *algorithm* of each kernel family, separate from its *schedule* (in the Halide
sense: what is computed vs how it is tiled, ordered and mapped onto a device). Agents then work
against the LKB (choose a family, a point, a policy and a schedule) instead of writing kernels.
That is SSOT applied to kernels, and it is what drives time to bit parity (TTBP) and time to
performance/energy victory (TTPV) toward zero-day model support.

**A shrinking residual means the LKB is converging.** Every hardware + model campaign should
leave the hardware-specific residual smaller than it found it.

The same ideas stated as engineering rules grounded in the math (bracketing trees and bit
parity, composable error budgets, families as parametric morphisms, fusion gain as laxity, an
equality-saturation fuser experiment): `references/lkb-math.md`.

## The registry

The LKB is not a new artifact. Its registry is **`kernels/<class>/common/KERNEL_FAMILIES.toml`**
(schema `crates/circuit/src/venn/families.rs`), resolved along the class chain like every other
class file: the nearest manifest is the base, a class's own manifest overrides families by id,
and evidence counts only on the class that measured it (`crates/circuit/src/hardware/class.rs`
`class_families`). What already exists maps onto it directly:

| LKB concept | Where it lives today |
|---|---|
| a family (one algorithm) | `[[family]]`: the ops it implements, its kernels, its compute unit and MMA atom, its numeric pipeline |
| its parameters | `[[family.param]]` with a kind |
| its instantiated points | `[[family.point]]` with `how` = `instantiation`, `copy` or `branch` |
| its evidence | `[[family.evidence]]`, keyed to `docs/kernel-perf/measurements.toml` rows |
| a schedule choice | `FUSIONS.toml` rules (which point at which rows), `HARDWARE.toml [defaults]`, the WxAy engine's tile tiers |
| hardware facts the schedule reads | `kernels/DEVICES.toml` (per SKU), `HARDWARE.toml` (per class) |
| the residual | the class's own sources in `kernels/<class>/common/`, its `KERNEL.toml [shadow]` entries, `kernels/FORKS.md` |

## Families

A family is one algorithm for one op class (e.g. "W4A16 GEMV", "paged decode attention",
"grouped MoE expert GEMM"), independent of the device it runs on. It names the circuit ops it
implements, the formats it accepts, the rows one launch covers, and where its arithmetic runs
(`compute`: tensor core with its MMA atom, CUDA core, memory). A kernel belongs to exactly one
family. Two kernels that compute the same thing on the same formats are one family with two
points, or one of them is residual.

## Parameter kinds

| Kind | What it is | Defines a point? | Examples |
|---|---|---|---|
| **runtime** | a kernel argument that sizes nothing | no | strides, counts, eps, scale pointers, top-k |
| **compile-time** | sizes registers, shared memory or unrolling | yes (an instantiation) | head_dim, group size, tile shape, rows per CTA, pipeline depth |
| **policy** | a plug-in of the family's template | yes | weight format, scale layout, activation quantizer, epilogue, routing scoring (the WxAy engine's load_w / prep_w / mma / store) |
| **numerics** | the order and precision of the arithmetic | yes | accumulation order (split-K count and reduction tree, K-loop order), accumulator dtype, rounding (RNE, tie rule), FMA contraction, where scales apply |

**Numerics is a parameter, not an accident.** Two classes that run the same family at the same
numerics point produce the same bits: cross-hardware bit parity (Tier 2 in
`references/bit-parity.md`) is then a property of the point, not a hope. So:
- state each family's numerics point (the `pipeline` declarations already state the
  accumulation and scale steps; the split and reduction order belong beside them);
- where the cost allows, a new class picks the **same** numerics point as the reference class
  (e.g. a split-K count fixed by the model shape rather than derived from the SM count), and
  gets byte identity with the reference box for free;
- where it does not (a different MMA atom, a split count the hardware needs), the point is a
  different numerics point, declared as such, and parity falls back to the declared tolerance
  and the transcript match rate.

Today `ParamKind` has runtime, compile and policy; numerics is expressed through the families'
pipeline declarations. Making it a first-class kind is a follow-up (below).

## The evidence envelope

A family's evidence envelope is the set of (class, point, row count) at which a microbench
record exists. "Optimized" is claimed only inside it; outside it a point is "shared, unmeasured",
whatever another class measured. The envelope grows by measurement only, and a campaign's first
job on a new class is to put the hot points (by step share) inside it.

## Promotion: how the LKB grows

A pattern joins the LKB (becomes a family point, a policy or a numerics point instead of a
copy) when:
1. **at least two hardware/model points use it** (two classes, or two models on one class), and
2. the parameterized form is **byte-identical** at every existing point and shows **no
   microbench regression** there (the stability gate, `references/parameterization.md`), and
   the existing gates are unchanged.

One user is a specialization; two users are a parameter. Promote when the second user appears,
not before: the LKB is **discovered bottom-up** from kernels that exist and are measured, never
designed top-down from kernels nobody has written. A parameter with one value is allowed only
when the code is already structured to vary it.

## The residual

The residual is every kernel that is not (yet) a point of an LKB family on the class:
- a class's own source that replaces a parent's (`KERNEL.toml [shadow]`);
- a class's own addition that duplicates a family's algorithm with a different schedule;
- a per-model or per-point copy (`how = "copy"`).

Residual is allowed, but **never silent**: each entry is a named leaf override with its reason
and its evidence (why the family's point could not serve here, measured), listed in the class's
`KERNEL.toml [shadow]` or as a family point with `how = "copy"`, and in `kernels/FORKS.md`. A
residual entry is a promotion candidate the moment a second user appears.

Hopper's residual at campaign start: **17 sources, 4062 lines** in `kernels/hopper/common/`
(2 `[shadow]` replacements of the W8A16 GEMVs, 15 additions), none of them named by any
FUSIONS.toml rule or KERNEL_FAMILIES.toml family.

## Convergence metrics

Recorded per campaign in the ledger (`ledger/<class>.toml`) and stated in every exit report:

| Metric | Definition | Direction |
|---|---|---|
| **LKB coverage** | share of the step (at C1, C16, C128; roofline first, nsys once a serve runs) run by kernels of LKB-registered families on this class; and its measured part (inside the evidence envelope) | up |
| **residual** | count and lines of hardware-specific kernels of the class (its own sources, shadows, copies) | down |
| **parameterization yield** | lines deleted by parameterization, and families or points promoted | up |
| **TTBP**, **TTPV-speed**, **TTPV-energy** | `references/bit-parity.md`, `references/speed-and-energy.md` | down |
| **zero-day readiness** | TTBP for a new model on already-supported hardware (no new class work) | down |

How to read them today, per (model, device):
- LKB coverage: `met circuit plan --checkpoint <id> --hardware <device> --precision declared`
  prints, per gap table, "Shared X% (measured on this class), shared-unmeasured Y%,
  parameterisation Z%, policy variant W%, novel V% of the step": coverage is `100 - V`, its
  measured part is `X`. Only planned kernels count: a class kernel no rule names is not covered.
- residual: the class's `common/` sources and their lines, plus its `[shadow]` and `copy` entries.

Every campaign exit report states two lines:
```
Promoted into the LKB: <family / point / policy / numerics point>, ... (or "none")
Residual delta: <before count, lines> -> <after count, lines> (<+/-n>)
```

## Follow-ups

- `met circuit venn` / `plan` printing LKB coverage and the residual for a (model, hardware)
  pair directly (the coverage is already in the gap tables; the residual needs the class's
  source list, which `ClassSources` holds).
- `numerics` as a fourth `ParamKind`, with the split and reduction order of each point stated.
- A class's own kernels modelled in its FUSIONS/KERNEL_FAMILIES overlays (for Hopper, audit
  item 6 of `targets/h100.md`): until then they are residual by definition.
