# The Latent Kernel Blueprint

**The goal.** Parameterize the differences between hardware classes and between kernels until a
basis of kernel *algorithms* remains that does not depend on the device. Each algorithm is kept
apart from its *schedule*, in the Halide sense: what is computed, as opposed to how it is tiled,
ordered and mapped onto a device. That basis is the **Latent Kernel Blueprint (LKB)**.

Contributors and agents then work against the LKB. They choose a family, a point, a policy and a
schedule instead of writing a kernel. This is single source of truth applied to kernels. It
shortens two intervals for every new hardware and model combination:

- **TTBP**, the time to bit parity;
- **TTPV**, the time to a performance and energy win.

Shortening both is what makes support on the day of a model's release possible.

**A shrinking residual means the LKB is converging.** Every hardware or model campaign should
leave the hardware-specific residual smaller than it found it.

[The Circuit Compiler](./circuit-compiler.md) describes the system the LKB lives in. [The LKB in
Mathematics](../appendix/lkb-math.md) states the same ideas as engineering rules: bracketing
trees and bit parity, composable error budgets, families as parametric morphisms, and fusion gain
as laxity.

## The registry

The LKB is not a separate artifact. Its registry is **`kernels/<class>/common/KERNEL_FAMILIES.toml`**,
with its schema in `crates/circuit/src/venn/families.rs`. It is resolved along the class chain like
every other class file:

- the nearest manifest is the base;
- a class's own manifest overrides families by id;
- evidence counts only on the class that measured it (`class_families` in
  `crates/circuit/src/hardware/class.rs`).

What already exists maps onto it directly:

| LKB concept | Where it lives |
|---|---|
| a family (one algorithm) | `[[family]]`: the ops it implements, its kernels, its compute unit and MMA atom, its numeric pipeline |
| its parameters | `[[family.param]]`, each with a kind |
| its points | `[[family.point]]`, with `how` = `instantiation`, `copy` or `branch` |
| its evidence | `[[family.evidence]]`, keyed to rows of `docs/kernel-perf/measurements.toml` |
| a schedule choice | the lowering rules in `FUSIONS.toml` (which point serves which rows), `HARDWARE.toml [defaults]`, the tile tiers |
| hardware facts the schedule reads | `kernels/DEVICES.toml` (per SKU), `HARDWARE.toml` (per class) |
| the LKB residual | the class's own sources in `kernels/<class>/common/`, per-model sources under `kernels/<class>/<model>/<quant>/`, `KERNEL.toml [shadow]` entries, `how = "copy"` points, `kernels/FORKS.md` |

## Families

A family is one algorithm for one op class, independent of the device it runs on. Examples are
"W4A16 GEMV", "paged decode attention" and "grouped MoE expert GEMM". A family records:

- the circuit ops it implements;
- the formats it accepts;
- the rows one launch covers;
- where its arithmetic runs (`compute`): tensor core with its MMA atom, CUDA core, or memory.

A kernel belongs to exactly one family. If two kernels compute the same thing on the same formats,
they are either one family with two points, or one of them is residual.

## Parameter kinds

| Kind | What it is | Defines a point? | Examples |
|---|---|---|---|
| **runtime** | a kernel argument that sizes nothing | no | strides, counts, eps, scale pointers, top-k |
| **compile-time** | sizes registers, shared memory or unrolling | yes (an instantiation) | head_dim, group size, tile shape, rows per CTA, pipeline depth |
| **policy** | a plug-in of the family's template | yes | weight format, scale layout, activation quantizer, epilogue, routing scoring |
| **numerics** | the order and precision of the arithmetic | yes | accumulation order (split-K count and reduction tree, K-loop order), accumulator dtype, rounding, FMA contraction, where scales apply |

**Numerics is a parameter, not an accident.** Two classes that run the same family at the same
numerics point produce the same bits, so cross-hardware bit parity becomes a property of the
point rather than a hope. It follows that:

- each family states its numerics point. The `pipeline` declarations already state the
  precision of each step (accumulate, scale). The split and reduction order belong beside them.
- where the cost allows, a new class picks the **same** numerics point as the reference class and
  gets byte identity with it for free. An example is a split-K count fixed by the model shape
  rather than derived from the SM count.
- where it does not (a different MMA atom, a split count the hardware needs), the point is a
  different numerics point and is declared as one. Parity then falls back to a declared tolerance
  and the transcript match rate.

**Not yet built.** `ParamKind` has `runtime`, `compile` and `policy` today. The pipeline
declarations record each step's precision, not its order, so a family's reduction tree is not
yet stated in data. A `numerics` kind with declared reduction trees is planned.

## Parity tiers

Bring-ups prove parity before any performance work:

- **Tier 1**: self-consistency on the target, bit-exact. This covers circuit vs legacy forward,
  eager vs graphed, determinism, and row invariance.
- **Tier 2**: reference correctness. Logits must fall within a declared tolerance, and transcripts
  are compared against a control. They are bit-exact wherever the arithmetic order allows.
- **Tier 3**: the accuracy bar.

The numerics point decides which form Tier 2 takes:
- same point as the reference: bytes;
- different point: a tolerance, derived as in [the LKB in mathematics](../appendix/lkb-math.md#3-rule-error-budgets-compose).

## The evidence envelope

A family's evidence envelope is the set of (class, point, row count) at which a microbench record
exists. "Optimized" is claimed only inside it. Outside it, a point is "shared, unmeasured",
whatever another class measured. The envelope grows only by measurement. A campaign's first job
on a new class is to bring the hot points (by share of the step) inside it.

## Promotion: how the LKB grows

A pattern joins the LKB, as a family point, a policy or a numerics point instead of a copy, when
both conditions hold:

1. **at least two hardware or model points use it** (two classes, or two models on one class);
2. the parameterized form is **byte-identical** at every existing point, shows **no microbench
   regression** there, and leaves the existing gates unchanged.

One user is a specialization; two users make a parameter. Promote when the second user appears,
not before. The LKB is **discovered bottom-up**, from kernels that exist and are measured. It is
never designed top-down from kernels nobody has written. A parameter with one value is allowed
only when the code is already structured to vary it.

## The LKB residual

The LKB residual is every kernel that is not, or not yet, a point of an LKB family on the class:

- a class's own source that replaces a parent's (`KERNEL.toml [shadow]`);
- a class's own addition that duplicates a family's algorithm with a different schedule;
- a per-model or per-point copy (`how = "copy"`).

The residual is not the residual stream: the word is always qualified as "LKB residual".

The residual is also not the same as an **uncovered op**. The Venn and the gap reports call an op
*novel* when no family lowers it at all. That is a gap in the LKB, and the fix is a new
generator. A residual kernel is the opposite case: a kernel that exists on the class but is not a
generator.

Residual is allowed, but **never silent**. Each entry is a named leaf override that records:
- why the family's point could not serve here;
- the measurement that shows it.

It is listed in the class's `KERNEL.toml [shadow]`, or as a family point with `how = "copy"`,
and in `kernels/FORKS.md`. A residual entry becomes a promotion candidate the moment a second
user appears.

## Convergence metrics

Each campaign records these metrics at its start and its exit:

| Metric | Definition | Direction |
|---|---|---|
| **LKB coverage** | the share of the step (at C1, C16, C128) run by kernels of LKB-registered families on this class, and its measured part (inside the evidence envelope) | up |
| **LKB residual** | count and lines of the class's hardware-specific kernels (its own sources, shadows, copies) | down |
| **parameterization yield** | lines deleted by parameterization, and families or points promoted | up |
| **TTBP**, **TTPV-speed**, **TTPV-energy** | time to bit parity; time to a speed win and to an energy win, measured separately | down |
| **zero-day readiness** | TTBP for a new model on already-supported hardware (no new class work) | down |

How to read them today, per (model, device):

- **LKB coverage.** `met circuit plan --checkpoint <id> --hardware <device> --precision declared`
  prints a line per gap table: "Shared X% (measured on this class), shared-unmeasured Y%,
  parameterisation Z%, policy variant W%, novel V% of the step". Coverage is `100 - V`, and its
  measured part is `X`.
  - Only planned kernels count. A class kernel that no lowering rule names is invisible to the
    plan.
  - So a class with a large residual can still show high coverage. Read the two metrics together
    until residual kernels are modelled in the class overlays.
- **LKB residual.** The class's `common/` sources and their lines, plus its `[shadow]` and `copy`
  entries.

Every campaign exit report states two lines:
```
Promoted into the LKB: <family / point / policy / numerics point>, ... (or "none")
LKB residual delta: <before count, lines> -> <after count, lines> (<+/-n>)
```

## Not yet built

- `met circuit lkb`, which prints LKB coverage, the evidence envelope and the LKB residual for a
  (model, hardware) pair and emits them as campaign-ledger fields. The coverage is already in the
  gap tables; the residual needs the class's source list, which `ClassSources`
  (`crates/circuit/src/hardware/sources.rs`) holds.
- `numerics` as a fourth parameter kind, with the split and reduction order of each point stated.
- A class's own kernels modelled in its `FUSIONS.toml` and `KERNEL_FAMILIES.toml` overlays. Until
  then they are residual by definition, and their share of the step is not measured.
