# ADR-0020: Kernels and architectures are built against shared blueprints

**Status:** Accepted
**Date:** 2026-10-05

## Context

The engine's kernel tree is organized per `(hardware, model, quantization)` target. Earlier
design text described the targets as sharing no kernels at all ("specialization is a directory,
not a template"). In practice, sources are shared through `kernels/<class>/common/`,
`[sources] use` and `[hardware] inherits`. The circuit compiler also plans every model on every
hardware class from one set of kernel families (`KERNEL_FAMILIES.toml`) and one set of lowering
rules (`FUSIONS.toml`).

Without an explicit rule, a new hardware class or model tends to add copies: per-head-dim
kernel files, per-class rewrites of an inherited kernel, per-model circuit blocks. Each copy
drifts and has to be measured and maintained on its own.

## Decision

- **Kernels.** Every kernel is either a point of a kernel family in the Latent Kernel Blueprint
  (an instantiation of its compile-time, policy or numerics parameters), or a named entry in the
  LKB residual. A residual entry is a `KERNEL.toml [shadow]` entry, a `how = "copy"` point, or
  a class-only source, and it records its reason and evidence. There is no third kind.
- **Architectures.** Every architecture is a layout of block families of the Latent
  Architecture Blueprint at a parameter point derived from the checkpoint, or it is refused with
  a reason (the architecture residual).
- **Promotion.** A pattern becomes a parameter when a second hardware or model point uses it, and
  only if the parameterized form is byte-identical at every existing point (golden plans for
  architectures, microbenches and parity for kernels) with no measured regression.
- **Metrics.** Every hardware or model campaign records, at its start and its exit:
  - LKB coverage (with its measured part), the LKB residual, and promotions;
  - architecture coverage, models expressible as parameter points, the LAB residual, and LAB
    promotions.

The definitions are in the book: [The Latent Kernel Blueprint](../../book/src/architecture/lkb.md),
[The Latent Architecture Blueprint](../../book/src/architecture/lab.md),
[The LKB in Mathematics](../../book/src/appendix/lkb-math.md).

## Consequences

- A new class or model starts by measuring how much of it the blueprints already express (the
  Venn and the plan reports), and it reports the residual it adds.
- Copies are allowed, but only as named residual with evidence. Each copy becomes a promotion
  candidate when a second user appears.
- Performance claims stay local to a target: a shared point counts as "optimized" on a class only
  where it was measured there.
- The book's category-theory appendix was rewritten to match. Its earlier reading, that the
  targets form a discrete category and the kernels are not factored through anything, is
  retired.
