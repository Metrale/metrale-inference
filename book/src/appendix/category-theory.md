# A Category-Theoretic Perspective

The Metrale Engine book argues its case in prose. The claim is narrower than "fast everywhere":

- every `(Hardware, Model, Quantization)` target gets kernels chosen, compiled and measured for
  that target;
- the kernel *algorithms* are shared across targets wherever sharing is byte-identical and
  costs nothing measurable;
- what cannot be shared is kept as a named, measured exception.

Category theory gives precise names for the structures that claim relies on, and this appendix
names them. It is not a proof of performance, not a tutorial in category theory, and not
required reading for anyone who wants to run or extend Metrale Engine. It is a lens.

Standard references for the underlying mathematics: Saunders Mac Lane, *Categories for the
Working Mathematician* (second edition); Emily Riehl, *Category Theory in Context* (freely
available). Sections 1–6 use only the first two chapters of either. The kernel structure in
Section 3 is worked out at engineering length in [The LKB in Mathematics](./lkb-math.md).

---

## 1. Targets, and where performance claims live

A **category** is a collection of objects together with arrows (morphisms) between them. It is
closed under composition and has an identity arrow on every object.

Metrale Engine's targets form a collection `𝒯` with one object per supported `(H, M, q)` triple.
In code these objects are `metrale_core::target::KernelTarget` values, one per compiled
`TargetPtxSet`. The leaf directories `kernels/<hw>/<model>/<quant>/` list them.

Earlier editions of this appendix made `𝒯` *discrete*: no arrows between distinct targets, and
no shared structure between their kernels. That is no longer how the engine is built.

- Kernel sources are shared through `kernels/<hw>/common/`, `[sources] use` and
  `[hardware] inherits`.
- The circuit compiler plans every model on every class from one set of kernel families.

What survives of the discrete reading is a statement about **evidence**, not about code:

> **A performance claim is local to an object.** A kernel measured on one target is
> "shared, unmeasured" on any other target until it is measured there.

The [evidence envelope](../architecture/lkb.md#the-evidence-envelope) is that rule in data:
`[[family.evidence]]` counts only on the class that recorded it.

## 2. `𝒯` as a product

The three axes decompose: `𝒯 ≅ Hw × Mod × Quant`. The product comes with projections
`π_Hw`, `π_Mod` and `π_Quant`, each of which reads off one coordinate. A **functor** is a
structure-preserving map between categories: it sends objects to objects and arrows to arrows,
and it respects identities and composition.

The product decomposition is visible in three places:

- **The directory tree.** `kernels/<hw>/<model>/<quant>/` mirrors the three factors.
- **The build-time wildcards.** `METRALE_TARGET_HW`, `METRALE_TARGET_MODEL` and
  `METRALE_TARGET_QUANT` in `crates/kernels/build.rs` select subsets of each factor independently.
- **The workspace crate split.**
  - Hw axis: `metrale-gpu-runtime` and `metrale-comm`.
  - Mod axis: `metrale-model-arch` and the architecture circuits.
  - Quant axis: `crates/model-layers/src/quant_format/` and `metrale-kernels`.

Orthogonal axes are the defining property of a product. Adding an object to `Hw` does not
change `Mod × Quant`.

## 3. Kernels factor through the Latent Kernel Blueprint

The kernel structure has two layers: a theory shared by every target, and a realization of it on
each hardware class.

**The theory.** The [Latent Kernel Blueprint](../architecture/lkb.md) (LKB) is presented by:
- **generators**: the kernel families of `KERNEL_FAMILIES.toml`, each with parameters and
  points;
- **relations**: the fusion rules proven bit-identical (`numerics = "bit_identical"` in
  `FUSIONS.toml`).

**Realization.** Each hardware class `H` gives a functor `F_H : LKB → Impl_H`. It turns family
points into launchable kernels through the class chain (`HARDWARE.toml inherits`), the class's
overlays, and the kernels it can run.

**The plan.** A model's circuit is lowered onto the LKB by the fuser (`L_H`). The plan on `H` is
`F_H ∘ L_H` applied to the circuit, and `met circuit plan --hardware` prints it.

**The residual.** Not every kernel on a class is the image of a generator. The **LKB residual**
of `H` is the part of `Impl_H` outside the image of `F_H`: shadows, class-only sources and
per-point copies. It is kept explicit and named, with its evidence. So the kernel set of a target
decomposes as

```text
Kernels(H, M, q)  =  F_H( points the plan of (M, q) uses )  ⊔  LKB-residual(H) used by (M, q)
```

**The engine's direction** is to shrink the right-hand summand: a new class should need a new
functor (data), not new generators (kernels). Each campaign reports the residual's size, so
convergence is a measured trend, not an assertion.

**How exact this is.** The reading is exact for the plan's *structure*: which kernels, composed
how. It is approximate for its *numbers*. Floating-point addition is not associative, so `F_H`
preserves a bit-identical relation only when both sides keep the same reduction order on `H`.
Otherwise it preserves the relation up to a derived error bound. [The LKB in
Mathematics](./lkb-math.md) states both cases as rules (Sections 2 and 3).

## 4. Build-to-runtime as a composition of functors

Three categories and two functors sit in a line:

```text
Sources  ──[ComputeTarget.compile]──►  Binaries  ──[embed + load]──►  KernelHandles
```

- `Sources` has one object per target. Its underlying data is the staged `.cu` / `.metal` /
  `.hip` files: the class's `common/` layer plus the target's own leaf directory.
- `Binaries` holds the compiled PTX / metallib / HSACO blobs.
- `KernelHandles` holds the runtime-resident entries returned by
  `GpuBackend::kernel(module, function)`.

The first arrow is the `ComputeTarget` trait in `crates/core/src/compute.rs`. It is a
vendor-indexed family of functors, one per `Vendor`. The rest of the diagram does not depend on
how the binaries were produced.

The factoring in Section 3 happens at the `Sources` level and in the plan, **ahead of time**.
Every point a target uses is instantiated and compiled for that target at build time. Nothing in
the runtime chooses among points by dispatching over shapes or types, and nothing is compiled
just in time.

## 5. The `GpuBackend` trait as an algebraic theory

An **algebraic theory** is a signature (operation symbols with arities) plus equations that every
implementation must satisfy. A **model** of the theory is a set with operations that satisfy the
equations.

The `GpuBackend` trait in `crates/gpu-runtime/src/gpu.rs` is such a theory.
- Its operations are `alloc`, `free`, `kernel`, `launch`, `synchronize`, `copy_h2d`, and so on.
- Its equations are unwritten but real, for example "`synchronize` serialises previously launched
  work on the given stream".

Two models ship:
- `MetraleCudaBackend` implements the theory with the CUDA driver API;
- `MockGpuBackend` records launches and returns the results the equations demand.

The business logic (scheduler, engine, layer code) is polymorphic over the choice of model.
`cargo test` evaluates it in `MockGpuBackend`, and production evaluates it in
`MetraleCudaBackend`. This is the formal meaning of [SBIO](../architecture/sbio.md): business
logic never performs I/O directly because it never commits to a model.

Section 3 uses the same idea one level down. The LKB is a theory, and each hardware class is a
model of it.

## 6. The compiled registry as a coproduct

A **coproduct** in `𝐒𝐞𝐭` is a disjoint union:

```text
all_ptx  ≅  ∐_{(H,M,q) ∈ 𝒯}  Binaries(H, M, q)
```

`metrale_kernels::all_ptx_sets()` returns it. Each target contributes one summand of compiled
modules, compiled for that target even where its sources are shared. Adding a target adds a
summand and leaves the existing summands unchanged.

Sharing therefore lives in the sources and the plan (Section 3). The compiled artifacts stay
per target (this section).

## 7. Where general frameworks sit in this picture

A general framework also factors its kernels through a smaller category of shared kernels.
Common forms:
- shape polymorphism inside one kernel;
- dtype dispatch on a runtime tag;
- just-in-time specialization on first call.

The LKB is a factoring too. The differences are where and how it is resolved:

1. **Ahead of time.** Every point is instantiated and compiled per target at build time, and the
   plan is fixed per (model, class, mode, rows) before serving. No runtime dispatch chooses among
   points.
2. **Measured per object.** A shared point is "optimized" on a target only inside its evidence
   envelope there (Section 1).
3. **Opt-out per kernel.** A target may replace any point with a residual kernel. It must name
   the kernel and give its evidence, and the replacement is counted.
4. **Numerics are a parameter.** Where two targets share a point's reduction order, their results
   are byte-identical. Where they do not, the difference is declared and bounded (Section 3).

Whether a particular framework's factoring costs speed on a particular target is an empirical
question, answered per target by [benchmarks](../operations/benchmarks.md), not by this appendix.

## 8. Reading the book through this lens

- The [Part II philosophy chapter](../architecture/philosophy.md) shows the product structure in
  code: the kernel tree is the coordinate system, and the crate split is the decomposition.
- [The Circuit Compiler](../architecture/circuit-compiler.md) and [the
  LKB](../architecture/lkb.md) describe Section 3 operationally: circuits, lowering, families,
  realization, and the residual.
- The [dispatch chapter](../architecture/dispatch.md) traces a request through the functor
  composition of Section 4.
- The [SBIO chapter](../architecture/sbio.md) is the operational version of Section 5.
- The [deep-dive chapters](../deep-dives/kernels.md) describe kernels at a single object each.

The lens supplies questions to ask in review:
- "Is this new kernel the image of an existing generator, or a named residual?"
- "Does this change keep the reduction order of a point, so parity stays byte-exact?"
- "Does this couple two factors of `Hw × Mod × Quant`?"

## 9. What this perspective does not prove

Category theory names structures. It does not measure throughput, verify kernel correctness or
port the engine to a new vendor. Everything above follows from the code being organized along
these lines. The formalism describes that organization and does none of the work.

- **Performance** is empirical. See [Benchmarking](../operations/benchmarks.md).
- **Correctness** is tested. Bit parity comes first, then accuracy. See
  [The LKB](../architecture/lkb.md#parity-tiers) and [Contributing](../project/contributing.md).
- **Porting a vendor** is design work. The categorical answer names the pieces (a `ComputeTarget`
  impl, a `GpuBackend` impl, a realization of the LKB, a residual) but not the effort.

---

**Further reading.** For the mathematics: Mac Lane, *Categories for the Working
Mathematician*, chapters I–III; Riehl, *Category Theory in Context*, chapters 1–4; and the
sources listed in [The LKB in Mathematics](./lkb-math.md#sources). For the engineering:
[Philosophy (Part II)](../architecture/philosophy.md), [The Circuit
Compiler](../architecture/circuit-compiler.md), [Kernel Dispatch
Pipeline](../architecture/dispatch.md), [SBIO](../architecture/sbio.md).
