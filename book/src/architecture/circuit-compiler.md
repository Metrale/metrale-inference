# The Circuit Compiler

The circuit compiler turns a checkpoint into the exact list of kernel launches that run it on a
given hardware class. It is the crate `metrale-circuit` (`crates/circuit`) and the `met circuit`
commands built on it. The crate is pure: it does no GPU work, file I/O, environment reads, clock
reads or random draws. Callers read the TOML files and pass their text in, and every plan is
deterministic and carries a digest.

This chapter names each stage and the vocabulary the rest of the book and the code use for it.
[The Latent Kernel Blueprint](./lkb.md) covers the kernel side in depth. [The Latent
Architecture Blueprint](./lab.md) covers the model side. [The LKB in
Mathematics](../appendix/lkb-math.md) states both precisely.

## The stages

```text
config.json ──config map──► circuit instance ──lowering──► kernel groups ──realization──► plan
 (checkpoint)   (<arch>.config.toml)  (a Σ-diagram)   (fuser + FUSIONS.toml)  (LKB points)  (class H)
                                                                                         │
                                                         circuit executor (--forward circuit)
```

| Stage | What it is | Where |
|---|---|---|
| **Op signature `Σ`** | the closed vocabulary circuits are written in: about thirty ops plus `linear(role)` and `act_quant(format)` | `OpKind` in `crates/circuit/src/ir.rs` |
| **Circuit** | a string diagram over `Σ`: ops wired by typed edges in execution order, organized in blocks and laid out per layer kind | `kernels/circuits/<arch>.toml`, `kernels/circuits/blocks/` |
| **Config map** | reads a checkpoint's `config.json` into the circuit's dims, switches and parameters; every key is classified or the checkpoint is refused | `kernels/circuits/<arch>.config.toml`, `crates/circuit/src/config_map.rs` |
| **Instance** | a circuit instantiated at one shape and one precision (declared by the checkpoint, or set by a serving policy) | `crates/circuit/src/instantiate.rs`, `kernels/circuits/INSTANCES.toml` |
| **Lowering** | the fuser covers the instance with groups, each produced by one lowering rule that maps an op pattern to kernels | `crates/circuit/src/fuser.rs`, `kernels/<class>/common/FUSIONS.toml` |
| **Kernel families** | the generators of the LKB: one algorithm each, with parameters and instantiated points | `kernels/<class>/common/KERNEL_FAMILIES.toml` |
| **Realization** | a hardware class turns family points into kernels it can launch, along its inheritance chain | `kernels/<class>/HARDWARE.toml`, `crates/circuit/src/hardware/` |
| **Plan** | the realized lowering for one (model, class, mode, rows), with its digest | `met circuit show`, `met circuit plan --hardware` |
| **Executor** | runs a plan: `met serve --forward circuit` | `crates/model-layers/src/circuit_exec/` |

## Tensor types

An edge carries a tensor type: rows × a dim expression × a format (`crates/circuit/src/format.rs`).
A format carries its scale granularity (per tensor, per token, per channel, per group or per
block). Memory layout is not yet part of the type. Layout choices are made inside kernels and in
the buffer planner, which assigns arena offsets and is a different thing from tensor layout.

## Lowering rules and their numerics tags

A rule in `FUSIONS.toml` is a **lowering clause**. Under the modes, row counts and serving
settings it names, it maps a pattern of ops to one or more kernels. The fuser applies rules
greedily by priority, and every node ends up in exactly one group. Every rule carries a
`numerics` tag that states how its kernel relates to running the same ops one by one:

| Tag | Meaning | Status in the LKB |
|---|---|---|
| `bit_identical` | byte-identical to the unfused chain, proven by the named microtest | a **relation**: an equation between diagrams |
| `reference` | the engine's default for this chain; its kernel **defines** the reference numerics | definitional, not claimed equal to the unfused chain |
| `differs` | a different numerics point; selected only when its `lever` is opted in | a parameter change, disclosed on records |

The tag is part of the plan digest. `met circuit diff` checks byte parity between the
hand-written forward and the circuit forward: first with reference rules only, then with every
rule.

### When a relation may become the default

A `bit_identical` rule produces the same bytes as the chain it replaces, so it cannot change
accuracy. It may therefore become the default lowering (its `when` setting turned on in the
serving policies) on two pieces of evidence, with no accuracy gate:

1. **Byte identity.** Its microtest passes, and `met circuit diff` shows identical logits with
   the rule on and off.
2. **No regression in speed and no regression in energy.** Measure with a same-box A/B: the
   same binary, the same box, the rule on vs off, across the concurrency ladder the gates use,
   with a control leg. Speed and energy are measured separately (`J = ∫ P dt`; one does not
   imply the other).

A relation that wins on one and loses on the other stays opt-in, as a flagged choice that
records both deltas. The same rule appears in the `FUSIONS.toml` schema notes and in
[ADR-0018](https://github.com/Metrale/metrale-inference/blob/main/docs/adr/0018-relations-become-defaults-on-bytes-speed-and-energy.md).

## Hardware classes and realization

A hardware class (`kernels/<class>/HARDWARE.toml`) names its architecture and the class it
inherits from. It also holds its serving defaults and its tensor-core policy. Its
`common/FUSIONS.toml` and `common/KERNEL_FAMILIES.toml` may inherit the parent's and override or
remove entries. The plan on a class uses the rules and families resolved along that chain,
restricted to kernels the class can run (`crates/circuit/src/hardware/avail.rs`).

The **tensor-core policy** is a constraint on the realization. It lists which ops must run on
tensor cores at which rows and formats, and every exemption is explicit. A plan that violates it
is refused.

Realization is exact for the plan's structure: which kernels, composed how. It is exact for the
plan's numbers only where a class keeps the same reduction order as the reference class. See
[parity tiers](./lkb.md#parity-tiers).

## Reading the reports in this vocabulary

| Command | What it reports | In LKB terms |
|---|---|---|
| `met circuit show` | the plan as stable text, one group per line, with rule, numerics tag, compute unit and numeric pipeline | the lowering, realized on the default class |
| `met circuit display` | the layer strip and diagrams with fused groups framed | the string diagram |
| `met circuit plan --hardware <device>` | fused plan, gap report, roofline estimates, memory fit, tensor-core audit | the realization on the device's class. "Novel" rows are **uncovered ops** (no generator yet). LKB coverage is `100 − novel` |
| `met circuit venn` | one model's ops classified against another's kernels: shared, shared-unmeasured, parameterization, policy variant, novel | two models' lowerings compared in one LKB. Parameterization rows are promotion opportunities |
| `met circuit diff` | logits byte parity, legacy vs circuit, reference rules vs all rules | the realization respects the relations it exercises |
| `met circuit memory` | memory per node, state and cache, and the largest concurrency that fits | — |
| `met circuit precision` | the precision of each step inside each node, required vs declared | the numerics a point declares (precision only; order is not yet declared) |

The [LKB chapter](./lkb.md#not-yet-built) lists what the reports do not compute yet, notably the
LKB residual.

## Recipes: a parameter point plus a serving policy

This is the target design. It is adopted, and existing recipes migrate to it in a later change.
A launch recipe is made of exactly three parts:

1. **The checkpoint's parameter point.** The architecture circuit and its dims, switches and
   precision are derived from the checkpoint by the config map. A recipe names the checkpoint
   and never restates its architecture.
2. **A serving policy.** These are the choices that may change what is computed or how the serve
   behaves, each stated explicitly:
   - the precision tier (`declared`, or a disclosed lower tier);
   - KV-cache format;
   - speculative decoding (draft, depth);
   - batch and KV limits;
   - the lowering settings that choose between byte-different kernels (see below);
   - opted-in levers.
3. **Resources.** Devices, nodes, memory utilization and ports.

**Choices that preserve semantics never appear in a recipe.** Fusions, projection
concatenations, scheduling and stream assignment produce the same bytes either way, so they are
the compiler's to make, chosen by cost.

**Which lowering settings belong in a recipe is derived from the rules, not decided by hand.** A
rule's `when` key selects among rules. A key belongs to the serving policy when the rules it
selects among can produce different bytes: a `reference` rule against another `reference` rule,
or any `differs` rule. A key that only switches `bit_identical` rules on or off belongs to the
compiler, and it becomes a default under the rule above.

The recipe checker requires every recipe-settable key to be explicit, so no serve default is
inherited silently. Under this design, the lowering part of its key list is computed from the
rules in this way rather than maintained by hand. The design note
[recipe shape](https://github.com/Metrale/metrale-inference/blob/main/docs/design/recipe-shape.md)
gives the full mapping and the migration order.

## Vocabulary in the code

| Term | Code |
|---|---|
| circuit | `Circuit` (`ir.rs`), `kernels/circuits/<arch>.toml` |
| op signature | `OpKind`, `LinearRole` (`ir.rs`) |
| tensor type | `Edge`, `Format`, `Scale` (`ir.rs`, `format.rs`) |
| lowering | `fuse`, `FusionPlan`, `Group` (`fuser.rs`) |
| lowering rule | `Rule` (`rules.rs`), `[[rule]]` in `FUSIONS.toml` |
| relation | a `Rule` with `Numerics::BitIdentical` |
| kernel family (generator) | `Family` (`venn/families.rs`), `[[family]]` |
| point | `Point`, `How` (`venn/families.rs`), `[[family.point]]` |
| parameter kind | `ParamKind` (`venn/families.rs`) |
| realization on a class | `ClassInfo`, `class_rules`, `class_families` (`hardware/class.rs`), `node_exec` (`hardware/exec.rs`) |
| uncovered op | `Class::Novel` (`venn/mod.rs`), the plan's placeholder groups |
| LKB residual | not yet modelled: `ClassSources` (`hardware/sources.rs`), `KERNEL.toml [shadow]`, `how = "copy"` |
| cost | `node_cost` (`venn/roofline.rs`), `DeviceRoofline` (`hardware/estimate.rs`) |
