# The LKB in Mathematics

This page states the [Latent Kernel Blueprint](../architecture/lkb.md) in the language of monoidal
categories, but only as far as that language produces rules an agent can apply. Each section
ends with the rule. Section 7 says plainly which parts are vocabulary and which are actionable.
The math is stated conservatively; where the engine only approximates a structure, the note
says so.

## 1. The mapping

Circuits are string diagrams: boxes (ops) wired by typed edges, composed in series and side by
side. String diagrams are the graphical calculus of symmetric monoidal categories (Joyal and
Street 1991; survey: Selinger 2011). A circuit also fans an edge out to several consumers
(a residual stream read twice), so strictly it lives in a symmetric monoidal category with
copying (a "gs-monoidal" or cartesian setting); nothing below depends on that detail.

| Math | Engine | Where |
|---|---|---|
| object | a tensor type: shape × format, where the format carries its scale granularity (layout is not yet part of the type) | edge formats in `kernels/circuits/<arch>.toml`, `crates/circuit/src/format.rs` |
| morphism | an op, and a kernel that implements it | the op vocabulary `crates/circuit/src/ir.rs`; kernels in `KERNEL_FAMILIES.toml` |
| `g ∘ f` (series) | an edge from one node to the next | circuit edges |
| `f ⊗ g` (parallel) | independent branches (q / k / v projections) | circuit blocks |
| string diagram | a circuit | `kernels/circuits/<arch>.toml`, `kernels/circuits/blocks/` |
| the op signature `Σ` | the closed op vocabulary circuits are written in | `OpKind` in `crates/circuit/src/ir.rs` |
| **lowering** `L_H` (from `Σ`-diagrams to LKB diagrams) | the fuser covers a circuit with rule groups; each rule is a clause `pattern ↦ kernel(s)` under its modes, rows and settings | `fuse` in `crates/circuit/src/fuser.rs`; rules in `FUSIONS.toml` |
| rewrite rule (equation between diagrams) | a lowering rule with `numerics = "bit_identical"`, proven by its microtest | `FUSIONS.toml` |
| theory presented by generators and relations | the LKB: families are generators, bit-identical rewrites are relations | `KERNEL_FAMILIES.toml` + the bit-identical rules |
| model of the theory: a functor `F_H : LKB → Impl_H` (functorial semantics, Lawvere 1963) | the **realization** on a hardware class: it turns the presentation into launchable kernels | `HARDWARE.toml` (`inherits`, `[build]`, `[defaults]`), the class's FUSIONS / KERNEL_FAMILIES overlays, kernel availability |
| the plan on class `H` | `F_H ∘ L_H` applied to a circuit | `met circuit plan --hardware` |
| checking that `F_H` respects the relations | byte parity of the fused vs unfused plan | `met circuit diff`, the rules' microtests |
| LKB residual | the part of `Impl_H` outside the image of `F_H`: kernels only this class has | `KERNEL.toml [shadow]`, the class's own sources, `kernels/FORKS.md` |
| convergence | the presentation stabilizes: a new class needs a new functor (data), not new generators (kernels) | the campaign metrics: LKB residual and promotions |

**Lowering rules are not all relations.** Each rule's `numerics` tag states how its clause
relates to lowering the same ops one by one:
- `bit_identical`: a relation, proven by a microtest;
- `reference`: definitional. The rule's kernel *defines* the reference numerics for that op
  chain. It is the engine's default and is not claimed equal to the unfused chain;
- `differs`: a different numerics point, selected only when its lever is opted in.

Most rules today are `reference`, so the presentation's relations are the few `bit_identical`
rules, while the rest of `FUSIONS.toml` is the lowering. Classes that inherit a parent's rules
(`inherits` in their `FUSIONS.toml`) inherit its lowering, minus their `remove` list.

**Where the analogy is loose.** A functor must preserve the relations exactly. With floating
point, `F_H` preserves a bit-identical relation only when the two sides keep the same arithmetic
order on that class (section 2); otherwise it preserves it up to an error bound (section 3). So
"hardware class = functor" is exact for the plan's structure (which kernels, composed how) and
approximate for its numbers. The rules below exist to make that approximation explicit.

**Rule 1.** Every new kernel is either (a) the image of an existing generator under the class's
functor (a point, policy or numerics choice of a family), or (b) a named residual generator.
There is no third kind.

## 2. Rule: bit parity is the same reduction bracketing tree

Exact addition is associative, which is why a circuit can leave the order of a sum unspecified.
Floating-point addition is not associative (Goldberg 1991; Higham 2002, ch. 4): `(a + b) + c`
and `a + (b + c)` round at different points and can differ in the last bits. A reduction of `n`
terms therefore has as many floating-point meanings as it has **bracketing trees** (binary trees
with `n` leaves, the order in which partial sums are formed and rounded).

So the **numerics parameter of a family is, concretely, a choice of bracketing tree per
reduction**, plus the rounding points and contraction:
- the tile split of the reduced dimension (K per thread, per warp, per CTA);
- the order inside a warp (shuffle tree) and across warps / CTAs (shared-memory or atomic
  order);
- the split-K combine (how many splits, combined in what order, in what precision);
- where partial sums are rounded (FP32 accumulator, BF16 output, intermediate stores);
- FMA contraction (`a*b + c` rounded once or twice; `--fmad` and compiler contraction);
- inside a tensor-core MMA, the instruction's own accumulation order and internal precision,
  which the ISA does not always specify and which differs between MMA atoms (e.g.
  `mma.sync` vs `wgmma`).

**Criterion.** For the same inputs, formats, rounding mode and denormal handling, two
implementations of a reduction give bit-identical results **if** they use the same bracketing
tree, the same rounding points and the same contraction (IEEE 754 makes each basic operation
deterministic). The converse does not hold strictly (two trees can agree on some inputs), but
on adversarial inputs different trees differ, so treat "different tree" as "not bit-identical".

**Rule 2a.** Where two classes run the same tree (same source, same tile split, same reduction
order, same split count, same flags, same MMA atom or no MMA, no vendor library call), demand
byte-exactness across classes; a difference is a bug.

**Rule 2b.** Where the trees differ, do not chase bits: apply the Tier 2 tolerance
(a declared tolerance, see the parity tiers in [the LKB](../architecture/lkb.md#parity-tiers)) and the transcript match rate.

**Rule 2c. Make the tree explicit, so TTBP is a design decision.** For each family point, state
the tree in data beside its `pipeline` declaration: reduced dimension, split per level (thread,
warp, CTA, split-K), the combine order at each level, rounding points, contraction, MMA atom.
Then "can this class match the reference box bit for bit?" is a comparison of two records, not
an experiment, and a new class can choose to keep the reference tree wherever the cost allows
(e.g. a split-K count fixed by the model shape instead of derived from the SM count). Choosing
the reference tree is how a class buys Tier 2 byte identity; choosing another tree is a
declared trade of identity for speed.

## 3. Rule: error budgets compose

Give the set of all kernels from `A` to `B` a distance: `d(f, f')` = the largest output
difference over the inputs that matter (a sup norm, in the units of `B`). This is the metric
enrichment of a category (Lawvere 1973); Tier 1 asks `d = 0`, Tier 2 asks `d ≤ ε`.

**The composition bound.** If `g` amplifies input differences by at most `L_g` (a Lipschitz
constant: `|g(x) - g(y)| ≤ L_g |x - y|`), then by the triangle inequality

```
d(g∘f, g'∘f')  ≤  d(g∘f, g∘f') + d(g∘f', g'∘f')  ≤  L_g · d(f, f') + d(g, g')
```

where `d(g, g')` is measured over the inputs `g` actually sees. For parallel branches with a max
norm, `d(f⊗g, f'⊗g') = max(d(f, f'), d(g, g'))`.

**Use.** Derive an end-to-end tolerance from per-op tolerances and amplification factors
instead of guessing one. Worked toy (one layer slice, `y = x + W · norm(x)`, values of order 1):
- `norm` (RMSNorm, BF16 output): own error `ε_n ≈ 4e-3` (half an ulp of BF16 at 1 is
  `2^-9 ≈ 2e-3`; take two ulps as the budget);
- `W·` (GEMV, FP32 accumulate, BF16 output): own error `ε_w ≈ 4e-3` (the output rounding
  dominates a well-conditioned FP32 sum); amplification `L_w` of input differences;
- `+ x` (residual add, BF16 output): `L = 1` in each input, own error `ε_a ≈ 4e-3`.

```
d(W·norm)        ≤  L_w · ε_n + ε_w
d(x + W·norm(x)) ≤  1 · d(W·norm) + ε_a  =  L_w · ε_n + ε_w + ε_a
```

With `L_w = 0.5` this is `0.002 + 0.004 + 0.004 = 0.010`; with `L_w = 4`, `0.024`.

**Caveats, stated honestly.**
- The worst-case `L_w` is the operator norm `‖W‖_∞` (largest row sum of `|W|`), which for a
  5120-wide projection can be in the tens to hundreds: a valid bound, usually far too
  pessimistic to use. The practical `L_w` is estimated **empirically, per format and per
  layer kind**: perturb real activations by a known amount and measure the output change. An
  empirical `L` is an estimate, not a bound; label the derived tolerance as such and check it
  against measured Tier 2 differences.
- BF16 and FP8 errors are relative, so use a scaled distance (relative to the tensor's RMS or
  per-row scale) when magnitudes vary across layers.
- Through many layers the bound compounds (products of `L`); use it per block and per short
  chain, not across a whole model, where the transcript match rate is the honest measure.

**Rule 3.** Every Tier 2 tolerance is derived: per-op `ε` from the format's rounding, `L` from a
recorded perturbation measurement, composed by the bound above, and written down beside the
check. A tolerance with no derivation is a guess and is treated as one.

## 4. Family schema = parametric morphisms (Para)

The Para construction (Cruttwell, Gavranović, Ghani, Wilson, Zanasi 2022; used in Gavranović
et al., "Categorical Deep Learning", ICML 2024) makes "a morphism with parameters" precise: a
morphism `A → B` in `Para(C)` is a pair `(P, f : P ⊗ A → B)`, and a 2-cell between `(P, f)`
and `(Q, g)` is a reparameterization `r : Q → P` with `g = f ∘ (r ⊗ A)`.

Onto `[[family.param]]`:
- **runtime** parameters (strides, counts, eps, pointers) are part of the input `A`: they never
  select a point, which is why the manifest says a runtime difference never makes an
  opportunity;
- **compile-time**, **policy** and **numerics** parameters make up `P`;
- a **point** is a choice `p : I → P`, giving the kernel `f(p, -) : A → B`; `how =
  instantiation | copy | branch` records how `p` is realised;
- a **reparameterization** is a map between parameter spaces that preserves the kernel, e.g.
  a renamed policy, a template argument that subsumes an old per-point copy.

**The promotion test** ([the LKB](../architecture/lkb.md#promotion-how-the-lkb-grows)). Two kernels `k1, k2 : A → B` are one family iff
there is one parametric `f : P ⊗ A → B` and points `p1, p2` with `k_i = f(p_i, -)` **at every
evidence point**, checked byte-for-byte (the stability gate). Equality "up to
reparameterization" is exactly what lets a per-point copy be deleted.

**Rule 4.** A new family parameter is declared with its kind, and the kind decides where it
lives: runtime in the argument list, compile-time as a template instantiation, policy as a
plug-in of the family's template, numerics as a declared bracketing tree (section 2).

## 5. Cost and energy are lax functors

Assign each plan a cost in `(ℝ≥0, +)`. For a plan built by composing `f` then `g`:

```
cost(g ∘ f)  ≤  cost(f) + cost(g)      (subadditivity: the fused composite is never dearer)
```

is the property a fusion must have; a functor that satisfies it only up to this inequality
is lax. **The fusion gain is exactly the laxity**: `cost(f) + cost(g) - cost(g ∘ f)`, which in
the roofline is the materialized round trip the fusion removes (the intermediate's bytes
written and read back, over bandwidth) plus the saved launch. A real fused kernel can violate
the inequality (register pressure, lost occupancy): then it is not a fusion worth keeping, and
the measured cost, not the roofline, decides.

**Energy is a second cost functor, not the same one.** `J = ∫ P dt`: a fusion that saves time
can raise power, so its energy laxity is measured on its own (NVML counter), never inferred from
time, and the two gains are reported separately.

**Not yet computed.** Today's roofline cost (`crates/circuit/src/venn/roofline.rs`) is per node
and does not depend on the plan: it counts every edge a node reads or writes, fused or not, and
no launch overhead. Under it every plan's cost is the sum of its nodes', so the modelled laxity
is zero. Computing it needs a cost per fused group that drops the fused edges' round trips and
adds a per-launch term.

**Rule 5.** A fusion rule is kept only if its time laxity is positive where it applies (measured
once a serve runs), and its energy laxity is recorded beside it; a fusion that wins one and
loses the other is a flagged choice, never a default.

## 6. Proposal: an equality-saturation fuser (an experiment, not built now)

**Today.** The fuser is greedy and deterministic: rules are tried by priority, each scanning
nodes in execution order, and a node joins at most one group (`crates/circuit/src/fuser.rs`).
A greedy cover can miss a cheaper combination when an early high-priority rule consumes a
node a better combination needed.

**The idea.** Equality saturation (Tate et al. 2009; `egg`, Willsey et al., POPL 2021) keeps
every equivalent form of a term at once in an e-graph, applies rewrite rules until no rule adds
anything (or a bound is hit), then extracts the cheapest equivalent term under a cost function.
Loaded with the circuit and **only `bit_identical` rewrites**, every extracted plan is equal to
the reference plan byte for byte: correct by construction, with the cost model choosing among
correct plans.

**What it can rewrite.** Only `bit_identical` rules are equations. A `reference` rule defines
the reference numerics, so the saturator keeps the reference lowering fixed and explores only the
forms that are equal to it under the relations. The space it can search therefore grows with the
number of `bit_identical` rules, which is a reason to prove more rules bit-identical before
building it.

**Scope of the experiment.**
- *Inputs*: a loaded circuit, the class's rules filtered to `bit_identical` (and `reference`),
  the available kernels, the policy and the run (mode, rows): exactly what `fuse` takes today.
- *Rule translation*: each rule becomes a conditional rewrite (pattern → fused kernel group),
  with its side conditions (rows, modes, settings, kernel availability, input formats) as
  guards.
- *Extraction*: under the roofline cost (section 5). Optimal extraction with shared
  sub-terms is NP-hard in general; start with egg's tree-cost extractor and compare against an
  ILP extractor on small plans.
- *Success metric*: across every golden plan (`kernels/circuits/plans/`, the hardware matrix),
  the extracted plan's estimated step cost is ≤ the greedy plan's; count the strictly better
  plans and their gain; where costs tie, the extracted plan's digest equals the greedy one
  (no churn). Every changed plan must then pass `met circuit diff` byte parity and an nsys
  check that the predicted gain is real.
- *Risks*: e-graph blow-up (bound iterations and node count; rules that only fuse never
  create unbounded growth, but associativity-like rules would); cost-model fidelity (the
  roofline mis-ranks small kernels and launch-bound steps: use measured costs where the
  evidence envelope has them); rules whose conditions depend on the whole plan.
- *Where it plugs in*: a second pure planner beside `fuse` in `crates/circuit` (SBIO: no I/O,
  same inputs, same `FusionPlan` output and digest), selected explicitly, compared plan by
  plan. The `egg` crate is MIT-licensed; its addition goes through `cargo deny`.

**Rule 6.** Until the experiment shows a gain on the golden plans with byte parity, the greedy
fuser stays the planner; the experiment's result, positive or not, is recorded.

## 7. What is vocabulary and what is actionable

- **Actionable**: section 2 (the parity criterion: record the bracketing tree, demand bytes
  where trees match, tolerances where they do not), section 3 (derive tolerances from
  per-op errors and measured amplification), section 6 (a bounded, measurable fuser
  experiment).
- **Precision, not new capability**: section 4 states what a family and a point are and what
  promotion tests; section 5 states what fusion optimizes and why time and energy are separate.
- **Vocabulary**: section 1. It names things the engine already has. Its value is Rule 1 and
  the definition of convergence, not the category theory itself; nobody needs to write a
  functor.

## Sources

- F. W. Lawvere, *Functorial Semantics of Algebraic Theories*, PhD thesis, Columbia, 1963.
- F. W. Lawvere, "Metric spaces, generalized logic, and closed categories", *Rend. Sem. Mat.
  Fis. Milano* 43, 1973.
- A. Joyal, R. Street, "The geometry of tensor calculus I", *Advances in Mathematics* 88, 1991.
- P. Selinger, "A survey of graphical languages for monoidal categories", in *New Structures
  for Physics*, Springer, 2011.
- G. Cruttwell, B. Gavranović, N. Ghani, P. Wilson, F. Zanasi, "Categorical Foundations of
  Gradient-Based Learning", ESOP 2022.
- B. Gavranović, P. Lessard, A. Dudzik, T. von Glehn, J. G. M. Araújo, P. Veličković,
  "Position: Categorical Deep Learning is an Algebraic Theory of All Architectures", ICML 2024.
- D. Goldberg, "What every computer scientist should know about floating-point arithmetic",
  *ACM Computing Surveys* 23(1), 1991.
- N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM, 2002.
- R. Tate, M. Stepp, Z. Tatlock, S. Lerner, "Equality Saturation: a New Approach to
  Optimization", POPL 2009.
- M. Willsey, C. Nandi, Y. R. Wang, O. Flatt, Z. Tatlock, P. Panchekha, "egg: Fast and
  Extensible Equality Saturation", POPL 2021.
- J. Ragan-Kelley et al., "Halide: a language and compiler for optimizing parallelism,
  locality, and recomputation in image processing pipelines", PLDI 2013 (algorithm vs
  schedule).
