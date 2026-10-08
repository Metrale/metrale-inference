# ADR-0018: A proven bit-identical relation may become the default on bytes, speed and energy

**Status:** Accepted
**Date:** 2026-10-05

## Context

The circuit compiler lowers an architecture circuit to kernels through the rules in
`kernels/<class>/common/FUSIONS.toml`. Each rule carries a `numerics` tag:

- `reference`: the engine's default for that op chain, which defines the reference numerics;
- `bit_identical`: byte-identical to running the chain unfused, proven by a named microtest;
- `differs`: a different numerics point, selected only by an opt-in lever.

A `bit_identical` rule applies only where a serving policy turns its `when` setting on. Several
proven `bit_identical` rules were off in every policy, and nothing stated what evidence would
turn one on by default. Two questions came up: whether such a change owes the accuracy gates
(hours of GPU time each), and which evidence decides it.

## Decision

A `bit_identical` rule may become the default (its `when` setting on in the serving policies
that can use it) when both of the following hold.

1. **Byte identity.** Its microtest passes, and `met circuit diff` shows identical logits with
   the rule on and off for the models that use it.
2. **No speed regression and no energy regression.**
   - Measured with a same-box A/B: the same binary, the same box, the rule on against off,
     across the concurrency ladder the gates use, with a control leg to bound run-to-run noise.
   - Speed and energy are measured and reported separately. Energy is `J = ∫ P dt` from the
     device's energy counter; it is never inferred from time.

**No accuracy gate is required.** Identical bytes give identical outputs, so they give identical
accuracy. The byte-identity proof stands in for the accuracy suite.

A rule that improves one of speed and energy and regresses the other does not become a default.
It stays an opt-in choice, and its records state both deltas.

This applies only to `bit_identical` rules. Changing a `reference` rule's kernel changes the
reference numerics, and selecting a `differs` rule changes the numerics point. Both keep their
existing gates.

## Consequences

- Turning on a proven fusion costs one A/B campaign, not a full accuracy campaign.
- Proving more `reference` rules bit-identical (each with a microtest) is worth doing for its own
  sake, because it moves them under this cheaper rule.
- Rule 1 has to be precise. A microtest that covers only some modes or row counts proves identity
  only there, so the rule's `modes` and `rows` must not exceed what the microtest covers.
- A setting that only switches `bit_identical` rules is the compiler's choice, not a recipe's
  (see [ADR-0019](0019-recipes-are-a-parameter-point-a-serving-policy-and-resources.md)).
