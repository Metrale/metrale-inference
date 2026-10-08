# ADR-0019: A recipe is a parameter point, a serving policy, and resources

**Status:** Accepted
**Date:** 2026-10-05

## Context

A launch recipe (`recipes/<family>/<name>.yaml`) mixes four kinds of setting:
- facts about the model the checkpoint already declares;
- choices that change what is computed (precision tier, KV-cache format);
- choices that only change how fast the same bytes are produced (some kernel levers);
- the resources to run on.

The circuit compiler now derives a model's architecture and declared precision from its
checkpoint, and it knows which lowering choices are byte-identical (ADR-0018). Recipes still
restate some of this, and some byte-identical choices are pinned by hand in more than one place.

## Decision

A recipe states exactly:

1. **The checkpoint's parameter point.** The checkpoint id. Architecture, dims, switches and
   declared precision are derived from it, never restated.
2. **A serving policy.** Every choice that may change the bytes computed or the serve's
   behavior: precision tier, KV-cache format, speculative decoding, batch and KV limits, the
   byte-changing lowering settings, and opted-in levers. Each is stated explicitly.
3. **Resources.** Devices, nodes, parallel degree, memory utilization, bind address and port.

**Semantics-preserving choices never appear in a recipe.** Fusions, concatenations, scheduling,
and settings that only switch `bit_identical` rules are the compiler's to make.

Which lowering settings are policy is computed from the rules, not listed. A `when` key is
policy if any rule it gates is `reference` or `differs`, and it belongs to the compiler if every
rule it gates is `bit_identical`. The fully-explicit recipe checker takes its lowering keys from
that computation and its serve-flag keys from the CLI definition, as it does today.

The design and the migration order are in [docs/design/recipe-shape.md](../design/recipe-shape.md).

## Consequences

- A recipe can no longer pin a byte-identical choice. Turning one on or off is a compiler
  default, decided under ADR-0018.
- Proving a `reference` rule `bit_identical` removes its setting from every recipe automatically.
- The policy of a circuit-backed recipe has one source, and the circuit instances refer to it.
- Recipes for models without a circuit keep the current schema until they have one, because their
  lowering policy cannot be derived.
- Existing recipes migrate as a separate stack, and every step must leave each recipe's resolved
  `met serve` command line unchanged.
