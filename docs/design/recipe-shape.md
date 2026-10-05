# Recipe shape: a parameter point, a serving policy, and resources

**Status:** target design, adopted ([ADR-0019](../adr/0019-recipes-are-a-parameter-point-a-serving-policy-and-resources.md)).
Existing recipes migrate in a later, separate stack (§5).
**Background:** [The Circuit Compiler](../../book/src/architecture/circuit-compiler.md),
[The Latent Architecture Blueprint](../../book/src/architecture/lab.md),
[ADR-0018](../adr/0018-relations-become-defaults-on-bytes-speed-and-energy.md).

## 1. The shape

A recipe (`recipes/<family>/<name>.yaml`) states exactly three things.

| Part | What it holds | Who decides it |
|---|---|---|
| **Parameter point** | the checkpoint id. The architecture, its dims and switches, and the declared precision are *derived* from the checkpoint by the config map | the checkpoint |
| **Serving policy** | every choice that may change the bytes computed or the serve's behavior: precision tier, KV-cache format, speculative decoding (draft, depth), batch and KV limits, the byte-changing lowering settings (§2), opted-in levers | the recipe, explicitly |
| **Resources** | container, nodes, devices, tensor/expert parallel degree, memory utilization, bind address, port | the recipe, explicitly |

Choices that preserve semantics are absent from recipes. Fusions, projection concatenations,
scheduling, stream assignment, and any setting that only switches `bit_identical` rules produce
the same bytes either way, so the compiler makes them by cost. Under ADR-0018 such a setting
becomes a default once it shows no speed or energy regression.

## 2. Which lowering settings are policy: derived, not listed

The lowering rules select among themselves through `when` keys (`kernels/<class>/common/FUSIONS.toml`).
For each key, look at the rules it gates:

- **serving policy**: some gated rule is `reference` or `differs`. Switching the key can then
  change bytes, because it changes the reference numerics or the numerics point;
- **compiler**: every gated rule is `bit_identical`. The key cannot change bytes.

A key is policy unless it is proven otherwise. Two `reference` rules that happen to agree are
still treated as different until one of them is proven `bit_identical`, and that proof moves
the key to the compiler.

On the gb10 rules at the time of writing, this gives:

| Key | Gated rules | Class |
|---|---|---|
| `row_tiers` | reference, differs | policy |
| `kv_cache_dtype` | reference | policy |
| `lm_head_dtype` | reference | policy |
| `ssm_h_dtype` | reference, differs | policy |
| `ssm_batched_recurrent` | reference, differs | policy |
| `ssm_ba_gates_hopper` | reference | policy |
| `gemv_sw` | reference, differs | policy |
| `w4a16_tc` | reference | policy |
| `decode_split_silu` | reference | policy |
| `rms_norm_act_quant` | bit_identical only | **compiler** |

`activation_quantization` is not a rule's `when` key. The precision resolver reads it, and it
selects formats, so it is policy.

The classification is a pure function of the rule set. It belongs in `crates/circuit` beside the
rules and is recomputed on every change. A rule whose numerics tag changes therefore moves its key
automatically, with no list to edit.

## 3. Consistency with the fully-explicit recipe checker

The recipe checker (`crates/server/src/recipe/explicit.rs`) requires every recipe-settable
`met serve` key to be explicit. It derives the key list from the serve CLI definition, so a new
flag is required automatically. This design extends that rule and does not replace it.

- The serve-flag part stays derived from the CLI.
- The lowering part is derived from the rules (§2):
  - a recipe for a checkpoint that has a circuit must state every **policy** key for the
    classes it runs on;
  - it must not state a **compiler** key.
- Both parts are refused on omission, so nothing is left to an engine default.

## 4. One source for the policy

Today a circuit-backed recipe's serving policy is written in two places:
- the recipe's `defaults:` and `env:`;
- the instance's `[instance.policy] settings` in `kernels/circuits/INSTANCES.toml`, which cites
  where the recipe sets each value.

The target is one source. The recipe holds the policy, and an instance names its recipe id and
keeps only what the golden-plan matrix needs (rows, modes, `golden`). A check proves that each
instance's resolved policy equals its recipe's.

Gate pins of serve levers (`[benchmarks.serve_env]` in `BENCH.toml`) become recipe policy too,
so a gate runs a recipe and does not restate one.

## 5. Migration order (a separate, later stack)

1. **Classifier.** A pure function `policy_keys(rules) -> { policy, compiler }` in
   `crates/circuit`, with tests on the toy rule set and on the gb10 rules (the table in §2 as
   the expected output). No behaviour change.
2. **Checker extension.** The recipe checker reads the classifier. During migration, version-2
   recipes are checked for the policy keys through their existing `defaults:` and `env:`
   spellings.
3. **Schema version 3.** Sections `checkpoint`, `policy`, `resources` and `metadata`. A mapper
   reads version 2 into version 3, so both parse during migration.
4. **Circuit-backed recipes first:** the five recipes listed in `INSTANCES.toml`. Instances then
   point at recipe ids (§4).
5. **Plan-only architectures:** recipes whose checkpoint has a circuit the executor does not run
   yet (MoE, Nemotron-H, dense GQA).
6. **Recipes with no circuit.** These stay on version 2. Their lowering policy cannot be derived
   until their architecture has a circuit, and they migrate when it does.
7. **Gate pins.** `[benchmarks.serve_env]` entries move into the recipes their gates name.
8. **Retire version 2** once step 6 is empty.

Each step leaves every recipe's resolved `met serve` command line byte-identical. A check renders
both versions to argv and compares them.
