# The Latent Architecture Blueprint

The [Latent Kernel Blueprint](./lkb.md) parameterizes kernels until a hardware-independent basis
remains. The **Latent Architecture Blueprint (LAB)** does the same for models.

Architectures are described as parameterized blocks. A new model is then a **parameter point**
of existing block families, plus a small, named **architecture residual** for what no family
expresses yet. Supporting a new checkpoint becomes a config-map entry instead of a new circuit.

## Where it lives

The LAB is not a separate artifact any more than the LKB is. It is the architecture circuits and
their config maps:

| LAB concept | Where it lives |
|---|---|
| a block family (one architectural component) | a `[block.<name>]` in `kernels/circuits/blocks/` or in a circuit file |
| its parameters | the circuit's `dims` (shape), switch dims with `when = "<switch>"` (structure), `params_when` (variants), node `binding`s (checkpoint tensor names) |
| a model's layout | `[layout]`: an `interval` with a period, or a `list` of layer kinds read from the config |
| a model's parameter point | what the config map (`kernels/circuits/<arch>.config.toml`) derives from `config.json`, per `[variant.<model_type>]` where needed |
| the architecture residual | the refusals: a `config.json` key or `model_type` that no circuit maps is refused with a reason (`crates/circuit/tests/checkpoint_refusals.rs`) |

## Parameter kinds

| Kind | What it does | Examples |
|---|---|---|
| **shape** | sizes edges; never changes which ops exist | hidden size, heads, KV heads, head_dim, experts, top-k, state size |
| **switch** | adds or elides nodes; a dim whose value is 0 or 1 | QK-norm, QKV bias, tied embeddings, an MTP head, a latent MoE projection |
| **variant** | selects one of several sub-blocks or op parameters | RoPE scheme, norm form (`x·w` or `x·(1+w)`), router scoring (softmax or sigmoid with bias), expert activation |
| **binding** | names the checkpoint tensors a node reads | `{L}.self_attn.q_proj`, where `{L}` is the layer's module path |

A parameter's kind on the architecture side does not fix its kind on the kernel side. The
lowering decides that, through each kernel family's parameter extractors
(`[[family.param]] from = "dim:…" | "op" | "setting:…"`). For example:
- head_dim becomes a compile-time point of the attention family;
- top-k becomes a runtime argument of the routing kernel;
- router scoring becomes a policy of the routing family.

This is where the two blueprints meet. One structural condition makes it work: a lowering rule
must be able to distinguish the variants it implements. A rule written for one router scoring
function must not match another.

## The axes along which supported models differ

| Axis | Values | How it is expressed today |
|---|---|---|
| attention | GQA, MQA, MLA | GQA and MQA through the KV-head dim; MLA is architecture residual |
| head_dim | per model | shape dim |
| attention window | full, sliding, interleaved | full only; sliding is refused by the config maps |
| attention output gate | none, sigmoid gate interleaved with Q | fixed per circuit |
| QK-norm, QKV bias | on / off | switches |
| positional scheme | none, RoPE (plain, partial, multi-section) | config parameters read by the executor |
| sequence mixer | full attention, gated delta net, Mamba-2 | separate blocks, chosen per layer kind |
| layer interleave | a period, an explicit list, a pattern string | `[layout]` plus the config map's layer source |
| FFN | dense SwiGLU, ReLU², MoE | blocks; MoE top-k and expert count are shape dims |
| MoE routing | softmax or sigmoid-with-bias, renormalization, shared expert (gated or not), latent projections | node parameters and switches |
| norms | RMSNorm with `w` or `1 + w`; placement | node parameter; placement fixed by the block |
| embeddings | tied or untied | a switch in one circuit, a parameter in others |
| draft heads | none, one MTP module, multi-module MTP | the circuit's `draft` blocks; `mtp ∈ {0, 1}` |
| other towers | vision, audio | outside the text circuit (ignored or refused) |

## Promotion

The [LKB promotion rule](./lkb.md#promotion-how-the-lkb-grows) applies unchanged. A structure
becomes a parameter when a second model needs it. One user is a specialization; two users make a
parameter.

A block family's parameterized form is accepted only if every existing instance produces a
byte-identical golden plan (`kernels/circuits/plans/`). Two circuit files carrying the same block
are the architecture's equivalent of a per-point kernel copy: they are a promotion candidate.

## Convergence metrics

Each campaign records these beside the LKB metrics:

| Metric | Definition | Direction |
|---|---|---|
| **architecture coverage** | the share of the checkpoints the engine tracks (the fixture set under `crates/circuit/tests/fixtures/checkpoints/`) that instantiate a circuit; and the share of circuits the executor runs | up |
| **models expressible as parameter points** | checkpoints that instantiate with no change to any circuit or block, only to a config map | up |
| **LAB residual** | refused `model_type`s and keys, each with the generator it is missing (an MLA block, a sliding-window attention variant, …) | down |
| **marginal lines per new model** | the circuit plus config-map lines a new checkpoint needed | down |
| **duplicated circuit lines** | lines of a block defined in more than one file | down |

Every campaign exit report states:
```
Promoted into the LAB: <block family / parameter / variant>, ... (or "none")
LAB residual delta: <refused model_types before> -> <after>
```

## Restructuring at the circuit level

Circuit-level changes to an architecture fall into two classes, and the compiler treats them
differently.

**Semantics-preserving.** These are changes that produce the same bytes:
- concatenating projections that read the same input (QKV, gate and up);
- fusing a layer's residual add into the next layer's norm;
- running independent branches on separate streams;
- permuting a cache or state layout;
- where a draft head attaches, when verification is exact.

They are relations of the LKB or plan-level choices. The compiler makes them by cost, and they
never appear in a [recipe](./circuit-compiler.md#recipes-a-parameter-point-plus-a-serving-policy).

**Semantics-changing.** These change the parameter point:
- a lower precision tier for some layers;
- a quantized KV cache;
- fewer routed experts, pruned or merged experts, skipped layers;
- a different numerics point, such as the row tier (`by_rows` against the batch-invariant
  `canonical`): both are correct, but their bytes differ at some row counts.

The circuit expresses each of them through existing mechanisms: dims, switches, precision maps.
Each sits behind a named serving-policy setting and is disclosed on records with its speed and
energy deltas. One that lowers precision below what the checkpoint declares, or removes
computation, must also pass the accuracy bar.
