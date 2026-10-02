# The circuit memory model: design and the reserve wiring

Status: 2026-10-02, with `met circuit memory` (crates/circuit/src/memory/, crates/server/src/cli/circuit_memory*.rs).

Owner directive (2026-10-02): "being able to see and define explicitly (for metadata and
tracking), for each node, the memory used. That way, we can define the base amount of memory
expected based on initial ISL's. We need to take into account the KV, SSM, and now with the
additions from [the new speculative-decoding techniques], other caches (e.g., caching of
acceptance tokens). We want to be able to use this as a utility for the circuit."

## 1. What the model is

One pure evaluator, `metrale_circuit::memory::evaluate`, over the instantiated circuit and
explicit counts. Nothing in the crate reads a file, the environment or an engine global.

| Term | Source of truth | Per node |
|---|---|---|
| Stored weights | the checkpoint's declared formats (`Format::weight_bytes`, scales included), `out x k` from the node's edges, routed experts x `experts` | the node |
| Derived copies | `kernels/circuits/COPIES.toml`: one rule per loader allocation (requantized serving copies, transposed twins, MMQ repacks, widened scales, leaked intermediates), matched by op, declared format, served format and serve settings | the node |
| Outside the circuit | the checkpoint's safetensors headers (norms, conv, gates, a vision tower), measured by the CLI | totals only |
| Activations | the buffer planner's arena for each fused run given (decode C, verify C x K, prefill chunk); the legacy arena (`BufferSizes::from_config` + the GDN two-phase buffers) is what the budget charges while the legacy forward is the default | the producing node, at the peak run |
| Workspace | `[[family.workspace]]` in KERNEL_FAMILIES.toml, sized at the widest run; `arena = true` marks scratch the legacy arena already holds | every node of the family |
| States | `[[block.<b>.state]]` recurrent and paged-KV declarations (`StatePlan`, M5) | the node that updates or writes it |
| Caches | `[[block.<b>.state]]` cache kinds (`StateKind::is_cache`): prefix and ring snapshots, the carry stash and tables, the WY tables, the accepted-hidden stash, the drafter capture, the prompt-lookup index (host), a token tree's mask and drafted ids | state table |
| Driver | `kernels/<class>/HARDWARE.toml [memory]` (#72's calibration: fixed + per mille of the util budget) plus measured chunk slack | totals only |

A cache whose count is absent sizes nothing; an unknown kind is refused at load with the list of
kinds. The counts are the engine's: the CLI computes them with the engine's own functions
(`ssm_reserve::pool_counts_with`, `mtp_state_slots_with`, `marconi_snapshot_slots`,
`decode_rollback_ring_slots`, `resolve_num_drafts`, `resolve_mtp_max_seqs`,
`resolve_kv_dtype_str`, `resolve_prefill_budget`, `BufferSizes::from_config`) from the
`ServeArgs` the serve's own parser builds out of a recipe and flags.

## 2. Accuracy (the ledger validation)

`crates/server/src/cli/circuit_memory_ledger_tests.rs` evaluates the model at five real boots'
settings and pool sizes and compares every term the boot's allocation ledger itemizes:

| Term | Tolerance | Worst error over the five boots |
|---|---:|---:|
| weights (stored + derived + outside) | 1% | -0.22% (Nemotron-3-Nano: the router is declared NVFP4 by the global ModelOpt algo but stored F32; the 52.8 MiB dequant scratch) |
| KV (target + MTP pools) | 0.1% | +0.01% |
| SSM pool | 0.1% | 0.00% |
| snapshots (Marconi, decode ring) | 0.5% | 0.00% |
| legacy arena | 1% | +0.02% |
| GDN two-phase buffers | 0.1% | -0.02% |

and #72's named reserve terms (SSM pool, Marconi, decode ring, carry stash, driver) equal the
model's to the logged MB.

## 3. Wiring #72's reserve to the model (proposal)

#72 (`pr/default-tier-kv`) plans the pre-load reserve from named terms
(`serve_phases/preflight/reserve_plan.rs` `ReservePlan::terms`). Today each term is computed by
its own function; the SSM pool already reads the circuit (`PoolPlan` over `StatePlan`, M5). The
minimal change that makes every term derive from the circuit model, in the order of risk:

1. **Marconi and the decode ring.** `marconi_bytes` and `ring_slots * slots * per_seq_blob`
   become `memory::caches::cache_terms` over the model's circuit states with
   `CacheInputs { prefix_snapshot_slots, ring }`. This also counts the last-hidden row the
   allocation adds (`ssm_snapshot_init.rs:88`, which the reserve leaves out today).
2. **The carry stash.** `runtime_headroom::carry_stash_bytes` becomes the `carry_stash` and
   `carry_table` caches with `CacheInputs { carry: (verify_slots, VERIFY_WY_TABLE_SEQS) }`;
   `GdnCarrySizes` stays the allocator's arithmetic and
   `cli/circuit_memory_parity_tests.rs` pins the two equal.
3. **The driver terms.** `DRIVER_FIXED_BYTES` and `DRIVER_BUDGET_PER_MILLE` move to
   `kernels/gb10/HARDWARE.toml [memory]` (already declared there), read through
   `memory::DriverTerms` at build time like the `[defaults]` levers
   (`metrale_kernels::TARGET_DEFAULTS`), so another class states its own calibration.
4. **The weights.** The post-load audit's `predicted_derived_bytes` (native-FP8 dense route
   only) is replaced by `memory::weights::node_weights` over COPIES.toml, which covers every
   default load path; `METRALE_DENSE_FP8`'s copies become rules with `when` keys.

Each step is a byte-identical refactor except (1)'s hidden row, whose size is printed. Steps
1-3 touch only `reserve_plan.rs` and `runtime_headroom.rs`; step 4 touches the post-load audit.
Until then the validation test above is the agreement proof.

## 4. Findings while validating

- **4.6 GiB of leaked load-time copies on the dense 27B** (both tiers): the BF16 intermediates of
  the FP8 attention requant (3.1 GiB, `attn_arms.rs:143-152`), the GDN `out_proj` FP8 predequant
  that the cast overwrites (1.4 GiB, `gdn_dequant.rs:418-424`), and the first `in_proj_ba`
  interleave. COPIES.toml marks them `leaked`; freeing them is a loader fix, not a model change.
- **Derived copies are 58% of the dense 27B's resident weights** (30.5 GiB on 21.8 GiB stored).
- **The declared circuit gave nvidia/Qwen3.6-35B-A3B-NVFP4's routed experts BF16**: ModelOpt
  names the fused `…mlp.experts` module, not its projections. `declared_precision.rs` now lets a
  routed expert's projection inherit it.
- **Nemotron-3-Nano's router is declared NVFP4 by the global ModelOpt algo** but stored F32
  (128 x 2688 per layer, 30 MiB over 23 layers); the declared plan has no router exclusion.

## 5. Follow-ups

- `hardware::estimate::footprint` (the `met circuit plan` memory fit) sizes weights with its own
  `out x k x experts` sum; it should read `memory::weights::node_weights` (which also stores the
  draft head's reused `lm_head` once). That changes the checked-in matrix reports, so it is left
  to its own change.
- FUSIONS.toml has no FP8-KV paged-attention rule on gb10, so an FP8-KV serve's attention plans as
  a placeholder group and its split-K workspace is not attributed.
