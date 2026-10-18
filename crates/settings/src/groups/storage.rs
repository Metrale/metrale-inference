// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The `storage` lever group: the `storage.*` levers a `met` process reads at run time.
//!
//! Owner: settings.
//! Invariants: rows in env-name order.

lever_group! {
    /// 2026-10-17: The `storage` group, resolved by `crate::resolve()`.
    pub struct StorageSettings { group: "storage", class: Runtime }
    ds41_direct_reads: Switch = "on", None, "METRALE_DS41_DIRECT_READS", "crates/model-weights/src/weights/gguf/expert_reads.rs", "`METRALE_DS41_DIRECT_READS=0` keeps every read on the page cache. `0` turns it off.";
    ds41_prefetch_k: Text = "unset", None, "METRALE_DS41_PREFETCH_K", "crates/model-arch/src/moe_v41.rs", "`METRALE_DS41_PREFETCH_K` (default 0 = off), read once: how many of the next layer's predicted experts to start reading while this layer computes (needs `METRALE_DS41_READER_POOL=1`).";
    ds41_reader_threads: Int = "16", None, "METRALE_DS41_READER_THREADS", "crates/model-arch/src/weight_loader/deepseek_v41/load_layers.rs", "16 readers: one 12.22 MiB expert reads in ~2 ms whatever the split, but a prefill's gather of hundreds runs at the disk's 10-11 GB/s only past eight threads (09-19 NVMe probe)";
    expert_rdma_dev: Text = "roceP2p1s0f1", None, "METRALE_EXPERT_RDMA_DEV", "crates/storage/src/links.rs", "The RDMA device of rail 0 for the KV, snapshot and expert peer tiers. Set to anything, empty included, it is used verbatim.";
    expert_rdma_gid: Int = "3", None, "METRALE_EXPERT_RDMA_GID", "crates/storage/src/links.rs", "The GID index of rail 0 for the KV, snapshot and expert peer tiers.";
    fast_load: Switch = "on", "--no-fast-load", "METRALE_FAST_LOAD", "crates/server/src/main_modules/serve_phases/load_levers.rs", "The O_DIRECT pipelined weight loader. `0` turns it off.";
    fast_load_prefetch_shards: Switch = "off", "--fast-load-prefetch-shards", "METRALE_FAST_LOAD_PREFETCH_SHARDS", "crates/server/src/main_modules/serve_phases/load_levers.rs", "Shard prefetch and readahead in the fast weight loader. `1` turns it on.";
    glm_layer_stream: Switch = "off", None, "METRALE_GLM_LAYER_STREAM", "crates/model-arch/src/weight_loader/glm5_next_load/layer_stream.rs", "`1`: GLM-5.3 load keeps every text layer's non-expert tensors (attention, dense MLP, shared expert, norms, mHC) out of the device weight store and reads each layer's from its shard when that layer is built, instead of uploading all of them and freeing them after the last layer. Lowers the load peak by their total (the store copies the binders never bind); the bytes the binders see are the same (test-gated), so no numerics change. Unset or `0` keeps the store copies; another value is refused at load.";
    lora_peer: Text = "unset", None, "METRALE_LORA_PEER", "crates/model-layers/src/lora/env.rs", "The host:port of the weight peer that stages adapters, when set and non-empty. Setting it also sets the model's `lora_rotatable`, as `METRALE_LORA_ROTATE` does.";
    lora_rdma_dev: Text = "unset", None, "METRALE_LORA_RDMA_DEV", "crates/model-weights/src/weight_lora_rdma.rs", "Read in `crates/model-weights/src/weight_lora_rdma.rs`.";
    lora_rdma_gid: Text = "unset", None, "METRALE_LORA_RDMA_GID", "crates/model-weights/src/weight_lora_rdma.rs", "LoRA env: DEV chains LORA→WEIGHT→EXPERT (an exported-but-EMPTY var counts as set — `first_set`); GID reads ONLY METRALE_LORA_RDMA_GID (no chain, deliberately).";
    no_gemv_exact_m_tiers: Presence = "off", None, "METRALE_NO_GEMV_EXACT_M_TIERS", "crates/model-layers/src/layers/w4a16_gemv_tiers.rs", "`METRALE_NO_GEMV_EXACT_M_TIERS` (presence-checked: any value, including `0`, disables) hides widths 5/6/7 from the decision, leaving widths 4 and 8. Any value turns it on, `0` included.";
    redzone_range_file: Path = "unset", None, "METRALE_REDZONE_RANGE_FILE", "crates/model-engine/src/model/trait_impl/decode_a_redzone.rs", "`METRALE_REDZONE_RANGE_FILE=<path>` — see the use site.";
    store_dir: Path = "unset", None, "METRALE_STORE_DIR", "crates/server/src/response_store/store_impl.rs", "Build from env. - `METRALE_STORE_MAX_ENTRIES` (default 10 000) - `METRALE_STORE_TTL_SECONDS` (default 86 400) - `METRALE_STORE_DIR` — when set, enable filesystem persistence and replay any...";
    warm_target_dir: Path = "unset", None, "METRALE_WARM_TARGET_DIR", "crates/bench/src/benchmarks/agentic/warm.rs", "Read in `crates/bench/src/benchmarks/agentic/warm.rs`.";
    weight_rdma_dev: Text = "unset", None, "METRALE_WEIGHT_RDMA_DEV", "crates/model-weights/src/weight_lora_rdma.rs", "Read in `crates/model-weights/src/weight_lora_rdma.rs`.";
    weight_rdma_gid: Int = "unset", None, "METRALE_WEIGHT_RDMA_GID", "crates/model-weights/src/weight_tier_rdma.rs", "Read in `crates/model-weights/src/weight_tier_rdma.rs`.";
}
