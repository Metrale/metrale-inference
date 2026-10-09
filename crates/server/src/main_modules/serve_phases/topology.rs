// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Serve topology: TP and EP sizes and ranks, the TP-local head
//! counts, and the NCCL communicator.
//!
//! Owner: server startup (`met serve`).
//! Invariants:
//! - A `Topology` from `resolve_topology` has `world_size == tp_size * ep_size`,
//!   or `world_size == tp_size == ep_size > 1`.
//! - Above one TP rank, `config`'s head counts are this rank's and its
//!   `pre_shard_heads()` are the checkpoint's.

#[cfg(feature = "nccl")]
use anyhow::Context;
use anyhow::Result;

use metrale_config::ModelConfig;

use crate::cli;

pub(crate) struct Topology {
    pub(crate) world_size: usize,
    pub(crate) tp_size: usize,
    pub(crate) ep_size: usize,
    pub(crate) tp_rank: usize,
    pub(crate) ep_rank: usize,
}

pub(crate) fn resolve_topology(
    args: &cli::ServeArgs,
    config: &mut ModelConfig,
) -> Result<Topology> {
    let (tp_size, ep_size) = if args.tp_size == 1 && args.ep_size == 1 && args.world_size > 1 {
        (1usize, args.world_size)
    } else {
        (args.tp_size.max(1), args.ep_size.max(1))
    };
    let derived_world = if tp_size == ep_size {
        tp_size
    } else {
        tp_size * ep_size
    };
    let world_size = if args.world_size <= 1 && (tp_size > 1 || ep_size > 1) {
        tracing::info!(
            "Auto-derived world_size={} from --tp-size {} --ep-size {} (rule: \
             tp==ep → overlapping = tp; else orthogonal = tp×ep). Pass \
             --world-size to override.",
            derived_world,
            tp_size,
            ep_size,
        );
        derived_world
    } else {
        args.world_size
    };
    let (tp_rank, ep_rank) = if tp_size == ep_size && tp_size == world_size && tp_size > 1 {
        (args.rank, args.rank)
    } else if world_size == tp_size * ep_size {
        (args.rank % tp_size, args.rank / tp_size)
    } else {
        anyhow::bail!(
            "Invalid parallelism topology: world_size={} but tp_size={} × ep_size={} = {}. \
             Either use orthogonal mesh (world = tp × ep) or overlapping groups \
             (world = tp = ep, used for 2-GPU TP+EP composition).",
            world_size,
            tp_size,
            ep_size,
            tp_size * ep_size,
        );
    };
    config.tp_rank = tp_rank;
    config.tp_world_size = tp_size;
    // 2026-09-26: Loaders size per-sequence state from `serve_max_seq_len`,
    // e.g. the GLM-5.3 DSA indexer cache (`glm5next_dsa/mod.rs`).
    config.serve_max_seq_len = args.max_seq_len;
    config.ep_rank = ep_rank;
    config.ep_world_size = ep_size;
    // 2026-10-09: `--moe-expert-layout`; `tp` is refused off its topology and for a loader
    // that does not slice experts over TP.
    config.moe_expert_layout = args.moe_expert_layout.0;
    config.moe_expert_layout.check_topology(tp_size, ep_size)?;
    if tp_size > 1 {
        let loader = metrale_model_engine::factory::loader_for_config(config)?;
        let support = loader.tp_support();
        let slices_experts = loader.slices_experts_over_tp();
        drop(loader);
        if config.moe_expert_layout == metrale_config::MoeExpertLayout::Tp && !slices_experts {
            anyhow::bail!(
                "--moe-expert-layout tp is not supported by the {} weight loader: it does not \
                 slice routed experts over TP. Use --moe-expert-layout ep.",
                config.model_type,
            );
        }
        if support == metrale_config::TpSupport::Unsupported {
            anyhow::bail!(
                "TP (--tp-size > 1) is not supported by the {} weight loader. \
                 Run with --tp-size 1 (EP-only). To extend TP to this architecture, \
                 wire `crate::tp_shard::slice_for_rank` per attention/MoE/SSM \
                 tensor in the loader and override `ModelWeightLoader::tp_support()` \
                 to return `Even`. See `weight_loader/minimax.rs` as the reference.",
                config.model_type,
            );
        }
        // 2026-10-08: `Even` refuses a head count that does not divide over `tp_size`, as
        // this phase always did; `Uneven` (GLM-5.3) splits it with `metrale_config::tp_split`.
        config.shard_heads_for_tp(support)?;
        tracing::info!(
            "TP-local head counts: num_attention_heads={}, num_key_value_heads={}, \
             linear_num_key_heads={}, linear_num_value_heads={}",
            config.num_attention_heads,
            config.num_key_value_heads,
            config.linear_num_key_heads,
            config.linear_num_value_heads,
        );
    }
    if world_size > 1 {
        let (start, end) = config.local_expert_range();
        tracing::info!(
            "Parallelism: global rank {}/{} (tp_rank={}/{}, ep_rank={}/{}), local experts [{}, {})",
            args.rank,
            world_size,
            tp_rank,
            tp_size,
            ep_rank,
            ep_size,
            start,
            end,
        );
        if config.moe_expert_layout == metrale_config::MoeExpertLayout::Tp {
            tracing::info!(
                "MoE expert layout tp: every routed expert's intermediate width is sliced over \
                 the {tp_size} TP ranks; all experts are local on every rank"
            );
        }
    }
    Ok(Topology {
        world_size,
        tp_size,
        ep_size,
        tp_rank,
        ep_rank,
    })
}

/// 2026-09-26: The NCCL communicator, or `None` when `world_size <= 1`. The
/// receive buffer holds `max_batch_tokens x max(hidden_size, vocab_size)`
/// elements of `ALL_REDUCE_DTYPE_BYTES` (`required_model_recv_bytes`).
#[cfg(feature = "nccl")]
pub(crate) fn init_nccl_comm(
    args: &cli::ServeArgs,
    gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
    world_size: usize,
    max_batch_tokens: usize,
    hidden_size: usize,
    vocab_size: usize,
) -> Result<Option<std::sync::Arc<dyn metrale_comm::CommBackend>>> {
    use metrale_comm::CommBackend;
    if world_size <= 1 {
        return Ok(None);
    }
    let recv_capacity = metrale_comm::nccl_backend::required_model_recv_bytes(
        max_batch_tokens,
        hidden_size,
        vocab_size,
    )
    .context("Failed to size the NCCL receive buffer")?;
    tracing::info!(
        "Initializing NCCL: rank {}/{}, master {}:{}, recv_buffer {} MiB \
         (max_batch_tokens={} × max(hidden_size,vocab_size)={} × {} B)",
        args.rank,
        world_size,
        args.master_addr,
        args.master_port,
        recv_capacity / (1024 * 1024),
        max_batch_tokens,
        hidden_size.max(vocab_size),
        metrale_comm::nccl_backend::ALL_REDUCE_DTYPE_BYTES,
    );
    let cuda_stream = gpu.default_stream();
    let backend = metrale_comm::NcclBackend::new(
        args.rank,
        world_size,
        &args.master_addr,
        args.master_port,
        cuda_stream,
        recv_capacity,
        args.all_reduce_oneshot_max_kb
            .checked_mul(1024)
            .context("--all-reduce-oneshot-max-kb overflows")?,
    )
    .context("Failed to initialize NCCL")?;
    tracing::info!("NCCL initialized: rank {}", backend.rank());
    Ok(Some(
        std::sync::Arc::new(backend) as std::sync::Arc<dyn metrale_comm::CommBackend>
    ))
}

/// 2026-09-26: A `cuda` build without `nccl` has no collectives:
/// `world_size > 1` is an error, and otherwise there is no communicator.
#[cfg(all(feature = "cuda", not(feature = "nccl")))]
pub(crate) fn init_nccl_comm(
    _args: &cli::ServeArgs,
    _gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
    world_size: usize,
    _max_batch_tokens: usize,
    _hidden_size: usize,
    _vocab_size: usize,
) -> Result<Option<std::sync::Arc<dyn metrale_comm::CommBackend>>> {
    if world_size > 1 {
        anyhow::bail!(
            "multi-rank NCCL is not available in this build (cuda feature \
             without nccl — SCALE/AMD gfx1151 has no NCCL library); \
             single-device only"
        );
    }
    Ok(None)
}

/// 2026-09-26: A `metal` build has no collectives: `world_size > 1` is an
/// error, and otherwise there is no communicator.
#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub(crate) fn init_nccl_comm(
    _args: &cli::ServeArgs,
    _gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
    world_size: usize,
    _max_batch_tokens: usize,
    _hidden_size: usize,
    _vocab_size: usize,
) -> Result<Option<std::sync::Arc<dyn metrale_comm::CommBackend>>> {
    if world_size > 1 {
        anyhow::bail!(
            "multi-rank NCCL is not available on Apple Silicon (metal feature); \
             single-device only"
        );
    }
    Ok(None)
}
