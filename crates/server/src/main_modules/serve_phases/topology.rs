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

use metrale_config::{IndexerCacheDtype, ModelConfig};

use crate::cli;

pub(crate) struct Topology {
    pub(crate) world_size: usize,
    pub(crate) tp_size: usize,
    pub(crate) ep_size: usize,
    pub(crate) tp_rank: usize,
    pub(crate) ep_rank: usize,
}

/// 2026-10-10: `(tp_size, ep_size, derived_world)` from the flags: the parallel sizes as
/// `resolve_topology` uses them, and the world size they imply when `--world-size` is left at
/// 1. Moved out of `resolve_topology` unchanged, so `serve_world_size` derives the same world.
fn parallel_sizes(args: &cli::ServeArgs) -> (usize, usize, usize) {
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
    (tp_size, ep_size, derived_world)
}

/// 2026-10-10: The serve's world size, as `resolve_topology` resolves it.
pub(crate) fn serve_world_size(args: &cli::ServeArgs) -> usize {
    let (tp_size, ep_size, derived_world) = parallel_sizes(args);
    if args.world_size <= 1 && (tp_size > 1 || ep_size > 1) {
        derived_world
    } else {
        args.world_size
    }
}

pub(crate) fn resolve_topology(
    args: &cli::ServeArgs,
    // 2026-10-17: `kv_paging` (the peer-paging backend) and the indexer layout.
    levers: &super::LoadLevers,
    config: &mut ModelConfig,
) -> Result<Topology> {
    let (tp_size, ep_size, derived_world) = parallel_sizes(args);
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
    set_indexer_cache_dtype(args, levers, config)?;
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

/// 2026-10-16: `--indexer-cache-dtype` and the rewind depth its FP8 tail ring is sized for: the
/// widest verify pass (`verify_pool_drafts`) when any drafter runs, else 0. `fp8` is refused off
/// its scope (`IndexerCacheDtype::check_serve`), which 2026-10-18 reads from the model's loader.
fn set_indexer_cache_dtype(
    args: &cli::ServeArgs,
    levers: &super::LoadLevers,
    config: &mut ModelConfig,
) -> Result<()> {
    let dtype = args.indexer_cache_dtype.0;
    let served = match dtype {
        IndexerCacheDtype::Bf16 => &[][..],
        IndexerCacheDtype::Fp8 => {
            metrale_model_engine::factory::loader_for_config(config)?.indexer_cache_dtypes()
        }
    };
    dtype.check_serve(
        &config.model_type,
        served,
        levers.prefix_cache_gate.glm_indexer_paged,
    )?;
    let drafting = args.speculative || args.prompt_lookup_config().is_some();
    config.indexer_cache_dtype = dtype;
    config.indexer_rewind_rows = if drafting {
        args.verify_pool_drafts()
    } else {
        0
    };
    if dtype != metrale_config::IndexerCacheDtype::Bf16 {
        tracing::info!(
            "indexer cache dtype {}: tail ring sized for a rewind of {} rows",
            dtype.name(),
            config.indexer_rewind_rows
        );
    }
    Ok(())
}

/// 2026-09-26: The NCCL communicator, or `None` when `world_size <= 1`. The
/// receive buffer holds `max_batch_tokens x max(hidden_size, vocab_size)`
/// elements of `ALL_REDUCE_DTYPE_BYTES` (`required_model_recv_bytes`).
#[cfg(feature = "nccl")]
pub(crate) fn init_nccl_comm(
    args: &cli::ServeArgs,
    settings: &metrale_settings::InferenceServerSettings,
    gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
    world_size: usize,
    max_batch_tokens: usize,
    hidden_size: usize,
    vocab_size: usize,
) -> Result<Option<std::sync::Arc<dyn metrale_comm::CommBackend>>> {
    if world_size <= 1 {
        return Ok(None);
    }
    let recv_capacity =
        metrale_comm::required_model_recv_bytes(max_batch_tokens, hidden_size, vocab_size)
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
        metrale_comm::ALL_REDUCE_DTYPE_BYTES,
    );
    let backend = metrale_comm::bootstrap(
        args.rank,
        world_size,
        &args.master_addr,
        args.master_port,
        gpu.default_stream(),
        recv_capacity,
        metrale_comm::CommLevers {
            oneshot_max_bytes: args
                .all_reduce_oneshot_max_kb
                .checked_mul(1024)
                .context("--all-reduce-oneshot-max-kb overflows")?,
            diagnostics: settings.telemetry.comm_diagnostics.as_deref(),
            poll: &settings.runtime.comm_poll,
        },
    )
    .context("Failed to initialize NCCL")?;
    tracing::info!("NCCL initialized: rank {}", backend.rank());
    Ok(Some(backend))
}

/// 2026-10-21: The control communicator's port (`METRALE_GLM_EP_CTRL_COMM`): `None` when the
/// lever is off or the serve has one rank; required when it is on (no default), and refused
/// where its bootstrap would collide with the main communicator's (`master_port`, and the
/// `master_port + 1` its reconnect binds) or its own reconnect port would.
pub(crate) fn ctrl_port(
    lever: bool,
    port: Option<i64>,
    master_port: u16,
    world_size: usize,
) -> Result<Option<u16>> {
    if !lever || world_size <= 1 {
        return Ok(None);
    }
    let port: u16 =
        crate::lever_value::narrow_opt("METRALE_GLM_EP_CTRL_PORT", port)?.ok_or_else(|| {
            anyhow::anyhow!(
                "METRALE_GLM_EP_CTRL_COMM=1 requires METRALE_GLM_EP_CTRL_PORT (the control \
                 communicator's bootstrap port; there is no default)"
            )
        })?;
    let taken = [master_port, master_port.wrapping_add(1)];
    anyhow::ensure!(
        port != 0 && !taken.contains(&port) && !taken.contains(&port.wrapping_add(1)),
        "METRALE_GLM_EP_CTRL_PORT={port} collides with the main communicator's bootstrap ports \
         {taken:?} (--master-port and its reconnect port); pick a port at least 2 away"
    );
    Ok(Some(port))
}

/// 2026-10-21: Bootstrap the control communicator on `port`, its operations on `stream` (the
/// model's greatest-priority control stream). It carries only control frames: no receive
/// buffer to speak of, no one-shot all-reduce.
#[cfg(feature = "nccl")]
pub(crate) fn init_ctrl_comm(
    args: &cli::ServeArgs,
    settings: &metrale_settings::InferenceServerSettings,
    world_size: usize,
    port: u16,
    stream: u64,
) -> Result<std::sync::Arc<dyn metrale_comm::CommBackend>> {
    tracing::info!(
        "Initializing the NCCL control communicator: rank {}/{world_size}, master {}:{port}",
        args.rank,
        args.master_addr,
    );
    metrale_comm::bootstrap(
        args.rank,
        world_size,
        &args.master_addr,
        port,
        stream,
        // 2026-10-21: World 2 requires a non-zero two-rank all-reduce buffer; this communicator
        // never all-reduces, so one page.
        4096,
        metrale_comm::CommLevers {
            oneshot_max_bytes: 0,
            diagnostics: settings.telemetry.comm_diagnostics.as_deref(),
            poll: &settings.runtime.comm_poll,
        },
    )
    .context("Failed to initialize the NCCL control communicator")
}

/// 2026-10-21: A build without `nccl` has no control communicator.
#[cfg(not(feature = "nccl"))]
pub(crate) fn init_ctrl_comm(
    _args: &cli::ServeArgs,
    _settings: &metrale_settings::InferenceServerSettings,
    _world_size: usize,
    _port: u16,
    _stream: u64,
) -> Result<std::sync::Arc<dyn metrale_comm::CommBackend>> {
    anyhow::bail!("METRALE_GLM_EP_CTRL_COMM needs a build with NCCL")
}

/// 2026-09-26: A `cuda` build without `nccl` has no collectives:
/// `world_size > 1` is an error, and otherwise there is no communicator.
#[cfg(all(feature = "cuda", not(feature = "nccl")))]
pub(crate) fn init_nccl_comm(
    _args: &cli::ServeArgs,
    _settings: &metrale_settings::InferenceServerSettings,
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
    _settings: &metrale_settings::InferenceServerSettings,
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

#[cfg(test)]
mod ctrl_port_tests {
    use super::ctrl_port;

    /// 2026-10-21: The port is required with the lever on a multi-rank serve and ignored
    /// otherwise; a port the main communicator binds (its own or its reconnect's), or one whose
    /// reconnect port it binds, is refused, as is one out of `u16` range.
    #[test]
    fn the_ctrl_port_is_required_and_kept_off_the_main_communicators_ports() {
        assert_eq!(ctrl_port(false, None, 29500, 3).unwrap(), None);
        assert_eq!(
            ctrl_port(true, None, 29500, 1).unwrap(),
            None,
            "one rank: no channel"
        );
        let e = ctrl_port(true, None, 29500, 3).unwrap_err().to_string();
        assert!(e.contains("requires METRALE_GLM_EP_CTRL_PORT"), "{e}");
        assert_eq!(ctrl_port(true, Some(29510), 29500, 3).unwrap(), Some(29510));
        assert_eq!(ctrl_port(true, Some(29502), 29500, 3).unwrap(), Some(29502));
        for taken in [29499, 29500, 29501] {
            assert!(ctrl_port(true, Some(taken), 29500, 3).is_err(), "{taken}");
        }
        assert!(ctrl_port(true, Some(0), 29500, 3).is_err());
        assert!(ctrl_port(true, Some(70_000), 29500, 3).is_err());
    }
}
