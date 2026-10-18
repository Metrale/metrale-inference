// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Step 4 of `build_model`, the buffer arena, moved out of `build.rs`; sized with the
//! `ArenaLevers` projected from the startup settings.
//!
//! Owner: metrale-model-engine.
//! Invariants:
//! - The lean arena is for `glm5_next` stacks only; on another model type the lever is refused.

use anyhow::Result;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::buffers::{ArenaLevers, BufferArena};
use metrale_gpu_runtime::gpu::GpuBackend;

/// 2026-09-25: The buffer arena. 2026-10-01: Sized for the slot ceiling (`auto` resolves after
/// it, at KV sizing); the preflight sized the arena reserve for the same ceiling.
/// 2026-10-12: `METRALE_GLM_ARENA_LEAN=1`: a `glm5_next` stack gets the arena with the other
/// architectures' regions at 256 B (`BufferSizes::lean_hc_stack`); on another model type the
/// lever is refused rather than ignored.
pub(super) fn buffer_arena(
    config: &ModelConfig,
    max_batch_tokens: usize,
    max_seq_len: usize,
    kv_block_size: usize,
    slot_ceiling: usize,
    arena_levers: &ArenaLevers,
    gpu: &dyn GpuBackend,
    settings: &metrale_settings::InferenceServerSettings,
) -> Result<BufferArena> {
    let arena_lean = metrale_model_arch::glm5next_layer::arena_lean(settings);
    anyhow::ensure!(
        !arena_lean || config.model_type == "glm5_next",
        "METRALE_GLM_ARENA_LEAN=1 is for glm5_next models, not {:?}",
        config.model_type
    );
    if arena_lean {
        let sizes = metrale_gpu_runtime::buffers::BufferSizes::from_config_with(
            config,
            max_batch_tokens,
            max_seq_len,
            kv_block_size,
            slot_ceiling,
            arena_levers,
        )
        .lean_hc_stack();
        BufferArena::from_sizes(config, sizes, max_batch_tokens, slot_ceiling, gpu)
    } else {
        BufferArena::new_with(
            config,
            max_batch_tokens,
            max_seq_len,
            kv_block_size,
            slot_ceiling,
            arena_levers,
            gpu,
        )
    }
}
