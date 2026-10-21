// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: GPU runtime: the `GpuBackend` trait with its CUDA, Metal and mock
//! implementations, device buffers, streams, the kernel registry, the cuBLASLt,
//! CUTLASS and FlashInfer bridges, and the process-wide mid-chunk SSM tail-capture
//! switch.
//!
//! Owner: gpu-runtime.
//! Invariants: none beyond the types.

#![deny(warnings)]
#![deny(clippy::all)]

pub mod buffers;
#[cfg(feature = "cuda")]
pub mod cublaslt;
// 2026-09-25: Without the `cuda` feature these modules come from the
// `*_metal_stub.rs` files, which keep the entry points metrale-model-layers calls
// without a `cfg`; their GPU entry points are `unreachable!`.
#[cfg(not(feature = "cuda"))]
#[path = "cublaslt_metal_stub.rs"]
pub mod cublaslt;
#[cfg(feature = "cuda")]
pub mod cuda_backend;
#[cfg(feature = "cuda")]
pub mod cutlass;
#[cfg(not(feature = "cuda"))]
#[path = "cutlass_metal_stub.rs"]
pub mod cutlass;
#[cfg(feature = "cuda")]
pub mod flashinfer;
#[cfg(not(feature = "cuda"))]
#[path = "flashinfer_metal_stub.rs"]
pub mod flashinfer;
pub mod gpu;
#[path = "gpu_args.rs"]
mod gpu_args;
pub mod graph_hoist;
pub mod host_heap;
pub mod host_sync_counts;
pub mod kernel_args;
#[cfg(feature = "metal")]
pub mod metal_backend;
pub mod op_cache;
pub mod pinned_hosts;
pub mod prefill_tail_mode;
pub mod timing;

/// 2026-09-25: The last multiple of `block_size` strictly below `total_tokens`, or
/// `None` when `block_size` is 0 or `total_tokens <= block_size`.
///
/// The SSM tail snapshot is placed here: in-pass by `prepare_midchunk_capture`
/// (metrale-model-engine), or, with the `scheduler.ssm_tail_ckpt` setting on and
/// `ssm_tail_midchunk_enabled` off, by ending a prefill chunk here
/// (`run_standard` and `run_batched_mixed` in met).
pub fn ssm_tail_boundary(total_tokens: usize, block_size: usize) -> Option<usize> {
    if block_size == 0 || total_tokens <= block_size {
        return None;
    }
    let boundary = ((total_tokens - 1) / block_size) * block_size;
    (boundary > 0).then_some(boundary)
}

/// 2026-10-17: The mid-chunk SSM tail capture switch from its resolved setting text
/// (`scheduler.ssm_tail_midchunk`, `METRALE_SSM_TAIL_MIDCHUNK`): on unless `0`.
pub fn ssm_tail_midchunk_on(setting: Option<&str>) -> bool {
    setting != Some("0")
}

/// 2026-09-25: Publish the serve's mid-chunk capture switch before the first
/// `ssm_tail_midchunk_enabled` read. 2026-10-17: The serve publishes `false` for
/// `--no-ssm-tail-midchunk`, else [`ssm_tail_midchunk_on`] of its resolved setting. The cell is
/// set once: the first published value, or the first read, wins.
///
/// ROTI: a startup static, until the model reads the switch from its startup settings.
pub fn set_ssm_tail_midchunk(on: bool) {
    let _ = SSM_TAIL_MIDCHUNK.set(on);
}

static SSM_TAIL_MIDCHUNK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// 2026-09-25: Mid-chunk SSM tail capture: the published switch, or the declared on when
/// nothing was published. While it is on the scheduler does not end chunks
/// at `ssm_tail_boundary`, and `prepare_midchunk_capture` (metrale-model-engine)
/// plans an in-pass capture there, which it does only on `metrale_scale` builds.
pub fn ssm_tail_midchunk_enabled() -> bool {
    *SSM_TAIL_MIDCHUNK.get_or_init(|| ssm_tail_midchunk_on(None))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_zero_setting_turns_the_midchunk_capture_off() {
        // 2026-10-17: The deleted environment reader's rule: anything but `0` is on.
        assert!(ssm_tail_midchunk_on(None));
        assert!(!ssm_tail_midchunk_on(Some("0")));
        assert!(ssm_tail_midchunk_on(Some("1")));
        assert!(ssm_tail_midchunk_on(Some("off")));
    }

    #[test]
    fn the_first_published_midchunk_switch_is_final() {
        // 2026-09-25: The cell is process-global with no reset, so no other test
        // in this crate may touch it.
        set_ssm_tail_midchunk(false);
        assert!(!ssm_tail_midchunk_enabled());
        set_ssm_tail_midchunk(true);
        assert!(!ssm_tail_midchunk_enabled(), "a published switch is final");
    }

    #[test]
    fn the_tail_boundary_is_the_last_block_strictly_below_the_prompt() {
        // 2026-09-25: No boundary gives `None`, not 0: a snapshot at token 0
        // would restore nothing.
        assert_eq!(ssm_tail_boundary(0, 16), None);
        assert_eq!(ssm_tail_boundary(16, 16), None, "not the prompt's own end");
        assert_eq!(ssm_tail_boundary(17, 16), Some(16));
        assert_eq!(ssm_tail_boundary(32, 16), Some(16), "strictly below");
        assert_eq!(ssm_tail_boundary(33, 16), Some(32));
        assert_eq!(ssm_tail_boundary(100, 0), None, "no division by zero");
    }
}

#[cfg(feature = "cuda")]
pub mod cuda_host;
#[cfg(feature = "cuda")]
pub mod kernel;
pub mod mlx_int8;
#[cfg(feature = "cuda")]
mod ptx_globals;
#[cfg(feature = "cuda")]
mod ptx_pdl;
#[cfg(feature = "cuda")]
pub mod registry;
#[cfg(feature = "cuda")]
mod registry_pdl;
#[cfg(feature = "cuda")]
pub mod stream;
