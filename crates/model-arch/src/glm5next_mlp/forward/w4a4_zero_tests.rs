// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: `METRALE_GLM_MOE_ZERO_IN_SWIGLU` on the recording mock: when the routed SwiGLU
//! zeroes `expert_out` (`zeroes_in_swiglu`, which `forward_moe` also reads to skip its memset)
//! and the launch it then makes. The bytes are checked on a GPU by the `pdl_chain_gate` example.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::KernelHandle;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;

use super::tests::{
    SWIGLU, SWIGLU_ZERO, cfg, cfg_tp, f32_arg, kernels, ptr_arg, run_experts_on, u32_arg,
};
use super::{expert_out_bytes, zeroes_in_swiglu};

/// 2026-10-21: On at every width the slot GEMV or the sweep serves whose down projection leaves
/// rows unwritten (2..=16 rows; one row only without the own-slots down sweep); off with the
/// lever off, without the kernel, and on the grouped GEMM's widths.
#[test]
fn the_swiglu_zeroes_exactly_where_the_memset_would_run_and_the_kernel_is_there() {
    let gpu = MockGpuBackend::new();
    let k = kernels(&gpu);
    for c in [cfg(), cfg_tp()] {
        assert!(!zeroes_in_swiglu(&k, &c, 1), "the own-slots down writes every row");
        for rows in 2..=16 {
            assert!(zeroes_in_swiglu(&k, &c, rows), "{rows} rows");
        }
        assert!(!zeroes_in_swiglu(&k, &c, 17), "17 rows take the grouped GEMM");
    }
    let mut no_own = k;
    no_own.w4a4_moe_slots_sweep = KernelHandle(0);
    assert!(zeroes_in_swiglu(&no_own, &cfg(), 1));
    let mut off = k;
    off.levers.mlp.zero_in_swiglu = false;
    let mut absent = k;
    absent.swiglu_zero = KernelHandle(0);
    for rows in [1, 2, 16] {
        assert!(!zeroes_in_swiglu(&off, &cfg(), rows), "lever off, {rows} rows");
        assert!(!zeroes_in_swiglu(&absent, &cfg(), rows), "no kernel, {rows} rows");
    }
}

/// 2026-10-21: The zeroing SwiGLU takes the plain one's arguments and grid, then `expert_out` and
/// its 16-byte word count; with the lever off the plain SwiGLU runs in the same place.
#[test]
fn the_zeroing_swiglu_is_the_plain_launch_plus_expert_out() {
    for rows in [2usize, 3, 12] {
        let (_g, on, ws) = run_experts_on(cfg(), rows, |_| {});
        let (_g, off, _) = run_experts_on(cfg(), rows, |k| k.levers.mlp.zero_in_swiglu = false);
        let (z, p) = (&on[3], &off[3]);
        assert_eq!((z.func, p.func), (SWIGLU_ZERO, SWIGLU), "{rows} rows");
        assert_eq!((z.grid, z.block), (p.grid, p.block), "{rows} rows");
        assert_eq!(z.args[..5], p.args[..], "{rows} rows");
        assert_eq!(ptr_arg(z, 5), ws.expert_out);
        assert_eq!(
            u32_arg(z, 6) as usize * 16,
            expert_out_bytes(&cfg(), rows),
            "{rows} rows"
        );
        assert_eq!(expert_out_bytes(&cfg(), rows), rows * 8 * 4096 * 2);
        assert_eq!((u32_arg(z, 3), f32_arg(z, 4)), ((rows * 8 * 2048) as u32, 10.0));
    }
}
