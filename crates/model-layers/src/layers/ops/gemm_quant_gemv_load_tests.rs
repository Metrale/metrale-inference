// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: Which kernel lookups `DenseGemvBatchmDirect::load` makes. Every
//! lookup is recorded by the kernel audit and a failed one refuses the boot
//! (`serve_phases/kernel_gate.rs`), so the load must look up exactly the
//! compiled R's entries and nothing else. A probe of R = 4 before R = 2 once
//! refused every hopper boot of a GDN model while GPU tests passed.
//!
//! Owner: model-layers (kernel lookup).
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::mock::MockGpuBackend;

use super::*;

const MODULE: &str = "dense_gemv_bf16_batchm";

fn entries(r: u32) -> Vec<(String, String)> {
    (2..=DENSE_GEMV_BATCHM_MAX_M)
        .map(|m| {
            (
                MODULE.to_owned(),
                format!("dense_gemv_bf16_batchm_r{r}_m{m}"),
            )
        })
        .collect()
}

/// 2026-10-14: A build that compiled no direct entries makes no lookup at all,
/// so it leaves no failed row in the audit.
#[test]
fn no_compiled_r_makes_no_lookup() {
    let gpu = MockGpuBackend::new();
    assert!(DenseGemvBatchmDirect::load_compiled(&gpu, None).is_none());
    assert!(gpu.kernel_lookups_snapshot().is_empty());
}

/// 2026-10-14: The compiled R's entries, every M once, in order, and no other
/// R: the lookup of an R the build did not compile is what the gate refused.
#[test]
fn the_compiled_r_is_the_only_one_looked_up() {
    for r in [1, 2, 4] {
        let gpu = MockGpuBackend::new();
        let direct = DenseGemvBatchmDirect::load_compiled(&gpu, Some(r)).expect("all entries");
        assert_eq!(direct.outputs_per_group, r);
        assert_eq!(direct.by_m.len(), (DENSE_GEMV_BATCHM_MAX_M - 1) as usize);
        assert_eq!(gpu.kernel_lookups_snapshot(), entries(r));
    }
}

/// 2026-10-14: One missing entry of the compiled set disables the route rather
/// than launching a zero handle, and its failed lookup stays in the audit
/// (here: the mock's lookup log) for the gate to refuse.
#[test]
fn a_missing_entry_of_the_compiled_set_disables_the_route() {
    let gpu = MockGpuBackend::new();
    gpu.deny_kernel(MODULE, "dense_gemv_bf16_batchm_r2_m7");
    assert!(DenseGemvBatchmDirect::load_compiled(&gpu, Some(2)).is_none());
    let looked_up = gpu.kernel_lookups_snapshot();
    assert_eq!(
        looked_up.last().map(|(_, f)| f.as_str()),
        Some("dense_gemv_bf16_batchm_r2_m7")
    );
    assert_eq!(looked_up, entries(2)[..6].to_vec());
}

/// 2026-10-14: The constant `load` reads is the one build.rs baked for this
/// build's target; a hopper build compiles R = 2 (kernels/hopper/HARDWARE.toml).
#[test]
fn load_reads_the_baked_r() {
    let gpu = MockGpuBackend::new();
    let direct = DenseGemvBatchmDirect::load(&gpu);
    assert_eq!(
        direct.map(|d| d.outputs_per_group),
        metrale_kernels::TARGET_BATCHM_DIRECT_R
    );
}
