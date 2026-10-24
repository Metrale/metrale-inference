// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: `sweep_entries`: the names the lever selects, that each is a sweep entry of the
//! kernel source with the launch mode its name claims, and the fallback of an absent `_sl` entry.
//! Whether the compiled `_sl` entries really hold no `griddepcontrol.wait` is checked on the real
//! PTX by the `pdl_chain_gate` example.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::mock::MockGpuBackend;

use super::*;

#[test]
fn the_lever_picks_the_serial_twins_and_off_keeps_the_plain_entries() {
    assert_eq!(
        names(true),
        [
            "w4a4_gemv_mx8_moe_union_sweep_sl",
            "w4a4_gemv_mx16_moe_union_sweep_sl",
        ]
    );
    assert_eq!(
        names(false),
        [
            "w4a4_gemv_mx8_moe_union_sweep",
            "w4a4_gemv_mx16_moe_union_sweep",
        ]
    );
}

/// 2026-10-21: Each name is an entry of the kernel source; the `_sl` twins are the SL
/// instantiation and the plain entries are not, at the same MB, PF, MINB, KH and OWN.
#[test]
fn every_name_is_a_sweep_entry_with_its_launch_mode() {
    const SRC: &str = include_str!("../../../../../kernels/gb10/common/w4a4_gemv_mx_moe.cu");
    let entry = |name: &str| -> Vec<&str> {
        let head = format!("W4A4_SWEEP_ENTRY({name}, ");
        let line = SRC
            .lines()
            .find(|l| l.starts_with(&head))
            .unwrap_or_else(|| panic!("{name}: no W4A4_SWEEP_ENTRY"));
        line[head.len()..line.len() - 1].split(", ").collect()
    };
    for (plain, sl) in GATE_UP_SWEEPS {
        let (p, s) = (entry(plain), entry(sl));
        assert_eq!(p.last(), Some(&"false"), "{plain}: SL");
        assert_eq!(s.last(), Some(&"true"), "{sl}: SL");
        assert_eq!(p[..p.len() - 1], s[..s.len() - 1], "{sl}: MB, PF, MINB, KH, OWN");
    }
}

#[test]
fn an_absent_serial_twin_falls_back_to_its_plain_entry() {
    let gpu = MockGpuBackend::new();
    gpu.deny_kernel(W4A4_MOE_MODULE, "w4a4_gemv_mx16_moe_union_sweep_sl");
    let h = gate_up_sweeps(&gpu, true);
    assert!(h.iter().all(|h| h.0 != 0), "{h:?}");
    let looked: Vec<String> = gpu
        .kernel_lookups_snapshot()
        .into_iter()
        .map(|(_, f)| f)
        .collect();
    assert_eq!(
        looked,
        [
            "w4a4_gemv_mx8_moe_union_sweep_sl",
            "w4a4_gemv_mx16_moe_union_sweep_sl",
            "w4a4_gemv_mx16_moe_union_sweep",
        ]
    );
    let off = MockGpuBackend::new();
    gate_up_sweeps(&off, false);
    assert_eq!(
        off.kernel_lookups_snapshot()
            .into_iter()
            .map(|(_, f)| f)
            .collect::<Vec<_>>(),
        [
            "w4a4_gemv_mx8_moe_union_sweep",
            "w4a4_gemv_mx16_moe_union_sweep",
        ]
    );
}
