// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: Host-side tests of the W8A8 decode launchers: segment
//! validation, the K split, the token-tile choice and the availability hook.
//! The kernels' arithmetic is tested on the GPU by the model-arch example
//! `w8a8_decode_microtest`.

use super::*;

fn w(n: u32, k: u32, format: WeightQuantFormat) -> Fp8Weight {
    Fp8Weight {
        weight: DevicePtr(0x1000),
        row_scale: DevicePtr(0x2000),
        n,
        k,
        scale_format: format,
    }
}

fn kernels_all() -> W8a8Kernels {
    let h = KernelHandle(7);
    W8a8Kernels {
        quant_row: h,
        quant_g128: h,
        quant_silu_row: h,
        quant_silu_g128: h,
        rowscale: [h; 5],
        blk128: [h; 5],
    }
}

fn scratch(q_bytes: usize, scale_bytes: usize) -> W8a8Scratch {
    W8a8Scratch {
        q: DevicePtr(0x3000),
        q_bytes,
        scale: DevicePtr(0x4000),
        scale_bytes,
    }
}

/// 2026-09-28: Segment boundaries are the prefix sums, and a missing segment
/// sits at `n` so the kernel never selects it.
#[test]
fn boundaries_are_prefix_sums() {
    let r = WeightQuantFormat::Fp8PerRow;
    let one = W8a8Weight::new(&[w(5120, 5120, r)]).unwrap();
    assert_eq!((one.n(), one.boundaries()), (5120, (5120, 5120)));
    let two = W8a8Weight::new(&[w(10240, 5120, r), w(6144, 5120, r)]).unwrap();
    assert_eq!((two.n(), two.boundaries()), (16384, (10240, 16384)));
    let three = W8a8Weight::new(&[w(12288, 5120, r), w(1024, 5120, r), w(1024, 5120, r)]).unwrap();
    assert_eq!((three.n(), three.boundaries()), (14336, (12288, 13312)));
}

/// 2026-09-28: Every malformed stack is refused: a K the kernel cannot tile,
/// mixed K or layout, an unsupported layout, an interior segment off the row
/// tile (per-row) or scale block (block-scaled), and 0 or 4 segments. The last
/// segment may have any row count.
#[test]
fn malformed_stacks_are_refused() {
    let (r, b) = (
        WeightQuantFormat::Fp8PerRow,
        WeightQuantFormat::Fp8BlockScaled,
    );
    assert!(W8a8Weight::new(&[]).is_err());
    assert!(W8a8Weight::new(&[w(16, 128, r); 4]).is_err());
    assert!(W8a8Weight::new(&[w(16, 5000, r)]).is_err());
    assert!(W8a8Weight::new(&[w(16, 5120, r), w(16, 6144, r)]).is_err());
    assert!(W8a8Weight::new(&[w(128, 5120, r), w(128, 5120, b)]).is_err());
    assert!(W8a8Weight::new(&[w(16, 5120, WeightQuantFormat::Nvfp4)]).is_err());
    assert!(W8a8Weight::new(&[w(24, 5120, r), w(16, 5120, r)]).is_err());
    assert!(W8a8Weight::new(&[w(64, 5120, b), w(128, 5120, b)]).is_err());
    assert!(W8a8Weight::new(&[w(16, 5120, r), w(24, 5120, r)]).is_ok());
    assert!(W8a8Weight::new(&[w(128, 5120, b), w(64, 5120, b)]).is_ok());
    assert!(W8a8Weight::new(&[w(248077, 5120, r)]).is_ok());
}

/// 2026-09-28: 8, 16, 32, 64 and 128 are the last rows of each token-tile width.
#[test]
fn entry_edges() {
    let cases = [
        (1, 0),
        (8, 0),
        (9, 1),
        (16, 1),
        (17, 2),
        (32, 2),
        (33, 3),
        (64, 3),
        (65, 4),
        (128, 4),
    ];
    for (rows, e) in cases {
        assert_eq!(entry_index(rows), e, "rows={rows}");
        assert!(ENTRIES[e].starts_with(["mb1", "mb2", "mb4", "mb8", "mb16"][e]));
    }
}

/// 2026-09-28: The hook is false outside 1..=256 rows, with an unresolved
/// kernel of the weight's layout, and when the scratch is one byte short of
/// the activation or of its scales; true at the exact fit.
#[test]
fn availability_hook_edges() {
    let (r, b) = (
        WeightQuantFormat::Fp8PerRow,
        WeightQuantFormat::Fp8BlockScaled,
    );
    let wr = W8a8Weight::new(&[w(5120, 6144, r)]).unwrap();
    let wb = W8a8Weight::new(&[w(5120, 6144, b)]).unwrap();
    let k = kernels_all();
    let (rows, kb) = (W8A8_MAX_ROWS, 6144 / 128);
    let exact_r = scratch(rows * 6144, rows * 4);
    let exact_b = scratch(rows * 6144, rows * kb * 4);
    assert!(w8a8_decode_available(&k, &wr, rows, &exact_r));
    assert!(w8a8_decode_available(&k, &wb, rows, &exact_b));
    assert!(!w8a8_decode_available(&k, &wb, rows, &exact_r));
    assert!(!w8a8_decode_available(&k, &wr, 0, &exact_r));
    assert!(!w8a8_decode_available(&k, &wr, rows + 1, &exact_r));
    let short_q = scratch(rows * 6144 - 1, rows * 4);
    assert!(!w8a8_decode_available(&k, &wr, rows, &short_q));
    let short_s = scratch(rows * 6144, rows * 4 - 1);
    assert!(!w8a8_decode_available(&k, &wr, rows, &short_s));
    let mut missing = k;
    missing.blk128[4] = KernelHandle(0);
    assert!(w8a8_decode_available(&missing, &wr, 256, &exact_r));
    assert!(!w8a8_decode_available(&missing, &wb, 256, &exact_b));
    assert!(!w8a8_decode_available(
        &W8a8Kernels::none(),
        &wr,
        1,
        &exact_r
    ));
}
