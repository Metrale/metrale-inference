// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: `deinterleave_qg` against the permutation it defines, computed on the host:
//! every row's `[Q_h | G_h]` pairs become `[Q_0..Q_n | G_0..G_n]`, and nothing between rows
//! or past the last row is written. The shapes cover its 16-byte vector path (the Flash-Next
//! gated Q: 24 heads of 256, at one, eight and 59 tokens), its scalar path (head_dim not a
//! multiple of 8; a buffer starting 2 bytes past alignment), a stride that leaves some rows
//! aligned and others not (both paths in one launch), and a row smaller than the block.
//!
//! GPU test: run with
//! ```text
//! cargo test -p metrale-model-layers --release --lib deinterleave_qg_tests -- --ignored --nocapture
//! ```
//! on a host whose build has the `qwen3.8-flash-next/nvfp4` kernel set.

use metrale_gpu_runtime::gpu::GpuBackend;

use super::deinterleave_qg;

/// 2026-10-14: (num_heads, head_dim, stride in elements, tokens, byte offset of the buffer).
const CASES: [(usize, usize, usize, usize, usize); 7] = [
    (24, 256, 13_312, 1, 0),
    (24, 256, 13_312, 8, 0),
    (24, 256, 13_312, 59, 0),
    (24, 256, 12_289, 3, 0),
    (24, 256, 13_312, 2, 2),
    (5, 36, 360, 4, 0),
    (2, 8, 40, 5, 0),
];
/// 2026-10-14: Elements past the last row that must stay as they were.
const TAIL: usize = 64;

/// 2026-10-14: Distinct-looking u16 values from a xorshift stream.
fn values(n: usize, seed: u32) -> Vec<u16> {
    let mut x = seed | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x as u16
        })
        .collect()
}

/// 2026-10-14: The permutation on the host, row by row, in place.
fn reference(data: &mut [u16], (nh, hd, stride, tokens): (usize, usize, usize, usize)) {
    for t in 0..tokens {
        let row = &mut data[t * stride..][..2 * nh * hd];
        let src = row.to_vec();
        for h in 0..nh {
            row[h * hd..][..hd].copy_from_slice(&src[2 * h * hd..][..hd]);
            row[(nh + h) * hd..][..hd].copy_from_slice(&src[(2 * h + 1) * hd..][..hd]);
        }
    }
}

fn bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[test]
#[ignore]
fn deinterleave_qg_writes_the_permutation_and_nothing_else() {
    let set = metrale_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("qwen3.8-flash-next/nvfp4 not in this build");
    let gpu = metrale_gpu_runtime::cuda_backend::MetraleCudaBackend::new(0, &set.modules)
        .expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let k = g.kernel("ssm_preprocess", "deinterleave_qg").unwrap();
    for (c, (nh, hd, stride, tokens, offset)) in CASES.into_iter().enumerate() {
        let n = (tokens - 1) * stride + 2 * nh * hd + TAIL;
        let host = values(n, 0x9E37_79B9 ^ c as u32);
        let base = g.alloc(n * 2 + offset).unwrap();
        let p = base.offset(offset);
        g.copy_h2d(&bytes(&host), p).unwrap();
        deinterleave_qg(
            g,
            k,
            p,
            tokens as u32,
            nh as u32,
            hd as u32,
            stride as u32,
            0,
        )
        .unwrap();
        g.synchronize(0).unwrap();
        let mut got = vec![0u8; n * 2];
        g.copy_d2h(p, &mut got).unwrap();
        let mut want = host.clone();
        reference(&mut want, (nh, hd, stride, tokens));
        assert_ne!(want, host, "case {c}: the permutation moved nothing");
        assert_eq!(
            got,
            bytes(&want),
            "case {c}: ({nh}, {hd}, {stride}, {tokens}, +{offset})"
        );
    }
}
