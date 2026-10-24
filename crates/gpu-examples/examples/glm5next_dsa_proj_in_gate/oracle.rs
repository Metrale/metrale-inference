// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The kernels, the oracle and the output readback of the DSA proj-in gate.
//!
//! Owner: model-arch examples (GLM-5.3 kernels).
//! Invariants: none beyond the types.

use super::*;

pub(super) fn outs(g: &dyn GpuBackend, b: &Bufs) -> Result<Vec<Vec<u8>>> {
    g.synchronize(0)?;
    b.out
        .iter()
        .zip(ROW_BYTES)
        .map(|(&p, w)| read(g, p, b.m * w))
        .collect()
}

/// 2026-10-21: `split`: resolve the split and pair entries (the base PTX has neither).
pub(super) fn kernels(g: &dyn GpuBackend, split: bool) -> Result<Kernels> {
    Ok(Kernels {
        w8a8: W8a8Kernels::load(g),
        split4: if split {
            g.kernel(W8A8_SPLIT4_MODULE, W8A8_SPLIT4_ENTRY)?
        } else {
            KernelHandle(0)
        },
        rms: g.kernel("rms_norm_vanilla", "rms_norm_vanilla")?,
        few_f32: g.kernel(
            "dense_gemv_bf16_batchm_few",
            "dense_gemv_bf16_batchm_few_fp32out",
        )?,
        pair: if split {
            g.kernel(
                "dense_gemv_bf16_batchm_few",
                "dense_gemv_bf16_batchm_few_fp32out_pair",
            )?
        } else {
            KernelHandle(0)
        },
    })
}

/// 2026-10-21: The oracle's outputs per M on backend `g`, from the fixed seed.
pub(super) fn oracle(g: &dyn GpuBackend, k: &Kernels) -> Result<Vec<Vec<Vec<u8>>>> {
    let mut rng = Lcg(0x6473_6170);
    let l = layer(g, &mut rng)?;
    ROWS_TESTED
        .iter()
        .map(|&m| {
            let b = bufs(g, m, &hidden_rows(&mut rng, m))?;
            poison(g, &b)?;
            old(g, k, &l, &b, 0)?;
            outs(g, &b)
        })
        .collect()
}
