// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: `dsa_pool_keys_store_rows_bf16` against `dsa_pool_keys_store_rows`: over a flat
//! cache (block table NULL) of GLM-5.3's indexer width, every pool's BF16 key must be the RN
//! rounding of the FP32 key the FP32 store writes, in the first half of the same slot, and the
//! second half untouched.
//!
//! Owner: model-arch examples.
//! Invariants: returns an error unless every pool's bytes match.

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::{D, KP, Rng, bf16_rn, read, up};

/// 2026-10-23: Tokens of the flat cache (a ragged tail: the last 3 rows complete no pool).
const ROWS: usize = 4 * 997 + 3;

fn bf16(rng: &mut Rng, n: usize, scale: f32) -> Vec<u16> {
    (0..n).map(|_| bf16_rn(rng.unit() * scale)).collect()
}

pub fn gate(gpu: &dyn GpuBackend, rng: &mut Rng) -> Result<()> {
    let f32k = gpu.kernel("dsa_scores_warp", "dsa_pool_keys_store_rows")?;
    let bf16k = gpu.kernel("dsa_scores_heads_bf16", "dsa_pool_keys_store_rows_bf16")?;
    let pools = ROWS / KP;
    let k = up(gpu, &bf16(rng, ROWS * D, 3.0))?;
    let g = up(gpu, &bf16(rng, ROWS * D, 6.0))?;
    let ape: Vec<f32> = (0..KP * D).map(|_| rng.unit()).collect();
    let ape = up(gpu, &ape)?;
    let bytes = pools * D * 4;
    let run = |kernel, fill: u8| -> Result<Vec<u8>> {
        let pk = gpu.alloc(bytes)?;
        gpu.memset(pk, fill, bytes)?;
        KernelLaunch::new(gpu, kernel)
            .grid([1, ROWS as u32, 1])
            .block([128, 1, 1])
            .arg_ptr(DevicePtr::NULL)
            .arg_i32(0)
            .arg_ptr(k)
            .arg_ptr(g)
            .arg_ptr(pk)
            .arg_ptr(ape)
            .arg_u32(D as u32)
            .arg_u32(KP as u32)
            .arg_ptr(DevicePtr::NULL)
            .arg_u32(0)
            .arg_u32(64)
            .arg_u32(0)
            .arg_u32(0)
            .launch(0)?;
        let out = read(gpu, pk, bytes)?;
        gpu.free(pk)?;
        Ok(out)
    };
    let want = run(f32k, 0x00)?;
    let got = run(bf16k, 0x5A)?;
    for p in 0..pools {
        let slot = &got[p * D * 4..(p + 1) * D * 4];
        for d in 0..D {
            let o = (p * D + d) * 4;
            let x = f32::from_le_bytes([want[o], want[o + 1], want[o + 2], want[o + 3]]);
            let h = u16::from_le_bytes([slot[2 * d], slot[2 * d + 1]]);
            ensure!(
                h == bf16_rn(x),
                "dsa_pool_keys_store_rows_bf16: pool {p} channel {d}: {h:#06x}, the RN rounding \
                 of {x} is {:#06x}",
                bf16_rn(x)
            );
        }
        ensure!(
            slot[D * 2..].iter().all(|&b| b == 0x5A),
            "dsa_pool_keys_store_rows_bf16: pool {p} wrote past its BF16 half"
        );
    }
    for p in [k, g, ape] {
        gpu.free(p)?;
    }
    println!(
        "store: dsa_pool_keys_store_rows_bf16 is the RN rounding of dsa_pool_keys_store_rows \
         ({pools} pools x {D})"
    );
    Ok(())
}
