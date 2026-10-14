// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: Causal prefill attention gives a row the same bytes whatever rows follow it: the
//! premise of stopping dense attention at a QSA inert bound (`QsaLayer::dense_rows`). For each
//! BF16 HDIM-256 kernel a QSA layer can run (`attn_prefill_paged_64` and its 128-row twin over a
//! paged cache, `attn_prefill_64` and its twin over contiguous K/V), a launch of the leading `d`
//! rows (`kv_len = q_offset + d`) must equal the first `d` rows of the full launch, byte for
//! byte. Cases: the first 4096-row chunk of a 32k prompt cut at the model's bound (2051), and a
//! chunk after a cached prefix that straddles it.
//!
//! GPU test: run with
//! ```text
//! cargo test -p metrale-model-layers --release --lib prefill_attn_rows_tests -- --ignored --nocapture
//! ```

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::AttnFa128Kernels;

const HD: usize = 256;
const NQ: usize = 24;
const NKV: usize = 2;
const BS: usize = 16;
const SENTINEL: u8 = 0x5a;
/// 2026-10-14: (q_len, q_offset, kept rows d).
const CASES: [(usize, usize, usize); 2] = [(4096, 0, 2051), (300, 1900, 151)];

struct Rng(u64);
impl Rng {
    /// 2026-10-14: BF16 bytes of values in [-scale, scale).
    fn bf16_bytes(&mut self, n: usize, scale: f32) -> Vec<u8> {
        (0..n)
            .flat_map(|_| {
                self.0 = self
                    .0
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = (self.0 >> 40) as f32 / (1u64 << 24) as f32;
                let f = (2.0 * u - 1.0) * scale;
                ((f.to_bits() >> 16) as u16).to_le_bytes()
            })
            .collect()
    }
}

fn upload(g: &dyn GpuBackend, b: &[u8]) -> DevicePtr {
    let p = g.alloc(b.len()).unwrap();
    g.copy_h2d(b, p).unwrap();
    p
}

/// 2026-10-14: Run `launch(out, rows)` into a sentinel-filled buffer; its bytes, or `None` when
/// the kernel does not apply on this target (a twin the target does not compile).
fn run(
    g: &dyn GpuBackend,
    rows: usize,
    launch: &dyn Fn(DevicePtr, u32) -> Result<bool>,
) -> Result<Option<Vec<u8>>> {
    let bytes = rows * NQ * HD * 2;
    let out = g.alloc(bytes)?;
    g.memset(out, SENTINEL, bytes)?;
    let ran = launch(out, rows as u32)?;
    g.synchronize(0)?;
    let mut raw = vec![0u8; bytes];
    g.copy_d2h(out, &mut raw)?;
    g.free(out)?;
    Ok(ran.then_some(raw))
}

#[test]
#[ignore]
fn leading_rows_of_causal_prefill_attention_do_not_depend_on_later_rows() -> Result<()> {
    let set = metrale_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("qwen3.8-flash-next/nvfp4 not in this build");
    let gpu = metrale_gpu_runtime::cuda_backend::MetraleCudaBackend::new(0, &set.modules)?;
    let g: &dyn GpuBackend = &gpu;
    let twins = AttnFa128Kernels::resolve(g);
    let paged64 = g.kernel("prefill_paged", "attn_prefill_paged_64")?;
    let contig64 = g.kernel("attn_prefill", "attn_prefill_64")?;
    let isd = 1.0 / (HD as f32).sqrt();
    let (h, kh, hd, bs) = (NQ as u32, NKV as u32, HD as u32, BS as u32);
    let mut rng = Rng(0x5153_4144_2026_1014);
    for (q_len, q_off, d) in CASES {
        let kv = q_off + q_len;
        let pages = kv.div_ceil(BS);
        let table: Vec<u8> = (0..pages as i32)
            .rev()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        let q = upload(g, &rng.bf16_bytes(q_len * NQ * HD, 3.0));
        let kc = upload(g, &rng.bf16_bytes(pages * BS * NKV * HD, 1.0));
        let vc = upload(g, &rng.bf16_bytes(pages * BS * NKV * HD, 1.0));
        let bt = upload(g, &table);
        let qo = q_off as u32;
        let paged_old = |o, n: u32| {
            crate::layers::ops::prefill_attention_paged_64(
                g,
                paged64,
                q,
                kc,
                vc,
                o,
                bt,
                n,
                qo + n,
                qo,
                h,
                kh,
                hd,
                bs,
                0,
                isd,
                0,
            )
            .map(|_| true)
        };
        let paged_twin =
            |o, n: u32| twins.paged(g, q, kc, vc, o, bt, n, qo + n, qo, h, kh, hd, bs, 0, isd, 0);
        let mut kernels: Vec<(&str, &dyn Fn(DevicePtr, u32) -> Result<bool>)> =
            vec![("paged_64", &paged_old), ("paged twin", &paged_twin)];
        // 2026-10-14: Contiguous K/V are the cache's first rows: the cache-skip path starts at 0.
        let contig_old = |o, n: u32| {
            crate::layers::ops::prefill_attention_64(
                g, contig64, q, kc, vc, o, n, 1, h, kh, hd, isd, true, 0, 0,
            )
            .map(|_| true)
        };
        let contig_twin =
            |o, n: u32| twins.contiguous(g, q, kc, vc, o, n, 1, h, kh, hd, isd, true, 0, 0);
        if q_off == 0 {
            kernels.push(("contiguous_64", &contig_old));
            kernels.push(("contiguous twin", &contig_twin));
        }
        for (name, launch) in kernels {
            let Some(full) = run(g, q_len, launch)? else {
                println!("{name}: not on this target");
                continue;
            };
            let kept = run(g, d, launch)?.expect("applied at the full length");
            ensure!(
                full.chunks_exact(2).all(|v| v != [SENTINEL; 2]),
                "{name}: unwritten values"
            );
            let row = NQ * HD * 2;
            let diff = kept
                .chunks_exact(2)
                .zip(full[..d * row].chunks_exact(2))
                .filter(|(a, b)| a != b)
                .count();
            println!("{name} q_len {q_len} q_offset {q_off} kept {d}: {diff} values differ");
            ensure!(
                diff == 0,
                "{name} q_len {q_len} q_offset {q_off}: {diff} of the first {d} rows' values differ"
            );
        }
        for p in [q, kc, vc, bt] {
            g.free(p)?;
        }
    }
    Ok(())
}
