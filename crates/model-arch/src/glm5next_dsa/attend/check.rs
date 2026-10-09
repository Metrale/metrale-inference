// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: `METRALE_GLM_DSA_DECODE_HB_CHECK=<dir>`, a diagnostic for the head-batched DSA
//! decode on a serving path: after each eager head-batched launch (never inside a graph
//! capture), the per-head kernel runs the same inputs into a scratch buffer, the two outputs are
//! compared on the host, and the first rows that differ by more than [`DUMP_REL`] of their
//! head's scale are written to `<dir>/case_<n>/` with everything needed to replay them
//! (`glm5next_dsa_hb_replay`). It synchronizes the stream and copies to the host on every call:
//! a debugging aid, never a serving setting.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - Off (no extra launch, no host copy) unless the variable names a directory.
//! - At most [`MAX_DUMPS`] cases are written per process.

use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, Result};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::{DsaDecodeInputs, DsaDecodePaging, Glm5NextDsaConfig, launch_per_head};
use crate::glm5next_dsa::select::DsaSelectGeometry;

/// 2026-10-09: A row whose worst difference exceeds this fraction of its head's largest |O| is
/// dumped: about 8 BF16 ulps, far above the reordering noise (under one ulp).
const DUMP_REL: f32 = 1.0 / 32.0;
const MAX_DUMPS: usize = 8;

static DUMPS: AtomicUsize = AtomicUsize::new(0);
static CALLS: AtomicUsize = AtomicUsize::new(0);
static BAD_CALLS: AtomicUsize = AtomicUsize::new(0);

/// 2026-10-09: The check directory, read once; `None` when unset.
pub(super) fn check_dir() -> Result<Option<&'static str>> {
    static V: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let v = V.get_or_init(|| std::env::var("METRALE_GLM_DSA_DECODE_HB_CHECK").ok());
    if let Some(d) = v {
        std::fs::create_dir_all(d)
            .with_context(|| format!("METRALE_GLM_DSA_DECODE_HB_CHECK={d}"))?;
    }
    Ok(v.as_deref())
}

fn bf16s(b: &[u8]) -> Vec<f32> {
    b.chunks(2)
        .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect()
}

fn read(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; n];
    gpu.copy_d2h(p, &mut b)?;
    Ok(b)
}

fn i32s(b: &[u8]) -> Vec<i32> {
    b.chunks(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// 2026-10-09: Run the per-head kernel on `inputs` into scratch, compare with the head-batched
/// output already in `inputs.out`, and dump the worst offending rows.
#[allow(clippy::too_many_arguments)]
pub(super) fn against_per_head(
    gpu: &dyn GpuBackend,
    per_head: KernelHandle,
    cfg: &Glm5NextDsaConfig,
    geom: &DsaSelectGeometry,
    paging: &DsaDecodePaging,
    inputs: &DsaDecodeInputs,
    stream: u64,
    dir: &str,
) -> Result<()> {
    let (rows, heads, kvl) = (paging.num_seqs, paging.num_q_heads, cfg.kv_lora_rank);
    let n = rows * heads * kvl * 2;
    let scratch = gpu.alloc(n)?;
    let reference = DsaDecodeInputs {
        out: scratch,
        ..*inputs
    };
    launch_per_head(gpu, per_head, cfg, geom, paging, &reference, stream)?;
    gpu.synchronize(stream)?;
    let hb = bf16s(&read(gpu, inputs.out, n)?);
    let rf = bf16s(&read(gpu, scratch, n)?);
    gpu.free(scratch)?;
    let seq_lens = i32s(&read(gpu, inputs.seq_lens, rows * 4)?);
    let calls = CALLS.fetch_add(1, Ordering::Relaxed) + 1;
    let mut worst = (0f32, 0usize, 0usize);
    for r in 0..rows {
        if seq_lens[r] == 0 {
            continue;
        }
        for h in 0..heads {
            let o = (r * heads + h) * kvl;
            let scale = rf[o..o + kvl].iter().fold(0f32, |m, v| m.max(v.abs()));
            let d = (0..kvl).fold(0f32, |m, i| m.max((hb[o + i] - rf[o + i]).abs()));
            let rel = if scale > 0.0 { d / scale } else { d };
            if rel > worst.0 || rel.is_nan() {
                worst = (rel, r, h);
            }
        }
    }
    let bad = worst.0.is_nan() || worst.0 > DUMP_REL;
    if bad {
        let b = BAD_CALLS.fetch_add(1, Ordering::Relaxed) + 1;
        tracing::warn!(
            "DSA HB check: call {calls} ({rows} rows, {heads} heads): row {} head {} differs by \
             {:.4} of its scale (seq_len {}); {b} bad calls so far",
            worst.1,
            worst.2,
            worst.0,
            seq_lens[worst.1]
        );
        if DUMPS.fetch_add(1, Ordering::Relaxed) < MAX_DUMPS {
            dump(
                gpu,
                cfg,
                geom,
                paging,
                inputs,
                &hb,
                &rf,
                worst,
                seq_lens[worst.1],
                dir,
            )?;
        }
    } else if calls.is_multiple_of(1000) {
        tracing::warn!(
            "DSA HB check: {calls} calls, {} bad; this one worst {:.5}",
            BAD_CALLS.load(Ordering::Relaxed),
            worst.0
        );
    }
    Ok(())
}

/// 2026-10-09: Write one row: `meta.json`, `q.bin` (BF16 `[heads, kv_lora]`), `sel.bin` (i32
/// `[out_width]`), `tokens.bin` (`[out_width, kv_lora]` FP8, the latent of each valid entry,
/// zeros elsewhere), `out_hb.bin` and `out_ref.bin` (BF16 `[heads, kv_lora]`).
#[allow(clippy::too_many_arguments)]
fn dump(
    gpu: &dyn GpuBackend,
    cfg: &Glm5NextDsaConfig,
    geom: &DsaSelectGeometry,
    paging: &DsaDecodePaging,
    inputs: &DsaDecodeInputs,
    hb: &[f32],
    rf: &[f32],
    worst: (f32, usize, usize),
    seq_len: i32,
    dir: &str,
) -> Result<()> {
    let (_, r, h) = worst;
    let (heads, kvl, w) = (paging.num_q_heads, cfg.kv_lora_rank, geom.out_width);
    let case = format!("{dir}/case_{}", DUMPS.load(Ordering::Relaxed));
    std::fs::create_dir_all(&case)?;
    let q = read(gpu, inputs.q.offset(r * heads * kvl * 2), heads * kvl * 2)?;
    let sel = i32s(&read(gpu, inputs.sel_indices.offset(r * w * 4), w * 4)?);
    let bs = paging.block_size;
    let n_blocks = (seq_len.max(0) as usize).div_ceil(bs);
    let bt = i32s(&read(
        gpu,
        inputs
            .block_tables
            .offset(r * paging.max_blocks_per_seq * 4),
        n_blocks.max(1) * 4,
    )?);
    let mut tokens = vec![0u8; w * kvl];
    for (j, &t) in sel.iter().enumerate() {
        if t >= 0 && t < seq_len {
            let t = t as usize;
            let off = bt[t / bs] as u64 * paging.cache_stride_bytes + (t % bs * kvl) as u64;
            let row = read(gpu, inputs.k_cache.offset(off as usize), kvl)?;
            tokens[j * kvl..(j + 1) * kvl].copy_from_slice(&row);
        }
    }
    let to_bf16 = |v: &[f32]| -> Vec<u8> {
        v.iter()
            .flat_map(|x| half::bf16::from_f32(*x).to_le_bytes())
            .collect()
    };
    let o = r * heads * kvl;
    std::fs::write(format!("{case}/q.bin"), &q)?;
    std::fs::write(
        format!("{case}/sel.bin"),
        sel.iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<u8>>(),
    )?;
    std::fs::write(format!("{case}/tokens.bin"), &tokens)?;
    std::fs::write(
        format!("{case}/out_hb.bin"),
        to_bf16(&hb[o..o + heads * kvl]),
    )?;
    std::fs::write(
        format!("{case}/out_ref.bin"),
        to_bf16(&rf[o..o + heads * kvl]),
    )?;
    let meta = serde_json::json!({
        "row": r, "rows": paging.num_seqs, "worst_head": h, "worst_rel": worst.0,
        "heads": heads, "kv_lora": kvl, "out_width": w, "seq_len": seq_len,
        "block_size": bs, "cache_stride_bytes": paging.cache_stride_bytes,
        "max_blocks_per_seq": paging.max_blocks_per_seq,
        "k_scale": inputs.k_scale, "v_scale": inputs.v_scale,
        "block_table": bt,
    });
    std::fs::write(
        format!("{case}/meta.json"),
        serde_json::to_vec_pretty(&meta)?,
    )?;
    tracing::warn!("DSA HB check: dumped row {r} to {case}");
    Ok(())
}
