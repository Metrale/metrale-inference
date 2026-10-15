// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: The host side of the unabsorbed prefill: this rank's `q_b_proj` rows and
//! `o_proj` columns, and `kv_b_proj`'s two halves laid out per head for
//! `glm5next_mla_head_bmm`, plus the f64 reference of that kernel.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - No arithmetic: every weight is a checkpoint value moved to a new place, so the BF16 upload
//!   holds the checkpoint's bits (the absorbed path rounds products instead).
//! - `w_uk_t[h][c][r] = kv_b[h*(nope+vd) + r][c]` and `w_uv[h][r][c] = kv_b[h*(nope+vd) + nope +
//!   r][c]`, so `head_bmm(q_b x, w_uk_t)` is `q_absorb x` and `o_proj head_bmm(a, w_uv)` is
//!   `o_absorb a` in exact arithmetic (`build::absorb_q`, `build::absorb_o`).

use anyhow::{Result, bail, ensure};

use super::super::Glm5NextDsaConfig;
use super::super::build::{LoadFn, row_slice, shard_host};
use super::super::tp::DsaTpPlan;

/// 2026-10-15: One rank's unabsorbed weights, f32 on the host.
pub struct UnabsorbHost {
    /// 2026-10-15: `[local_heads * qk_nope_head_dim, q_lora_rank]`, this rank's heads.
    pub q_b: Vec<f32>,
    /// 2026-10-15: `[hidden, local_heads * v_head_dim]`, this rank's columns (row-parallel).
    pub o_proj: Vec<f32>,
    /// 2026-10-15: `[local_heads, kv_lora_rank, qk_nope_head_dim]` ([`split_w_uk_t`]).
    pub w_uk_t: Vec<f32>,
    /// 2026-10-15: `[local_heads, v_head_dim, kv_lora_rank]` ([`split_w_uv`]).
    pub w_uv: Vec<f32>,
}

fn check_kv_b(cfg: &Glm5NextDsaConfig, kv_b_local: &[f32], heads: usize) -> Result<()> {
    if cfg.qk_rope_head_dim != 0 {
        bail!(
            "unabsorb: NoPE only; qk_rope_head_dim is {}",
            cfg.qk_rope_head_dim
        );
    }
    let want = heads * (cfg.qk_nope_head_dim + cfg.v_head_dim) * cfg.kv_lora_rank;
    ensure!(
        kv_b_local.len() == want,
        "unabsorb: kv_b_proj slice has {} elems, expected {want}",
        kv_b_local.len()
    );
    Ok(())
}

/// 2026-10-15: `kv_b_proj`'s K half per head, transposed:
/// `w_uk_t[h][c][r] = kv_b_local[(h*(nope+vd) + r)*kvl + c]`, `[heads, kvl, nope]`.
pub fn split_w_uk_t(
    cfg: &Glm5NextDsaConfig,
    kv_b_local: &[f32],
    heads: usize,
) -> Result<Vec<f32>> {
    check_kv_b(cfg, kv_b_local, heads)?;
    let (nope, vd, kvl) = (cfg.qk_nope_head_dim, cfg.v_head_dim, cfg.kv_lora_rank);
    let mut out = vec![0f32; heads * kvl * nope];
    for h in 0..heads {
        for r in 0..nope {
            let src = &kv_b_local[(h * (nope + vd) + r) * kvl..][..kvl];
            for (c, &v) in src.iter().enumerate() {
                out[(h * kvl + c) * nope + r] = v;
            }
        }
    }
    Ok(out)
}

/// 2026-10-15: `kv_b_proj`'s V half per head, as stored:
/// `w_uv[h][r][c] = kv_b_local[(h*(nope+vd) + nope + r)*kvl + c]`, `[heads, vd, kvl]`.
pub fn split_w_uv(cfg: &Glm5NextDsaConfig, kv_b_local: &[f32], heads: usize) -> Result<Vec<f32>> {
    check_kv_b(cfg, kv_b_local, heads)?;
    let (nope, vd, kvl) = (cfg.qk_nope_head_dim, cfg.v_head_dim, cfg.kv_lora_rank);
    let mut out = Vec::with_capacity(heads * vd * kvl);
    for h in 0..heads {
        let base = (h * (nope + vd) + nope) * kvl;
        out.extend_from_slice(&kv_b_local[base..base + vd * kvl]);
    }
    Ok(out)
}

/// 2026-10-15: This rank's [`UnabsorbHost`]: the plan's `q_b_proj` rows and `o_proj` columns,
/// and its heads' `kv_b_proj` rows split per head. `load` returns a checkpoint tensor by
/// layer-relative name, as for `build::build_dsa_weights`.
pub fn unabsorb_for_rank(
    cfg: &Glm5NextDsaConfig,
    plan: &DsaTpPlan,
    load: LoadFn<'_>,
) -> Result<UnabsorbHost> {
    let get = |n: &str| -> Result<Vec<f32>> { load(&format!("self_attn.{n}")) };
    let plan_of = |n: &str| {
        plan.get(n)
            .ok_or_else(|| anyhow::anyhow!("no shard plan for {n}"))
    };
    let q_b = shard_host(plan_of("q_b_proj")?, &get("q_b_proj.weight")?);
    let o_proj = shard_host(plan_of("o_proj")?, &get("o_proj.weight")?);
    let lh = plan.local_heads;
    ensure!(
        q_b.len() == lh * cfg.qk_head_dim() * cfg.q_lora_rank,
        "unabsorb: q_b_proj shard has {} elems, expected {}",
        q_b.len(),
        lh * cfg.qk_head_dim() * cfg.q_lora_rank
    );
    ensure!(
        o_proj.len() == cfg.hidden * lh * cfg.v_head_dim,
        "unabsorb: o_proj shard has {} elems, expected {}",
        o_proj.len(),
        cfg.hidden * lh * cfg.v_head_dim
    );
    let rows = cfg.qk_nope_head_dim + cfg.v_head_dim;
    let kv_b_local = row_slice(
        &get("kv_b_proj.weight")?,
        cfg.kv_lora_rank,
        plan.head_start * rows,
        (plan.head_start + lh) * rows,
    );
    Ok(UnabsorbHost {
        q_b,
        o_proj,
        w_uk_t: split_w_uk_t(cfg, &kv_b_local, lh)?,
        w_uv: split_w_uv(cfg, &kv_b_local, lh)?,
    })
}

/// 2026-10-15: `glm5next_mla_head_bmm` in f64:
/// `C[i, h*n + j] = Σ_k A[i, h*kk + k] · B[h][j][k]` for `m` rows and `heads` heads.
pub fn head_bmm_ref(
    a: &[f64],
    b: &[f64],
    m: usize,
    heads: usize,
    kk: usize,
    n: usize,
) -> Vec<f64> {
    let mut c = vec![0f64; m * heads * n];
    for i in 0..m {
        for h in 0..heads {
            let a_row = &a[i * heads * kk + h * kk..][..kk];
            for j in 0..n {
                let b_row = &b[(h * n + j) * kk..][..kk];
                c[i * heads * n + h * n + j] = a_row.iter().zip(b_row).map(|(x, y)| x * y).sum();
            }
        }
    }
    c
}

/// 2026-10-15: `C[i, j] = Σ_k A[i, k] · W[j, k]` in f64 (a dense projection, `W` `[n, kk]`).
pub fn dense_ref(a: &[f64], w: &[f64], m: usize, n: usize, kk: usize) -> Vec<f64> {
    let mut c = vec![0f64; m * n];
    for i in 0..m {
        let a_row = &a[i * kk..][..kk];
        for j in 0..n {
            c[i * n + j] = a_row.iter().zip(&w[j * kk..][..kk]).map(|(x, y)| x * y).sum();
        }
    }
    c
}
