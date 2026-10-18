// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Bind one GLM-5.3 MLP site for a rank: TP-slice the dense FFN or shared expert,
//! and build the routed experts' pointer tables over GLOBAL expert ids.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - TP: `gate_proj`/`up_proj` (`[inter, hidden]`) are split by row and `down_proj`
//!   (`[hidden, inter]`) by column, over the same `TpSlice` of `inter`, so the dense output is
//!   a partial sum.
//! - EP: an expert is owned whole by one rank. In each pointer table, an id another rank owns
//!   keeps a null `packed` pointer. 2026-10-09: Under the `tp` expert layout every id is local
//!   (each a `moe_intermediate`-wide slice), so no entry is null.
//! - The router weight and bias are loaded unsliced on every rank.

use anyhow::{Result, bail};
use metrale_config::TpSlice;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::Glm5NextMlpConfig;
use super::precision::{GroupPrecision, MlpKernel};
use super::weights::{
    Glm5NextDenseMlpWeights, Glm5NextExpertPtrTable, Glm5NextExpertWeights, Glm5NextMoePtrTables,
    Glm5NextMoeWeights, Nvfp4Proj,
};

/// 2026-09-25: A BF16 or F32 tensor as host `f32`, by layer-relative name.
pub type LoadFn<'a> = &'a dyn Fn(&str) -> Result<Vec<f32>>;
/// 2026-09-25: One routed expert's weights, by GLOBAL id. Experts are not sliced, so the
/// loader's closure (`bind_expert`) can return NVFP4 pointers straight from the weight store.
/// 2026-10-09: Under the `tp` expert layout the closure returns this rank's slice, uploaded by
/// the loader (`glm5_next_load::expert_tp`), `moe_intermediate` wide.
pub type ExpertFn<'a> = &'a dyn Fn(usize) -> Result<Glm5NextExpertWeights>;

/// 2026-09-25: Rows `[start, end)` of a `[rows, row_elems]` row-major tensor, for a
/// column-parallel projection.
fn row_slice(v: &[f32], row_elems: usize, start: usize, end: usize) -> Vec<f32> {
    v[start * row_elems..end * row_elems].to_vec()
}

/// 2026-09-25: Columns `[start, end)` of every row, for the row-parallel `down_proj`.
fn col_slice(v: &[f32], row_elems: usize, start: usize, end: usize) -> Vec<f32> {
    v.chunks(row_elems)
        .flat_map(|r| r[start..end].iter().copied())
        .collect()
}

fn up_bf16(gpu: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = v
        .iter()
        .flat_map(|x| half::bf16::from_f32(*x).to_le_bytes())
        .collect();
    let p = gpu.alloc(b.len().max(1))?;
    gpu.copy_h2d(&b, p)?;
    Ok(p)
}

fn up_f32(gpu: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = gpu.alloc(b.len().max(1))?;
    gpu.copy_h2d(&b, p)?;
    Ok(p)
}

/// 2026-10-08: This rank's `[gate, up, down]` host values of one SwiGLU MLP of width
/// `full_inter` with `hidden` rows: `gate_proj`/`up_proj` (`[inter, hidden]`) keep the rows of
/// `cols` and `down_proj` (`[hidden, inter]`) its columns. `load` returns a projection by its
/// leaf name. Errors when `cols` is empty or runs past `full_inter`, or a tensor has the wrong
/// element count.
pub fn slice_dense_mlp(
    hidden: usize,
    full_inter: usize,
    cols: TpSlice,
    prefix: &str,
    load: LoadFn<'_>,
) -> Result<[Vec<f32>; 3]> {
    if cols.len == 0 || cols.end() > full_inter {
        bail!(
            "GLM MLP {prefix}: columns {:?} do not fit intermediate {full_inter}",
            cols.range()
        );
    }
    let get = |n: &str| -> Result<Vec<f32>> {
        let v = load(&format!("{prefix}.{n}"))?;
        if v.len() != full_inter * hidden {
            bail!(
                "GLM MLP {prefix}.{n}: {} elements, expected {}",
                v.len(),
                full_inter * hidden
            );
        }
        Ok(v)
    };
    let (gate, up, down) = (
        get("gate_proj.weight")?,
        get("up_proj.weight")?,
        get("down_proj.weight")?,
    );
    Ok([
        row_slice(&gate, hidden, cols.start, cols.end()),
        row_slice(&up, hidden, cols.start, cols.end()),
        col_slice(&down, full_inter, cols.start, cols.end()),
    ])
}

/// 2026-09-25: TP-slice and upload one BF16 SwiGLU MLP, a dense layer or a routed layer's
/// shared expert. `full_inter` is the unsliced width. 2026-10-08: The rank keeps columns
/// `cols` (`Glm5NextMlpConfig::dense_slice` or `shared_slice`), which need not be
/// `full_inter / tp` wide; [`slice_dense_mlp`] does the slicing and states the refusals.
pub fn build_dense_mlp(
    gpu: &dyn GpuBackend,
    cfg: &Glm5NextMlpConfig,
    full_inter: usize,
    cols: TpSlice,
    prefix: &str,
    load: LoadFn<'_>,
) -> Result<Glm5NextDenseMlpWeights> {
    let [gate, up, down] = slice_dense_mlp(cfg.hidden, full_inter, cols, prefix, load)?;
    Ok(Glm5NextDenseMlpWeights {
        gate_proj: up_bf16(gpu, &gate)?,
        up_proj: up_bf16(gpu, &up)?,
        down_proj: up_bf16(gpu, &down)?,
    })
}

/// 2026-09-25: One projection's pointer table over all `num_experts` GLOBAL ids. An id another
/// rank owns keeps a null `packed` pointer, which the expert kernels skip.
fn build_expert_ptr_table(
    gpu: &dyn GpuBackend,
    cfg: &Glm5NextMlpConfig,
    experts: &[Glm5NextExpertWeights],
    proj: impl Fn(&Glm5NextExpertWeights) -> Nvfp4Proj,
) -> Result<Glm5NextExpertPtrTable> {
    let n = cfg.num_experts;
    let mut packed = vec![0u8; n * 8];
    let mut scale = vec![0u8; n * 8];
    let mut scale2 = vec![0u8; n * 4];
    for id in 0..n {
        let Some(local) = cfg.local_slot(id) else {
            continue;
        };
        let p = proj(&experts[local]);
        packed[id * 8..id * 8 + 8].copy_from_slice(&p.packed.0.to_le_bytes());
        scale[id * 8..id * 8 + 8].copy_from_slice(&p.scale.0.to_le_bytes());
        scale2[id * 4..id * 4 + 4].copy_from_slice(&p.scale_2.to_le_bytes());
    }
    let packed_ptrs = gpu.alloc(packed.len())?;
    gpu.copy_h2d(&packed, packed_ptrs)?;
    let scale_ptrs = gpu.alloc(scale.len())?;
    gpu.copy_h2d(&scale, scale_ptrs)?;
    let scale2_vals = gpu.alloc(scale2.len())?;
    gpu.copy_h2d(&scale2, scale2_vals)?;
    Ok(Glm5NextExpertPtrTable {
        packed_ptrs,
        scale_ptrs,
        scale2_vals,
    })
}

/// 2026-10-08: The routed experts' precision plan, given whether every bound projection carries
/// a static activation scale (`precision::GroupPrecision::resolve`).
pub type PrecisionFn<'a> = &'a dyn Fn(bool) -> Result<GroupPrecision>;

/// 2026-09-25: Bind one routed MoE site for this rank: the router and bias unsliced, the shared
/// expert TP-sliced (`cfg.shared_slice()`), and the `local_experts` routed experts this EP rank
/// owns. 2026-10-08: Then the experts' precision plan, and their uniform activation scales when
/// the plan reaches W4A4 at some row count up to `max_rows`.
pub fn build_moe(
    gpu: &dyn GpuBackend,
    cfg: &Glm5NextMlpConfig,
    full_shared_inter: usize,
    load: LoadFn<'_>,
    expert: ExpertFn<'_>,
    precision: PrecisionFn<'_>,
    max_rows: usize,
) -> Result<Glm5NextMoeWeights> {
    // 2026-09-25: The router weight and bias are loaded whole on every rank, so every rank
    // selects the same experts.
    let router = load("mlp.gate.weight")?;
    if router.len() != cfg.num_experts * cfg.hidden {
        bail!(
            "GLM MoE mlp.gate.weight: {} elements, expected {} ({} experts x {} hidden)",
            router.len(),
            cfg.num_experts * cfg.hidden,
            cfg.num_experts,
            cfg.hidden
        );
    }
    let bias = load("mlp.gate.e_score_correction_bias")?;
    if bias.len() != cfg.num_experts {
        bail!(
            "GLM MoE e_score_correction_bias: {} entries, expected {}",
            bias.len(),
            cfg.num_experts
        );
    }

    let shared = build_dense_mlp(
        gpu,
        cfg,
        full_shared_inter,
        cfg.shared_slice(),
        "mlp.shared_experts",
        load,
    )?;

    // 2026-09-25: Ascending GLOBAL id: slot `i` holds id `local_expert_range().start + i`, the
    // inverse of `Glm5NextMlpConfig::local_slot`.
    let mut experts = Vec::with_capacity(cfg.local_experts);
    for id in cfg.local_expert_range() {
        experts.push(expert(id)?);
    }

    let precision = precision(super::build_w4a4::experts_have_scales(&experts))?;
    let act_scales = if precision.reaches(MlpKernel::W4a4Static, max_rows) {
        Some(super::build_w4a4::expert_act_scales(&experts)?)
    } else {
        None
    };

    let ptrs = Glm5NextMoePtrTables {
        gate: build_expert_ptr_table(gpu, cfg, &experts, |e| e.gate_proj)?,
        up: build_expert_ptr_table(gpu, cfg, &experts, |e| e.up_proj)?,
        down: build_expert_ptr_table(gpu, cfg, &experts, |e| e.down_proj)?,
    };

    Ok(Glm5NextMoeWeights {
        router: up_bf16(gpu, &router)?,
        // 2026-09-25: FP32: `glm5next_router_topk` reads `bias` as `const float*`.
        router_bias: up_f32(gpu, &bias)?,
        shared,
        experts,
        ptrs,
        precision,
        act_scales,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-09-25: `row_slice` takes rows and `col_slice` takes columns, on tensors whose values
    /// encode their index.
    #[test]
    fn dense_slicing_takes_rows_for_gate_and_columns_for_down() {
        let gate: Vec<f32> = (0..4)
            .flat_map(|r| (0..3).map(move |c| (r * 10 + c) as f32))
            .collect();
        assert_eq!(
            row_slice(&gate, 3, 2, 4),
            vec![20., 21., 22., 30., 31., 32.]
        );

        let down: Vec<f32> = (0..3)
            .flat_map(|r| (0..4).map(move |c| (r * 10 + c) as f32))
            .collect();
        assert_eq!(col_slice(&down, 4, 2, 4), vec![2., 3., 12., 13., 22., 23.]);
        // 2026-09-25: On a square tensor the two slices have the same length and different
        // values, so only the values tell them apart.
        let sq: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let by_row = row_slice(&sq, 4, 2, 4);
        let by_col = col_slice(&sq, 4, 2, 4);
        assert_eq!(by_row.len(), by_col.len());
        assert_ne!(by_row, by_col);
    }

    /// 2026-10-08: The 2048-wide shared expert at TP=3 (688/680/680 columns): every rank's
    /// `gate`/`up` rows and `down` columns rejoin, in rank order, to the full tensors, and each
    /// rank's three tensors cover the same intermediate indices.
    #[test]
    fn three_uneven_ranks_partition_the_dense_mlp() {
        let (hidden, inter) = (4usize, 2048usize);
        // 2026-10-08: `[inter, hidden]` values `i * hidden + h` and `[hidden, inter]` values
        // `-(h * inter + i)`, so every element names its own position.
        let gate: Vec<f32> = (0..inter * hidden).map(|x| x as f32).collect();
        let down: Vec<f32> = (0..hidden * inter).map(|x| -(x as f32)).collect();
        let load = |n: &str| -> Result<Vec<f32>> {
            Ok(if n.ends_with("down_proj.weight") {
                down.clone()
            } else {
                gate.clone()
            })
        };
        let ranks: Vec<[Vec<f32>; 3]> = (0..3)
            .map(|r| {
                let cols = metrale_config::tp_split(inter, 3, r, 8).unwrap();
                slice_dense_mlp(hidden, inter, cols, "mlp.shared_experts", &load).unwrap()
            })
            .collect();
        assert_eq!(
            ranks
                .iter()
                .map(|t| t[0].len() / hidden)
                .collect::<Vec<_>>(),
            vec![688, 680, 680]
        );
        let rows: Vec<f32> = ranks.iter().flat_map(|t| t[0].clone()).collect();
        assert_eq!(rows, gate, "gate rows rejoin in rank order");
        for h in 0..hidden {
            let row: Vec<f32> = ranks
                .iter()
                .flat_map(|t| t[2].chunks(t[2].len() / hidden).nth(h).unwrap().to_vec())
                .collect();
            assert_eq!(
                row,
                down[h * inter..(h + 1) * inter],
                "down row {h} rejoins"
            );
        }
        // 2026-10-08: Rank 1's first gate row and first down column are intermediate index
        // 688: the gate-up output and the down input meet on the same columns.
        assert_eq!(ranks[1][0][0], (688 * hidden) as f32);
        assert_eq!(ranks[1][2][0], -688.0);
    }

    /// 2026-10-08: A column range past the width, an empty one, or a tensor of the wrong size is
    /// refused.
    #[test]
    fn bad_columns_and_sizes_are_refused() {
        let ok = |_: &str| -> Result<Vec<f32>> { Ok(vec![0.0; 16 * 2]) };
        let short = |_: &str| -> Result<Vec<f32>> { Ok(vec![0.0; 16 * 2 - 1]) };
        let s = |start, len| TpSlice { start, len };
        assert!(slice_dense_mlp(2, 16, s(8, 8), "m", &ok).is_ok());
        for bad in [s(8, 9), s(0, 0)] {
            let e = slice_dense_mlp(2, 16, bad, "m", &ok).unwrap_err();
            assert!(e.to_string().contains("do not fit"), "{e}");
        }
        let e = slice_dense_mlp(2, 16, s(0, 8), "m", &short).unwrap_err();
        assert!(e.to_string().contains("elements, expected 32"), "{e}");
    }
}
