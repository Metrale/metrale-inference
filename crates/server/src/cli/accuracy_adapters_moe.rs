// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Launch adapters of the grouped routed-expert kernels (`moe_grouped` cases): the
//! engine's own sort (`moe_fp8_grouped_sort`) groups the case's routing by expert, the routed
//! experts' weights go into per-expert pointer tables as the loader builds them, the projection
//! runs through the engine launcher, and its rows, written by sorted position, come back in the
//! case's slot order through the sort's `token_to_perm` (the blend's mapping). A down input is
//! placed at its sorted positions the same way. The `_lean` launchers first run the production
//! repack (`nvfp4_tc_lean_repack`) over the uploaded tables, as the loader does.
//!
//! Owner: server CLI.
//! Invariants:
//! - The shared-expert block rows every grouped launch runs read the first routed expert's
//!   weights into their own buffers; those are read back only for their guard bands.
//! - Every buffer a kernel writes is a sentinel-filled output with guard bands; the slot
//!   permutation and the BF16 hi + lo decode are `metrale_accuracy::refs::moe_grouped`'s.

use anyhow::{Result, ensure};
use metrale_accuracy::case::{Case, Enc, Tensor};
use metrale_accuracy::refs::moe_grouped as mg;
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::{Fp8Weight, QuantizedWeight, WeightQuantFormat};

use super::accuracy_adapters::{Adapter, handle, not_runnable};
use super::accuracy_gpu::Dev;

/// 2026-10-09: The adapter of each grouped-expert launcher.
pub(crate) const ADAPTERS: &[(&str, Adapter)] = &[
    (
        "moe_nvfp4_grouped_tc::moe_expert_gate_up_act_nvfp4_grouped_tc",
        nvfp4_gate_up,
    ),
    (
        "moe_nvfp4_grouped_tc::moe_expert_gate_up_act_nvfp4_grouped_tc_lean",
        nvfp4_gate_up,
    ),
    (
        "moe_nvfp4_grouped_tc::moe_expert_down_act_nvfp4_grouped_tc",
        nvfp4_down,
    ),
    (
        "moe_nvfp4_grouped_tc::moe_expert_down_act_nvfp4_grouped_tc_lean",
        nvfp4_down,
    ),
    (
        "moe_shared_expert_fused_fp8_grouped::moe_expert_gate_up_act_fp8_grouped",
        fp8_gate_up,
    ),
    (
        "moe_shared_expert_fused_fp8_grouped::moe_expert_down_act_fp8_grouped",
        fp8_down,
    ),
];

const SORT: &str = "moe_fp8_grouped_sort::moe_fp8_grouped_sort";
const LEAN_REPACK: &str = "moe_nvfp4_grouped_tc::nvfp4_tc_lean_repack";

fn need(ok: bool, why: impl FnOnce() -> String) -> Result<()> {
    if ok { Ok(()) } else { Err(not_runnable(why())) }
}

fn tensor<'a>(case: &'a Case, name: &str) -> Result<&'a Tensor> {
    case.tensor(name).map_err(not_runnable)
}

fn scalar(case: &Case, name: &str) -> Result<f64> {
    case.scalar(name).map_err(not_runnable)
}

/// 2026-10-09: Upload raw bytes (pointer tables, staged rows) through the device's tracked
/// allocations.
fn raw(dev: &mut Dev<'_>, bytes: Vec<u8>) -> Result<DevicePtr> {
    let n = bytes.len();
    dev.upload(&Tensor {
        enc: Enc::Ue8m0,
        dims: vec![n],
        bytes: std::sync::Arc::new(bytes),
    })
}

/// 2026-10-09: The routing's sizes and the engine sort's outputs, with `token_to_perm` read.
struct Sorted {
    tokens: usize,
    top_k: usize,
    experts: usize,
    out: ops::Fp8GroupedSortOut,
    perm: Vec<i32>,
    routed: Vec<usize>,
}

impl Sorted {
    fn slots(&self) -> usize {
        self.tokens * self.top_k
    }

    fn cap(&self) -> u32 {
        ops::fp8_grouped_active_cap(self.tokens as u32, self.top_k as u32, self.experts as u32)
    }
}

fn sort(dev: &mut Dev<'_>, case: &Case) -> Result<Sorted> {
    let ids = tensor(case, "ids")?;
    let (tokens, top_k) = (ids.dims[0], ids.dims[1]);
    let experts = scalar(case, "experts")? as usize;
    let te = tokens * top_k;
    let id_ptr = dev.upload(ids)?;
    let out = ops::Fp8GroupedSortOut {
        sorted_token_ids: dev.output(te * 4)?,
        sorted_expert_ids: dev.output(te * 4)?,
        expert_offsets: dev.output((experts + 1) * 4)?,
        token_to_perm: dev.output(te * 4)?,
        active_experts: dev.output(experts * 4)?,
        active_count: dev.output(4)?,
    };
    let kernel = handle(dev, SORT)?;
    ops::moe_fp8_grouped_sort(
        dev.gpu,
        kernel,
        ops::Fp8GroupedSortOut { ..out },
        id_ptr,
        te as u32,
        experts as u32,
        top_k as u32,
        dev.stream,
    )?;
    let perm: Vec<i32> = dev
        .read(out.token_to_perm, te * 4)?
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let (ids, _) = mg::routing(case).map_err(not_runnable)?;
    let routed = mg::slots_by_expert(&ids).into_keys().collect();
    Ok(Sorted {
        tokens,
        top_k,
        experts,
        out,
        perm,
        routed,
    })
}

/// 2026-10-09: One projection's per-expert tables (weights, block scales, NVFP4 global scales),
/// null for experts no slot routes to, and the first routed expert's buffers for the shared
/// expert's block rows.
struct Tables {
    w: DevicePtr,
    s: DevicePtr,
    s2: DevicePtr,
    first: (DevicePtr, DevicePtr, f32),
}

fn tables(dev: &mut Dev<'_>, case: &Case, s: &Sorted, p: &str, lean: bool) -> Result<Tables> {
    let (mut w, mut sc) = (vec![0u64; s.experts], vec![0u64; s.experts]);
    let mut s2 = vec![0f32; s.experts];
    let mut first = None;
    let repack = if lean {
        Some((handle(dev, LEAN_REPACK)?, raw(dev, vec![0u8; 4])?))
    } else {
        None
    };
    for &e in &s.routed {
        let name = mg::weight_name(p, e);
        let wt = tensor(case, &name)?;
        let (wp, sp) = (
            dev.upload(wt)?,
            dev.upload(tensor(case, &format!("{name}_block"))?)?,
        );
        let g = scalar(case, &format!("{name}_global"))? as f32;
        if let Some((kernel, bad)) = repack {
            let (n, k) = (wt.dims[0] as u32, wt.dims[1] as u32);
            need(ops::nvfp4_lean_repack_shape_ok(n, k), || {
                format!("the lean repack does not admit [{n}, {k}]")
            })?;
            ops::nvfp4_tc_lean_repack(dev.gpu, kernel, wp, sp, n, k, bad, dev.stream)?;
        }
        w[e] = wp.0;
        sc[e] = sp.0;
        s2[e] = g;
        first.get_or_insert((wp, sp, g));
    }
    if let Some((_, bad)) = repack {
        ensure!(
            dev.read(bad, 4)?.iter().all(|&b| b == 0),
            "nvfp4_tc_lean_repack flagged a negative or NaN scale"
        );
    }
    let le = |v: &[u64]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    Ok(Tables {
        w: raw(dev, le(&w))?,
        s: raw(dev, le(&sc))?,
        s2: raw(dev, s2.iter().flat_map(|x| x.to_le_bytes()).collect())?,
        first: first.ok_or_else(|| not_runnable("the case routes no expert".into()))?,
    })
}

fn nvfp4_tables(t: &Tables) -> ops::Nvfp4ExpertTables {
    ops::Nvfp4ExpertTables {
        packed_ptrs: t.w,
        scale_ptrs: t.s,
        scale2_vals: t.s2,
    }
}

fn nvfp4_shared(t: &Tables) -> QuantizedWeight {
    let mut q = QuantizedWeight::null();
    (q.weight, q.weight_scale, q.weight_scale_2) = t.first;
    q
}

fn fp8_shared(t: &Tables, n: usize, k: usize) -> Fp8Weight {
    Fp8Weight {
        weight: t.first.0,
        row_scale: t.first.1,
        n: n as u32,
        k: k as u32,
        scale_format: WeightQuantFormat::Fp8BlockScaled,
    }
}

/// 2026-10-09: The case's output width, its projection input width, and the weight rows.
fn widths(case: &Case, p: &str, s: &Sorted) -> Result<(usize, usize)> {
    let e = *s
        .routed
        .first()
        .ok_or_else(|| not_runnable("the case routes no expert".into()))?;
    let w = tensor(case, &mg::weight_name(p, e))?;
    need(w.dims[0] == case.out.0[1], || {
        format!(
            "weight rows {} for {} output columns",
            w.dims[0], case.out.0[1]
        )
    })?;
    Ok((w.dims[0], w.dims[1]))
}

/// 2026-10-09: Slot-ordered input rows at their sorted positions.
fn by_position(dev: &mut Dev<'_>, x: &Tensor, s: &Sorted) -> Result<DevicePtr> {
    let row = x.bytes.len() / s.slots();
    let staged = mg::rows_by_position(&x.bytes, &s.perm, row).map_err(not_runnable)?;
    raw(dev, staged)
}

fn nvfp4_gate_up(dev: &mut Dev<'_>, case: &Case, kernel: KernelHandle) -> Result<Vec<u8>> {
    let lean = case.launcher.ends_with("_lean");
    let s = sort(dev, case)?;
    let (n, k) = widths(case, "gate", &s)?;
    let geo = ops::NVFP4_GROUPED_GATE_UP_TC;
    need(
        ops::nvfp4_grouped_tc_shape_ok(n as u32, k as u32, geo),
        || format!("the tensor-core gate+up does not admit n={n} k={k}"),
    )?;
    let (gate, up) = (
        tables(dev, case, &s, "gate", lean)?,
        tables(dev, case, &s, "up", lean)?,
    );
    let x = dev.upload(tensor(case, "x")?)?;
    // 2026-10-09: The act buffers are FP32-sized rows holding N hi then N lo BF16 values.
    let act = dev.output(s.slots() * n * 4)?;
    let sh_act = dev.output(s.tokens * n * 4)?;
    ops::moe_expert_gate_up_act_nvfp4_grouped(
        dev.gpu,
        kernel,
        geo,
        x,
        nvfp4_tables(&gate),
        nvfp4_tables(&up),
        act,
        s.out.expert_offsets,
        s.out.sorted_token_ids,
        s.out.active_experts,
        s.out.active_count,
        &nvfp4_shared(&gate),
        &nvfp4_shared(&up),
        sh_act,
        n as u32,
        k as u32,
        s.cap(),
        s.tokens as u32,
        dev.stream,
    )?;
    let by_pos = dev.read(act, s.slots() * n * 4)?;
    dev.read(sh_act, s.tokens * n * 4)?;
    mg::pair_rows_by_slot(&by_pos, &s.perm, n).map_err(anyhow::Error::msg)
}

fn nvfp4_down(dev: &mut Dev<'_>, case: &Case, kernel: KernelHandle) -> Result<Vec<u8>> {
    let lean = case.launcher.ends_with("_lean");
    let s = sort(dev, case)?;
    let (n, k) = widths(case, "down", &s)?;
    let geo = ops::NVFP4_GROUPED_DOWN_TC;
    need(
        ops::nvfp4_grouped_tc_shape_ok(n as u32, k as u32, geo),
        || format!("the tensor-core down does not admit n={n} k={k}"),
    )?;
    let x = tensor(case, "x")?;
    need(x.enc == Enc::Bf16 && x.dims == [s.slots(), 2 * k], || {
        format!(
            "a hi + lo input [{}, {}] bf16, not {:?}",
            s.slots(),
            2 * k,
            x.dims
        )
    })?;
    let down = tables(dev, case, &s, "down", lean)?;
    let act = by_position(dev, x, &s)?;
    let sh_act = raw(dev, vec![0u8; s.tokens * 2 * k * 2])?;
    let out = dev.output(s.slots() * n * 2)?;
    let sh_out = dev.output(s.tokens * n * 2)?;
    ops::moe_expert_down_act_nvfp4_grouped(
        dev.gpu,
        kernel,
        geo,
        act,
        nvfp4_tables(&down),
        out,
        s.out.expert_offsets,
        s.out.active_experts,
        s.out.active_count,
        sh_act,
        &nvfp4_shared(&down),
        sh_out,
        n as u32,
        k as u32,
        s.cap(),
        s.tokens as u32,
        dev.stream,
    )?;
    let by_pos = dev.read(out, s.slots() * n * 2)?;
    dev.read(sh_out, s.tokens * n * 2)?;
    mg::rows_by_slot(&by_pos, &s.perm, n * 2).map_err(anyhow::Error::msg)
}

fn fp8_gate_up(dev: &mut Dev<'_>, case: &Case, kernel: KernelHandle) -> Result<Vec<u8>> {
    let s = sort(dev, case)?;
    let (n, k) = widths(case, "gate", &s)?;
    let (gate, up) = (
        tables(dev, case, &s, "gate", false)?,
        tables(dev, case, &s, "up", false)?,
    );
    let x = dev.upload(tensor(case, "x")?)?;
    let act = dev.output(s.slots() * n * 4)?;
    let sh_act = dev.output(s.tokens * n * 4)?;
    ops::moe_expert_gate_up_act_fp8_grouped(
        dev.gpu,
        kernel,
        ops::FP8_GROUPED_GATE_UP_SCALAR,
        x,
        gate.w,
        gate.s,
        up.w,
        up.s,
        act,
        s.out.expert_offsets,
        s.out.sorted_token_ids,
        s.out.active_experts,
        s.out.active_count,
        &fp8_shared(&gate, n, k),
        &fp8_shared(&up, n, k),
        sh_act,
        n as u32,
        k as u32,
        s.cap(),
        s.tokens as u32,
        dev.stream,
    )?;
    let by_pos = dev.read(act, s.slots() * n * 4)?;
    dev.read(sh_act, s.tokens * n * 4)?;
    mg::rows_by_slot(&by_pos, &s.perm, n * 4).map_err(anyhow::Error::msg)
}

fn fp8_down(dev: &mut Dev<'_>, case: &Case, kernel: KernelHandle) -> Result<Vec<u8>> {
    let s = sort(dev, case)?;
    let (n, k) = widths(case, "down", &s)?;
    let x = tensor(case, "x")?;
    need(x.enc == Enc::F32 && x.dims == [s.slots(), k], || {
        format!("an f32 input [{}, {k}], not {:?}", s.slots(), x.dims)
    })?;
    let down = tables(dev, case, &s, "down", false)?;
    let act = by_position(dev, x, &s)?;
    let sh_act = raw(dev, vec![0u8; s.tokens * k * 4])?;
    let out = dev.output(s.slots() * n * 2)?;
    let sh_out = dev.output(s.tokens * n * 2)?;
    ops::moe_expert_down_act_fp8_grouped(
        dev.gpu,
        kernel,
        ops::FP8_GROUPED_DOWN_SCALAR,
        act,
        down.w,
        down.s,
        out,
        s.out.expert_offsets,
        s.out.active_experts,
        s.out.active_count,
        sh_act,
        &fp8_shared(&down, n, k),
        sh_out,
        n as u32,
        k as u32,
        s.cap(),
        s.tokens as u32,
        dev.stream,
    )?;
    let by_pos = dev.read(out, s.slots() * n * 2)?;
    dev.read(sh_out, s.tokens * n * 2)?;
    mg::rows_by_slot(&by_pos, &s.perm, n * 2).map_err(anyhow::Error::msg)
}
