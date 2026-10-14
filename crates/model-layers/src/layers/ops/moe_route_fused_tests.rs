// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: The Hopper fused router (`moe_route_fused_hopper`) writes the bytes of the three
//! launches it replaces (`dense_gemv_batchm_split` -> `moe_topk_softmax_rows_warp` ->
//! `moe_fp8_grouped_sort`) on the Flash-Next router (512 experts, hidden 2560, top 10) at
//! {1, 2, 3, 8, 13, 16} rows, normalized or not: logits, picks, weights, expert offsets, the
//! active list and count, the sorted expert ids, each expert's run of token ids as a multiset
//! (the order inside a run is the sort's atomics in either form) and a `token_to_perm` that
//! inverts the sort. The fixture has realistic magnitudes (logits of a few units, so BF16
//! rounding ties many of them), router rows copied onto other experts (exact ties the lower
//! index must win), a zero row (every logit +0), a NaN row (nothing pickable: the warp
//! kernel's pseudo pick of expert 0), a row at the BF16 maximum (huge and non-finite logits)
//! and two equal rows (experts repeated across rows).
//! Also: the sort outputs inside the logits buffer as the decode lays them out, the
//! completion counter back at zero after each launch, a row's bytes alone equal its bytes in
//! 16 rows, and both forms' time per launch with the router weight cold in L2.
//!
//! GPU test: run with
//! ```text
//! cargo test -p metrale-model-layers --release moe_route_fused_tests -- --ignored --nocapture --test-threads 1
//! ```
//! on a Hopper host whose build has the `qwen3.8-flash-next/nvfp4` kernel set
//! (`METRALE_TARGET_HW=hopper`).

use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::nvfp4_moe_tc_split_tests::{EXPERTS, HIDDEN, TOP_K, download, time_us, upload};
use super::{
    DENSE_GEMV_BATCHM_MAX_M, Fp8GroupedSortOut, MOE_ROUTE_FUSED_MAX_ROWS, dense_gemv_batchm_split,
    moe_fp8_grouped_sort, moe_route_fused, moe_route_fused_shape_ok, moe_topk_softmax_rows_warp,
};
use crate::weight_map::DenseWeight;

const SENTINEL: u8 = 0xAB;
const WIDTHS: [u32; 6] = [1, 2, 3, 8, 13, 16];
const TIMED: [u32; 5] = [1, 2, 4, 8, 16];
const MAX_ROWS: u32 = MOE_ROUTE_FUSED_MAX_ROWS;
/// 2026-10-14: Input row of zeros: every logit +0, so the picks are experts 0..TOP_K.
const ZERO_ROW: u32 = 1;
/// 2026-10-14: Input row of NaN: no logit can be picked, so every pick is the warp kernel's
/// pseudo pick of expert 0.
const NAN_ROW: u32 = 2;
/// 2026-10-14: Input row at the BF16 maximum: huge and non-finite logits.
const HUGE_ROW: u32 = 3;
/// 2026-10-14: Two equal input rows: the same picks, so their experts repeat across rows.
const TWIN_ROWS: (u32, u32) = (4, 5);
/// 2026-10-14: (kept, copy): the router row of `kept` copied onto `copy`, an exact tie.
const TWIN_EXPERTS: [(u32, u32); 3] = [(0, 511), (7, 300), (100, 101)];
/// 2026-10-14: Router weight copies the timing rotates through: 24 x 2.6 MB exceeds the L2.
const COLD_COPIES: usize = 24;

struct Kernels {
    gemv: KernelHandle,
    topk: KernelHandle,
    sort: KernelHandle,
    fused: KernelHandle,
}

fn kernels(g: &dyn GpuBackend) -> Kernels {
    let k = |module: &str, name: &str| g.kernel(module, name).unwrap();
    Kernels {
        gemv: k("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm"),
        topk: k("moe_topk", "moe_topk_softmax_rows_warp"),
        sort: k("moe_fp8_grouped_sort", "moe_fp8_grouped_sort"),
        fused: k("moe_route_fused_hopper", "moe_route_fused_hopper"),
    }
}

/// 2026-10-14: `n` BF16 values `scale * u`, u uniform over 2001 levels in [-1, 1].
fn bf16_values(n: usize, seed: u32, scale: f32) -> Vec<u16> {
    let mut x = seed | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            let v = ((x % 2001) as f32 / 1000.0 - 1.0) * scale;
            (v.to_bits() >> 16) as u16
        })
        .collect()
}

fn le16(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// 2026-10-14: (input `[MAX_ROWS, HIDDEN]`, router weight `[EXPERTS, HIDDEN]`) as uploaded bytes.
fn fixture_bytes() -> (Vec<u8>, Vec<u8>) {
    let h = HIDDEN as usize;
    let mut input = bf16_values(MAX_ROWS as usize * h, 11, 2.0);
    let row = |r: u32| r as usize * h..(r as usize + 1) * h;
    input[row(ZERO_ROW)].fill(0);
    input[row(NAN_ROW)].fill(0x7FC0);
    input[row(HUGE_ROW)].fill(0x7F7F);
    input.copy_within(row(TWIN_ROWS.0), TWIN_ROWS.1 as usize * h);
    let mut gate = bf16_values(EXPERTS as usize * h, 29, 0.04);
    for (kept, copy) in TWIN_EXPERTS {
        gate.copy_within(row(kept), copy as usize * h);
    }
    (le16(&input), le16(&gate))
}

fn sentinel(g: &dyn GpuBackend, n: usize) -> DevicePtr {
    let p = g.alloc(n).unwrap();
    g.memset(p, SENTINEL, n).unwrap();
    p
}

/// 2026-10-14: Bytes of the sort outputs for `te` slots, laid out as the grouped decode lays
/// them (`forward_nvfp4_grouped_decode`), with the active list `EXPERTS` long.
fn sort_bytes(te: usize) -> usize {
    let ne = EXPERTS as usize;
    te * 4 * 3 + (ne + 1) * 4 + ne * 4 + 4
}

fn sort_out(base: DevicePtr, te: usize) -> Fp8GroupedSortOut {
    let ne = EXPERTS as usize;
    let token_to_perm = base.offset(te * 8 + (ne + 1) * 4);
    let active_experts = token_to_perm.offset(te * 4);
    Fp8GroupedSortOut {
        sorted_token_ids: base,
        sorted_expert_ids: base.offset(te * 4),
        expert_offsets: base.offset(te * 8),
        token_to_perm,
        active_experts,
        active_count: active_experts.offset(ne * 4),
    }
}

/// 2026-10-14: One routing's downloaded outputs; `logits` empty when the sort overwrote them.
#[derive(Debug)]
struct Routed {
    logits: Vec<u8>,
    idx: Vec<u32>,
    w: Vec<u8>,
    sort: Vec<i32>,
}

fn words(b: &[u8]) -> impl Iterator<Item = [u8; 4]> + '_ {
    b.chunks_exact(4).map(|c| c.try_into().unwrap())
}

fn read(
    g: &dyn GpuBackend,
    logits: Option<DevicePtr>,
    idx: DevicePtr,
    w: DevicePtr,
    sort: DevicePtr,
    m: u32,
) -> Routed {
    g.synchronize(0).unwrap();
    let te = (m * TOP_K) as usize;
    Routed {
        logits: logits.map_or_else(Vec::new, |p| download(g, p, (m * EXPERTS * 2) as usize)),
        idx: words(&download(g, idx, te * 4))
            .map(u32::from_le_bytes)
            .collect(),
        w: download(g, w, te * 4),
        sort: words(&download(g, sort, sort_bytes(te)))
            .map(i32::from_le_bytes)
            .collect(),
    }
}

/// 2026-10-14: The three launches over the first `m` rows, the router split into `y_blocks`
/// block rows.
fn three(
    g: &dyn GpuBackend,
    k: &Kernels,
    (input, gate): (DevicePtr, DevicePtr),
    m: u32,
    y_blocks: u32,
    normalize: bool,
) -> Routed {
    let te = (m * TOP_K) as usize;
    let logits = sentinel(g, (MAX_ROWS * EXPERTS * 2) as usize);
    let (idx, w, sort) = (
        sentinel(g, te * 4),
        sentinel(g, te * 4),
        sentinel(g, sort_bytes(te)),
    );
    let weight = DenseWeight { weight: gate };
    dense_gemv_batchm_split(
        g, k.gemv, input, &weight, logits, m, y_blocks, EXPERTS, HIDDEN, EXPERTS, 0,
    )
    .unwrap();
    moe_topk_softmax_rows_warp(g, k.topk, logits, idx, w, EXPERTS, TOP_K, normalize, m, 0).unwrap();
    moe_fp8_grouped_sort(
        g,
        k.sort,
        sort_out(sort, te),
        idx,
        m * TOP_K,
        EXPERTS,
        TOP_K,
        0,
    )
    .unwrap();
    read(g, Some(logits), idx, w, sort, m)
}

/// 2026-10-14: The fused launch over the first `m` rows, its sort outputs in their own buffer or
/// (`aliased`) at the start of the logits buffer. Asserts the counter is zero afterwards.
#[allow(clippy::too_many_arguments)]
fn fused(
    g: &dyn GpuBackend,
    k: &Kernels,
    (input, gate): (DevicePtr, DevicePtr),
    counter: DevicePtr,
    m: u32,
    normalize: bool,
    aliased: bool,
) -> Routed {
    let te = (m * TOP_K) as usize;
    let logits_bytes = ((MAX_ROWS * EXPERTS * 2) as usize).max(sort_bytes(te));
    let logits = sentinel(g, logits_bytes);
    let (idx, w) = (sentinel(g, te * 4), sentinel(g, te * 4));
    let sort = if aliased {
        logits
    } else {
        sentinel(g, sort_bytes(te))
    };
    moe_route_fused(
        g,
        k.fused,
        input,
        gate,
        logits,
        idx,
        w,
        sort_out(sort, te),
        counter,
        m,
        EXPERTS,
        HIDDEN,
        TOP_K,
        normalize,
        0,
    )
    .unwrap();
    let out = read(g, (!aliased).then_some(logits), idx, w, sort, m);
    assert_eq!(download(g, counter, 4), [0; 4], "m={m}: counter not reset");
    out
}

/// 2026-10-14: The sort outputs with each expert's run of token ids sorted, after checking that
/// `token_to_perm` inverts the sort of `idx`.
fn canonical_sort(r: &Routed, m: u32) -> Vec<i32> {
    let (te, ne) = ((m * TOP_K) as usize, EXPERTS as usize);
    let s = &r.sort;
    let (tokens, experts) = (&s[..te], &s[te..2 * te]);
    let offsets = &s[2 * te..2 * te + ne + 1];
    let perm = &s[2 * te + ne + 1..3 * te + ne + 1];
    let mut seen = vec![false; te];
    for (slot, &p) in perm.iter().enumerate() {
        let p = usize::try_from(p).expect("negative perm");
        assert!(
            p < te && !seen[p],
            "m={m}: token_to_perm is not a permutation"
        );
        seen[p] = true;
        assert_eq!(
            tokens[p],
            (slot / TOP_K as usize) as i32,
            "m={m} slot {slot}: token"
        );
        assert_eq!(experts[p], r.idx[slot] as i32, "m={m} slot {slot}: expert");
    }
    let mut runs = tokens.to_vec();
    for e in 0..ne {
        runs[offsets[e] as usize..offsets[e + 1] as usize].sort_unstable();
    }
    let count = usize::try_from(s[s.len() - 1]).expect("negative active count");
    let active = &s[3 * te + ne + 1..][..count];
    [&runs, experts, offsets, active, &s[s.len() - 1..]].concat()
}

fn assert_same(got: &Routed, want: &Routed, m: u32, label: &str) {
    if !want.logits.is_empty() && !got.logits.is_empty() {
        assert_eq!(got.logits, want.logits, "{label} m={m}: logits");
    }
    assert_eq!(got.idx, want.idx, "{label} m={m}: picks");
    assert_eq!(got.w, want.w, "{label} m={m}: weights");
    assert_eq!(
        canonical_sort(got, m),
        canonical_sort(want, m),
        "{label} m={m}: sort outputs"
    );
}

/// 2026-10-14: The fixture's designed cases hold in `r` (so the comparison covers them).
fn assert_fixture_cases(r: &Routed, m: u32) {
    let k = TOP_K as usize;
    let row = |t: u32| &r.idx[t as usize * k..(t as usize + 1) * k];
    if m > ZERO_ROW {
        assert_eq!(row(ZERO_ROW), (0..TOP_K).collect::<Vec<_>>(), "zero row");
    }
    if m > NAN_ROW {
        assert_eq!(row(NAN_ROW), vec![0; k], "NaN row: pseudo picks");
    }
    if m > TWIN_ROWS.1 {
        assert_eq!(row(TWIN_ROWS.0), row(TWIN_ROWS.1), "equal rows");
    }
    for t in 0..m {
        for (kept, copy) in TWIN_EXPERTS {
            let at = |e: u32| row(t).iter().position(|&x| x == e);
            if let Some(c) = at(copy) {
                assert!(
                    at(kept).is_some_and(|a| a < c),
                    "row {t}: tied expert {copy} before {kept}"
                );
            }
        }
    }
}

fn setup() -> (
    metrale_gpu_runtime::cuda_backend::MetraleCudaBackend,
    (Vec<u8>, Vec<u8>),
) {
    let set = metrale_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("qwen3.8-flash-next/nvfp4 not in this build");
    let gpu = metrale_gpu_runtime::cuda_backend::MetraleCudaBackend::new(0, &set.modules)
        .expect("CUDA backend");
    (gpu, fixture_bytes())
}

fn zeroed_counter(g: &dyn GpuBackend) -> DevicePtr {
    let p = g.alloc(4).unwrap();
    g.memset(p, 0, 4).unwrap();
    p
}

#[test]
fn moe_route_fused_admits_the_kernels_shapes_only() {
    for m in 1..=MAX_ROWS {
        assert!(moe_route_fused_shape_ok(m, HIDDEN, EXPERTS, TOP_K), "m={m}");
    }
    assert!(moe_route_fused_shape_ok(1, 4096, 512, 32));
    assert!(moe_route_fused_shape_ok(1, 8, 1, 1));
    for (m, h, e, k) in [
        (0, HIDDEN, EXPERTS, TOP_K),
        (MAX_ROWS + 1, HIDDEN, EXPERTS, TOP_K),
        (1, 4104, EXPERTS, TOP_K),
        (1, 2564, EXPERTS, TOP_K),
        (1, 0, EXPERTS, TOP_K),
        (1, HIDDEN, 513, TOP_K),
        (1, HIDDEN, 0, 0),
        (1, HIDDEN, EXPERTS, 0),
        (1, HIDDEN, EXPERTS, 33),
        (1, HIDDEN, 8, 9),
    ] {
        assert!(
            !moe_route_fused_shape_ok(m, h, e, k),
            "m={m} h={h} e={e} k={k}"
        );
    }
}

#[test]
#[ignore]
fn moe_route_fused_writes_the_three_launches_bytes() {
    let (gpu, (input, gate)) = setup();
    let g: &dyn GpuBackend = &gpu;
    let k = kernels(g);
    let bufs = (upload(g, &input), upload(g, &gate));
    let counter = zeroed_counter(g);
    for m in WIDTHS {
        for normalize in [true, false] {
            let label = format!("normalize={normalize}");
            let want = three(
                g,
                &k,
                bufs,
                m,
                m.div_ceil(DENSE_GEMV_BATCHM_MAX_M),
                normalize,
            );
            assert_same(
                &three(g, &k, bufs, m, m, normalize),
                &want,
                m,
                &format!("{label} one row per block row"),
            );
            assert_fixture_cases(&want, m);
            let got = fused(g, &k, bufs, counter, m, normalize, false);
            assert_same(&got, &want, m, &label);
            assert_same(
                &fused(g, &k, bufs, counter, m, normalize, false),
                &got,
                m,
                &format!("{label} relaunch"),
            );
            assert_same(
                &fused(g, &k, bufs, counter, m, normalize, true),
                &want,
                m,
                &format!("{label} aliased"),
            );
        }
    }
    println!("ROUTE_FUSED_BYTES PASS widths={WIDTHS:?}");
}

#[test]
#[ignore]
fn moe_route_fused_rows_are_batch_invariant() {
    let (gpu, (input, gate)) = setup();
    let g: &dyn GpuBackend = &gpu;
    let k = kernels(g);
    let bufs = (upload(g, &input), upload(g, &gate));
    let counter = zeroed_counter(g);
    for normalize in [true, false] {
        let all = fused(g, &k, bufs, counter, MAX_ROWS, normalize, false);
        for m in WIDTHS {
            let part = fused(g, &k, bufs, counter, m, normalize, false);
            let (te, lb) = ((m * TOP_K) as usize, (m * EXPERTS * 2) as usize);
            assert_eq!(
                part.logits,
                all.logits[..lb],
                "normalize={normalize} m={m}: logits"
            );
            assert_eq!(
                part.idx,
                all.idx[..te],
                "normalize={normalize} m={m}: picks"
            );
            assert_eq!(
                part.w,
                all.w[..te * 4],
                "normalize={normalize} m={m}: weights"
            );
        }
    }
    println!("ROUTE_FUSED_INVARIANT PASS widths={WIDTHS:?}");
}

#[test]
#[ignore]
fn moe_route_fused_cold_timing() {
    let (gpu, (input, gate)) = setup();
    let g: &dyn GpuBackend = &gpu;
    let k = kernels(g);
    let d_input = upload(g, &input);
    let gates: Vec<DenseWeight> = (0..COLD_COPIES)
        .map(|_| DenseWeight {
            weight: upload(g, &gate),
        })
        .collect();
    let counter = zeroed_counter(g);
    let te_max = (MAX_ROWS * TOP_K) as usize;
    let logits = sentinel(
        g,
        ((MAX_ROWS * EXPERTS * 2) as usize).max(sort_bytes(te_max)),
    );
    let (idx, w) = (sentinel(g, te_max * 4), sentinel(g, te_max * 4));
    let weight_mb = gate.len() as f64 / 1e6;
    for m in TIMED {
        let te = (m * TOP_K) as usize;
        let out = move || sort_out(logits, te);
        let mut q = 0;
        let three_us = time_us(g, || {
            q = (q + 1) % COLD_COPIES;
            dense_gemv_batchm_split(
                g, k.gemv, d_input, &gates[q], logits, m, 1, EXPERTS, HIDDEN, EXPERTS, 0,
            )
            .unwrap();
            moe_topk_softmax_rows_warp(g, k.topk, logits, idx, w, EXPERTS, TOP_K, true, m, 0)
                .unwrap();
            moe_fp8_grouped_sort(g, k.sort, out(), idx, m * TOP_K, EXPERTS, TOP_K, 0).unwrap();
        });
        let fused_us = time_us(g, || {
            q = (q + 1) % COLD_COPIES;
            moe_route_fused(
                g,
                k.fused,
                d_input,
                gates[q].weight,
                logits,
                idx,
                w,
                out(),
                counter,
                m,
                EXPERTS,
                HIDDEN,
                TOP_K,
                true,
                0,
            )
            .unwrap();
        });
        println!(
            "ROUTE_FUSED_US m={m} three={three_us:.2} fused={fused_us:.2} saved={:.2} three_tbs={:.2} fused_tbs={:.2}",
            three_us - fused_us,
            weight_mb / three_us,
            weight_mb / fused_us,
        );
    }
}
