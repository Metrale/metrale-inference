// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Tests of the tp expert layout without a GPU: the slice geometry, the NVFP4 byte
//! slicing and padding, the expert MLP summed over the rank slices against the whole expert,
//! and the MLP config's layout refusals.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use metrale_cache::kv_dequant::{NVFP4_E2M1_LUT, e4m3_lut};
use metrale_config::{ModelConfig, MoeExpertLayout, parse_config};

use super::*;
use crate::glm5next_mlp::Glm5NextMlpConfig;

const CONFIG: &str =
    include_str!("../../../model-engine/tests/fixtures/glm53-nvfp4-9e0d74e3-config.json");

fn slices(full: usize, tp: usize) -> Vec<ExpertSlice> {
    (0..tp)
        .map(|r| expert_slice(full, tp, r).unwrap())
        .collect()
}

/// 2026-10-09: GLM-5.3's 2048 over three ranks: 768 / 640 / 640, contiguous, no padding.
#[test]
fn glm53_width_splits_768_640_640() {
    let s = slices(2048, 3);
    let got: Vec<_> = s.iter().map(|s| (s.start, s.real, s.len)).collect();
    assert_eq!(got, [(0, 768, 768), (768, 640, 640), (1408, 640, 640)]);
    assert!(s.iter().all(|s| s.pad() == 0 && s.full == 2048));
}

/// 2026-10-09: For many widths and rank counts, the slices partition the padded width in
/// unit-aligned order, their real parts partition the checkpoint width, padding sits only on
/// the last rank and is narrower than a unit.
#[test]
fn slices_partition_the_width_and_pad_only_the_tail() {
    for tp in 2..=4 {
        for full in (16usize..=2048).step_by(16) {
            if full.next_multiple_of(EXPERT_TP_UNIT) / EXPERT_TP_UNIT < tp {
                assert!(expert_slice(full, tp, 0).is_err(), "{full} over {tp}");
                continue;
            }
            let s = slices(full, tp);
            let (mut at, mut real_at) = (0, 0);
            for (r, x) in s.iter().enumerate() {
                assert_eq!(x.start, at, "{full}/{tp} rank {r}");
                assert_eq!(x.start, real_at, "{full}/{tp} rank {r}");
                assert!(x.len.is_multiple_of(EXPERT_TP_UNIT) && x.real >= 1);
                assert!(x.real.is_multiple_of(16), "real part splits a scale group");
                if r + 1 < tp {
                    assert_eq!(x.pad(), 0, "{full}/{tp}: padding before the last rank");
                }
                at += x.len;
                real_at += x.real;
            }
            assert_eq!(at, full.next_multiple_of(EXPERT_TP_UNIT));
            assert_eq!(real_at, full);
        }
    }
}

/// 2026-10-09: A width that is not a 16-multiple, a zero width, or too few units are refused.
#[test]
fn bad_widths_are_refused() {
    for (full, tp) in [(0, 3), (2040, 3), (128, 3), (256, 3)] {
        assert!(expert_slice(full, tp, 0).is_err(), "{full} over {tp}");
    }
    assert!(expert_slice(384, 3, 3).is_err(), "rank out of range");
}

/// 2026-10-09: An `[n, k]` NVFP4 weight whose every packed byte and scale names its position.
fn tagged(n: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    let p = (0..n * k / 2).map(|i| (i * 7 + 1) as u8 | 1).collect();
    let s = (0..n * k / 16).map(|i| (i * 13 + 3) as u8 | 1).collect();
    (p, s)
}

/// 2026-10-09: Gate/up rows and down columns of every rank rejoin, in rank order, to the whole
/// tensor; the padded rows/columns are all zero bytes. Run on an even split and a padded one.
#[test]
fn byte_slices_rejoin_to_the_whole_and_pad_with_zeros() {
    let hidden = 64usize;
    for (full, tp) in [(640usize, 3usize), (336, 3), (256, 2)] {
        let (gp, gs) = tagged(full, hidden);
        let (dp, ds) = tagged(hidden, full);
        let mut rows_p = Vec::new();
        let mut rows_s = Vec::new();
        let mut cols: Vec<(Vec<u8>, Vec<u8>)> = vec![(Vec::new(), Vec::new()); hidden];
        for s in slices(full, tp) {
            let (p, sc) = slice_expert_rows(&gp, &gs, hidden, &s).unwrap();
            assert_eq!(
                (p.len(), sc.len()),
                (s.len * hidden / 2, s.len * hidden / 16)
            );
            let (real_p, pad_p) = p.split_at(s.real * hidden / 2);
            let (real_s, pad_s) = sc.split_at(s.real * hidden / 16);
            assert!(pad_p.iter().chain(pad_s).all(|&b| b == 0), "row pad");
            rows_p.extend_from_slice(real_p);
            rows_s.extend_from_slice(real_s);

            let (p, sc) = slice_expert_cols(&dp, &ds, hidden, &s).unwrap();
            assert_eq!(
                (p.len(), sc.len()),
                (hidden * s.len / 2, hidden * s.len / 16)
            );
            for (h, (rp, rs)) in p.chunks(s.len / 2).zip(sc.chunks(s.len / 16)).enumerate() {
                assert!(
                    rp[s.real / 2..]
                        .iter()
                        .chain(&rs[s.real / 16..])
                        .all(|&b| b == 0)
                );
                cols[h].0.extend_from_slice(&rp[..s.real / 2]);
                cols[h].1.extend_from_slice(&rs[..s.real / 16]);
            }
        }
        assert_eq!((rows_p, rows_s), (gp, gs), "{full}/{tp}: gate rows rejoin");
        for (h, (p, s)) in cols.into_iter().enumerate() {
            let (pb, sb) = (full / 2, full / 16);
            assert_eq!(
                p,
                dp[h * pb..(h + 1) * pb],
                "{full}/{tp}: down row {h} codes"
            );
            assert_eq!(
                s,
                ds[h * sb..(h + 1) * sb],
                "{full}/{tp}: down row {h} scales"
            );
        }
    }
}

/// 2026-10-09: Host dequantization of an `[n, k]` NVFP4 weight (no global scale).
fn dequant(p: &[u8], s: &[u8], n: usize, k: usize) -> Vec<f32> {
    (0..n * k)
        .map(|i| {
            let (r, c) = (i / k, i % k);
            let b = p[r * k / 2 + c / 2];
            let code = if c % 2 == 0 { b & 0xF } else { b >> 4 };
            NVFP4_E2M1_LUT[code as usize] * e4m3_lut()[s[r * k / 16 + c / 16] as usize]
        })
        .collect()
}

/// 2026-10-09: GLM's clamped SwiGLU expert on the host: `down(silu(min(g, L)) * clamp(u, L))`.
fn expert_out(x: &[f32], g: &[f32], u: &[f32], d: &[f32], inter: usize, hidden: usize) -> Vec<f32> {
    let lim = 10.0f32;
    let dot = |w: &[f32], v: &[f32]| w.iter().zip(v).map(|(a, b)| a * b).sum::<f32>();
    let act: Vec<f32> = (0..inter)
        .map(|i| {
            let gv = dot(&g[i * hidden..(i + 1) * hidden], x).min(lim);
            let uv = dot(&u[i * hidden..(i + 1) * hidden], x).clamp(-lim, lim);
            gv / (1.0 + (-gv).exp()) * uv
        })
        .collect();
    (0..hidden)
        .map(|h| dot(&d[h * inter..(h + 1) * inter], &act))
        .collect()
}

/// 2026-10-09: Realistic random NVFP4 (codes anywhere, scales 0.25 to 2).
fn random(n: usize, k: usize, seed: &mut u64) -> (Vec<u8>, Vec<u8>) {
    let mut next = || {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    };
    let p = (0..n * k / 2).map(|_| next() as u8).collect();
    let s = (0..n * k / 16)
        .map(|_| 0x28 + (next() % 24) as u8)
        .collect();
    (p, s)
}

/// 2026-10-09: The expert MLP over each rank's slices, summed over ranks, equals the whole
/// expert to FP32 rounding, padded columns included; and a slicer that dropped a rank's last
/// unit (negative control) is far off.
#[test]
fn rank_slices_sum_to_the_whole_expert() {
    let hidden = 64usize;
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    for (full, tp) in [(640usize, 3usize), (336, 3)] {
        let (gp, gs) = random(full, hidden, &mut seed);
        let (up, us) = random(full, hidden, &mut seed);
        let (dp, ds) = random(hidden, full, &mut seed);
        let x: Vec<f32> = (0..hidden)
            .map(|i| ((i * 37 % 23) as f32 - 11.0) / 40.0)
            .collect();
        let whole = expert_out(
            &x,
            &dequant(&gp, &gs, full, hidden),
            &dequant(&up, &us, full, hidden),
            &dequant(&dp, &ds, hidden, full),
            full,
            hidden,
        );
        let rank_out = |s: &ExpertSlice, drop_tail: bool| {
            let rows = |p: &[u8], sc: &[u8]| {
                let (p, sc) = slice_expert_rows(p, sc, hidden, s).unwrap();
                dequant(&p, &sc, s.len, hidden)
            };
            let (p, sc) = slice_expert_cols(&dp, &ds, hidden, s).unwrap();
            let mut d = dequant(&p, &sc, hidden, s.len);
            if drop_tail {
                for row in d.chunks_mut(s.len) {
                    row[s.len - EXPERT_TP_UNIT..].fill(0.0);
                }
            }
            expert_out(&x, &rows(&gp, &gs), &rows(&up, &us), &d, s.len, hidden)
        };
        let sum = |drop: bool| {
            let mut acc = vec![0f32; hidden];
            for s in slices(full, tp) {
                for (a, v) in acc.iter_mut().zip(rank_out(&s, drop)) {
                    *a += v;
                }
            }
            acc
        };
        let err = |got: &[f32]| {
            let num: f32 = got.iter().zip(&whole).map(|(a, b)| (a - b).powi(2)).sum();
            (num / whole.iter().map(|b| b * b).sum::<f32>()).sqrt()
        };
        assert!(
            err(&sum(false)) < 1e-5,
            "{full}/{tp}: rel {}",
            err(&sum(false))
        );
        assert!(
            err(&sum(true)) > 1e-2,
            "{full}/{tp}: the control did not move"
        );
    }
}

fn glm(tp: usize, ep: usize, rank: usize, layout: MoeExpertLayout) -> ModelConfig {
    let mut c = parse_config(CONFIG).expect("the real checkpoint config parses");
    c.tp_world_size = tp;
    c.tp_rank = rank;
    c.ep_world_size = ep;
    c.ep_rank = if ep > 1 { rank } else { 0 };
    c.moe_expert_layout = layout;
    c
}

/// 2026-10-09: The tp layout at TP 3: every rank holds all 288 experts, its slice of each, and
/// every routed id maps to a local slot; the site output needs the all-reduce.
#[test]
fn tp_layout_config_holds_every_expert_sliced() {
    let widths: Vec<usize> = (0..3)
        .map(|r| {
            let c = Glm5NextMlpConfig::from_config(&glm(3, 1, r, MoeExpertLayout::Tp)).unwrap();
            assert_eq!(c.local_experts, 288);
            assert_eq!(c.local_expert_range(), 0..288);
            assert!((0..288).all(|id| c.local_slot(id) == Some(id)), "rank {r}");
            assert!(c.needs_all_reduce());
            assert_eq!(c.expert_shard, ExpertShard::Sliced(slices(2048, 3)[r]));
            assert_eq!(
                c.local_shared_intermediate,
                [688, 680, 680][r],
                "shared unchanged"
            );
            c.moe_intermediate
        })
        .collect();
    assert_eq!(widths, [768, 640, 640]);
}

/// 2026-10-09: The ep layout at TP 3 / EP 3 is the whole-expert plan: 96 experts of 2048 each.
#[test]
fn ep_layout_config_is_the_whole_expert_plan() {
    for r in 0..3 {
        let c = Glm5NextMlpConfig::from_config(&glm(3, 3, r, MoeExpertLayout::Ep)).unwrap();
        assert_eq!(c.expert_shard, ExpertShard::Whole);
        assert_eq!((c.moe_intermediate, c.local_experts), (2048, 96));
        assert_eq!(c.local_expert_range(), r * 96..(r + 1) * 96);
    }
}

/// 2026-10-09: The tp layout is refused beside EP and at one TP rank; a config whose fields
/// disagree with its slice fails `validate`.
#[test]
fn tp_layout_refusals() {
    for (tp, ep) in [(3, 3), (1, 1)] {
        let e = Glm5NextMlpConfig::from_config(&glm(tp, ep, 0, MoeExpertLayout::Tp)).unwrap_err();
        assert!(e.to_string().contains("--moe-expert-layout tp"), "{e}");
    }
    let ok = Glm5NextMlpConfig::from_config(&glm(3, 1, 1, MoeExpertLayout::Tp)).unwrap();
    for bad in [
        Glm5NextMlpConfig {
            moe_intermediate: 2048,
            ..ok
        },
        Glm5NextMlpConfig {
            local_experts: 96,
            ..ok
        },
        Glm5NextMlpConfig {
            ep_world_size: 3,
            ..ok
        },
    ] {
        let e = bad.validate().unwrap_err();
        assert!(e.to_string().contains("tp expert layout"), "{e}");
    }
}
