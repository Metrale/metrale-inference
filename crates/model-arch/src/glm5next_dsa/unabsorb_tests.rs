// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: G1 of MLA-UNABSORB, on the host: the unabsorbed prefill's weights (`host`) give
//! exactly the absorbed path's products on every tensor-parallel rank, with negative controls
//! (a head-stride error, swapped K/V halves, another rank's `o_proj` columns) that must fire;
//! the lever's parse, geometry check, launch-grid refusals, site tier table and footprint; and
//! the load-time state on a recording mock (off touches nothing; on uploads checkpoint bits).
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: none beyond the types.

use anyhow::{Result, bail};
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::build::absorb_for_rank;
use super::host::{
    UnabsorbHost, dense_ref, head_bmm_ref, split_w_uk_t, split_w_uv, unabsorb_for_rank,
};
use super::lever::{ENV, O_PROJ_SITE, Q_B_SITE, SITE_RULES, parse};
use super::*;

/// 2026-10-15: A small NoPE layer; `nope` != `vd` so a K/V width mix-up changes shapes.
fn cfg(local_heads: usize, nope: usize, vd: usize) -> Glm5NextDsaConfig {
    Glm5NextDsaConfig {
        hidden: 7,
        index_heads: 4,
        index_head_dim: 8,
        index_kpool: 4,
        index_topk: 16,
        always_select_tail: true,
        local_heads,
        q_lora_rank: 6,
        kv_lora_rank: 4,
        qk_nope_head_dim: nope,
        qk_rope_head_dim: 0,
        v_head_dim: vd,
        max_context: 16_384,
    }
}

/// 2026-10-15: GLM-5.3-Flash's DSA geometry on one rank of a 3-way split (22 of 64 heads).
fn glm(local_heads: usize) -> Glm5NextDsaConfig {
    Glm5NextDsaConfig {
        hidden: 4096,
        q_lora_rank: 1536,
        kv_lora_rank: 512,
        qk_nope_head_dim: 256,
        v_head_dim: 256,
        ..cfg(local_heads, 256, 256)
    }
}

/// 2026-10-15: Small integers in `-3..=3` from a seed: every product and sum below stays an
/// integer far inside f32's exact range, so the absorbed (f32) and unabsorbed (f64) paths must
/// agree bit for bit when the layouts are right.
fn ints(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s % 7) as i64 - 3) as f32
        })
        .collect()
}

fn f64s(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| x as f64).collect()
}

struct Ckpt {
    q_b: Vec<f32>,
    kv_b: Vec<f32>,
    o: Vec<f32>,
}

impl Ckpt {
    fn new(c: &Glm5NextDsaConfig, heads: usize, mk: impl Fn(usize, u64) -> Vec<f32>) -> Self {
        let (nope, vd, kvl) = (c.qk_nope_head_dim, c.v_head_dim, c.kv_lora_rank);
        Self {
            q_b: mk(heads * nope * c.q_lora_rank, 1),
            kv_b: mk(heads * (nope + vd) * kvl, 2),
            o: mk(c.hidden * heads * vd, 3),
        }
    }

    fn load(&self, n: &str) -> Result<Vec<f32>> {
        Ok(match n {
            "self_attn.q_b_proj.weight" => self.q_b.clone(),
            "self_attn.kv_b_proj.weight" => self.kv_b.clone(),
            "self_attn.o_proj.weight" => self.o.clone(),
            other => bail!("unexpected tensor {other}"),
        })
    }
}

/// 2026-10-15: The largest |absorbed - unabsorbed| over `m` rows of query and output, for a
/// rank's absorbed weights and its (possibly corrupted) unabsorbed ones.
fn gap(c: &Glm5NextDsaConfig, q_abs: &[f32], o_abs: &[f32], u: &UnabsorbHost, m: usize) -> f64 {
    let (lh, nope, vd, kvl) = (
        c.local_heads,
        c.qk_nope_head_dim,
        c.v_head_dim,
        c.kv_lora_rank,
    );
    let x = f64s(&ints(m * c.q_lora_rank, 11));
    let a = f64s(&ints(m * lh * kvl, 12));
    let q_ref = dense_ref(&x, &f64s(q_abs), m, lh * kvl, c.q_lora_rank);
    let mid = dense_ref(&x, &f64s(&u.q_b), m, lh * nope, c.q_lora_rank);
    let q_un = head_bmm_ref(&mid, &f64s(&u.w_uk_t), m, lh, nope, kvl);
    let o_ref = dense_ref(&a, &f64s(o_abs), m, c.hidden, lh * kvl);
    let mid = head_bmm_ref(&a, &f64s(&u.w_uv), m, lh, kvl, vd);
    let o_un = dense_ref(&mid, &f64s(&u.o_proj), m, c.hidden, lh * vd);
    q_ref
        .iter()
        .zip(&q_un)
        .chain(o_ref.iter().zip(&o_un))
        .map(|(r, u)| (r - u).abs())
        .fold(0.0, f64::max)
}

/// 2026-10-15: `(cfg, plan)` of every rank of a 3-way split of 5 heads (2, 2, 1).
fn ranks(nope: usize, vd: usize) -> Vec<(Glm5NextDsaConfig, DsaTpPlan)> {
    (0..3)
        .map(|rank| {
            let local = metrale_config::tp_split(5, 3, rank, 1).unwrap().len;
            let c = cfg(local, nope, vd);
            let plan = DsaTpPlan::new(rank, 3, 5, &c).unwrap();
            (c, plan)
        })
        .collect()
}

/// 2026-10-15: On integer weights and activations, `q_b_proj` then the per-head `W_UK^T` is
/// exactly `q_absorb`, and the per-head `W_UV` then `o_proj` exactly `o_absorb`, on every rank
/// of an uneven split (so the head offsets of `q_b_proj`, `kv_b_proj` and `o_proj` are all
/// exercised). 17 rows: more than one row of every shape.
#[test]
fn unabsorbed_equals_absorbed_exactly_on_every_rank() {
    let ck = Ckpt::new(&cfg(5, 3, 5), 5, ints);
    for (c, plan) in ranks(3, 5) {
        let (q_abs, o_abs) = absorb_for_rank(&c, &plan, &|n| ck.load(n)).unwrap();
        let u = unabsorb_for_rank(&c, &plan, &|n| ck.load(n)).unwrap();
        assert_eq!(
            gap(&c, &q_abs, &o_abs, &u, 17),
            0.0,
            "rank {}",
            plan.head_start
        );
    }
}

/// 2026-10-15: Real-valued weights: the two paths agree to f32 rounding of the absorbed
/// products (the absorbed weights are f32 sums, the reference here is f64).
#[test]
fn unabsorbed_matches_absorbed_on_real_weights() {
    let wave = |n: usize, s: u64| -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32) * (0.11 + s as f32 * 0.13)).sin())
            .collect()
    };
    let ck = Ckpt::new(&cfg(5, 3, 5), 5, wave);
    for (c, plan) in ranks(3, 5) {
        let (q_abs, o_abs) = absorb_for_rank(&c, &plan, &|n| ck.load(n)).unwrap();
        let u = unabsorb_for_rank(&c, &plan, &|n| ck.load(n)).unwrap();
        let g = gap(&c, &q_abs, &o_abs, &u, 9);
        assert!(g < 1e-4, "rank {}: gap {g}", plan.head_start);
    }
}

/// 2026-10-15: The comparison detects the layout errors it exists for. Each control corrupts
/// one weight the way a plausible bug would and must open a gap: two heads' `W_UK^T` swapped
/// (a head-stride error), `W_UV` taken from the K half (swapped halves; `nope == vd` so the
/// shapes still fit), and another rank's `o_proj` columns (a wrong shard offset).
#[test]
fn negative_controls_fire() {
    let ck = Ckpt::new(&cfg(5, 4, 4), 5, ints);
    let rk = ranks(4, 4);
    let (c, plan) = &rk[1];
    let (q_abs, o_abs) = absorb_for_rank(c, plan, &|n| ck.load(n)).unwrap();
    let fresh = || unabsorb_for_rank(c, plan, &|n| ck.load(n)).unwrap();
    assert_eq!(gap(c, &q_abs, &o_abs, &fresh(), 5), 0.0, "control baseline");

    let mut u = fresh();
    let per = c.kv_lora_rank * c.qk_nope_head_dim;
    let (h0, h1) = u.w_uk_t.split_at_mut(per);
    h0.swap_with_slice(&mut h1[..per]);
    assert!(gap(c, &q_abs, &o_abs, &u, 5) > 0.0, "head-stride control");

    let mut u = fresh();
    let rows = c.qk_nope_head_dim + c.v_head_dim;
    let kv_local = &ck.kv_b[plan.head_start * rows * c.kv_lora_rank..]
        [..plan.local_heads * rows * c.kv_lora_rank];
    u.w_uv = (0..plan.local_heads)
        .flat_map(|h| {
            kv_local[h * rows * c.kv_lora_rank..][..c.v_head_dim * c.kv_lora_rank].to_vec()
        })
        .collect();
    assert!(
        gap(c, &q_abs, &o_abs, &u, 5) > 0.0,
        "swapped-halves control"
    );

    let mut u = fresh();
    let (c0, p0) = &rk[0];
    u.o_proj = unabsorb_for_rank(c0, p0, &|n| ck.load(n)).unwrap().o_proj;
    assert!(gap(c, &q_abs, &o_abs, &u, 5) > 0.0, "shard-offset control");
}

/// 2026-10-15: The per-head layouts index by value: with `kv_b[i] = i`, `w_uk_t[h][c][r]` is
/// row `h*(nope+vd) + r`, column `c`, and `w_uv[h][r][c]` row `h*(nope+vd) + nope + r`.
#[test]
fn split_layouts_index_the_checkpoint_rows() {
    let c = cfg(2, 3, 5);
    let (nope, vd, kvl) = (3, 5, 4);
    let kv_b: Vec<f32> = (0..2 * (nope + vd) * kvl).map(|i| i as f32).collect();
    let uk = split_w_uk_t(&c, &kv_b, 2).unwrap();
    let uv = split_w_uv(&c, &kv_b, 2).unwrap();
    assert_eq!((uk.len(), uv.len()), (2 * kvl * nope, 2 * vd * kvl));
    for h in 0..2 {
        for cc in 0..kvl {
            for r in 0..nope {
                let want = ((h * (nope + vd) + r) * kvl + cc) as f32;
                assert_eq!(uk[(h * kvl + cc) * nope + r], want, "uk h{h} c{cc} r{r}");
            }
            for r in 0..vd {
                let want = ((h * (nope + vd) + nope + r) * kvl + cc) as f32;
                assert_eq!(uv[(h * vd + r) * kvl + cc], want, "uv h{h} r{r} c{cc}");
            }
        }
    }
    let e = split_w_uv(&c, &kv_b[1..], 2).unwrap_err();
    assert!(e.to_string().contains("expected 64"), "{e}");
    let mut rope = c;
    rope.qk_rope_head_dim = 2;
    assert!(split_w_uk_t(&rope, &kv_b, 2).is_err());
}

/// 2026-10-15: Unset and `0` are off; `declared` and `dsa` are the two options; anything else
/// is refused with the variable's name.
#[test]
fn the_lever_parses_three_values_and_refuses_the_rest() {
    assert_eq!(parse(None), Ok(MlaUnabsorb::Off));
    assert_eq!(parse(Some("0")), Ok(MlaUnabsorb::Off));
    assert_eq!(parse(Some("declared")), Ok(MlaUnabsorb::Declared));
    assert_eq!(parse(Some("dsa")), Ok(MlaUnabsorb::Dsa));
    for bad in ["", "1", "on", "DSA", "declared "] {
        let e = parse(Some(bad)).unwrap_err();
        assert!(e.contains(ENV), "{bad:?}: {e}");
    }
    for m in [MlaUnabsorb::Off, MlaUnabsorb::Declared, MlaUnabsorb::Dsa] {
        assert_eq!(parse(Some(m.name())), Ok(m));
    }
}

/// 2026-10-15: GLM-5.3's geometry passes; each dimension the kernel is compiled for refuses
/// alone.
#[test]
fn the_geometry_check_refuses_each_other_dimension() {
    check_cfg(&glm(22)).unwrap();
    let edits: [fn(&mut Glm5NextDsaConfig); 4] = [
        |c| c.qk_rope_head_dim = 64,
        |c| c.qk_nope_head_dim = 192,
        |c| c.v_head_dim = 128,
        |c| c.kv_lora_rank = 256,
    ];
    for (i, edit) in edits.iter().enumerate() {
        let mut c = glm(22);
        edit(&mut c);
        assert!(check_cfg(&c).is_err(), "edit {i}");
    }
}

/// 2026-10-15: The grid is `(N/128, ceil(M/128), heads)`; no rows, more rows than the scratch,
/// no heads, a null or unaligned pointer are refused.
#[test]
fn the_launch_grid_and_its_refusals() {
    let p = [DevicePtr(0x1000), DevicePtr(0x2000), DevicePtr(0x3000)];
    assert_eq!(
        head_bmm_grid(HeadBmm::Q, 8208, 22, 8208, p).unwrap(),
        [4, 65, 22]
    );
    assert_eq!(
        head_bmm_grid(HeadBmm::O, 1, 22, 8208, p).unwrap(),
        [2, 1, 22]
    );
    assert_eq!(
        head_bmm_grid(HeadBmm::O, 128, 1, 128, p).unwrap(),
        [2, 1, 1]
    );
    assert_eq!(
        head_bmm_grid(HeadBmm::O, 129, 1, 129, p).unwrap(),
        [2, 2, 1]
    );
    assert!(head_bmm_grid(HeadBmm::Q, 0, 22, 8208, p).is_err());
    assert!(head_bmm_grid(HeadBmm::Q, 8209, 22, 8208, p).is_err());
    assert!(head_bmm_grid(HeadBmm::Q, 16, 0, 8208, p).is_err());
    for i in 0..3 {
        for bad in [0u64, 0x1008] {
            let mut q = p;
            q[i] = DevicePtr(bad);
            assert!(
                head_bmm_grid(HeadBmm::O, 16, 22, 8208, q).is_err(),
                "ptr {i} {bad:#x}"
            );
        }
    }
}

/// 2026-10-15: The DSA site tier table is complete and unambiguous: every DSA site the
/// registry lists has exactly one row per mode, except the unabsorbed sites with the lever off
/// (none: listing one is an error); `declared` keeps them at the checkpoint, `dsa` gives them
/// the knob; the absorbed sites follow the knob in every mode.
#[test]
fn the_site_tier_table_is_complete() {
    let sites = crate::glm5next_w4a16_dense::DSA_SITES;
    for (_, name, _) in SITE_RULES {
        assert!(sites.contains(&name), "{name} is not a DSA site");
    }
    for mode in [MlaUnabsorb::Off, MlaUnabsorb::Declared, MlaUnabsorb::Dsa] {
        for name in sites {
            let rows = SITE_RULES
                .iter()
                .filter(|(m, n, _)| *m == mode && *n == name)
                .count();
            let unabsorbed = name == Q_B_SITE || name == O_PROJ_SITE;
            let want = usize::from(mode.is_on() || !unabsorbed);
            assert_eq!(rows, want, "{name} under {mode:?}");
            let rule = site_rule(name, mode);
            match (mode, unabsorbed) {
                (MlaUnabsorb::Off, true) => assert!(rule.is_err(), "{name}"),
                (MlaUnabsorb::Declared, true) => assert_eq!(rule.unwrap(), SiteRule::Checkpoint),
                _ => assert_eq!(rule.unwrap(), SiteRule::DsaKnob, "{name} {mode:?}"),
            }
        }
    }
    assert!(site_rule("dsa.kv_a_proj", MlaUnabsorb::Declared).is_err());
}

/// 2026-10-15: The footprint the readback discloses, at GLM-5.3's rank shape (22 heads, 8208
/// verify rows): 92.5 MB of scratch and 75.0 MB of BF16 weights per DSA layer.
#[test]
fn the_footprint_at_glm_shape() {
    let c = glm(22);
    assert_eq!(mid_bytes(&c, 8208), 8208 * 22 * 256 * 2);
    assert_eq!(mid_bytes(&c, 8208), 92_471_296);
    assert_eq!(
        weight_bytes(&c),
        2 * (22 * 256 * 1536 + 4096 * 22 * 256 + 2 * 22 * 512 * 256)
    );
    assert_eq!(weight_bytes(&c), 74_973_184);
}

/// 2026-10-15: A one-head rank at GLM's per-head widths with a small `q_lora` and `hidden`, so
/// the load-time state is built on the mock at the geometry `check_cfg` accepts.
fn small_glm() -> (Glm5NextDsaConfig, DsaTpPlan, Ckpt) {
    let mut c = glm(1);
    c.q_lora_rank = 8;
    c.hidden = 16;
    let plan = DsaTpPlan::new(0, 1, 1, &c).unwrap();
    let ck = Ckpt::new(&c, 1, ints);
    (c, plan, ck)
}

/// 2026-10-15: Off allocates nothing, resolves no kernel and builds no weights, even for a
/// geometry the lever would refuse.
#[test]
fn off_touches_nothing() {
    let gpu = MockGpuBackend::new();
    let before = gpu.live_bytes();
    let bad = cfg(1, 3, 5);
    let t = TextUnabsorb::new(&gpu, &bad, 8208, true, 11, MlaUnabsorb::Off).unwrap();
    let plan = DsaTpPlan::new(0, 1, 1, &bad).unwrap();
    let ck = Ckpt::new(&bad, 1, ints);
    assert!(
        t.weights(&gpu, &bad, &plan, &|n| ck.load(n))
            .unwrap()
            .is_none()
    );
    assert_eq!(gpu.live_bytes(), before);
    assert!(gpu.kernel_lookups_snapshot().is_empty());
    assert!(gpu.launches_snapshot().is_empty());
}

/// 2026-10-15: On, a geometry the kernel is not compiled for is refused at load.
#[test]
fn on_refuses_a_geometry_it_cannot_run() {
    let gpu = MockGpuBackend::new();
    for mode in [MlaUnabsorb::Declared, MlaUnabsorb::Dsa] {
        let e = TextUnabsorb::new(&gpu, &cfg(1, 3, 5), 16, true, 1, mode)
            .err()
            .expect("refused");
        assert!(e.to_string().contains(ENV), "{e}");
    }
}

/// 2026-10-15: On, the shared scratch is allocated once at `new` and every layer points at it;
/// per layer, each layer allocates its own. The uploaded weights hold the checkpoint values
/// moved (BF16 of small integers is exact), never products, and both entries are resolved.
#[test]
fn on_uploads_checkpoint_bits_and_sizes_the_scratch() {
    let (c, plan, ck) = small_glm();
    let rows = 40;
    for shared in [true, false] {
        let gpu = MockGpuBackend::new();
        let base = gpu.live_bytes().unwrap();
        let t = TextUnabsorb::new(&gpu, &c, rows, shared, 2, MlaUnabsorb::Declared).unwrap();
        let after_new = gpu.live_bytes().unwrap() - base;
        assert_eq!(after_new, if shared { mid_bytes(&c, rows) } else { 0 });
        let lookups = gpu.kernel_lookups_snapshot();
        for f in ["glm5next_mla_head_bmm_q", "glm5next_mla_head_bmm_o"] {
            assert!(
                lookups.contains(&(launch::MODULE.to_string(), f.to_string())),
                "{f}"
            );
        }
        let a = t
            .weights(&gpu, &c, &plan, &|n| ck.load(n))
            .unwrap()
            .unwrap();
        let b = t
            .weights(&gpu, &c, &plan, &|n| ck.load(n))
            .unwrap()
            .unwrap();
        assert_eq!(
            a.mid == b.mid,
            shared,
            "scratch shared iff the workspace is"
        );
        assert_eq!((a.mode, a.mid_rows), (MlaUnabsorb::Declared, rows));
        let per_layer = weight_bytes(&c) + if shared { 0 } else { mid_bytes(&c, rows) };
        assert_eq!(gpu.live_bytes().unwrap() - base, after_new + 2 * per_layer);

        let h = unabsorb_for_rank(&c, &plan, &|n| ck.load(n)).unwrap();
        let bf16 = |v: &[f32]| -> Vec<u8> {
            v.iter()
                .flat_map(|&x| ((x.to_bits() >> 16) as u16).to_le_bytes())
                .collect()
        };
        for (ptr, host, what) in [
            (a.q_b, &h.q_b, "q_b"),
            (a.o_proj, &h.o_proj, "o_proj"),
            (a.w_uk_t, &h.w_uk_t, "w_uk_t"),
            (a.w_uv, &h.w_uv, "w_uv"),
        ] {
            assert_eq!(gpu.read_alloc(ptr).unwrap(), bf16(host), "{what}");
        }
        assert_eq!(h.q_b, ck.q_b, "one rank: q_b_proj moved whole");
        assert_eq!(h.o_proj, ck.o, "one rank: o_proj moved whole");
    }
}
