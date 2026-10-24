// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-10-19: The checkpoint builder. The end-to-end test serves the golden's parameters as
//! checkpoint tensors (bf16 and f32 bytes, the Engram table as e4m3 codes with E8M0 scales)
//! and requires the forward to match the golden-built forward bit for bit; the rest pin the
//! dtype routing and every refusal on hand-made tensors.
//!
//! Owner: model-arch, DeepSeek-V4.1 reference.
//! Invariants: none beyond the types.

use std::collections::HashMap;

use super::super::engram::{EngramTables, f32_to_e4m3_rne};
use super::super::mat::{InputQuant, Weight, e4m3};
use super::super::model::{ModelCfg, ModelState, ModelWeights, forward};
use super::super::{Golden, fixed_int, fixed_value, regen_param};
use super::*;

#[derive(Default)]
struct MapSource(HashMap<String, (RawDtype, Vec<u64>, Vec<u8>)>);

impl MapSource {
    fn put(&mut self, name: &str, dtype: RawDtype, shape: &[u64], bytes: Vec<u8>) {
        self.0
            .insert(name.to_string(), (dtype, shape.to_vec(), bytes));
    }
}

impl TensorSource for MapSource {
    fn tensor(&self, name: &str) -> Result<RawTensor<'_>, Error> {
        let (dtype, shape, bytes) = self
            .0
            .get(name)
            .ok_or_else(|| refuse(name, "absent".into()))?;
        Ok(RawTensor {
            dtype: *dtype,
            shape: shape.clone(),
            bytes,
        })
    }
}

/// 2026-10-19: The golden's parameters as checkpoint tensors in their declared dtypes.
fn golden_source(g: &Golden, t: &EngramTables) -> MapSource {
    let mut src = MapSource::default();
    let metas = g.0["weights_meta"].as_object().expect("weights_meta");
    for (name, m) in metas {
        let shape: Vec<u64> = m["shape"]
            .as_array()
            .expect("shape")
            .iter()
            .map(|d| d.as_u64().expect("dim"))
            .collect();
        let n: u64 = shape.iter().product();
        let engram_layer = || {
            let l: usize = name.split('.').nth(1).unwrap().parse().unwrap();
            t.layer_ids
                .iter()
                .position(|&x| x == l)
                .expect("engram layer")
        };
        let (dtype, bytes) = match (m["kind"].as_str().unwrap(), m["dtype"].as_str().unwrap()) {
            ("f32", "bfloat16") => {
                let v = regen_param(g, name);
                assert!(
                    v.iter().all(|x| x.to_bits() & 0xFFFF == 0),
                    "{name}: bf16-exact"
                );
                let b = v
                    .iter()
                    .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes());
                (RawDtype::Bf16, b.collect())
            }
            ("f32", "float32") => {
                let b = regen_param(g, name).into_iter().flat_map(f32::to_le_bytes);
                (RawDtype::F32, b.collect())
            }
            ("fp8", _) => {
                let codes = (0..n).map(|i| f32_to_e4m3_rne(fixed_value(name, i, 0.5, 0.0)));
                (RawDtype::F8E4m3, codes.collect())
            }
            ("e8m0", _) => {
                let s = &t.scales[engram_layer()];
                assert_eq!(s.len() as u64, n, "{name}");
                let b = s.iter().map(|x| {
                    assert_eq!(
                        x.to_bits() & 0x807F_FFFF,
                        0,
                        "{name}: a positive power of two"
                    );
                    (x.to_bits() >> 23) as u8
                });
                (RawDtype::F8E8m0, b.collect())
            }
            other => panic!("{name}: unexpected {other:?}"),
        };
        src.put(name, dtype, &shape, bytes);
    }
    src
}

#[test]
fn golden_tensors_through_the_checkpoint_give_the_golden_forward_bit_for_bit() {
    let g = Golden::load();
    let c = ModelCfg::from_golden(&g);
    let t = EngramTables::from_golden(&g);
    let src = golden_source(&g, &t);
    let ck = model_weights(&src, &c, &t).expect("golden tensors form a checkpoint");
    let gold = ModelWeights::from_golden(&g, &c, &t);
    assert!(ck.layers[2].comp_wgate.is_some() && ck.layers[4].comp_wgate.is_none());
    assert!(ck.layers[5].idx_wq_b.is_some() && ck.layers[5].idx_wk.is_none());
    assert!(ck.layers[1].engram.is_some() && ck.layers[0].engram.is_none());
    let prefill = g.fixture_u64("prefill_len") as usize;
    let ids: Vec<i64> = (0..(prefill + 2) as u64)
        .map(|i| fixed_int("input_ids", i, c.vocab as u64) as i64)
        .collect();
    let (mut sa, mut sb) = (ModelState::new(&c, &t), ModelState::new(&c, &t));
    for (lo, hi) in [
        (0, prefill),
        (prefill, prefill + 1),
        (prefill + 1, prefill + 2),
    ] {
        let a = forward(&ids[lo..hi], lo, &c, &ck, &mut sa, &t);
        let b = forward(&ids[lo..hi], lo, &c, &gold, &mut sb, &t);
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&a.logits), bits(&b.logits), "logits at start {lo}");
        assert_eq!(bits(&a.h_final), bits(&b.h_final), "h_final at start {lo}");
    }
}

fn e8m0_byte(exp: i32) -> u8 {
    (exp + 127) as u8
}

/// 2026-10-19: A `[64, 64]` e4m3 weight with a distinct scale per 32x32 tile.
fn e4m3_source() -> MapSource {
    let mut s = MapSource::default();
    let codes: Vec<u8> = (0..64 * 64).map(|i| (i % 0x7E) as u8).collect();
    s.put("w.weight", RawDtype::F8E4m3, &[64, 64], codes);
    let scales = vec![e8m0_byte(-1), e8m0_byte(0), e8m0_byte(1), e8m0_byte(2)];
    s.put("w.scale", RawDtype::F8E8m0, &[2, 2], scales);
    s
}

#[test]
fn e4m3_tiles_take_their_own_scale_and_the_input_follows_the_site() {
    let s = e4m3_source();
    let m = matrix(&s, "w", 64, 64, MX_BLOCK, Site::Linear).unwrap();
    assert_eq!(m.input_quant(), InputQuant::Fp8Block);
    let mut row = vec![0f32; 64];
    for (r, scales) in [(0usize, [0.5f32, 1.0]), (33, [2.0, 4.0])] {
        m.decode_row(r, 64, &mut row);
        for (j, &v) in row.iter().enumerate() {
            let code = ((r * 64 + j) % 0x7E) as u8;
            assert_eq!(v, e4m3(code) * scales[j / 32], "row {r} col {j}");
        }
    }
    let w = matrix(&s, "w", 64, 64, MX_BLOCK, Site::Widened).unwrap();
    assert_eq!(w.input_quant(), InputQuant::Exact);
}

#[test]
fn fp4_is_two_codes_per_byte_with_a_scale_per_32_and_quantizes_its_input() {
    let mut s = MapSource::default();
    // 2026-10-19: low nibble 1 (0.5), high nibble 0xA (-1.0).
    s.put("w.weight", RawDtype::I8, &[2, 32], vec![0xA1; 64]);
    s.put(
        "w.scale",
        RawDtype::F8E8m0,
        &[2, 2],
        vec![e8m0_byte(0), e8m0_byte(1), e8m0_byte(2), e8m0_byte(3)],
    );
    let m = matrix(&s, "w", 2, 64, MX_BLOCK, Site::Linear).unwrap();
    assert_eq!(m.input_quant(), InputQuant::Fp8Block);
    let mut row = vec![0f32; 64];
    m.decode_row(1, 64, &mut row);
    assert_eq!((row[0], row[1], row[32], row[33]), (2.0, -4.0, 4.0, -8.0));
}

#[test]
fn bf16_and_f32_weights_take_exact_input_at_any_site() {
    let mut s = MapSource::default();
    s.put(
        "b.weight",
        RawDtype::Bf16,
        &[1, 2],
        vec![0x80, 0x3F, 0x00, 0xC0],
    );
    s.put(
        "f.weight",
        RawDtype::F32,
        &[1, 2],
        [1.5f32, -0.25]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect(),
    );
    let b = matrix(&s, "b", 1, 2, MX_BLOCK, Site::Linear).unwrap();
    let f = matrix(&s, "f", 1, 2, MX_BLOCK, Site::Linear).unwrap();
    let mut row = [0f32; 2];
    assert_eq!(b.input_quant(), InputQuant::Exact);
    b.decode_row(0, 2, &mut row);
    assert_eq!(row, [1.0, -2.0]);
    assert_eq!(f.input_quant(), InputQuant::Exact);
    f.decode_row(0, 2, &mut row);
    assert_eq!(row, [1.5, -0.25]);
}

#[track_caller]
fn refused<T>(r: Result<T, Error>, needle: &str) {
    let Err(e) = r else {
        panic!("must refuse ({needle})")
    };
    assert_eq!(e.code(), ErrorCode::InvalidArgument);
    let msg = e.to_string();
    assert!(msg.contains(needle), "{msg} lacks {needle}");
}

#[test]
fn malformed_checkpoint_tensors_are_refused() {
    let lin = |s: &MapSource| matrix(s, "w", 64, 64, MX_BLOCK, Site::Linear).map(|_| ());
    lin(&e4m3_source()).expect("the control is accepted");
    let mut s = e4m3_source();
    s.0.remove("w.scale");
    refused(lin(&s), "w.scale: absent");
    let mut s = e4m3_source();
    s.put("w.scale", RawDtype::F8E8m0, &[4, 1], vec![127; 4]);
    refused(lin(&s), "shape [4, 1], expected [2, 2]");
    let mut s = e4m3_source();
    s.put("w.scale", RawDtype::F32, &[2, 2], vec![0; 16]);
    refused(lin(&s), "where an E8M0 scale is expected");
    let mut s = e4m3_source();
    s.put(
        "w.scale",
        RawDtype::F8E8m0,
        &[2, 2],
        vec![127, 0xFF, 127, 127],
    );
    refused(lin(&s), "NaN code");
    let mut s = e4m3_source();
    s.put("w.weight", RawDtype::F8E4m3, &[64, 32], vec![0; 64 * 32]);
    refused(lin(&s), "shape [64, 32], expected [64, 64]");
    let mut s = e4m3_source();
    s.put("w.weight", RawDtype::I8, &[64, 64], vec![0; 64 * 64]);
    refused(lin(&s), "shape [64, 64], expected [64, 32]");
    let mut s = e4m3_source();
    s.put("w.weight", RawDtype::F8E8m0, &[64, 64], vec![0; 64 * 64]);
    refused(lin(&s), "is not a weight dtype");

    let mut v = MapSource::default();
    v.put("v", RawDtype::F32, &[3], vec![0; 12]);
    vector(&v, "v", 3).expect("the control is accepted");
    refused(vector(&v, "v", 4), "3 elements, expected 4");
    v.put("v", RawDtype::F32, &[3], vec![0; 8]);
    refused(vector(&v, "v", 3), "8 bytes for 3 elements");
    v.put("v", RawDtype::F8E4m3, &[3], vec![0; 3]);
    refused(vector(&v, "v", 3), "where bf16 or f32 is expected");
}
