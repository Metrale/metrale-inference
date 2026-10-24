// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-10-19: The cache rows: their kernels and sizes at the checkpoint's head geometry, the
//! byte layout the host decode reads (the oracle the device rows are checked against), and, on
//! a device, the packing kernels against the reference's in-place quantisers bit for bit.
//!
//! Owner: model-arch, DeepSeek-V4.1.
//! Invariants: none beyond the types.

use super::*;

fn rows(head_dim: usize, index_hd: usize) -> Result<V41Rows> {
    V41Rows::resolve(&declared()?, head_dim, index_hd)
}

#[test]
fn declared_rows_at_the_checkpoint_geometry() {
    let r = rows(512, 128).unwrap();
    assert_eq!(
        [r.window, r.latent, r.index_key].map(|x| (x.codec.name(), x.pack, x.bytes)),
        [
            ("mxfp8", RowPack::Fp8E8m0, 512 + 16),
            (
                "nvfp4",
                RowPack::Fp4 {
                    group: 16,
                    e4m3_scale: true
                },
                256 + 32
            ),
            (
                "mxfp4",
                RowPack::Fp4 {
                    group: 32,
                    e4m3_scale: false
                },
                64 + 4
            ),
        ]
    );
}

#[test]
fn rows_that_split_a_group_are_refused() {
    assert!(rows(48, 128).is_err(), "48 window values split a 32-group");
    assert!(rows(512, 48).is_err(), "48 index values split a 32-group");
    assert!(rows(0, 128).is_err(), "an empty row");
}

/// 2026-10-19: Only the unrotated block-scaled codecs a kernel here packs resolve; the plan
/// refuses the rest before a row is sized.
#[test]
fn a_codec_without_a_packing_kernel_is_refused() {
    let (w, l, k) = (codec("mxfp8"), codec("nvfp4"), codec("mxfp4"));
    for name in ["bf16", "fp8", "turbo4", "turbo8", "turbo3_tcq", "fp8_row"] {
        let c = codec(name);
        assert!(LatentKvPlan::new(c, l, k).is_err(), "{name} as window");
        assert!(LatentKvPlan::new(w, l, c).is_err(), "{name} as index key");
    }
}

#[test]
fn e2m1_rows_read_the_low_nibble_first_and_keep_negative_zero() {
    // 2026-10-19: 32 values, one e8m0 scale 2^1 (byte 128). Byte 0 = 0x21: value 0 is code 1
    // (0.5), value 1 is code 2 (1.0). Byte 1 = 0x8F: value 2 is code F (-6), value 3 is code 8.
    let mut row = vec![0u8; 16 + 1];
    row[0] = 0x21;
    row[1] = 0x8F;
    row[16] = 128;
    let v = decode_rows(codec("mxfp4"), &row, 1, 32).unwrap();
    assert_eq!(&v[..3], &[1.0, 2.0, -12.0]);
    assert!(v[3] == 0.0 && v[3].is_sign_negative(), "code 8 is -0");
    assert!(v[4..].iter().all(|x| x.to_bits() == 0));
}

#[test]
fn each_group_reads_its_own_scale_after_the_codes() {
    // 2026-10-19: Latent row of 32: codes 0x22 everywhere (1.0, 1.0), e4m3 scales 0x38 (1.0)
    // then 0x40 (2.0): values 0..16 are 1, values 16..32 are 2.
    let mut row = vec![0x22u8; 16];
    row.extend([0x38, 0x40]);
    let v = decode_rows(codec("nvfp4"), &row, 1, 32).unwrap();
    assert!(v[..16].iter().all(|&x| x == 1.0) && v[16..].iter().all(|&x| x == 2.0));
    // 2026-10-19: Two window rows of 32: e4m3 code 0x40 (2.0) and e8m0 scales 2^1, then 2^-127
    // (byte 0, the smallest scale, not zero).
    let mut rows = vec![0x40u8; 32];
    rows.push(128);
    rows.extend(vec![0x40u8; 32]);
    rows.push(0);
    let v = decode_rows(codec("mxfp8"), &rows, 2, 32).unwrap();
    assert!(v[..32].iter().all(|&x| x == 4.0));
    assert!(v[32..].iter().all(|&x| x == 2f32.powi(-126)));
}

#[test]
fn a_short_buffer_and_a_codec_without_a_host_decode_are_refused() {
    let w = codec("mxfp8");
    assert!(decode_rows(w, &[0u8; 32], 1, 32).is_err());
    assert!(decode_rows(w, &[0u8; 33], 1, 32).is_ok());
    assert!(decode_rows(codec("turbo2"), &[0u8; 64], 1, 32).is_err());
}

#[cfg(feature = "cuda")]
mod device {
    use super::*;
    use crate::attn_v41::AttnV41;
    use crate::deepseek_v41_layer::levers::tests::at_defaults;
    use crate::deepseek_v41_ref::attn::act_quant_inplace;
    use crate::deepseek_v41_ref::compress::{fp4_quant_e4m3_inplace, fp4_quant_e8m0_inplace};
    use crate::deepseek_v41_ref::to_bf16_rne;
    use metrale_gpu_runtime::gpu::GpuBackend;

    /// 2026-10-19: bf16 values over 40 binades, with exact zeros, negative zeros, an all-zero
    /// block and values on e2m1/e4m3 rounding ties (the deterministic stream is the test's own;
    /// any fixed seed works).
    fn inputs(n: usize) -> Vec<f32> {
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        (0..n)
            .map(|i| {
                s = s
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let u = (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
                let binade = ((i / 32) % 40) as i32 - 20;
                match i % 97 {
                    _ if (64..96).contains(&i) => 0.0,
                    3 => -0.0,
                    5 => 0.75 * 2f32.powi(binade),
                    7 => -2.5 * 2f32.powi(binade),
                    _ => to_bf16_rne(u * 2f32.powi(binade)),
                }
            })
            .collect()
    }

    fn pack(
        g: &dyn GpuBackend,
        attn: &AttnV41,
        row: &V41Row,
        x: &[f32],
        rows: usize,
        values: usize,
    ) -> Vec<u8> {
        let bytes: Vec<u8> = x
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let src = g.alloc(bytes.len()).unwrap();
        g.copy_h2d(&bytes, src).unwrap();
        let len = rows * row.bytes;
        let out = g.alloc(len).unwrap();
        let stream = g.default_stream();
        attn.pack_rows(g, row, src, rows, values, out, stream)
            .unwrap();
        g.synchronize(stream).unwrap();
        let mut packed = vec![0u8; len];
        g.copy_d2h(out, &mut packed).unwrap();
        g.free(src).unwrap();
        g.free(out).unwrap();
        packed
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    #[ignore = "requires a CUDA GB10 + the deepseek-v4-flash kernel target"]
    fn packed_rows_decode_to_the_reference_quantisation() {
        let g = crate::attn_v41::tests::backend();
        let attn = AttnV41::new(&g, crate::attn_v41::tp_tests::cfg(1), at_defaults()).unwrap();
        let (n_rows, hd, ihd) = (12, 512, 128);
        let r = rows(hd, ihd).unwrap();
        let x = inputs(n_rows * hd);
        type Quant = fn(&mut [f32]);
        let latent: Quant = |v| fp4_quant_e4m3_inplace(v, 16);
        let index: Quant = |v| fp4_quant_e8m0_inplace(v, 32);
        let cases: [(&V41Row, usize, Quant); 3] = [
            (&r.window, hd, act_quant_inplace),
            (&r.latent, hd, latent),
            (&r.index_key, ihd, index),
        ];
        for (row, values, quant) in cases {
            let n = n_rows * hd / values;
            let mut want = x.clone();
            quant(&mut want);
            let packed = pack(&g, &attn, row, &x, n, values);
            let got = decode_rows(row.codec, &packed, n, values).unwrap();
            let name = row.codec.name();
            assert_eq!(bits(&got), bits(&want), "{name}");
            // 2026-10-19: Control: the same bytes with their scale bytes read as the other
            // element disagree, so the comparison sees which scale the kernel wrote.
            let s = spec(row.codec);
            let (value, scale) = lut_elems(s.host_dequant).unwrap();
            let other = match scale {
                Elem::E8m0 => Elem::E4m3,
                Elem::E4m3 | Elem::E2m1 => Elem::E8m0,
            };
            let data = s.data_bytes(values as u64) as usize;
            let group = s.group as usize;
            let wrong = decode(&packed, n, values, group, data, row.bytes, (value, other)).unwrap();
            assert_ne!(bits(&wrong), bits(&want), "{name} control");
        }
        attn.free(&g).unwrap();
    }
}
