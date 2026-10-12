// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: Bit-identity gate: the device K=2 MTP verdict (`mtp_k2_verdict`) and the device-step
//! kernels (`predicated_copy_2d`, `k2_verify_meta`, `k2_step_advance`) against their host
//! specifications (`speculative::k2_device_verdict`, `speculative::device_step`), word for word.
//!
//! Owner: model-arch examples (speculative kernels).
//! Invariants:
//! - The run fails (panics) unless every case gives the specification's five words.
//!
//! Cases: accepts and rejects, ids 0 and the vocabulary's top, equal argmaxes, the
//! `DEVICE_DRAFT` sentinel as a draft, and pseudo-random ids from the GLM-5.3 vocabulary.
//!
//!   cargo run -p metrale-model-arch --release --example mtp_k2_verdict_gate \
//!       --features cuda,gpu-examples

use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_layers::layers::ops::{
    k2_step_advance, k2_verify_meta, mtp_k2_verdict, predicated_copy_2d,
};
use metrale_model_layers::speculative::device_step;
use metrale_model_layers::speculative::{DEVICE_DRAFT, K2_DEVICE_VERDICT_WORDS, k2_device_verdict};

fn words(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|w| w.to_le_bytes()).collect()
}

fn main() -> anyhow::Result<()> {
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &gpu;
    let k = g.kernel("mtp_k2_verdict", "mtp_k2_verdict")?;
    let stream = g.default_stream();
    let (argmax, ids, out) = (
        g.alloc(8)?,
        g.alloc(8)?,
        g.alloc(K2_DEVICE_VERDICT_WORDS * 4)?,
    );
    const V: u32 = 154_856;
    let mut cases: Vec<([u32; 2], u32)> = vec![
        ([7, 9], 7),
        ([7, 9], 9),
        ([0, 0], 0),
        ([0, 5], 1),
        ([V - 1, 3], V - 1),
        ([V - 1, V - 1], V - 1),
        ([12, 34], DEVICE_DRAFT),
    ];
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    for i in 0..4000 {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let a0 = (s % V as u64) as u32;
        let a1 = ((s >> 21) % V as u64) as u32;
        // 2026-10-11: Half the drafts are row 0's argmax, so accepts and rejects both appear.
        let draft = if i % 2 == 0 {
            a0
        } else {
            ((s >> 40) % V as u64) as u32
        };
        cases.push(([a0, a1], draft));
    }
    let mut mismatches = 0usize;
    for (am, draft) in &cases {
        g.copy_h2d(&words(am), argmax)?;
        // 2026-10-11: Row 0's token is not read; row 1 is the draft.
        g.copy_h2d(&words(&[0xDEAD_BEEF, *draft]), ids)?;
        g.memset(out, 0xFF, K2_DEVICE_VERDICT_WORDS * 4)?;
        mtp_k2_verdict(g, k, argmax, ids, out, stream)?;
        g.synchronize(stream)?;
        let mut b = [0u8; K2_DEVICE_VERDICT_WORDS * 4];
        g.copy_d2h(out, &mut b)?;
        let got: Vec<u32> = b
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let want = k2_device_verdict(*am, *draft);
        if got != want {
            mismatches += 1;
            if mismatches <= 5 {
                eprintln!("argmax {am:?} draft {draft}: device {got:?} != host {want:?}");
            }
        }
    }
    for p in [argmax, ids, out] {
        g.free(p)?;
    }
    assert_eq!(
        mismatches,
        0,
        "{mismatches} of {} cases differ",
        cases.len()
    );
    println!(
        "mtp_k2_verdict gate PASS: {} cases, word for word",
        cases.len()
    );
    device_step_kernels(g)?;
    Ok(())
}
fn read(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Vec<u8> {
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b).unwrap();
    b
}

/// 2026-10-11: The device-step kernels against `speculative::device_step`.
fn device_step_kernels(g: &dyn GpuBackend) -> anyhow::Result<()> {
    let stream = g.default_stream();
    let k_copy = g.kernel("mtp_device_step", "predicated_copy_2d")?;
    let k_meta = g.kernel("mtp_device_step", "k2_verify_meta")?;
    let k_adv = g.kernel("mtp_device_step", "k2_step_advance")?;
    let flag = g.alloc(4)?;
    // 2026-10-11: predicated_copy_2d: aligned (vector) and unaligned (byte) shapes, 3 rows with
    // pitches above the width; the copy happens exactly when the flag matches, and nothing outside
    // the rows changes.
    for (width, src_pitch, dst_pitch, src_off, dst_off) in [
        (4096usize, 8192usize, 6144usize, 0usize, 0usize),
        (1000, 1013, 1500, 3, 5),
        (16, 32, 48, 0, 0),
    ] {
        let h = 3usize;
        let src_bytes = src_off + src_pitch * h;
        let dst_bytes = dst_off + dst_pitch * h;
        let src: Vec<u8> = (0..src_bytes).map(|i| (i * 7 + 3) as u8).collect();
        let dst0: Vec<u8> = (0..dst_bytes).map(|i| (i * 13 + 1) as u8).collect();
        let (ds, dd) = (g.alloc(src_bytes)?, g.alloc(dst_bytes)?);
        g.copy_h2d(&src, ds)?;
        for (f, m) in [(0u32, 0u32), (1, 0), (1, 1), (0, 1)] {
            g.copy_h2d(&dst0, dd)?;
            g.copy_h2d(&words(&[f]), flag)?;
            predicated_copy_2d(
                g,
                k_copy,
                flag,
                m,
                ds.offset(src_off),
                src_pitch,
                dd.offset(dst_off),
                dst_pitch,
                width,
                h,
                stream,
            )?;
            g.synchronize(stream)?;
            let got = read(g, dd, dst_bytes);
            let mut want = dst0.clone();
            if device_step::predicated_copy_applies(f, m) {
                for r in 0..h {
                    let s0 = src_off + r * src_pitch;
                    let d0 = dst_off + r * dst_pitch;
                    want[d0..d0 + width].copy_from_slice(&src[s0..s0 + width]);
                }
            }
            assert_eq!(got, want, "copy width {width} flag {f} match {m}");
        }
        g.free(ds)?;
        g.free(dd)?;
    }
    // 2026-10-11: k2_verify_meta over lengths across block boundaries and past the table.
    let table: Vec<u32> = vec![7, 3, 9, 12, 5];
    let (state, bt) = (g.alloc(4)?, g.alloc(table.len() * 4)?);
    let (pos, slots, lens) = (g.alloc(8)?, g.alloc(16)?, g.alloc(8)?);
    g.copy_h2d(&words(&table), bt)?;
    for len in [0usize, 1, 14, 15, 16, 31, 47, 63, 79, 80, 200] {
        g.copy_h2d(&words(&[len as u32]), state)?;
        k2_verify_meta(
            g,
            k_meta,
            state,
            bt,
            table.len() as u32,
            16,
            999,
            pos,
            slots,
            lens,
            stream,
        )?;
        g.synchronize(stream)?;
        let want = device_step::k2_verify_meta(len, 16, 999, |i| table.get(i).copied());
        let want_bytes: Vec<u8> = want
            .positions
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .chain(want.slots.iter().flat_map(|w| w.to_le_bytes()))
            .chain(want.seq_lens.iter().flat_map(|w| w.to_le_bytes()))
            .collect();
        let mut got = read(g, pos, 8);
        got.extend(read(g, slots, 16));
        got.extend(read(g, lens, 8));
        assert_eq!(got, want_bytes, "meta at length {len}");
    }
    // 2026-10-11: k2_step_advance over both verdicts.
    let verdict = g.alloc(4)?;
    for (len, acc) in [(100u32, false), (100, true), (0, true)] {
        g.copy_h2d(&words(&[len]), state)?;
        g.copy_h2d(&words(&[u32::from(acc)]), verdict)?;
        k2_step_advance(g, k_adv, verdict, state, stream)?;
        g.synchronize(stream)?;
        let got = u32::from_le_bytes(read(g, state, 4).try_into().unwrap());
        assert_eq!(
            got as usize,
            device_step::k2_step_advance(len as usize, acc)
        );
    }
    for p in [flag, state, bt, pos, slots, lens, verdict] {
        g.free(p)?;
    }
    println!("device-step kernels gate PASS");
    Ok(())
}
