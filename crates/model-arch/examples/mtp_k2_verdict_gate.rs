// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: Bit-identity gate: the device K=2 MTP verdict (`mtp_k2_verdict`) against its host
//! specification (`speculative::k2_device_verdict`), word for word.
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
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_layers::layers::ops::mtp_k2_verdict;
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
    Ok(())
}
