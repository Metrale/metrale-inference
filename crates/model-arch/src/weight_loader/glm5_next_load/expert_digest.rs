// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-24: `METRALE_GLM_EXPERT_DIGEST=<file>`: a SHA-256 of every routed-expert device
//! buffer a tp-layout load uploads, read back from the device after the upload, so two loads
//! (`METRALE_GLM_EXPERT_FAST_LOAD` on and off) can be proven to place the same bytes.
//!
//! A diagnostic, never a serve setting: each expert costs six blocking device read-backs and
//! hashes. Lines are appended (one file per load; use a fresh path) in bind order and compared
//! after `sort`, one per buffer:
//! `L<layer> E<id> <proj> <part> <bytes> <sha256>`, plus one per projection with the bits of
//! its global and activation scales: `L<layer> E<id> <proj> scales <scale_2 bits> <input bits>`.
//!
//! Owner: model-arch weight loader.
//! Invariants:
//! - The digested bytes are the device's, not the host blobs the upload was given.

use std::io::Write;

use sha2::{Digest, Sha256};

use super::*;

/// 2026-10-24: The digest file of one layer's binder.
pub(super) struct ExpertDigest {
    pub path: std::path::PathBuf,
    pub layer: usize,
}

/// 2026-10-24: The projection and part of each of an expert's six slot buffers, in slot order.
const PARTS: [(&str, &str); 6] = [
    ("gate_proj", "codes"),
    ("gate_proj", "scales"),
    ("up_proj", "codes"),
    ("up_proj", "scales"),
    ("down_proj", "codes"),
    ("down_proj", "scales"),
];

/// 2026-10-24: The digest lines of one expert: `device[i]` holds `sizes[i]` bytes.
pub(super) fn lines(
    layer: usize,
    id: usize,
    device: &[Vec<u8>; 6],
    scalars: &[(f32, Option<f32>); 3],
) -> Vec<String> {
    let mut out: Vec<String> = PARTS
        .iter()
        .zip(device)
        .map(|((p, part), b)| {
            let h: String = Sha256::digest(b)
                .iter()
                .map(|x| format!("{x:02x}"))
                .collect();
            format!("L{layer} E{id} {p} {part} {} {h}", b.len())
        })
        .collect();
    for (p, (s2, a)) in ["gate_proj", "up_proj", "down_proj"].iter().zip(scalars) {
        let a = a.map_or("none".to_string(), |v| format!("{:08x}", v.to_bits()));
        out.push(format!(
            "L{layer} E{id} {p} scales {:08x} {a}",
            s2.to_bits()
        ));
    }
    out
}

impl ExpertDigest {
    /// 2026-10-24: Read expert `id`'s six buffers back from `ptrs` (each of its `blobs` length)
    /// and append their lines to the file.
    pub(super) fn record(
        &self,
        gpu: &dyn GpuBackend,
        id: usize,
        blobs: &[&[u8]; 6],
        ptrs: &[DevicePtr; 6],
        scalars: &[(f32, Option<f32>); 3],
    ) -> Result<()> {
        let mut device: [Vec<u8>; 6] = Default::default();
        for ((d, b), p) in device.iter_mut().zip(blobs).zip(ptrs) {
            *d = vec![0u8; b.len()];
            gpu.copy_d2h(*p, d)?;
        }
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .with_context(|| {
                format!("METRALE_GLM_EXPERT_DIGEST: opening {}", self.path.display())
            })?;
        let text = lines(self.layer, id, &device, scalars).join("\n") + "\n";
        f.write_all(text.as_bytes())
            .with_context(|| format!("METRALE_GLM_EXPERT_DIGEST: writing {}", self.path.display()))
    }
}
