// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: One launch's host operands in canonical layouts. References decode them exactly;
//! GPU adapters convert them to each kernel's layout; mutations edit them. Canonical layouts:
//! row-major, an E2M1 tensor packs two values per byte with the even index in the low nibble,
//! a scale tensor is row-major over (rows, groups).
//!
//! Owner: metrale-accuracy.
//! Invariants:
//! - A tensor's bytes always decode under its format (lengths are checked at construction).
//! - Decoding never masks NaN (see [`crate::elem`]).

use std::collections::BTreeMap;

use crate::elem::{self, Elem};

/// 2026-10-09: The element encoding of a host tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Enc {
    /// 2026-10-09: bfloat16, 2 bytes LE.
    Bf16,
    /// 2026-10-09: f32, 4 bytes LE.
    F32,
    /// 2026-10-09: i32, 4 bytes LE.
    I32,
    /// 2026-10-09: FP8 E4M3, 1 byte.
    E4m3,
    /// 2026-10-09: FP4 E2M1, two per byte (even index in the low nibble).
    E2m1x2,
    /// 2026-10-09: Unsigned E4M3 scale, 1 byte.
    Ue4m3,
    /// 2026-10-09: UE8M0 scale, 1 byte.
    Ue8m0,
}

impl Enc {
    /// 2026-10-09: Bytes holding `n` elements.
    pub fn bytes_for(self, n: usize) -> usize {
        match self {
            Enc::Bf16 => 2 * n,
            Enc::F32 | Enc::I32 => 4 * n,
            Enc::E4m3 | Enc::Ue4m3 | Enc::Ue8m0 => n,
            Enc::E2m1x2 => n.div_ceil(2),
        }
    }

    /// 2026-10-09: The rounding model of a value of this encoding (`None` for i32).
    pub fn elem(self) -> Option<Elem> {
        match self {
            Enc::Bf16 => Some(elem::BF16),
            Enc::F32 => Some(elem::F32),
            Enc::I32 => None,
            Enc::E4m3 => Some(elem::E4M3),
            Enc::E2m1x2 => Some(elem::E2M1),
            Enc::Ue4m3 => Some(elem::UE4M3),
            Enc::Ue8m0 => Some(elem::UE8M0),
        }
    }
}

/// 2026-10-09: A host tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    /// 2026-10-09: Encoding.
    pub enc: Enc,
    /// 2026-10-09: Dimensions, outermost first.
    pub dims: Vec<usize>,
    /// 2026-10-09: Bytes.
    /// (shared: a mutation copies only the tensor it edits, `Arc::make_mut`).
    pub bytes: std::sync::Arc<Vec<u8>>,
}

impl Tensor {
    /// 2026-10-09: Encode `values` (each already a value of the encoding, else an error).
    pub fn encode(enc: Enc, dims: Vec<usize>, values: &[f64]) -> Result<Tensor, String> {
        let n: usize = dims.iter().product();
        if values.len() != n {
            return Err(format!("{} values for dims {dims:?}", values.len()));
        }
        let mut bytes = Vec::with_capacity(enc.bytes_for(n));
        let not = |v: f64| format!("{v} is not a {enc:?} value");
        match enc {
            Enc::Bf16 => {
                for &v in values {
                    bytes.extend_from_slice(
                        &elem::f64_to_bf16(v).ok_or_else(|| not(v))?.to_le_bytes(),
                    );
                }
            }
            Enc::F32 => {
                for &v in values {
                    if f64::from(v as f32) != v {
                        return Err(not(v));
                    }
                    bytes.extend_from_slice(&(v as f32).to_le_bytes());
                }
            }
            Enc::I32 => {
                for &v in values {
                    if v.fract() != 0.0 || v.abs() > f64::from(i32::MAX) {
                        return Err(not(v));
                    }
                    bytes.extend_from_slice(&(v as i32).to_le_bytes());
                }
            }
            Enc::E4m3 | Enc::Ue4m3 => {
                for &v in values {
                    if enc == Enc::Ue4m3 && v < 0.0 {
                        return Err(not(v));
                    }
                    bytes.push(elem::f64_to_e4m3(v).ok_or_else(|| not(v))?);
                }
            }
            Enc::Ue8m0 => {
                for &v in values {
                    let e = v.log2();
                    if v <= 0.0 || e.fract() != 0.0 || !(-127.0..=127.0).contains(&e) {
                        return Err(not(v));
                    }
                    bytes.push((e as i32 + 127) as u8);
                }
            }
            Enc::E2m1x2 => {
                for pair in values.chunks(2) {
                    let lo = elem::f64_to_e2m1(pair[0]).ok_or_else(|| not(pair[0]))?;
                    let hi = match pair.get(1) {
                        Some(&v) => elem::f64_to_e2m1(v).ok_or_else(|| not(v))?,
                        None => 0,
                    };
                    bytes.push(lo | (hi << 4));
                }
            }
        }
        Ok(Tensor {
            enc,
            dims,
            bytes: std::sync::Arc::new(bytes),
        })
    }

    /// 2026-10-09: Element count.
    pub fn len(&self) -> usize {
        self.dims.iter().product()
    }

    /// 2026-10-09: No elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 2026-10-09: Element `i`, exactly.
    pub fn get(&self, i: usize) -> f64 {
        let b = &self.bytes;
        match self.enc {
            Enc::Bf16 => elem::bf16_to_f64(u16::from_le_bytes([b[2 * i], b[2 * i + 1]])),
            Enc::F32 => f64::from(f32::from_le_bytes([
                b[4 * i],
                b[4 * i + 1],
                b[4 * i + 2],
                b[4 * i + 3],
            ])),
            Enc::I32 => f64::from(i32::from_le_bytes([
                b[4 * i],
                b[4 * i + 1],
                b[4 * i + 2],
                b[4 * i + 3],
            ])),
            Enc::E4m3 => elem::e4m3_to_f64(b[i]),
            Enc::Ue4m3 => elem::ue4m3_to_f64(b[i]),
            Enc::Ue8m0 => elem::ue8m0_to_f64(b[i]),
            Enc::E2m1x2 => elem::e2m1_to_f64(if i.is_multiple_of(2) {
                b[i / 2] & 0xf
            } else {
                b[i / 2] >> 4
            }),
        }
    }

    /// 2026-10-09: Every element, exactly.
    pub fn values(&self) -> Vec<f64> {
        (0..self.len()).map(|i| self.get(i)).collect()
    }
}

/// 2026-10-09: One launch: its operands, scalars and output.
#[derive(Debug, Clone, PartialEq)]
pub struct Case {
    /// 2026-10-09: Family.
    pub family: String,
    /// 2026-10-09: The entry point launched (`module::function`).
    pub kernel: String,
    /// 2026-10-09: The entry point whose launcher runs it (the contract's kernel; differs from
    /// `kernel` only in a wrong-symbol mutation, which keeps the launcher and swaps the symbol).
    pub launcher: String,
    /// 2026-10-09: Op key of the family pipeline.
    pub op: String,
    /// 2026-10-09: Input tensors by name (the reference's canonical names).
    pub tensors: BTreeMap<String, Tensor>,
    /// 2026-10-09: Runtime scalars by name (eps, theta, softmax scale, global scales).
    pub scalars: BTreeMap<String, f64>,
    /// 2026-10-09: Output dims (rows, cols) and encoding.
    pub out: (Vec<usize>, Enc),
    /// 2026-10-09: Shards launched separately (tensor-parallel); empty is one launch.
    pub split: Vec<Shard>,
}

/// 2026-10-09: One shard of a split launch: the weight rows (output columns as computed) it
/// runs, and the first output column it writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shard {
    /// 2026-10-09: First weight row.
    pub lo: usize,
    /// 2026-10-09: One past the last weight row.
    pub hi: usize,
    /// 2026-10-09: Output column of row `lo`.
    pub out_at: usize,
}

impl Case {
    /// 2026-10-09: Tensor `name`; an error names it.
    pub fn tensor(&self, name: &str) -> Result<&Tensor, String> {
        self.tensors
            .get(name)
            .ok_or_else(|| format!("case has no tensor `{name}`"))
    }

    /// 2026-10-09: Scalar `name`; an error names it.
    pub fn scalar(&self, name: &str) -> Result<f64, String> {
        self.scalars
            .get(name)
            .copied()
            .ok_or_else(|| format!("case has no scalar `{name}`"))
    }
}

#[cfg(test)]
#[path = "case_tests.rs"]
mod tests;
