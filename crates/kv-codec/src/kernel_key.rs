// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: Typed kernel ids for the KV path (TIDE) and the kernel set resolved from them
//! once at startup (ROTI). The set is generic over the backend's handle type, so this core
//! crate names no device type.
//!
//! Owner: kv-codec.
//! Invariants:
//! - A [`KvKernelKey`] names one kernel entry point: an operation on one codec pair at one
//!   head-dim class. A missing optional kernel is `None` in [`ResolvedKvKernels`], never another
//!   codec's kernel.

use crate::KvCodecId;
use crate::spec::fnv_words;

/// 2026-10-18: A KV-path kernel operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum KvOp {
    /// 2026-10-18: Quantize and write new K/V rows into their blocks.
    Write = 0,
    /// 2026-10-18: Paged decode attention.
    DecodeAttn = 1,
    /// 2026-10-18: Paged decode attention specialized for head_dim 512.
    DecodeAttn512 = 2,
    /// 2026-10-18: The split-K partial pass of paged decode attention.
    DecodeSplitK = 3,
    /// 2026-10-18: The reduction after `DecodeSplitK`.
    DecodeReduce = 4,
    /// 2026-10-18: Paged prefill attention.
    PrefillAttn = 5,
    /// 2026-10-18: Paged prefill attention, 64-row tile.
    PrefillAttn64 = 6,
    /// 2026-10-18: The forward WHT bookend.
    WhtFwd = 7,
    /// 2026-10-18: The inverse WHT bookend.
    WhtInv = 8,
    /// 2026-10-18: Re-quantize resident blocks between two codecs.
    Requant = 9,
    /// 2026-10-18: Dequantize blocks to bf16.
    DequantBf16 = 10,
}

/// 2026-10-18: The head-dim specialization a kernel was compiled for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum HeadDimClass {
    /// 2026-10-18: The model's own head dim (the kernel reads it at run time or from `HDIM`).
    Model = 0,
    /// 2026-10-18: A kernel specialized for head_dim 128.
    Hd128 = 1,
    /// 2026-10-18: A kernel specialized for head_dim 512.
    Hd512 = 2,
}

/// 2026-10-18: One kernel entry point.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct KvKernelKey {
    /// 2026-10-18: The operation.
    pub op: KvOp,
    /// 2026-10-18: The K-side codec (the only codec of a symmetric kernel).
    pub k: KvCodecId,
    /// 2026-10-18: The V-side codec of a fused asymmetric kernel; `None` when symmetric.
    pub v: Option<KvCodecId>,
    /// 2026-10-18: The head-dim specialization.
    pub hd: HeadDimClass,
}

impl KvKernelKey {
    /// 2026-10-18: A stable 64-bit identity of the key.
    pub const fn digest(&self) -> u64 {
        let v = match self.v {
            Some(v) => v as u64 + 1,
            None => 0,
        };
        fnv_words(&[self.op as u64, self.k as u64, v, self.hd as u64])
    }
}

/// 2026-10-18: The kernels one layer class needs, resolved once. `H` is the backend's kernel
/// handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResolvedKvKernels<H: Copy> {
    /// 2026-10-18: The cache write (both sides; a fused pair writes in one launch).
    pub write: H,
    /// 2026-10-18: Decode attention.
    pub decode: H,
    /// 2026-10-18: The head_dim-512 decode kernel, when the codec pair has one.
    pub decode_512: Option<H>,
    /// 2026-10-18: The split-K pass and its reduction, when the codec pair has them.
    pub splitk: Option<(H, H)>,
    /// 2026-10-18: Prefill attention, when the codec pair has it.
    pub prefill: Option<H>,
    /// 2026-10-18: The WHT forward and inverse bookends, when either side is rotated.
    pub wht: Option<(H, H)>,
}

/// 2026-10-18: Every resolved kernel of a serve, one entry per layer class.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KvKernelSet<H: Copy> {
    /// 2026-10-18: The resolved kernels.
    pub handles: alloc::boxed::Box<[ResolvedKvKernels<H>]>,
}

impl<H: Copy> ResolvedKvKernels<H> {
    /// 2026-10-18: The same kernels with each handle turned into another by `f`, in field order;
    /// the first failure is returned and `f` is not called again.
    pub fn try_map<G: Copy, E>(
        self,
        mut f: impl FnMut(H) -> Result<G, E>,
    ) -> Result<ResolvedKvKernels<G>, E> {
        let write = f(self.write)?;
        let decode = f(self.decode)?;
        let decode_512 = match self.decode_512 {
            Some(h) => Some(f(h)?),
            None => None,
        };
        let splitk = match self.splitk {
            Some((a, b)) => Some((f(a)?, f(b)?)),
            None => None,
        };
        let prefill = match self.prefill {
            Some(h) => Some(f(h)?),
            None => None,
        };
        let wht = match self.wht {
            Some((a, b)) => Some((f(a)?, f(b)?)),
            None => None,
        };
        Ok(ResolvedKvKernels {
            write,
            decode,
            decode_512,
            splitk,
            prefill,
            wht,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::ResolvedKvKernels;

    fn all(base: u32) -> ResolvedKvKernels<u32> {
        ResolvedKvKernels {
            write: base + 1,
            decode: base + 2,
            decode_512: Some(base + 3),
            splitk: Some((base + 4, base + 5)),
            prefill: Some(base + 6),
            wht: Some((base + 7, base + 8)),
        }
    }

    /// 2026-10-18: Every field maps to its own image, in field order, and `None` stays `None`.
    #[test]
    fn try_map_maps_each_field_to_its_own_image_in_order() {
        let mut seen = alloc::vec::Vec::new();
        let got = all(0).try_map(|h| -> Result<u32, ()> {
            seen.push(h);
            Ok(h + 100)
        });
        assert_eq!(got, Ok(all(100)));
        assert_eq!(seen, [1, 2, 3, 4, 5, 6, 7, 8]);
        let bare = ResolvedKvKernels {
            decode_512: None,
            splitk: None,
            prefill: None,
            wht: None,
            ..all(0)
        };
        let mapped = bare.try_map(|h| Ok::<u32, ()>(h + 100)).unwrap();
        assert_eq!(
            (mapped.decode_512, mapped.splitk, mapped.prefill, mapped.wht),
            (None, None, None, None)
        );
    }

    /// 2026-10-18: The first failure is returned and nothing after it is mapped.
    #[test]
    fn try_map_stops_at_the_first_failure() {
        let mut seen = alloc::vec::Vec::new();
        let got = all(0).try_map(|h| {
            seen.push(h);
            if h == 5 { Err(h) } else { Ok(h) }
        });
        assert_eq!(got, Err(5));
        assert_eq!(seen, [1, 2, 3, 4, 5]);
    }
}
