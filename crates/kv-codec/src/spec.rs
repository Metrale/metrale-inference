// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: The description of one KV-cache codec ([`KvCodecSpec`]) and the layout math
//! every byte count in the engine is derived from.
//!
//! Owner: kv-codec.
//! Invariants:
//! - `data_bytes`, `side_bytes` and `block_bytes` are pure functions of the spec fields; no
//!   caller re-derives a size from a codec's name (SSOT).
//! - A byte-sized side element of `per_values` values contributes
//!   `(values / per_values) * elem_bytes`, with each division flooring exactly as the
//!   pre-registry formulas did; a sub-byte element rounds the packed total up.
//! - `digest` hashes only layout fields, so two codecs with the same byte layout but
//!   different codebooks share a digest; the spill header pairs it with the id.

use crate::KvCodecId;

/// 2026-10-18: The launch-argument family of a codec. Device code dispatches on this, never on
/// the codec id, so a new codec that fits an existing family needs no new executor code. It is
/// closed on purpose: a new family is a design change, and every match over it has no wildcard.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum AbiId {
    /// 2026-10-18: Values only (bf16).
    Dense = 0,
    /// 2026-10-18: Values plus a per-tensor or per-head scale held outside the block (fp8).
    DenseScaled = 1,
    /// 2026-10-18: Values plus an in-block side tensor per group (nvfp4, turbo4/3/2/8).
    BlockScaled = 2,
    /// 2026-10-18: Self-contained 128-value groups rotated and decoded inside the codec's own
    /// kernels against a device codebook (turbo1/2/3_tcq, turbo4_n128). Each group is one record
    /// (norm, codes, trellis start state), so the side tensors give byte counts, not plane
    /// offsets, and no host LUT reads it.
    Trellis = 3,
    /// 2026-10-18: A compressed latent ring (MLA-style). Declared; no row uses it yet.
    Latent = 4,
}

/// 2026-10-18: What a side tensor holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SideKind {
    /// 2026-10-18: A multiplicative group scale.
    Scale = 0,
    /// 2026-10-18: A group norm (turbo8's bf16 per 16).
    Norm = 1,
    /// 2026-10-18: Sign bits.
    Sign = 2,
    /// 2026-10-18: Trellis start states.
    TrellisState = 3,
}

/// 2026-10-18: The element type of a side tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SideElem {
    /// 2026-10-18: FP8 E4M3, 8 bits.
    E4m3 = 0,
    /// 2026-10-18: E8M0 power of two, 8 bits.
    E8m0 = 1,
    /// 2026-10-18: BF16, 16 bits.
    Bf16 = 2,
    /// 2026-10-18: F32, 32 bits.
    F32 = 3,
    /// 2026-10-18: One bit.
    Bit = 4,
    /// 2026-10-18: IEEE FP16, 16 bits (the 128-group codecs' norm).
    F16 = 5,
    /// 2026-10-18: A 16-bit unsigned slot (a trellis start state and its padding).
    U16 = 6,
}

impl SideElem {
    /// 2026-10-18: Bits per element.
    pub const fn bits(self) -> u64 {
        match self {
            SideElem::E4m3 | SideElem::E8m0 => 8,
            SideElem::Bf16 | SideElem::F16 | SideElem::U16 => 16,
            SideElem::F32 => 32,
            SideElem::Bit => 1,
        }
    }
}

/// 2026-10-18: One side tensor: one `elem` per `per_values` values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SideTensor {
    /// 2026-10-18: What it holds.
    pub kind: SideKind,
    /// 2026-10-18: Its element type.
    pub elem: SideElem,
    /// 2026-10-18: Values per element; never 0.
    pub per_values: u32,
}

/// 2026-10-18: The basis the cached values are stored in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Rotation {
    /// 2026-10-18: The model's own basis.
    None,
    /// 2026-10-18: Walsh-Hadamard rotated over the whole head, for these head dims (the
    /// `wht_bf16` bookends wrap attention).
    Wht {
        /// 2026-10-18: Head dims the rotation kernel supports.
        sizes: &'static [u32],
    },
    /// 2026-10-18: Fast WHT over fixed blocks of `block` values.
    Fwht {
        /// 2026-10-18: Values per rotated block.
        block: u32,
    },
    /// 2026-10-18: Rotated per `block` values inside the codec's own write and attention
    /// kernels, so attention needs no bookends; the head dim must be a whole number of blocks.
    InKernel {
        /// 2026-10-18: Values per rotated block.
        block: u32,
    },
}

/// 2026-10-18: Trellis-coded quantization parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TrellisSpec {
    /// 2026-10-18: Trellis states.
    pub states: u16,
    /// 2026-10-18: Bits per coded value.
    pub bits: u8,
}

/// 2026-10-18: The host-side dequantization a spilled or swapped block needs. A codec with
/// `None` cannot be host-dequantized, so high-speed swap does not engage for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum HostLutId {
    /// 2026-10-18: Values are already bf16.
    Copy = 0,
    /// 2026-10-18: FP8 E4M3 values times the per-tensor scale.
    E4m3Tensor = 1,
    /// 2026-10-18: E2M1 values times an E4M3 group scale.
    E2m1E4m3 = 2,
    /// 2026-10-18: The 16-level turbo4 codebook times an E4M3 group scale.
    Turbo4Lut = 3,
    /// 2026-10-18: The 8-level turbo3 codebook times an E4M3 group scale.
    Turbo3Lut = 4,
    /// 2026-10-18: FP8 E4M3 values times a BF16 group norm.
    E4m3Bf16Norm = 5,
    /// 2026-10-19: FP8 E4M3 values times an E8M0 group scale.
    E4m3E8m0 = 6,
    /// 2026-10-19: E2M1 values times an E8M0 group scale.
    E2m1E8m0 = 7,
}

/// 2026-10-18: How many leading and trailing attention layers a plan keeps at the boundary
/// codec when `--kv-high-precision-layers` is left at auto. Lossy low-bit codecs accumulate
/// quantization error across layers; the boundary layers bound it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AutoBoundary {
    /// 2026-10-18: No boundary layers.
    None,
    /// 2026-10-18: ceil(n / 3), at least 2.
    OneThird,
    /// 2026-10-18: ceil(4n / 5), at least 4.
    FourFifths,
}

impl AutoBoundary {
    /// 2026-10-18: The boundary count for `num_attention_layers` layers; `None` when the codec
    /// asks for none.
    pub const fn layers(self, num_attention_layers: u32) -> Option<u32> {
        let n = num_attention_layers;
        match self {
            AutoBoundary::None => None,
            AutoBoundary::OneThird => Some(max(n.div_ceil(3), 2)),
            AutoBoundary::FourFifths => Some(max(n.saturating_mul(4).div_ceil(5), 4)),
        }
    }
}

const fn max(a: u32, b: u32) -> u32 {
    if a > b { a } else { b }
}

const fn contains(set: &[u32], x: u32) -> bool {
    let mut i = 0;
    while i < set.len() {
        if set[i] == x {
            return true;
        }
        i += 1;
    }
    false
}

/// 2026-10-18: The geometry of one side of one paged block.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlockGeometry {
    /// 2026-10-18: Tokens per block.
    pub block_size: u32,
    /// 2026-10-18: KV heads in the layer.
    pub num_kv_heads: u32,
    /// 2026-10-18: Head dimension.
    pub head_dim: u32,
}

impl BlockGeometry {
    /// 2026-10-18: Values in one side of one block.
    pub const fn values(&self) -> u64 {
        self.block_size as u64 * self.num_kv_heads as u64 * self.head_dim as u64
    }
}

/// 2026-10-18: What a codec's rows hold. An attention codec is a `--kv-cache-dtype` side; an
/// indexer codec stores a sparse-attention indexer's key rows (`--indexer-cache-dtype`) and is
/// never accepted as an attention side.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CodecRole {
    /// 2026-10-18: The keys or values of an attention layer.
    Attention = 0,
    /// 2026-10-18: A sparse-attention indexer's compressed key rows.
    Indexer = 1,
    /// 2026-10-19: A compressed-latent attention layer.s window or latent rows that only that
    /// layer.s own kernels read (`LatentKvPlan`); never a `--kv-cache-dtype` side.
    Latent = 2,
}

/// 2026-10-18: Everything the engine knows about one codec. Every row lives in the registry
/// table (`registry.rs`).
#[derive(Debug)]
pub struct KvCodecSpec {
    /// 2026-10-18: The id; `spec(id).id == id` is checked at compile time.
    pub id: KvCodecId,
    /// 2026-10-18: The canonical spelling (`--kv-cache-dtype` for an attention codec).
    pub name: &'static str,
    /// 2026-10-18: What the rows hold; only an attention codec parses as a `--kv-cache-dtype`.
    pub role: CodecRole,
    /// 2026-10-18: The display label telemetry prints (e.g. `Turbo4`); distinct per codec.
    pub label: &'static str,
    /// 2026-10-18: Bits per stored value.
    pub value_bits: u8,
    /// 2026-10-18: Values per quantization group; 0 when the codec has none.
    pub group: u32,
    /// 2026-10-18: In-block side tensors, in storage order after the values.
    pub side: &'static [SideTensor],
    /// 2026-10-18: The stored basis.
    pub rotation: Rotation,
    /// 2026-10-18: Trellis parameters, when trellis coded.
    pub trellis: Option<TrellisSpec>,
    /// 2026-10-18: The launch-argument family.
    pub abi: AbiId,
    /// 2026-10-18: Host dequantization, when the host can read the block.
    pub host_dequant: Option<HostLutId>,
    /// 2026-10-18: True when a write then read returns the input bits.
    pub lossless: bool,
    /// 2026-10-18: Head dims a decode kernel of this codec's own is selected for today.
    pub head_dims: &'static [u32],
    /// 2026-10-18: True when the scale is calibrated online per tensor (fp8).
    pub online_calibration: bool,
    /// 2026-10-18: The boundary layers an auto `--kv-high-precision-layers` gives this codec
    /// when it is served on both sides.
    pub auto_boundary: AutoBoundary,
    /// 2026-10-18: The CITATIONS.md heading of the method the codec implements; `None` for a
    /// plain hardware number format, which no paper introduces.
    pub citation: Option<&'static str>,
}

impl KvCodecSpec {
    /// 2026-10-18: True when the stored values are in a rotated basis that attention wraps in
    /// the rotation bookends. An in-kernel rotation is false: its own kernels undo it.
    pub const fn rotates(&self) -> bool {
        match self.rotation {
            Rotation::None | Rotation::InKernel { .. } => false,
            Rotation::Wht { .. } | Rotation::Fwht { .. } => true,
        }
    }

    /// 2026-10-18: True when the stored values are in any rotated basis, undone by the rotation
    /// bookends or by the codec's own kernels. Such a side has no dense attention to fall back
    /// to, so every kernel that reads it must be its own.
    pub const fn stores_rotated(&self) -> bool {
        match self.rotation {
            Rotation::None => false,
            Rotation::InKernel { .. } | Rotation::Wht { .. } | Rotation::Fwht { .. } => true,
        }
    }

    /// 2026-10-18: True when this side is rotated at `head_dim`: a WHT rotation lists the head
    /// dims its kernel supports, and a block FWHT needs whole blocks.
    pub const fn rotates_at(&self, head_dim: u32) -> bool {
        match self.rotation {
            Rotation::None | Rotation::InKernel { .. } => false,
            Rotation::Wht { sizes } => contains(sizes, head_dim),
            Rotation::Fwht { block } => block != 0 && head_dim.is_multiple_of(block),
        }
    }

    /// 2026-10-18: True when this codec's kernels can hold a `head_dim` row. An in-kernel
    /// rotation needs whole blocks at a head dim its kernels are built for (`head_dims` lists
    /// them all); every other codec stores any head dim.
    pub const fn fits_head_dim(&self, head_dim: u32) -> bool {
        match self.rotation {
            Rotation::InKernel { block } => {
                block != 0 && head_dim.is_multiple_of(block) && contains(self.head_dims, head_dim)
            }
            Rotation::None | Rotation::Wht { .. } | Rotation::Fwht { .. } => true,
        }
    }

    /// 2026-10-18: True when the stored values are plain bf16, so a host probe can read them
    /// without dequantizing (the host LUT is a copy).
    pub const fn stores_bf16(&self) -> bool {
        matches!(self.host_dequant, Some(HostLutId::Copy))
    }

    /// 2026-10-18: Bytes of packed values: `values * value_bits / 8`.
    pub const fn data_bytes(&self, values: u64) -> u64 {
        values * self.value_bits as u64 / 8
    }

    /// 2026-10-18: Bytes of every in-block side tensor for `values` values.
    pub const fn side_bytes(&self, values: u64) -> u64 {
        let mut total = 0;
        let mut i = 0;
        while i < self.side.len() {
            let t = self.side[i];
            let elems = values / t.per_values as u64;
            let bits = t.elem.bits();
            total += if bits.is_multiple_of(8) {
                elems * (bits / 8)
            } else {
                (elems * bits).div_ceil(8)
            };
            i += 1;
        }
        total
    }

    /// 2026-10-18: Bytes of one side of one block: values then side tensors.
    pub const fn block_bytes(&self, g: &BlockGeometry) -> u64 {
        let values = g.values();
        self.data_bytes(values) + self.side_bytes(values)
    }

    /// 2026-10-18: A stable FNV-1a hash of the layout fields (bits, group, side tensors,
    /// rotation, trellis, ABI). It changes exactly when the stored bytes change meaning.
    pub const fn digest(&self) -> u64 {
        let mut h = Fnv::new().byte(self.value_bits).word(self.group as u64);
        let mut i = 0;
        while i < self.side.len() {
            let t = self.side[i];
            h = h
                .byte(t.kind as u8)
                .byte(t.elem as u8)
                .word(t.per_values as u64);
            i += 1;
        }
        h = match self.rotation {
            Rotation::None => h.byte(0),
            Rotation::Wht { sizes } => {
                let mut h = h.byte(1);
                let mut j = 0;
                while j < sizes.len() {
                    h = h.word(sizes[j] as u64);
                    j += 1;
                }
                h
            }
            Rotation::Fwht { block } => h.byte(2).word(block as u64),
            Rotation::InKernel { block } => h.byte(3).word(block as u64),
        };
        h = match self.trellis {
            None => h.byte(0),
            Some(t) => h.byte(1).word(t.states as u64).byte(t.bits),
        };
        h.byte(self.abi as u8).finish()
    }
}

/// 2026-10-18: FNV-1a 64, usable in `const fn`.
#[derive(Clone, Copy)]
pub(crate) struct Fnv(u64);

impl Fnv {
    pub(crate) const fn new() -> Self {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
    pub(crate) const fn byte(self, b: u8) -> Self {
        Fnv((self.0 ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3))
    }
    pub(crate) const fn word(self, w: u64) -> Self {
        let bytes = w.to_le_bytes();
        let mut h = self;
        let mut i = 0;
        while i < 8 {
            h = h.byte(bytes[i]);
            i += 1;
        }
        h
    }
    pub(crate) const fn finish(self) -> u64 {
        self.0
    }
}

/// 2026-10-18: FNV-1a 64 over `parts`, for digests built outside this module.
pub(crate) const fn fnv_words(parts: &[u64]) -> u64 {
    let mut h = Fnv::new();
    let mut i = 0;
    while i < parts.len() {
        h = h.word(parts[i]);
        i += 1;
    }
    h.finish()
}
