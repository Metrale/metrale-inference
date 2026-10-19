// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: One spec constant per registered codec. The registry (`registry.rs`) binds each
//! to its id; the layout golden test (`tests/layout.rs`) holds them to the byte sizes the cache
//! allocated before the registry existed.
//!
//! Owner: kv-codec.
//! Invariants:
//! - Each constant's `id` and `name` match its registry row (compile-time check in
//!   `registry.rs`).

use crate::KvCodecId;
use crate::spec::{
    AbiId, AutoBoundary, CodecRole, HostLutId, KvCodecSpec, Rotation, SideElem, SideKind,
    SideTensor,
};

/// 2026-10-18: Values per group of every block-scaled codec registered today.
const GROUP16: u32 = 16;

/// 2026-10-18: One E4M3 scale per 16 values (nvfp4, turbo4/3/2).
const E4M3_SCALE_PER_16: &[SideTensor] = &[SideTensor {
    kind: SideKind::Scale,
    elem: SideElem::E4m3,
    per_values: GROUP16,
}];

/// 2026-10-18: One BF16 norm per 16 values (turbo8).
const BF16_NORM_PER_16: &[SideTensor] = &[SideTensor {
    kind: SideKind::Norm,
    elem: SideElem::Bf16,
    per_values: GROUP16,
}];

/// 2026-10-18: Head dims the `wht_bf16` bookend kernel rotates.
const WHT_SIZES: &[u32] = &[128, 256, 512];

/// 2026-10-18: The CITATIONS.md heading of the TurboQuant+ KV-cache prior art.
const TURBOQUANT: Option<&str> = Some("TurboQuant+ KV cache");

/// 2026-10-18: BF16 values.
pub(crate) const BF16: KvCodecSpec = KvCodecSpec {
    id: KvCodecId::Bf16,
    name: "bf16",
    role: CodecRole::Attention,
    label: "BF16",
    value_bits: 16,
    group: 0,
    side: &[],
    rotation: Rotation::None,
    trellis: None,
    abi: AbiId::Dense,
    host_dequant: Some(HostLutId::Copy),
    lossless: true,
    head_dims: &[128, 256, 512],
    online_calibration: false,
    auto_boundary: AutoBoundary::None,
    citation: None,
};

/// 2026-10-18: FP8 E4M3 values; the per-tensor scale lives outside the block.
pub(crate) const FP8_E4M3_TENSOR: KvCodecSpec = KvCodecSpec {
    id: KvCodecId::Fp8,
    name: "fp8",
    role: CodecRole::Attention,
    label: "FP8",
    value_bits: 8,
    group: 0,
    side: &[],
    rotation: Rotation::None,
    trellis: None,
    abi: AbiId::DenseScaled,
    host_dequant: Some(HostLutId::E4m3Tensor),
    lossless: false,
    head_dims: &[128, 256, 512],
    online_calibration: true,
    auto_boundary: AutoBoundary::None,
    citation: None,
};

/// 2026-10-18: E2M1 values plus an E4M3 scale per 16.
pub(crate) const NVFP4_E4M3X16: KvCodecSpec = KvCodecSpec {
    id: KvCodecId::Nvfp4,
    name: "nvfp4",
    role: CodecRole::Attention,
    label: "NVFP4",
    value_bits: 4,
    group: GROUP16,
    side: E4M3_SCALE_PER_16,
    rotation: Rotation::None,
    trellis: None,
    abi: AbiId::BlockScaled,
    host_dequant: Some(HostLutId::E2m1E4m3),
    lossless: false,
    head_dims: &[128, 256],
    online_calibration: false,
    auto_boundary: AutoBoundary::None,
    citation: None,
};

/// 2026-10-18: WHT-rotated 16-level codebook in the nvfp4 byte layout.
pub(crate) const TURBO4: KvCodecSpec = KvCodecSpec {
    id: KvCodecId::Turbo4,
    name: "turbo4",
    role: CodecRole::Attention,
    label: "Turbo4",
    value_bits: 4,
    group: GROUP16,
    side: E4M3_SCALE_PER_16,
    rotation: Rotation::Wht { sizes: WHT_SIZES },
    trellis: None,
    abi: AbiId::BlockScaled,
    host_dequant: Some(HostLutId::Turbo4Lut),
    lossless: false,
    head_dims: &[128, 256, 512],
    online_calibration: false,
    auto_boundary: AutoBoundary::OneThird,
    citation: TURBOQUANT,
};

/// 2026-10-18: WHT-rotated 8-level codebook, 8 values in 3 bytes.
pub(crate) const TURBO3: KvCodecSpec = KvCodecSpec {
    id: KvCodecId::Turbo3,
    name: "turbo3",
    role: CodecRole::Attention,
    label: "Turbo3",
    value_bits: 3,
    group: GROUP16,
    side: E4M3_SCALE_PER_16,
    rotation: Rotation::Wht { sizes: WHT_SIZES },
    trellis: None,
    abi: AbiId::BlockScaled,
    host_dequant: Some(HostLutId::Turbo3Lut),
    lossless: false,
    head_dims: &[128, 256],
    online_calibration: false,
    auto_boundary: AutoBoundary::OneThird,
    citation: TURBOQUANT,
};

/// 2026-10-18: WHT-rotated 4-level codebook, 4 values per byte. No host LUT exists, so
/// high-speed swap does not engage for it.
pub(crate) const TURBO2: KvCodecSpec = KvCodecSpec {
    id: KvCodecId::Turbo2,
    name: "turbo2",
    role: CodecRole::Attention,
    label: "Turbo2",
    value_bits: 2,
    group: GROUP16,
    side: E4M3_SCALE_PER_16,
    rotation: Rotation::Wht { sizes: WHT_SIZES },
    trellis: None,
    abi: AbiId::BlockScaled,
    host_dequant: None,
    lossless: false,
    head_dims: &[128, 256],
    online_calibration: false,
    auto_boundary: AutoBoundary::FourFifths,
    citation: TURBOQUANT,
};

/// 2026-10-18: WHT-rotated FP8 E4M3 values plus a BF16 norm per 16.
pub(crate) const TURBO8: KvCodecSpec = KvCodecSpec {
    id: KvCodecId::Turbo8,
    name: "turbo8",
    role: CodecRole::Attention,
    label: "Turbo8",
    value_bits: 8,
    group: GROUP16,
    side: BF16_NORM_PER_16,
    rotation: Rotation::Wht { sizes: WHT_SIZES },
    trellis: None,
    abi: AbiId::BlockScaled,
    host_dequant: Some(HostLutId::E4m3Bf16Norm),
    lossless: false,
    head_dims: &[128, 256, 512],
    online_calibration: false,
    auto_boundary: AutoBoundary::OneThird,
    citation: TURBOQUANT,
};

/// 2026-10-18: Channels of the FP8 indexer kernels (`DSA_FP8_D` in `dsa_indexer_fp8.cu`): one
/// pool entry is one row of this many values under one scale, rotated by a WHT of this size.
const FP8_ROW_CHANNELS: u32 = 128;

/// 2026-10-18: FP8 E4M3 indexer rows, each rotated by a 128-point WHT and scaled by one F32
/// power of two: the GLM-5.3 DSA indexer cache (`--indexer-cache-dtype fp8`). A pool's entry is
/// `128 + 4` bytes, the term `IndexerRowFormat` computed by hand before the registry.
pub(crate) const FP8_ROW_F32SCALE: KvCodecSpec = KvCodecSpec {
    id: KvCodecId::Fp8RowF32,
    name: "fp8_row",
    role: CodecRole::Indexer,
    label: "FP8Row",
    value_bits: 8,
    group: FP8_ROW_CHANNELS,
    side: &[SideTensor {
        kind: SideKind::Scale,
        elem: SideElem::F32,
        per_values: FP8_ROW_CHANNELS,
    }],
    rotation: Rotation::Wht {
        sizes: &[FP8_ROW_CHANNELS],
    },
    trellis: None,
    abi: AbiId::BlockScaled,
    host_dequant: None,
    lossless: false,
    head_dims: &[FP8_ROW_CHANNELS],
    online_calibration: false,
    auto_boundary: AutoBoundary::None,
    citation: None,
};
