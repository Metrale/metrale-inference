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
    SideTensor, TrellisSpec,
};

/// 2026-10-18: Values per group of the nvfp4 and turbo4/3/2/8 codecs.
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

/// 2026-10-19: Values per group of the OCP microscaling (MX) codecs.
const GROUP32: u32 = 32;

/// 2026-10-19: One E8M0 (power-of-two) scale per 32 values (mxfp8, mxfp4).
const E8M0_SCALE_PER_32: &[SideTensor] = &[SideTensor {
    kind: SideKind::Scale,
    elem: SideElem::E8m0,
    per_values: GROUP32,
}];

/// 2026-10-19: MXFP8 window rows of a compressed-latent attention layer: FP8 E4M3 values and an
/// E8M0 scale per 32, the format the DeepSeek-V4.1 reference stores its sliding-window keys in.
pub(crate) const MXFP8_E8M0X32: KvCodecSpec = KvCodecSpec {
    id: KvCodecId::Mxfp8,
    name: "mxfp8",
    role: CodecRole::Latent,
    label: "MXFP8",
    value_bits: 8,
    group: GROUP32,
    side: E8M0_SCALE_PER_32,
    rotation: Rotation::None,
    trellis: None,
    abi: AbiId::BlockScaled,
    host_dequant: Some(HostLutId::E4m3E8m0),
    lossless: false,
    head_dims: &[512],
    online_calibration: false,
    auto_boundary: AutoBoundary::None,
    citation: None,
};

/// 2026-10-19: MXFP4 indexer key rows: E2M1 values and an E8M0 scale per 32, the format the
/// DeepSeek-V4.1 reference stores its compressed index keys in.
pub(crate) const MXFP4_E8M0X32: KvCodecSpec = KvCodecSpec {
    id: KvCodecId::Mxfp4,
    name: "mxfp4",
    role: CodecRole::Indexer,
    label: "MXFP4",
    value_bits: 4,
    group: GROUP32,
    side: E8M0_SCALE_PER_32,
    rotation: Rotation::None,
    trellis: None,
    abi: AbiId::BlockScaled,
    host_dequant: Some(HostLutId::E2m1E8m0),
    lossless: false,
    head_dims: &[128],
    online_calibration: false,
    auto_boundary: AutoBoundary::None,
    citation: None,
};

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

/// 2026-10-18: Values per group of the 128-group rotated codecs (`metrale-kv-tq128`).
const GROUP128: u32 = 128;

/// 2026-10-18: Head dims the 128-group kernels serve (`paged_decode_attn_tq128.cu`).
const TQ128_HEAD_DIMS: &[u32] = &[128, 256, 512];

/// 2026-10-18: The CITATIONS.md heading of the trellis-coded KV-cache prior art.
const TCQ: Option<&str> = Some("Trellis-coded KV cache (TCQ)");

/// 2026-10-18: One FP16 norm per 128 values.
const F16_NORM: SideTensor = SideTensor {
    kind: SideKind::Norm,
    elem: SideElem::F16,
    per_values: GROUP128,
};

/// 2026-10-18: The trellis codecs' per-group side: the norm, then the start-state slot.
const TCQ_SIDE: &[SideTensor] = &[
    F16_NORM,
    SideTensor {
        kind: SideKind::TrellisState,
        elem: SideElem::U16,
        per_values: GROUP128,
    },
];

/// 2026-10-18: A 128-group rotated codec row; only the id, name, label, bits, side, trellis
/// and auto boundary differ between the four.
const fn tq128(
    id: KvCodecId,
    name: &'static str,
    label: &'static str,
    value_bits: u8,
    side: &'static [SideTensor],
    trellis: Option<TrellisSpec>,
    auto_boundary: AutoBoundary,
) -> KvCodecSpec {
    KvCodecSpec {
        id,
        name,
        role: CodecRole::Attention,
        label,
        value_bits,
        group: GROUP128,
        side,
        rotation: Rotation::InKernel { block: GROUP128 },
        trellis,
        abi: AbiId::Trellis,
        host_dequant: None,
        lossless: false,
        head_dims: TQ128_HEAD_DIMS,
        online_calibration: false,
        auto_boundary,
        citation: TCQ,
    }
}

/// 2026-10-18: 512-state bitshift trellis, 3 bits per value: 52 bytes per 128 values
/// (3.25 bits per value).
pub(crate) const TURBO3_TCQ: KvCodecSpec = tq128(
    KvCodecId::Turbo3Tcq,
    "turbo3_tcq",
    "Turbo3Tcq",
    3,
    TCQ_SIDE,
    Some(TrellisSpec {
        states: 512,
        bits: 3,
    }),
    AutoBoundary::OneThird,
);

/// 2026-10-18: 256-state bitshift trellis, 2 bits per value: 36 bytes per 128 values
/// (2.25 bits per value). Below turbo2's 2.5 bits, so it takes turbo2's wide boundary.
pub(crate) const TURBO2_TCQ: KvCodecSpec = tq128(
    KvCodecId::Turbo2Tcq,
    "turbo2_tcq",
    "Turbo2Tcq",
    2,
    TCQ_SIDE,
    Some(TrellisSpec {
        states: 256,
        bits: 2,
    }),
    AutoBoundary::FourFifths,
);

/// 2026-10-18: 256-state bitshift trellis, 1 bit per value: 20 bytes per 128 values
/// (1.25 bits per value). The lossiest row, so it takes the wide boundary turbo2 takes.
pub(crate) const TURBO1_TCQ: KvCodecSpec = tq128(
    KvCodecId::Turbo1Tcq,
    "turbo1_tcq",
    "Turbo1Tcq",
    1,
    TCQ_SIDE,
    Some(TrellisSpec {
        states: 256,
        bits: 1,
    }),
    AutoBoundary::FourFifths,
);

/// 2026-10-18: The turbo4 16-level codebook over one rotated 128-value group with one FP16 norm:
/// 66 bytes per 128 values (4.125 bits per value).
pub(crate) const TURBO4_N128: KvCodecSpec = tq128(
    KvCodecId::Turbo4N128,
    "turbo4_n128",
    "Turbo4N128",
    4,
    &[F16_NORM],
    None,
    AutoBoundary::OneThird,
);
