// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: `TrellisKv` against the mock backend: which pairs build state, the uploaded
//! books, every launch's argument list in its `.cu` declaration order, and each refusal. The
//! codecs are found through the registry (`AbiId::Trellis`), so the tests cover a codec added
//! later without an edit.

use super::*;
use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend};
use metrale_kv_codec::{KvCodecId, kv_kernel_names};

const KERNEL: KernelHandle = KernelHandle(0x77);
const STREAM: u64 = 5;
const K_SIDE: KvSideArgs = KvSideArgs::Trellis {
    pool: DevicePtr(0xA0),
    block_stride: 6656,
};
const V_SIDE: KvSideArgs = KvSideArgs::Trellis {
    pool: DevicePtr(0xB0),
    block_stride: 7000,
};

/// 2026-10-18: Every registry attention codec of the `Trellis` family.
fn trellis_codecs() -> Vec<KvCodecId> {
    let all: Vec<KvCodecId> = KvCodecId::ATTENTION
        .into_iter()
        .filter(|&id| spec(id).abi == AbiId::Trellis)
        .collect();
    assert!(!all.is_empty(), "the registry has no Trellis codec");
    all
}

fn state(gpu: &MockGpuBackend, id: KvCodecId) -> TrellisKv {
    let pair = KvPair::symmetric(id);
    let names = kv_kernel_names(pair, 256).unwrap();
    TrellisKv::for_pair(gpu, pair, 256, &names)
        .unwrap()
        .unwrap_or_else(|| panic!("{} built no trellis state", spec(id).name))
}

fn f32s(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// 2026-10-18: A launch as `grid|block|args`: a buffer as `p<addr>`, a 4-byte scalar as its
/// `u32` bits, an 8-byte one as `w<u64>`.
fn render_for(gpu: &MockGpuBackend, func: u64) -> Vec<String> {
    gpu.launches_snapshot()
        .iter()
        .map(|l| {
            assert_eq!((l.func, l.stream, l.shared_mem), (func, STREAM, 0));
            let args: Vec<String> = l
                .args
                .iter()
                .map(|a| match a {
                    MockArg::Buffer(p) => format!("p{:x}", p.0),
                    MockArg::Bytes(b) if b.len() == 4 => {
                        u32::from_le_bytes(b[..].try_into().unwrap()).to_string()
                    }
                    MockArg::Bytes(b) if b.len() == 8 => {
                        format!("w{}", u64::from_le_bytes(b[..].try_into().unwrap()))
                    }
                    MockArg::Bytes(b) => panic!("unexpected {}-byte scalar", b.len()),
                })
                .collect();
            format!("{:?}|{:?}|{}", l.grid, l.block, args.join(" "))
        })
        .collect()
}

/// 2026-10-18: The launches of the layer's write or decode `KERNEL`.
fn render(gpu: &MockGpuBackend) -> Vec<String> {
    render_for(gpu, KERNEL.0)
}

fn write_common(num_tokens: u32) -> WriteCommon {
    WriteCommon {
        key: DevicePtr(0x11),
        value: DevicePtr(0x12),
        slot_mapping: DevicePtr(0x13),
        num_tokens,
        num_kv_heads: 4,
        head_dim: 256,
        block_size: 16,
        key_stride: 1024,
        value_stride: 1040,
    }
}

fn decode_common(sliding_window: u32) -> DecodeCommon {
    DecodeCommon {
        q: DevicePtr(0x21),
        output: DevicePtr(0x22),
        block_tables: DevicePtr(0x23),
        seq_lens: DevicePtr(0x24),
        max_blocks_per_seq: 33,
        num_seqs: 3,
        num_q_heads: 24,
        num_kv_heads: 4,
        head_dim: 256,
        block_size: 16,
        inv_sqrt_d: 0.0625,
        q_stride: 6144,
        sliding_window,
    }
}

fn code(e: anyhow::Error) -> Option<ErrorCode> {
    e.downcast_ref::<metrale_error::Error>().map(|e| e.code())
}

/// 2026-10-18: Each trellis codec has a kv-tq128 format; its state uploads exactly that format's
/// K and V books (and its write midpoints when it has them) and resolves the table's prefill
/// kernel.
#[test]
fn each_trellis_codec_uploads_its_own_books_and_resolves_its_prefill() {
    for id in trellis_codecs() {
        let gpu = MockGpuBackend::new();
        let s = state(&gpu, id);
        let codec = Tq128Codec::named(spec(id).name).expect("a kv-tq128 format");
        assert_eq!(s.codec, codec);
        let book = |p| gpu.read_alloc(p).unwrap();
        assert_eq!(book(s.k_book), f32s(codec.codebook(KvSide::Key)));
        assert_eq!(book(s.v_book), f32s(codec.codebook(KvSide::Value)));
        assert_eq!(
            s.mids.map(book),
            codec.midpoints().map(f32s),
            "{}",
            codec.name()
        );
        let prefill = kv_kernel_names(KvPair::symmetric(id), 256)
            .unwrap()
            .prefill
            .unwrap();
        let lookups = gpu.kernel_lookups_snapshot();
        assert_eq!(
            lookups,
            [(prefill.module.to_owned(), prefill.func.to_owned())]
        );
    }
}

/// 2026-10-18: A pair with no trellis side builds no state and touches the device not at all.
#[test]
fn other_pairs_build_nothing() {
    let mut checked = 0;
    for pair in KvPair::all() {
        if spec(pair.k()).abi == AbiId::Trellis || spec(pair.v()).abi == AbiId::Trellis {
            continue;
        }
        let Ok(names) = kv_kernel_names(pair, 128) else {
            continue;
        };
        let gpu = MockGpuBackend::new();
        assert!(
            TrellisKv::for_pair(&gpu, pair, 128, &names)
                .unwrap()
                .is_none()
        );
        assert!(gpu.kernel_lookups_snapshot().is_empty(), "{pair}");
        checked += 1;
    }
    assert!(checked > 0);
}

/// 2026-10-18: A table without a prefill kernel for the codec is refused at construction, not
/// at the first prefill.
#[test]
fn a_missing_prefill_kernel_is_refused_at_construction() {
    let id = trellis_codecs()[0];
    let pair = KvPair::symmetric(id);
    let mut names = kv_kernel_names(pair, 256).unwrap();
    names.prefill = None;
    let e = TrellisKv::for_pair(&MockGpuBackend::new(), pair, 256, &names)
        .err()
        .expect("refused");
    assert_eq!(code(e), Some(ErrorCode::KvAbiUnimplemented));
}

/// 2026-10-18: The write is two launches, K then V, one warp per (token, head, 128-group):
/// 7 tokens x 4 heads x 2 groups = 56 groups, 14 blocks of 4 warps. A codec with write
/// midpoints passes them right after the book.
#[test]
fn the_write_launches_k_then_v_in_declaration_order() {
    for id in trellis_codecs() {
        let gpu = MockGpuBackend::new();
        let s = state(&gpu, id);
        s.write(&gpu, KERNEL, &write_common(7), K_SIDE, V_SIDE, STREAM)
            .unwrap();
        let mids = s.mids.map(|m| format!(" p{:x}", m.0)).unwrap_or_default();
        let side = |src: &str, pool: &str, book: DevicePtr, stride: u32, block: u64| {
            format!(
                "[14, 1, 1]|[128, 1, 1]|p{src} p{pool} p13 p{:x}{mids} 7 4 256 16 {stride} w{block}",
                book.0
            )
        };
        assert_eq!(
            render(&gpu),
            [
                side("11", "a0", s.k_book, 1024, 6656),
                side("12", "b0", s.v_book, 1040, 7000)
            ],
            "{}",
            spec(id).name
        );
    }
}

/// 2026-10-18: Zero tokens launch nothing (a grid of zero blocks is a launch error).
#[test]
fn an_empty_write_launches_nothing() {
    let gpu = MockGpuBackend::new();
    let s = state(&gpu, trellis_codecs()[0]);
    s.write(&gpu, KERNEL, &write_common(0), K_SIDE, V_SIDE, STREAM)
        .unwrap();
    assert!(gpu.launches_snapshot().is_empty());
}

/// 2026-10-18: Decode passes both books, then the shape, the softmax scale, each side's decode
/// scale from kv-tq128, the Q stride and both block strides.
#[test]
fn the_decode_launch_passes_each_sides_book_and_scale() {
    for id in trellis_codecs() {
        let gpu = MockGpuBackend::new();
        let s = state(&gpu, id);
        s.decode(&gpu, KERNEL, &decode_common(0), K_SIDE, V_SIDE, STREAM)
            .unwrap();
        let (ks, vs) = (
            s.codec.decode_scale(KvSide::Key).to_bits(),
            s.codec.decode_scale(KvSide::Value).to_bits(),
        );
        let want = format!(
            "[24, 3, 1]|[256, 1, 1]|p21 pa0 pb0 p22 p23 p24 p{:x} p{:x} 33 24 4 256 16 {} {ks} {vs} \
             6144 w6656 w7000",
            s.k_book.0,
            s.v_book.0,
            0.0625f32.to_bits()
        );
        assert_eq!(render(&gpu), [want], "{}", spec(id).name);
    }
}

fn prefill_args(sliding_window: u32, k: KvSideArgs) -> TrellisPrefill {
    TrellisPrefill {
        q: DevicePtr(0x31),
        out: DevicePtr(0x32),
        block_table: DevicePtr(0x33),
        seq_len_start: 40,
        n: 9,
        nq: 24,
        nkv: 4,
        hd: 256,
        bs: 16,
        inv_sqrt_d: 0.0625,
        sliding_window,
        k,
        v: V_SIDE,
    }
}

/// 2026-10-18: Prefill runs one block per (query head, chunk token) through the resolved
/// prefill kernel, with the Q stride of a contiguous `[n, nq, hd]` chunk.
#[test]
fn the_prefill_launch_covers_every_head_and_chunk_token() {
    let gpu = MockGpuBackend::new();
    let s = state(&gpu, trellis_codecs()[0]);
    s.prefill(&gpu, &prefill_args(0, K_SIDE), STREAM).unwrap();
    let (ks, vs) = (
        s.codec.decode_scale(KvSide::Key).to_bits(),
        s.codec.decode_scale(KvSide::Value).to_bits(),
    );
    let want = format!(
        "[24, 9, 1]|[256, 1, 1]|p31 pa0 pb0 p32 p33 p{:x} p{:x} 40 24 4 256 16 {} {ks} {vs} 6144 \
         w6656 w7000",
        s.k_book.0,
        s.v_book.0,
        0.0625f32.to_bits()
    );
    // 2026-10-18: The mock hands every kernel lookup the handle 0xDEAD.
    assert_eq!(render_for(&gpu, 0xDEAD), [want]);
}

/// 2026-10-18: The kernels apply no sliding window, so one is refused before any launch.
#[test]
fn a_sliding_window_is_refused_before_launching() {
    let gpu = MockGpuBackend::new();
    let s = state(&gpu, trellis_codecs()[0]);
    let e = s
        .decode(&gpu, KERNEL, &decode_common(128), K_SIDE, V_SIDE, STREAM)
        .unwrap_err();
    assert_eq!(code(e), Some(ErrorCode::KvCodecFeatureUnsupported));
    let e = s
        .prefill(&gpu, &prefill_args(128, K_SIDE), STREAM)
        .unwrap_err();
    assert_eq!(code(e), Some(ErrorCode::KvCodecFeatureUnsupported));
    assert!(gpu.launches_snapshot().is_empty());
}

/// 2026-10-18: A side of another family is refused, not launched with a misread stride.
#[test]
fn a_side_of_another_family_is_refused() {
    let gpu = MockGpuBackend::new();
    let s = state(&gpu, trellis_codecs()[0]);
    let dense = KvSideArgs::Dense {
        pool: DevicePtr(0xA0),
        block_stride: 6656,
    };
    let e = s
        .write(&gpu, KERNEL, &write_common(7), dense, V_SIDE, STREAM)
        .unwrap_err();
    assert_eq!(code(e), Some(ErrorCode::KvCodecFeatureUnsupported));
    let e = s
        .decode(&gpu, KERNEL, &decode_common(0), K_SIDE, dense, STREAM)
        .unwrap_err();
    assert_eq!(code(e), Some(ErrorCode::KvCodecFeatureUnsupported));
    let e = s
        .prefill(&gpu, &prefill_args(0, dense), STREAM)
        .unwrap_err();
    assert_eq!(code(e), Some(ErrorCode::KvCodecFeatureUnsupported));
    assert!(gpu.launches_snapshot().is_empty());
}
