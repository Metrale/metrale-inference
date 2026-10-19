// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: Golden argument lists of the fused KV launchers, one per side shape. Each list is
//! the declaration order of the `.cu` entry point it launches; the lists were recorded from the
//! per-pair `ops::*` launchers the packer replaced, which matched it on all nine pairs and the
//! symmetric block-scaled cache before they were deleted.

use super::*;
use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend};
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};

const KERNEL: KernelHandle = KernelHandle(0x77);
const STREAM: u64 = 11;

fn ptr(n: u64) -> DevicePtr {
    DevicePtr(0x1000 + n)
}

const DENSE_K: KvSideArgs = KvSideArgs::Dense {
    pool: DevicePtr(0x1003),
    block_stride: 1001,
};
const SCALED_K: KvSideArgs = KvSideArgs::DenseScaled {
    pool: DevicePtr(0x1003),
    block_stride: 1001,
    scale: 0.375,
};
const BLOCK_K: KvSideArgs = KvSideArgs::BlockScaled {
    pool: DevicePtr(0x1003),
    block_stride: 1001,
    data_bytes: 501,
};
const BLOCK_V: KvSideArgs = KvSideArgs::BlockScaled {
    pool: DevicePtr(0x1004),
    block_stride: 2002,
    data_bytes: 602,
};

fn write_common() -> WriteCommon {
    WriteCommon {
        key: ptr(1),
        value: ptr(2),
        slot_mapping: ptr(5),
        num_tokens: 7,
        num_kv_heads: 4,
        head_dim: 256,
        block_size: 16,
        key_stride: 1024,
        value_stride: 1040,
    }
}

fn decode_common() -> DecodeCommon {
    DecodeCommon {
        q: ptr(1),
        output: ptr(6),
        block_tables: ptr(7),
        seq_lens: ptr(8),
        max_blocks_per_seq: 33,
        num_seqs: 3,
        num_q_heads: 24,
        num_kv_heads: 4,
        head_dim: 256,
        block_size: 16,
        inv_sqrt_d: 0.0625,
        q_stride: 6144,
        sliding_window: 9,
    }
}

/// 2026-10-18: The one launch `f` records, rendered as `grid|block|args`: a buffer as `p<addr>`,
/// a 4-byte scalar as its decimal `u32` bits, an 8-byte one as `w<u64>`.
fn render(f: impl FnOnce(&MockGpuBackend) -> anyhow::Result<()>) -> String {
    let gpu = MockGpuBackend::new();
    f(&gpu).unwrap();
    let launches = gpu.launches_snapshot();
    assert_eq!(launches.len(), 1);
    let l = &launches[0];
    assert_eq!((l.func, l.stream, l.shared_mem), (KERNEL.0, STREAM, 0));
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
}

const W: &str = "[7, 1, 1]|[256, 1, 1]|p1001 p1002 p1003 p1004 p1005 4 256 16 1024 1040";
const D: &str =
    "[24, 3, 1]|[256, 1, 1]|p1001 p1003 p1004 p1006 p1007 p1008 33 24 4 256 16 1031798784";
/// 2026-10-18: `0.375_f32.to_bits()`.
const SCALE: &str = "1052770304";

fn write(k: KvSideArgs, tails: Tails) -> String {
    render(|g| launch_fused_write(g, KERNEL, &write_common(), k, BLOCK_V, tails, STREAM))
}

fn decode(k: KvSideArgs, tails: Tails) -> String {
    render(|g| launch_fused_decode(g, KERNEL, &decode_common(), k, BLOCK_V, tails, STREAM))
}

/// 2026-10-18: A symmetric block-scaled cache (`reshape_and_cache_flash_nvfp4`,
/// `paged_decode_attn_nvfp4`): K's stride and data bytes once, no window.
#[test]
fn a_shared_block_scaled_cache_passes_one_tail() {
    assert_eq!(write(BLOCK_K, Tails::Shared), format!("{W} w1001 w501"));
    assert_eq!(
        decode(BLOCK_K, Tails::Shared),
        format!("{D} 6144 w1001 w501")
    );
}

/// 2026-10-18: A block-scaled pair (`*_turbo4k_turbo3v` and the other two turbo pairs): each
/// side's stride and data bytes, then the window for decode.
#[test]
fn a_block_scaled_pair_passes_both_tails() {
    assert_eq!(
        write(BLOCK_K, Tails::PerSide),
        format!("{W} w1001 w501 w2002 w602")
    );
    assert_eq!(
        decode(BLOCK_K, Tails::PerSide),
        format!("{D} 6144 w1001 w501 w2002 w602 9")
    );
}

/// 2026-10-18: A dense K with a block-scaled V (`*_bf16k_turbo{2,3,4}v`): the write takes K's
/// stride, the decode none of K's.
#[test]
fn a_dense_k_passes_its_stride_to_the_write_only() {
    assert_eq!(
        write(DENSE_K, Tails::PerSide),
        format!("{W} w1001 w2002 w602")
    );
    assert_eq!(
        decode(DENSE_K, Tails::PerSide),
        format!("{D} 6144 w2002 w602 9")
    );
}

/// 2026-10-18: A scaled K (`*_fp8k_turbo{2,3,4}v`): its scale after the shape for the write and
/// before the Q stride for the decode.
#[test]
fn a_scaled_k_passes_its_scale_where_each_kernel_declares_it() {
    assert_eq!(
        write(SCALED_K, Tails::PerSide),
        format!("{W} {SCALE} w1001 w2002 w602")
    );
    assert_eq!(
        decode(SCALED_K, Tails::PerSide),
        format!("{D} {SCALE} 6144 w2002 w602 9")
    );
}
