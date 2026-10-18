// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: The layout golden test: every codec's `block_bytes` equals the size the cache's
//! pre-registry `block_bytes_dims` match allocated, frozen here as literals over a geometry grid
//! that includes group-unaligned shapes (where each floor division must match exactly).

use metrale_kv_codec::{BlockGeometry, KvCodecId, spec};

/// 2026-10-18: Pre-registry bytes per block side, captured from `KvCacheConfig` at the base
/// commit, columns in `KvCodecId::ALL` order: bf16, fp8, nvfp4, turbo4, turbo3, turbo2, turbo8.
const GOLDEN: [((u32, u32, u32), [u64; 7]); 10] = [
    ((16, 1, 64), [2048, 1024, 576, 576, 448, 320, 1152]),
    ((16, 2, 128), [8192, 4096, 2304, 2304, 1792, 1280, 4608]),
    ((16, 4, 256), [32768, 16384, 9216, 9216, 7168, 5120, 18432]),
    ((16, 8, 128), [32768, 16384, 9216, 9216, 7168, 5120, 18432]),
    ((32, 1, 512), [32768, 16384, 9216, 9216, 7168, 5120, 18432]),
    ((16, 3, 96), [9216, 4608, 2592, 2592, 2016, 1440, 5184]),
    ((1, 1, 16), [32, 16, 9, 9, 7, 5, 18]),
    ((1, 1, 24), [48, 24, 13, 13, 10, 7, 26]),
    ((1, 1, 8), [16, 8, 4, 4, 3, 2, 8]),
    (
        (16, 16, 256),
        [131072, 65536, 36864, 36864, 28672, 20480, 73728],
    ),
];

#[test]
fn every_codec_reproduces_the_pre_registry_block_bytes() {
    assert_eq!(
        KvCodecId::ALL.len(),
        7,
        "a new codec needs its golden column"
    );
    for ((block_size, num_kv_heads, head_dim), want) in GOLDEN {
        let g = BlockGeometry {
            block_size,
            num_kv_heads,
            head_dim,
        };
        for (id, want) in KvCodecId::ALL.into_iter().zip(want) {
            assert_eq!(spec(id).block_bytes(&g), want, "{} at {g:?}", id.name());
        }
    }
}

#[test]
fn data_plus_side_is_the_block() {
    // 2026-10-18: The executors pass `data_bytes` as the offset of the side tensors, so the two
    // halves must tile the block exactly.
    let g = BlockGeometry {
        block_size: 16,
        num_kv_heads: 3,
        head_dim: 96,
    };
    for id in KvCodecId::ALL {
        let s = spec(id);
        assert_eq!(
            s.data_bytes(g.values()) + s.side_bytes(g.values()),
            s.block_bytes(&g),
            "{}",
            id.name()
        );
    }
}

/// 2026-10-18: The layout digest of every codec, frozen: a change means stored bytes changed
/// meaning, which must be a deliberate, declared edit (spill headers carry it).
const DIGESTS: [u64; 7] = [
    0xa89f_eadf_c9f3_efa5,
    0xfe83_ccdc_f382_286a,
    0xd478_1dfb_6338_f053,
    0x1603_3c16_9f52_3881,
    0x3f74_d3f0_d400_9f6e,
    0x4df8_5baf_dc74_0077,
    0xbd91_2453_1ff2_35d8,
];

#[test]
fn layout_digests_are_frozen() {
    let got: Vec<u64> = KvCodecId::ALL.iter().map(|&id| spec(id).digest()).collect();
    assert_eq!(got, DIGESTS, "got {got:#x?}");
}

#[test]
fn only_codebook_differences_share_a_digest() {
    // 2026-10-18: nvfp4 and turbo4 share a byte layout but not a basis (turbo4 is WHT-rotated),
    // so their digests differ; no two registered codecs collide.
    for (i, a) in KvCodecId::ALL.iter().enumerate() {
        for b in &KvCodecId::ALL[i + 1..] {
            assert_ne!(
                spec(*a).digest(),
                spec(*b).digest(),
                "{} and {} share a layout digest",
                a.name(),
                b.name()
            );
        }
    }
}
