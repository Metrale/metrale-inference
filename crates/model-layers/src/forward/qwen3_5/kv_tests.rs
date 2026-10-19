// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: The Metal KV cache against the forward it replaced. The legacy sizes, kernel
//! names and launch sequences below are transcribed from the deleted `MetalKvDtype` match arms
//! and the turbo/bf16 branches of `forward_full_attention`, written out literally; the code
//! under test derives them from the codec specs and the Metal kernel table instead.

use metrale_error::{Error, ErrorCode};
use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend};
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};
use metrale_kv_codec::{
    KernelName, KvPair, ResolvedKvKernels, metal_kv_kernel_names, metal_kv_pairs,
};

use super::{KvStepDims, KvStepIo, LayerKvCache, MetalKvKernels, append_and_attend};
use crate::forward::qwen3_5::Qwen35Kernels;

/// 2026-10-18: The three branches the legacy forward had.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Legacy {
    Bf16,
    SymTurbo,
    Bf16KTurboV,
}

/// 2026-10-18: Every dtype the legacy enum parsed, with its branch.
const LEGACY: &[(&str, Legacy)] = &[
    ("bf16", Legacy::Bf16),
    ("turbo8", Legacy::SymTurbo),
    ("turbo4", Legacy::SymTurbo),
    ("turbo3", Legacy::SymTurbo),
    ("turbo2", Legacy::SymTurbo),
    ("bf16k_turbo4v", Legacy::Bf16KTurboV),
    ("bf16k_turbo3v", Legacy::Bf16KTurboV),
    ("bf16k_turbo2v", Legacy::Bf16KTurboV),
];

fn pair(s: &str) -> KvPair {
    s.parse().expect("accepted spelling")
}

fn code(e: &anyhow::Error) -> Option<ErrorCode> {
    e.downcast_ref::<Error>().map(Error::code)
}

/// 2026-10-18: The deleted `LayerKvCache::alloc` size match, verbatim.
fn legacy_bytes(dtype: &str, max_seq: u32, kv_dim: u32) -> (usize, usize, usize, usize) {
    let n = (max_seq * kv_dim) as usize;
    let s = (max_seq * kv_dim / 16) as usize;
    match dtype {
        "bf16" => (n * 2, n * 2, 0, 0),
        "turbo8" => (n, n, s * 2, s * 2),
        "turbo4" => (n / 2, n / 2, s, s),
        "turbo3" => (n * 3 / 8, n * 3 / 8, s, s),
        "turbo2" => (n / 4, n / 4, s, s),
        "bf16k_turbo4v" => (n * 2, n / 2, 0, s),
        "bf16k_turbo3v" => (n * 2, n * 3 / 8, 0, s),
        "bf16k_turbo2v" => (n * 2, n / 4, 0, s),
        other => panic!("{other} was not a legacy dtype"),
    }
}

/// 2026-10-18: Each legacy dtype allocates the same four buffers at the same sizes, with a
/// scale buffer exactly where the legacy cache had one.
#[test]
fn alloc_matches_the_legacy_sizes() {
    for &(name, _) in LEGACY {
        for (max_seq, kv_dim) in [(1, 16), (7, 48), (37, 1024), (4100, 512)] {
            let gpu = MockGpuBackend::new();
            let kv = LayerKvCache::alloc(&gpu, pair(name), max_seq, kv_dim).unwrap();
            let len = |p: Option<DevicePtr>| p.map_or(0, |p| gpu.read_alloc(p).unwrap().len());
            let got = (
                len(Some(kv.k)),
                len(Some(kv.v)),
                len(kv.k_scales),
                len(kv.v_scales),
            );
            assert_eq!(
                got,
                legacy_bytes(name, max_seq, kv_dim),
                "{name} {max_seq}x{kv_dim}"
            );
            assert_eq!(kv.capacity, max_seq);
        }
    }
}

/// 2026-10-18: A pair the Metal table does not serve is refused before any buffer exists.
#[test]
fn alloc_refuses_pairs_metal_does_not_serve() {
    for name in ["fp8", "nvfp4", "fp8k_turbo4v", "turbo4k_turbo3v"] {
        let gpu = MockGpuBackend::new();
        let err = LayerKvCache::alloc(&gpu, pair(name), 8, 64)
            .err()
            .unwrap_or_else(|| panic!("{name} allocated"));
        assert_eq!(code(&err), Some(ErrorCode::KvCapabilityMissing), "{name}");
    }
}

/// 2026-10-18: A quantized side needs its group to divide KV_DIM; a bf16 cache does not.
#[test]
fn alloc_refuses_a_kv_dim_the_group_does_not_divide() {
    let gpu = MockGpuBackend::new();
    assert!(LayerKvCache::alloc(&gpu, pair("bf16"), 4, 24).is_ok());
    for name in ["turbo4", "turbo8", "bf16k_turbo2v"] {
        assert!(
            LayerKvCache::alloc(&gpu, pair(name), 4, 24).is_err(),
            "{name}"
        );
    }
}

/// 2026-10-18: The legacy `Qwen35Kernels::resolve` lookups that are not KV kernels, verbatim.
/// Its KV lookups are golden-tested against the legacy names in the codec crate's
/// `metal_table_tests`, so here they come from the Metal table.
const LEGACY_NON_KV_LOOKUPS: &[(&str, &str)] = &[
    ("rms_norm", "rms_norm"),
    ("rope_apply", "rope_apply"),
    ("sigmoid_gate", "sigmoid_gate"),
    ("add_rms_norm", "add_rms_norm"),
    ("qwen35_qkv_split", "qwen35_qkv_split"),
    ("causal_conv1d_update_l2norm", "causal_conv1d_update_l2norm"),
    ("gdn_helpers", "gdn_compute_gate"),
    ("gdn_helpers", "sigmoid_bf16_to_f32"),
    ("gated_delta_rule_decode", "gated_delta_rule_decode"),
];

/// 2026-10-18: Startup looks up the legacy non-KV kernels and every kernel of every pair the
/// Metal table serves: nothing missing, so a broken kernel still fails at startup, and nothing
/// else.
#[test]
fn resolve_looks_up_the_legacy_kernel_set() {
    let gpu = MockGpuBackend::new();
    Qwen35Kernels::resolve(&gpu).unwrap();
    let mut got = gpu.kernel_lookups_snapshot();
    got.sort();
    got.dedup();
    let name = |n: KernelName| (n.module.to_owned(), n.func.to_owned());
    let mut want: Vec<(String, String)> = LEGACY_NON_KV_LOOKUPS
        .iter()
        .map(|&(m, f)| (m.to_owned(), f.to_owned()))
        .collect();
    for p in metal_kv_pairs() {
        let k = metal_kv_kernel_names(p).unwrap();
        want.extend([name(k.write), name(k.decode)]);
        want.extend(k.wht.into_iter().flat_map(|(a, b)| [name(a), name(b)]));
    }
    want.sort();
    want.dedup();
    assert_eq!(
        got.len(),
        27,
        "the legacy resolve looked up 27 distinct kernels"
    );
    assert_eq!(got, want);
}

/// 2026-10-18: A missing turbo kernel fails startup, as the legacy hard lookup did.
#[test]
fn a_missing_metal_kv_kernel_fails_resolve() {
    let gpu = MockGpuBackend::new();
    let k = metal_kv_kernel_names(pair("bf16k_turbo3v")).unwrap();
    gpu.deny_kernel(k.decode.module, k.decode.func);
    assert!(Qwen35Kernels::resolve(&gpu).is_err());
}

/// 2026-10-18: The resolved set refuses a pair the Metal table does not serve.
#[test]
fn resolved_kernels_refuse_an_unserved_pair() {
    let gpu = MockGpuBackend::new();
    let kernels = MetalKvKernels::resolve(&gpu).unwrap();
    let err = kernels
        .get(pair("fp8"))
        .expect_err("fp8 has no Metal kernels");
    assert_eq!(code(&err), Some(ErrorCode::KvCapabilityMissing));
    for &(name, _) in LEGACY {
        assert!(kernels.get(pair(name)).is_ok(), "{name}");
    }
}

const WRITE: u64 = 11;
const DECODE: u64 = 12;
const FWD: u64 = 13;
const INV: u64 = 14;

type Launch = (u64, [u32; 3], [u32; 3], Vec<MockArg>);

fn u(x: u32) -> MockArg {
    MockArg::Bytes(x.to_le_bytes().to_vec())
}

fn f(x: f32) -> MockArg {
    MockArg::Bytes(x.to_le_bytes().to_vec())
}

fn buf(p: DevicePtr) -> MockArg {
    MockArg::Buffer(p)
}

/// 2026-10-18: The launches the legacy forward issued for one token, branch by branch.
fn legacy_launches(kind: Legacy, kv: &LayerKvCache, io: &KvStepIo, d: &KvStepDims) -> Vec<Launch> {
    let (nh, nkv, hd) = (d.num_heads, d.num_kv_heads, d.head_dim);
    let wht = |h, rows, x| (h, [rows, 1, 1], [32, 1, 1], vec![u(hd), buf(x)]);
    let head = vec![
        u(nkv),
        u(hd),
        u(d.cache_pos),
        buf(io.k),
        buf(io.v),
        buf(kv.k),
        buf(kv.v),
    ];
    let attn_head = vec![u(d.seq_len_attn), u(nh), u(nkv), u(hd), f(d.scale)];
    let grid = [(nkv * hd / 16).div_ceil(64), 1, 1];
    match kind {
        Legacy::Bf16 => {
            let mut attn = attn_head;
            attn.extend([buf(io.q), buf(kv.k), buf(kv.v), buf(io.out)]);
            vec![
                (WRITE, [hd, nkv, 1], [1, 1, 1], head),
                (DECODE, [nh, 1, 1], [32, 1, 1], attn),
            ]
        }
        Legacy::SymTurbo => {
            let (ks, vs) = (kv.k_scales.unwrap(), kv.v_scales.unwrap());
            let mut append = head;
            append.extend([buf(ks), buf(vs)]);
            let mut attn = attn_head;
            attn.extend([f(d.sparse_v), buf(io.q), buf(kv.k), buf(kv.v)]);
            attn.extend([buf(ks), buf(vs), buf(io.out)]);
            vec![
                wht(FWD, nkv, io.k),
                wht(FWD, nkv, io.v),
                (WRITE, grid, [64, 1, 1], append),
                wht(FWD, nh, io.q),
                (DECODE, [nh, 1, 1], [32, 1, 1], attn),
                wht(INV, nh, io.out),
            ]
        }
        Legacy::Bf16KTurboV => {
            let vs = kv.v_scales.unwrap();
            let mut append = head;
            append.push(buf(vs));
            let mut attn = attn_head;
            attn.extend([f(d.sparse_v), buf(io.q), buf(kv.k), buf(kv.v)]);
            attn.extend([buf(vs), buf(io.out)]);
            vec![
                wht(FWD, nkv, io.v),
                (WRITE, grid, [64, 1, 1], append),
                (DECODE, [nh, 1, 1], [32, 1, 1], attn),
                wht(INV, nh, io.out),
            ]
        }
    }
}

/// 2026-10-18: For every legacy dtype the KV step issues the legacy launches: same kernels, same
/// order, grids, blocks and arguments, so the Metal forward is unchanged byte for byte.
#[test]
fn append_and_attend_issues_the_legacy_launches() {
    let kernels = ResolvedKvKernels {
        write: KernelHandle(WRITE),
        decode: KernelHandle(DECODE),
        decode_512: None,
        splitk: None,
        prefill: None,
        wht: Some((KernelHandle(FWD), KernelHandle(INV))),
    };
    let io = KvStepIo {
        q: DevicePtr(0x1_0000),
        k: DevicePtr(0x2_0000),
        v: DevicePtr(0x3_0000),
        out: DevicePtr(0x4_0000),
    };
    let d = KvStepDims {
        num_heads: 12,
        num_kv_heads: 5,
        head_dim: 256,
        cache_pos: 6,
        seq_len_attn: 7,
        scale: 0.0625,
        sparse_v: 1e-3,
    };
    for &(name, kind) in LEGACY {
        let gpu = MockGpuBackend::new();
        let kv = LayerKvCache::alloc(&gpu, pair(name), 8, d.num_kv_heads * d.head_dim).unwrap();
        append_and_attend(&gpu, &kernels, &kv, &io, &d, 3).unwrap();
        let got: Vec<Launch> = gpu
            .launches_snapshot()
            .into_iter()
            .map(|l| {
                assert_eq!((l.shared_mem, l.stream), (0, 3), "{name}");
                (l.func, l.grid, l.block, l.args)
            })
            .collect();
        assert_eq!(got, legacy_launches(kind, &kv, &io, &d), "{name}");
    }
}
