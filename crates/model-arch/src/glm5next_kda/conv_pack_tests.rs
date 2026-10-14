// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: The fused pack + conv on the mock backend: the lever's parse, the grid, and the
//! launch plan of each [`Pack`] (which launch writes `qkv_proj`, which buffers the fused kernel
//! reads and writes, and that nothing but the pack changes in `front_end`). The bytes are
//! compared on a GPU by `examples/glm5next_kda_conv_pack_bench.rs`.
//!
//! Owner: model-arch (GLM-5.3-Flash KDA).
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend, MockLaunch};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::decode_rows_tests::{
    CONV_TOKENS, CONV_TOKENS_PARTS, CONV_WINDOW, cfg, layer, seq_state, since,
};
use super::super::{Glm5NextKdaWorkspace, Pack};
use super::{conv_pack_spans, parse_conv_pack};

/// 2026-10-13: `kda_pack_qkv_bf16`'s mock handle in `decode_rows_tests::layer`.
const PACK: u64 = 0x312;

fn u32_arg(v: u32) -> MockArg {
    MockArg::Bytes(v.to_le_bytes().to_vec())
}

type Key = (u64, [u32; 3], [u32; 3], u32, Vec<MockArg>);

fn key(l: &MockLaunch) -> Key {
    (l.func, l.grid, l.block, l.shared_mem, l.args.clone())
}

fn funcs(l: &[MockLaunch]) -> Vec<u64> {
    l.iter().map(|x| x.func).collect()
}

/// 2026-10-13: Only `0`, `1` or unset are accepted; a typo fails instead of silently serving
/// the unfused path.
#[test]
fn the_lever_takes_only_zero_or_one() {
    assert_eq!(parse_conv_pack(None), Ok(false));
    assert_eq!(parse_conv_pack(Some("0")), Ok(false));
    assert_eq!(parse_conv_pack(Some("1")), Ok(true));
    for bad in ["", "on", "true", "2", " 1"] {
        assert!(parse_conv_pack(Some(bad)).is_err(), "{bad:?}");
    }
}

/// 2026-10-13: GLM-5.3 at TP=3 (conv_dim 8448, 33 channel blocks) on 48 SMs runs six row
/// spans, 198 blocks; a channel grid wider than the target still gets one span.
#[test]
fn spans_fill_the_sms() {
    assert_eq!(conv_pack_spans(48, 33), 6);
    assert_eq!(conv_pack_spans(48, 3), 64);
    assert_eq!(conv_pack_spans(48, 500), 1);
    assert_eq!(conv_pack_spans(48, 0), 192);
}

/// 2026-10-13: `front_end` under `InConv` launches exactly what it launches under `InFrontEnd`
/// less the one pack.
#[test]
fn front_end_drops_only_the_pack() {
    let gpu = MockGpuBackend::new();
    let l = layer(&gpu);
    let ws = Glm5NextKdaWorkspace::new(&gpu, &cfg(), 16).unwrap();
    let hidden = gpu.alloc(16 * cfg().hidden * 2).unwrap();
    let run = |pack| {
        let from = gpu.launch_count();
        l.front_end(&gpu, hidden, 7, &ws, pack, 0).unwrap();
        since(&gpu, from)
    };
    let (sep, fused) = (run(Pack::InFrontEnd), run(Pack::InConv));
    assert_eq!(funcs(&sep).iter().filter(|&&f| f == PACK).count(), 1);
    let sep_less_pack: Vec<Key> = sep.iter().filter(|x| x.func != PACK).map(key).collect();
    assert_eq!(fused.iter().map(key).collect::<Vec<_>>(), sep_less_pack);
}

/// 2026-10-13: `conv_tokens` under `InConv` launches the fused kernel over the three
/// projections of the `k` rows (k one `k * qkv` past q, v two), writing `qkv_proj` and
/// `conv_out` on the sequence's conv state, then the window advance over the `qkv_proj` it
/// wrote; under `InFrontEnd` the packed conv and the same advance. The fused grid is the same
/// at every row count.
#[test]
fn the_fused_conv_reads_the_parts_and_writes_the_pack() {
    let gpu = MockGpuBackend::new();
    let l = layer(&gpu);
    let c = cfg();
    let (qkv, cd) = (c.qkv_dim(), c.conv_dim());
    let ws = Glm5NextKdaWorkspace::new(&gpu, &c, 16).unwrap();
    let st = seq_state(&gpu);
    let run = |k: usize, pack| {
        let from = gpu.launch_count();
        l.conv_tokens(&gpu, k, &st, &ws, pack, 0).unwrap();
        since(&gpu, from)
    };
    let sep = run(7, Pack::InFrontEnd);
    assert_eq!(funcs(&sep), vec![CONV_TOKENS, CONV_WINDOW]);
    assert_eq!(sep[0].args[1], MockArg::Buffer(ws.qkv_proj));

    let got = run(7, Pack::InConv);
    assert_eq!(funcs(&got), vec![CONV_TOKENS_PARTS, CONV_WINDOW]);
    let (f, win) = (&got[0], &got[1]);
    let spans = conv_pack_spans(gpu.sm_count().unwrap(), 3);
    assert_eq!((f.grid, f.block), ([3, spans, 1], [256, 1, 1]));
    let want = vec![
        MockArg::Buffer(st.conv),
        MockArg::Buffer(ws.qkv_parts),
        MockArg::Buffer(ws.qkv_parts.offset(7 * qkv * 2)),
        MockArg::Buffer(ws.qkv_parts.offset(2 * 7 * qkv * 2)),
        MockArg::Buffer(l.weights.conv.weight),
        MockArg::Buffer(DevicePtr::NULL),
        MockArg::Buffer(ws.qkv_proj),
        MockArg::Buffer(ws.conv_out),
        u32_arg(cd as u32),
        u32_arg(qkv as u32),
        u32_arg(7),
        u32_arg(c.conv_kernel as u32),
        u32_arg(c.qk_channels() as u32),
        u32_arg(c.head_dim as u32),
        MockArg::Bytes(c.l2_eps.to_le_bytes().to_vec()),
    ];
    assert_eq!(f.args, want);
    assert_eq!(
        key(win),
        key(&sep[1]),
        "the advance reads the qkv_proj the fused kernel wrote"
    );

    for k in [1usize, 12, 16] {
        assert_eq!(run(k, Pack::InConv)[0].grid, f.grid, "grid at {k} rows");
    }
}

/// 2026-10-13: Without the lever (the test process does not set it), every caller packs in
/// `front_end`, and `decode_k` launches the unfused pair.
#[test]
fn unset_lever_keeps_the_pack_in_front_end() {
    let gpu = MockGpuBackend::new();
    let l = layer(&gpu);
    assert_eq!(l.pack_for(true).unwrap(), Pack::InFrontEnd);
    assert_eq!(l.pack_for(false).unwrap(), Pack::InFrontEnd);
    let ws = Glm5NextKdaWorkspace::new(&gpu, &cfg(), 16).unwrap();
    let st = seq_state(&gpu);
    let hidden = gpu.alloc(16 * cfg().hidden * 2).unwrap();
    let from = gpu.launch_count();
    l.decode_k_with(&gpu, hidden, 7, &st, &ws, &[], 0, true)
        .unwrap();
    let got = funcs(&since(&gpu, from));
    assert_eq!(got.iter().filter(|&&f| f == PACK).count(), 1);
    assert!(got.contains(&CONV_TOKENS) && !got.contains(&CONV_TOKENS_PARTS));
}
