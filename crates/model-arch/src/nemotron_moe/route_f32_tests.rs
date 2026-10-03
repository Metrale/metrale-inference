// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: Which kernels route the tokens of a layer whose checkpoint stores the router
//! FP32: on decode, on the one-token prefill (the per-token path) and on the multi-token prefill
//! (the sorted path), only the FP32 router GEMM reads the router weight and only the FP32 top-k
//! routes; no BF16 router GEMV/GEMM and no BF16 top-k runs. The BF16 router is the control: the
//! same filters see its routing launches.
//!
//! Owner: model-arch (Nemotron-H).
//! Invariants: none beyond the types.

use metrale_cache::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};
use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend, MockLaunch};
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};
use metrale_model_layers::layer::{
    EmptyLayerState, ForwardContext, MoeLoraRoute, TransformerLayer,
};
use metrale_model_layers::layers::ops::{DerivedWeights, GemmDispatch, ModelLevers, ModelStats};

use super::RouterF32;
use crate::nemotron_moe::prefill_weights::tests::{config, layer};

const ROUTER_F32: u64 = 0xF32A;
const TOPK_F32: u64 = 0xF32B;
const TOPK_BF16: u64 = 0xB16B;
const TOPK_BF16_BATCHED: u64 = 0xB16C;

fn reads(launch: &MockLaunch, p: DevicePtr) -> bool {
    launch.args.iter().any(|a| match a {
        MockArg::Buffer(b) => *b == p,
        MockArg::Bytes(b) => b.as_slice() == p.0.to_le_bytes().as_slice(),
    })
}

/// 2026-10-02: What one call routed with: the launches that read the router weight, and the
/// top-k launches by handle.
struct Routing {
    weight_readers: Vec<MockLaunch>,
    topk: Vec<u64>,
    logits_f32_readers: usize,
}

/// 2026-10-02: Run decode (`tokens == 0`) or a prefill of `tokens` on a fresh layer.
fn route(gate_f32: bool, tokens: usize) -> Routing {
    let gpu = MockGpuBackend::new();
    let config = config();
    let mut l = layer(&gpu, &config, gate_f32);
    // 2026-10-02: The mock returns one handle for every kernel name; distinct handles here
    // tell the routing kernels apart.
    l.topk_sigmoid_k = KernelHandle(TOPK_BF16);
    l.topk_sigmoid_batched_k = KernelHandle(TOPK_BF16_BATCHED);
    if let Some(r) = l.router_f32.as_mut() {
        *r = RouterF32 {
            gemm: KernelHandle(ROUTER_F32),
            topk: KernelHandle(TOPK_F32),
        };
    }
    let buffers = BufferArena::new(&config, 8, 256, 256, 8, &gpu).unwrap();
    // 2026-10-02: The prefill GEMMs size their grids by the N tile a CUDA module publishes.
    gpu.set_kernel_n_tile(KernelHandle(0xDEAD), 64);
    let (dispatch, derived) = (GemmDispatch::defaults(), DerivedWeights::new());
    let (levers, stats) = (ModelLevers::defaults(), ModelStats::new());
    let ctx = ForwardContext {
        buffers: &buffers,
        hc_row_offset: 0,
        gpu: &gpu,
        config: &config,
        dispatch: &dispatch,
        derived: &derived,
        levers: &levers,
        stats: &stats,
        attn_metadata: None,
        profile: false,
        comm: None,
        graph_capture: false,
        decode_step: tokens == 0,
        gdn_exact_replay: false,
        gdn_write_on_accept: false,
        token_ids: None,
        host_token_ids: None,
        routed_lora_layers: None,
        midchunk_capture: None,
        moe_lora_route: MoeLoraRoute::Fold,
    };
    let (hidden, residual) = (buffers.hidden_states(), buffers.residual());
    let start = gpu.launch_count();
    if tokens == 0 {
        l.decode_inner(hidden, residual, &ctx, 7).unwrap();
    } else {
        let kv = KvCacheConfig {
            block_size: 16,
            num_kv_heads: 1,
            head_dim: 16,
            num_layers: 1,
            dtype: KvCacheDtype::Fp8,
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
        };
        let mut kv = PagedKvCache::new(kv, 2, &gpu).unwrap();
        let (mut a, mut b, mut c) = (vec![], vec![], vec![]);
        l.prefill(
            hidden,
            residual,
            tokens,
            &mut EmptyLayerState,
            &mut kv,
            0,
            &mut a,
            &mut b,
            &mut c,
            0,
            &ctx,
            7,
        )
        .unwrap();
    }
    let launches = gpu.launches_snapshot()[start..].to_vec();
    let logits_f32 = buffers.gate_logits_f32();
    Routing {
        weight_readers: launches
            .iter()
            .filter(|x| reads(x, l.weights.gate.weight))
            .cloned()
            .collect(),
        topk: launches
            .iter()
            .map(|x| x.func)
            .filter(|f| [TOPK_F32, TOPK_BF16, TOPK_BF16_BATCHED].contains(f))
            .collect(),
        logits_f32_readers: launches.iter().filter(|x| reads(x, logits_f32)).count(),
    }
}

/// 2026-10-02: Path A: decode, the per-token prefill and the sorted prefill each route once,
/// through the FP32 router over every token, and the FP32 top-k reads its FP32 logits.
#[test]
fn an_fp32_router_routes_every_path_in_fp32() {
    for tokens in [0usize, 1, 4] {
        let r = route(true, tokens);
        let rows = tokens.max(1) as u32;
        assert_eq!(
            r.weight_readers.len(),
            1,
            "tokens={tokens}: one router read"
        );
        let g = &r.weight_readers[0];
        assert_eq!(g.func, ROUTER_F32, "tokens={tokens}: the FP32 router GEMM");
        assert_eq!(
            g.grid,
            [1, rows, 1],
            "tokens={tokens}: 2 experts, one row per token"
        );
        assert_eq!(
            r.topk,
            vec![TOPK_F32],
            "tokens={tokens}: only the FP32 top-k"
        );
        assert_eq!(
            r.logits_f32_readers, 2,
            "tokens={tokens}: GEMM writes, top-k reads"
        );
    }
}

/// 2026-10-02: Path B (control): a BF16 router routes through the BF16 kernels, so the filters
/// above do see routing launches; the FP32 kernels never run.
#[test]
fn a_bf16_router_keeps_the_bf16_routing() {
    for (tokens, topk) in [(0usize, TOPK_BF16), (1, TOPK_BF16), (4, TOPK_BF16_BATCHED)] {
        let r = route(false, tokens);
        assert_eq!(r.weight_readers.len(), 1, "tokens={tokens}");
        assert_ne!(r.weight_readers[0].func, ROUTER_F32, "tokens={tokens}");
        assert_eq!(r.topk, vec![topk], "tokens={tokens}");
        assert_eq!(r.logits_f32_readers, 0, "tokens={tokens}");
    }
}
