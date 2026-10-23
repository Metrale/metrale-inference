// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: The pipelined segmented GEMV launch: its constants against the kernel source, the
//! block cap at the GLM-5.3 serving shapes, and the launch it makes on a recording mock backend.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend};

use super::*;
use crate::layers::ops::W4A16_GEMV_SEG_MAX_ROWS;
use crate::weight_map::QuantizedWeight;

const CUH: &str = include_str!("../../../../../kernels/gb10/common/w4a16_gemv_pipe.cuh");
const CU: &str = include_str!("../../../../../kernels/gb10/common/w4a16_gemv_pipe.cu");
/// 2026-10-23: The other instantiation, whose stage takes two inputs and an F32
/// ([`PipeInput::Pair`]).
const FOLD: &str = include_str!("../../../../../kernels/gb10/common/glm5next_swiglu_down.cu");

/// 2026-10-21: The host's constants are the kernel's: block size, segment count, entries and
/// their launch bounds, and `w4a16_gemv_seg`'s staging layout (the shared-memory size it reuses).
/// 2026-10-23: The template is `w4a16_gemv_pipe.cuh`; each entry file instantiates it.
#[test]
fn host_constants_match_the_kernel() {
    assert!(CUH.contains(&format!(
        "#define PIPE_THREADS {}\n",
        W4A16_GEMV_PIPE_WARPS * 32
    )));
    assert!(CUH.contains(&format!("#define PIPE_WARPS {W4A16_GEMV_PIPE_WARPS}\n")));
    assert_eq!(CUH.matches("PIPE_PARAMS(").count(), W4A16_GEMV_SEG_MAX + 1);
    let bounds = format!("__launch_bounds__(PIPE_THREADS, {W4A16_GEMV_PIPE_BLOCKS_PER_SM})");
    for m in 1..=2 {
        assert!(CU.contains(&format!("{bounds} w4a16_gemv_pipe_m{m}(")));
        assert!(CU.contains(&format!(
            "w4a16_gemv_pipe_body<{m}, 1>(PipeRowsIn{{A, K}}, seg, K);"
        )));
        assert!(FOLD.contains(&format!("{bounds} glm5next_swiglu_down_pipe_m{m}(")));
        assert!(FOLD.contains(&format!(
            "w4a16_gemv_pipe_body<{m}, 1>(SwigluIn{{gate, up, limit, K}}"
        )));
        assert_eq!(
            w4a16_gemv_pipe_entry(m),
            Some(format!("w4a16_gemv_pipe_m{m}")).as_deref()
        );
    }
    assert_eq!(CU.matches("extern \"C\" __global__").count(), 2);
    assert!(CU.contains("const __nv_bfloat16* __restrict__ A, PIPE_SEG_PARAMS, unsigned int K)"));
    assert!(FOLD.contains(
        "const __nv_bfloat16* __restrict__ gate, const __nv_bfloat16* __restrict__ up, float limit,\n    PIPE_SEG_PARAMS, unsigned int K)"
    ));
    assert_eq!(w4a16_gemv_pipe_entry(0), None);
    assert_eq!(w4a16_gemv_pipe_entry(3), None);
    assert!(
        CUH.contains(
            "sm[256u * j + 64u * (2u * (r & 1u) + h) + (r >> 1)] = in((unsigned int)m, u);"
        )
    );
    assert!(CUH.contains("return ((const uint4*)(A + (unsigned long long)m * K))[u];"));
    assert!(CUH.contains("const unsigned int J = (K16 + 127) / 128;"));
    assert!(CUH.contains("float* s_lut = (float*)(s_a + 256u * J * M);"));
    assert!(CUH.contains("uint4* sm = s_a + 256u * J * m;"));
    for src in [CUH, CU, FOLD] {
        assert!(!src.contains("__shared__ float"), "no static shared memory");
    }
}

/// 2026-10-21: The contract is `w4a16_gemv_seg`'s at a row count with an entry.
#[test]
fn contract_is_the_seg_contract_at_an_entry() {
    let six = [2816, 2816, 2816, 128, 22, 128];
    assert!(w4a16_gemv_pipe_ok(4096, &six, 1));
    assert!(w4a16_gemv_pipe_ok(4096, &six, 2));
    assert!(w4a16_gemv_pipe_ok(768, &[4096], 2));
    assert!(w4a16_gemv_pipe_ok(11264, &[4096], 2));
    for m in 3..=W4A16_GEMV_SEG_MAX_ROWS {
        assert!(w4a16_gemv_seg_rows_ok(4096, &six, m));
        assert!(!w4a16_gemv_pipe_ok(4096, &six, m), "no m{m} entry");
    }
    assert!(!w4a16_gemv_pipe_ok(4096, &six, 0));
    assert!(!w4a16_gemv_pipe_ok(4096 + 16, &six, 2));
    assert!(!w4a16_gemv_pipe_ok(4096, &[], 2));
    assert!(!w4a16_gemv_pipe_ok(4096, &[1; 7], 2));
}

/// 2026-10-21: On 48 SMs: the shared gate|up (192 blocks), shared down and KDA o (512 blocks,
/// 2.67 waves) run on one resident wave of 192 blocks; the KDA six-group (1091 blocks), DSA
/// q_absorb (1408) and o_absorb (24 KB of staged input per row: four blocks do not fit) one
/// output per warp. The thresholds sit exactly at their edges.
#[test]
fn cap_at_the_serving_shapes() {
    let six = [2816, 2816, 2816, 128, 22, 128];
    for m in 1..=2 {
        assert_eq!(w4a16_gemv_pipe_cap(&[768, 768], 4096, m, 48), 192);
        assert_eq!(w4a16_gemv_pipe_cap(&[4096], 768, m, 48), 192);
        assert_eq!(w4a16_gemv_pipe_cap(&[4096], 2816, m, 48), 192);
        assert_eq!(w4a16_gemv_pipe_cap(&six, 4096, m, 48), u32::MAX);
        assert_eq!(w4a16_gemv_pipe_cap(&[11264], 1536, m, 48), u32::MAX);
    }
    // 2026-10-21: o_absorb stages 24 KB per row: 4 x (24576 + 64 + 1024) > 100 KB at one row.
    assert_eq!(w4a16_gemv_pipe_cap(&[4096], 11264, 1, 48), u32::MAX);
    assert_eq!(w4a16_gemv_pipe_cap(&[4096], 10240, 1, 48), 192);
    // 2026-10-21: Three waves of 192 blocks are 4608 rows; one more block's worth is uncapped.
    assert_eq!(w4a16_gemv_pipe_cap(&[4608], 4096, 2, 48), 192);
    assert_eq!(w4a16_gemv_pipe_cap(&[4609], 4096, 2, 48), u32::MAX);
    // 2026-10-21: Four blocks of K 6144 at 2 rows need 4 x (24576 + 64 + 1024) > 100 KB.
    assert_eq!(w4a16_gemv_pipe_cap(&[4096], 4096 + 2048, 1, 48), 192);
    assert_eq!(w4a16_gemv_pipe_cap(&[4096], 4096 + 2048, 2, 48), u32::MAX);
    assert_eq!(w4a16_gemv_pipe_cap(&[4096], 4096, 2, 0), u32::MAX);
    assert_eq!(w4a16_gemv_pipe_blocks(1536, 192), 192);
    assert_eq!(w4a16_gemv_pipe_blocks(4096, 192), 192);
    assert_eq!(w4a16_gemv_pipe_blocks(22, 192), 3);
    assert_eq!(w4a16_gemv_pipe_blocks(1, 192), 1);
}

fn bytes(a: &MockArg) -> &[u8] {
    match a {
        MockArg::Bytes(b) => b,
        MockArg::Buffer(p) => panic!("expected bytes, got buffer {p}"),
    }
}

fn ptr(a: &MockArg) -> DevicePtr {
    match a {
        MockArg::Buffer(p) => *p,
        MockArg::Bytes(_) => panic!("expected a buffer"),
    }
}

/// 2026-10-21: The shared down at 2 rows: the `m2` entry, the capped grid, two rows of staged
/// input, the segment in slot 0, the empty slots, then K; a refused shape, row count or cap
/// launches nothing.
#[test]
fn the_launch_carries_the_policy_grid() {
    let gpu = MockGpuBackend::new();
    let segs = [W4a16Seg {
        weight: QuantizedWeight {
            weight: DevicePtr(0x1000),
            weight_scale: DevicePtr(0x1100),
            weight_scale_2: 0.5,
            ..QuantizedWeight::null()
        },
        output: DevicePtr(0x9000),
        n: 4096,
    }];
    let a = DevicePtr(0x5000);
    w4a16_gemv_pipe(&gpu, a, &segs, 2, 768, 48, 7).unwrap();
    let l = gpu.launches_snapshot();
    assert_eq!(l.len(), 1);
    assert_eq!(
        (l[0].grid, l[0].block, l[0].shared_mem, l[0].stream),
        ([192, 1, 1], [256, 1, 1], 2 * 4096 + 64, 7)
    );
    assert_eq!(l[0].args.len(), 1 + 5 * W4A16_GEMV_SEG_MAX + 1);
    assert_eq!(ptr(&l[0].args[0]), a);
    assert_eq!(ptr(&l[0].args[1]), segs[0].weight.weight);
    assert_eq!(ptr(&l[0].args[2]), segs[0].weight.weight_scale);
    assert_eq!(bytes(&l[0].args[3]), 0.5f32.to_le_bytes());
    assert_eq!(ptr(&l[0].args[4]), segs[0].output);
    assert_eq!(bytes(&l[0].args[5]), 4096u32.to_le_bytes());
    for i in 1..W4A16_GEMV_SEG_MAX {
        assert_eq!(ptr(&l[0].args[1 + 5 * i]), DevicePtr::NULL);
        assert_eq!(bytes(&l[0].args[5 + 5 * i]), 0u32.to_le_bytes());
    }
    assert_eq!(bytes(&l[0].args[31]), 768u32.to_le_bytes());
    assert!(
        gpu.kernel_lookups_snapshot()
            .contains(&(W4A16_GEMV_PIPE_MODULE.into(), "w4a16_gemv_pipe_m2".into()))
    );
    w4a16_gemv_pipe_at(&gpu, a, &segs, 1, 768, u32::MAX, 7).unwrap();
    let l = gpu.launches_snapshot();
    assert_eq!((l[1].grid, l[1].shared_mem), ([512, 1, 1], 4096 + 64));
    assert!(w4a16_gemv_pipe(&gpu, a, &segs, 3, 768, 48, 7).is_err());
    assert!(w4a16_gemv_pipe(&gpu, a, &segs, 2, 768 + 16, 48, 7).is_err());
    assert!(w4a16_gemv_pipe_at(&gpu, a, &segs, 2, 768, 0, 7).is_err());
    assert_eq!(gpu.launches_snapshot().len(), 2);
}

/// 2026-10-23: A staged `Pair` input pushes its two pointers and the F32 ahead of the segments
/// (the fold entry's parameter order), on the given kernel and the policy grid; the segments and
/// K follow as for the plain entry.
#[test]
fn a_pair_input_leads_the_arguments() {
    let gpu = MockGpuBackend::new();
    let segs = [W4a16Seg {
        weight: QuantizedWeight {
            weight: DevicePtr(0x1000),
            weight_scale: DevicePtr(0x1100),
            weight_scale_2: 0.5,
            ..QuantizedWeight::null()
        },
        output: DevicePtr(0x9000),
        n: 4096,
    }];
    let k = metrale_gpu_runtime::gpu::KernelHandle(0x77);
    let input = PipeInput::Pair(DevicePtr(0x5000), DevicePtr(0x6000), 10.0);
    let cap = w4a16_gemv_pipe_cap(&[4096], 768, 2, 48);
    w4a16_gemv_pipe_staged(&gpu, k, input, &segs, 2, 768, cap, 7).unwrap();
    let l = gpu.launches_snapshot();
    assert_eq!(l.len(), 1);
    assert_eq!(l[0].func, k.0);
    assert_eq!((l[0].grid, l[0].shared_mem), ([192, 1, 1], 2 * 4096 + 64));
    assert_eq!(l[0].args.len(), 3 + 5 * W4A16_GEMV_SEG_MAX + 1);
    assert_eq!(ptr(&l[0].args[0]), DevicePtr(0x5000));
    assert_eq!(ptr(&l[0].args[1]), DevicePtr(0x6000));
    assert_eq!(bytes(&l[0].args[2]), 10.0f32.to_le_bytes());
    assert_eq!(ptr(&l[0].args[3]), segs[0].weight.weight);
    assert_eq!(bytes(&l[0].args[33]), 768u32.to_le_bytes());
    assert!(w4a16_gemv_pipe_staged(&gpu, k, input, &segs, 3, 768, cap, 7).is_err());
    assert_eq!(gpu.launches_snapshot().len(), 1);
}
