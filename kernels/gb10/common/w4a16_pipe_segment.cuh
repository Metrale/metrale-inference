// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-23: A persistent segment over a W4A16 GEMV chain (P3 prototype): GLM-5.3's dense MLP /
// shared expert, gate|up GEMV -> clamped SwiGLU -> down GEMV, as the three phases of ONE launch
// (segment.cuh) instead of three launches (w4a16_gemv_pipe_m<M> over gate|up,
// glm5next_swiglu_clamp, w4a16_gemv_pipe_m<M> over the activation).
//
// Owner: gb10 kernels.
// Invariants:
// - Every byte the segment writes (gate, up, act, out) equals the three launches': phase 0 task t
//   is block t of the gate|up launch of g0 blocks (w4a16_gemv_pipe_body_at), phase 1 writes
//   glm5next_swiglu_clamp_bf16x8 of the same words glm5next_swiglu_clamp rounds (the elementwise
//   kernel's bytes, glm5next_swiglu.cuh), phase 2 task t is block t of the down launch of g2 blocks.
// - Phases 1 and 2 read what earlier phases of this launch wrote, through L2 (PipeRowsInCg,
//   __ldcg).
// - Dynamic shared memory: the larger of the two GEMV phases' (w4a16_gemv_pipe.cuh contract).
#pragma once

#include "segment.cuh"
#include "w4a16_gemv_pipe.cuh"
#include "glm5next_swiglu.cuh"

// 2026-10-23: Words (8 BF16) of the activation one phase-1 task writes.
#define DENSE_SEG_ACT_WORDS PIPE_THREADS

struct DenseSegArgs {
    const __nv_bfloat16* x;
    PipeSeg gate_up[6];
    PipeSeg down[6];
    __nv_bfloat16* act;
    float limit;
    unsigned int hidden, inter;
};

template <int M>
struct DenseSegRun {
    const DenseSegArgs& a;
    unsigned int g0, g2;
    __device__ __forceinline__ void operator()(unsigned int phase, unsigned int t,
                                               unsigned int /*ntasks*/) const {
        if (phase == 0) {
            w4a16_gemv_pipe_body_at<M, 1>(PipeRowsIn{a.x, a.hidden}, a.gate_up, a.hidden, t, g0);
        } else if (phase == 1) {
            const unsigned int words = M * a.inter / 8;
            const unsigned int w = t * DENSE_SEG_ACT_WORDS + threadIdx.x;
            if (w < words) {
                const uint4 g = __ldcg((const uint4*)a.gate_up[0].out + w);
                const uint4 u = __ldcg((const uint4*)a.gate_up[1].out + w);
                ((uint4*)a.act)[w] = glm5next_swiglu_clamp_bf16x8(g, u, a.limit);
            }
        } else {
            w4a16_gemv_pipe_body_at<M, 1>(PipeRowsInCg{a.act, a.inter}, a.down, a.inter, t, g2);
        }
    }
};

// 2026-10-23: The segment of `M` rows: g0 gate|up tasks, the SwiGLU tasks, g2 down tasks.
template <int M>
__device__ __forceinline__ void w4a16_dense_segment(SegCtl* ctl, const DenseSegArgs& a,
                                                    unsigned int g0, unsigned int g2) {
    const unsigned int act_tasks = (M * a.inter / 8 + DENSE_SEG_ACT_WORDS - 1) / DENSE_SEG_ACT_WORDS;
    const unsigned int n[SEGMENT_MAX_PHASES] = {g0, act_tasks, g2, 0};
    segment_run(ctl, n, DenseSegRun<M>{a, g0, g2});
}

// 2026-10-23: An entry's parameters: x, gate (0) and up (1) segments, down (2), act, limit,
// hidden, inter, the two GEMV phases' block counts (b0, b2) and the control words.
#define DENSE_SEG_PARAMS                                                                     \
    const __nv_bfloat16* __restrict__ x, PIPE_PARAMS(0), PIPE_PARAMS(1), PIPE_PARAMS(2),    \
        __nv_bfloat16* act, float limit, unsigned int hidden, unsigned int inter,           \
        unsigned int b0, unsigned int b2, SegCtl* ctl

#define DENSE_SEG_ARGS(a)                                                                    \
    const DenseSegArgs a = {x,                                                               \
                            {{p0, s0, g0, c0, n0}, {p1, s1, g1, c1, n1}, {}, {}, {}, {}},  \
                            {{p2, s2, g2, c2, n2}, {}, {}, {}, {}, {}},                     \
                            act, limit, hidden, inter}
