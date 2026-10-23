// SPDX-License-Identifier: MIT OR Apache-2.0
// metrale: instantiates w4a16_gemv (w4a16_gemv_pipe.cuh)

// 2026-10-23: glm5next_swiglu_down_pipe_m<M>: GLM-5.3's shared-expert down projection with the
// clamped SwiGLU folded into its input stage: w4a16_gemv_pipe_m<M> over the rows
// bf16(swiglu_clamp(gate, up)) (glm5next_swiglu.cuh), computed while staging instead of read
// from the activation buffer glm5next_swiglu_clamp writes. One launch and one buffer round trip
// fewer per layer.
//
// Owner: gb10 kernels.
// Invariants:
// - Every output's bytes equal glm5next_swiglu_clamp then w4a16_gemv_pipe_m<M> over its output:
//   the staged words are that kernel's bytes (glm5next_swiglu_clamp_bf16x8, same expression and
//   rounding under the common tree's flags) and the rest is the template's.
// - gate and up: M contiguous BF16 rows of K each, 16-byte aligned (the dense MLP scratch).

#include "w4a16_gemv_pipe.cuh"
#include "glm5next_swiglu.cuh"

// 2026-10-23: The input stage: word u of row m of the SwiGLU of gate and up.
struct SwigluIn {
    const __nv_bfloat16* __restrict__ gate;
    const __nv_bfloat16* __restrict__ up;
    float limit;
    unsigned int K;
    __device__ __forceinline__ uint4 operator()(unsigned int m, unsigned int u) const {
        const unsigned long long row = (unsigned long long)m * K;
        return glm5next_swiglu_clamp_bf16x8(((const uint4*)(gate + row))[u],
                                            ((const uint4*)(up + row))[u], limit);
    }
};

// 2026-10-23: Each entry names its kernel directly so the build's entry-point scan finds it.
extern "C" __global__ void __launch_bounds__(PIPE_THREADS, 4) glm5next_swiglu_down_pipe_m1(
    const __nv_bfloat16* __restrict__ gate, const __nv_bfloat16* __restrict__ up, float limit,
    PIPE_SEG_PARAMS, unsigned int K) {
    PIPE_SEG_ARRAY(seg);
    w4a16_gemv_pipe_body<1, 1>(SwigluIn{gate, up, limit, K}, seg, K);
}

extern "C" __global__ void __launch_bounds__(PIPE_THREADS, 4) glm5next_swiglu_down_pipe_m2(
    const __nv_bfloat16* __restrict__ gate, const __nv_bfloat16* __restrict__ up, float limit,
    PIPE_SEG_PARAMS, unsigned int K) {
    PIPE_SEG_ARRAY(seg);
    w4a16_gemv_pipe_body<2, 1>(SwigluIn{gate, up, limit, K}, seg, K);
}
