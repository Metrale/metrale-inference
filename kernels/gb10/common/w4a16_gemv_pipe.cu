// SPDX-License-Identifier: MIT OR Apache-2.0
// metrale: instantiates w4a16_gemv (w4a16_gemv_pipe.cuh)

// 2026-10-21: w4a16_gemv_pipe_m<M>: the w4a16_gemv_pipe template (w4a16_gemv_pipe.cuh, which
// holds the method, the measurements and the invariants) over M contiguous input rows of K.
// 2026-10-23: The template moved to the header unchanged but for the input stage, PipeRowsIn
// here, so other instantiations (glm5next_swiglu_down.cu) share every output's arithmetic.
//
// Owner: gb10 kernels.
// Invariants: as w4a16_gemv_pipe.cuh; A 16-byte aligned, M contiguous rows of K.

#include "w4a16_gemv_pipe.cuh"

// 2026-10-21: Each entry names its kernel directly so the build's entry-point scan finds it.
extern "C" __global__ void __launch_bounds__(PIPE_THREADS, 4) w4a16_gemv_pipe_m1(
    const __nv_bfloat16* __restrict__ A, PIPE_SEG_PARAMS, unsigned int K) {
    PIPE_SEG_ARRAY(seg);
    w4a16_gemv_pipe_body<1, 1>(PipeRowsIn{A, K}, seg, K);
}

extern "C" __global__ void __launch_bounds__(PIPE_THREADS, 4) w4a16_gemv_pipe_m2(
    const __nv_bfloat16* __restrict__ A, PIPE_SEG_PARAMS, unsigned int K) {
    PIPE_SEG_ARRAY(seg);
    w4a16_gemv_pipe_body<2, 1>(PipeRowsIn{A, K}, seg, K);
}
