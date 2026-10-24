// SPDX-License-Identifier: MIT OR Apache-2.0
// metrale: instantiates glm_dense_seg (w4a16_pipe_segment.cuh)

// 2026-10-23: glm5next_dense_seg_pipe_m<M>: GLM-5.3's dense MLP / shared expert (gate|up, clamped
// SwiGLU, down) as one persistent segment (w4a16_pipe_segment.cuh over segment.cuh and the pipe
// template): the P3 prototype of DESIGN-fuser-megakernel.md §4.2, METRALE_GLM_DENSE_SEGMENT.
//
// Owner: gb10 kernels.
// Invariants: as w4a16_pipe_segment.cuh (every byte equals the three launches it replaces).

#include "w4a16_gemv_pipe.cuh"
#include "w4a16_pipe_segment.cuh"

// 2026-10-23: Each entry names its kernel directly so the build's entry-point scan finds it.
extern "C" __global__ void __launch_bounds__(PIPE_THREADS, 4) glm5next_dense_seg_pipe_m1(
    DENSE_SEG_PARAMS) {
    DENSE_SEG_ARGS(a);
    w4a16_dense_segment<1>(ctl, a, b0, b2, sm_budget);
}

extern "C" __global__ void __launch_bounds__(PIPE_THREADS, 4) glm5next_dense_seg_pipe_m2(
    DENSE_SEG_PARAMS) {
    DENSE_SEG_ARGS(a);
    w4a16_dense_segment<2>(ctl, a, b0, b2, sm_budget);
}
