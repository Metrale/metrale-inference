// SPDX-License-Identifier: MIT OR Apache-2.0
// 2026-10-24: fatbin_tests.rs fixture: one module-scope global and two entries, one of which
// waits on its grid dependencies (griddepcontrol.wait) and one of which does not.
__device__ unsigned int fixture_counter;

extern "C" __global__ void fixture_waits(unsigned int* out) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    out[threadIdx.x] = fixture_counter + threadIdx.x;
}

extern "C" __global__ void fixture_plain(unsigned int* out) {
    out[threadIdx.x] = threadIdx.x;
}
