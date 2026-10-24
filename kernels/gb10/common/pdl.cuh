// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-21: Programmatic dependent launch (PDL) inside a kernel. The host launches a kernel
// programmatically only when its PTX entry body holds `griddepcontrol.wait` (gpu-runtime
// ptx_pdl.rs) and the model switched the mode on (METRALE_GLM_PDL); such a kernel may then start
// while its predecessor on the stream is still running.
//
//   pdl_wait():    blocks until every prerequisite grid has completed and its memory is visible.
//                  Nothing the predecessor (or anything before it) writes may be read, and nothing
//                  may be written to global memory, before this call. Only data that is constant
//                  for the whole run (weights) may be loaded ahead of it.
//   pdl_trigger(): lets the dependent kernel launch once every block of this grid has called it
//                  or exited. It never relaxes visibility (the dependent's own pdl_wait still waits
//                  for this grid to complete), so it is safe anywhere; placed right after
//                  pdl_wait() it lets the next kernel's prologue overlap this kernel's body.
//
// Without the launch attribute both are no-ops, so a kernel's bytes never depend on whether PDL is
// on: PDL changes only timing. A read hoisted above pdl_wait() is a race, not a numerics change
// (pdl_chain_gate replays the chain against a serialised reference to catch one).
//
// Owner: gb10 kernels.
// Invariants: sm_90 and newer only; on older targets both compile to nothing.
#pragma once

__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

__device__ __forceinline__ void pdl_trigger() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

// 2026-10-21: The common prologue split: wait for the predecessor, then let the successor launch.
__device__ __forceinline__ void pdl_enter() {
    pdl_wait();
    pdl_trigger();
}
