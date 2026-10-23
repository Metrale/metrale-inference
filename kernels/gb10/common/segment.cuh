// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-23: The persistent-segment task loop (DESIGN-fuser-megakernel.md §4.2, P3 prototype):
// one launch runs the tasks of several dependent phases. Each CTA takes the next task by ticket
// (atomicAdd), so tasks start in table order; a task of phase p > 0 first waits (thread 0 spins,
// with backoff) until every task of phase p - 1 has completed, then the block runs it and
// publishes its completion (every thread fences its stores, the block syncs, thread 0 counts).
// Flags, not grid barriers: a task waits only on tasks that were already taken by running CTAs,
// so the loop cannot deadlock whatever the grid and whatever else shares the SMs.
//
// Owner: gb10 kernels.
// Invariants:
// - SegCtl is zero at launch: the last CTA to leave resets it (so a graph replays it as captured).
// - Phase p's tasks see every global store of phase p - 1 (fence before the count, acquire after
//   the wait); they must read such data through L2 (ld.global.cg), never the non-coherent path.
// - A task's body may use the block's dynamic shared memory freely: the loop syncs the block
//   before each task.
// - Body contract: `run(phase, task, ntasks)`, called by every thread of the block.
#pragma once

#define SEGMENT_MAX_PHASES 4

// 2026-10-23: The control words of one segment launch (device global, zero-initialised).
struct SegCtl {
    unsigned int ticket;
    unsigned int exited;
    unsigned int done[SEGMENT_MAX_PHASES];
};

__device__ __forceinline__ unsigned int segment_ld_acquire(const unsigned int* p) {
    unsigned int v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

template <class Run>
__device__ __forceinline__ void segment_run(SegCtl* ctl, const unsigned int (&n)[SEGMENT_MAX_PHASES],
                                            const Run& run) {
    __shared__ unsigned int s_task;
    unsigned int total = 0;
    #pragma unroll
    for (int p = 0; p < SEGMENT_MAX_PHASES; p++) total += n[p];
    for (;;) {
        __syncthreads();
        if (threadIdx.x == 0) s_task = atomicAdd(&ctl->ticket, 1u);
        __syncthreads();
        unsigned int local = s_task;
        if (local >= total) break;
        unsigned int phase = 0;
        #pragma unroll
        for (int p = 0; p < SEGMENT_MAX_PHASES - 1; p++) {
            if (phase == (unsigned int)p && local >= n[p]) {
                local -= n[p];
                phase = p + 1;
            }
        }
        if (phase > 0 && threadIdx.x == 0) {
            unsigned int ns = 32;
            while (segment_ld_acquire(&ctl->done[phase - 1]) < n[phase - 1]) {
                __nanosleep(ns);
                ns = ns < 64 ? 2 * ns : ns;
            }
        }
        __syncthreads();
        run(phase, local, n[phase]);
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) atomicAdd(&ctl->done[phase], 1u);
    }
    if (threadIdx.x == 0) {
        __threadfence();
        if (atomicAdd(&ctl->exited, 1u) == gridDim.x - 1) {
            ctl->ticket = 0;
            #pragma unroll
            for (int p = 0; p < SEGMENT_MAX_PHASES; p++) ctl->done[p] = 0;
            __threadfence();
            ctl->exited = 0;
        }
    }
}
