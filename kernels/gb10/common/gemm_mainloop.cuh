// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-05: The tile mainloop family's device side (book/src/appendix/layouts.md, atom bundles in
// crates/circuit/src/venn/atoms.rs). A GEMM tile kernel is a point of it: an atom bundle (the MMA
// instruction, the operand copy instructions, the shared-memory row layout and its swizzle) plus a
// schedule (here: a multistage ring of asynchronous copies) plus the kernel's own body (what one
// stage loads, what one stage computes, how it folds scales). The bundle's pieces are the only
// inline assembly; a kernel names them instead of writing its own.
//
// Owner: gb10 kernels.
// Invariants:
// - Every atom is a single instruction with the operand order the PTX ISA states; nothing here
//   reorders arithmetic, so a kernel moved onto these atoms keeps its reduction tree and its
//   bytes (checked by comparing the PTX before and after a move).
// - SwizzledRows<ROW_BYTES, B, M, S> is the layout `row * ROW_BYTES + chunk * 2^M` followed by the
//   XOR swizzle sigma(B, M, S) of crates/layout; its static_asserts state when the closed form
//   used here equals that composition.
// - The multistage schedule keeps STAGES - 1 copy groups in flight: before computing step s it
//   waits until stage s has landed and every thread has finished with the stage the next load
//   overwrites.

#pragma once
#include <cuda_bf16.h>

namespace gml {

__device__ __forceinline__ unsigned smem_addr(const void* p) {
    return (unsigned)__cvta_generic_to_shared(p);
}

// 2026-10-05: Copy atoms, global to shared memory, asynchronous. `pred` false zero-fills the
// destination without reading `src`.
struct CopyAsync16 {
    static constexpr int BYTES = 16;
    __device__ __forceinline__ static void copy(unsigned dst, const void* src, bool pred) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(pred ? 16 : 0));
    }
};
struct CopyAsync4 {
    static constexpr int BYTES = 4;
    __device__ __forceinline__ static void copy(unsigned dst, const void* src, bool pred) {
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(dst), "l"(src), "r"(pred ? 4 : 0));
    }
};
struct CopyAsyncGroups {
    __device__ __forceinline__ static void commit() { asm volatile("cp.async.commit_group;\n" ::); }
    template <int N>
    __device__ __forceinline__ static void wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
};

// 2026-10-05: Copy atom, shared memory to registers: four 8x8 b16 matrices, one row address per lane.
struct LoadMatrixX4 {
    __device__ __forceinline__ static void load(unsigned addr, unsigned& d0, unsigned& d1, unsigned& d2, unsigned& d3) {
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                     : "r"(addr));
    }
};

// 2026-10-05: MMA atom: m16n8k32, E4M3 x E4M3, F32 accumulation in place, one warp.
struct MmaE4m3M16N8K32 {
    static constexpr int M = 16, N = 8, K = 32;
    __device__ __forceinline__ static void mma(float* acc, const unsigned* a, unsigned b0, unsigned b1) {
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
            : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
              "f"(acc[0]), "f"(acc[1]), "f"(acc[2]), "f"(acc[3]));
    }
};

constexpr int log2_exact(int v) { return v <= 1 ? 0 : 1 + log2_exact(v / 2); }

// 2026-10-05: Byte offset of 16-byte chunk `ch` of row `row` in a [rows][ROW_BYTES] tile under
// sigma(B, M, S): offset = row * ROW_BYTES + ch * 2^M, then bits [M + S, M + S + B) XORed into
// bits [M, M + B). With the source field at or above the row bits, that is the closed form below:
// the chunk XORed with B bits of the row.
template <int ROW_BYTES, int B, int M, int S>
struct SwizzledRows {
    static constexpr int ROW_BITS = log2_exact(ROW_BYTES);
    static constexpr int ROW_SHIFT = M + S - ROW_BITS;
    static constexpr unsigned MASK = (1u << B) - 1u;
    static_assert((1 << ROW_BITS) == ROW_BYTES, "rows are a power of two bytes");
    static_assert(M + B <= ROW_BITS, "the swizzled chunk field lies inside a row");
    static_assert(ROW_SHIFT >= 0, "the source field lies in the row index");
    __device__ __forceinline__ static unsigned offset(unsigned row, unsigned ch) {
        return row * ROW_BYTES + ((ch ^ ((row >> ROW_SHIFT) & MASK)) << M);
    }
};

// 2026-10-05: An atom bundle: the instructions a tile kernel is built from.
template <class MmaT, class CopyT, class SmallCopyT, class GroupsT, class LoadT, class RowsT>
struct Bundle {
    using Mma = MmaT;
    using Copy = CopyT;
    using SmallCopy = SmallCopyT;
    using Groups = GroupsT;
    using Load = LoadT;
    using Rows = RowsT;
};

// 2026-10-05: The multistage schedule: `load(step, stage)` issues one step's copies into ring slot
// `stage`, `compute(step, stage)` consumes it. Steps run in ascending order, one compute per step.
template <class B, int STAGES, class LoadFn, class ComputeFn>
__device__ __forceinline__ void multistage(unsigned n_steps, LoadFn&& load, ComputeFn&& compute) {
    #pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if ((unsigned)s < n_steps) load((unsigned)s, (unsigned)s);
        B::Groups::commit();
    }
    for (unsigned step = 0; step < n_steps; step++) {
        B::Groups::template wait<STAGES - 2>();
        __syncthreads();  // 2026-10-05: stage `step` visible; stage `step - 1` free for the next load
        {
            const unsigned nxt = step + STAGES - 1;
            if (nxt < n_steps) load(nxt, nxt % STAGES);
            B::Groups::commit();
        }
        compute(step, step % STAGES);
    }
    B::Groups::template wait<0>();
}

// 2026-10-05: The BF16 epilogue of an m16n8 accumulator tile: warp (wm, wn) of a WARPS_M x WARPS_N
// grid holds MI x NI fragments; tile row r goes to C row `row_base + r` for r < rows_valid.
template <int BM, int BN, int WARPS_M, int WARPS_N>
__device__ __forceinline__ void store_bf16_rows(
    __nv_bfloat16* __restrict__ C,
    unsigned int N,
    unsigned long long row_base,
    int rows_valid,
    unsigned int cta_n,
    const float (&outer)[BM / WARPS_M / 16][BN / WARPS_N / 8][4]
) {
    constexpr int WM = BM / WARPS_M, WN = BN / WARPS_N, MI = WM / 16, NI = WN / 8;
    const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const unsigned wm0 = (warp / WARPS_N) * WM, wn0 = (warp % WARPS_N) * WN;
    const unsigned gid = lane >> 2, tig = lane & 3u;
    #pragma unroll
    for (int mi = 0; mi < MI; mi++) {
        const int r0 = (int)(wm0 + mi * 16 + gid), r1 = r0 + 8;
        #pragma unroll
        for (int ni = 0; ni < NI; ni++) {
            const unsigned col = cta_n + wn0 + ni * 8 + tig * 2;
            if (r0 < rows_valid)
                *(__nv_bfloat162*)&C[(row_base + r0) * N + col] = __floats2bfloat162_rn(outer[mi][ni][0], outer[mi][ni][1]);
            if (r1 < rows_valid)
                *(__nv_bfloat162*)&C[(row_base + r1) * N + col] = __floats2bfloat162_rn(outer[mi][ni][2], outer[mi][ni][3]);
        }
    }
}

}
