// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-14: `moe_w4a16_grouped_gemm_ptrtable_wgmma`: the grouped W4A16 MoE prefill GEMM of
// `moe_w4a16_grouped_gemm_ptrtable_pipe` (moe_w4a16_grouped_pipe_hopper.cu) on Hopper's
// warpgroup MMA, with the same arguments and the same output layout C[row][n].
//
// - Operands swapped: the dequantised weights are the wgmma A operand (64 weight rows per
//   warpgroup, from registers) and the expert's token rows are the B operand (up to 128 rows,
//   K-major in shared memory, 128-byte swizzle). A CTA = 2 warpgroups = 128 output columns x up to
//   128 token rows; wgmma N is 128 for every unit (rows past the unit's are zero-filled and not
//   stored). With two or more N variants in one function ptxas serialises every wgmma (C7514), so
//   a short unit pays full-width MMAs; the per-unit dequantisation, which does not shrink with the
//   rows, costs about as much as those MMAs.
// - Persistent 1-D grid, one CTA per SM. A CTA scans the experts' 128-row tile counts into shared
//   memory and strides over the work units (expert, 128-row tile, 128-column N tile), N fastest.
//   K advances 64 per stage over WG_STAGES cp.async stages; the load cursor runs WG_AHEAD stages
//   ahead of the MMAs across unit boundaries, so a short K (the down projection) does not refill
//   the pipeline per unit.
// - A weight is BF16(E2M1 * E4M3 * scale2), product in FP32 left to right, rounded once to BF16,
//   by the same helpers (ntc_e2m1x4_bf16, ntc_e4m3_bf16x2, ntc_hmul2, the scale-pair rounding) the
//   pipe kernel uses: every staged weight is bit-identical to the pipe's by construction.
// - Each output is an FP32 accumulator starting at +0.0, advanced by one tensor-core k16 step at a
//   time in ascending k16 order, rounded once to BF16 (round to nearest even): the pipe kernel's
//   order. NOT identical by construction: whether a warpgroup k16 step sums its 16 products into
//   the accumulator exactly as an m16n8k16 mma.sync does. The products commute (W x X = X x W
//   exactly); the in-step reduction is the hardware's. The GPU test
//   (moe_w4a16_wgmma_tests.rs) compares the bytes with the base and pipe kernels; the launcher
//   runs this kernel only when METRALE_MOE_PREFILL_WGMMA is set.
//
// Requirements (the host admits only these shapes; the kernel traps on the rest): K % 64 == 0,
// N % 128 == 0, each expert's packed pointer 16-byte aligned and scale pointer 4-byte aligned, and
// at least WG_SMEM_BYTES + (E + 1) * 4 bytes of dynamic shared memory (the launcher's
// MOE_W4A16_WGMMA_LAUNCH, crates/model-layers/src/layers/ops/moe_grouped_a/wgmma.rs). The rows
// of an expert with a null packed pointer are not written (the pipe kernel's rule).
//
// Owner: hopper kernels.
// Invariants: the staged weights equal the pipe kernel's bit for bit; the K order of the
// accumulation is ascending k16 into one FP32 accumulator per output.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "tc_weight_formats.cuh"  // 2026-10-14: ntc_e2m1x4_bf16, ntc_e4m3_bf16x2, ntc_hmul2
#include "moe_w4a16_grouped_hopper.cuh"  // 2026-10-14: pipe_scale_pair, pipe_scan_tiles

#define WG_MT 128
#define WG_NT 128
#define WG_KS 64
#define WG_STAGES 8
// 2026-10-14: Loads run this many stages ahead of the MMAs; the two remaining slots belong to the
// stage being multiplied and the one before it, whose wgmma may still be in flight.
#define WG_AHEAD (WG_STAGES - 2)
#define WG_THREADS 256
#define WG_GROUP 16
#define WG_B_BYTES (WG_MT * WG_KS * 2)
#define WG_W_BYTES (WG_NT * WG_KS / 2)
#define WG_S_BYTES (WG_NT * WG_KS / WG_GROUP)
// 2026-10-14: The output staging tile, [128 token rows][128 + 8 BF16]: the 8-element pad spreads
// the 2-byte stores of a warp over distinct banks.
#define WG_CS_LD (WG_NT + 8)
#define WG_CS_BYTES (WG_MT * WG_CS_LD * 2)
// 2026-10-14: 1024 bytes of slack let the kernel align the swizzled B slots to 1024 bytes.
#define WG_SMEM_BYTES (1024 + WG_STAGES * (WG_B_BYTES + WG_W_BYTES + WG_S_BYTES) + WG_CS_BYTES)

__device__ __forceinline__ unsigned int wg_smem_addr(const void* p) {
    return (unsigned int)__cvta_generic_to_shared(p);
}

// 2026-10-14: 16 bytes global -> shared, zero-filled when !valid (no global read then).
__device__ __forceinline__ void wg_cp16(unsigned int dst, const void* src, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src),
                 "r"(valid ? 16 : 0));
}

__device__ __forceinline__ void wg_cp4(unsigned int dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(dst), "l"(src));
}

__device__ __forceinline__ void wg_cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N>
__device__ __forceinline__ void wg_cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

// 2026-10-14: The cp.async writes (generic proxy) become visible to the wgmma reads (async proxy).
__device__ __forceinline__ void wg_fence_proxy() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }

__device__ __forceinline__ void wg_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void wg_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

// 2026-10-14: Shared-memory matrix descriptor of a K-major B slot with the 128-byte swizzle: start
// address >> 4 (bits 0-13), leading byte offset 16 (unused by this layout), stride byte offset
// 1024 (one 8-row x 128-byte swizzle atom; bits 32-45), base offset 0 (slots are 1024-aligned),
// layout 1 = 128-byte swizzle (bits 62-63). Advancing K by 16 BF16 adds 32 bytes (2) to the start.
__device__ __forceinline__ unsigned long long wg_desc(unsigned int smem_addr) {
    return (unsigned long long)((smem_addr & 0x3FFFFu) >> 4) | (1ull << 16) | (64ull << 32) |
           (1ull << 62);
}

// 2026-10-14: One m64nNk16 warpgroup MMA, D += A x B: A = four BF16-pair registers (the
// m16n8k16 A fragment of this warp's 16 rows), B from the descriptor, D = N / 2 FP32 registers.
template <int N>
__device__ __forceinline__ void wg_mma(float* d, const unsigned int* a, unsigned long long desc);

template <>
__device__ __forceinline__ void wg_mma<128>(float* d, const unsigned int* a,
                                              unsigned long long desc) {
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %69, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
        "{"
        "%0, %1, %2, %3, %4, %5, %6, %7, "
        "%8, %9, %10, %11, %12, %13, %14, %15, "
        "%16, %17, %18, %19, %20, %21, %22, %23, "
        "%24, %25, %26, %27, %28, %29, %30, %31, "
        "%32, %33, %34, %35, %36, %37, %38, %39, "
        "%40, %41, %42, %43, %44, %45, %46, %47, "
        "%48, %49, %50, %51, %52, %53, %54, %55, "
        "%56, %57, %58, %59, %60, %61, %62, %63"
        "}, {%64, %65, %66, %67}, %68, p, 1, 1, 0;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
          "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]),
          "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]),
          "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]),
          "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]),
          "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]),
          "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]),
          "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]),
          "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));
}

// 2026-10-14: One persistent work unit: the expert's tables and scale2, its first output row and
// row count (<= WG_MT), and the first output column.
struct WgUnit {
    const unsigned char* w;
    const unsigned char* s;
    float s2;
    unsigned int row0;
    int rows;
    unsigned int n0;
};

// 2026-10-14: Unit i of this CTA (i-th of its grid-strided units). Traps on a misaligned table.
__device__ __forceinline__ WgUnit wg_unit(unsigned int i, const int* first_tile,
                                          unsigned int num_experts, unsigned int n_tiles,
                                          const unsigned long long* __restrict__ B_packed_ptrs,
                                          const unsigned long long* __restrict__ B_scale_ptrs,
                                          const float* __restrict__ scale2_vals,
                                          const int* __restrict__ expert_offsets) {
    const unsigned int w = blockIdx.x + i * gridDim.x;
    const unsigned int mt = w / n_tiles;
    const unsigned int e = pipe_tile_expert(first_tile, num_experts, mt);
    const int m_start = expert_offsets[e];
    const int m_local = (int)(mt - (unsigned int)first_tile[e]) * WG_MT;
    const int left = expert_offsets[e + 1] - m_start - m_local;
    WgUnit u;
    u.w = (const unsigned char*)B_packed_ptrs[e];
    u.s = (const unsigned char*)B_scale_ptrs[e];
    if (u.w != 0 && ((((unsigned long long)u.w) & 15ull) != 0 ||
                     (((unsigned long long)u.s) & 3ull) != 0)) {
        __trap();
    }
    u.s2 = scale2_vals[e];
    u.row0 = (unsigned int)(m_start + m_local);
    u.rows = left < WG_MT ? left : WG_MT;
    u.n0 = (w - mt * n_tiles) * WG_NT;
    return u;
}

// 2026-10-14: The four A fragments (one per k16 step) of this thread's two weight rows r0 and
// r0 + 8 for one 64-K stage: a[4 kk + 0 / 1] = rows r0 / r0 + 8 at k 16 kk + 2q, + 1; a[4 kk + 2 /
// 3] = the same rows at k 16 kk + 8 + 2q, + 1. A staged row is 32 packed bytes, its 16-byte half h
// stored at h ^ ((row >> 2) & 1), then its 4 E4M3 scale bytes (one per k16 group).
__device__ __forceinline__ void wg_dequant(unsigned int* a, const unsigned char* w_slot,
                                           const unsigned int* s_slot, unsigned int r0,
                                           unsigned int q, float s2) {
    const unsigned int sel = q | ((q + 4u) << 4);
#pragma unroll
    for (int rr = 0; rr < 2; rr++) {
        const unsigned int r = r0 + rr * 8, sw = (r >> 2) & 1u;
        const uint4 lo = *(const uint4*)(w_slot + r * 32 + sw * 16);
        const uint4 hi = *(const uint4*)(w_slot + r * 32 + (sw ^ 1u) * 16);
        const unsigned int words[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
        const unsigned int sc = s_slot[r];
#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            const unsigned int s = ntc_e4m3_bf16x2((sc >> (8 * kk)) & 0xFFu);
            unsigned int d[2];
            ntc_e2m1x4_bf16(__byte_perm(words[2 * kk], words[2 * kk + 1], sel), d);
            a[4 * kk + rr] = pipe_scale_pair(ntc_hmul2(d[0], s), s2);
            a[4 * kk + 2 + rr] = pipe_scale_pair(ntc_hmul2(d[1], s), s2);
        }
    }
}

// 2026-10-14: What every unit of a CTA shares: the B slot base (shared address), the weight and
// scale slots, the output staging tile, this thread's weight row r0 and lane quad q, its index t,
// and the K stages per unit.
struct WgShared {
    unsigned int b_s;
    const unsigned char* w_g;
    const unsigned int* s_g;
    __nv_bfloat16* Cs;
    unsigned int r0, q, t, num_k;
};

// 2026-10-14: One unit's K loop and its store: global stages c0 .. c0 + num_k, wgmma N = WG_MT.
// Stage c: wait for its copies, retire the MMAs of stage c - 2 (whose slot the next load reuses),
// issue the load WG_AHEAD ahead, dequantise into the register set stage c - 2 used, then the four
// k16 MMAs. The caller zeroes `acc` (WG_MT / 2 floats) before the unit.
template <typename Loader>
__device__ __forceinline__ void wg_run_unit(float* acc, Loader& load_next, const WgShared& sh,
                                            const WgUnit& u, unsigned int c0,
                                            __nv_bfloat16* __restrict__ C, unsigned int N) {
    unsigned int a0[16], a1[16];
    auto stage = [&](unsigned int c, unsigned int* a) {
        wg_cp_wait<WG_AHEAD - 1>();
        wg_fence_proxy();
        wg_wait<1>();
        __syncthreads();
        load_next();
        const unsigned int slot = c % WG_STAGES;
        if (u.w != 0) {
            wg_dequant(a, sh.w_g + slot * WG_W_BYTES, sh.s_g + slot * (WG_S_BYTES / 4), sh.r0,
                       sh.q, u.s2);
        } else {
            // 2026-10-14: An expert without tables: its unit's MMAs run on zero weights and its
            // rows are not stored.
#pragma unroll
            for (int i = 0; i < 16; i++) a[i] = 0u;
        }
        const unsigned long long desc = wg_desc(sh.b_s + slot * WG_B_BYTES);
        wg_fence();
#pragma unroll
        for (int kk = 0; kk < 4; kk++) wg_mma<WG_MT>(acc, a + 4 * kk, desc + 2ull * kk);
        wg_commit();
    };
#pragma unroll 1
    for (unsigned int kt = 0; kt < sh.num_k; kt += 2) {
        stage(c0 + kt, a0);
        if (kt + 1 < sh.num_k) stage(c0 + kt + 1, a1);
    }
    wg_wait<0>();
#pragma unroll
    for (int i = 0; i < WG_MT / 2; i++) asm volatile("" : "+f"(acc[i])::"memory");
    // 2026-10-14: Accumulator acc[4j + e] of this thread is weight row r0 (+ 8 for e >= 2), token
    // row 8j + 2q (+ 1 for odd e); the staging tile is [token][column].
    __nv_bfloat16* Cs = sh.Cs;
#pragma unroll
    for (int j = 0; j < WG_MT / 8; j++) {
        const unsigned int tr = 8 * j + 2 * sh.q;
        Cs[tr * WG_CS_LD + sh.r0] = __float2bfloat16_rn(acc[4 * j]);
        Cs[(tr + 1) * WG_CS_LD + sh.r0] = __float2bfloat16_rn(acc[4 * j + 1]);
        Cs[tr * WG_CS_LD + sh.r0 + 8] = __float2bfloat16_rn(acc[4 * j + 2]);
        Cs[(tr + 1) * WG_CS_LD + sh.r0 + 8] = __float2bfloat16_rn(acc[4 * j + 3]);
    }
    __syncthreads();
    if (u.w == 0) return;
#pragma unroll
    for (int i = 0; i < (WG_MT * WG_NT / 8) / WG_THREADS; i++) {
        const unsigned int ch = sh.t + i * WG_THREADS, tr = ch >> 4, col = (ch & 15) * 8;
        if ((int)tr >= u.rows) break;
        *(uint4*)(C + (unsigned long long)(u.row0 + tr) * N + u.n0 + col) =
            *(const uint4*)(Cs + tr * WG_CS_LD + col);
    }
}

// 2026-10-14: Persistent grid of any size (the launcher uses one CTA per SM); a CTA with no unit
// exits after the scan.
extern "C" __global__ void __launch_bounds__(WG_THREADS, 1)
moe_w4a16_grouped_gemm_ptrtable_wgmma(
    const __nv_bfloat16* __restrict__ A,                   // 2026-10-14: [num_tokens, K]
    const unsigned long long* __restrict__ B_packed_ptrs,  // 2026-10-14: [E] -> [N, K/2] E2M1
    const unsigned long long* __restrict__ B_scale_ptrs,   // 2026-10-14: [E] -> [N, K/16] E4M3
    const float* __restrict__ scale2_vals,                 // 2026-10-14: [E]
    __nv_bfloat16* __restrict__ C,                         // 2026-10-14: [total_expanded, N]
    const int* __restrict__ expert_offsets,                // 2026-10-14: [E + 1] prefix sum
    const int* __restrict__ sorted_token_ids,              // 2026-10-14: row of A, or NULL
    unsigned int num_experts, unsigned int N, unsigned int K) {
    extern __shared__ __align__(16) unsigned char wg_raw[];
    __shared__ int warp_tiles[WG_THREADS / 32];
    {
        unsigned int dyn;
        asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dyn));
        if (dyn < WG_SMEM_BYTES + (num_experts + 1) * 4 || (K % WG_KS) != 0 ||
            (N % WG_NT) != 0) {
            __trap();
        }
    }
    // 2026-10-14: Layout from the first 1024-aligned byte: B slots [S][128 rows][128 bytes],
    // weight slots [S][128 rows][32 bytes], scale slots [S][128 rows][4 bytes], the output
    // staging tile; the tile scan follows at WG_SMEM_BYTES from the start.
    const unsigned int raw_s = wg_smem_addr(wg_raw);
    unsigned char* base = wg_raw + ((((raw_s + 1023u) & ~1023u)) - raw_s);
    const unsigned int b_s = wg_smem_addr(base);
    const unsigned int w_s = b_s + WG_STAGES * WG_B_BYTES;
    const unsigned int s_s = w_s + WG_STAGES * WG_W_BYTES;
    const unsigned char* w_g = base + WG_STAGES * WG_B_BYTES;
    const unsigned int* s_g = (const unsigned int*)(w_g + WG_STAGES * WG_W_BYTES);
    __nv_bfloat16* Cs = (__nv_bfloat16*)(w_g + WG_STAGES * (WG_W_BYTES + WG_S_BYTES));
    int* first_tile = (int*)(wg_raw + WG_SMEM_BYTES);

    const int total =
        pipe_scan_tiles<WG_MT, WG_THREADS>(first_tile, warp_tiles, expert_offsets, num_experts);
    const unsigned int n_tiles = N / WG_NT, num_k = K / WG_KS;
    const unsigned int units = (unsigned int)total * n_tiles;
    const unsigned int my_units =
        blockIdx.x < units ? (units - 1u - blockIdx.x) / gridDim.x + 1u : 0u;
    const unsigned int steps = my_units * num_k;
    if (steps == 0) return;

    const unsigned int t = threadIdx.x, lane = t & 31;
    const unsigned int g = lane >> 2, q = lane & 3;
    // 2026-10-14: This thread's weight rows: warpgroup t / 128 owns rows 64 (t / 128) .. + 64, its
    // warp w (of 4) rows 16 w .. + 16 of those, as the wgmma A fragment places them.
    const unsigned int r0 = (t >> 7) * 64 + ((t >> 5) & 3) * 16 + g;
    const unsigned int half_K = K / 2, groups_K = K / WG_GROUP;

    // 2026-10-14: Load slots, fixed per thread: token rows (t / 8) + 32 i, 16-byte chunk t % 8
    // (swizzled by row % 8); packed row t / 2, half t % 2; scale row t (t < 128).
    const unsigned int b_ch = t & 7, b_row0 = t >> 3;
    const unsigned int w_row = t >> 1, w_h = t & 1;
    const unsigned int w_dst = w_row * 32 + ((w_h ^ ((w_row >> 2) & 1u)) << 4);
    const __nv_bfloat16* ld_tok[4];
    unsigned int ld_unit = 0, ld_kt = 0, ld_step = 0;
    int ld_rows = 0;
    const unsigned char* ld_w = 0;
    const unsigned char* ld_sc = 0;
    bool ld_wok = false;

    auto load_next = [&]() {
        if (ld_step < steps) {
            if (ld_kt == 0) {
                const WgUnit u = wg_unit(ld_unit, first_tile, num_experts, n_tiles, B_packed_ptrs,
                                         B_scale_ptrs, scale2_vals, expert_offsets);
                ld_rows = u.rows;
                ld_wok = u.w != 0;
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    const unsigned int row = b_row0 + 32 * i;
                    const unsigned int src =
                        (int)row < u.rows ? (sorted_token_ids ? (unsigned int)sorted_token_ids[u.row0 + row]
                                                              : u.row0 + row)
                                          : 0u;
                    ld_tok[i] = A + (unsigned long long)src * K + b_ch * 8;
                }
                ld_w = ld_wok ? u.w + (unsigned long long)(u.n0 + w_row) * half_K + w_h * 16 : 0;
                ld_sc = ld_wok ? u.s + (unsigned long long)(u.n0 + (t & 127)) * groups_K : 0;
            }
            const unsigned int slot = ld_step % WG_STAGES;
            const unsigned int k0 = ld_kt * WG_KS;
            const unsigned int bslot = b_s + slot * WG_B_BYTES;
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const unsigned int row = b_row0 + 32 * i;
                wg_cp16(bslot + row * 128 + ((b_ch ^ (row & 7u)) << 4), ld_tok[i] + k0,
                        (int)row < ld_rows);
            }
            if (ld_wok) {
                wg_cp16(w_s + slot * WG_W_BYTES + w_dst, ld_w + k0 / 2, true);
                if (t < WG_NT) wg_cp4(s_s + slot * WG_S_BYTES + t * 4, ld_sc + k0 / WG_GROUP);
            }
            if (++ld_kt == num_k) {
                ld_kt = 0;
                ld_unit++;
            }
        }
        ld_step++;
        wg_cp_commit();
    };

#pragma unroll 1
    for (int p = 0; p < WG_AHEAD; p++) load_next();

    // 2026-10-14: The units run in order; the loads stay WG_AHEAD stages ahead across them.
    const WgShared sh = {b_s, w_g, s_g, Cs, r0, q, t, num_k};
    unsigned int c = 0;
    float acc[WG_MT / 2];
#pragma unroll 1
    for (unsigned int i = 0; i < my_units; i++, c += num_k) {
        const WgUnit u = wg_unit(i, first_tile, num_experts, n_tiles, B_packed_ptrs, B_scale_ptrs,
                                 scale2_vals, expert_offsets);
#pragma unroll
        for (int j = 0; j < WG_MT / 2; j++) acc[j] = 0.0f;
        wg_run_unit(acc, load_next, sh, u, c, C, N);
    }
    wg_cp_wait<0>();
}

// 2026-10-14: Output columns per work unit, read by the host to check the N tile it admits.
extern "C" __device__ unsigned int moe_w4a16_grouped_gemm_ptrtable_wgmma_n_tile = WG_NT;
