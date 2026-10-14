// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-14: Hopper NVFP4 MoE decode (W4A16): `moe_expert_gate_up_act_nvfp4_grouped_hsm` and
// `moe_expert_down_act_nvfp4_grouped_hsm` return the bytes of the split pair
// `moe_expert_gate_up_act_nvfp4_grouped_tc_split` / `moe_expert_down_act_nvfp4_grouped_tc_mt1`
// (gb10/common/moe_nvfp4_grouped_tc.cu), with the same arguments, grid rows and block-row contract
// (tc_block_rows), from weights staged whole in shared memory.
//
// Why: the split pair's warps read a 16-column tile's weights from global memory one load group
// (2 chunks, 2 KB per warp) at a time with one group in flight, so at 1..16 rows each warp waits
// on K / 256 dependent DRAM round trips and an H100 reaches ~0.9 TB/s. Here every warp issues its
// whole tile (16 rows x K / 2 packed bytes plus 16 x K / 16 scale bytes) as cp.async 16-byte copies
// up front, one mbarrier per load group, then runs the unchanged warp body on shared memory.
//
// Owner: hopper kernels.
// Invariants:
// - Class B: each warp runs gtc_warp / gtc_warp_routed (gb10/common/moe_grouped_tc.cuh) with the
//   split pair's template arguments (gate+up PART 1 / PART 2, MT 1, G 2; down MT 1, G 1) through
//   the Nvfp4G16Smem policy below, whose load / scale return the 16 bytes and the 2 scale bytes
//   Nvfp4G16 reads from global memory for the same (row, chunk, lane), and whose frag and out are
//   Nvfp4G16's. Every MMA, its operands and its order, and the epilogue, are therefore the split
//   pair's: the bytes are equal for every row count, and a row's bytes do not depend on the other
//   rows of its launch (moe_grouped_tc.cuh).
// - Grids: gate+up (N / HSM_GU_COLS, cap + S) with HSM_GU_THREADS (one gate warp, one up warp),
//   down (N / HSM_DN_COLS, cap + S) with HSM_DN_THREADS, S = ceil(num_tokens / TC_ROWS). Dynamic
//   shared memory at least hsm_smem_bytes(warps, nst, xs) (the host's
//   `nvfp4_grouped_hsm_smem_bytes`, crates/model-layers/src/layers/ops/nvfp4_moe_grouped.rs);
//   the kernels trap on less, on K % (128 * G) != 0 and on a packed pointer that is not 16-byte
//   aligned. Scale pointers may have any alignment (16-, 4- or 1-byte copies).
// - A null routed weight pointer zeroes that expert's act or output rows, as the split pair does.

#include <cuda_bf16.h>

#include "moe_grouped_tc.cuh"
#include "tc_weight_formats.cuh"

#define HSM_GU_THREADS 64
#define HSM_GU_COLS 16
#define HSM_GU_G 2
#define HSM_DN_WARPS 4
#define HSM_DN_THREADS (HSM_DN_WARPS * 32)
#define HSM_DN_COLS (HSM_DN_WARPS * 16)
#define HSM_DN_G 1

__device__ __forceinline__ unsigned int hsm_addr(const void* p) {
    return (unsigned int)__cvta_generic_to_shared(p);
}

__device__ __forceinline__ void hsm_wait(unsigned int bar) {
    unsigned int done;
    do {
        asm volatile(
            "{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0;\nselp.u32 %0, 1, 0, p;\n}\n"
            : "=r"(done) : "r"(bar) : "memory");
    } while (!done);
}

// 2026-10-14: Bytes of dynamic shared memory: `warps` warps of `nst` mbarriers (8 bytes each,
// rounded up to 16), `xs` exchange floats, then 9 * K bytes per warp (packed tile, scale tile).
__host__ __device__ constexpr unsigned int hsm_head_bytes(unsigned int warps, unsigned int nst, unsigned int xs) {
    return (warps * nst * 8u + 15u) / 16u * 16u + xs * 4u;
}

// 2026-10-14: The NVFP4 policy on one staged 16 x K tile: packed bytes of chunk c, half h (rows
// g + 8h), lane (4g + t) at uint4 (c * 2 + h) * 32 + lane (Nvfp4G16's 16 bytes at
// (col + g + 8h) * K / 2 + 16t + 64c); scale bytes row-major [16, K / 16] as in global memory.
// Load group st = chunk / G is complete once mbarrier st has flipped; the scales are in group 0.
// `col` is not used: a Mat is one warp's staged tile (MT 1).
template <int G> struct Nvfp4G16Smem {
    static constexpr int CHUNK_K = Nvfp4G16::CHUNK_K;
    static constexpr bool FOLDS = false;
    static constexpr float ACT_LIFT = 1.0f;
    struct Mat { const uint4* w; const unsigned char* s; unsigned int bar; float s2; };
    struct Tile { const uint4* w; const unsigned char* sr[2]; unsigned int bar; float s2; };
    using Sc = Nvfp4G16::Sc;
    static __device__ __forceinline__ Tile tile(const Mat& M, unsigned int, unsigned int g, unsigned int t, unsigned int K) {
        Tile T;
        T.w = M.w + 4 * g + t;
        #pragma unroll
        for (int h = 0; h < 2; h++) T.sr[h] = M.s + (g + 8 * h) * (K / 16) + 2 * t;
        T.bar = M.bar;
        T.s2 = M.s2;
        return T;
    }
    static __device__ __forceinline__ uint4 load(const Tile& T, int h, unsigned int chunk) {
        hsm_wait(T.bar + 8u * (chunk / G));
        return T.w[(chunk * 2 + h) * 32];
    }
    static __device__ __forceinline__ Sc scale(const Tile& T, int h, unsigned int chunk) {
        hsm_wait(T.bar);
        return *(const unsigned short*)(T.sr[h] + chunk * 8);
    }
    static __device__ __forceinline__ void frag(const uint4& lo, const uint4& hi, Sc sg, Sc sh, int j, unsigned int* a) {
        Nvfp4G16::frag(lo, hi, sg, sh, j, a);
    }
    static __device__ __forceinline__ bool fold_at(unsigned int) { return false; }
    static __device__ __forceinline__ float fold_scale(const Tile&, unsigned int) { return 1.0f; }
    static __device__ __forceinline__ float out(const Tile& T, float x) { return x * T.s2; }
};

// 2026-10-14: One warp stages the 16-row tile at `col` of the row-major NVFP4 matrix (w, s) into
// `sw` (8 * K bytes) and `ss` (K bytes) and returns its Mat. Lane 0 initialises the K / 128 / G
// mbarriers at `bar` (32 arrivals each); every lane issues the scale copies, then per load group
// its packed copies and a cp.async arrive on that group's barrier, so a flipped barrier st also
// covers every earlier copy of every lane.
template <int G>
__device__ __forceinline__ typename Nvfp4G16Smem<G>::Mat hsm_stage(
    const unsigned char* __restrict__ w, const unsigned char* __restrict__ s, float s2, unsigned int col,
    unsigned int K, uint4* sw, unsigned char* ss, unsigned long long* bars
) {
    const unsigned int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const unsigned int nst = K / 128 / G, bar = hsm_addr(bars);
    if (lane == 0)
        for (unsigned int st = 0; st < nst; st++)
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 32;\n" :: "r"(bar + 8u * st) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    __syncwarp();
    const unsigned char* sg = s + (unsigned long long)col * (K / 16);
    const unsigned long long sa = (unsigned long long)sg;
    if ((sa & 15ull) == 0) {
        for (unsigned int i = lane; i < K / 16; i += 32)
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(hsm_addr(ss + 16 * i)), "l"(sg + 16 * i) : "memory");
    } else if ((sa & 3ull) == 0) {
        for (unsigned int i = lane; i < K / 4; i += 32)
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(hsm_addr(ss + 4 * i)), "l"(sg + 4 * i) : "memory");
    } else {
        for (unsigned int i = lane; i < K; i += 32) ss[i] = sg[i];
    }
    for (unsigned int st = 0; st < nst; st++) {
        #pragma unroll
        for (int c = 0; c < G; c++) {
            const unsigned int chunk = st * G + c;
            #pragma unroll
            for (int h = 0; h < 2; h++) {
                const unsigned char* src = w + (unsigned long long)(col + g + 8 * h) * (K / 2) + t * 16 + chunk * 64;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                             :: "r"(hsm_addr(sw + (chunk * 2 + h) * 32 + lane)), "l"(src) : "memory");
            }
        }
        asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" :: "r"(bar + 8u * st) : "memory");
    }
    __syncwarp();
    return {sw, ss, bar, s2};
}

// 2026-10-14: The layout check every entry runs first (uniform across the block).
__device__ __forceinline__ void hsm_check(unsigned int need, unsigned int K, unsigned int G) {
    unsigned int dyn;
    asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dyn));
    if (dyn < need || K % (128u * G) != 0u) __trap();
}

// 2026-10-14: tc_block_rows with the active count and the active expert read together (the
// expert is used only when i < active_count[0]; i < cap, the active list's length).
__device__ __forceinline__ bool hsm_block_rows(
    const int* __restrict__ expert_offsets, const int* __restrict__ active_experts,
    const int* __restrict__ active_count, unsigned int num_tokens,
    bool* is_shared, unsigned int* expert, unsigned int* begin, unsigned int* end
) {
    const unsigned int y = blockIdx.y;
    const unsigned int shared_slots = (num_tokens + TC_ROWS - 1) / TC_ROWS;
    *is_shared = (y < shared_slots);
    if (*is_shared) {
        *expert = 0; *begin = y * TC_ROWS; *end = min(num_tokens, (y + 1) * TC_ROWS);
    } else {
        const unsigned int i = y - shared_slots;
        const int count = active_count[0];
        const int e = active_experts[i];
        if ((int)i >= count) return false;
        *expert = (unsigned int)e;
        *begin = (unsigned int)expert_offsets[e];
        *end = (unsigned int)expert_offsets[e + 1];
    }
    return *begin < *end;
}

extern "C" __global__ void __launch_bounds__(HSM_GU_THREADS) moe_expert_gate_up_act_nvfp4_grouped_hsm(
    const __nv_bfloat16* __restrict__ A, const unsigned long long* __restrict__ gate_packed_ptrs,
    const unsigned long long* __restrict__ gate_scale_ptrs, const float* __restrict__ gate_scale2,
    const unsigned long long* __restrict__ up_packed_ptrs, const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2, __nv_bfloat16* __restrict__ act, const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids, const int* __restrict__ active_experts, const int* __restrict__ active_count,
    const unsigned char* __restrict__ sh_gate_packed, const unsigned char* __restrict__ sh_gate_scale, float sh_gate_s2,
    const unsigned char* __restrict__ sh_up_packed, const unsigned char* __restrict__ sh_up_scale, float sh_up_s2,
    __nv_bfloat16* __restrict__ sh_act, unsigned int N, unsigned int K, unsigned int cap, unsigned int num_tokens
) {
    using P = Nvfp4G16Smem<HSM_GU_G>;
    extern __shared__ __align__(16) unsigned char hsm_smem[];
    const unsigned int nst = K / 128 / HSM_GU_G, warp = threadIdx.x >> 5;
    const unsigned int head = hsm_head_bytes(2, nst, gtc_split_xs_floats(1));
    hsm_check(head + 2 * 9 * K, K, HSM_GU_G);
    bool is_shared;
    unsigned int expert, begin, end;
    if (!hsm_block_rows(expert_offsets, active_experts, active_count, num_tokens, &is_shared, &expert, &begin, &end)) return;
    const unsigned int f0 = blockIdx.x * HSM_GU_COLS;
    float* const xs = (float*)(hsm_smem + head - gtc_split_xs_floats(1) * 4);
    unsigned long long* const bars = (unsigned long long*)hsm_smem + warp * nst;
    uint4* const sw = (uint4*)(hsm_smem + head + warp * 9 * K);
    unsigned char* const ss = hsm_smem + head + warp * 9 * K + 8 * K;
    const bool by_pos = is_shared;
    const unsigned char *pw, *psc;
    float s2;
    if (is_shared) {
        pw = warp == 0 ? sh_gate_packed : sh_up_packed;
        psc = warp == 0 ? sh_gate_scale : sh_up_scale;
        s2 = warp == 0 ? sh_gate_s2 : sh_up_s2;
    } else {
        const unsigned char* Pg = (const unsigned char*)gate_packed_ptrs[expert];
        const unsigned char* Pu = (const unsigned char*)up_packed_ptrs[expert];
        const unsigned char* Sg = (const unsigned char*)gate_scale_ptrs[expert];
        const unsigned char* Su = (const unsigned char*)up_scale_ptrs[expert];
        const float g2 = gate_scale2[expert], u2 = up_scale2[expert];
        if (Pg == 0 || Pu == 0) {
            for (unsigned int pos = begin; pos < end; pos++)
                for (unsigned int i = threadIdx.x; i < HSM_GU_COLS; i += HSM_GU_THREADS)
                    for (unsigned int hl = 0; hl < 2; hl++)
                        act[(unsigned long long)pos * 2 * N + hl * N + f0 + i] = __float2bfloat16(0.0f);
            return;
        }
        pw = warp == 0 ? Pg : Pu;
        psc = warp == 0 ? Sg : Su;
        s2 = warp == 0 ? g2 : u2;
    }
    if ((((unsigned long long)pw) & 15ull) != 0) __trap();
    const P::Mat M = hsm_stage<HSM_GU_G>(pw, psc, s2, f0, K, sw, ss, bars);
    __nv_bfloat16* const out = is_shared ? sh_act : act;
    if (is_shared) {
        if (warp == 0) gtc_warp<P, true, 1, 1, HSM_GU_G, 1, HSM_GU_THREADS>(A, sorted_token_ids, by_pos, begin, end, M, M, out, N, K, f0, xs);
        else gtc_warp<P, true, 1, 1, HSM_GU_G, 2, HSM_GU_THREADS>(A, sorted_token_ids, by_pos, begin, end, M, M, out, N, K, f0, xs);
    } else {
        if (warp == 0) gtc_warp_routed<P, true, 1, HSM_GU_G, 1, HSM_GU_THREADS>(A, sorted_token_ids, by_pos, begin, end, M, M, out, N, K, f0, xs);
        else gtc_warp_routed<P, true, 1, HSM_GU_G, 2, HSM_GU_THREADS>(A, sorted_token_ids, by_pos, begin, end, M, M, out, N, K, f0, xs);
    }
}

extern "C" __global__ void __launch_bounds__(HSM_DN_THREADS) moe_expert_down_act_nvfp4_grouped_hsm(
    const __nv_bfloat16* __restrict__ act, const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs, const float* __restrict__ scale2, __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets, const int* __restrict__ active_experts, const int* __restrict__ active_count,
    const __nv_bfloat16* __restrict__ sh_act, const unsigned char* __restrict__ sh_down_packed,
    const unsigned char* __restrict__ sh_down_scale, float sh_down_s2, __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N, unsigned int K, unsigned int cap, unsigned int num_tokens
) {
    using P = Nvfp4G16Smem<HSM_DN_G>;
    extern __shared__ __align__(16) unsigned char hsm_smem[];
    const unsigned int nst = K / 128 / HSM_DN_G, warp = threadIdx.x >> 5;
    const unsigned int head = hsm_head_bytes(HSM_DN_WARPS, nst, 0);
    hsm_check(head + HSM_DN_WARPS * 9 * K, K, HSM_DN_G);
    bool is_shared;
    unsigned int expert, begin, end;
    if (!hsm_block_rows(expert_offsets, active_experts, active_count, num_tokens, &is_shared, &expert, &begin, &end)) return;
    const unsigned int f0 = blockIdx.x * HSM_DN_COLS + warp * 16;
    unsigned long long* const bars = (unsigned long long*)hsm_smem + warp * nst;
    uint4* const sw = (uint4*)(hsm_smem + head + warp * 9 * K);
    unsigned char* const ss = hsm_smem + head + warp * 9 * K + 8 * K;
    const unsigned char *pw, *psc;
    float s2;
    if (is_shared) {
        pw = sh_down_packed; psc = sh_down_scale; s2 = sh_down_s2;
    } else {
        pw = (const unsigned char*)packed_ptrs[expert];
        psc = (const unsigned char*)scale_ptrs[expert];
        s2 = scale2[expert];
        if (pw == 0) {
            for (unsigned int pos = begin; pos < end; pos++)
                for (unsigned int i = threadIdx.x; i < HSM_DN_COLS; i += HSM_DN_THREADS)
                    C[(unsigned long long)pos * N + blockIdx.x * HSM_DN_COLS + i] = __float2bfloat16(0.0f);
            return;
        }
    }
    if ((((unsigned long long)pw) & 15ull) != 0) __trap();
    const P::Mat M = hsm_stage<HSM_DN_G>(pw, psc, s2, f0, K, sw, ss, bars);
    if (is_shared)
        gtc_warp<P, false, 1, 1, HSM_DN_G>(sh_act, nullptr, true, begin, end, M, M, sh_down_out, N, K, f0);
    else
        gtc_warp_routed<P, false, 1, HSM_DN_G>(act, nullptr, true, begin, end, M, M, C, N, K, f0);
}
