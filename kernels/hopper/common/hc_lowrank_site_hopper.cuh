// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-14: `hc_lowrank_site`: one launch for the whole decode-width collapse of a Qwen3.8-
// Flash-Next low-rank hyper-connection site, in place of `hc_pre_stage_bf16` (GB10 model file
// hyper_connection.cu) followed by `hc_lowrank_down_inject` and `hc_lowrank_up_mix`. Included by
// hc_lowrank_gemv_hopper.cu after its helpers (hlg_dot8, hlg_butterfly_step) and qhc_act.cuh,
// so it is compiled with the same flags as those kernels.
//
// Outputs, bit for bit those of the three-launch route at every T (the byte gate is
// crates/model-layers/src/layers/ops/hyper_connection_site_tests.rs):
//   normed [T, hc*H] BF16, low [T, rank] BF16, inj [T, hc] FP32 (with inject_w), y [T, H] BF16.
//
// Blocks take roles in the order they start (a ticket from a global counter), so a block only
// ever waits for blocks that are already running or done, whatever the residency:
//   tickets [0, NS)          stage: one (token, stream) each, NS = T * hc
//   tickets [NS, NS + ND)    down:  one (row group g, virtual warp w) each, ND = 8 * NG,
//                            NG = ceil(rows / 16), rows = rank + (hc with inject_w, else 0)
//   tickets [NS + ND, end)   up:    16 hidden indices each, H / 16 blocks
// Down and up blocks start their weight tiles' copies to shared memory (cp.async) before they
// wait, so the weight stream overlaps the stage and the down phase. Waiting is one thread polling
// a counter with ld.acquire.gpu, then a block barrier; publishing is a block barrier, then one
// thread's __threadfence and atomicAdd (the cooperative-groups grid barrier's pattern). Data
// written inside the launch is read with ld.global.cg (L2), never the non-coherent path.
//
// Reduction trees (each the one of the kernel it replaces, so the bits match):
// - stage: `hc_pre_stage_bf16` runs 1024 threads per block. Here thread i of 256 plays virtual
//   threads i + 256 m (m < 4): virtual thread v sums x[d]^2 for d = v, v + 1024, ... in order,
//   its warp (virtual warp v / 32 = i / 32 + 8 m, the same 32 lanes) reduces with the same
//   shfl_down ladder, and thread 0 adds the 32 virtual-warp partials in ascending order.
// - down: `hc_lowrank_down_inject` reduces (row, token) over 256 threads: thread v sums chunks
//   v + 256 j (j ascending), each warp reduces by the butterfly (lane pairs across bit 4, then
//   3, 2, 1, 0), and the 8 warp sums are added in ascending warp order from 0. Here a down block
//   of virtual warp w computes warp w's sum for 16 rows: lane l sums the same chunks 32 w + l +
//   256 j in the same order, the same butterfly reduces them, and the partial goes to `partial`
//   [8, rows, T]. The last of a group's 8 blocks to finish adds the 8 partials in warp order from
//   0 and applies the same epilogue. A float sum of two values is commutative, so the butterfly's
//   result does not depend on which lane holds which value.
// - up: `hc_lowrank_up_mix` per warp: lane l sums chunks l, l + 32, ..., the same butterfly
//   reduces each (stream, token) sum, and the streams are added in ascending order. Its passes
//   hold 4 tokens instead of 8; which sums share a pass does not change any sum's tree.
//
// Requirements (the launcher admits only these; the kernel traps on a wrong grid or shared-memory
// size): blockDim.x == 256; gridDim.x == NS + ND + H / 16; hc * H a multiple of 2048 and at most
// 8 * 2048; H a multiple of 16; rank a multiple of 8, at most 512; hc <= 4; T <= 16; rows <= 16 *
// HCS_MAX_GROUPS; every pointer 16-byte aligned; `partial` holds 8 * rows * T floats; dynamic
// shared memory hcs_smem_bytes(T, hc, H, rank).
//
// Two launches of this kernel must not overlap in time (they share the counters, as callers of
// one hc_lowrank_scratch already share its rows). The last block to finish zeroes the counters,
// and module globals start at zero, so every launch starts from zero.
//
// Owner: hopper kernels.
// Invariants: outputs equal the three-launch route's bytes and each token's are independent of T
// (`ops/hyper_connection_site_tests.rs`).

#pragma once

#define HCS_THREADS 256
#define HCS_WARPS (HCS_THREADS / 32)
// 2026-10-14: The down tree's virtual warps (`HLG_THREADS / 32` of the kernel it replaces).
#define HCS_VWARPS 8
#define HCS_GROUP_ROWS 16              // 2026-10-14: 2 rows per warp
#define HCS_DOWN_TOKENS 8              // 2026-10-14: tokens per down pass (x staged per pass)
#define HCS_UP_COLS 16                 // 2026-10-14: 2 hidden indices per warp
// 2026-10-14: Tokens per up pass: 4 streams x 4 tokens = 16 sums, half the replaced kernel's 32,
// so the accumulators fit the 3-blocks-per-SM register budget. Each sum's tree is unchanged.
#define HCS_UP_TOKENS 4
#define HCS_MAX_NJ 8                   // 2026-10-14: 256-chunk segments of K, K <= 16384
#define HCS_STAGE_THREADS 1024         // 2026-10-14: hc_pre_stage_bf16's block width
#define HCS_MAX_GROUPS 64
#define HCS_MAX_T 16                   // 2026-10-14: the decode GEMVs' admitted widths
#define HCS_TICKET 0
#define HCS_STAGED 1
#define HCS_DOWNED 2
#define HCS_EXITED 3
#define HCS_GROUP0 4

// 2026-10-14: Launch counters; zero at module load and again at the end of every launch.
__device__ unsigned int hcs_sync[HCS_GROUP0 + HCS_MAX_GROUPS];

__device__ __forceinline__ unsigned int hcs_ld_acquire(const unsigned int* p) {
    unsigned int v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

// 2026-10-14: Block-wide wait until *ctr >= target.
__device__ __forceinline__ void hcs_wait(const unsigned int* ctr, unsigned int target) {
    if (threadIdx.x == 0) {
        while (hcs_ld_acquire(ctr) < target) __nanosleep(32);
    }
    __syncthreads();
}

// 2026-10-14: Block-wide publish of everything the block wrote: barrier, fence, increment.
// Returns the counter's previous value on thread 0.
__device__ __forceinline__ unsigned int hcs_publish(unsigned int* ctr) {
    __syncthreads();
    unsigned int old = 0u;
    if (threadIdx.x == 0) {
        __threadfence();
        old = atomicAdd(ctr, 1u);
    }
    return old;
}

// 2026-10-14: 16-byte global -> shared copy; src_bytes 0 writes zeros (and reads nothing).
__device__ __forceinline__ void hcs_cp16(uint4* dst, const void* src, unsigned int src_bytes) {
    const unsigned int d = (unsigned int)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(d), "l"(src), "r"(src_bytes)
                 : "memory");
}

__device__ __forceinline__ void hcs_cp_commit_wait_all() {
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::: "memory");
}

__device__ __forceinline__ void hcs_cp_commit() {
    asm volatile("cp.async.commit_group;" ::: "memory");
}

// 2026-10-14: One (token, stream) of hc_pre_stage_bf16, in its 1024-thread order.
__device__ __forceinline__ void hcs_stage(
    const float* __restrict__ streams, const __nv_bfloat16* __restrict__ norm_w,
    __nv_bfloat16* normed, float* red, unsigned int q, unsigned int H, unsigned int hc, float eps
) {
    const unsigned int t = q / hc;
    const unsigned int s = q % hc;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5;
    const float* xs = streams + (size_t)t * hc * H + (size_t)s * H;
    #pragma unroll
    for (unsigned int m = 0; m < HCS_STAGE_THREADS / HCS_THREADS; ++m) {
        float acc = 0.0f;
        for (unsigned int d = tid + m * HCS_THREADS; d < H; d += HCS_STAGE_THREADS) {
            float v = __ldg(xs + d);
            acc += v * v;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
        }
        if (lane == 0) red[m * HCS_WARPS + warp] = acc;
    }
    __syncthreads();
    if (tid == 0) {
        float tot = 0.0f;
        for (unsigned int w2 = 0; w2 < HCS_STAGE_THREADS / 32; ++w2) tot += red[w2];
        red[HCS_STAGE_THREADS / 32] = rsqrtf(tot / (float)H + eps);
    }
    __syncthreads();
    const float rms = red[HCS_STAGE_THREADS / 32];
    const __nv_bfloat16* ws = norm_w + (size_t)s * H;
    __nv_bfloat16* os = normed + (size_t)t * hc * H + (size_t)s * H;
    for (unsigned int d = tid; d < H; d += HCS_THREADS) {
        os[d] = __float2bfloat16(xs[d] * rms * (1.0f + (float)ws[d]));
    }
}

// 2026-10-14: Butterfly reduce of 16 values: lanes pair across bit 4 on all 16, then the
// reduce-scatter steps across bits 3..0. Lane L returns the warp sum of v[L % 16], over the same
// pairing as hlg_reduce_scatter32.
__device__ __forceinline__ float hcs_reduce_scatter16(float (&v)[32]) {
    const unsigned int lane = threadIdx.x & 31u;
    #pragma unroll
    for (int j = 0; j < 16; ++j) v[j] += __shfl_xor_sync(0xFFFFFFFFu, v[j], 16);
    hlg_butterfly_step<8>(v, lane);
    hlg_butterfly_step<4>(v, lane);
    hlg_butterfly_step<2>(v, lane);
    hlg_butterfly_step<1>(v, lane);
    return v[0];
}

// 2026-10-14: Row r of [down_w ; inject_w], or null past the last row.
__device__ __forceinline__ const __nv_bfloat16* hcs_down_row(
    const __nv_bfloat16* __restrict__ down_w, const __nv_bfloat16* __restrict__ inject_w,
    unsigned int r, unsigned int rank, unsigned int rows, unsigned int K
) {
    return r < rank ? down_w + (size_t)r * K
         : r < rows ? inject_w + (size_t)(r - rank) * K
         : nullptr;
}

// 2026-10-14: Virtual warp w of row group g: weight tile [16][nj][32] then x [8][nj][32] chunks.
// Partials of this warp go to partial[(w * rows + row) * T + t]; the group's last block reduces.
__device__ __forceinline__ void hcs_down(
    const __nv_bfloat16* __restrict__ down_w, const __nv_bfloat16* __restrict__ inject_w,
    const __nv_bfloat16* normed, float* partial, __nv_bfloat16* low, float* inj_out,
    uint4* tile, uint4* xs, unsigned int q, unsigned int T, unsigned int K, unsigned int rank,
    unsigned int rows, float inv_hc, unsigned int n_stage, unsigned int* last_flag
) {
    const unsigned int g = q / HCS_VWARPS;
    const unsigned int w = q % HCS_VWARPS;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5;
    const unsigned int k8n = K / 8u;
    const unsigned int nj = k8n / 256u;
    const unsigned int seg = nj * 32u;

    for (unsigned int i = tid; i < HCS_GROUP_ROWS * seg; i += HCS_THREADS) {
        const unsigned int rl = i / seg;
        const unsigned int j = (i % seg) / 32u;
        const unsigned int c = 32u * w + (i % 32u) + 256u * j;
        const __nv_bfloat16* row = hcs_down_row(down_w, inject_w, g * HCS_GROUP_ROWS + rl, rank, rows, K);
        hcs_cp16(tile + i, row != nullptr ? row + (size_t)c * 8u : down_w, row != nullptr ? 16u : 0u);
    }
    hcs_cp_commit();
    hcs_wait(hcs_sync + HCS_STAGED, n_stage);

    const uint4* x4 = reinterpret_cast<const uint4*>(normed);
    for (unsigned int t0 = 0; t0 < T; t0 += HCS_DOWN_TOKENS) {
        const unsigned int nt = min((unsigned int)HCS_DOWN_TOKENS, T - t0);
        if (t0 > 0) __syncthreads();
        for (unsigned int i = tid; i < nt * seg; i += HCS_THREADS) {
            const unsigned int t = i / seg;
            const unsigned int j = (i % seg) / 32u;
            xs[i] = __ldcg(x4 + (size_t)(t0 + t) * k8n + 32u * w + (i % 32u) + 256u * j);
        }
        hcs_cp_commit_wait_all();
        __syncthreads();

        float acc[32];
        #pragma unroll
        for (int i = 0; i < 32; ++i) acc[i] = 0.0f;
        const uint4* w0 = tile + (2u * warp) * seg + lane;
        const uint4* w1 = w0 + seg;
        #pragma unroll
        for (unsigned int j = 0; j < HCS_MAX_NJ; ++j) {
            if (j < nj) {
                const uint4 wa = w0[j * 32u];
                const uint4 wb = w1[j * 32u];
                #pragma unroll
                for (int t = 0; t < HCS_DOWN_TOKENS; ++t) {
                    if ((unsigned int)t < nt) {
                        const uint4 x = xs[((unsigned int)t * nj + j) * 32u + lane];
                        hlg_dot8(acc[t], wa, x);
                        hlg_dot8(acc[HCS_DOWN_TOKENS + t], wb, x);
                    }
                }
            }
        }
        const float sum = hcs_reduce_scatter16(acc);
        const unsigned int row = g * HCS_GROUP_ROWS + 2u * warp + (lane & 15u) / HCS_DOWN_TOKENS;
        const unsigned int t = lane % HCS_DOWN_TOKENS;
        if (lane < 16u && row < rows && t < nt) {
            partial[((size_t)w * rows + row) * T + t0 + t] = sum;
        }
    }

    const unsigned int old = hcs_publish(hcs_sync + HCS_GROUP0 + g);
    if (tid == 0) {
        if (old == HCS_VWARPS - 1u) __threadfence();
        *last_flag = old == HCS_VWARPS - 1u ? 1u : 0u;
    }
    __syncthreads();
    if (*last_flag == 0u) return;

    // 2026-10-14: Last block of the group: the 8 warp sums in warp order, then the epilogue.
    const unsigned int row = g * HCS_GROUP_ROWS + tid / 16u;
    const unsigned int t = tid % 16u;
    if (row < rows && t < T) {
        float sum = 0.0f;
        #pragma unroll
        for (int w2 = 0; w2 < HCS_VWARPS; ++w2) sum += __ldcg(partial + ((size_t)w2 * rows + row) * T + t);
        const float p = __bfloat162float(__float2bfloat16(sum)) * inv_hc;
        if (row < rank) {
            low[(size_t)t * rank + row] = __float2bfloat16(qhc_silu(p));
        } else {
            inj_out[(size_t)t * (rows - rank) + (row - rank)] = 2.0f * qhc_sigmoid(p);
        }
    }
    hcs_publish(hcs_sync + HCS_DOWNED);
    if (tid == 0) atomicExch(hcs_sync + HCS_GROUP0 + g, 0u);
}

// 2026-10-14: Hidden indices [d0, d0 + 16): weight tile [hc][16][r8n] chunks, then low [T][r8n].
__device__ __forceinline__ void hcs_up(
    const __nv_bfloat16* __restrict__ up_w, const __nv_bfloat16* normed, const __nv_bfloat16* low,
    __nv_bfloat16* __restrict__ y_out, uint4* tile, uint4* lows, unsigned int q, unsigned int T,
    unsigned int H, unsigned int hc, unsigned int rank, float inv_hc, unsigned int n_groups
) {
    const unsigned int d0 = q * HCS_UP_COLS;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5;
    const unsigned int r8n = rank / 8u;
    const unsigned int seg = HCS_UP_COLS * r8n;

    for (unsigned int i = tid; i < hc * seg; i += HCS_THREADS) {
        const unsigned int s = i / seg;
        const unsigned int d = d0 + (i % seg) / r8n;
        const unsigned int c = i % r8n;
        hcs_cp16(tile + i, up_w + ((size_t)s * H + d) * rank + (size_t)c * 8u, 16u);
    }
    hcs_cp_commit();
    hcs_wait(hcs_sync + HCS_DOWNED, n_groups);

    const uint4* l4 = reinterpret_cast<const uint4*>(low);
    for (unsigned int i = tid; i < T * r8n; i += HCS_THREADS) lows[i] = __ldcg(l4 + i);
    hcs_cp_commit_wait_all();
    __syncthreads();

    const uint4 zero = make_uint4(0u, 0u, 0u, 0u);
    const size_t hc_dim = (size_t)hc * H;
    #pragma unroll 1
    for (unsigned int i = 0; i < 2u; ++i) {
        const unsigned int dl = 2u * warp + i;
        const unsigned int d = d0 + dl;
        #pragma unroll 1
        for (unsigned int t0 = 0; t0 < T; t0 += HCS_UP_TOKENS) {
            const unsigned int nt = min((unsigned int)HCS_UP_TOKENS, T - t0);
            float acc[32];
            #pragma unroll
            for (int k = 0; k < 32; ++k) acc[k] = 0.0f;
            for (unsigned int k = lane; k < r8n; k += 32u) {
                uint4 w[HLG_STREAMS];
                #pragma unroll
                for (int s = 0; s < HLG_STREAMS; ++s) {
                    w[s] = (unsigned int)s < hc ? tile[((unsigned int)s * HCS_UP_COLS + dl) * r8n + k] : zero;
                }
                #pragma unroll
                for (int t = 0; t < HCS_UP_TOKENS; ++t) {
                    if ((unsigned int)t < nt) {
                        const uint4 x = lows[(t0 + (unsigned int)t) * r8n + k];
                        #pragma unroll
                        for (int s = 0; s < HLG_STREAMS; ++s) hlg_dot8(acc[s * HCS_UP_TOKENS + t], w[s], x);
                    }
                }
            }
            const float part = hcs_reduce_scatter16(acc);
            const unsigned int s = (lane & 15u) / HCS_UP_TOKENS;
            const unsigned int t = lane % HCS_UP_TOKENS;
            float term = 0.0f;
            if (lane < 16u && s < hc && t < nt) {
                const float u = __bfloat162float(__float2bfloat16(part));
                const float nv = __bfloat162float(__ldcg(normed + (t0 + t) * hc_dim + (size_t)s * H + d));
                term = qhc_sigmoid(u) * nv;
            }
            float mixed = 0.0f;
            #pragma unroll
            for (int s2 = 0; s2 < HLG_STREAMS; ++s2) {
                const float v = __shfl_sync(0xFFFFFFFFu, term, s2 * HCS_UP_TOKENS + (int)t);
                if ((unsigned int)s2 < hc) mixed += v;
            }
            if (lane < nt) y_out[(size_t)(t0 + lane) * H + d] = __float2bfloat16(mixed * inv_hc);
        }
    }
}

// 2026-10-14: Grid [T * hc + 8 * ceil(rows / 16) + H / 16], block 256, dynamic shared memory
// 16 * max(16 * nj * 32, 16 * hc * r8n) + 16 * max(8 * nj * 32, T * r8n) bytes, nj = hc * H /
// 2048, r8n = rank / 8 (`hc_lowrank_site.rs` computes it).
extern "C" __global__ void __launch_bounds__(HCS_THREADS, 3) hc_lowrank_site(
    const float* __restrict__ streams,           // 2026-10-14: [T, hc, H] FP32
    const __nv_bfloat16* __restrict__ norm_w,    // 2026-10-14: [hc * H]
    const __nv_bfloat16* __restrict__ down_w,    // 2026-10-14: [rank, hc * H]
    const __nv_bfloat16* __restrict__ inject_w,  // 2026-10-14: [hc, hc * H] or null
    const __nv_bfloat16* __restrict__ up_w,      // 2026-10-14: [hc * H, rank]
    __nv_bfloat16* normed,                       // 2026-10-14: [T, hc * H] out
    float* partial,                              // 2026-10-14: [8, rows, T] scratch
    __nv_bfloat16* low,                          // 2026-10-14: [T, rank] out
    __nv_bfloat16* __restrict__ y_out,           // 2026-10-14: [T, H] out
    float* inj_out,                              // 2026-10-14: [T, hc] out with inject_w
    const unsigned int T,
    const unsigned int H,
    const unsigned int hc,
    const unsigned int rank,
    const float eps,
    const float inv_hc
) {
    extern __shared__ uint4 hcs_smem[];
    __shared__ unsigned int s_ticket;
    __shared__ unsigned int s_last;

    const unsigned int K = hc * H;
    const unsigned int nj = K / 2048u;
    const unsigned int r8n = rank / 8u;
    const unsigned int rows = rank + (inject_w != nullptr ? hc : 0u);
    const unsigned int n_stage = T * hc;
    const unsigned int n_groups = (rows + HCS_GROUP_ROWS - 1u) / HCS_GROUP_ROWS;
    const unsigned int n_down = HCS_VWARPS * n_groups;
    const unsigned int tile_u4 = max(HCS_GROUP_ROWS * nj * 32u, HCS_UP_COLS * hc * r8n);
    const unsigned int stage_u4 = max(HCS_DOWN_TOKENS * nj * 32u, T * r8n);
    unsigned int smem_bytes;
    asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(smem_bytes));
    if (blockDim.x != HCS_THREADS || gridDim.x != n_stage + n_down + H / HCS_UP_COLS
        || n_groups > HCS_MAX_GROUPS || nj == 0u || nj > HCS_MAX_NJ || hc > HLG_STREAMS
        || T > HCS_MAX_T || smem_bytes < 16u * (tile_u4 + stage_u4)) {
        __trap();
    }

    if (threadIdx.x == 0) s_ticket = atomicAdd(hcs_sync + HCS_TICKET, 1u);
    __syncthreads();
    const unsigned int q = s_ticket;
    uint4* tile = hcs_smem;
    uint4* stage = hcs_smem + tile_u4;

    if (q < n_stage) {
        hcs_stage(streams, norm_w, normed, reinterpret_cast<float*>(stage), q, H, hc, eps);
        hcs_publish(hcs_sync + HCS_STAGED);
    } else if (q < n_stage + n_down) {
        hcs_down(down_w, inject_w, normed, partial, low, inj_out, tile, stage, q - n_stage, T, K,
                 rank, rows, inv_hc, n_stage, &s_last);
    } else {
        hcs_up(up_w, normed, low, y_out, tile, stage, q - n_stage - n_down, T, H, hc, rank, inv_hc,
               n_groups);
    }

    // 2026-10-14: The last block out zeroes the launch counters for the next launch.
    if (threadIdx.x == 0) {
        __threadfence();
        if (atomicAdd(hcs_sync + HCS_EXITED, 1u) == gridDim.x - 1u) {
            atomicExch(hcs_sync + HCS_TICKET, 0u);
            atomicExch(hcs_sync + HCS_STAGED, 0u);
            atomicExch(hcs_sync + HCS_DOWNED, 0u);
            atomicExch(hcs_sync + HCS_EXITED, 0u);
        }
    }
}
