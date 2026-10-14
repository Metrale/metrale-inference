// SPDX-License-Identifier: MIT OR Apache-2.0

#include <cuda_bf16.h>
#include <assert.h>

// 2026-09-25: MoE token routing: sort the (token, k) slots by expert, gather rows, reduce
// expert outputs back to tokens, blend in the shared expert, and build the grouped-GEMM
// work-list.
//
// Owner: gb10 kernels.
// Invariants:
// - A slot is token * topk + k. topk_ids, topk_weights and token_to_perm are indexed by slot.
// - After moe_sort_by_expert, expert e owns the sorted rows
//   [expert_offsets[e], expert_offsets[e + 1]).
// - The row order inside one expert's range follows shared-memory atomic order, so it
//   is not deterministic from run to run.

// 2026-09-25: permuted[row] = hidden_states[sorted_token_ids[row]]. One block per row;
// the threads stride over hidden_size.
extern "C" __global__ void moe_permute_tokens(
    const __nv_bfloat16* __restrict__ hidden_states,
    __nv_bfloat16* __restrict__ permuted,
    const int* __restrict__ sorted_token_ids,
    unsigned int hidden_size,
    unsigned int total_expanded
) {
    unsigned int row = blockIdx.x;
    unsigned int col = threadIdx.x;

    if (row >= total_expanded) return;

    int src_token = sorted_token_ids[row];


    for (unsigned int c = col; c < hidden_size; c += blockDim.x) {
        permuted[row * hidden_size + c] = hidden_states[src_token * hidden_size + c];
    }
}

// 2026-09-25: output[token] = sum over k of topk_weights[token, k] *
// expert_output[token * topk + k]. It reads expert_output in slot order, not sorted
// order: sorted_token_ids is not read. Slots at or past total_expanded are skipped.


extern "C" __global__ void moe_unpermute_reduce(
    const __nv_bfloat16* __restrict__ expert_output,
    __nv_bfloat16* __restrict__ output,
    const int* __restrict__ sorted_token_ids,
    const float* __restrict__ topk_weights,
    unsigned int hidden_size,
    unsigned int num_tokens,
    unsigned int topk,
    unsigned int total_expanded
) {

    unsigned int token = blockIdx.x;
    unsigned int col = threadIdx.x;

    if (token >= num_tokens) return;

    for (unsigned int c = col; c < hidden_size; c += blockDim.x) {
        float acc = 0.0f;
        for (unsigned int k = 0; k < topk; k++) {
            unsigned int perm_row = token * topk + k;
            if (perm_row < total_expanded) {
                float w = topk_weights[token * topk + k];
                float val = __bfloat162float(expert_output[perm_row * hidden_size + c]);
                acc += w * val;
            }
        }
        output[token * hidden_size + c] = __float2bfloat16(acc);
    }
}

// 2026-09-25: expert_counts[topk_ids[slot]] += 1 for each of the num_tokens * topk slots.
// The kernel only adds, so the caller zeroes expert_counts first.
extern "C" __global__ void moe_count_experts(
    const int* __restrict__ topk_ids,
    int* __restrict__ expert_counts,
    unsigned int num_tokens,
    unsigned int topk
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = num_tokens * topk;
    if (idx < total) {
        int expert_id = topk_ids[idx];
        atomicAdd(&expert_counts[expert_id], 1);
    }
}

// 2026-09-25: output[token] = sum over k of topk_weights[token, k] *
// expert_output[token_to_perm[token * topk + k]], with token_to_perm from
// moe_sort_by_expert. One block per token; the threads stride over hidden_size.

extern "C" __global__ void moe_unpermute_reduce_indexed(
    const __nv_bfloat16* __restrict__ expert_output,
    __nv_bfloat16* __restrict__ output,
    const int* __restrict__ token_to_perm,
    const float* __restrict__ topk_weights,
    unsigned int hidden_size,
    unsigned int num_tokens,
    unsigned int topk
) {
    unsigned int token = blockIdx.x;
    if (token >= num_tokens) return;

    for (unsigned int c = threadIdx.x; c < hidden_size; c += blockDim.x) {
        float acc = 0.0f;
        for (unsigned int k = 0; k < topk; k++) {
            int perm_row = token_to_perm[token * topk + k];
            float w = topk_weights[token * topk + k];
            float val = __bfloat162float(expert_output[perm_row * hidden_size + c]);
            acc += w * val;
        }
        output[token * hidden_size + c] = __float2bfloat16(acc);
    }
}

// 2026-09-25: output[token] += sigmoid(dot(normed[token], gate_weight)) * shared_out[token].
// A null gate_weight uses a factor of 1. One block per token. blockDim.x must be a
// multiple of 32 and at most 256: s_dot_partial holds one partial sum per warp, 8 slots.




extern "C" __global__ void moe_batched_blend(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ shared_out,
    const __nv_bfloat16* __restrict__ normed,
    const __nv_bfloat16* __restrict__ gate_weight,
    unsigned int hidden_size,
    unsigned int num_tokens
) {
    __shared__ float s_dot_partial[8];

    unsigned int token = blockIdx.x;
    if (token >= num_tokens) return;

    unsigned int tid = threadIdx.x;
    unsigned int warp_id = tid / 32;
    unsigned int lane = tid % 32;

    const __nv_bfloat16* my_normed = normed + token * hidden_size;
    const __nv_bfloat16* my_shared = shared_out + token * hidden_size;
    __nv_bfloat16* my_output = output + token * hidden_size;



    float local_dot = 0.0f;
    if (gate_weight != 0) {
        for (unsigned int i = tid; i < hidden_size; i += blockDim.x) {
            float n = __bfloat162float(my_normed[i]);
            float g = __bfloat162float(gate_weight[i]);
            local_dot += n * g;
        }
    }


    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_dot += __shfl_down_sync(0xFFFFFFFF, local_dot, offset);
    }
    if (lane == 0) s_dot_partial[warp_id] = local_dot;
    __syncthreads();


    float gate_scalar;
    if (tid == 0) {
        if (gate_weight == 0) {

            gate_scalar = 1.0f;
        } else {
            float total = 0.0f;
            for (unsigned int w = 0; w < blockDim.x / 32; w++) {
                total += s_dot_partial[w];
            }
            gate_scalar = 1.0f / (1.0f + __expf(-total));
        }
        s_dot_partial[0] = gate_scalar;
    }
    __syncthreads();
    gate_scalar = s_dot_partial[0];


    for (unsigned int i = tid; i < hidden_size; i += blockDim.x) {
        float o = __bfloat162float(my_output[i]);
        float s = __bfloat162float(my_shared[i]);
        my_output[i] = __float2bfloat16(o + gate_scalar * s);
    }
}

// 2026-09-25: Counting sort of the total_expanded slots by expert id (topk_ids is indexed
// by slot). Writes expert_offsets[0..=num_experts] as the prefix sum of the per-expert
// counts, and for each slot at sorted position pos: sorted_token_ids[pos] = slot / topk,
// sorted_expert_ids[pos] = its expert, token_to_perm[slot] = pos.
// Runs as one block: thread 0 builds the prefix sum.



extern "C" __global__ void moe_sort_by_expert(
    const unsigned int* __restrict__ topk_ids,
    int* __restrict__ sorted_token_ids,
    int* __restrict__ sorted_expert_ids,
    int* __restrict__ expert_offsets,
    int* __restrict__ token_to_perm,
    unsigned int total_expanded,
    unsigned int num_experts,
    unsigned int topk
) {
    // 2026-09-25: counts and offsets hold 1024 experts; num_experts is not checked against that.
    __shared__ unsigned int counts[1024];
    __shared__ unsigned int offsets[1025];


    for (unsigned int i = threadIdx.x; i < num_experts; i += blockDim.x)
        counts[i] = 0;
    __syncthreads();


    for (unsigned int i = threadIdx.x; i < total_expanded; i += blockDim.x)
        atomicAdd(&counts[topk_ids[i]], 1);
    __syncthreads();


    if (threadIdx.x == 0) {
        offsets[0] = 0;
        for (unsigned int e = 0; e < num_experts; e++)
            offsets[e + 1] = offsets[e] + counts[e];
        for (unsigned int e = 0; e <= num_experts; e++)
            expert_offsets[e] = (int)offsets[e];
    }
    __syncthreads();


    for (unsigned int i = threadIdx.x; i < num_experts; i += blockDim.x)
        counts[i] = 0;
    __syncthreads();


    for (unsigned int i = threadIdx.x; i < total_expanded; i += blockDim.x) {
        unsigned int expert_id = topk_ids[i];
        unsigned int pos = offsets[expert_id] + atomicAdd(&counts[expert_id], 1);
        sorted_token_ids[pos] = (int)(i / topk);
        sorted_expert_ids[pos] = (int)expert_id;
        token_to_perm[i] = (int)pos;
    }
}

// 2026-09-25: Build the compacted work-list read by moe_fp8_grouped_gemm and
// moe_w8a8_grouped_gemm_pm4: one item per (expert, m_tile, n_tile) tile of an expert that
// has rows and a non-null weight pointer. The GEMMs skip a null weight pointer as well.
//   worklist[w*2 + 0] = expert_id
//   worklist[w*2 + 1] = (mt << 6) | nt  (m-tile and n-tile indices; the GEMMs decode >> 6 and & 0x3F)
//   total_tiles[0]    = number of items
// m_tile and n_tiles must match the GEMM's tile shape (PM4_M_TILE rows, PM4_N_TILE
// columns). The worklist capacity is not checked: the caller sizes it for the worst case.
// Items are in expert, then m_tile, then n_tile order. 2026-09-27: up to 1024 experts the whole
// block builds the list in parallel; above that thread 0 alone does.
// The GEMM that reads total_tiles and worklist must run on the same stream after this
// kernel; no event orders them.













#define TILE_WL_MAX_EXPERTS 1024

extern "C" __global__ void moe_build_tile_worklist(
    const int* __restrict__ expert_offsets,
    const unsigned long long* __restrict__ B_weight_ptrs,
    unsigned int* __restrict__ worklist,
    int* __restrict__ total_tiles,
    unsigned int num_experts,
    unsigned int n_tiles,
    unsigned int m_tile
) {
    // 2026-09-27: The whole block builds the list: per-expert item counts in parallel, one
    // serial prefix over them, then every item written in parallel at its prefix position.
    // The list (order and content) is the one the thread-0 loop below emits.
    __shared__ unsigned int base[TILE_WL_MAX_EXPERTS + 1];
    if (num_experts <= TILE_WL_MAX_EXPERTS) {
        for (unsigned int e = threadIdx.x; e < num_experts; e += blockDim.x) {
            int M_e = expert_offsets[e + 1] - expert_offsets[e];
            base[e + 1] = (M_e <= 0 || B_weight_ptrs[e] == 0)
                ? 0u : (((unsigned int)M_e + m_tile - 1) / m_tile) * n_tiles;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            base[0] = 0;
            for (unsigned int e = 0; e < num_experts; e++) base[e + 1] += base[e];
            total_tiles[0] = (int)base[num_experts];
        }
        __syncthreads();
        const unsigned int total = base[num_experts];
        for (unsigned int w = threadIdx.x; w < total; w += blockDim.x) {
            unsigned int lo = 0, hi = num_experts;   // 2026-09-27: last e with base[e] <= w
            while (hi - lo > 1) {
                unsigned int mid = (lo + hi) >> 1;
                if (base[mid] <= w) lo = mid; else hi = mid;
            }
            unsigned int local = w - base[lo];
            unsigned int mt = local / n_tiles;
            unsigned int nt = local - mt * n_tiles;
            assert(mt < (1u << 26) && nt < 64u);
            worklist[w * 2 + 0] = lo;
            worklist[w * 2 + 1] = (mt << 6) | nt;
        }
        return;
    }

    if (threadIdx.x != 0) return;

    unsigned int w = 0;
    for (unsigned int e = 0; e < num_experts; e++) {
        int m_start = expert_offsets[e];
        int M_e = expert_offsets[e + 1] - m_start;
        if (M_e <= 0 || B_weight_ptrs[e] == 0) continue;

        unsigned int mt_e = ((unsigned int)M_e + m_tile - 1) / m_tile;
        for (unsigned int mt = 0; mt < mt_e; mt++) {
            for (unsigned int nt = 0; nt < n_tiles; nt++) {
                // 2026-09-25: the packed word holds nt in its low 6 bits and mt in the upper 26.

                assert(mt < (1u << 26) && nt < 64u);
                worklist[w * 2 + 0] = e;
                worklist[w * 2 + 1] = (mt << 6) | nt;
                w++;
            }
        }
    }
    total_tiles[0] = (int)w;
}

// 2026-10-14: moe_sort_by_expert over many blocks, in a stable order: inside each expert's range
// the slots sit in increasing slot order, so the outputs are deterministic and are the stable
// counting sort's (moe_sort_by_expert's offsets, any of its orders). Three launches on one stream:
//   moe_sort_mb_hist    grid nb = ceil(total / MOE_SORT_MB_CHUNK), block 256: the per-block counts
//                       blk_hist[b * num_experts + e];
//   moe_sort_mb_scan    one block of num_experts threads rounded up to whole warps: blk_hist in
//                       place to each block's first row inside the expert (an exclusive scan over
//                       b), and expert_offsets[0..=num_experts] as moe_sort_by_expert writes them;
//   moe_sort_mb_scatter grid nb, block 256 (MOE_SORT_MB_WARPS warps): the three outputs.
// Block b owns slots [b * CHUNK, (b + 1) * CHUNK); warp w of it the MOE_SORT_MB_PER_WARP slots
// from b * CHUNK + w * PER_WARP, lane l of round r the slot + r * 32 + l. Ranks inside a warp
// come from __match_any_sync, so no atomic decides an order.
// num_experts is at most MOE_SORT_MB_MAX_EXPERTS (the caller checks). An id >= num_experts is
// skipped (not counted, not placed) where moe_sort_by_expert would index past its counts.
#define MOE_SORT_MB_CHUNK 1024u
#define MOE_SORT_MB_WARPS 8u
#define MOE_SORT_MB_PER_WARP (MOE_SORT_MB_CHUNK / MOE_SORT_MB_WARPS)
#define MOE_SORT_MB_ROUNDS (MOE_SORT_MB_PER_WARP / 32u)
#define MOE_SORT_MB_MAX_EXPERTS 1024u

extern "C" __global__ void moe_sort_mb_hist(
    const unsigned int* __restrict__ topk_ids,
    unsigned int* __restrict__ blk_hist,
    unsigned int total_expanded,
    unsigned int num_experts
) {
    __shared__ unsigned int cnt[MOE_SORT_MB_MAX_EXPERTS];
    for (unsigned int e = threadIdx.x; e < num_experts; e += blockDim.x) cnt[e] = 0;
    __syncthreads();
    const unsigned int base = blockIdx.x * MOE_SORT_MB_CHUNK;
    const unsigned int end = min(base + MOE_SORT_MB_CHUNK, total_expanded);
    for (unsigned int i = base + threadIdx.x; i < end; i += blockDim.x) {
        const unsigned int id = topk_ids[i];
        if (id < num_experts) atomicAdd(&cnt[id], 1u);
    }
    __syncthreads();
    for (unsigned int e = threadIdx.x; e < num_experts; e += blockDim.x)
        blk_hist[blockIdx.x * num_experts + e] = cnt[e];
}

// 2026-10-14: bounded at MOE_SORT_MB_MAX_EXPERTS threads, so the widest scan block fits the
// register file (unbounded, ptxas gave it 72 registers: 1024 threads would not launch).
extern "C" __global__ void __launch_bounds__(MOE_SORT_MB_MAX_EXPERTS) moe_sort_mb_scan(
    unsigned int* __restrict__ blk_hist,
    int* __restrict__ expert_offsets,
    unsigned int num_blocks,
    unsigned int num_experts
) {
    __shared__ unsigned int warp_tot[32];
    const unsigned int e = threadIdx.x;
    const unsigned int lane = e & 31u;
    const unsigned int warp = e >> 5;
    unsigned int total = 0;
    if (e < num_experts) {
        // 2026-10-14: sixteen loads in flight before their stores.
        for (unsigned int b0 = 0; b0 < num_blocks; b0 += 16u) {
            unsigned int v[16];
#pragma unroll
            for (unsigned int j = 0; j < 16u; j++)
                v[j] = b0 + j < num_blocks ? blk_hist[(b0 + j) * num_experts + e] : 0u;
#pragma unroll
            for (unsigned int j = 0; j < 16u; j++) {
                if (b0 + j < num_blocks) blk_hist[(b0 + j) * num_experts + e] = total;
                total += v[j];
            }
        }
    }
    unsigned int incl = total;
#pragma unroll
    for (unsigned int o = 1; o < 32u; o <<= 1) {
        const unsigned int y = __shfl_up_sync(0xffffffffu, incl, o);
        if (lane >= o) incl += y;
    }
    if (lane == 31u) warp_tot[warp] = incl;
    __syncthreads();
    if (warp == 0) {
        unsigned int s = lane < (blockDim.x >> 5) ? warp_tot[lane] : 0u;
#pragma unroll
        for (unsigned int o = 1; o < 32u; o <<= 1) {
            const unsigned int y = __shfl_up_sync(0xffffffffu, s, o);
            if (lane >= o) s += y;
        }
        warp_tot[lane] = s;
    }
    __syncthreads();
    const unsigned int excl = incl - total + (warp ? warp_tot[warp - 1] : 0u);
    if (e < num_experts) expert_offsets[e] = (int)excl;
    if (e + 1u == num_experts) expert_offsets[num_experts] = (int)(excl + total);
}

extern "C" __global__ void moe_sort_mb_scatter(
    const unsigned int* __restrict__ topk_ids,
    const unsigned int* __restrict__ blk_hist,
    const int* __restrict__ expert_offsets,
    int* __restrict__ sorted_token_ids,
    int* __restrict__ sorted_expert_ids,
    int* __restrict__ token_to_perm,
    unsigned int total_expanded,
    unsigned int num_experts,
    unsigned int topk
) {
    // 2026-10-14: wcnt[w * num_experts + e]: warp w's count of expert e, then its first row.
    __shared__ unsigned int wcnt[MOE_SORT_MB_WARPS * MOE_SORT_MB_MAX_EXPERTS];
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    for (unsigned int i = threadIdx.x; i < MOE_SORT_MB_WARPS * num_experts; i += blockDim.x)
        wcnt[i] = 0;
    const unsigned int first = blockIdx.x * MOE_SORT_MB_CHUNK + warp * MOE_SORT_MB_PER_WARP + lane;
    unsigned int id[MOE_SORT_MB_ROUNDS];
    unsigned int peers[MOE_SORT_MB_ROUNDS];
#pragma unroll
    for (unsigned int r = 0; r < MOE_SORT_MB_ROUNDS; r++) {
        const unsigned int slot = first + r * 32u;
        id[r] = slot < total_expanded ? topk_ids[slot] : 0xffffffffu;
    }
    __syncthreads();
    unsigned int* const mine = wcnt + warp * num_experts;
#pragma unroll
    for (unsigned int r = 0; r < MOE_SORT_MB_ROUNDS; r++) {
        peers[r] = __match_any_sync(0xffffffffu, id[r]);
        if (id[r] < num_experts && lane == (unsigned int)(__ffs(peers[r]) - 1))
            mine[id[r]] += __popc(peers[r]);
        __syncwarp();
    }
    __syncthreads();
    for (unsigned int e = threadIdx.x; e < num_experts; e += blockDim.x) {
        unsigned int run = (unsigned int)expert_offsets[e] + blk_hist[blockIdx.x * num_experts + e];
        for (unsigned int w = 0; w < MOE_SORT_MB_WARPS; w++) {
            const unsigned int c = wcnt[w * num_experts + e];
            wcnt[w * num_experts + e] = run;
            run += c;
        }
    }
    __syncthreads();
    const unsigned int below = (1u << lane) - 1u;
#pragma unroll
    for (unsigned int r = 0; r < MOE_SORT_MB_ROUNDS; r++) {
        const bool live = id[r] < num_experts;
        if (live) {
            const unsigned int slot = first + r * 32u;
            const unsigned int pos = mine[id[r]] + __popc(peers[r] & below);
            sorted_token_ids[pos] = (int)(slot / topk);
            sorted_expert_ids[pos] = (int)id[r];
            token_to_perm[slot] = (int)pos;
        }
        __syncwarp();
        if (live && lane == (unsigned int)(__ffs(peers[r]) - 1)) mine[id[r]] += __popc(peers[r]);
        __syncwarp();
    }
}
