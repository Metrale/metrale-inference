// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-10: The sharded feed argmax. Each rank turns its slice of a logits row into two u64
// keys, the keys combine with a u64 max all-reduce (16 bytes instead of the row), and the
// combined keys resolve to the token argmax_bf16_batch_feed (argmax_feed.cu) returns for the
// whole row, ties included. crates/sampling/src/feed_argmax_key.rs is the host specification;
// its tests pin the equivalence for every split of a row.
//
// Owner: gb10 kernels.
// Invariants:
// - keys[0] (p) and keys[1] (q) are zeroed before argmax_key_shard_bf16 runs; every launch
//   over a slice only raises them (atomicMax), so slices may be keyed in any order and on any
//   rank before the max all-reduce.
// - The p key reproduces argmax_bf16_batch's 1024-thread scan and halving tree: value, then
//   the smaller bit-reversed thread id (i mod 1024), then the smaller i; values at or below
//   -1e30 and NaN give 0. The q key: the value with the mask ids at -inf, then the larger i;
//   NaN gives 0. -0.0 keys as +0.0.

#include <cuda_bf16.h>
#include <math_constants.h>

#define KEY_TID_BITS 10u
#define KEY_STRIDE_BITS 22u
#define KEY_BLOCK 1024u

// 2026-10-10: Bit tests rather than float compares for -0.0 and NaN, so no math-flag choice
// can fold them away.
__device__ __forceinline__ bool key_is_nan(float v) {
    return (__float_as_uint(v) & 0x7FFFFFFFu) > 0x7F800000u;
}

__device__ __forceinline__ unsigned int key_order_bits(float v) {
    unsigned int b = __float_as_uint(v);
    if ((b & 0x7FFFFFFFu) == 0u) {
        b = 0u;
    }
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__device__ __forceinline__ unsigned int key_bitrev_tid(unsigned int t) {
    return __brev(t) >> (32u - KEY_TID_BITS);
}

__device__ __forceinline__ unsigned long long plain_key(float v, unsigned int i) {
    if (key_is_nan(v) || v <= -1e30f) {
        return 0ull;
    }
    const unsigned int t = i % KEY_BLOCK;
    const unsigned int k = i / KEY_BLOCK;
    const unsigned int tie = ((~key_bitrev_tid(t) & ((1u << KEY_TID_BITS) - 1u)) << KEY_STRIDE_BITS)
        | (~k & ((1u << KEY_STRIDE_BITS) - 1u));
    return ((unsigned long long)key_order_bits(v) << 32) | tie;
}

__device__ __forceinline__ unsigned long long masked_key(
    float v, unsigned int i, unsigned int m0, unsigned int m1) {
    const float w = (i == m0 || i == m1) ? -CUDART_INF_F : v;
    if (key_is_nan(w)) {
        return 0ull;
    }
    return ((unsigned long long)key_order_bits(w) << 32) | i;
}

__device__ __forceinline__ unsigned long long warp_max_u64(unsigned long long x) {
    for (int o = 16; o > 0; o >>= 1) {
        const unsigned long long y = __shfl_xor_sync(0xFFFFFFFFu, x, o);
        x = y > x ? y : x;
    }
    return x;
}

// 2026-10-10: Keys of `shard[0..n_local]`, whose entry j is global index `base + j`. Any grid;
// blockDim.x a multiple of 32, at most 1024.
extern "C" __global__ void argmax_key_shard_bf16(
    const __nv_bfloat16* __restrict__ shard,
    const unsigned int* __restrict__ masks,  // 2026-10-10: [2], (m0, m1); 0xFFFFFFFF = none
    unsigned long long* __restrict__ keys,   // 2026-10-10: [2], (p, q)
    unsigned int base,
    unsigned int n_local
) {
    const unsigned int m0 = masks[0];
    const unsigned int m1 = masks[1];
    unsigned long long p = 0ull;
    unsigned long long q = 0ull;
    for (unsigned int j = blockIdx.x * blockDim.x + threadIdx.x; j < n_local;
         j += gridDim.x * blockDim.x) {
        const float v = __bfloat162float(shard[j]);
        const unsigned int i = base + j;
        const unsigned long long kp = plain_key(v, i);
        const unsigned long long kq = masked_key(v, i, m0, m1);
        p = kp > p ? kp : p;
        q = kq > q ? kq : q;
    }
    __shared__ unsigned long long s_p[32];
    __shared__ unsigned long long s_q[32];
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    p = warp_max_u64(p);
    q = warp_max_u64(q);
    if (lane == 0) {
        s_p[warp] = p;
        s_q[warp] = q;
    }
    __syncthreads();
    if (warp == 0) {
        const unsigned int warps = blockDim.x >> 5;
        p = lane < warps ? s_p[lane] : 0ull;
        q = lane < warps ? s_q[lane] : 0ull;
        p = warp_max_u64(p);
        q = warp_max_u64(q);
        if (lane == 0) {
            if (p != 0ull) {
                atomicMax(&keys[0], p);
            }
            if (q != 0ull) {
                atomicMax(&keys[1], q);
            }
        }
    }
}

// 2026-10-10: The token for the combined keys: the max of this rank's `keys` and the
// `n_peers` peer key pairs that a one-shot exchange left at `peers + slot * slot_bytes`
// (CommBackend::exchange_partials; n_peers = 0 when `keys` already holds every slice), then
// q's index when p's lands on a mask id, else p's (argmax_bf16_batch_feed's rule). One thread.
extern "C" __global__ void argmax_key_resolve(
    const unsigned long long* __restrict__ keys,
    const unsigned char* __restrict__ peers,
    unsigned long long slot_bytes,
    unsigned int n_peers,
    const unsigned int* __restrict__ masks,
    unsigned int* __restrict__ cell
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    unsigned long long kp = keys[0];
    unsigned long long kq = keys[1];
    for (unsigned int s = 0; s < n_peers; ++s) {
        const unsigned long long* pk =
            (const unsigned long long*)(peers + (unsigned long long)s * slot_bytes);
        kp = pk[0] > kp ? pk[0] : kp;
        kq = pk[1] > kq ? pk[1] : kq;
    }
    unsigned int p = 0u;
    if (kp != 0ull) {
        const unsigned int tie = (unsigned int)kp;
        const unsigned int t =
            key_bitrev_tid(~(tie >> KEY_STRIDE_BITS) & ((1u << KEY_TID_BITS) - 1u));
        const unsigned int k = ~tie & ((1u << KEY_STRIDE_BITS) - 1u);
        p = k * KEY_BLOCK + t;
    }
    cell[0] = (p == masks[0] || p == masks[1]) ? (unsigned int)kq : p;
}
