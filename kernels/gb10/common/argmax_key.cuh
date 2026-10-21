// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-21: The argmax keys of argmax_key.cu, shared with screen_head.cu: a u64 key per logit
// whose max over any split of a row names the token argmax_bf16 / argmax_bf16_batch return for
// the whole row, ties included (the p key), or the masked feed pick (the q key). Moved out of
// argmax_key.cu unchanged; crates/sampling/src/feed_argmax_key.rs is the host specification.
//
// Owner: gb10 kernels.
// Invariants: those of argmax_key.cu.

#pragma once

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

// 2026-10-21: The token a nonzero p key names (argmax_key_resolve's decode, moved here); 0 for
// the zero key, as argmax_bf16 returns 0 when no logit is above the -1e30 floor.
__device__ __forceinline__ unsigned int plain_key_index(unsigned long long kp) {
    if (kp == 0ull) {
        return 0u;
    }
    const unsigned int tie = (unsigned int)kp;
    const unsigned int t =
        key_bitrev_tid(~(tie >> KEY_STRIDE_BITS) & ((1u << KEY_TID_BITS) - 1u));
    const unsigned int k = ~tie & ((1u << KEY_STRIDE_BITS) - 1u);
    return k * KEY_BLOCK + t;
}
