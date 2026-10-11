// SPDX-License-Identifier: MIT OR Apache-2.0
// 2026-10-11: The GLM-5.3 vocab-sharded draft head's cross-rank (max, argmax) exchange on the
// device (glm5next_mtp_head/head_xchg.rs is the host form and the oracle).
//
// Owner: gb10 kernels.
// Invariants:
// - The buffer is `world` BF16 logit lanes, then three BF16 base-256 digit lanes per rank:
//   4 * world lanes. A rank writes only its own lanes and zeros the rest, so a BF16 SUM
//   all-reduce delivers every rank's lanes untouched.
// - glm5next_mtp_xchg_pack writes the bytes head_xchg::pack writes: the logit lane is the head's
//   own BF16 output copied bit for bit, each digit d is the BF16 of an integer 0..=255, exact
//   (the host truncates the f32 bits, which for such an integer is the same value).
// - glm5next_mtp_xchg_pick decides as head_xchg::pick: the highest logit, a tie to the lowest
//   rank (strictly greater wins), the global id summed from the winner's digits.

#include <cuda_bf16.h>

// 2026-10-11: One block of 32 threads. `arg` holds this rank's local argmax (u32) of its vocab
// shard starting at row `v0`; `logits` that shard's BF16 logits.
extern "C" __global__ void glm5next_mtp_xchg_pack(
    const unsigned int* __restrict__ arg,
    const __nv_bfloat16* __restrict__ logits,
    __nv_bfloat16* __restrict__ xchg,
    unsigned int world,
    unsigned int rank,
    unsigned int v0
) {
    const unsigned int lanes = 4u * world;
    const unsigned int local = arg[0];
    const unsigned int g = v0 + local;
    for (unsigned int i = threadIdx.x; i < lanes; i += blockDim.x) {
        __nv_bfloat16 v = __float2bfloat16(0.0f);
        if (i == rank) {
            v = logits[local];
        } else if (i >= world) {
            const unsigned int owner = (i - world) / 3u;
            const unsigned int d = (i - world) % 3u;
            if (owner == rank) {
                v = __float2bfloat16((float)((g >> (8u * d)) & 0xFFu));
            }
        }
        xchg[i] = v;
    }
}

// 2026-10-11: One thread. Writes the winning global token id to `out[0]`.
extern "C" __global__ void glm5next_mtp_xchg_pick(
    const __nv_bfloat16* __restrict__ xchg,
    unsigned int world,
    unsigned int* __restrict__ out
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    unsigned int win = 0;
    for (unsigned int r = 1; r < world; ++r) {
        if (__bfloat162float(xchg[r]) > __bfloat162float(xchg[win])) win = r;
    }
    unsigned int idx = 0;
    for (unsigned int d = 0; d < 3u; ++d) {
        const unsigned int digit = (unsigned int)__bfloat162float(xchg[world + win * 3u + d]);
        idx += digit << (8u * d);
    }
    out[0] = idx;
}
