// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-11: Device-step kernels for the K=2 MTP step (METRALE_GLM_MTP_DEVICE_STEP parts 3+4):
// the moves that today wait for the host's verdict, made on the device from the device verdict
// (mtp_k2_verdict.cu), so the host can read the verdict a step late. Host specifications:
// metrale_model_layers::speculative::device_step (Rust); their tests pin these rules.
//
// Owner: gb10 kernels.
// Invariants:
// - predicated_copy_2d copies row r (width bytes from src + r*src_pitch to dst + r*dst_pitch) for
//   every r < height exactly when flag[0] == match; otherwise it writes nothing. Rows must not
//   overlap. 16-byte vectors when every address, pitch and the width are 16-byte aligned, bytes
//   otherwise.
// - k2_verify_meta writes the 2-row verify metadata for a sequence of length state[0]: positions
//   L and L+1, KV slots block_table[pos / bs] * bs + pos % bs (dummy_block past table_len), and
//   lengths L+1 and L+2, the values TransformerModel::decode_verify_graphed computes on the host.
// - k2_step_advance adds 1 + verdict[0] (the committed rows) to state[0].

#include <stdint.h>

extern "C" __global__ void predicated_copy_2d(
    const unsigned int* __restrict__ flag,
    unsigned int match,
    const unsigned char* __restrict__ src,
    unsigned long long src_pitch,
    unsigned char* __restrict__ dst,
    unsigned long long dst_pitch,
    unsigned long long width,
    unsigned int height
) {
    if (flag[0] != match) {
        return;
    }
    const unsigned int r = blockIdx.y;
    if (r >= height) {
        return;
    }
    const unsigned char* s = src + (unsigned long long)r * src_pitch;
    unsigned char* d = dst + (unsigned long long)r * dst_pitch;
    const unsigned long long tid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    const bool vec = (((uintptr_t)s | (uintptr_t)d | width) & 15ull) == 0ull;
    if (vec) {
        const uint4* s4 = reinterpret_cast<const uint4*>(s);
        uint4* d4 = reinterpret_cast<uint4*>(d);
        for (unsigned long long i = tid; i < width / 16ull; i += stride) {
            d4[i] = s4[i];
        }
    } else {
        for (unsigned long long i = tid; i < width; i += stride) {
            d[i] = s[i];
        }
    }
}

extern "C" __global__ void k2_verify_meta(
    const unsigned int* __restrict__ state,
    const int* __restrict__ block_table,
    unsigned int table_len,
    unsigned int block_size,
    int dummy_block,
    unsigned int* __restrict__ positions,
    long long* __restrict__ slots,
    int* __restrict__ seq_lens
) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    const unsigned int len = state[0];
    for (unsigned int t = 0u; t < 2u; ++t) {
        const unsigned int pos = len + t;
        const unsigned int blk = pos / block_size;
        const int phys = blk < table_len ? block_table[blk] : dummy_block;
        positions[t] = pos;
        slots[t] = (long long)phys * (long long)block_size + (long long)(pos % block_size);
        seq_lens[t] = (int)(pos + 1u);
    }
}

extern "C" __global__ void k2_step_advance(
    const unsigned int* __restrict__ verdict,
    unsigned int* __restrict__ state
) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    state[0] += 1u + verdict[0];
}
