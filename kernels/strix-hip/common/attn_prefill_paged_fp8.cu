// SPDX-License-Identifier: MIT OR Apache-2.0
// forked-from: kernels/gb10/common/attn_prefill_paged_fp8.cu (2026-10-13; 70 of 82 lines differ, see kernels/FORKS.md)

// 2026-10-13: Kernels `attn_prefill_paged_fp8` and `_64` on strix-hip: paged prefill flash attention over an FP8 E4M3
// KV cache. Q is contiguous BF16. The loader dequantizes K and V to BF16 while it fills shared memory, with k_scale for
// K and v_scale for V: the loader gb10's attn_prefill_paged_fp8_batched.cu uses when METRALE_ATTN_FP8_SMEM is not
// defined.
//
// gb10's attn_prefill_paged_fp8.cu, which this tree compiled through [sources] until now, defines
// METRALE_ATTN_FP8_SMEM: it copies raw E4M3 bytes into shared memory and leaves the dequant to gb10's
// prefill_paged_compute.cuh at the MMA. This tree's prefill_paged_compute.cuh has no such path and reads shared memory
// as BF16, so it attended over raw bytes. A prefill reaches this kernel when it reads its prefix from the FP8 cache,
// i.e. every chunk after the first, which then came out wrong.
//
// Owner: strix-hip kernels.
// Invariants: none beyond the types.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ __nv_bfloat16 fp8_to_bf16(__nv_fp8_storage_t b, float scale) {
    float v = __half2float(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3)) * scale;
    return __float2bfloat16(v);
}

// 2026-10-13: One thread dequantizes 8 E4M3 values and stores them as one uint4 of BF16; rows at or past `kv_l` are
// zeroed. `fp8_cache_stride` is the per-block stride in bytes.
#define LOAD_KV_TILE(cache, bt, smem, kv_s, kv_l, kvh, t, stride) \
    do { \
        const float _sc = ((const void*)(cache) == (const void*)K_cache) ? k_scale : v_scale; \
        const unsigned int _cpr = HDIM / 8; \
        for (unsigned int _i = t; _i < TILE_CHUNKS; _i += (stride)) { \
            unsigned int _row = _i / _cpr, _col = (_i % _cpr) * 8; \
            unsigned int _pos = (kv_s) + _row; \
            if (_pos < (kv_l)) { \
                unsigned int _lb = _pos / cache_block_size; \
                unsigned int _bo = _pos % cache_block_size; \
                unsigned int _pb = (unsigned int)(bt)[_lb]; \
                const __nv_fp8_storage_t* _base = (const __nv_fp8_storage_t*)(cache) \
                    + (unsigned long long)_pb * fp8_cache_stride \
                    + (unsigned long long)_bo * num_kv_heads * head_dim \
                    + (unsigned long long)(kvh) * head_dim + _col; \
                __nv_bfloat16 _v[8]; \
                for (int _j = 0; _j < 8; _j++) \
                    _v[_j] = fp8_to_bf16(_base[_j], _sc); \
                *((uint4*)&(smem)[_row][_col]) = *((uint4*)_v); \
            } else { *((uint4*)&(smem)[_row][_col]) = make_uint4(0,0,0,0); } \
        } \
    } while(0)

#define KERNEL_NAME attn_prefill_paged_fp8
#define K_CACHE_TYPE const void* __restrict__
#define V_CACHE_TYPE const void* __restrict__
#define KERNEL_EXTRA_PARAMS \
    , const float inv_sqrt_d \
    , const float k_scale \
    , const float v_scale \
    , const unsigned long long fp8_cache_stride

#define KERNEL_PREAMBLE

#include "prefill_paged_compute.cuh"
