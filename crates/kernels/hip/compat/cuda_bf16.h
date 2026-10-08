#pragma once
#include <hip/hip_bf16.h>
typedef __hip_bfloat16  __nv_bfloat16;
typedef __hip_bfloat162 __nv_bfloat162;
#ifndef METRALE_CVTA_COMPAT
#define METRALE_CVTA_COMPAT
#define __cvta_generic_to_shared(p) ((unsigned long long)(size_t)(p))
#endif
#ifndef METRALE_BF16_RN_COMPAT
#define METRALE_BF16_RN_COMPAT
// 2026-10-07: ROCm 7.2.1's hip_bf16.h declares no __float2bfloat16_rn. Its
// __float2bfloat16 rounds to nearest even (hip_bf16.h builds its own
// __float22bfloat162_rn on it), which is the CUDA _rn contract. The macro
// also wins over a HIP release that declares the function itself.
static __host__ __device__ inline __hip_bfloat16 metrale_float2bfloat16_rn(float f) {
    return __float2bfloat16(f);
}
#define __float2bfloat16_rn metrale_float2bfloat16_rn
#endif
