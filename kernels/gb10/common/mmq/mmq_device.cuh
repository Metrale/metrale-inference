#pragma once

// MMQ device primitives: target capability, warp size and reductions, the trap for an
// uncompiled arch, dp4a, PDL wait, and the E4M3 / E2M1 scalar conversions.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cfloat>
#include <cstdint>
#include <cstdio>

#define STRINGIZE_IMPL(...) #__VA_ARGS__
#define STRINGIZE(...) STRINGIZE_IMPL(__VA_ARGS__)

#define WARP_SIZE 32

// While BW spans CC 1000, 1100 & 1200, we are integrating Tensor Core instructions available to 1200 family, see
// https://docs.nvidia.com/cutlass/media/docs/cpp/blackwell_functionality.html#blackwell-sm120-gemms
#define MMQ_CC_BLACKWELL       1200
#define MMQ_CC_RUBIN           1300

// Block-scaled FP4 MMA: compute capability 12.x (consumer Blackwell, GB10).
#if __CUDA_ARCH__ >= MMQ_CC_BLACKWELL && __CUDA_ARCH__ < MMQ_CC_RUBIN
#    define BLACKWELL_MMA_AVAILABLE
#endif // __CUDA_ARCH__ >= MMQ_CC_BLACKWELL && __CUDA_ARCH__ < MMQ_CC_RUBIN

#define MMQ_PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))

static constexpr __device__ int mmq_physical_warp_size() {
    return 32;
}

[[noreturn]]
static __device__ void no_device_code(
    const char * file_name, const int line, const char * function_name, const int arch, const char * arch_list) {
    printf("%s:%d: ERROR: CUDA kernel %s has no device code compatible with CUDA arch %d. this module was compiled for: %s\n",
           file_name, line, function_name, arch, arch_list);
    __trap();

    (void) no_device_code; // suppress unused function warning
}

#define NO_DEVICE_CODE no_device_code(__FILE__, __LINE__, __FUNCTION__, __CUDA_ARCH__, STRINGIZE(__CUDA_ARCH_LIST__))

static __device__ __forceinline__ void mmq_pdl_sync() {
    cudaGridDependencySynchronize();
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, width);
    }
    return x;
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_max(float x) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, offset, width));
    }
    return x;
}

static __device__ __forceinline__ int mmq_dp4a(const int a, const int b, int c) {
    return __dp4a(a, b, c);
}

static __device__ __forceinline__ float mmq_ue4m3_to_fp32(uint8_t x) {
    const uint32_t bits = x * (x != 0x7F && x != 0xFF); // Convert NaN to 0.0f to match CPU implementation.
    const __nv_fp8_e4m3 xf = *reinterpret_cast<const __nv_fp8_e4m3 *>(&bits);
    return static_cast<float>(xf) / 2;
}

static __device__ __forceinline__ uint8_t mmq_fp32_to_ue4m3(float x) {
#if defined(BLACKWELL_MMA_AVAILABLE) // This is used for NVFP4 subblock scale quantizations only
    if (!(x > 0.0f)) {
        return 0;
    }
    const __nv_fp8_e4m3 xf(x);
    return xf.__x;
#else
     NO_DEVICE_CODE; // Used only for NVFP4 Scales for Activations, only for Blackwell
#endif // defined(BLACKWELL_MMA_AVAILABLE)
}

__device__ __forceinline__ uint8_t mmq_float_to_fp4_e2m1(float x, float e) {
    const uint8_t sign_bit = (x < 0.0f) << 3;
    float         ax       = fabsf(x) * e;

    // Positive LUT
    static constexpr float pos_lut[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };

    int   best_i   = 0;
    float best_err = fabsf(ax - pos_lut[0]);

#pragma unroll
    for (int i = 1; i < 8; ++i) {
        const float err = fabsf(ax - pos_lut[i]);
        if (err < best_err) {
            best_err = err;
            best_i   = i;
        }
    }

    return static_cast<uint8_t>(best_i | sign_bit);
}
