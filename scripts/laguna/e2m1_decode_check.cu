// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-07: Standalone constructed CUDA oracle for the actual shared decoder.
// Compile PTX and the driver host separately; see the performance diagnostic doc.
// No checkpoint, learned payload, or model support registration is involved.
#ifndef E2M1_HOST
#include "../../kernels/gb10/minimax-m2-229b/nvfp4/moe_w4a16_grouped_gemm.cu"
#else
#include <cuda.h>
#endif
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifndef E2M1_HOST
// 2026-10-07: A literal independent table retains signed zero, unlike abs-only controls.
extern "C" __global__ void compare_decode(unsigned* output) {
    const float oracle[16] = {
        0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
        -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
    const unsigned scales2[11] = {
        0u, 0x80000000u, 0x3f800000u, 0xbf800000u, 0x3e000000u,
        0x3f9e064bu, 0x0d800000u, 0x71800000u, 0x7f800000u,
        0xff800000u, 0x7fc00000u};
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= 16 * 256 * 11) return;
    unsigned nibble = row % 16, scale = (row / 16) % 256;
    __nv_fp8_e4m3 fp8;
    *(unsigned char*)&fp8 = scale;
    float second = __uint_as_float(scales2[row / (16 * 256)]);
    float actual = moe_e2m1_bit_value(nibble), expected = oracle[nibble];
    float a1 = actual * (float)fp8, e1 = expected * (float)fp8;
    float a2 = a1 * second, e2 = e1 * second;
    unsigned* out = output + row * 10;
    out[0] = __float_as_uint(actual); out[1] = __float_as_uint(expected);
    out[2] = __float_as_uint(a1); out[3] = __float_as_uint(e1);
    out[4] = __float_as_uint(a2); out[5] = __float_as_uint(e2);
    out[6] = __bfloat16_as_ushort(__float2bfloat16(a2));
    out[7] = __bfloat16_as_ushort(__float2bfloat16(e2));
    out[8] = __float_as_uint(moe_e2m1_bit_value(nibble & 7));
    out[9] = __float_as_uint(expected * ((float)fp8 * second));
}

#else
static void checked(CUresult status) {
    if (status != CUDA_SUCCESS) {
        const char* message = nullptr;
        cuGetErrorString(status, &message);
        std::fprintf(stderr, "%s\n", message ? message : "CUDA driver failure");
        std::exit(2);
    }
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    checked(cuInit(0));
    CUdevice gpu;
    CUcontext context;
    CUmodule module;
    CUfunction function;
    checked(cuDeviceGet(&gpu, 0));
    checked(cuCtxCreate(&context, nullptr, 0, gpu));
    checked(cuModuleLoad(&module, argv[1]));
    checked(cuModuleGetFunction(&function, module, "compare_decode"));
    constexpr unsigned rows = 16 * 256 * 11;
    std::vector<unsigned> result(rows * 10);
    CUdeviceptr device;
    checked(cuMemAlloc(&device, result.size() * sizeof(unsigned)));
    void* args[] = {&device};
    checked(cuLaunchKernel(function, (rows + 255) / 256, 1, 1, 256, 1, 1, 0, nullptr, args, nullptr));
    checked(cuMemcpyDtoH(result.data(), device, result.size() * sizeof(unsigned)));
    checked(cuMemFree(device));
    checked(cuModuleUnload(module));
    checked(cuCtxDestroy(context));
    unsigned errors = 0, sign_control = 0, order_control = 0;
    for (unsigned row = 0; row < rows; ++row) {
        const unsigned* p = result.data() + row * 10;
        for (unsigned stage = 0; stage < 4; ++stage) errors += p[2 * stage] != p[2 * stage + 1];
        sign_control += p[8] != p[1];
        // 2026-10-07: Only finite pairs establish a numerical reassociation difference.
        if ((p[9] & 0x7f800000u) != 0x7f800000u &&
            (p[5] & 0x7f800000u) != 0x7f800000u) order_control += p[9] != p[5];
    }
    std::printf("rows=%u raw_stage_errors=%u sign_control=%u finite_order_control=%u\n",
                rows, errors, sign_control, order_control);
    return errors || !sign_control || !order_control;
}

#endif
