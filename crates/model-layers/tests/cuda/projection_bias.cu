// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-07: Executes the production CUDA epilogue against an independent bit
// oracle; detects the known-bad round-accumulator-before-bias variant.
// Build from repository root on the authorized GPU host:
// nvcc --fmad=false -arch=sm_121 -o /tmp/projection-bias-test \
//   crates/model-layers/tests/cuda/projection_bias.cu
// Run /tmp/projection-bias-test [receipt.json] to preserve raw bit arrays.
// This is primitive parity, not F.linear or model parity.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include "../../../../kernels/gb10/common/projection_bias.cu"

#define CHECK_CUDA(expr) do { \
    const cudaError_t error = (expr); \
    if (error != cudaSuccess) { \
        std::fprintf(stderr, "%s: %s\n", #expr, cudaGetErrorString(error)); \
        return 1; \
    } \
} while (0)

// 2026-10-07: Host integer RNE oracle independent of CUDA BF16 intrinsics.
static uint16_t bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7fffffffU) > 0x7f800000U) return (bits >> 16) | 0x40U;
    return (bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16;
}
static float widen(uint16_t value) {
    uint32_t bits = uint32_t(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}
static bool same(uint16_t lhs, uint16_t rhs) {
    return lhs == rhs || (std::isnan(widen(lhs)) && std::isnan(widen(rhs)));
}

int main(int argc, char** argv) {
    if (argc > 2) return 2;
    const unsigned cols = 259, rows = 3, count = rows * cols;
    const float cases[] = {1.00390625f, -1.00390625f, 0.0f, -0.0f,
        1.01171875f, -1.01171875f, 0x1p-126f, 0x1p-133f,
        65536.0f, -65536.0f, INFINITY, -INFINITY, NAN};
    std::vector<float> accum(count);
    std::vector<uint16_t> bias(cols), expected(count), observed(count), known_bad(count);
    for (unsigned c = 0; c < cols; ++c) bias[c] = bf16(c % 2 ? -0.00390625f : 0.00390625f);
    unsigned detections = 0;
    for (unsigned i = 0; i < count; ++i) {
        accum[i] = cases[i % (sizeof(cases) / sizeof(cases[0]))];
        const float b = widen(bias[i % cols]);
        expected[i] = bf16(accum[i] + b);
        known_bad[i] = bf16(widen(bf16(accum[i])) + b);
        detections += !same(expected[i], known_bad[i]);
    }
    if (expected[0] != 0x3f81 || detections == 0) {
        std::fprintf(stderr, "invalid oracle or ineffective rounding control\n");
        return 1;
    }
    float* d_accum = nullptr;
    __nv_bfloat16 *d_bias = nullptr, *d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_accum, count * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_bias, cols * sizeof(uint16_t)));
    CHECK_CUDA(cudaMalloc(&d_out, count * sizeof(uint16_t)));
    CHECK_CUDA(cudaMemcpy(d_accum, accum.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bias, bias.data(), cols * sizeof(uint16_t), cudaMemcpyHostToDevice));
    projection_bias_bf16<<<(count + 255) / 256, 256>>>(d_accum, d_bias, d_out, count, cols);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(observed.data(), d_out, count * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    for (unsigned i = 0; i < count; ++i) {
        if (!same(expected[i], observed[i])) {
            std::fprintf(stderr, "element %u: expected %04x, got %04x\n", i, expected[i], observed[i]);
            return 1;
        }
    }
    // 2026-10-07: Integer bits keep NaNs and signed zero lossless in offline replay.
    if (argc == 2) {
        FILE* receipt = std::fopen(argv[1], "w");
        if (!receipt) { std::perror("receipt"); return 1; }
        std::fprintf(receipt, "{\"schema\":1,\"rows\":%u,\"cols\":%u,\"accum_f32_bits\":[", rows, cols);
        for (unsigned i = 0; i < count; ++i) {
            uint32_t bits;
            std::memcpy(&bits, &accum[i], sizeof(bits));
            std::fprintf(receipt, "%s%u", i ? "," : "", bits);
        }
        auto array = [receipt](const char* name, const std::vector<uint16_t>& data) {
            std::fprintf(receipt, "],\"%s\":[", name);
            for (unsigned i = 0; i < data.size(); ++i)
                std::fprintf(receipt, "%s%u", i ? "," : "", unsigned(data[i]));
        };
        array("bias_bf16_bits", bias);
        array("expected_bf16_bits", expected);
        array("observed_bf16_bits", observed);
        array("known_bad_bf16_bits", known_bad);
        std::fprintf(receipt, "]}\n");
        if (std::fclose(receipt) != 0) { std::perror("receipt close"); return 1; }
    }
    CHECK_CUDA(cudaFree(d_accum));
    CHECK_CUDA(cudaFree(d_bias));
    CHECK_CUDA(cudaFree(d_out));
    std::printf("PASS %u elements, %u known-bad rounding detections\n", count, detections);
    return 0;
}
