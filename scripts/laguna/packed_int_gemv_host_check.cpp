// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-07: Host check of the strix-hip packed INT4 / INT8 GEMV lane arithmetic
// (kernels/strix-hip/laguna-xs-2.1/int4/packed_int_dequant.cuh), with no GPU.
//
// It runs the header's decode and per-lane partial for all 32 lanes of a wave and reduces
// them in the kernel's xor-shuffle tree order, so its output is the value the device kernel
// should store bit for bit. That emulation is compared with:
//   1. literal known answers (hand-packed words, hand-computed outputs: the same cases as
//      crates/model-layers/src/quant_format/packed_int_tests.rs);
//   2. an independent double-precision reference over per-element (q * scale) * x;
//   3. known-bad controls: the same checks with a most-significant-first decode and with a
//      two's-complement decode must fail;
//   4. literal BF16 rounding cases (ties to even, NaN), since the store must be bit-exact.
// Optional: `--tensor <packed.i32> <scale.bf16> <N> <K> <bits>` runs (2) on a raw
// weight_packed / weight_scale pair cut from a checkpoint (e.g. by HTTP range request).
//
// Build and run (any C++17 compiler; exit status 0 means every check passed):
//   c++ -std=c++17 -O2 -ffp-contract=off scripts/laguna/packed_int_gemv_host_check.cpp \
//       -o /tmp/packed_int_check && /tmp/packed_int_check
//
// TODO(gfx1151): the GPU gate launches packed_int_gemv.cu with the same inputs and compares
// device output bytes with this emulation (expected bit-identical) and the reference.

#include "../../kernels/strix-hip/laguna-xs-2.1/int4/packed_int_dequant.cuh"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

namespace {

int failures = 0;

void expect(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL %s\n", what);
        ++failures;
    }
}

// 2026-10-07: Decoders under test: the layout's, and two known-bad controls.
enum class Decode { Layout, MsbFirst, TwosComplement };

int decode(Decode d, int bits, uint32_t word, int i) {
    const int per = 32 / bits;
    const uint32_t mask = (1u << bits) - 1u;
    switch (d) {
        case Decode::Layout:
            return bits == 4 ? pi_code<4>(word, i) : pi_code<8>(word, i);
        case Decode::MsbFirst:
            return (int)((word >> (bits * (per - 1 - i))) & mask) - (1 << (bits - 1));
        case Decode::TwosComplement: {
            int u = (int)((word >> (bits * i)) & mask);
            return u >= (1 << (bits - 1)) ? u - (1 << bits) : u;
        }
    }
    return 0;
}

uint16_t bf16(float f) { return pi_f32_to_bf16_rn(f); }

// 2026-10-07: The device kernel's value for one output: 32 lane partials, xor-shuffle tree
// 16, 8, 4, 2, 1 (lane 0's view), BF16 store.
float emulate_column(int bits, const uint32_t* row_words, const uint16_t* row_scales,
                     const uint16_t* x, int k) {
    float lanes[32];
    for (int l = 0; l < 32; ++l) {
        lanes[l] = bits == 4 ? pi_row_lane_partial<4>(row_words, row_scales, x, k, l, 32)
                             : pi_row_lane_partial<8>(row_words, row_scales, x, k, l, 32);
    }
    for (int off = 16; off >= 1; off /= 2) {
        float next[32];
        for (int l = 0; l < 32; ++l) {
            next[l] = lanes[l] + lanes[l ^ off];
        }
        std::memcpy(lanes, next, sizeof(lanes));
    }
    return pi_bf16_to_f32(pi_f32_to_bf16_rn(lanes[0]));
}

// 2026-10-07: Independent reference: sum_k (q_k * scale_g) * x_k in double, with the
// decoder chosen by `d` (Layout is the truth; the controls must disagree with it).
double reference_column(Decode d, int bits, const uint32_t* row_words,
                        const uint16_t* row_scales, const uint16_t* x, int k) {
    const int per = 32 / bits;
    double acc = 0.0;
    for (int kk = 0; kk < k; ++kk) {
        int q = decode(d, bits, row_words[kk / per], kk % per);
        double w = (double)q * (double)pi_bf16_to_f32(row_scales[kk / 128]);
        acc += w * (double)pi_bf16_to_f32(x[kk]);
    }
    return acc;
}

// 2026-10-07: Literal known answers; returns the number of mismatches under `d`.
int known_answers(Decode d) {
    int bad = 0;
    const uint32_t int4[3] = {0x76543210u, 0xFEDCBA98u, 0x8888888Fu};
    for (int k = 0; k < 16; ++k) bad += decode(d, 4, int4[k / 8], k % 8) != k - 8;
    bad += decode(d, 4, int4[2], 0) != 7;
    for (int i = 1; i < 8; ++i) bad += decode(d, 4, int4[2], i) != 0;
    const int want8[4] = {-127, -128, 127, 0};
    for (int i = 0; i < 4; ++i) bad += decode(d, 8, 0x80FF0001u, i) != want8[i];

    // 2026-10-07: INT4 [2, 256] GEMV: y0 = -144 (x = 1), y1 = 7 (x[0] = 3, x[128] = -4).
    std::vector<uint32_t> w(64);
    for (int j = 0; j < 32; ++j) w[j] = j % 2 ? 0xFEDCBA98u : 0x76543210u;
    for (int j = 32; j < 64; ++j) w[j] = 0x88888888u;
    w[32] = w[48] = 0x8888888Fu;
    const uint16_t s[4] = {bf16(0.25f), bf16(2.0f), bf16(0.5f), bf16(0.125f)};
    std::vector<uint16_t> ones(256, bf16(1.0f)), sparse(256, bf16(0.0f));
    sparse[0] = bf16(3.0f);
    sparse[128] = bf16(-4.0f);
    bad += reference_column(d, 4, &w[0], &s[0], ones.data(), 256) != -144.0;
    bad += reference_column(d, 4, &w[32], &s[2], sparse.data(), 256) != 7.0;
    // 2026-10-07: INT8 [1, 128], every word 0x80FF0001, scale 0.5, x = 1: -2048.
    std::vector<uint32_t> w8(32, 0x80FF0001u);
    const uint16_t s8 = bf16(0.5f);
    bad += reference_column(d, 8, w8.data(), &s8, ones.data(), 128) != -2048.0;
    return bad;
}

// 2026-10-07: The kernel's BF16 store is round to nearest, ties to even, NaN kept quiet:
// the emulation is bit-exact only if this is.
int bf16_rounding_mismatches() {
    const uint32_t in[6] = {0x3F808000u, 0x3F818000u, 0x3F80C000u, 0x3F807FFFu, 0xBF80C000u,
                            0x7F800001u};
    const uint16_t want[6] = {0x3F80u, 0x3F82u, 0x3F81u, 0x3F80u, 0xBF81u, 0x7FC0u};
    int bad = 0;
    for (int i = 0; i < 6; ++i) {
        float f;
        std::memcpy(&f, &in[i], 4);
        bad += pi_f32_to_bf16_rn(f) != want[i];
    }
    return bad;
}

uint32_t lcg(uint64_t& state) {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    return (uint32_t)(state >> 32);
}

// 2026-10-07: Emulation vs reference on one [n, k] weight; tolerance is one BF16 ulp of the
// output plus fp32 accumulation slack relative to sum |w * x|.
bool emulation_matches_reference(int bits, int n, int k, const uint32_t* words,
                                 const uint16_t* scales, const uint16_t* x, double* worst) {
    const int per = 32 / bits;
    bool ok = true;
    for (int row = 0; row < n; ++row) {
        const uint32_t* rw = words + (size_t)row * (k / per);
        const uint16_t* rs = scales + (size_t)row * (k / 128);
        double ref = reference_column(Decode::Layout, bits, rw, rs, x, k);
        double mag = 0.0;
        for (int kk = 0; kk < k; ++kk) {
            mag += std::fabs((double)decode(Decode::Layout, bits, rw[kk / per], kk % per) *
                             pi_bf16_to_f32(rs[kk / 128]) * pi_bf16_to_f32(x[kk]));
        }
        double got = emulate_column(bits, rw, rs, x, k);
        double tol = std::fabs(ref) * (1.0 / 128.0) + mag * 1e-6 + 1e-30;
        double err = std::fabs(got - ref);
        if (err / tol > *worst) *worst = err / tol;
        ok = ok && err <= tol;
    }
    return ok;
}

void random_case(int bits, int n, int k, uint64_t seed) {
    uint64_t st = seed;
    std::vector<uint32_t> words((size_t)n * k * bits / 32);
    for (auto& w : words) w = lcg(st);
    std::vector<uint16_t> scales((size_t)n * (k / 128)), x(k);
    for (auto& s : scales) s = bf16((float)(lcg(st) % 1000 + 1) * 1e-4f);
    for (auto& v : x) v = bf16(((float)(lcg(st) % 2001) - 1000.0f) / 250.0f);
    double worst = 0.0;
    bool ok = emulation_matches_reference(bits, n, k, words.data(), scales.data(), x.data(),
                                          &worst);
    std::printf("random int%d [%d, %d]: worst err/tol %.3f\n", bits, n, k, worst);
    expect(ok, "random emulation vs reference");
    // 2026-10-07: Known-bad control on the same data: the MSB-first reference must disagree.
    const int per = 32 / bits;
    int differ = 0;
    for (int row = 0; row < n; ++row) {
        const uint32_t* rw = words.data() + (size_t)row * (k / per);
        const uint16_t* rs = scales.data() + (size_t)row * (k / 128);
        double a = reference_column(Decode::Layout, bits, rw, rs, x.data(), k);
        double b = reference_column(Decode::MsbFirst, bits, rw, rs, x.data(), k);
        differ += std::fabs(a - b) > std::fabs(a) * (1.0 / 128.0) + 1e-6;
    }
    expect(differ > n / 2, "MSB-first control is detected on random data");
}

// 2026-10-07: Grouped-expert slot mapping: slot s reads activation row s / x_row_div of
// expert ids[s]; an out-of-range id is a zero output.
void grouped_case() {
    const int k = 128, n = 2, experts = 3, top_k = 2, tokens = 2;
    std::vector<std::vector<uint32_t>> w(experts, std::vector<uint32_t>(n * 16));
    std::vector<std::vector<uint16_t>> s(experts, std::vector<uint16_t>(n));
    for (int e = 0; e < experts; ++e) {
        for (int j = 0; j < n * 16; ++j) w[e][j] = 0x88888888u + (uint32_t)(e + 1);  // q0 = e + 1
        for (int r = 0; r < n; ++r) s[e][r] = bf16((float)(r + 1));
    }
    std::vector<uint16_t> x((size_t)tokens * k, bf16(0.0f));
    x[0] = bf16(1.0f);       // token 0, k = 0
    x[k] = bf16(10.0f);      // token 1, k = 0
    const int ids[tokens * top_k] = {2, 0, 1, 7};
    for (int slot = 0; slot < tokens * top_k; ++slot) {
        for (int col = 0; col < n; ++col) {
            int e = ids[slot];
            float got = 0.0f;
            if (e >= 0 && e < experts) {
                got = emulate_column(4, &w[e][col * 16], &s[e][col], &x[(slot / top_k) * k], k);
            }
            // 2026-10-07: q at k = 0 is e + 1 (each word's first code), scale col + 1.
            float want = (e >= 0 && e < experts)
                             ? (float)(e + 1) * (float)(col + 1) * (slot / top_k ? 10.0f : 1.0f)
                             : 0.0f;
            if (got != want) {
                std::printf("slot %d col %d: got %g want %g\n", slot, col, got, want);
                ++failures;
            }
        }
    }
}

std::vector<char> read_file(const char* path) {
    std::ifstream f(path, std::ios::binary);
    return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

void tensor_case(const char* packed, const char* scale, int n, int k, int bits) {
    std::vector<char> p = read_file(packed), sc = read_file(scale);
    if (p.size() != (size_t)n * k * bits / 8 || sc.size() != (size_t)n * (k / 128) * 2) {
        std::printf("tensor sizes %zu / %zu do not match [%d, %d] int%d\n", p.size(),
                    sc.size(), n, k, bits);
        ++failures;
        return;
    }
    std::vector<uint32_t> words(p.size() / 4);
    std::memcpy(words.data(), p.data(), p.size());
    std::vector<uint16_t> scales(sc.size() / 2);
    std::memcpy(scales.data(), sc.data(), sc.size());
    uint64_t st = 7;
    std::vector<uint16_t> x(k);
    for (auto& v : x) v = bf16(((float)(lcg(st) % 2001) - 1000.0f) / 1000.0f);
    double worst = 0.0;
    bool ok = emulation_matches_reference(bits, n, k, words.data(), scales.data(), x.data(),
                                          &worst);
    std::printf("tensor int%d [%d, %d]: worst err/tol %.3f\n", bits, n, k, worst);
    expect(ok, "checkpoint tensor emulation vs reference");
}

}  // namespace

int main(int argc, char** argv) {
    expect(known_answers(Decode::Layout) == 0, "layout decoder passes the known answers");
    expect(bf16_rounding_mismatches() == 0, "BF16 store rounds to nearest even");
    int msb = known_answers(Decode::MsbFirst), twos = known_answers(Decode::TwosComplement);
    std::printf("known-bad controls: msb-first %d mismatches, two's complement %d\n", msb, twos);
    expect(msb > 0, "MSB-first control fails the known answers");
    expect(twos > 0, "two's-complement control fails the known answers");
    random_case(4, 64, 2048, 1);
    random_case(4, 64, 512, 2);
    random_case(8, 64, 2048, 3);
    random_case(8, 64, 512, 4);
    grouped_case();
    if (argc == 7 && std::strcmp(argv[1], "--tensor") == 0) {
        tensor_case(argv[2], argv[3], std::atoi(argv[4]), std::atoi(argv[5]), std::atoi(argv[6]));
    }
    std::printf("%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures,
                failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
