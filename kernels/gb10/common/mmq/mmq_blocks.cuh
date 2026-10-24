#pragma once

// Quantized block formats read by the MMQ tiles and the K-quant GEMVs, their QK/QR/QI
// constants, the type ids that select a format, and the per-format traits.

#include "mmq_device.cuh"

// Values match ggml's ggml_type ids, which the template instantiations are keyed by.
enum mmq_type {
    MMQ_TYPE_Q2_K  = 10,
    MMQ_TYPE_Q3_K  = 11,
    MMQ_TYPE_Q4_K  = 12,
    MMQ_TYPE_Q6_K  = 14,
    MMQ_TYPE_NVFP4 = 40, // NVFP4 (4 blocks, E4M3 scale)
    MMQ_TYPE_Q2_0  = 42, // PrismML Ternary-Bonsai: 2-bit codes, value (code-1)*d, group 128
};

// QK = number of values after dequantization
// QK_K = super-block size

#define QK_K 256
#define K_SCALE_SIZE 12

// QR = QK / number of values before dequantization
// QI = number of 32 bit integers before dequantization

#define QI2_0 (QK2_0 / 32)
#define QR2_0 1

#define QI_NVFP4 (QK_NVFP4 / (4 * QR_NVFP4))
#define QR_NVFP4 2

#define QI8_0 (QK8_0 / (4 * QR8_0))
#define QR8_0 1

#define QI8_1 (QK8_1 / (4 * QR8_1))
#define QR8_1 1

#define QI2_K (QK_K / (4*QR2_K))
#define QR2_K 4

#define QI3_K (QK_K / (4*QR3_K))
#define QR3_K 4

#define QI4_K (QK_K / (4*QR4_K))
#define QR4_K 2

#define QI6_K (QK_K / (4*QR6_K))
#define QR6_K 2

// PrismML Ternary-Bonsai Q2_0: fp16 block scale d @front, then 128 2-bit codes
// (4 per byte, low-bits-first). Dequant value = (code - 1) * d. 34 bytes / block.
// Byte-identical to Metrale Engine's on-disk `WeightDtype::PackedQ2_0{group:128}` — the
// native decode GEMV (q2_0_gemv_vec.cu) reads the same layout, so no repack.
#define QK2_0 128
typedef struct {
    half d;            // delta (block scale)
    uint8_t qs[QK2_0 / 4];  // 128 codes, 4 per byte
} block_q2_0;
static_assert(sizeof(block_q2_0) == sizeof(half) + QK2_0 / 4, "wrong q2_0 block size/padding");

#define QK_NVFP4 64
#define QK_NVFP4_SUB 16  // sub-block size for per-group scales
typedef struct {
    uint8_t d[QK_NVFP4/QK_NVFP4_SUB]; // UE4M3 scales (4 bytes, one per 16-element sub-block)
    uint8_t qs[QK_NVFP4/2];           // packed 4-bit E2M1 values (32 bytes)
} block_nvfp4;
static_assert(sizeof(block_nvfp4) == sizeof(uint8_t)*(QK_NVFP4/QK_NVFP4_SUB) + QK_NVFP4/2, "wrong nvfp4 block size/padding");

#define QK8_0 32

#define QK8_1 32
typedef struct {
    __extension__ union {
        struct {
            half d; // delta
            half s; // d * sum(qs[i])
        } data;
        half2 ds;
    };
    int8_t qs[QK8_1]; // quants
} block_q8_1;
static_assert(sizeof(block_q8_1) == 2*sizeof(half) + QK8_1, "wrong q8_1 block size/padding");

//
// Super-block quantization structures
//

// 2-bit quantization
// weight is represented as x = a * q + b
// 16 blocks of 16 elements each
// Effectively 2.625 bits per weight
typedef struct {
    uint8_t scales[QK_K/16]; // scales and mins, quantized with 4 bits
    uint8_t qs[QK_K/4];      // quants
    __extension__ union {
        struct {
            half d;    // super-block scale for quantized scales
            half dmin; // super-block scale for quantized mins
        } data;
        half2 dm;
    };
} block_q2_K;
static_assert(sizeof(block_q2_K) == 2*sizeof(half) + QK_K/16 + QK_K/4, "wrong q2_K block size/padding");

// 3-bit quantization
// weight is represented as x = a * q
// 16 blocks of 16 elements each
// Effectively 3.4375 bits per weight
typedef struct {
    uint8_t hmask[QK_K/8]; // quants - high bit
    uint8_t qs[QK_K/4];    // quants - low 2 bits
    uint8_t scales[12];    // scales, quantized with 6 bits
    half d;           // super-block scale
} block_q3_K;
static_assert(sizeof(block_q3_K) == sizeof(half) + QK_K / 4 + QK_K / 8 + 12, "wrong q3_K block size/padding");

// 4-bit quantization
// 8 blocks of 32 elements each
// weight is represented as x = a * q + b
// Effectively 4.5 bits per weight
typedef struct {
    __extension__ union {
        struct {
            half d;    // super-block scale for quantized scales
            half dmin; // super-block scale for quantized mins
        } data;
        half2 dm;
    };
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K/2];           // 4--bit quants
} block_q4_K;
static_assert(sizeof(block_q4_K) == 2*sizeof(half) + K_SCALE_SIZE + QK_K/2, "wrong q4_K block size/padding");

// 6-bit quantization
// weight is represented as x = a * q
// 16 blocks of 16 elements each
// Effectively 6.5625 bits per weight
typedef struct {
    uint8_t ql[QK_K/2];      // quants, lower 4 bits
    uint8_t qh[QK_K/4];      // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits
    half d;             // super-block scale
} block_q6_K;
static_assert(sizeof(block_q6_K) == sizeof(half) + QK_K / 16 + 3*QK_K/4, "wrong q6_K block size/padding");

// e2m1 values (doubled)
// ref: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
static const __device__ int8_t kvalues_mxfp4[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
};

template <mmq_type type>
struct mmq_quant_traits;

template<>
struct mmq_quant_traits<MMQ_TYPE_Q2_0> {
    static constexpr int qk = QK2_0;
    static constexpr int qr = QR2_0;
    static constexpr int qi = QI2_0;
};

template<>
struct mmq_quant_traits<MMQ_TYPE_NVFP4> {
    static constexpr int qk = QK_NVFP4;
    static constexpr int qr = QR_NVFP4;
    static constexpr int qi = QI_NVFP4;
};

template<>
struct mmq_quant_traits<MMQ_TYPE_Q2_K> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR2_K;
    static constexpr int qi = QI2_K;
};

template<>
struct mmq_quant_traits<MMQ_TYPE_Q3_K> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR3_K;
    static constexpr int qi = QI3_K;
};

template<>
struct mmq_quant_traits<MMQ_TYPE_Q4_K> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR4_K;
    static constexpr int qi = QI4_K;
};

template<>
struct mmq_quant_traits<MMQ_TYPE_Q6_K> {
    static constexpr int qk = QK_K;
    static constexpr int qr = QR6_K;
    static constexpr int qi = QI6_K;
};
