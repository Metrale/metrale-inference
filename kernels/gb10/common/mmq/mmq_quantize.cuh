#pragma once

// Activation quantizers: rows of F32 or BF16 to block_q8_1_mmq (three scale layouts) or to
// block_fp4_mmq (NVFP4, one E4M3 scale per 16 values).

#include "mmq_layout.cuh"

// Device body of the NVFP4 activation quantizer; nvfp4_mmq.cu's extern "C" entries call it (BF16 in).
template <typename src_t = float>
static __device__ __forceinline__ void quantize_mmq_nvfp4_worker(
        const src_t * __restrict__ x, const int32_t * __restrict__ ids, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2) {
#if defined(BLACKWELL_MMA_AVAILABLE)

    const int64_t i0_base = ((int64_t) blockDim.x * blockIdx.y + threadIdx.x) * QK_NVFP4_SUB;
    if (i0_base >= ne0) {
        return;
    }

    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;
    const int64_t i01 = ids ? ids[i1] : i1;
    const int64_t k_block = i0_base / QK_K;
    const int64_t blocks_per_col = (ne0 + QK_K - 1) / QK_K;
    if (k_block >= blocks_per_col) {
        return;
    }

    const int64_t ib = blockIdx.z * ((int64_t) blocks_per_col * ne1) + k_block * ne1 + blockIdx.x;
    block_fp4_mmq * y = (block_fp4_mmq *) vy;
    block_fp4_mmq * yb = y + ib;

    const int sub = (i0_base % QK_K) / QK_NVFP4_SUB;

    float vals_raw[QK_NVFP4_SUB];
    float amax_raw = 0.0f;
    const int64_t base_idx = i3 * s03 + i2 * s02 + i01 * s01;
#pragma unroll
    for (int k = 0; k < QK_NVFP4_SUB; k++) {
        const int64_t i00 = i0_base + k;
        if (i00 < ne00) {
            const float v = static_cast<float>(x[base_idx + i00]);
            vals_raw[k] = v;
            amax_raw = fmaxf(amax_raw, fabsf(v));
        } else {
            vals_raw[k] = 0.0f;
        }
    }

    static constexpr int test_offsets[5] = { 0, -1, 1, -2, 2};
    const int first_fp8_code = (int) mmq_fp32_to_ue4m3(amax_raw / 6.0f);

    float best_err = FLT_MAX;
    uint8_t fp8_code = 0;
    float subblock_scale = 0.0f;

#pragma unroll // Check +/- 2 to find best code to reduce NVFP4 activation loss. Negligible overhead on Blackwell.
    for (int i = 0; i < 5; i++) {
        const int test_code = first_fp8_code + test_offsets[i];
        if (test_code < 0 || test_code > 0x7e) {
            continue;
        }
        const uint8_t code = (uint8_t) test_code;
        const float test_scale = mmq_ue4m3_to_fp32(code);
        const float test_inv_scale = test_scale > 0.0f ? 0.5f / test_scale : 0.0f;
        float cur_err = 0.0f;
#pragma unroll
        for (int k = 0; k < QK_NVFP4_SUB; ++k) {
            const float v = vals_raw[k];
            const uint8_t q = mmq_float_to_fp4_e2m1(v, test_inv_scale);
            const float err_diff = fabsf(v) - fabsf(kvalues_mxfp4[q & 0x7]) * test_scale;
            cur_err = fmaf(err_diff, err_diff, cur_err);
        }

        if (cur_err < best_err) {
            best_err = cur_err;
            fp8_code = test_code;
            subblock_scale = test_scale;
        }
    }

    const float inv_scale = subblock_scale > 0.0f ? 0.5f / subblock_scale : 0.0f;
    uint32_t q0 = 0;
    uint32_t q1 = 0;
#pragma unroll // this is faster than the previous __nv_fp4x4_e2m1
    for (int k = 0; k < QK_NVFP4_SUB / 4; ++k) {
        q0 |= (uint32_t) mmq_float_to_fp4_e2m1(vals_raw[k +  0], inv_scale) << (8 * k);
        q0 |= (uint32_t) mmq_float_to_fp4_e2m1(vals_raw[k +  8], inv_scale) << (8 * k + 4);
        q1 |= (uint32_t) mmq_float_to_fp4_e2m1(vals_raw[k +  4], inv_scale) << (8 * k);
        q1 |= (uint32_t) mmq_float_to_fp4_e2m1(vals_raw[k + 12], inv_scale) << (8 * k + 4);
    }

    uint32_t * yqs = reinterpret_cast<uint32_t *>(yb->qs);
    yqs[2 * sub + 0] = q0;
    yqs[2 * sub + 1] = q1;
    reinterpret_cast<uint8_t *>(yb->d4)[sub] = fp8_code;
#else
    NO_DEVICE_CODE; // This is for Blackwell NVFP4 activations only.
#endif // defined(BLACKWELL_MMA_AVAILABLE)

}

// Device body of the q8_1 activation quantizer; the extern "C" entries of q4k_mmq.cu and kquant_moe.cu call it.
template <mmq_q8_1_ds_layout ds_layout, typename src_t = float> // METRALE: src_t for bf16-input activations
static __device__ __forceinline__ void quantize_mmq_q8_1_worker(
        const src_t * __restrict__ x, const int32_t * __restrict__ ids, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int ne1, const int ne2) {

    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum   = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

    const int64_t i0 = ((int64_t)blockDim.x*blockIdx.y + threadIdx.x)*4;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;

    const int64_t i00 = i0;
    mmq_pdl_sync();
    const int64_t i01 = ids ? ids[i1] : i1;
    const int64_t i02 = i2;
    const int64_t i03 = i3;

    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;

    const int64_t ib0 = blockIdx.z*((int64_t)gridDim.x*gridDim.y*blockDim.x/QK8_1); // first block of channel
    const int64_t ib  = ib0 + (i0 / (4*QK8_1))*ne1 + blockIdx.x;                    // block index in channel
    const int64_t iqs = i0 % (4*QK8_1);                                             // quant index in block

    // Load 4 values per thread and calculate max. abs. value between them:
    // METRALE: src_t==bf16 reads 4 consecutive bf16 (Metrale Engine activations); float path = original float4 load.
    const int64_t eidx = i03*s03 + i02*s02 + i01*s01 + i00;
    float4 xi;
    if (i0 >= ne00) { xi = make_float4(0.0f, 0.0f, 0.0f, 0.0f); }
    else if constexpr (sizeof(src_t) == 2) {
        const __nv_bfloat16 * xb = (const __nv_bfloat16 *) x + eidx;
        xi = make_float4(__bfloat162float(xb[0]), __bfloat162float(xb[1]), __bfloat162float(xb[2]), __bfloat162float(xb[3]));
    } else {
        xi = ((const float4 *) x)[eidx/4];
    }
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));

    // Exchange max. abs. value between vals_per_scale/4 threads.
#pragma unroll
    for (int offset = vals_per_scale/8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }

    float sum;
    if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = xi.x + xi.y + xi.z + xi.w;

        // Calculate sums across vals_per_sum/4 threads.
#pragma unroll
        for (int offset = vals_per_sum/8; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
        }
    }

    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = roundf(xi.x*d_inv);
    q.y = roundf(xi.y*d_inv);
    q.z = roundf(xi.z*d_inv);
    q.w = roundf(xi.w*d_inv);

    // Write back 4 int8 values as a single 32 bit value for better memory bandwidth:
    char4 * yqs4 = (char4 *) y[ib].qs;
    yqs4[iqs/4] = q;

    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        if (iqs % 16 != 0 || iqs >= 96) {
            return;
        }

        y[ib].d2s6[2 + iqs/16] = sum;

        if (iqs % 64 != 0) {
            return;
        }

        const float d = 1.0f / d_inv;

        y[ib].d2s6[iqs/64] = d;

        return;
    }

    if (iqs % 32 != 0) {
        return;
    }

    const float d = 1.0f / d_inv;

    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
        y[ib].ds4[iqs/32] = make_half2(d, sum);
    } else {
        y[ib].d4[iqs/32]  = d;
    }
}
