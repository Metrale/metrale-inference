// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-26: Metal kernels for NLLB-200: bf16 kernels, including batched decode and the
// cache scatter/gather.
//
// Weight layout is HuggingFace `nn.Linear`: weight is [out=N, in=K] row-major,
// consumed transposed (C[m,n] = bias[n] + sum_k A[m,k] * W[n,k]).


#include <metal_stdlib>
using namespace metal;

kernel void nllb_embed_bf16(
    device const uint *ids [[buffer(0)]],
    device const bfloat *table [[buffer(1)]],
    device bfloat *out [[buffer(2)]],
    constant uint &d [[buffer(3)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]])
{
    uint tok = tgid.x;
    uint tid = tid3.x;
    ulong id = ids[tok];
    for (uint i = tid; i < d; i += 256) {
        out[(ulong)tok * d + i] = table[id * d + i];
    }
}

kernel void nllb_scale_bf16(
    device bfloat *x [[buffer(0)]],
    constant uint &n [[buffer(1)]],
    constant float &s [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) {
        x[gid] = bfloat(float(x[gid]) * s);
    }
}

kernel void nllb_add_bf16(
    device bfloat *dst [[buffer(0)]],
    device const bfloat *src [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) {
        dst[gid] = bfloat(float(dst[gid]) + float(src[gid]));
    }
}

kernel void nllb_bias_bf16(
    device bfloat *c [[buffer(0)]],
    device const bfloat *bias [[buffer(1)]],
    constant uint &M [[buffer(2)]],
    constant uint &N [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    ulong total = (ulong)M * N;
    if (gid < total) {
        c[gid] = bfloat(float(c[gid]) + float(bias[gid % N]));
    }
}

kernel void nllb_relu_bf16(
    device bfloat *x [[buffer(0)]],
    constant uint &n [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) {
        x[gid] = bfloat(fmax(float(x[gid]), 0.0f));
    }
}

kernel void nllb_layernorm_bf16(
    device bfloat *x [[buffer(0)]],
    device const bfloat *w [[buffer(1)]],
    device const bfloat *b [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &dim [[buffer(4)]],
    constant float &eps [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]])
{
    uint row = tgid.x;
    if (row >= rows) {
        return;
    }
    uint tid = tid3.x;
    threadgroup float sm[256];
    device bfloat *rowp = x + (ulong)row * dim;

    float local = 0.0f;
    for (uint i = tid; i < dim; i += 256) {
        local += float(rowp[i]);
    }
    sm[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sm[tid] += sm[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = sm[0] / float(dim);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    local = 0.0f;
    for (uint i = tid; i < dim; i += 256) {
        float dv = float(rowp[i]) - mean;
        local += dv * dv;
    }
    sm[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sm[tid] += sm[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv = rsqrt(sm[0] / float(dim) + eps);
    for (uint i = tid; i < dim; i += 256) {
        float v = (float(rowp[i]) - mean) * inv * float(w[i]) + float(b[i]);
        rowp[i] = bfloat(v);
    }
}

kernel void nllb_add_row_bf16(
    device bfloat *dst [[buffer(0)]],
    device const bfloat *row [[buffer(1)]],
    constant uint &n [[buffer(2)]],
    constant uint &d [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < n) {
        dst[gid] = bfloat(float(dst[gid]) + float(row[gid % d]));
    }
}

kernel void nllb_scatter_batched(
    device const bfloat *src [[buffer(0)]],
    device bfloat *dst [[buffer(1)]],
    constant uint &pos [[buffer(2)]],
    constant uint &B [[buffer(3)]],
    constant uint &stride [[buffer(4)]],
    constant uint &d [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    ulong total = (ulong)B * d;
    if (gid >= total) {
        return;
    }
    uint b = gid / d;
    uint i = gid % d;
    dst[((ulong)b * stride + pos) * d + i] = src[gid];
}

kernel void nllb_gather_batched(
    device const bfloat *src [[buffer(0)]],
    device bfloat *dst [[buffer(1)]],
    device const uint *perm [[buffer(2)]],
    constant uint &B [[buffer(3)]],
    constant uint &used [[buffer(4)]],
    constant uint &stride [[buffer(5)]],
    constant uint &d [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
    ulong total = (ulong)B * used * d;
    if (gid >= total) {
        return;
    }
    uint i = gid % d;
    uint t = (gid / d) % used;
    uint b = gid / (used * d);
    uint parent = perm[b];
    dst[((ulong)b * stride + t) * d + i] = src[((ulong)parent * stride + t) * d + i];
}

kernel void nllb_gemv_bf16(
    device const bfloat *x [[buffer(0)]],
    device const bfloat *w [[buffer(1)]],
    device const bfloat *bias [[buffer(2)]],
    device bfloat *y [[buffer(3)]],
    constant uint &N [[buffer(4)]],
    constant uint &K [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]])
{
    if (row >= N) {
        return;
    }
    threadgroup float partial[32];
    float acc = 0.0f;
    for (uint k = tid; k < K; k += tg_size) {
        acc += float(x[k]) * float(w[(ulong)row * K + k]);
    }
    float simd_acc = simd_sum(acc);
    if (simd_lane_id == 0) {
        partial[simd_group_id] = simd_acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_simds = (tg_size + 31u) / 32u;
    if (simd_group_id == 0) {
        float v = (tid < num_simds) ? partial[tid] : 0.0f;
        v = simd_sum(v);
        if (tid == 0) {
            y[row] = bfloat(v + float(bias[row]));
        }
    }
}

kernel void nllb_attn_kv_bf16(
    device const bfloat *q [[buffer(0)]],
    device const bfloat *k [[buffer(1)]],
    device const bfloat *v [[buffer(2)]],
    device bfloat *out [[buffer(3)]],
    constant uint &tq [[buffer(4)]],
    constant uint &tk [[buffer(5)]],
    constant uint &H [[buffer(6)]],
    constant uint &D [[buffer(7)]],
    constant float &scale [[buffer(8)]],
    constant uint &causal [[buffer(9)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]])
{
    uint qh = tgid.x;
    uint query = qh / H;
    uint head = qh % H;
    uint tid = tid3.x;
    threadgroup float scratch[256];
    threadgroup float *scores = scratch;
    threadgroup float *red = scratch + tk;
    uint dmodel = H * D;
    ulong bq = (ulong)query * dmodel + (ulong)head * D;
    uint kmax = causal != 0 ? (query + 1) : tk;

    for (uint j = 0; j < kmax; j++) {
        ulong bk = (ulong)j * dmodel + (ulong)head * D;
        red[tid] = float(q[bq + tid]) * float(k[bk + tid]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = D >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                red[tid] += red[tid + stride];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) {
            scores[j] = red[0] * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        float m = -1.0e30f;
        for (uint j = 0; j < kmax; j++) {
            m = fmax(m, scores[j]);
        }
        float sum = 0.0f;
        for (uint j = 0; j < kmax; j++) {
            scores[j] = exp(scores[j] - m);
            sum += scores[j];
        }
        for (uint j = 0; j < kmax; j++) {
            scores[j] /= sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc = 0.0f;
    for (uint j = 0; j < kmax; j++) {
        acc += scores[j] * float(v[(ulong)j * dmodel + (ulong)head * D + tid]);
    }
    out[bq + tid] = bfloat(acc);
}

kernel void nllb_attn_bdecode(
    device const bfloat *q [[buffer(0)]],
    device const bfloat *kc [[buffer(1)]],
    device const bfloat *vc [[buffer(2)]],
    device bfloat *out [[buffer(3)]],
    constant uint &B [[buffer(4)]],
    constant uint &stride [[buffer(5)]],
    device const uint *tk [[buffer(6)]],
    constant uint &H [[buffer(7)]],
    constant uint &D [[buffer(8)]],
    constant float &scale [[buffer(9)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid3 [[thread_position_in_threadgroup]])
{
    uint bh = tgid.x;
    uint b = bh / H;
    uint head = bh % H;
    if (b >= B) {
        return;
    }
    uint tid = tid3.x;
    uint t = tk[b];
    threadgroup float scratch[256];
    threadgroup float *scores = scratch;
    threadgroup float *red = scratch + t;
    uint dmodel = H * D;
    ulong qbase = (ulong)b * dmodel + (ulong)head * D;
    ulong cbase = (ulong)b * stride * dmodel + (ulong)head * D;

    for (uint j = 0; j < t; j++) {
        red[tid] = float(q[qbase + tid]) * float(kc[cbase + (ulong)j * dmodel + tid]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = D >> 1; s > 0; s >>= 1) {
            if (tid < s) {
                red[tid] += red[tid + s];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) {
            scores[j] = red[0] * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        float m = -1.0e30f;
        for (uint j = 0; j < t; j++) {
            m = fmax(m, scores[j]);
        }
        float sum = 0.0f;
        for (uint j = 0; j < t; j++) {
            scores[j] = exp(scores[j] - m);
            sum += scores[j];
        }
        for (uint j = 0; j < t; j++) {
            scores[j] /= sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc = 0.0f;
    for (uint j = 0; j < t; j++) {
        acc += scores[j] * float(vc[cbase + (ulong)j * dmodel + tid]);
    }
    out[qbase + tid] = bfloat(acc);
}
