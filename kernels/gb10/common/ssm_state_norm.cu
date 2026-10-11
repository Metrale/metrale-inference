// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: A pass over the SSM h state of every SSM layer in one launch.
// 2026-10-10: Only the Mamba-2 non-finite count remains; the per-head Frobenius-norm clamp it
// replaced was also applied to GDN states after every prefill chunk, and is gone: no SSM state
// is rescaled (the references bound nothing).
//
// Owner: gb10 kernels.
// Invariants: none beyond the types.
//
// h_state_ptrs holds one device pointer per layer; each layer's state is
// [num_heads, k_dim, v_dim] with v contiguous. Grid (num_heads, num_layers), block v_dim,
// thread tid owning column tid (mamba2_state_finite_guard in model-engine
// trait_impl/mamba2_state_guard.rs, dims from ssm_state_norm_dims).

// 2026-09-29: Count the non-finite values of a Mamba-2 h state (the reference bounds nothing,
// and prefill states reach per-head norms of 11275). *count must be zeroed before the launch.
// Each thread adds its column's count once.
extern "C" __global__ void ssm_state_nonfinite_count(
    const float* const* __restrict__ h_state_ptrs,
    unsigned int num_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int* __restrict__ count
) {
    const unsigned int head = blockIdx.x;
    const unsigned int layer = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    if (head >= num_heads || tid >= v_dim) return;
    const float* H = h_state_ptrs[layer] + (unsigned long long)head * k_dim * v_dim;
    unsigned int bad = 0;
    for (unsigned int j = 0; j < k_dim; j++) {
        if (!isfinite(H[j * v_dim + tid])) bad++;
    }
    if (bad) atomicAdd(count, bad);
}
