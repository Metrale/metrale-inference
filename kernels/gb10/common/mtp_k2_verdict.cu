// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-11: The K=2 MTP verdict on the device (METRALE_GLM_MTP_DEVICE_STEP). A K=2 verify
// judges one draft (row 1's token) against the target's argmax of row 0; the verdict the host
// computes for a greedy request without logit processing is a pure function of three ids, so
// every rank can take it from its own copy of them, without the host. The host specification is
// metrale_model_layers::speculative::k2_device_verdict (k2_wire.rs); its tests pin the words.
//
// Owner: gb10 kernels.
// Invariants:
// - out[0] accepted (draft == argmax[0]), out[1] the next token (argmax[1] after an accept,
//   argmax[0] after a reject), out[2] the draft, out[3] argmax[0], out[4] argmax[1].
// - One thread writes every word; launched as grid (1,1,1), block (32,1,1).

extern "C" __global__ void mtp_k2_verdict(
    const unsigned int* __restrict__ argmax,
    const unsigned int* __restrict__ token_ids,
    unsigned int* __restrict__ out
) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    const unsigned int a0 = argmax[0];
    const unsigned int a1 = argmax[1];
    const unsigned int draft = token_ids[1];
    const unsigned int accepted = draft == a0 ? 1u : 0u;
    out[0] = accepted;
    out[1] = accepted ? a1 : a0;
    out[2] = draft;
    out[3] = a0;
    out[4] = a1;
}
