// SPDX-License-Identifier: MIT OR Apache-2.0
// The one ABI dispatch; kv-codec-ban must not scan it.
pub fn with_abi(a: AbiId) -> &'static str {
    match a {
        AbiId::Dense => "paged_decode_attn_bf16",
        AbiId::DenseScaled => "paged_decode_attn_fp8",
    }
}
