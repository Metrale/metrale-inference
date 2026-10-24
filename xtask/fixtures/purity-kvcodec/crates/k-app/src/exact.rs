// SPDX-License-Identifier: MIT OR Apache-2.0
pub fn kernels(d: &str) -> [String; 3] {
    [
        format!("reshape_and_cache_{d}"),
        "attn_prefill_paged_64".to_owned(),
        "paged_decode_attn_splitk_reduce".to_owned(),
    ]
}
