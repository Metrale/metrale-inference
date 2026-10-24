// SPDX-License-Identifier: MIT OR Apache-2.0
// KvCacheDtype::Fp8 and "paged_decode_fp8" in a comment are prose.
/* match x { KvCodecId::Bf16 => "reshape_and_cache" } */
pub const MESSAGE: &str = "paged_decode failed for this request";
pub fn all() -> usize {
    KvCodecId::ALL.len()
}
pub fn bytes(id: KvCodecId, g: &Geometry) -> u64 {
    match spec(id).abi {
        Abi::Wide => 0,
        _ => spec(id).block_bytes(g),
    }
}
