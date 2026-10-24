// SPDX-License-Identifier: MIT OR Apache-2.0
// The registry crate names its own codecs; kv-codec-ban must not scan it.
pub fn name(id: KvCodecId) -> &'static str {
    match id {
        KvCodecId::Bf16 => "bf16",
        KvCodecId::Fp8 => "fp8",
    }
}
