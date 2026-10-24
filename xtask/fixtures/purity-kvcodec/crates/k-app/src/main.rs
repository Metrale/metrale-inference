// SPDX-License-Identifier: MIT OR Apache-2.0
fn main() {
    let d: KvCacheDtype = parse();
    let id = KvCodecId::Turbo4;
    let name = match id {
        KvCodecId::Fp8 => "paged_decode_attn_fp8",
        _ => "wht_bf16_fwd",
    };
    drop((d, name));
}
