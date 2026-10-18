// SPDX-License-Identifier: MIT OR Apache-2.0
pub fn abi(a: AbiId) -> u8 {
    match a {
        AbiId::Dense => "wht_bf16_inv".len() as u8,
        _ => 1,
    }
}
