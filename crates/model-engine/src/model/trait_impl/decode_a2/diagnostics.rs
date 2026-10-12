// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-06: Decode diagnostics interpret the hidden arena's BF16 bytes without
//! joining adjacent values into an unrelated FP32 bit pattern.

/// 2026-10-06: Widen eight little-endian BF16 values exactly, including special values.
pub(super) fn bf16_hidden_preview(bytes: &[u8; 16]) -> [f32; 8] {
    std::array::from_fn(|i| {
        f32::from_bits(u32::from(u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]])) << 16)
    })
}

#[cfg(test)]
mod tests {
    use super::bf16_hidden_preview;

    #[test]
    fn decodes_independent_bf16_values_in_little_endian_order() {
        let bytes = [
            0x80, 0x3f, 0x00, 0xc0, 0x00, 0x3f, 0x00, 0x00, 0x00, 0x80, 0x80, 0x40, 0x80, 0xbf,
            0x80, 0x3e,
        ];
        let got = bf16_hidden_preview(&bytes);
        let expected = [1.0f32, -2.0, 0.5, 0.0, -0.0, 4.0, -1.0, 0.25];
        assert_eq!(got.map(f32::to_bits), expected.map(f32::to_bits));
        // 2026-10-06: Detection control for the former paired-BF16-as-FP32 reader.
        let paired = f32::from_le_bytes(bytes[..4].try_into().unwrap());
        assert_ne!(paired.to_bits(), expected[0].to_bits());
    }

    #[test]
    fn preserves_non_finite_values_and_smallest_bf16_subnormal() {
        let bytes = [
            0x80, 0x7f, 0x80, 0xff, 0xc1, 0x7f, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00,
        ];
        let got = bf16_hidden_preview(&bytes);
        assert_eq!(got[0], f32::INFINITY);
        assert_eq!(got[1], f32::NEG_INFINITY);
        assert!(got[2].is_nan());
        assert_eq!(got[2].to_bits(), 0x7fc1_0000);
        assert_eq!(got[3].to_bits(), 0x0001_0000);
    }
}
