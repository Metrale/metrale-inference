// SPDX-License-Identifier: MIT OR Apache-2.0
pub fn d(x: KvCacheDtype) -> &'static str { match x { KvCodecId::Fp8 => "paged_decode_fp8" } }
