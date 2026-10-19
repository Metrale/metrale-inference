// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: The KV-cache codec registry. The `kv_codec_registry!` invocation at the bottom
//! of this file **is** the registry: it generates [`KvCodecId`], the [`SPEC`] table,
//! [`KvCodecId::ALL`], [`KvCodecId::name`] and the [`PAIR_ALIASES`] table. Adding a codec means
//! adding one row to the invocation and one spec constant in `rows.rs`.
//!
//! Owner: kv-codec.
//! Invariants:
//! - `spec(id).id == id` and `spec(id).name == id.name()` for every id, and discriminants are
//!   unique and ascending (checked at compile time below). Discriminants may leave gaps.
//! - A pair alias names two registered codecs; no alias spells a codec name.

use crate::rows;
use crate::spec::{AutoBoundary, CodecRole, KvCodecSpec};

/// 2026-10-18: Generate the codec id enum, its tables and the pair alias table from one list.
macro_rules! kv_codec_registry {
    (
        codecs {
            $( $(#[$meta:meta])* $variant:ident = $disc:literal : $name:literal => $spec:path ),+ $(,)?
        }
        pairs {
            $( $long:literal | $short:literal => ($k:ident, $v:ident) @ $ab:ident ),* $(,)?
        }
    ) => {
        /// 2026-10-18: A registered KV-cache codec. The discriminant is the wire and stored value
        /// (TIDE); it is never renumbered. Outside this crate every match needs a wildcard arm,
        /// which the `kv-codec-ban` gate counts: device code dispatches on [`crate::AbiId`].
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
        #[repr(u8)]
        #[non_exhaustive]
        pub enum KvCodecId {
            $( $(#[$meta])* $variant = $disc ),+
        }

        /// 2026-10-18: Registered codecs.
        const COUNT: usize = [$( $disc ),+].len();

        /// 2026-10-18: The spec of every codec, in table order (`KvCodecId::ALL`); [`spec`]
        /// finds a codec's row by its discriminant, so wire values may leave gaps.
        pub static SPEC: [KvCodecSpec; COUNT] = ROWS;

        /// 2026-10-18: [`SPEC`] as a constant, for the compile-time derivations below.
        const ROWS: [KvCodecSpec; COUNT] = [ $( $spec ),+ ];

        impl KvCodecId {
            /// 2026-10-18: Every codec, in table order (ascending discriminant).
            pub const ALL: [KvCodecId; COUNT] = [ $( KvCodecId::$variant ),+ ];

            /// 2026-10-18: The canonical spelling (`--kv-cache-dtype` for an attention codec).
            pub const fn name(self) -> &'static str {
                match self { $( KvCodecId::$variant => $name ),+ }
            }

            /// 2026-10-18: The codec with discriminant `raw`, or `None` when none is registered.
            pub const fn from_u8(raw: u8) -> Option<Self> {
                match raw {
                    $( $disc => Some(KvCodecId::$variant), )+
                    _ => None,
                }
            }
        }

        /// 2026-10-18: The fused (K, V) pairs the CLI accepts by name. Each has a long and a
        /// short spelling; both parse to the same `(k, v)`.
        pub static PAIR_ALIASES: &[PairAlias] = &ALIAS_ROWS;

        /// 2026-10-18: [`PAIR_ALIASES`] as a constant, for [`by_pair_name`].
        const ALIAS_ROWS: [PairAlias; [$( $long ),*].len()] = [
            $( PairAlias {
                long: $long,
                short: $short,
                k: KvCodecId::$k,
                v: KvCodecId::$v,
                auto_boundary: AutoBoundary::$ab,
            } ),*
        ];
    };
}

/// 2026-10-18: One accepted spelling of an asymmetric (K, V) pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PairAlias {
    /// 2026-10-18: The canonical spelling, e.g. `turbo4k_turbo3v`.
    pub long: &'static str,
    /// 2026-10-18: The short spelling, e.g. `turbo4k3v`.
    pub short: &'static str,
    /// 2026-10-18: The K-side codec.
    pub k: KvCodecId,
    /// 2026-10-18: The V-side codec.
    pub v: KvCodecId,
    /// 2026-10-18: The boundary layers an auto `--kv-high-precision-layers` gives the pair.
    pub auto_boundary: AutoBoundary,
}

kv_codec_registry! {
    codecs {
        /// 2026-10-18: BF16, 2 bytes per value.
        Bf16 = 0 : "bf16" => rows::BF16,
        /// 2026-10-18: FP8 E4M3, 1 byte per value, one online-calibrated scale per tensor.
        Fp8 = 1 : "fp8" => rows::FP8_E4M3_TENSOR,
        /// 2026-10-18: E2M1 values plus an E4M3 scale per 16: 4.5 bits per value.
        Nvfp4 = 2 : "nvfp4" => rows::NVFP4_E4M3X16,
        /// 2026-10-18: WHT-rotated 16-level codebook in the nvfp4 byte layout.
        Turbo4 = 3 : "turbo4" => rows::TURBO4,
        /// 2026-10-18: WHT-rotated 8-level codebook, 8 values in 3 bytes plus an E4M3 scale per
        /// 16: 3.5 bits per value.
        Turbo3 = 4 : "turbo3" => rows::TURBO3,
        /// 2026-10-18: WHT-rotated 4-level codebook, 4 values per byte plus an E4M3 scale per
        /// 16: 2.5 bits per value.
        Turbo2 = 5 : "turbo2" => rows::TURBO2,
        /// 2026-10-18: WHT-rotated FP8 E4M3 values plus a BF16 norm per 16: 9 bits per value.
        Turbo8 = 6 : "turbo8" => rows::TURBO8,
        /// 2026-10-18: 512-state trellis-coded 3-bit codes per rotated 128-value group: 3.25 bits
        /// per value.
        Turbo3Tcq = 7 : "turbo3_tcq" => rows::TURBO3_TCQ,
        /// 2026-10-18: 256-state trellis-coded 2-bit codes per rotated 128-value group: 2.25 bits
        /// per value.
        Turbo2Tcq = 8 : "turbo2_tcq" => rows::TURBO2_TCQ,
        /// 2026-10-18: 256-state trellis-coded 1-bit codes per rotated 128-value group: 1.25 bits
        /// per value.
        Turbo1Tcq = 9 : "turbo1_tcq" => rows::TURBO1_TCQ,
        /// 2026-10-18: GLM-5.3 DSA indexer rows: WHT-rotated FP8 E4M3, one F32 scale per
        /// 128-value row. An indexer codec, never a `--kv-cache-dtype` side. Wire values 10 and
        /// 11 are reserved for the DeepSeek-V4.1 rows.
        Fp8RowF32 = 12 : "fp8_row" => rows::FP8_ROW_F32SCALE,
        /// 2026-10-18: The turbo4 16-level codebook per rotated 128-value group with one FP16
        /// norm: 4.125 bits per value.
        Turbo4N128 = 13 : "turbo4_n128" => rows::TURBO4_N128,
    }
    pairs {
        "turbo4k_turbo3v" | "turbo4k3v" => (Turbo4, Turbo3) @ OneThird,
        "turbo4k_turbo8v" | "turbo4k8v" => (Turbo4, Turbo8) @ OneThird,
        "turbo3k_turbo8v" | "turbo3k8v" => (Turbo3, Turbo8) @ OneThird,
        "bf16k_turbo4v" | "bf16k4v" => (Bf16, Turbo4) @ OneThird,
        "bf16k_turbo3v" | "bf16k3v" => (Bf16, Turbo3) @ FourFifths,
        "fp8k_turbo4v" | "fp8k4v" => (Fp8, Turbo4) @ OneThird,
        "fp8k_turbo3v" | "fp8k3v" => (Fp8, Turbo3) @ OneThird,
        "bf16k_turbo2v" | "bf16k2v" => (Bf16, Turbo2) @ OneThird,
        "fp8k_turbo2v" | "fp8k2v" => (Fp8, Turbo2) @ OneThird,
    }
}

/// 2026-10-18: The spec of `id`.
pub const fn spec(id: KvCodecId) -> &'static KvCodecSpec {
    &SPEC[SLOT[id as usize] as usize]
}

/// 2026-10-18: The codec whose canonical spelling is `name`, attention or indexer; `None` for
/// any other text. A `const fn`, so a caller can bind a codec at compile time without naming a
/// variant.
pub const fn by_name(name: &str) -> Option<KvCodecId> {
    let mut i = 0;
    while i < COUNT {
        if const_str_eq(ROWS[i].name, name) {
            return Some(ROWS[i].id);
        }
        i += 1;
    }
    None
}

/// 2026-10-18: The (K, V) codecs a `--kv-cache-dtype` spelling names: a pair alias's long or
/// short spelling, or an attention codec's name (both sides). `None` for any other text,
/// including an indexer codec's name. A `const fn`, so a caller can bind a pair at compile time
/// without naming a variant.
pub(crate) const fn by_pair_name(name: &str) -> Option<(KvCodecId, KvCodecId)> {
    let mut i = 0;
    while i < ALIAS_ROWS.len() {
        let a = &ALIAS_ROWS[i];
        if const_str_eq(a.long, name) || const_str_eq(a.short, name) {
            return Some((a.k, a.v));
        }
        i += 1;
    }
    match by_name(name) {
        Some(id) if ROWS[SLOT[id as usize] as usize].role as u8 == CodecRole::Attention as u8 => {
            Some((id, id))
        }
        Some(_) | None => None,
    }
}

/// 2026-10-18: The table row of each discriminant (`u8::MAX` where none is registered).
const SLOT: [u8; 256] = {
    let mut slot = [u8::MAX; 256];
    let mut i = 0;
    while i < COUNT {
        assert!(i < u8::MAX as usize, "more codecs than a u8 slot holds");
        assert!(
            slot[KvCodecId::ALL[i] as usize] == u8::MAX,
            "two codecs share a discriminant"
        );
        #[allow(clippy::cast_possible_truncation)]
        let row = i as u8;
        slot[KvCodecId::ALL[i] as usize] = row;
        i += 1;
    }
    slot
};

/// 2026-10-18: Codecs whose role is `role`.
const fn role_count(role: CodecRole) -> usize {
    let mut n = 0;
    let mut i = 0;
    while i < COUNT {
        if ROWS[i].role as u8 == role as u8 {
            n += 1;
        }
        i += 1;
    }
    n
}

impl KvCodecId {
    /// 2026-10-18: Every attention codec (a `--kv-cache-dtype` side), in table order.
    pub const ATTENTION: [KvCodecId; role_count(CodecRole::Attention)] = {
        let mut out = [KvCodecId::ALL[0]; role_count(CodecRole::Attention)];
        let (mut i, mut n) = (0, 0);
        while i < COUNT {
            if ROWS[i].role as u8 == CodecRole::Attention as u8 {
                out[n] = ROWS[i].id;
                n += 1;
            }
            i += 1;
        }
        out
    };
}

/// 2026-10-18: Compile-time proof that the table is in ascending discriminant order and that
/// each row carries its own id and name.
const _: () = {
    let mut i = 0;
    while i < COUNT {
        let id = KvCodecId::ALL[i];
        assert!(
            i == 0 || (KvCodecId::ALL[i - 1] as u8) < id as u8,
            "KvCodecId::ALL is not in ascending discriminant order"
        );
        assert!(
            ROWS[i].id as u8 == id as u8,
            "SPEC row carries another codec's id"
        );
        assert!(
            const_str_eq(ROWS[i].name, id.name()),
            "SPEC row name differs from the registry name"
        );
        i += 1;
    }
};

/// 2026-10-18: `a == b` in a `const` context.
pub(crate) const fn const_str_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}
