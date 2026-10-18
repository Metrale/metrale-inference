// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: The KV-cache codec registry. The `kv_codec_registry!` invocation at the bottom
//! of this file **is** the registry: it generates [`KvCodecId`], the [`SPEC`] table,
//! [`KvCodecId::ALL`], [`KvCodecId::name`] and the [`PAIR_ALIASES`] table. Adding a codec means
//! adding one row to the invocation and one spec constant in `rows.rs`.
//!
//! Owner: kv-codec.
//! Invariants:
//! - `SPEC[id as usize].id == id` and `SPEC[id as usize].name == id.name()` for every id
//!   (checked at compile time below).
//! - A pair alias names two registered codecs; no alias spells a codec name.

use crate::rows;
use crate::spec::KvCodecSpec;

/// 2026-10-18: Generate the codec id enum, its tables and the pair alias table from one list.
macro_rules! kv_codec_registry {
    (
        codecs {
            $( $(#[$meta:meta])* $variant:ident = $disc:literal : $name:literal => $spec:path ),+ $(,)?
        }
        pairs {
            $( $long:literal | $short:literal => ($k:ident, $v:ident) ),* $(,)?
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

        /// 2026-10-18: The spec of every codec, indexed by discriminant.
        pub static SPEC: [KvCodecSpec; COUNT] = [ $( $spec ),+ ];

        impl KvCodecId {
            /// 2026-10-18: Every codec, in discriminant order.
            pub const ALL: [KvCodecId; COUNT] = [ $( KvCodecId::$variant ),+ ];

            /// 2026-10-18: The canonical `--kv-cache-dtype` spelling.
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
        pub static PAIR_ALIASES: &[PairAlias] = &[
            $( PairAlias { long: $long, short: $short, k: KvCodecId::$k, v: KvCodecId::$v } ),*
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
    }
    pairs {
        "turbo4k_turbo3v" | "turbo4k3v" => (Turbo4, Turbo3),
        "turbo4k_turbo8v" | "turbo4k8v" => (Turbo4, Turbo8),
        "turbo3k_turbo8v" | "turbo3k8v" => (Turbo3, Turbo8),
        "bf16k_turbo4v" | "bf16k4v" => (Bf16, Turbo4),
        "bf16k_turbo3v" | "bf16k3v" => (Bf16, Turbo3),
        "fp8k_turbo4v" | "fp8k4v" => (Fp8, Turbo4),
        "fp8k_turbo3v" | "fp8k3v" => (Fp8, Turbo3),
        "bf16k_turbo2v" | "bf16k2v" => (Bf16, Turbo2),
        "fp8k_turbo2v" | "fp8k2v" => (Fp8, Turbo2),
    }
}

/// 2026-10-18: The spec of `id`.
pub const fn spec(id: KvCodecId) -> &'static KvCodecSpec {
    &SPEC[id as usize]
}

/// 2026-10-18: Compile-time proof that the table is in discriminant order and that each row
/// carries its own id and name.
const _: () = {
    let mut i = 0;
    while i < COUNT {
        let id = KvCodecId::ALL[i];
        assert!(
            id as usize == i,
            "KvCodecId::ALL is not in discriminant order"
        );
        assert!(
            SPEC[i].id as usize == i,
            "SPEC row carries another codec's id"
        );
        assert!(
            const_str_eq(SPEC[i].name, id.name()),
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
