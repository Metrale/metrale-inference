// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: Fixed-width identifiers of the platform contracts. Each is raw bytes in
//! memory and lower-case hex in JSON, so a trace id can never be passed where a commit is
//! expected and two spellings of one id compare equal.
//!
//! Owner: qci-dump (contract binding).
//! Invariants: `to_hex` is lower case and exactly `2 * LEN` characters; `from_hex` accepts
//! exactly that many hex digits in either case and nothing else.

use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// 2026-10-11: Why a hex id did not parse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdError {
    NotHex,
    WrongLength,
}

impl fmt::Display for IdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotHex => f.write_str("not hexadecimal"),
            Self::WrongLength => f.write_str("wrong number of hex digits"),
        }
    }
}

const HEX: &[u8; 16] = b"0123456789abcdef";

fn nibble(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

/// 2026-10-11: Decode exactly `N` bytes of hex, either case.
pub(crate) fn decode<const N: usize>(text: &str) -> Result<[u8; N], IdError> {
    let raw = text.as_bytes();
    if raw.iter().any(|c| nibble(*c).is_none()) {
        return Err(IdError::NotHex);
    }
    if raw.len() != 2 * N {
        return Err(IdError::WrongLength);
    }
    let mut out = [0u8; N];
    for (i, pair) in raw.chunks_exact(2).enumerate() {
        out[i] = (nibble(pair[0]).ok_or(IdError::NotHex)? << 4)
            | nibble(pair[1]).ok_or(IdError::NotHex)?;
    }
    Ok(out)
}

/// 2026-10-11: Lower-case hex of `bytes`.
pub(crate) fn encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(HEX[usize::from(b >> 4)] as char);
        out.push(HEX[usize::from(b & 0x0f)] as char);
    }
    out
}

macro_rules! hex_id {
    ($(#[$doc:meta])* $name:ident, $len:expr) => {
        $(#[$doc])*
        #[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(pub [u8; $len]);

        impl $name {
            /// 2026-10-11: Width in bytes.
            pub const LEN: usize = $len;

            /// 2026-10-11: Exactly `2 * LEN` hex digits, either case.
            pub fn from_hex(text: &str) -> Result<Self, IdError> {
                decode::<$len>(text).map(Self)
            }

            /// 2026-10-11: Lower-case hex.
            pub fn to_hex(&self) -> String {
                encode(&self.0)
            }

            /// 2026-10-11: True when every byte is zero, which no contract id may be.
            pub fn is_zero(&self) -> bool {
                self.0.iter().all(|b| *b == 0)
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}({})", stringify!($name), self.to_hex())
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.to_hex())
            }
        }

        impl Serialize for $name {
            fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                s.serialize_str(&self.to_hex())
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                let text = String::deserialize(d)?;
                Self::from_hex(&text).map_err(|e| {
                    serde::de::Error::custom(format!("{}: {e}", stringify!($name)))
                })
            }
        }
    };
}

hex_id!(
    /// 2026-10-11: The id of one engine request, sent by the agent as `x-request-id`.
    RequestId,
    16
);
hex_id!(
    /// 2026-10-11: The W3C trace-id of the request, minted at the front door.
    TraceId,
    16
);
hex_id!(
    /// 2026-10-11: A W3C span (parent) id.
    SpanId,
    8
);
hex_id!(
    /// 2026-10-11: A full git commit: a runtime build or an immutable model revision.
    GitCommit,
    20
);
hex_id!(
    /// 2026-10-11: A SHA-256 digest of file bytes or of a document's canonical JSON.
    Sha256Digest,
    32
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_round_trips_lower_case_and_accepts_upper_case() {
        let id = TraceId::from_hex("4BF92F3577B34DA6A3CE929D0E0E4736").unwrap();
        assert_eq!(id.to_hex(), "4bf92f3577b34da6a3ce929d0e0e4736");
        assert_eq!(id, TraceId::from_hex(&id.to_hex()).unwrap());
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, "\"4bf92f3577b34da6a3ce929d0e0e4736\"");
        assert_eq!(serde_json::from_str::<TraceId>(&json).unwrap(), id);
    }

    #[test]
    fn wrong_width_and_non_hex_are_refused() {
        assert_eq!(
            SpanId::from_hex("00f067aa0ba902b"),
            Err(IdError::WrongLength)
        );
        assert_eq!(
            SpanId::from_hex("00f067aa0ba902b7a"),
            Err(IdError::WrongLength)
        );
        assert_eq!(SpanId::from_hex("00f067aa0ba902bg"), Err(IdError::NotHex));
        assert_eq!(GitCommit::from_hex("main"), Err(IdError::NotHex));
        assert!(serde_json::from_str::<RequestId>("\"req_abc\"").is_err());
    }

    #[test]
    fn zero_is_detected() {
        assert!(SpanId([0; 8]).is_zero());
        assert!(!SpanId([0, 0, 0, 0, 0, 0, 0, 1]).is_zero());
    }
}
