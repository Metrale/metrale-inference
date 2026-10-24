// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Serde for fixed-length byte arrays longer than serde's 32-element derive limit:
//! written as one byte string, read back only at exactly `N` bytes.
//!
//! Owner: blueprint (sealed kernels).
//! Invariants: a length other than `N` is a decode error, never a truncation or padding.

use core::fmt;

use serde::de::{Error as _, SeqAccess, Visitor};
use serde::{Deserializer, Serializer};

/// 2026-10-22: Serialize `v` as a byte string.
pub fn serialize<S: Serializer, const N: usize>(v: &[u8; N], s: S) -> Result<S::Ok, S::Error> {
    s.serialize_bytes(v)
}

struct Fixed<const N: usize>;

impl<'de, const N: usize> Visitor<'de> for Fixed<N> {
    type Value = [u8; N];

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{N} bytes")
    }

    fn visit_bytes<E: serde::de::Error>(self, v: &[u8]) -> Result<[u8; N], E> {
        v.try_into().map_err(|_| E::invalid_length(v.len(), &self))
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<[u8; N], A::Error> {
        let mut out = [0u8; N];
        for (i, o) in out.iter_mut().enumerate() {
            *o = seq
                .next_element()?
                .ok_or_else(|| A::Error::invalid_length(i, &self))?;
        }
        if seq.next_element::<u8>()?.is_some() {
            return Err(A::Error::invalid_length(N + 1, &self));
        }
        Ok(out)
    }
}

/// 2026-10-22: Deserialize exactly `N` bytes.
pub fn deserialize<'de, D: Deserializer<'de>, const N: usize>(d: D) -> Result<[u8; N], D::Error> {
    d.deserialize_bytes(Fixed::<N>)
}
