// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The `alloc` names a `std` crate's prelude would give, for the `no_std` build.
//!
//! Owner: metrale-latlas-pack.
//! Invariants: only re-exports of `alloc` items.

pub(crate) use alloc::collections::{BTreeMap, BTreeSet};
pub(crate) use alloc::format;
pub(crate) use alloc::string::{String, ToString};
pub(crate) use alloc::vec;
pub(crate) use alloc::vec::Vec;
