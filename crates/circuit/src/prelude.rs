// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The `alloc` names a `std` crate's prelude would give, for the `no_std` build.
//! Every module that allocates starts with `use crate::prelude::*;`.
//!
//! Owner: metrale-circuit.
//! Invariants: only re-exports of `alloc` items.

pub(crate) use alloc::boxed::Box;
pub(crate) use alloc::format;
pub(crate) use alloc::string::{String, ToString};
pub(crate) use alloc::vec;
pub(crate) use alloc::vec::Vec;
