// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The location of the `kernels/` tree, for build scripts only: depend on this crate
//! and read `DEP_METRALE_KERNEL_TREE_ROOT`, the directory holding `kernels/` (the repository root
//! in a checkout, this crate's own directory once published). The library is empty by design.
//!
//! Owner: kernel tree (location).
//! Invariants:
//! - The build script reads nothing under `kernels/`, so an edit there never rebuilds this crate
//!   or, through it, any dependent.

#![no_std]
#![forbid(unsafe_code)]
#![deny(
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
