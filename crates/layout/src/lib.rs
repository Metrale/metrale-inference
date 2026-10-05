// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: Layouts as functions. A layout maps an index `i` in `[0, size)` to an offset: the
//! index is expanded into mixed-radix digits (first digit fastest) and each digit is multiplied by
//! its stride; a swizzle then XORs a field of offset bits into a lower field. Tiling, partitioning
//! a tile among threads and staging through shared memory are compositions of such functions
//! (book/src/appendix/layouts.md).
//!
//! Owner: metrale-layout.
//! Invariants:
//! - Pure: no I/O, no allocation beyond the results, no randomness.
//! - Every operation's result is checked against the function it stands for by evaluation over
//!   its whole domain before it is returned; a closed form that would not equal it is an error,
//!   never a wrong answer.
//! - Only what kernels need: one level of modes over digits, integer strides, XOR swizzles.

mod check;
mod layout;
mod ops;
mod swizzle;

#[cfg(test)]
mod layout_tests;

pub use check::{BankReport, bank_conflicts, vectors_aligned};
pub use layout::{Digit, Layout, LayoutError, Mode};
pub use ops::{coalesce, complement, compose, divide, product};
pub use swizzle::Swizzle;
