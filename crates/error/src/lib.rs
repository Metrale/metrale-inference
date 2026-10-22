// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The CERF error registry: one fieldless `#[repr(u16)]` [`ErrorCode`] enum whose
//! discriminant is the stable code, one form template per code, and one pointer-sized
//! [`Error`] (code + call-site origin + arguments) rendered lazily on `Display`.
//!
//! Build errors only with [`error!`]: `error!(ErrorCode::NotImplemented, "memcpy_2d_async",
//! "SCALE")`. The argument count is checked against the form at compile time. Text is never
//! identity: equality, hashing and matching use the code alone.
//!
//! # Code ranges (owner-confirmed, append-only)
//!
//! | range | owner | sub-blocks |
//! |---|---|---|
//! | `0x0000–0x00FF` | general (any crate) | `Generic`, `Io`, `InvalidArgument`, `NotImplemented`, `Unavailable` |
//! | `0x0100–0x01FF` | abstract-io ports and hosts | time `0x0100` (reserved), entropy `0x0110`, env `0x0118`, blob `0x0120–0x013F`, proc `0x0140`, http `0x0150`, transport `0x0170`, host construction `0x0180` |
//! | `0x0200–0x02FF` | sched-core (`Device`, scheduler, lora) and device backends | sched-core `0x0200–0x027F`, device backends `0x0280–0x02FF` (`DeviceDriver`) |
//! | `0x0300–0x03FF` | serve-identity and host-probe (bench support) | serve env `0x0300–0x030F` |
//! | `0x0400–0x04FF` | device settings projections (`metrale-settings` `groups::device*`) | out of range `0x0400` |
//! | `0x0500–0x05FF` | KV-cache codec registry and plan (`metrale-kv-codec`) | unknown codec `0x0500`, pair `0x0501`, transition `0x0502`, capability `0x0503`, spill `0x0504`, plan `0x0505`, ladder `0x0506`, ABI `0x0507`, indexer geometry `0x0508`, host dequant `0x0509`, codec feature `0x050a` |
//! | `0x0600–0x06FF` | blueprints (`metrale-blueprint`, `metrale-blueprint-trust`) | framing `0x0600–0x0603`, trust and signatures `0x0604–0x0609`, decode `0x060a`, kernel resolution `0x060b–0x060d`, recipe projection `0x060e`, keys `0x060f`, codec `0x0610` |
//!
//! A code is never renumbered or removed; `codes.lock` and its test enforce that.
//!
//! Owner: error.
//! Invariants:
//! - `no_std` + `alloc`, no `unsafe`, no dependencies, no features; builds for
//!   `wasm32-unknown-unknown` and `wasm32v1-none` (NSCC).
//! - No trait objects (NDIC).

#![no_std]
#![forbid(unsafe_code)]
#![deny(
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

extern crate alloc;

mod arg;
mod code;
mod error;
#[doc(hidden)]
pub mod registry;

pub use arg::ErrArg;
pub use code::{CodeRange, ErrorCode};
pub use error::Error;

/// 2026-10-17: The engine's result type.
pub type Result<T> = core::result::Result<T, Error>;

/// 2026-10-17: Paths `error!` expands to in the caller's crate.
#[doc(hidden)]
pub mod __private {
    pub use alloc::boxed::Box;
}

/// 2026-10-17: Build an [`Error`] from a registry code and its form's arguments, recording the
/// call site's `file:line` as the origin. The code must be a constant path (`ErrorCode::X`).
///
/// ```
/// use metrale_error::{error, ErrorCode};
/// let e = error!(ErrorCode::NotImplemented, "memcpy_2d_async", "SCALE");
/// assert_eq!(e.to_string(), "memcpy_2d_async is not yet implemented on SCALE");
/// ```
///
/// The same call with one argument too few does not compile:
///
/// ```compile_fail
/// use metrale_error::{error, ErrorCode};
/// let e = error!(ErrorCode::NotImplemented, "memcpy_2d_async");
/// ```
///
/// Nor with one too many:
///
/// ```compile_fail
/// use metrale_error::{error, ErrorCode};
/// let e = error!(ErrorCode::NotImplemented, "memcpy_2d_async", "SCALE", 3u32);
/// ```
#[macro_export]
macro_rules! error {
    ($code:expr $(, $arg:expr)* $(,)?) => {{
        const __METRALE_ERROR_ARGS: usize = <[()]>::len(&[$($crate::__unit!($arg)),*]);
        const _: () = ::core::assert!(
            $code.placeholder_count() == __METRALE_ERROR_ARGS,
            "error!: the argument count differs from the number of placeholders in the form"
        );
        $crate::Error::__from_parts(
            $code,
            ::core::concat!(::core::file!(), ":", ::core::line!()),
            $crate::__args!($($arg),*),
        )
    }};
}

/// 2026-10-17: Maps one argument to `()` so `error!` can count arguments in a const.
#[doc(hidden)]
#[macro_export]
macro_rules! __unit {
    ($arg:expr) => {
        ()
    };
}

/// 2026-10-17: The argument slice `error!` stores: `None` for no arguments.
#[doc(hidden)]
#[macro_export]
macro_rules! __args {
    () => {
        ::core::option::Option::None
    };
    ($($arg:expr),+) => {
        ::core::option::Option::Some(
            $crate::__private::Box::new([$($crate::ErrArg::from($arg)),+])
                as $crate::__private::Box<[$crate::ErrArg]>,
        )
    };
}
