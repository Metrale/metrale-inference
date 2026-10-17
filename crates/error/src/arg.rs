// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: [`ErrArg`], the closed set of values an error form's `{}` placeholders take.
//!
//! Owner: error.
//! Invariants:
//! - No trait objects (NDIC): an argument is one of the listed variants, rendered by `Display`.
//! - Rendering an argument never re-scans its text for placeholders.

use alloc::boxed::Box;
use alloc::string::String;
use core::fmt;

use crate::{Error, ErrorCode};

/// 2026-10-17: One positional argument of an [`Error`], stored unrendered until `Display`.
#[derive(Clone, Debug)]
pub enum ErrArg {
    /// 2026-10-17: A static string: names, operations, call sites. No allocation.
    Str(&'static str),
    /// 2026-10-17: An owned string, for text only known at run time (paths, keys).
    String(String),
    /// 2026-10-17: Unsigned integers: sizes, counts, limits, statuses (BU64: wide enough for any).
    U64(u64),
    /// 2026-10-17: Signed integers, such as a vendor status that can be negative.
    I64(i64),
    /// 2026-10-17: Floats, such as a measured ratio in a refusal.
    F64(f64),
    /// 2026-10-17: A flag.
    Bool(bool),
    /// 2026-10-17: Another code, named without its arguments; renders `Name (0xNNNN)`.
    Code(ErrorCode),
    /// 2026-10-17: A whole error as the cause; renders that error's message.
    Nested(Box<Error>),
}

impl fmt::Display for ErrArg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrArg::Str(s) => f.write_str(s),
            ErrArg::String(s) => f.write_str(s),
            ErrArg::U64(v) => fmt::Display::fmt(v, f),
            ErrArg::I64(v) => fmt::Display::fmt(v, f),
            ErrArg::F64(v) => fmt::Display::fmt(v, f),
            ErrArg::Bool(v) => fmt::Display::fmt(v, f),
            ErrArg::Code(c) => write!(f, "{} ({:#06x})", c.name(), c.as_u16()),
            ErrArg::Nested(e) => fmt::Display::fmt(e, f),
        }
    }
}

impl From<&'static str> for ErrArg {
    fn from(v: &'static str) -> Self {
        ErrArg::Str(v)
    }
}

impl From<String> for ErrArg {
    fn from(v: String) -> Self {
        ErrArg::String(v)
    }
}

impl From<ErrorCode> for ErrArg {
    fn from(v: ErrorCode) -> Self {
        ErrArg::Code(v)
    }
}

impl From<Error> for ErrArg {
    fn from(v: Error) -> Self {
        ErrArg::Nested(Box::new(v))
    }
}

/// 2026-10-17: `From` for primitives whose conversion into the variant's type is lossless.
macro_rules! lossless_from {
    ($variant:ident($wide:ty): $($t:ty),*) => {
        $(
            impl From<$t> for ErrArg {
                fn from(v: $t) -> Self {
                    ErrArg::$variant(<$wide>::from(v))
                }
            }
        )*
    };
}

lossless_from!(U64(u64): u8, u16, u32, u64);
lossless_from!(I64(i64): i8, i16, i32, i64);
lossless_from!(F64(f64): f32, f64);
lossless_from!(Bool(bool): bool);

/// 2026-10-17: `usize` has no `From` into `u64`; every Rust target's pointer width is at most
/// 64 bits, which the `try_from` turns into a compile-checked fact via the const below.
impl From<usize> for ErrArg {
    fn from(v: usize) -> Self {
        const _: () = assert!(usize::BITS <= u64::BITS);
        ErrArg::U64(u64::try_from(v).unwrap_or(u64::MAX))
    }
}

/// 2026-10-17: As for `usize`: `isize` is at most 64 bits wide on every target.
impl From<isize> for ErrArg {
    fn from(v: isize) -> Self {
        const _: () = assert!(isize::BITS <= i64::BITS);
        ErrArg::I64(i64::try_from(v).unwrap_or(i64::MAX))
    }
}
