// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: [`Error`]: one pointer-sized value holding a code, the call-site origin and the
//! form's arguments, rendered only when displayed.
//!
//! Owner: error.
//! Invariants:
//! - `size_of::<Error>() == size_of::<usize>()`, and `Result<(), Error>` is the same size.
//! - Equality and hashing see the code only; origin and arguments never decide identity.
//! - Rendering scans only the form; an argument's own braces are written verbatim.

use alloc::boxed::Box;
use core::fmt;
use core::hash::{Hash, Hasher};

use crate::{ErrArg, ErrorCode};

/// 2026-10-17: The heap part of an [`Error`].
#[derive(Clone)]
struct Inner {
    code: ErrorCode,
    origin: &'static str,
    args: Option<Box<[ErrArg]>>,
}

/// 2026-10-17: The engine's error. Build it with [`crate::error!`], which checks the argument
/// count against the code's form at compile time and records `file:line` as the origin.
#[derive(Clone)]
pub struct Error(Box<Inner>);

impl Error {
    /// 2026-10-17: The constructor `error!` expands to. Calling it directly skips the
    /// compile-time argument-count check, so it is hidden; a mismatch renders the missing
    /// placeholders as `{}` and ignores surplus arguments.
    #[doc(hidden)]
    pub fn __from_parts(
        code: ErrorCode,
        origin: &'static str,
        args: Option<Box<[ErrArg]>>,
    ) -> Self {
        Self(Box::new(Inner { code, origin, args }))
    }

    /// 2026-10-17: The code: the error's identity.
    pub fn code(&self) -> ErrorCode {
        self.0.code
    }

    /// 2026-10-17: The `file:line` of the `error!` call that built this error.
    pub fn origin(&self) -> &'static str {
        self.0.origin
    }

    /// 2026-10-17: The form's arguments, in placeholder order (empty for a form without `{}`).
    pub fn args(&self) -> &[ErrArg] {
        self.0.args.as_deref().unwrap_or(&[])
    }
}

/// 2026-10-17: Write `form` with its `{}` placeholders filled from `args` in order, and `{{`/`}}`
/// as single braces. The form passed the registry's const check, so no other brace occurs.
fn render(form: &str, args: &[ErrArg], f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let bytes = form.as_bytes();
    let mut next_arg = args.iter();
    let mut run = 0;
    let mut i = 0;
    while i + 1 < bytes.len() {
        let pair = (bytes[i], bytes[i + 1]);
        if !matches!(pair, (b'{', b'{') | (b'}', b'}') | (b'{', b'}')) {
            i += 1;
            continue;
        }
        f.write_str(&form[run..i])?;
        match pair {
            (b'{', b'}') => match next_arg.next() {
                Some(arg) => fmt::Display::fmt(arg, f)?,
                None => f.write_str("{}")?,
            },
            (b'{', _) => f.write_str("{")?,
            _ => f.write_str("}")?,
        }
        i += 2;
        run = i;
    }
    f.write_str(&form[run..])
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        render(self.0.code.form(), self.args(), f)
    }
}

impl fmt::Debug for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Error {{ code: {:#06x} {}, origin: {}, message: \"{}\" }}",
            self.0.code.as_u16(),
            self.0.code.name(),
            self.0.origin,
            self
        )
    }
}

impl core::error::Error for Error {}

impl PartialEq for Error {
    fn eq(&self, other: &Self) -> bool {
        self.0.code == other.0.code
    }
}

impl Eq for Error {}

impl Hash for Error {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.code.hash(state);
    }
}
