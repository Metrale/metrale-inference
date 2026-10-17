// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The `error_registry!` macro that generates [`crate::ErrorCode`] and
//! [`crate::CodeRange`], and the const form parser its compile-time checks use.
//!
//! Owner: error.
//! Invariants:
//! - A form may contain only `{}` (a placeholder), `{{` and `}}` (literal braces); any other
//!   brace fails the build through a `const _: () = assert!(..)` per variant.
//! - Every code lies inside the range block it is declared in, and range blocks are disjoint
//!   and ascending; both are checked at compile time.

/// 2026-10-17: The number of `{}` placeholders in `form`, or `None` when the form holds any
/// brace other than `{}`, `{{` or `}}`.
pub const fn form_placeholders(form: &str) -> Option<usize> {
    let b = form.as_bytes();
    let mut i = 0;
    let mut n = 0;
    while i < b.len() {
        let next = if i + 1 < b.len() {
            Some(b[i + 1])
        } else {
            None
        };
        match (b[i], next) {
            (b'{', Some(b'{')) | (b'}', Some(b'}')) => i += 2,
            (b'{', Some(b'}')) => {
                n += 1;
                i += 2;
            }
            (b'{', _) | (b'}', _) => return None,
            _ => i += 1,
        }
    }
    Some(n)
}

/// 2026-10-17: Whether `ranges` (as `(lo, hi)` bounds, in declaration order) are each
/// non-empty, ascending and disjoint.
pub const fn ranges_ascending_disjoint(ranges: &[(u16, u16)]) -> bool {
    let mut i = 0;
    while i < ranges.len() {
        if ranges[i].0 > ranges[i].1 {
            return false;
        }
        if i > 0 && ranges[i - 1].1 >= ranges[i].0 {
            return false;
        }
        i += 1;
    }
    true
}

/// 2026-10-17: Declares the registry: range blocks, each holding `Name = 0xNNNN => "form",`
/// entries. Generates `CodeRange`, the fieldless `#[repr(u16)]` `ErrorCode`, its const
/// accessors, and the compile-time form and range checks. Crate-private: there is one registry.
macro_rules! error_registry {
    (
        $(
            $(#[$rmeta:meta])*
            $range:ident = $lo:literal ..= $hi:literal {
                $(
                    $(#[$meta:meta])*
                    $name:ident = $code:literal => $form:literal,
                )*
            }
        )*
    ) => {
        /// 2026-10-17: A reserved code range (one per crate or domain).
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum CodeRange {
            $( $(#[$rmeta])* $range, )*
        }

        impl CodeRange {
            /// 2026-10-17: Every range, in ascending order.
            pub const ALL: &'static [CodeRange] = &[$(CodeRange::$range),*];

            /// 2026-10-17: The inclusive `(first, last)` codes of the range.
            pub const fn bounds(self) -> (u16, u16) {
                match self { $( CodeRange::$range => ($lo, $hi), )* }
            }

            /// 2026-10-17: Whether `code` lies inside the range.
            pub const fn contains(self, code: u16) -> bool {
                let (lo, hi) = self.bounds();
                lo <= code && code <= hi
            }
        }

        /// 2026-10-17: The stable error code. The discriminant is the code: append-only, never
        /// renumbered (`codes.lock`). Each variant's message is its [`ErrorCode::form`].
        #[repr(u16)]
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub enum ErrorCode {
            $( $( $(#[$meta])* $name = $code, )* )*
        }

        impl ErrorCode {
            /// 2026-10-17: Every code, in declaration order.
            pub const ALL: &'static [ErrorCode] = &[$( $( ErrorCode::$name, )* )*];

            /// 2026-10-17: The numeric code (the `#[repr(u16)]` discriminant).
            pub const fn as_u16(self) -> u16 {
                self as u16
            }

            /// 2026-10-17: The variant name, as written in `codes.lock`.
            pub const fn name(self) -> &'static str {
                match self { $( $( ErrorCode::$name => stringify!($name), )* )* }
            }

            /// 2026-10-17: The form template: `{}` is a placeholder, `{{`/`}}` literal braces.
            pub const fn form(self) -> &'static str {
                match self { $( $( ErrorCode::$name => $form, )* )* }
            }

            /// 2026-10-17: The range block the code is declared in.
            pub const fn range(self) -> CodeRange {
                match self { $( $( ErrorCode::$name => CodeRange::$range, )* )* }
            }

            /// 2026-10-17: The number of `{}` placeholders in [`ErrorCode::form`], which is the
            /// number of arguments `error!` must be given.
            pub const fn placeholder_count(self) -> usize {
                match $crate::registry::form_placeholders(self.form()) {
                    Some(n) => n,
                    // 2026-10-17: Unreachable: every form passed the per-variant const check.
                    None => panic!("invalid form in the error registry"),
                }
            }

            /// 2026-10-17: The code with numeric value `code`, or `None` if none is registered.
            pub const fn from_u16(code: u16) -> Option<ErrorCode> {
                match code {
                    $( $( $code => Some(ErrorCode::$name), )* )*
                    _ => None,
                }
            }
        }

        const _: () = ::core::assert!(
            $crate::registry::ranges_ascending_disjoint(&[$( ($lo, $hi) ),*]),
            "error_registry: range blocks must be non-empty, ascending and disjoint"
        );
        $( $(
            const _: () = ::core::assert!(
                $crate::registry::form_placeholders($form).is_some(),
                concat!("error_registry: the form of ", stringify!($name),
                    " may use only placeholders and doubled braces")
            );
            const _: () = ::core::assert!(
                $lo <= $code && $code <= $hi,
                concat!("error_registry: ", stringify!($name), " lies outside range ",
                    stringify!($range))
            );
        )* )*
    };
}

pub(crate) use error_registry;

#[cfg(test)]
mod tests {
    use super::{form_placeholders, ranges_ascending_disjoint};

    #[test]
    fn form_parser_counts_placeholders_and_escapes() {
        assert_eq!(form_placeholders(""), Some(0));
        assert_eq!(form_placeholders("plain"), Some(0));
        assert_eq!(form_placeholders("{}"), Some(1));
        assert_eq!(form_placeholders("{} on {}"), Some(2));
        assert_eq!(form_placeholders("{{}}"), Some(0));
        assert_eq!(form_placeholders("{{{}}}"), Some(1));
        assert_eq!(form_placeholders("a {{literal}} {}"), Some(1));
    }

    #[test]
    fn form_parser_rejects_every_other_brace() {
        for bad in [
            "{", "}", "{x}", "{0}", "{:?}", "a { b", "a } b", "{}}", "{{}", "}{",
        ] {
            assert_eq!(form_placeholders(bad), None, "{bad:?} must be rejected");
        }
    }

    #[test]
    fn range_check_rejects_overlap_order_and_empty() {
        assert!(ranges_ascending_disjoint(&[(0, 0xFF), (0x100, 0x1FF)]));
        assert!(ranges_ascending_disjoint(&[]));
        assert!(!ranges_ascending_disjoint(&[(0, 0x100), (0x100, 0x1FF)]));
        assert!(!ranges_ascending_disjoint(&[(0x100, 0x1FF), (0, 0xFF)]));
        assert!(!ranges_ascending_disjoint(&[(0x10, 0x0F)]));
    }
}
