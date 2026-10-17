// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `lever_group!`, the one place a lever is declared, and its helpers.
//!
//! A row `field: Kind = default, flag, env, reader, doc;` becomes one `LeverSpec` (path
//! `group.field`), one `Rule`, and, in the `struct` form, one typed field resolved by
//! `resolve::pick`. The `specs` form (build, dev and tool levers, which no `met` process
//! resolves) emits only the rows.
//!
//! Owner: settings.
//! Invariants: the field type follows from `(Kind, default)` alone; a pair with no type is a
//! compile error, never a silent fallback.

macro_rules! lever_group {
    (
        $(#[$meta:meta])*
        pub struct $Name:ident { group: $group:literal, class: $class:ident }
        $( $field:ident : $kind:ident = $default:tt , $flag:tt , $env:literal , $reader:literal , $doc:literal ; )*
    ) => {
        $(#[$meta])*
        #[derive(Clone, Debug, PartialEq)]
        pub struct $Name {
            $( pub $field: $crate::__lever_ty!($kind $default), )*
        }

        impl $Name {
            $crate::__lever_rows! { group: $group, class: $class,
                $( $field : $kind = $default , $flag , $env , $reader , $doc ; )* }

            /// 2026-10-17: Every field of the group, CLI > recipe > env > declared default.
            ///
            /// # Errors
            /// The first lever whose value or declared default does not parse.
            pub fn resolve<E: $crate::Env>(
                src: &$crate::sources::Sources<'_, E>,
                prov: &mut $crate::provenance::Provenance,
            ) -> ::core::result::Result<Self, $crate::SettingsError> {
                ::core::result::Result::Ok(Self {
                    $( $field: $crate::resolve::pick(
                        src, prov, $env, $crate::__lever_rule!($kind), $default,
                    )?, )*
                })
            }
        }
    };
    (
        $(#[$meta:meta])*
        pub specs $Name:ident { group: $group:literal, class: $class:ident }
        $( $field:ident : $kind:ident = $default:tt , $flag:tt , $env:literal , $reader:literal , $doc:literal ; )*
    ) => {
        $(#[$meta])*
        pub enum $Name {}

        impl $Name {
            $crate::__lever_rows! { group: $group, class: $class,
                $( $field : $kind = $default , $flag , $env , $reader , $doc ; )* }
        }
    };
}

/// 2026-10-17: The `LEVERS` and `RULES` consts both `lever_group!` forms emit, written once.
#[macro_export]
#[doc(hidden)]
macro_rules! __lever_rows {
    (
        group: $group:literal, class: $class:ident,
        $( $field:ident : $kind:ident = $default:tt , $flag:tt , $env:literal , $reader:literal , $doc:literal ; )*
    ) => {
        pub const LEVERS: &'static [$crate::spec::LeverSpec] = &[
            $( $crate::spec::LeverSpec {
                env: $env,
                path: ::core::concat!($group, ".", ::core::stringify!($field)),
                ty: $crate::__lever_spec_ty!($kind),
                default: $default,
                flag: $crate::__lever_flag!($flag),
                class: $crate::spec::Class::$class,
                reader: $reader,
                doc: $doc,
            }, )*
        ];
        pub const RULES: &'static [$crate::spec::Rule] = &[ $( $crate::__lever_rule!($kind), )* ];
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! __lever_ty {
    (Switch "on") => { bool };
    (Switch "off") => { bool };
    (Switch "target-declared") => { ::core::option::Option<bool> };
    (Presence "off") => { bool };
    (Int "unset") => { ::core::option::Option<i64> };
    (Int "built-in") => { ::core::option::Option<i64> };
    (Int $d:literal) => { i64 };
    (Float "unset") => { ::core::option::Option<f64> };
    (Float $d:literal) => { f64 };
    (Text "unset") => { ::core::option::Option<$crate::__rt::String> };
    (Path "unset") => { ::core::option::Option<$crate::__rt::String> };
    (Text $d:literal) => { $crate::__rt::String };
    (Path $d:literal) => { $crate::__rt::String };
    ($k:ident $d:tt) => {
        ::core::compile_error!(::core::concat!(
            "lever_group!: no field type for ",
            ::core::stringify!($k),
            " = ",
            ::core::stringify!($d)
        ))
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! __lever_spec_ty {
    (Presence) => {
        $crate::spec::Ty::Switch
    };
    ($k:ident) => {
        $crate::spec::Ty::$k
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! __lever_rule {
    ($k:ident) => {
        $crate::spec::Rule::$k
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! __lever_flag {
    (None) => {
        ::core::option::Option::None
    };
    ($f:literal) => {
        ::core::option::Option::Some($f)
    };
}
