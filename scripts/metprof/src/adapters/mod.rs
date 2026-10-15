// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the real implementations of the ports: the filesystem and the
//! nsys sqlite export (through the in-crate read-only SQLite reader).

pub mod files;
pub mod nsys;
pub mod sqlite;
