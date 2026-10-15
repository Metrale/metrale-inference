// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the pure core: plain structs in, plain structs and strings out.
//! Nothing here reads a file, the environment or a clock.

pub mod capture;
pub mod config;
pub mod diff;
pub mod envdiff;
pub mod error;
pub mod families;
pub mod model;
pub mod report;
pub mod repr;
pub mod segment;
pub mod spec;
pub mod stats;
pub mod tables;
pub mod union;
pub mod verdict;
