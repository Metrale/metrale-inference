// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Holds the GPU microtests, parity gates and benches that used to live in
//! the `examples/` directory of `crates/model-arch`. This package has no library code;
//! Cargo needs one non-example target to build a package, and the examples are the product.
//!
//! Owner: gpu-examples.
//! Invariants: not a default workspace member; build it with `-p metrale-gpu-examples`.
