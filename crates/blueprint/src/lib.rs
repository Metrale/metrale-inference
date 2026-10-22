// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Blueprints (`.kbp`): one serve, fully stated, content-addressed and signed.
//!
//! - [`types`]: the body (checkpoint, hardware, circuit, plans, kernel set, typed settings,
//!   evidence) and its canonical postcard encoding;
//! - [`frame`]: the container (72-byte header, signed body, unsigned locator table, inline
//!   payload, signature block), verification before decode, lite ↔ full;
//! - [`resolve`]: kernel bytes from `internal` and `inline` sources, checked by hash;
//! - [`transport`]: the `kbp1.` + base64 text form;
//! - [`recipe_view`]: a recipe as a projection of a blueprint, both directions;
//! - [`export`]: the store's `kbp-card/v1` and `kbp-graph/v1` and their JSON Schemas.
//!
//! Owner: blueprint.
//! Invariants:
//! - `no_std` + `alloc`, no `unsafe`, no I/O; builds for both wasm32 targets (NSCC), so the
//!   store verifies with this same code.
//! - The circuit instance and its plans are `metrale_circuit`'s own types, serialized by their
//!   serde derives (SSOT); this crate defines no copy of them.

#![no_std]
#![forbid(unsafe_code)]
#![deny(
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

extern crate alloc;

pub mod codec;
pub mod export;
pub mod frame;
pub mod ids;
pub mod recipe_view;
pub mod resolve;
pub mod transport;
pub mod types;

pub use ids::{ArtifactId, BlueprintId, LineageId, Sha256, Sha256Stream};
pub use types::{BODY_SCHEMA, Blueprint, OLDEST_BODY_SCHEMA};
