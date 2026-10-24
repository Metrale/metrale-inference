// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Blueprint trust: who may sign a `.kbp`, and the check that they did.
//!
//! A signature is hybrid: ML-DSA (FIPS 204, deterministic, with a versioned context string)
//! **and** Ed25519 over the context-prefixed, length-prefixed message; a signature verifies
//! only when both halves do. Conventions are `metrale-auth`'s (seeds, `sign_deterministic`,
//! `verify_with_context`, kid-selected keys), with this crate's own contexts ([`Purpose`]).
//!
//! The chain is one level: a root key ([`Anchor`], compiled into the binary or, in a declared
//! dev mode, supplied by the caller) signs a [`KeyCert`] for a scoped signing key and the
//! [`Revocations`] list; the signing key signs the payload. [`verify_signed`] checks, in order:
//! the kid selects a certificate, the certificate's issuer is an anchor, the certificate's
//! signature, its decode, revocation, validity window, scope, and only then the payload
//! signature.
//!
//! Owner: blueprint.
//! Invariants:
//! - `no_std` + `alloc`, no `unsafe`, no I/O, no clock (the caller passes `now`), builds for
//!   both wasm32 targets (NSCC).
//! - Nothing a signature covers is trusted before the signature verifies.

#![no_std]
#![forbid(unsafe_code)]
#![deny(
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

extern crate alloc;

mod anchors;
mod cert;
mod keys;
mod names;
mod verify;

pub use anchors::{
    ANCHORS, Anchor, CompiledAnchor, EMBEDDED_REVOKED, REVOCATIONS_FLOOR, TrustAnchors,
};
pub use cert::{KeyCert, Revocations, SignedCert, SignedRevocations, certify, sign_revocations};
pub use keys::{HybridSig, PublicKeys, Signer, lv_cat, verify_hybrid};
pub use names::{Kid, Purpose, Scope, Suite};
pub use verify::{Revoked, SigBlock, TrustPolicy, Verified, verify_revocations, verify_signed};

/// 2026-10-22: The ML-DSA and Ed25519 seed length: the 32-byte secret a key file holds.
pub const SEED_LEN: usize = 32;
