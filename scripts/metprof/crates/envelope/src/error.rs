// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: one typed error for every way a seal, an open, a wrap or an
//! unwrap can fail. No variant carries key material; the authentication
//! failures (`Decrypt`, `Unwrap`) carry nothing at all, so a caller cannot
//! learn which part of a forged input was wrong.

/// Every failure of this crate.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EnvelopeError {
    /// The input is shorter than the structure it announces.
    #[error("truncated envelope: {0}")]
    Truncated(&'static str),
    /// The first four bytes are not `KKB1`.
    #[error("not a KKB1 envelope (bad magic)")]
    BadMagic,
    /// The header does not parse as the KKB1 header schema.
    #[error("malformed header: {0}")]
    HeaderJson(String),
    /// The header parses but is not in its canonical byte form.
    #[error("header is not canonical JSON")]
    NonCanonicalHeader,
    /// The header is larger than a u32 length can describe.
    #[error("header too large")]
    HeaderTooLarge,
    /// `v` names a format version this crate does not read.
    #[error("unsupported envelope version {0}")]
    UnsupportedVersion(u32),
    /// `alg`, `kdf` or `wrapped_dek.scheme` names an algorithm this crate does not use.
    #[error("unsupported {field}: {value}")]
    Unsupported { field: &'static str, value: String },
    /// A kid outside `[A-Za-z0-9._-]{1,64}` (or `.`/`..`).
    #[error("invalid kid")]
    InvalidKid,
    /// A path outside `[A-Za-z0-9._-/]`, empty, absolute, or with an empty, `.` or `..` segment.
    #[error("invalid path")]
    InvalidPath,
    /// A required text field (encoding, publisher_build) is empty.
    #[error("missing required field: {0}")]
    MissingField(&'static str),
    /// The header's `info` is not `kkb/v1/<kid>/<path>` for a valid path.
    #[error("header info is not kkb/v1/<kid>/<path>")]
    BadInfo,
    /// The header names a different kid than the caller expects.
    #[error("kid mismatch: expected {expected}, found {found}")]
    KidMismatch { expected: String, found: String },
    /// The header names a different `info` (kid + path) than the caller expects.
    #[error("info mismatch: expected {expected}, found {found}")]
    InfoMismatch { expected: String, found: String },
    /// AES-256-GCM authentication failed: wrong key, or header/ciphertext/tag altered.
    #[error("decryption failed (authentication)")]
    Decrypt,
    /// Decryption succeeded but the plaintext length differs from the header.
    #[error("plaintext length mismatch: header {expected}, actual {actual}")]
    PlaintextLength { expected: u64, actual: u64 },
    /// Decryption succeeded but the plaintext sha256 differs from the header.
    #[error("plaintext sha256 mismatch")]
    PlaintextDigest,
    /// The plaintext is beyond the AES-GCM per-message limit (2^36 - 32 bytes).
    #[error("plaintext too large for AES-GCM")]
    PlaintextTooLarge,
    /// The DEK unwrap failed: wrong secret key, wrong kid, or wrapped fields altered.
    #[error("key unwrap failed (authentication)")]
    Unwrap,
    /// The X25519 exchange produced the all-zero secret (a low-order public key).
    #[error("x25519 exchange is not contributory")]
    NonContributory,
    /// An ML-KEM-768 encapsulation key fails the FIPS 203 modulus check.
    #[error("invalid public key: {0}")]
    InvalidPublicKey(&'static str),
    /// A byte field has the wrong length.
    #[error("{field}: expected {expected} bytes, got {actual}")]
    BadLength {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    /// The entropy port failed.
    #[error("entropy source failed: {0}")]
    Entropy(String),
    /// The clock port failed.
    #[error("clock failed: {0}")]
    Clock(String),
}

/// Result alias for this crate.
pub type Result<T> = core::result::Result<T, EnvelopeError>;
