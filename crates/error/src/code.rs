// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The registry itself: every [`ErrorCode`], its stable number and its form,
//! grouped by the owner-confirmed range blocks (table in the crate docs).
//!
//! Owner: error.
//! Invariants:
//! - Append-only: a code is never renumbered or removed. A new code takes the next free number
//!   in its block and gets a line appended to `crates/error/codes.lock`.
//! - Prefer a specific code to [`ErrorCode::Generic`]; its uses are ratcheted
//!   (`tests/generic_ratchet.rs`).

use crate::registry::error_registry;

error_registry! {
    /// 2026-10-17: `0x0000–0x00FF`: universal codes any crate may use.
    General = 0x0000..=0x00FF {
        /// 2026-10-17: Unclassified; the one argument is the whole message. Ratcheted.
        Generic = 0x0000 => "{}",
        /// 2026-10-17: An i/o failure a host reports; the argument describes it.
        Io = 0x0001 => "i/o error: {}",
        /// 2026-10-17: A caller passed a value the callee refuses: (argument, reason).
        InvalidArgument = 0x0002 => "invalid argument {}: {}",
        /// 2026-10-17: A declared, reported gap: (operation, backend or host).
        NotImplemented = 0x0003 => "{} is not yet implemented on {}",
        /// 2026-10-17: A capability that exists but cannot serve now: (what).
        Unavailable = 0x0004 => "{} is unavailable",
    }
    /// 2026-10-17: `0x0100–0x01FF`: abstract-io ports and hosts (abstract-io-2.md D17).
    /// Sub-blocks: time `0x0100` (reserved; cannot fail), entropy `0x0110`, env `0x0118`,
    /// blob `0x0120–0x013F`, proc `0x0140`, http `0x0150`, transport `0x0170`,
    /// host construction `0x0180`.
    AbstractIo = 0x0100..=0x01FF {
        /// 2026-10-17: Entropy source unavailable.
        EntropyUnavailable = 0x0110 => "entropy source unavailable: {}",
        /// 2026-10-17: An environment variable is not valid unicode.
        EnvNotUnicode = 0x0118 => "environment variable {} is not valid unicode",
        /// 2026-10-17: A blob namespace is unavailable.
        BlobUnavailable = 0x0120 => "blob namespace {} unavailable",
        /// 2026-10-17: A blob was not found.
        BlobNotFound = 0x0121 => "blob {} not found",
        /// 2026-10-17: A blob key is invalid.
        BlobBadKey = 0x0122 => "invalid blob key {}",
        /// 2026-10-17: A blob namespace is full.
        BlobFull = 0x0123 => "blob namespace {} is full",
        /// 2026-10-17: A blob is corrupt.
        BlobCorrupt = 0x0124 => "blob {} is corrupt",
        /// 2026-10-17: A blob operation failed with an OS error.
        BlobOs = 0x0125 => "blob {}: os error {}",
        /// 2026-10-17: Process spawning is unavailable.
        ProcUnavailable = 0x0140 => "process spawning unavailable",
        /// 2026-10-17: A program was not found.
        ProcNotFound = 0x0141 => "program {} not found",
        /// 2026-10-17: Permission was denied running a program.
        ProcPermission = 0x0142 => "permission denied running {}",
        /// 2026-10-17: A program timed out.
        ProcTimeout = 0x0143 => "{} timed out after {} ms",
        /// 2026-10-17: A program wrote too much output.
        ProcOutputTooLarge = 0x0144 => "{} wrote more than {} bytes",
        /// 2026-10-17: A process operation failed with an OS error.
        ProcOs = 0x0145 => "{}: os error {}",
        /// 2026-10-17: A host is unreachable.
        HttpUnreachable = 0x0150 => "{} is unreachable",
        /// 2026-10-17: An HTTP request timed out.
        HttpTimeout = 0x0151 => "{} timed out after {} ms",
        /// 2026-10-17: An HTTP response exceeded a size limit.
        HttpTooLarge = 0x0152 => "{} exceeded {} bytes",
        /// 2026-10-17: An HTTP protocol error occurred.
        HttpProtocol = 0x0153 => "http protocol error from {}: {}",
        /// 2026-10-17: An HTTP response was truncated.
        HttpTruncated = 0x0154 => "response from {} was truncated",
        /// 2026-10-17: An HTTP operation failed with an OS error.
        HttpOs = 0x0155 => "{}: os error {}",
        /// 2026-10-17: A stream was closed.
        SinkClosed = 0x0170 => "stream {} closed",
        /// 2026-10-17: A stream is full.
        SinkFull = 0x0171 => "stream {} full",
        /// 2026-10-17: A host configuration error.
        HostConfig = 0x0180 => "host config: {}",
        /// 2026-10-17: A host capability is unavailable.
        HostUnavailable = 0x0181 => "host capability {} unavailable",
    }
    /// 2026-10-17: `0x0200–0x02FF`: sched-core (the `Device` trait, scheduler, lora) and the
    /// device backends.
    SchedDevice = 0x0200..=0x02FF {
    }
}
