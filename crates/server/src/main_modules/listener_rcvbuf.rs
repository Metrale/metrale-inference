// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The listener's TCP receive buffer. On a host whose
//! `net.ipv4.tcp_rmem` default is below the kernel's own, the server sizes the
//! listener's buffer itself, so a large request body is not held back.
//!
//! The defect it removes: with a `tcp_rmem` default of 87380 (a common
//! "high-bandwidth" tuning, e.g. `4096 87380 134217728`), the first receive
//! window on loopback is smaller than one segment of loopback's 64 KiB MTU. A
//! request body above about 43 KB then waits on the kernel's 200 ms window
//! probe before the server sees it, once or twice per request. Measured on
//! GB10: a 32k-token prompt (127 KB of JSON) reached the server 208-216 ms late
//! on every request, while the server's own TTFT was unchanged. A 4 KB body was
//! not delayed. At the kernel default, 131072, no request was delayed.
//!
//! Owner: server (HTTP layer).
//! Invariants:
//! - A host at or above `KERNEL_DEFAULT_TCP_RMEM_BYTES` is left alone, so its
//!   accepted sockets keep receive autotuning.
//! - The listener's buffer is set before the first accept; accepted sockets
//!   inherit it.

use std::io;

/// 2026-10-03: The `tcp_rmem` default Linux ships (the middle field of
/// `net.ipv4.tcp_rmem`). Below it, the first loopback window can be smaller
/// than one segment, which causes the delay this module removes. 87380 was
/// measured to delay; 131072 was not.
pub(crate) const KERNEL_DEFAULT_TCP_RMEM_BYTES: u64 = 131_072;

/// 2026-10-03: The receive buffer the listener asks for on such a host. An
/// explicit `SO_RCVBUF` turns off receive autotuning for the accepted sockets,
/// so this value is their buffer for good. It must hold a large request in
/// flight: a 32k-token prompt is ~127 KB of JSON, and requests with tools or
/// images run to several MB, under the 32 MiB body limit. 4 MiB covers that
/// in a few windows. The kernel doubles the value it is given, then caps it at
/// `net.core.rmem_max`; any setting, even a capped one, removed the delay in
/// the measurements above.
pub(crate) const LISTENER_RCVBUF_BYTES: usize = 4 * 1024 * 1024;

/// 2026-10-03: What to do with the listener's receive buffer, given the host's
/// `tcp_rmem` default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RcvbufPlan {
    /// 2026-10-03: The host's default is at or above the kernel's; autotuning stays on.
    Keep { default_bytes: u64 },
    /// 2026-10-03: The host's default is below the kernel's; set `bytes` on the listener.
    Set { default_bytes: u64, bytes: usize },
    /// 2026-10-03: The default could not be read (not Linux, or `/proc` unreadable); the
    /// listener is left alone.
    Unknown,
}

/// 2026-10-03: The plan for a host whose `tcp_rmem` default is `default_bytes`
/// (`None` when it could not be read).
pub(crate) fn plan(default_bytes: Option<u64>) -> RcvbufPlan {
    match default_bytes {
        None => RcvbufPlan::Unknown,
        Some(d) if d >= KERNEL_DEFAULT_TCP_RMEM_BYTES => RcvbufPlan::Keep { default_bytes: d },
        Some(d) => RcvbufPlan::Set {
            default_bytes: d,
            bytes: LISTENER_RCVBUF_BYTES,
        },
    }
}

/// 2026-10-03: The middle field (the default) of a `net.ipv4.tcp_rmem` value,
/// `"min default max"`, separated by whitespace.
pub(crate) fn parse_tcp_rmem_default(text: &str) -> io::Result<u64> {
    let fields: Vec<&str> = text.split_whitespace().collect();
    let [_, default, _] = fields.as_slice() else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("expected three fields `min default max`, got {text:?}"),
        ));
    };
    default.parse::<u64>().map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("default field {default:?} of {text:?}: {e}"),
        )
    })
}

/// 2026-10-03: The host's `tcp_rmem` default, read from `/proc` (it is per
/// network namespace, so this is the one the listener's sockets get).
#[cfg(target_os = "linux")]
pub(crate) fn read_tcp_rmem_default() -> io::Result<u64> {
    let text = std::fs::read_to_string("/proc/sys/net/ipv4/tcp_rmem")?;
    parse_tcp_rmem_default(&text)
}

/// 2026-10-03: Off Linux there is no `tcp_rmem` to read.
#[cfg(not(target_os = "linux"))]
pub(crate) fn read_tcp_rmem_default() -> io::Result<u64> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "net.ipv4.tcp_rmem exists only on Linux",
    ))
}

/// 2026-10-03: Set `bytes` as the listener's receive buffer and return the
/// size the kernel reports back (doubled, and capped at `net.core.rmem_max`).
pub(crate) fn set_listener_rcvbuf(
    listener: &tokio::net::TcpListener,
    bytes: usize,
) -> io::Result<usize> {
    let sock = socket2::SockRef::from(listener);
    sock.set_recv_buffer_size(bytes)?;
    sock.recv_buffer_size()
}

/// 2026-10-03: Read the host's default, apply the plan to `listener` and log
/// what was done. A failure to set the buffer is logged as a warning: the
/// server still serves, but large request bodies can be delayed.
pub(crate) fn size_listener_rcvbuf(listener: &tokio::net::TcpListener) -> RcvbufPlan {
    let default = match read_tcp_rmem_default() {
        Ok(d) => Some(d),
        Err(e) => {
            tracing::debug!("TCP receive buffer left as is: tcp_rmem not read ({e})");
            None
        }
    };
    let decided = plan(default);
    match decided {
        RcvbufPlan::Keep { default_bytes } => tracing::debug!(
            "TCP receive buffer left as is: net.ipv4.tcp_rmem default {default_bytes} B \
             >= {KERNEL_DEFAULT_TCP_RMEM_BYTES} B"
        ),
        RcvbufPlan::Set {
            default_bytes,
            bytes,
        } => match set_listener_rcvbuf(listener, bytes) {
            Ok(effective) => tracing::info!(
                "TCP receive buffer: net.ipv4.tcp_rmem default is {default_bytes} B, below the \
                 kernel's {KERNEL_DEFAULT_TCP_RMEM_BYTES} B, which delays request bodies over \
                 ~43 KB by 200 ms or more on loopback. Set SO_RCVBUF {bytes} B on the listener \
                 (kernel reports {effective} B); accepted sockets inherit it"
            ),
            Err(e) => tracing::warn!(
                "TCP receive buffer: net.ipv4.tcp_rmem default is {default_bytes} B, below the \
                 kernel's {KERNEL_DEFAULT_TCP_RMEM_BYTES} B, and setting SO_RCVBUF {bytes} B \
                 failed ({e}). Request bodies over ~43 KB may be delayed by 200 ms or more; \
                 raise the middle field of net.ipv4.tcp_rmem to {KERNEL_DEFAULT_TCP_RMEM_BYTES}"
            ),
        },
        RcvbufPlan::Unknown => {}
    }
    decided
}

#[cfg(test)]
#[path = "listener_rcvbuf_tests.rs"]
mod tests;
