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
//! 2026-10-22: Superseded for larger bodies: at the kernel default the first
//! window is 131072 B, and a body written in one go that is larger than it
//! waits on the same probe. Measured on GB10 rank 0 (`tcp_rmem` 4096 131072
//! 134217728, GLM-5.3-Flash serve): bodies of 32-96 KB reached the server in
//! 6-18 ms, 128-480 KB in 210-270 ms on every request (a 128k-token prompt is
//! ~480 KB), while the same bytes sent as two writes (headers, then body)
//! arrived in 6 ms. The 2026-10-03 run sent 127 KB, under that window. So the
//! listener is now sized on every host whose default is below the buffer it
//! needs, not only below the kernel's default.
//!
//! Owner: server (HTTP layer).
//! Invariants:
//! - A host whose `tcp_rmem` default is at least `LISTENER_RCVBUF_BYTES` is left
//!   alone, so its accepted sockets keep receive autotuning.
//! - The listener's buffer is set before the first accept; accepted sockets
//!   inherit it.

use std::io;

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
    /// 2026-10-03: The host's default is at least [`LISTENER_RCVBUF_BYTES`] (2026-10-22: was
    /// the kernel's default); autotuning stays on.
    Keep { default_bytes: u64 },
    /// 2026-10-03: The host's default is below [`LISTENER_RCVBUF_BYTES`]; set `bytes` on the
    /// listener.
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
        Some(d) if d >= LISTENER_RCVBUF_BYTES as u64 => RcvbufPlan::Keep { default_bytes: d },
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
             >= {LISTENER_RCVBUF_BYTES} B"
        ),
        RcvbufPlan::Set {
            default_bytes,
            bytes,
        } => match set_listener_rcvbuf(listener, bytes) {
            Ok(effective) => tracing::info!(
                "TCP receive buffer: net.ipv4.tcp_rmem default is {default_bytes} B, so a \
                 request body larger than the first window ({default_bytes} B or less) waits \
                 200 ms or more on loopback. Set SO_RCVBUF {bytes} B on the listener (kernel \
                 reports {effective} B); accepted sockets inherit it"
            ),
            Err(e) => tracing::warn!(
                "TCP receive buffer: net.ipv4.tcp_rmem default is {default_bytes} B, and setting \
                 SO_RCVBUF {bytes} B failed ({e}). Request bodies over ~{default_bytes} B may \
                 be delayed by 200 ms or more; raise the middle field of net.ipv4.tcp_rmem \
                 to {bytes}"
            ),
        },
        RcvbufPlan::Unknown => {}
    }
    decided
}

#[cfg(test)]
#[path = "listener_rcvbuf_tests.rs"]
mod tests;
