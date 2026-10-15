// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the TTFT client's I/O: one HTTP/1.1 exchange over a plain TCP
//! socket (no TLS), timed from just before the request is written, and a salt
//! from the clock, the process id and a counter.

use std::io::{ErrorKind, Read as _, Write as _};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::core::error::{Error, Result};
use crate::core::ttft::client::Endpoint;
use crate::core::ttft::sse::Read;
use crate::ports::{ChatTransport, SaltSource};

const READ_BUFFER: usize = 64 * 1024;

pub struct TcpChat;

fn connect(ep: &Endpoint, timeout: Duration) -> Result<TcpStream> {
    let name = format!("{}:{}", ep.host, ep.port);
    let addrs = name
        .to_socket_addrs()
        .map_err(|e| Error(format!("cannot resolve {name}: {e}")))?;
    let mut last = format!("cannot resolve {name}: no address");
    for a in addrs {
        match TcpStream::connect_timeout(&a, timeout) {
            Ok(s) => return Ok(s),
            Err(e) => last = format!("cannot connect to {name}: {e}"),
        }
    }
    Err(Error(last))
}

impl ChatTransport for TcpChat {
    fn exchange(&self, ep: &Endpoint, request: &[u8], timeout_s: f64) -> Result<Vec<Read>> {
        let timeout = Duration::try_from_secs_f64(timeout_s)
            .map_err(|e| Error(format!("timeout {timeout_s} s: {e}")))?;
        let name = format!("{}:{}", ep.host, ep.port);
        let io = |what: &str, e: std::io::Error| Error(format!("{what} {name}: {e}"));
        let mut s = connect(ep, timeout)?;
        s.set_nodelay(true).map_err(|e| io("cannot configure", e))?;
        s.set_read_timeout(Some(timeout))
            .map_err(|e| io("cannot configure", e))?;
        s.set_write_timeout(Some(timeout))
            .map_err(|e| io("cannot configure", e))?;
        let t0 = Instant::now();
        s.write_all(request).map_err(|e| io("cannot write to", e))?;
        let mut reads = Vec::new();
        let mut buf = vec![0u8; READ_BUFFER];
        loop {
            let Some(left) = timeout.checked_sub(t0.elapsed()).filter(|d| !d.is_zero()) else {
                return Err(Error(format!(
                    "{name}: no complete response within {timeout_s} s"
                )));
            };
            s.set_read_timeout(Some(left))
                .map_err(|e| io("cannot configure", e))?;
            match s.read(&mut buf) {
                Ok(0) => return Ok(reads),
                Ok(n) => reads.push(Read {
                    at_ns: t0.elapsed().as_nanos() as u64,
                    bytes: buf[..n].to_vec(),
                }),
                Err(e) if e.kind() == ErrorKind::Interrupted => {}
                Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {
                    return Err(Error(format!(
                        "{name}: no complete response within {timeout_s} s"
                    )));
                }
                Err(e) => return Err(io("read failed from", e)),
            }
        }
    }
}

pub struct ClockSalt;

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

impl SaltSource for ClockSalt {
    fn fresh(&self) -> u64 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        splitmix64(now ^ (u64::from(std::process::id()) << 32) ^ splitmix64(n))
    }
}
