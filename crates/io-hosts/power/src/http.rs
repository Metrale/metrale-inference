// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: [`HttpPoll`]: a plain HTTP/1.1 GET per frame (a smart plug's local status API:
//! Tasmota `/cm?cmnd=Status%208`, Shelly `/rpc/Switch.GetStatus?id=0`), std::net only, with an
//! explicit timeout and body limit. No TLS: these APIs are LAN-local and plain.
//!
//! Owner: metrale-io-host-power.
//! Invariants: a non-200 status, a body over the limit, or a closed connection is
//! `ExternalMeterTransport`.

use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

use crate::external::{Transport, transport_error};

/// 2026-10-22: GET `path` from `host:port` each frame.
pub struct HttpPoll {
    meter: String,
    host: String,
    port: u16,
    path: String,
    timeout: Duration,
    max_bytes: usize,
}

impl HttpPoll {
    /// 2026-10-22: Every argument explicit: no default timeout or limit.
    #[must_use]
    pub fn new(
        meter: &str,
        host: &str,
        port: u16,
        path: &str,
        timeout: Duration,
        max_bytes: usize,
    ) -> Self {
        Self {
            meter: meter.to_string(),
            host: host.to_string(),
            port,
            path: path.to_string(),
            timeout,
            max_bytes,
        }
    }

    fn fetch(&self) -> Result<Vec<u8>, String> {
        let addr = (self.host.as_str(), self.port)
            .to_socket_addrs()
            .map_err(|e| format!("resolve {}: {e}", self.host))?
            .next()
            .ok_or_else(|| format!("{} resolves to nothing", self.host))?;
        let mut s = TcpStream::connect_timeout(&addr, self.timeout)
            .map_err(|e| format!("connect {addr}: {e}"))?;
        s.set_read_timeout(Some(self.timeout))
            .map_err(|e| e.to_string())?;
        s.set_write_timeout(Some(self.timeout))
            .map_err(|e| e.to_string())?;
        write!(
            s,
            "GET {} HTTP/1.1\r\nHost: {}\r\nAccept: */*\r\nConnection: close\r\n\r\n",
            self.path, self.host
        )
        .map_err(|e| format!("send: {e}"))?;
        let mut raw = Vec::new();
        s.take(u64::try_from(self.max_bytes).unwrap_or(u64::MAX) + 1)
            .read_to_end(&mut raw)
            .map_err(|e| format!("receive: {e}"))?;
        if raw.len() > self.max_bytes {
            return Err(format!("response exceeds {} bytes", self.max_bytes));
        }
        parse_response(&raw)
    }
}

/// 2026-10-22: The body of a complete HTTP/1.x 200 response (identity, Content-Length or
/// chunked); why not otherwise.
pub fn parse_response(raw: &[u8]) -> Result<Vec<u8>, String> {
    let split = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .ok_or("the connection closed before the headers ended")?;
    let head = std::str::from_utf8(&raw[..split]).map_err(|_| "headers are not UTF-8")?;
    let body = &raw[split + 4..];
    let mut lines = head.split("\r\n");
    let status = lines.next().unwrap_or_default();
    let code = status.split_whitespace().nth(1).unwrap_or_default();
    if !status.starts_with("HTTP/1.") || code != "200" {
        return Err(format!("status line {status:?}"));
    }
    let header = |name: &str| {
        head.split("\r\n").skip(1).find_map(|l| {
            let (k, v) = l.split_once(':')?;
            k.trim()
                .eq_ignore_ascii_case(name)
                .then(|| v.trim().to_string())
        })
    };
    if header("transfer-encoding").is_some_and(|v| v.eq_ignore_ascii_case("chunked")) {
        return dechunk(body);
    }
    match header("content-length") {
        Some(n) => {
            let n: usize = n.parse().map_err(|_| format!("content-length {n:?}"))?;
            if body.len() < n {
                return Err(format!("body truncated: {} of {n} bytes", body.len()));
            }
            Ok(body[..n].to_vec())
        }
        None => Ok(body.to_vec()),
    }
}

fn dechunk(mut b: &[u8]) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    loop {
        let eol = b
            .windows(2)
            .position(|w| w == b"\r\n")
            .ok_or("chunk size line truncated")?;
        let size_txt = std::str::from_utf8(&b[..eol]).map_err(|_| "chunk size is not UTF-8")?;
        let size_txt = size_txt.split(';').next().unwrap_or_default().trim();
        let n =
            usize::from_str_radix(size_txt, 16).map_err(|_| format!("chunk size {size_txt:?}"))?;
        b = &b[eol + 2..];
        if n == 0 {
            return Ok(out);
        }
        if b.len() < n + 2 {
            return Err("chunk truncated".to_string());
        }
        out.extend_from_slice(&b[..n]);
        b = &b[n + 2..];
    }
}

impl Transport for HttpPoll {
    fn next_frame(&mut self) -> metrale_abstract_io::Result<Vec<u8>> {
        self.fetch()
            .map_err(|why| transport_error(&self.meter, why))
    }
}
