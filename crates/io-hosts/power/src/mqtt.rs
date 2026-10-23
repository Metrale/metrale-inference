// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: [`MqttSubscribe`]: a minimal MQTT 3.1.1 subscriber (CONNECT, SUBSCRIBE at QoS 0,
//! receive PUBLISH, PINGREQ on the keep-alive) over std::net: a plug's `tele/<topic>/SENSOR`
//! (Tasmota) or `<id>/status/switch:0` (Shelly) stream, each payload one frame. One exact topic,
//! no wildcards (a wildcard would mix meters). No TLS and no credentials: a LAN broker.
//!
//! Owner: metrale-io-host-power.
//! Invariants: a refused CONNACK/SUBACK, a closed connection, a packet over the limit or no
//! matching PUBLISH within the frame timeout is `ExternalMeterTransport`.

use std::io::{ErrorKind, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::{Duration, Instant};

use crate::external::{Transport, transport_error};

/// 2026-10-22: Everything the connection needs, stated by the caller.
#[derive(Clone, Debug)]
pub struct MqttConfig {
    pub host: String,
    pub port: u16,
    pub client_id: String,
    pub topic: String,
    /// 2026-10-22: Seconds, > 0; a PINGREQ goes out every half of it while waiting.
    pub keepalive_s: u16,
    /// 2026-10-22: The longest wait for one matching PUBLISH.
    pub frame_timeout: Duration,
    /// 2026-10-22: The largest packet accepted.
    pub max_bytes: usize,
}

/// 2026-10-22: An open subscription.
pub struct MqttSubscribe {
    meter: String,
    cfg: MqttConfig,
    s: TcpStream,
}

fn put_str(out: &mut Vec<u8>, s: &str) -> Result<(), String> {
    let n = u16::try_from(s.len()).map_err(|_| format!("{s:?} is longer than 65535 bytes"))?;
    out.extend_from_slice(&n.to_be_bytes());
    out.extend_from_slice(s.as_bytes());
    Ok(())
}

/// 2026-10-22: A packet with its fixed header and variable-length remaining length.
pub fn packet(header: u8, body: &[u8]) -> Vec<u8> {
    let mut out = vec![header];
    let mut n = body.len();
    loop {
        let mut byte = u8::try_from(n % 128).expect("n % 128 fits a byte");
        n /= 128;
        if n > 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if n == 0 {
            break;
        }
    }
    out.extend_from_slice(body);
    out
}

impl MqttSubscribe {
    /// 2026-10-22: Connect, subscribe and wait for both acknowledgements.
    pub fn connect(meter: &str, cfg: MqttConfig) -> metrale_abstract_io::Result<Self> {
        Self::open(meter, cfg).map_err(|why| transport_error(meter, why))
    }

    fn open(meter: &str, cfg: MqttConfig) -> Result<Self, String> {
        if cfg.keepalive_s == 0 || cfg.topic.is_empty() || cfg.topic.contains(['+', '#']) {
            return Err(format!(
                "needs keepalive > 0 and one exact topic, got {cfg:?}"
            ));
        }
        let addr = (cfg.host.as_str(), cfg.port)
            .to_socket_addrs()
            .map_err(|e| format!("resolve {}: {e}", cfg.host))?
            .next()
            .ok_or_else(|| format!("{} resolves to nothing", cfg.host))?;
        let s = TcpStream::connect_timeout(&addr, cfg.frame_timeout)
            .map_err(|e| format!("connect {addr}: {e}"))?;
        s.set_write_timeout(Some(cfg.frame_timeout))
            .map_err(|e| e.to_string())?;
        let mut me = Self {
            meter: meter.to_string(),
            cfg,
            s,
        };
        let mut body = Vec::new();
        put_str(&mut body, "MQTT")?;
        body.extend_from_slice(&[4, 0x02]);
        body.extend_from_slice(&me.cfg.keepalive_s.to_be_bytes());
        put_str(&mut body, &me.cfg.client_id)?;
        me.send(&packet(0x10, &body))?;
        let deadline = Instant::now() + me.cfg.frame_timeout;
        let (h, b) = me.recv(deadline)?.ok_or("no CONNACK before the timeout")?;
        if h >> 4 != 2 || b.len() != 2 || b[1] != 0 {
            return Err(format!("CONNACK refused: header {h:#x} body {b:?}"));
        }
        let mut sub = vec![0, 1];
        put_str(&mut sub, &me.cfg.topic)?;
        sub.push(0);
        me.send(&packet(0x82, &sub))?;
        let (h, b) = me.recv(deadline)?.ok_or("no SUBACK before the timeout")?;
        if h >> 4 != 9 || b.len() < 3 || b[2] == 0x80 {
            return Err(format!("SUBACK refused: header {h:#x} body {b:?}"));
        }
        Ok(me)
    }

    fn send(&mut self, p: &[u8]) -> Result<(), String> {
        self.s.write_all(p).map_err(|e| format!("send: {e}"))
    }

    fn tick(&self) -> Duration {
        Duration::from_millis(u64::from(self.cfg.keepalive_s) * 500)
    }

    /// 2026-10-22: The next packet (header byte, body), `None` when `deadline` passes first;
    /// sends a PINGREQ each keep-alive half while idle.
    fn recv(&mut self, deadline: Instant) -> Result<Option<(u8, Vec<u8>)>, String> {
        let mut first = [0u8; 1];
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Ok(None);
            }
            self.s
                .set_read_timeout(Some(left.min(self.tick())))
                .map_err(|e| e.to_string())?;
            match self.s.read(&mut first) {
                Ok(0) => return Err("the broker closed the connection".to_string()),
                Ok(_) => break,
                Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {
                    self.send(&[0xC0, 0x00])?;
                }
                Err(e) => return Err(format!("receive: {e}")),
            }
        }
        self.s
            .set_read_timeout(Some(self.cfg.frame_timeout))
            .map_err(|e| e.to_string())?;
        let mut len = 0usize;
        for i in 0..4 {
            let mut b = [0u8; 1];
            self.s
                .read_exact(&mut b)
                .map_err(|e| format!("length: {e}"))?;
            len |= usize::from(b[0] & 0x7f) << (7 * i);
            if b[0] & 0x80 == 0 {
                if len > self.cfg.max_bytes {
                    return Err(format!(
                        "packet of {len} bytes exceeds {}",
                        self.cfg.max_bytes
                    ));
                }
                let mut body = vec![0u8; len];
                self.s
                    .read_exact(&mut body)
                    .map_err(|e| format!("body: {e}"))?;
                return Ok(Some((first[0], body)));
            }
        }
        Err("malformed remaining length".to_string())
    }

    fn next_publish(&mut self) -> Result<Vec<u8>, String> {
        let deadline = Instant::now() + self.cfg.frame_timeout;
        loop {
            let Some((h, b)) = self.recv(deadline)? else {
                return Err(format!(
                    "no PUBLISH on {} within {:?}",
                    self.cfg.topic, self.cfg.frame_timeout
                ));
            };
            if h >> 4 != 3 {
                continue;
            }
            let qos = (h >> 1) & 3;
            if b.len() < 2 {
                return Err("PUBLISH too short".to_string());
            }
            let tl = usize::from(u16::from_be_bytes([b[0], b[1]]));
            let mut at = 2 + tl;
            if b.len() < at {
                return Err("PUBLISH topic truncated".to_string());
            }
            let topic = &b[2..at];
            if qos > 0 {
                if b.len() < at + 2 {
                    return Err("PUBLISH packet id truncated".to_string());
                }
                let id = [b[at], b[at + 1]];
                at += 2;
                if qos == 1 {
                    self.send(&[0x40, 0x02, id[0], id[1]])?;
                }
            }
            if topic == self.cfg.topic.as_bytes() {
                return Ok(b[at..].to_vec());
            }
        }
    }
}

impl Transport for MqttSubscribe {
    fn next_frame(&mut self) -> metrale_abstract_io::Result<Vec<u8>> {
        self.next_publish()
            .map_err(|why| transport_error(&self.meter, why))
    }
}
