// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Each transport against an in-process fake (a temp file for serial and hidraw, a
//! TcpListener thread for HTTP and for an MQTT broker), read through a real protocol.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::time::Duration;

use metrale_abstract_io::MonoNs;
use metrale_abstract_io::power::{EnergyCount, PowerProbe, Rail, SourceId};
use metrale_error::ErrorCode;
use metrale_power::external::{
    HidField, HidReportProtocol, JsonPathProtocol, KeyValueLineProtocol, KvKey, MeterPlacement,
    Scale,
};

use super::*;
use crate::http::{HttpPoll, parse_response};
use crate::mqtt::{MqttConfig, MqttSubscribe, packet};

fn placement(rail: Rail) -> MeterPlacement {
    MeterPlacement {
        source: SourceId("meter0".to_string()),
        device: "fake".to_string(),
        rail,
        update_period_ms: 1000.0,
    }
}

fn temp(tag: &str, bytes: &[u8]) -> PathBuf {
    let p = std::env::temp_dir().join(format!("metrale-ext-{tag}-{}", std::process::id()));
    std::fs::write(&p, bytes).unwrap();
    p
}

fn shelly() -> JsonPathProtocol {
    JsonPathProtocol::new(
        placement(Rail::Wall),
        Some(("apower", Scale { to_si: 1.0 })),
        Some(("aenergy.total", Scale { to_si: 3600.0 }, 3.6)),
    )
    .unwrap()
}

#[test]
fn serial_lines_feed_a_key_value_protocol_and_eof_is_a_transport_error() {
    let p = temp("serial", b"P=17.02W E=10J\r\n\r\nP=18.5W E=12J\n");
    let proto = KeyValueLineProtocol::new(
        placement(Rail::Module),
        '=',
        Some(KvKey {
            key: "P".into(),
            unit: "W".into(),
            scale: Scale { to_si: 1.0 },
        }),
        Some((
            KvKey {
                key: "E".into(),
                unit: "J".into(),
                scale: Scale { to_si: 1.0 },
            },
            1.0,
        )),
    )
    .unwrap();
    let mut m = ExternalMeter::new(SerialLines::open("meter0", &p).unwrap(), proto);
    assert_eq!(
        m.read(MonoNs(1)).unwrap().rails[&Rail::Module].instant_w,
        Some(17.02)
    );
    assert_eq!(
        m.read(MonoNs(2)).unwrap().energy[&Rail::Module],
        EnergyCount(12)
    );
    assert_eq!(
        m.read(MonoNs(3)).unwrap_err().code(),
        ErrorCode::ExternalMeterTransport
    );
    std::fs::remove_file(p).unwrap();
}

#[test]
fn a_missing_serial_device_is_unavailable() {
    let e = SerialLines::open("meter0", std::path::Path::new("/nonexistent/ttyACM9"))
        .err()
        .unwrap();
    assert_eq!(e.code(), ErrorCode::PowerSourceUnavailable);
}

#[test]
fn hidraw_reports_are_read_whole_and_a_short_one_is_a_transport_error() {
    let p = temp("hid", &[0x00, 0x10, 0x27, 0xAA, 0x00, 0x20]);
    let proto = HidReportProtocol::new(
        placement(Rail::Module),
        4,
        Some(HidField {
            offset: 1,
            width: 2,
            little_endian: true,
            signed: false,
            scale: Scale { to_si: 0.001 },
        }),
        None,
    )
    .unwrap();
    let mut m = ExternalMeter::new(
        HidrawReports::open("meter0", &p, proto.report_len()).unwrap(),
        proto,
    );
    assert_eq!(
        m.read(MonoNs(0)).unwrap().rails[&Rail::Module].instant_w,
        Some(10.0)
    );
    assert_eq!(
        m.read(MonoNs(0)).unwrap_err().code(),
        ErrorCode::ExternalMeterTransport
    );
    std::fs::remove_file(p).unwrap();
}

fn serve_http(responses: Vec<Vec<u8>>) -> u16 {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = l.local_addr().unwrap().port();
    std::thread::spawn(move || {
        for r in responses {
            let (mut c, _) = l.accept().unwrap();
            let mut req = Vec::new();
            let mut b = [0u8; 256];
            while !req.windows(4).any(|w| w == b"\r\n\r\n") {
                let n = c.read(&mut b).unwrap();
                assert!(n > 0, "client closed before its request ended");
                req.extend_from_slice(&b[..n]);
            }
            assert!(req.starts_with(b"GET /rpc/Switch.GetStatus?id=0 HTTP/1.1\r\n"));
            c.write_all(&r).unwrap();
            c.shutdown(std::net::Shutdown::Write).unwrap();
        }
    });
    port
}

#[test]
fn an_http_poll_reads_a_shelly_status_including_chunked_bodies() {
    let body = br#"{"apower":17.25,"aenergy":{"total":1.5}}"#;
    let mut plain =
        format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n", body.len()).into_bytes();
    plain.extend_from_slice(body);
    let mut chunked = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n".to_vec();
    chunked.extend_from_slice(format!("{:x}\r\n", 10).as_bytes());
    chunked.extend_from_slice(&body[..10]);
    chunked.extend_from_slice(format!("\r\n{:x}\r\n", body.len() - 10).as_bytes());
    chunked.extend_from_slice(&body[10..]);
    chunked.extend_from_slice(b"\r\n0\r\n\r\n");
    let port = serve_http(vec![
        plain,
        chunked,
        b"HTTP/1.1 503 Busy\r\nContent-Length: 0\r\n\r\n".to_vec(),
    ]);
    let t = HttpPoll::new(
        "meter0",
        "127.0.0.1",
        port,
        "/rpc/Switch.GetStatus?id=0",
        Duration::from_secs(2),
        4096,
    );
    let mut m = ExternalMeter::new(t, shelly());
    for _ in 0..2 {
        let s = m.read(MonoNs(0)).unwrap();
        assert_eq!(s.rails[&Rail::Wall].instant_w, Some(17.25));
        assert_eq!(s.energy[&Rail::Wall], EnergyCount(1500));
    }
    assert_eq!(
        m.read(MonoNs(0)).unwrap_err().code(),
        ErrorCode::ExternalMeterTransport
    );
}

#[test]
fn http_response_parsing_refuses_truncation_and_bad_status() {
    assert!(parse_response(b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\nabc").is_err());
    assert!(parse_response(b"HTTP/1.1 404 Not Found\r\n\r\n").is_err());
    assert!(parse_response(b"HTTP/1.1 200 OK\r\nContent-").is_err());
    assert_eq!(
        parse_response(b"HTTP/1.1 200 OK\r\n\r\nbody").unwrap(),
        b"body"
    );
}

#[test]
fn a_closed_http_connection_is_a_transport_error() {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = l.local_addr().unwrap().port();
    std::thread::spawn(move || {
        let (c, _) = l.accept().unwrap();
        drop(c);
    });
    let mut t = HttpPoll::new(
        "meter0",
        "127.0.0.1",
        port,
        "/",
        Duration::from_secs(2),
        4096,
    );
    assert_eq!(
        t.next_frame().unwrap_err().code(),
        ErrorCode::ExternalMeterTransport
    );
}

fn read_packet(c: &mut std::net::TcpStream) -> (u8, Vec<u8>) {
    let mut h = [0u8; 2];
    c.read_exact(&mut h).unwrap();
    let mut b = vec![0u8; usize::from(h[1])];
    c.read_exact(&mut b).unwrap();
    (h[0], b)
}

fn publish(topic: &str, payload: &[u8]) -> Vec<u8> {
    let mut b = u16::try_from(topic.len()).unwrap().to_be_bytes().to_vec();
    b.extend_from_slice(topic.as_bytes());
    b.extend_from_slice(payload);
    packet(0x30, &b)
}

fn broker(connack_rc: u8, close_after_suback: bool) -> u16 {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = l.local_addr().unwrap().port();
    std::thread::spawn(move || {
        let (mut c, _) = l.accept().unwrap();
        let (h, b) = read_packet(&mut c);
        assert_eq!(h, 0x10);
        assert_eq!(&b[2..6], b"MQTT");
        c.write_all(&[0x20, 0x02, 0x00, connack_rc]).unwrap();
        if connack_rc != 0 {
            return;
        }
        let (h, b) = read_packet(&mut c);
        assert_eq!(h, 0x82);
        c.write_all(&[0x90, 0x03, b[0], b[1], 0x00]).unwrap();
        if close_after_suback {
            return;
        }
        c.write_all(&publish("tele/other/SENSOR", br#"{"apower":1}"#))
            .unwrap();
        c.write_all(&publish(
            "shellies/plug0/status",
            br#"{"apower":42.5,"aenergy":{"total":2}}"#,
        ))
        .unwrap();
        std::thread::sleep(Duration::from_millis(200));
    });
    port
}

fn cfg(port: u16) -> MqttConfig {
    MqttConfig {
        host: "127.0.0.1".into(),
        port,
        client_id: "metrale-test".into(),
        topic: "shellies/plug0/status".into(),
        keepalive_s: 10,
        frame_timeout: Duration::from_secs(2),
        max_bytes: 4096,
    }
}

#[test]
fn an_mqtt_subscription_skips_other_topics_and_reads_the_meter() {
    let t = MqttSubscribe::connect("meter0", cfg(broker(0, false))).unwrap();
    let mut m = ExternalMeter::new(t, shelly());
    let s = m.read(MonoNs(5)).unwrap();
    assert_eq!(s.rails[&Rail::Wall].instant_w, Some(42.5));
    assert_eq!(s.energy[&Rail::Wall], EnergyCount(2000));
}

#[test]
fn a_refused_connack_a_wildcard_topic_and_a_closed_broker_are_transport_errors() {
    let e = MqttSubscribe::connect("meter0", cfg(broker(5, false)))
        .err()
        .unwrap();
    assert_eq!(e.code(), ErrorCode::ExternalMeterTransport);
    let mut wild = cfg(1);
    wild.topic = "shellies/+/status".into();
    assert!(MqttSubscribe::connect("meter0", wild).is_err());
    let mut t = MqttSubscribe::connect("meter0", cfg(broker(0, true))).unwrap();
    assert_eq!(
        t.next_frame().unwrap_err().code(),
        ErrorCode::ExternalMeterTransport
    );
}

#[test]
fn a_malformed_frame_over_a_good_transport_is_a_protocol_error() {
    let p = temp("badframe", b"{\"apower\": \n");
    let mut m = ExternalMeter::new(SerialLines::open("meter0", &p).unwrap(), shelly());
    assert_eq!(
        m.read(MonoNs(0)).unwrap_err().code(),
        ErrorCode::ExternalMeterProtocol
    );
    std::fs::remove_file(p).unwrap();
}
