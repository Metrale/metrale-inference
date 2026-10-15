// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: varints and the record format of the SQLite file format
//! (sections 1.6 and 2.1 of the file-format document). Pure over byte slices;
//! every read is bounds-checked, so a corrupt record is an error, never a panic.

use crate::core::error::Result;
use crate::fail;

/// One decoded column value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Int(i64),
    Real(f64),
    Text(String),
    Blob(Vec<u8>),
}

/// A big-endian varint of 1..=9 bytes at `buf[at..]`: the value and its
/// length. The ninth byte contributes all eight bits.
pub fn varint(buf: &[u8], at: usize) -> Result<(u64, usize)> {
    let mut v: u64 = 0;
    for i in 0..9 {
        let Some(&b) = buf.get(at + i) else {
            fail!("varint runs past the end of its buffer")
        };
        if i == 8 {
            return Ok(((v << 8) | u64::from(b), 9));
        }
        v = (v << 7) | u64::from(b & 0x7f);
        if b & 0x80 == 0 {
            return Ok((v, i + 1));
        }
    }
    fail!("varint longer than nine bytes")
}

/// Bytes a serial type occupies in the record body.
fn body_len(serial: u64) -> Result<usize> {
    Ok(match serial {
        0 | 8 | 9 => 0,
        1 => 1,
        2 => 2,
        3 => 3,
        4 => 4,
        5 => 6,
        6 | 7 => 8,
        10 | 11 => fail!("reserved serial type {serial} in a record"),
        n => usize::try_from((n - 12) / 2)
            .map_err(|_| crate::core::error::Error("oversized value".to_string()))?,
    })
}

fn signed(bytes: &[u8]) -> i64 {
    let mut v: i64 = if bytes.first().is_some_and(|b| b & 0x80 != 0) {
        -1
    } else {
        0
    };
    for &b in bytes {
        v = (v << 8) | i64::from(b);
    }
    v
}

fn decode(serial: u64, bytes: &[u8]) -> Result<Value> {
    Ok(match serial {
        0 => Value::Null,
        1..=6 => Value::Int(signed(bytes)),
        7 => {
            let mut a = [0u8; 8];
            a.copy_from_slice(bytes);
            Value::Real(f64::from_be_bytes(a))
        }
        8 => Value::Int(0),
        9 => Value::Int(1),
        n if n % 2 == 0 => Value::Blob(bytes.to_vec()),
        _ => match std::str::from_utf8(bytes) {
            Ok(s) => Value::Text(s.to_string()),
            Err(_) => fail!("a TEXT value is not valid UTF-8"),
        },
    })
}

/// Decode the columns at positions `want` (record order) of one record.
/// A position past the record's last column is `None`: the record predates
/// an `ALTER TABLE ADD COLUMN`, and the caller decides what that means.
pub fn columns(rec: &[u8], want: &[usize]) -> Result<Vec<Option<Value>>> {
    let (hlen, n) = varint(rec, 0)?;
    let Ok(hlen) = usize::try_from(hlen) else {
        fail!("record header length overflows")
    };
    if hlen < n || hlen > rec.len() {
        fail!(
            "record header length {hlen} outside the record ({} bytes)",
            rec.len()
        );
    }
    let mut serials = Vec::new();
    let mut at = n;
    while at < hlen {
        let (s, l) = varint(&rec[..hlen], at)?;
        serials.push(s);
        at += l;
    }
    let mut offsets = Vec::with_capacity(serials.len());
    let mut body = hlen;
    for &s in &serials {
        offsets.push(body);
        body = body.saturating_add(body_len(s)?);
    }
    if body > rec.len() {
        fail!(
            "record body needs {body} bytes, the payload has {}",
            rec.len()
        );
    }
    let mut out = Vec::with_capacity(want.len());
    for &c in want {
        match serials.get(c) {
            None => out.push(None),
            Some(&s) => {
                let start = offsets[c];
                let bytes = &rec[start..start + body_len(s)?];
                out.push(Some(decode(s, bytes)?));
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varints_of_every_length_decode() {
        assert_eq!(varint(&[0x7f], 0).unwrap(), (127, 1));
        assert_eq!(varint(&[0x81, 0x00], 0).unwrap(), (128, 2));
        let nine = [0xff; 9];
        assert_eq!(varint(&nine, 0).unwrap(), (u64::MAX, 9));
        assert!(varint(&[0x80, 0x80], 0).is_err());
    }

    #[test]
    fn signed_integers_of_each_width_sign_extend() {
        assert_eq!(signed(&[0xff]), -1);
        assert_eq!(signed(&[0x80, 0x00]), -32768);
        assert_eq!(signed(&[0x00, 0x80, 0x00]), 32768);
        assert_eq!(signed(&[0xff, 0xff, 0xff, 0xff, 0xff, 0xfe]), -2);
    }

    #[test]
    fn a_record_decodes_and_a_short_record_reports_absent_columns() {
        // header: len 4, types int8 (1), text len 2 (17), null (0); body 0x05 "hi"
        let rec = [4, 1, 17, 0, 5, b'h', b'i'];
        let got = columns(&rec, &[0, 1, 2, 3]).unwrap();
        assert_eq!(
            got,
            vec![
                Some(Value::Int(5)),
                Some(Value::Text("hi".into())),
                Some(Value::Null),
                None
            ]
        );
    }

    #[test]
    fn a_body_past_the_payload_is_an_error() {
        let rec = [3, 6, 1, 0, 0];
        assert!(columns(&rec, &[0]).is_err());
        assert!(columns(&[9, 1], &[0]).is_err());
        assert!(columns(&[2, 10], &[0]).is_err());
    }
}
