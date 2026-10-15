// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the read-only SQLite reader against files written by the
//! `sqlite3` CLI (`fixtures/reader.sql`, `fixtures/wal.sql`), plus corrupt
//! and truncated images built here from them.

use std::path::{Path, PathBuf};

use metprof::adapters::sqlite::{Db, Value};

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("metprof-sqlite-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).expect("scratch dir");
    d
}

fn reader() -> Db {
    Db::open(&fixture("reader.sqlite")).expect("fixture opens")
}

fn rows(db: &Db, table: &str, cols: &[&str]) -> metprof::core::error::Result<Vec<Vec<Value>>> {
    let t = db
        .table(table)?
        .unwrap_or_else(|| panic!("table {table} present"));
    let mut out = Vec::new();
    db.scan(&t, cols, &mut |r| {
        out.push(r.to_vec());
        Ok(())
    })?;
    Ok(out)
}

fn int(i: i64) -> Value {
    Value::Int(i)
}

fn text(s: &str) -> Value {
    Value::Text(s.to_string())
}

#[test]
fn every_integer_width_float_null_blob_and_short_record_decodes() {
    use Value::Null;
    let got = rows(&reader(), "vals", &["id", "n", "x", "t", "b", "late"]).unwrap();
    let want: Vec<Vec<Value>> = vec![
        vec![int(1), int(0), Value::Real(0.5), text("zero"), Null, Null],
        vec![
            int(2),
            int(1),
            Value::Real(-2.25),
            text(""),
            Value::Blob(vec![0, 0xff, 0x10]),
            Null,
        ],
        vec![int(3), int(127), Value::Real(1e300), Null, Null, Null],
        vec![int(4), int(-128), Null, text("i8"), Null, Null],
        vec![
            int(5),
            int(32767),
            Value::Real(3.0),
            text("i16"),
            Null,
            Null,
        ],
        vec![int(6), int(-32768), Null, Null, Null, Null],
        vec![int(7), int(8_388_607), Null, text("i24"), Null, Null],
        vec![int(8), int(-8_388_608), Null, Null, Null, Null],
        vec![int(9), int(2_147_483_647), Null, text("i32"), Null, Null],
        vec![int(10), int(-2_147_483_648), Null, Null, Null, Null],
        vec![
            int(11),
            int(140_737_488_355_327),
            Null,
            text("i48"),
            Null,
            Null,
        ],
        vec![int(12), int(-140_737_488_355_328), Null, Null, Null, Null],
        vec![int(13), int(i64::MAX), Null, text("i64"), Null, Null],
        vec![int(14), int(i64::MIN), Null, Null, Null, Null],
        vec![int(15), Null, Null, Null, Null, Null],
        vec![int(16), int(2), Null, text("late"), Null, int(42)],
    ];
    assert_eq!(got, want);
}

#[test]
fn columns_come_back_in_the_order_asked_not_the_declared_order() {
    let got = rows(&reader(), "vals", &["t", "id"]).unwrap();
    assert_eq!(got[3], vec![text("i8"), int(4)]);
}

#[test]
fn a_multi_level_table_returns_every_row_in_rowid_order() {
    let data = std::fs::read(fixture("reader.sqlite")).unwrap();
    let interior = data.chunks(512).skip(1).filter(|p| p[0] == 0x05).count();
    assert!(
        interior >= 2,
        "fixture must force interior pages, has {interior}"
    );
    let got = rows(&reader(), "many", &["k", "sq"]).unwrap();
    assert_eq!(got.len(), 4000);
    for (i, r) in got.iter().enumerate() {
        let k = i as i64 + 1;
        assert_eq!(r, &vec![int(k), int(k * k)]);
    }
}

#[test]
fn a_text_value_spread_over_overflow_pages_is_reassembled() {
    let want = (1..=1500)
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(",");
    assert!(want.len() > 4 * 512, "value must need an overflow chain");
    let got = rows(&reader(), "big", &["id", "body"]).unwrap();
    assert_eq!(
        got,
        vec![
            vec![int(1), text("short")],
            vec![int(2), text(&want)],
            vec![int(3), text("after")]
        ]
    );
}

#[test]
fn quoted_identifiers_and_case_insensitive_lookup() {
    let got = rows(&reader(), "QUOTED NAME", &["START", "end", "v"]).unwrap();
    assert_eq!(got, vec![vec![int(5), int(6), text("q")]]);
}

#[test]
fn absent_table_is_none_and_absent_column_is_an_error() {
    let db = reader();
    assert!(db.table("nope").unwrap().is_none());
    let t = db.table("vals").unwrap().unwrap();
    assert!(t.has_column("LATE") && !t.has_column("zzz"));
    let e = db.scan(&t, &["id", "zzz"], &mut |_| Ok(())).unwrap_err();
    assert!(e.0.contains("has no column zzz"), "{e}");
}

#[test]
fn a_callback_error_stops_the_scan_and_propagates() {
    let db = reader();
    let t = db.table("many").unwrap().unwrap();
    let mut seen = 0;
    let e = db
        .scan(&t, &["k"], &mut |_| {
            seen += 1;
            if seen == 10 {
                Err(metprof::core::error::Error("stop".into()))
            } else {
                Ok(())
            }
        })
        .unwrap_err();
    assert!(e.0.ends_with("stop") && seen == 10, "{e} after {seen}");
}

#[test]
fn a_wal_mode_database_is_refused() {
    let e = Db::open(&fixture("wal.sqlite")).err().expect("refused");
    assert!(e.0.contains("WAL mode"), "{e}");
}

#[test]
fn a_wal_sidecar_is_refused() {
    let d = scratch("sidecar");
    let p = d.join("r.sqlite");
    std::fs::copy(fixture("reader.sqlite"), &p).unwrap();
    assert!(Db::open(&p).is_ok());
    std::fs::write(d.join("r.sqlite-wal"), b"").unwrap();
    let e = Db::open(&p).err().expect("refused");
    assert!(e.0.contains("-wal file exists"), "{e}");
}

#[test]
fn reads_leave_the_file_unchanged() {
    let p = fixture("reader.sqlite");
    let before = std::fs::read(&p).unwrap();
    for t in ["vals", "many", "big"] {
        rows(&Db::open(&p).unwrap(), t, &[]).unwrap();
    }
    assert_eq!(std::fs::read(&p).unwrap(), before);
}

#[test]
fn truncated_and_junk_images_are_errors() {
    let data = std::fs::read(fixture("reader.sqlite")).unwrap();
    for cut in [0, 50, 99, 100, 511, 3000, data.len() - 1] {
        let r = Db::from_bytes("cut", data[..cut].to_vec());
        assert!(r.is_err(), "a file cut at {cut} bytes opened");
    }
    assert!(Db::from_bytes("junk", b"this is not sqlite ".repeat(100)).is_err());
    let mut bad_page = data.clone();
    bad_page[16..18].copy_from_slice(&1000u16.to_be_bytes());
    assert!(Db::from_bytes("page", bad_page).is_err());
}

/// Every table and column, collecting errors; a corrupt image must never panic.
fn read_all(db: &Db) -> usize {
    let mut errors = 0;
    for (t, cols) in [
        ("vals", &["id", "n", "x", "t", "b", "late"][..]),
        ("many", &["k", "sq"]),
        ("big", &["body"]),
    ] {
        match db.table(t) {
            Ok(Some(table)) => {
                errors += usize::from(db.scan(&table, cols, &mut |_| Ok(())).is_err())
            }
            Ok(None) | Err(_) => errors += 1,
        }
    }
    errors
}

#[test]
fn corrupt_pages_are_errors_never_panics() {
    let data = std::fs::read(fixture("reader.sqlite")).unwrap();
    let mut errors = 0;
    for page in 1..data.len() / 512 {
        for fill in [0x00u8, 0xff, 0x05] {
            let mut b = data.clone();
            b[page * 512..(page + 1) * 512].fill(fill);
            if let Ok(db) = Db::from_bytes("corrupt", b) {
                errors += read_all(&db);
            }
        }
    }
    assert!(errors > 100, "only {errors} corruptions were detected");
}

#[test]
fn single_byte_corruption_anywhere_never_panics() {
    let data = std::fs::read(fixture("reader.sqlite")).unwrap();
    let mut errors = 0;
    for at in (0..data.len()).step_by(7) {
        for v in [0x00u8, 0xff] {
            let mut b = data.clone();
            b[at] = v;
            match Db::from_bytes("flip", b) {
                Ok(db) => errors += read_all(&db),
                Err(_) => errors += 1,
            }
        }
    }
    assert!(errors > 0);
}
