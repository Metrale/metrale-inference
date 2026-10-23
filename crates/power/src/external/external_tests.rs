// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Each external protocol reads the configured quantities, declares the rest
//! unavailable, and refuses a frame that does not carry what it is configured to read.

use alloc::string::ToString;

use metrale_abstract_io::MonoNs;
use metrale_abstract_io::power::{Declared, EnergyCount, Field, Rail, SourceId, SourceKind};
use metrale_error::ErrorCode;

use super::*;

fn placement(rail: Rail) -> MeterPlacement {
    MeterPlacement {
        source: SourceId("plug0".to_string()),
        device: "test meter".to_string(),
        rail,
        update_period_ms: 1000.0,
    }
}

const TASMOTA: &str = r#"{"StatusSNS":{"Time":"2026-10-22T20:00:00","ENERGY":{"TotalStartTime":"2026-01-01T00:00:00","Total":12.345,"Yesterday":0.5,"Today":0.25,"Power":[61.5,0],"ApparentPower":70,"Voltage":231,"Current":0.3}}}"#;
const SHELLY: &str = r#"{"id":0, "source":"init", "output":true, "apower":17.25, "voltage":229.8, "aenergy":{"total":1234.5,"by_minute":[1.0,2.0,3.0],"minute_ts":1700000000}, "temperature":{"tC":41.2}}"#;

#[test]
fn tasmota_power_and_total_are_read_at_their_paths() {
    let p = JsonPathProtocol::new(
        placement(Rail::Wall),
        Some(("StatusSNS.ENERGY.Power.0", Scale { to_si: 1.0 })),
        Some(("StatusSNS.ENERGY.Total", Scale { to_si: 3.6e6 }, 3.6)),
    )
    .unwrap();
    let s = p.parse(TASMOTA.as_bytes(), MonoNs(7)).unwrap();
    assert_eq!(s.at, MonoNs(7));
    assert_eq!(s.rails[&Rail::Wall].instant_w, Some(61.5));
    // 12.345 kWh = 44 442 000 J = 12 345 000 counts of 3.6 J (1 mWh).
    assert_eq!(s.energy[&Rail::Wall], EnergyCount(12_345_000));
    p.capabilities().check(&s).unwrap();
}

#[test]
fn shelly_gen2_power_and_energy_are_read() {
    let p = JsonPathProtocol::new(
        placement(Rail::Wall),
        Some(("apower", Scale { to_si: 1.0 })),
        Some(("aenergy.total", Scale { to_si: 3600.0 }, 3.6)),
    )
    .unwrap();
    let s = p.parse(SHELLY.as_bytes(), MonoNs(0)).unwrap();
    assert_eq!(s.rails[&Rail::Wall].instant_w, Some(17.25));
    assert_eq!(s.energy[&Rail::Wall], EnergyCount(1_234_500));
}

#[test]
fn a_missing_path_a_string_value_and_malformed_json_are_protocol_errors() {
    let p = JsonPathProtocol::new(
        placement(Rail::Wall),
        Some(("ENERGY.Power", Scale { to_si: 1.0 })),
        None,
    )
    .unwrap();
    for frame in [
        r#"{"ENERGY":{"Voltage":230}}"#,
        r#"{"ENERGY":{"Power":"61 W"}}"#,
        r#"{"ENERGY":{"Power":61"#,
        r#"{"ENERGY":[1,2]}"#,
        "not json",
    ] {
        let e = p.parse(frame.as_bytes(), MonoNs(0)).unwrap_err();
        assert_eq!(e.code(), ErrorCode::ExternalMeterProtocol, "{frame}");
    }
    assert!(p.parse(&[0xff, 0xfe], MonoNs(0)).is_err());
}

#[test]
fn json_extractor_walks_nested_objects_and_arrays_and_skips_siblings() {
    let doc = r#"{"a":{"s":"x\"}","arr":[{"v":1},{"v":[9,{"w":-2.5e1}]}]},"b":3}"#;
    assert_eq!(json_number_at(doc, &["b"]), Ok(3.0));
    assert_eq!(
        json_number_at(doc, &["a", "arr", "1", "v", "1", "w"]),
        Ok(-25.0)
    );
    assert_eq!(
        json_number_at(doc, &["a", "arr", "2"]),
        Err("path not found")
    );
    assert_eq!(
        json_number_at(doc, &["a", "s", "x"]),
        Err("path descends into a scalar")
    );
}

#[test]
fn a_negative_power_or_energy_is_refused() {
    let p = JsonPathProtocol::new(
        placement(Rail::Module),
        Some(("p", Scale { to_si: 1.0 })),
        Some(("e", Scale { to_si: 1.0 }, 1.0)),
    )
    .unwrap();
    assert!(p.parse(br#"{"p":-1,"e":1}"#, MonoNs(0)).is_err());
    assert!(p.parse(br#"{"p":1,"e":-1}"#, MonoNs(0)).is_err());
    assert!(p.parse(br#"{"p":1,"e":1}"#, MonoNs(0)).is_ok());
}

#[test]
fn capabilities_declare_the_meter_rail_only_and_everything_else_unavailable() {
    let p = JsonPathProtocol::new(
        placement(Rail::Module),
        Some(("p", Scale { to_si: 1.0 })),
        None,
    )
    .unwrap();
    let c = p.capabilities();
    assert_eq!(c.kind(), SourceKind::External);
    assert_eq!(
        c.declares(Field::InstantPower(Rail::Module)),
        Declared::Available
    );
    assert!(matches!(
        c.declares(Field::Energy(Rail::Module)),
        Declared::Unavailable(_)
    ));
    assert!(matches!(
        c.declares(Field::InstantPower(Rail::GpuCore)),
        Declared::Unavailable(_)
    ));
    assert!(matches!(
        c.declares(Field::SmClock),
        Declared::Unavailable(_)
    ));
    for f in Field::universe() {
        assert_ne!(c.declares(f), Declared::Undeclared, "{}", f.name());
    }
}

#[test]
fn a_meter_configured_with_nothing_is_refused() {
    let e = JsonPathProtocol::new(placement(Rail::Wall), None, None).unwrap_err();
    assert_eq!(e.code(), ErrorCode::ExternalMeterProtocol);
}

fn kv() -> KeyValueLineProtocol {
    KeyValueLineProtocol::new(
        placement(Rail::Module),
        '=',
        Some(KvKey {
            key: "P".to_string(),
            unit: "W".to_string(),
            scale: Scale { to_si: 1.0 },
        }),
        Some((
            KvKey {
                key: "E".to_string(),
                unit: "mWh".to_string(),
                scale: Scale { to_si: 3.6 },
            },
            0.036,
        )),
    )
    .unwrap()
}

#[test]
fn a_key_value_line_reads_power_and_energy_with_units() {
    let s = kv()
        .parse(b"V=20.01V, I=0.85A; P=17.02W E=1234.5mWh\r\n", MonoNs(3))
        .unwrap();
    assert_eq!(s.rails[&Rail::Module].instant_w, Some(17.02));
    // 1234.5 mWh = 4444.2 J = 123 450 counts of 0.036 J (0.01 mWh).
    assert_eq!(s.energy[&Rail::Module], EnergyCount(123_450));
}

#[test]
fn a_key_value_line_with_a_wrong_unit_a_missing_or_repeated_key_is_refused() {
    let p = kv();
    for line in [
        "P=17.02 E=1234.5mWh",
        "P=17.02W",
        "P=17.02W P=1W E=1mWh",
        "P=abcW E=1mWh",
        "P=17.02kW E=1mWh",
    ] {
        let e = p.parse(line.as_bytes(), MonoNs(0)).unwrap_err();
        assert_eq!(e.code(), ErrorCode::ExternalMeterProtocol, "{line}");
    }
}

fn hid() -> HidReportProtocol {
    HidReportProtocol::new(
        placement(Rail::Module),
        8,
        Some(HidField {
            offset: 1,
            width: 2,
            little_endian: true,
            signed: false,
            scale: Scale { to_si: 0.01 },
        }),
        Some((
            HidField {
                offset: 3,
                width: 4,
                little_endian: false,
                signed: false,
                scale: Scale { to_si: 1.0 },
            },
            1.0,
        )),
    )
    .unwrap()
}

#[test]
fn a_hid_report_reads_little_and_big_endian_fields() {
    let r = [0x01, 0x34, 0x12, 0x00, 0x01, 0x00, 0x02, 0xaa];
    let s = hid().parse(&r, MonoNs(0)).unwrap();
    assert!((s.rails[&Rail::Module].instant_w.unwrap() - 46.60).abs() < 1e-9);
    assert_eq!(s.energy[&Rail::Module], EnergyCount(0x0001_0002));
}

#[test]
fn a_hid_report_of_the_wrong_length_or_a_field_past_the_end_is_refused() {
    assert!(hid().parse(&[0u8; 7], MonoNs(0)).is_err());
    let bad = HidField {
        offset: 6,
        width: 4,
        little_endian: true,
        signed: false,
        scale: Scale { to_si: 1.0 },
    };
    assert!(HidReportProtocol::new(placement(Rail::Module), 8, Some(bad), None).is_err());
    let odd = HidField {
        width: 3,
        offset: 0,
        ..bad
    };
    assert!(HidReportProtocol::new(placement(Rail::Module), 8, Some(odd), None).is_err());
}

#[test]
fn a_signed_hid_power_below_zero_is_refused() {
    let f = HidField {
        offset: 0,
        width: 2,
        little_endian: true,
        signed: true,
        scale: Scale { to_si: 1.0 },
    };
    let p = HidReportProtocol::new(placement(Rail::Module), 2, Some(f), None).unwrap();
    assert!(p.parse(&[0xff, 0xff], MonoNs(0)).is_err());
    let s = p.parse(&[0x05, 0x00], MonoNs(0)).unwrap();
    assert_eq!(s.rails[&Rail::Module].instant_w, Some(5.0));
}
