// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The nvidia-smi line parser, the declaration a first line implies, and (ignored,
//! run by hand on a GPU box) the live child.

use metrale_abstract_io::MonoNs;
use metrale_abstract_io::power::{Declared, Field, PowerProbe, Rail};
use metrale_error::ErrorCode;

use super::*;

const GB10: &str = "17.73, 2405, 41, 0x0000000000000000, 580.159.03\n";

#[test]
fn a_good_line_parses_every_column() {
    let l = parse_line(GB10).unwrap();
    assert_eq!(l.power_avg_w, Some(17.73));
    assert_eq!(l.sm_mhz, Some(2405));
    assert_eq!(l.temp_c, Some(41.0));
    assert_eq!(l.reasons, Some(0));
    assert_eq!(l.driver.as_deref(), Some("580.159.03"));
}

#[test]
fn na_columns_parse_as_absent() {
    let l = parse_line("[N/A], 208, [Not Supported], 0x0000000000000001, 580.159.03").unwrap();
    assert_eq!(l.power_avg_w, None);
    assert_eq!(l.temp_c, None);
    assert_eq!(l.reasons, Some(1));
}

#[test]
fn garbage_lines_are_refused() {
    for bad in [
        "",
        "17.7, 2405",
        "abc, 2405, 41, 0x0, d",
        "17.7, 2405, 41, 12, d",
        "1,2,3,0x0,d,extra",
    ] {
        assert!(parse_line(bad).is_err(), "{bad:?}");
    }
}

#[test]
fn the_first_line_declares_average_power_not_instant_and_na_columns_unavailable() {
    let first = parse_line("18.1, 2405, [N/A], 0x0000000000000000, 580.159.03").unwrap();
    let c = declare("nvidia-smi0", &first, 250).unwrap();
    assert_eq!(
        c.declares(Field::AveragePower(Rail::GpuCore)),
        Declared::Available
    );
    assert!(
        matches!(c.declares(Field::InstantPower(Rail::GpuCore)), Declared::Unavailable(r) if r.contains("moving average"))
    );
    assert!(matches!(c.declares(Field::GpuTemp), Declared::Unavailable(r) if r.contains("[N/A]")));
    assert!(matches!(
        c.declares(Field::Energy(Rail::GpuCore)),
        Declared::Unavailable(_)
    ));
    let s = to_sample(&c, &parse_line(GB10).unwrap(), MonoNs(4)).unwrap();
    assert_eq!(s.rails[&Rail::GpuCore].average_w, Some(17.73));
    assert_eq!(
        s.temps.gpu_c, None,
        "a column declared unavailable is never filled in"
    );
    c.check(&s).unwrap();
}

#[test]
fn a_column_declared_available_that_later_reads_na_fails_the_read() {
    let c = declare("nvidia-smi0", &parse_line(GB10).unwrap(), 250).unwrap();
    let later = parse_line("[N/A], 2405, 41, 0x0000000000000000, 580.159.03").unwrap();
    let e = to_sample(&c, &later, MonoNs(0)).unwrap_err();
    assert_eq!(e.code(), ErrorCode::PowerReadFailed);
}

#[test]
fn the_idle_reason_bit_is_cleared_and_a_real_throttle_is_kept() {
    let c = declare("nvidia-smi0", &parse_line(GB10).unwrap(), 250).unwrap();
    let idle = parse_line("5.0, 208, 36, 0x0000000000000001, 580.159.03").unwrap();
    assert_eq!(to_sample(&c, &idle, MonoNs(0)).unwrap().throttle, Some(0));
    let capped = parse_line("90.0, 1500, 70, 0x0000000000000005, 580.159.03").unwrap();
    assert_eq!(
        to_sample(&c, &capped, MonoNs(0)).unwrap().throttle,
        Some(0x4)
    );
}

#[test]
fn a_missing_program_is_unavailable() {
    let e = NvidiaSmiProbe::open("/nonexistent/nvidia-smi", 0, 250)
        .err()
        .unwrap();
    assert_eq!(e.code(), ErrorCode::PowerSourceUnavailable);
}

/// 2026-10-22: Needs a GPU and nvidia-smi; read-only. `cargo test -p metrale-io-host-power --
/// --ignored live_nvidia_smi`.
#[test]
#[ignore]
fn live_nvidia_smi() {
    let mut p = NvidiaSmiProbe::open("nvidia-smi", 0, 250).unwrap();
    println!("caps: {:?}", p.capabilities());
    for k in 0..3 {
        let s = p.read(MonoNs(k)).unwrap();
        println!("sample {k}: {s:?}");
        p.capabilities().check(&s).unwrap();
    }
}
