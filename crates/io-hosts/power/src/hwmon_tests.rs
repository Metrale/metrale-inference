// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: hwmon discovery and mapping on fake sysfs trees: a mapped amdgpu PPT channel is
//! read as Soc power, an unmapped one is never reported, a temperatures-only root does not open.

use std::fs;
use std::path::{Path, PathBuf};

use metrale_abstract_io::MonoNs;
use metrale_abstract_io::power::{Declared, EnergyCount, Field, PowerProbe, Rail};
use metrale_error::ErrorCode;

use super::*;

fn tree(tag: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!("metrale-hwmon-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).unwrap();
    root
}

fn chip(root: &Path, dir: &str, name: &str, files: &[(&str, &str)]) {
    let d = root.join(dir);
    fs::create_dir_all(&d).unwrap();
    fs::write(d.join("name"), format!("{name}\n")).unwrap();
    for (f, v) in files {
        fs::write(d.join(f), format!("{v}\n")).unwrap();
    }
}

fn soc() -> HwmonMapping {
    HwmonMapping {
        chip: "amdgpu".to_string(),
        label: Some("PPT".to_string()),
        rail: Rail::Soc,
        update_period_ms: 1.0,
    }
}

#[test]
fn a_mapped_amdgpu_ppt_channel_reads_as_soc_power_and_energy() {
    let root = tree("amd");
    chip(&root, "hwmon0", "acpitz", &[("temp1_input", "41000")]);
    chip(
        &root,
        "hwmon3",
        "amdgpu",
        &[
            ("power1_average", "23456000"),
            ("power1_input", "25000000"),
            ("power1_label", "PPT"),
            ("energy1_input", "123456789"),
            ("energy1_label", "PPT"),
            ("power2_input", "9000000"),
            ("power2_label", "fastPPT"),
        ],
    );
    let mut p = HwmonProbe::open(&root, &[soc()]).unwrap();
    let c = p.capabilities().clone();
    assert_eq!(
        c.declares(Field::AveragePower(Rail::Soc)),
        Declared::Available
    );
    assert_eq!(
        c.declares(Field::InstantPower(Rail::Soc)),
        Declared::Available
    );
    assert_eq!(c.declares(Field::Energy(Rail::Soc)), Declared::Available);
    assert_eq!(c.counter(Rail::Soc).unwrap().unit_j, 1e-6);
    assert!(matches!(
        c.declares(Field::InstantPower(Rail::GpuCore)),
        Declared::Unavailable(_)
    ));
    assert!(
        c.version().contains("amdgpu/fastPPT/Power"),
        "{}",
        c.version()
    );
    let s = p.read(MonoNs(9)).unwrap();
    assert_eq!(s.rails[&Rail::Soc].average_w, Some(23.456));
    assert_eq!(s.rails[&Rail::Soc].instant_w, Some(25.0));
    assert_eq!(s.energy[&Rail::Soc], EnergyCount(123_456_789));
    c.check(&s).unwrap();
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn a_temperatures_only_root_does_not_open() {
    let root = tree("gb10");
    chip(&root, "hwmon0", "acpitz", &[("temp1_input", "36000")]);
    chip(
        &root,
        "hwmon1",
        "nvme",
        &[("temp1_input", "40000"), ("temp1_label", "Composite")],
    );
    let e = HwmonProbe::open(&root, &[soc()]).err().unwrap();
    assert_eq!(e.code(), ErrorCode::PowerSourceUnavailable);
    assert!(
        e.to_string().contains("no hwmon power or energy inputs"),
        "{e}"
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn an_unmapped_label_is_never_guessed() {
    let root = tree("unmapped");
    chip(&root, "hwmon2", "amdgpu", &[("power1_average", "1000000")]);
    let e = HwmonProbe::open(&root, &[soc()]).err().unwrap();
    assert_eq!(e.code(), ErrorCode::PowerSourceUnavailable);
    assert!(e.to_string().contains("none mapped"), "{e}");
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn two_inputs_on_one_field_are_refused() {
    let root = tree("dup");
    chip(
        &root,
        "hwmon2",
        "amdgpu",
        &[("power1_average", "1"), ("power1_label", "PPT")],
    );
    chip(
        &root,
        "hwmon3",
        "amdgpu",
        &[("power1_average", "2"), ("power1_label", "PPT")],
    );
    let e = HwmonProbe::open(&root, &[soc()]).err().unwrap();
    assert!(e.to_string().contains("two inputs"), "{e}");
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn a_garbage_value_fails_the_read() {
    let root = tree("garbage");
    chip(
        &root,
        "hwmon2",
        "amdgpu",
        &[("power1_average", "1"), ("power1_label", "PPT")],
    );
    let mut p = HwmonProbe::open(&root, &[soc()]).unwrap();
    fs::write(root.join("hwmon2/power1_average"), "nope\n").unwrap();
    assert_eq!(
        p.read(MonoNs(0)).unwrap_err().code(),
        ErrorCode::PowerReadFailed
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn a_missing_root_does_not_open() {
    let e = HwmonProbe::open(Path::new("/nonexistent/hwmon"), &[soc()])
        .err()
        .unwrap();
    assert_eq!(e.code(), ErrorCode::PowerSourceUnavailable);
}

/// 2026-10-22: This box's real tree, read-only; prints what opens or why not.
#[test]
#[ignore]
fn live_sys_class_hwmon() {
    let root = Path::new("/sys/class/hwmon");
    println!("discovered: {:?}", discover(root));
    match HwmonProbe::open(root, &[soc()]) {
        Ok(mut p) => {
            println!("caps {:?}", p.capabilities());
            println!("sample {:?}", p.read(MonoNs(0)));
        }
        Err(e) => println!("does not open: {e}"),
    }
}
