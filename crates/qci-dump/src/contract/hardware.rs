// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: `HardwareFingerprint` v1: only what decides whether a recipe runs the same
//! way (accelerators, memory, host architecture, driver, toolkit). No time, serial number
//! or hostname, so two machines of one build share one fingerprint and its digest. The
//! runtime does not probe it here; it reads a fingerprint document and digests it.
//! What identifies the hardware (vendor, product, architecture, count, CPU architecture,
//! unified memory) is required; memory sizes, CPU model, OS kernel, driver and toolkit are
//! [`Field`]s a probe may leave absent (mfs DECISIONS PC-V). Absence is part of the digest.
//!
//! Owner: qci-dump (contract binding).
//! Invariants: [`HardwareFingerprint::validate`] passes before a digest is used as a key.

use serde::{Deserialize, Serialize};

use super::field::Field;
use super::rules::{self, Violation, require};

/// 2026-10-11: A named piece of software at a version (a driver, a toolkit, a runner).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SoftwareVersion {
    pub name: String,
    pub version: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Vendor {
    Nvidia,
    Amd,
    Intel,
    Apple,
}

/// 2026-10-11: Identical accelerators, counted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AcceleratorGroup {
    pub vendor: Vendor,
    pub product: String,
    /// 2026-10-11: The kernel target (`sm_121`, `gfx1151`).
    pub architecture: String,
    pub memory_bytes: Field<u64>,
    pub count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CpuArch {
    Aarch64,
    X86_64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Host {
    pub cpu_arch: CpuArch,
    pub cpu_model: Field<String>,
    pub memory_bytes: Field<u64>,
    /// 2026-10-11: CPU and accelerators share one memory pool (GB10, Strix Halo).
    pub unified_memory: bool,
    pub os_kernel: Field<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HardwareFingerprint {
    pub version: u32,
    /// 2026-10-11: The class baselines and manifests key by (`gb10`, `strix-halo`).
    pub hardware_class: String,
    pub accelerators: Vec<AcceleratorGroup>,
    pub host: Host,
    pub driver: Field<SoftwareVersion>,
    pub toolkit: Field<SoftwareVersion>,
}

impl HardwareFingerprint {
    /// 2026-10-11: The schema rules serde cannot express.
    pub fn validate(&self) -> Result<(), Violation> {
        rules::version(self.version)?;
        require(rules::dns_label(&self.hardware_class), "/hardware_class")?;
        require(!self.accelerators.is_empty(), "/accelerators")?;
        for (i, a) in self.accelerators.iter().enumerate() {
            require(
                rules::text(&a.product),
                &format!("/accelerators/{i}/product"),
            )?;
            require(
                rules::text(&a.architecture),
                &format!("/accelerators/{i}/architecture"),
            )?;
            require(
                bytes(&a.memory_bytes),
                &format!("/accelerators/{i}/memory_bytes"),
            )?;
            require(a.count >= 1, &format!("/accelerators/{i}/count"))?;
        }
        let h = &self.host;
        require(text(&h.cpu_model), "/host/cpu_model")?;
        require(bytes(&h.memory_bytes), "/host/memory_bytes")?;
        require(text(&h.os_kernel), "/host/os_kernel")?;
        for (v, at) in [(&self.driver, "/driver"), (&self.toolkit, "/toolkit")] {
            if let Some(v) = v.value() {
                v.validate(at)?;
            }
        }
        Ok(())
    }
}

/// 2026-10-11: `field_bytes`: a present byte count is at least one.
fn bytes(f: &Field<u64>) -> bool {
    f.value().is_none_or(|b| *b >= 1)
}

/// 2026-10-11: `field_text`: a present value is `text`.
fn text(f: &Field<String>) -> bool {
    f.value().is_none_or(|s| rules::text(s))
}

impl SoftwareVersion {
    /// 2026-10-11: Both parts are `text`; `at` is this object's JSON pointer.
    pub fn validate(&self, at: &str) -> Result<(), Violation> {
        require(rules::text(&self.name), &format!("{at}/name"))?;
        require(rules::text(&self.version), &format!("{at}/version"))
    }
}

#[cfg(test)]
mod tests {
    use super::super::field::AbsentReason;
    use super::*;

    fn fingerprint() -> HardwareFingerprint {
        HardwareFingerprint {
            version: 1,
            hardware_class: "gb10".into(),
            accelerators: vec![AcceleratorGroup {
                vendor: Vendor::Nvidia,
                product: "NVIDIA GB10".into(),
                architecture: "sm_121".into(),
                memory_bytes: Field::Measured(128),
                count: 1,
            }],
            host: Host {
                cpu_arch: CpuArch::Aarch64,
                cpu_model: Field::Measured("Grace".into()),
                memory_bytes: Field::Measured(128),
                unified_memory: true,
                os_kernel: Field::Measured("6.17.0".into()),
            },
            driver: Field::Measured(SoftwareVersion {
                name: "nvidia".into(),
                version: "580".into(),
            }),
            toolkit: Field::Absent(AbsentReason::Unavailable),
        }
    }

    #[test]
    fn an_absent_quantity_passes_and_a_present_one_keeps_its_rule() {
        let mut fp = fingerprint();
        assert_eq!(fp.validate(), Ok(()));
        fp.host.cpu_model = Field::Absent(AbsentReason::Unsupported);
        fp.host.memory_bytes = Field::Absent(AbsentReason::Unavailable);
        fp.accelerators[0].memory_bytes = Field::Absent(AbsentReason::Unavailable);
        assert_eq!(fp.validate(), Ok(()));
        let refused = |edit: fn(&mut HardwareFingerprint), at: &str| {
            let mut fp = fingerprint();
            edit(&mut fp);
            assert_eq!(fp.validate(), Err(Violation(at.into())));
        };
        refused(
            |f| f.accelerators[0].memory_bytes = Field::Measured(0),
            "/accelerators/0/memory_bytes",
        );
        refused(
            |f| f.host.memory_bytes = Field::Derived(0),
            "/host/memory_bytes",
        );
        refused(
            |f| f.host.cpu_model = Field::Measured(String::new()),
            "/host/cpu_model",
        );
        refused(
            |f| f.host.os_kernel = Field::Measured(String::new()),
            "/host/os_kernel",
        );
        refused(
            |f| {
                f.driver = Field::Measured(SoftwareVersion {
                    name: String::new(),
                    version: "1".into(),
                })
            },
            "/driver/name",
        );
    }
}
