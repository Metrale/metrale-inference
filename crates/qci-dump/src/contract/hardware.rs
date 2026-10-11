// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: `HardwareFingerprint` v1: only what decides whether a recipe runs the same
//! way (accelerators, memory, host architecture, driver, toolkit). No time, serial number
//! or hostname, so two machines of one build share one fingerprint and its digest. The
//! runtime does not probe it here; it reads a fingerprint document and digests it.
//!
//! Owner: qci-dump (contract binding).
//! Invariants: [`HardwareFingerprint::validate`] passes before a digest is used as a key.

use serde::{Deserialize, Serialize};

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
    pub memory_bytes: u64,
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
    pub cpu_model: String,
    pub memory_bytes: u64,
    /// 2026-10-11: CPU and accelerators share one memory pool (GB10, Strix Halo).
    pub unified_memory: bool,
    pub os_kernel: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HardwareFingerprint {
    pub version: u32,
    /// 2026-10-11: The class baselines and manifests key by (`gb10`, `strix-halo`).
    pub hardware_class: String,
    pub accelerators: Vec<AcceleratorGroup>,
    pub host: Host,
    pub driver: SoftwareVersion,
    pub toolkit: SoftwareVersion,
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
                a.memory_bytes >= 1,
                &format!("/accelerators/{i}/memory_bytes"),
            )?;
            require(a.count >= 1, &format!("/accelerators/{i}/count"))?;
        }
        let h = &self.host;
        require(rules::text(&h.cpu_model), "/host/cpu_model")?;
        require(h.memory_bytes >= 1, "/host/memory_bytes")?;
        require(rules::text(&h.os_kernel), "/host/os_kernel")?;
        self.driver.validate("/driver")?;
        self.toolkit.validate("/toolkit")
    }
}

impl SoftwareVersion {
    /// 2026-10-11: Both parts are `text`; `at` is this object's JSON pointer.
    pub fn validate(&self, at: &str) -> Result<(), Violation> {
        require(rules::text(&self.name), &format!("{at}/name"))?;
        require(rules::text(&self.version), &format!("{at}/version"))
    }
}
