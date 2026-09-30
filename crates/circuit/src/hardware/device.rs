// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: `kernels/DEVICES.toml`: the GPU SKUs a circuit is planned for, the kernel class
//! each builds, its roofline and the tensor-core instruction kinds it runs natively, plus the
//! source guards that compile a kernel only where an instruction kind exists.
//!
//! Owner: metrale-circuit (hardware).
//! Invariants:
//! - Nothing defaults: every key the planner reads is required; an unknown top-level key, guard
//!   key, instruction family, `native_mma` key, source id or a duplicate device id is a typed
//!   error. A profile's other spec keys are carried for the roadmap and not read.
//! - A peak the device lacks (`fp4_block_scale` on Hopper, `0.0` in the research file) is
//!   `None` here, and `native_mma` must agree with the dense peaks.

use std::collections::{BTreeMap, BTreeSet};

use serde::Deserialize;

use super::HwError;

/// 2026-09-30: The tensor-core instruction family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MmaFamily {
    /// 2026-09-30: Warp-level `mma.sync` (SM 8.x-12.x; the GB10 path).
    MmaSync,
    /// 2026-09-30: Hopper warpgroup `wgmma` (operand B from shared memory).
    Wgmma,
    /// 2026-09-30: Blackwell datacentre `tcgen05` (accumulators in tensor memory).
    Tcgen05,
}

impl MmaFamily {
    /// 2026-09-30: The registry spelling.
    pub fn name(self) -> &'static str {
        match self {
            MmaFamily::MmaSync => "mma_sync",
            MmaFamily::Wgmma => "wgmma",
            MmaFamily::Tcgen05 => "tcgen05",
        }
    }

    fn parse(s: &str) -> Option<Self> {
        [MmaFamily::MmaSync, MmaFamily::Wgmma, MmaFamily::Tcgen05]
            .into_iter()
            .find(|f| f.name() == s)
    }
}

/// 2026-09-30: The operand format class an MMA multiplies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MmaKind {
    /// 2026-09-30: BF16 x BF16.
    Bf16,
    /// 2026-09-30: E4M3/E5M2 x E4M3/E5M2.
    Fp8,
    /// 2026-09-30: INT8 x INT8.
    Int8,
    /// 2026-09-30: NVFP4: E2M1 with an E4M3 scale per 16 elements, both operands.
    Fp4BlockScale,
    /// 2026-09-30: MX formats (UE8M0 scales per 32): FP8, FP6 or FP4 operands.
    Mxf8f6f4,
    /// 2026-09-30: FP16 x FP16.
    Fp16,
    /// 2026-09-30: TF32 x TF32.
    Tf32,
    /// 2026-09-30: FP64 tensor.
    Fp64,
    /// 2026-09-30: FP4 x FP8 without block scales.
    Fp4Fp8Unscaled,
    /// 2026-09-30: FP4 x FP8 with NVFP4 (E4M3 per 16) scales: W4A8 in one instruction.
    Fp4Fp8Nvfp4,
    /// 2026-09-30: FP4 x BF16.
    Fp4Bf16,
}

impl MmaKind {
    /// 2026-09-30: The registry spelling.
    pub fn name(self) -> &'static str {
        match self {
            MmaKind::Bf16 => "bf16",
            MmaKind::Fp8 => "fp8",
            MmaKind::Int8 => "int8",
            MmaKind::Fp4BlockScale => "fp4_block_scale",
            MmaKind::Mxf8f6f4 => "mxf8f6f4",
            MmaKind::Fp16 => "fp16",
            MmaKind::Tf32 => "tf32",
            MmaKind::Fp64 => "fp64",
            MmaKind::Fp4Fp8Unscaled => "fp4_fp8_unscaled",
            MmaKind::Fp4Fp8Nvfp4 => "fp4_fp8_nvfp4_scaled",
            MmaKind::Fp4Bf16 => "fp4_bf16",
        }
    }

    fn parse(s: &str) -> Option<Self> {
        [
            MmaKind::Bf16,
            MmaKind::Fp8,
            MmaKind::Int8,
            MmaKind::Fp4BlockScale,
            MmaKind::Mxf8f6f4,
            MmaKind::Fp16,
            MmaKind::Tf32,
            MmaKind::Fp64,
            MmaKind::Fp4Fp8Unscaled,
            MmaKind::Fp4Fp8Nvfp4,
            MmaKind::Fp4Bf16,
        ]
        .into_iter()
        .find(|k| k.name() == s)
    }

    /// 2026-09-30: The kind a research `native_mma` key names; `None` for an unknown key.
    fn of_native_key(key: &str) -> Option<Self> {
        Some(match key {
            "bf16_bf16" => MmaKind::Bf16,
            "fp16_fp16" => MmaKind::Fp16,
            "tf32_tf32" => MmaKind::Tf32,
            "fp8_fp8" => MmaKind::Fp8,
            "int8_int8" => MmaKind::Int8,
            "fp64" => MmaKind::Fp64,
            "fp4_fp4_nvfp4_block16" => MmaKind::Fp4BlockScale,
            "fp8_fp8_mx_block32" | "fp6_fp6" | "fp4_fp4_mxfp4_block32" => MmaKind::Mxf8f6f4,
            "fp4_fp8_unscaled" => MmaKind::Fp4Fp8Unscaled,
            "fp4_fp8_nvfp4_scaled" => MmaKind::Fp4Fp8Nvfp4,
            "fp4_bf16" => MmaKind::Fp4Bf16,
            _ => return None,
        })
    }
}

/// 2026-09-30: One instruction kind, `<family>.<kind>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Instr {
    /// 2026-09-30: Family.
    pub family: MmaFamily,
    /// 2026-09-30: Operand kind.
    pub kind: MmaKind,
}

impl Instr {
    /// 2026-09-30: Parse `mma_sync.fp8`.
    pub fn parse(s: &str) -> Result<Self, HwError> {
        let bad = || HwError::Registry(format!("`{s}` is not <mma_sync|wgmma|tcgen05>.<kind>"));
        let (f, k) = s.split_once('.').ok_or_else(bad)?;
        Ok(Instr {
            family: MmaFamily::parse(f).ok_or_else(bad)?,
            kind: MmaKind::parse(k).ok_or_else(bad)?,
        })
    }

    /// 2026-09-30: The registry spelling.
    pub fn name(self) -> String {
        format!("{}.{}", self.family.name(), self.kind.name())
    }
}

/// 2026-09-30: Dense tensor peaks, TFLOPS; a kind the device has no MMA for is `None`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Peaks {
    /// 2026-09-30: BF16.
    pub bf16: f64,
    /// 2026-09-30: FP8.
    pub fp8: Option<f64>,
    /// 2026-09-30: NVFP4 block-scaled.
    pub fp4_block_scale: Option<f64>,
}

/// 2026-09-30: One SKU.
#[derive(Debug, Clone, PartialEq)]
pub struct Device {
    /// 2026-09-30: `h100-sxm`.
    pub id: String,
    /// 2026-09-30: Marketing name.
    pub name: String,
    /// 2026-09-30: Kernel class, `kernels/<class>/`.
    pub class: String,
    /// 2026-09-30: Build arch (`sm_90a`), equal to the class's HARDWARE.toml.
    pub arch: String,
    /// 2026-09-30: Compute capability.
    pub compute_capability: String,
    /// 2026-09-30: SM count.
    pub sms: u32,
    /// 2026-09-30: Visible device memory, bytes (`memory_gib_visible`, else the datasheet GB,
    /// which tracks the visible GiB).
    pub memory_bytes: f64,
    /// 2026-09-30: Memory technology.
    pub memory_type: String,
    /// 2026-09-30: Datasheet DRAM bandwidth, GB/s.
    pub bandwidth_gbps: f64,
    /// 2026-09-30: L2, MB, when published.
    pub l2_mb: Option<f64>,
    /// 2026-09-30: Shared memory per SM, KB.
    pub smem_per_sm_kb: u32,
    /// 2026-09-30: Largest portable thread-block cluster.
    pub cluster_max: u32,
    /// 2026-09-30: Tensor memory per SM, KB (tcgen05 accumulators); 0 without.
    pub tmem_per_sm_kb: u32,
    /// 2026-09-30: The family fast kernels are written for.
    pub mma_family: MmaFamily,
    /// 2026-09-30: Instruction kinds executed natively (`mma_family` x the `native_mma` pairs).
    pub native: BTreeSet<Instr>,
    /// 2026-09-30: Dense datasheet peaks.
    pub peaks: Peaks,
    /// 2026-09-30: Where measured achievable figures live, when measured.
    pub measured: Option<String>,
    /// 2026-09-30: Share of memory the planner may use.
    pub usable_fraction: f64,
    /// 2026-09-30: Why that share.
    pub usable_why: String,
    /// 2026-09-30: Keys computed by the research, not published.
    pub derived: Vec<String>,
    /// 2026-09-30: Source ids (`S1`, ...), resolved in [`Registry::sources`].
    pub sources: Vec<String>,
}

impl Device {
    /// 2026-09-30: Any family runs `kind` natively.
    pub fn runs(&self, kind: MmaKind) -> bool {
        self.native.iter().any(|i| i.kind == kind)
    }
}

/// 2026-09-30: Whether a guarded region compiles when its macro is defined or when it is not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Polarity {
    /// 2026-09-30: `#ifdef M` / `#if defined(M)`: compiled only where `M` is defined.
    Ifdef,
    /// 2026-09-30: `#ifndef M`: compiled only where `M` is not defined.
    Ifndef,
}

/// 2026-09-30: A source guard and the instruction kind it stands for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Guard {
    /// 2026-09-30: The macro.
    pub macro_name: String,
    /// 2026-09-30: Its polarity.
    pub polarity: Polarity,
    /// 2026-09-30: What a kernel inside the region needs natively.
    pub requires: Instr,
}

/// 2026-09-30: The parsed registry.
#[derive(Debug, Clone, PartialEq)]
pub struct Registry {
    /// 2026-09-30: Devices, in file order.
    pub devices: Vec<Device>,
    /// 2026-09-30: Guards, in file order.
    pub guards: Vec<Guard>,
    /// 2026-09-30: Source id to citation.
    pub sources: BTreeMap<String, String>,
}

impl Registry {
    /// 2026-09-30: The device `id`; an unknown id lists the known ones.
    pub fn device(&self, id: &str) -> Result<&Device, HwError> {
        self.devices
            .iter()
            .find(|d| d.id == id)
            .ok_or_else(|| HwError::UnknownDevice {
                id: id.to_string(),
                known: self.devices.iter().map(|d| d.id.clone()).collect(),
            })
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct File {
    schema_version: u32,
    compiled: String,
    sources: BTreeMap<String, String>,
    guard: Vec<GuardFile>,
    device: Vec<DeviceFile>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GuardFile {
    #[serde(rename = "macro")]
    macro_name: String,
    polarity: String,
    requires: String,
}

#[derive(Deserialize)]
struct PeaksFile {
    bf16: f64,
    fp8: f64,
    fp4_nvfp4: f64,
}

// 2026-09-30: The planner's keys of one research profile. The other spec keys (clocks, NVLink,
// PCIe, TDP, sparse peaks, ridges, ...) are carried for the roadmap and not read here.
#[derive(Deserialize)]
struct DeviceFile {
    id: String,
    name: String,
    class: String,
    arch: String,
    compute_capability: String,
    sm_count: u32,
    memory_type: String,
    memory_gb_datasheet: f64,
    memory_gib_visible: Option<f64>,
    memory_bandwidth_gbps: f64,
    l2_mb: Option<f64>,
    smem_per_sm_kb: u32,
    tmem_per_sm_kb: u32,
    mma_family: String,
    max_cluster: u32,
    derived: Vec<String>,
    sources: Vec<String>,
    peak_tflops: PeaksFile,
    native_mma: BTreeMap<String, bool>,
    usable_fraction: f64,
    usable_why: String,
    measured: Option<String>,
}

/// 2026-09-30: Parse `kernels/DEVICES.toml`.
pub fn parse_devices(text: &str) -> Result<Registry, HwError> {
    let f: File = toml::from_str(text).map_err(|e| HwError::Registry(e.to_string()))?;
    if f.schema_version != 1 {
        return Err(HwError::Registry(format!(
            "schema_version {} (this build reads 1; compiled {})",
            f.schema_version, f.compiled
        )));
    }
    let guards = f
        .guard
        .into_iter()
        .map(|g| {
            let polarity = match g.polarity.as_str() {
                "ifdef" => Polarity::Ifdef,
                "ifndef" => Polarity::Ifndef,
                other => {
                    return Err(HwError::Registry(format!(
                        "guard {}: polarity `{other}` (ifdef | ifndef)",
                        g.macro_name
                    )));
                }
            };
            Ok(Guard {
                requires: Instr::parse(&g.requires)?,
                macro_name: g.macro_name,
                polarity,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut devices: Vec<Device> = Vec::with_capacity(f.device.len());
    for d in f.device {
        devices.push(device_of(d, &devices, &f.sources)?);
    }
    Ok(Registry {
        devices,
        guards,
        sources: f.sources,
    })
}

fn device_of(
    d: DeviceFile,
    seen: &[Device],
    sources: &BTreeMap<String, String>,
) -> Result<Device, HwError> {
    let field = |detail: String| HwError::Registry(format!("device {}: {detail}", d.id));
    if seen.iter().any(|x| x.id == d.id) {
        return Err(field("duplicate id".into()));
    }
    if let Some(s) = d.sources.iter().find(|s| !sources.contains_key(*s)) {
        return Err(field(format!("source `{s}` is not in [sources]")));
    }
    let mma_family = MmaFamily::parse(&d.mma_family)
        .ok_or_else(|| field(format!("mma_family `{}`", d.mma_family)))?;
    let mut native = BTreeSet::new();
    for (key, on) in &d.native_mma {
        let kind = MmaKind::of_native_key(key)
            .ok_or_else(|| field(format!("native_mma key `{key}` is no instruction kind")))?;
        if *on {
            native.insert(Instr {
                family: mma_family,
                kind,
            });
        }
    }
    if !(d.usable_fraction > 0.0 && d.usable_fraction <= 1.0) {
        return Err(field(format!("usable_fraction {}", d.usable_fraction)));
    }
    let p = &d.peak_tflops;
    let listed = |k: MmaKind| native.iter().any(|i| i.kind == k);
    for (kind, peak) in [
        (MmaKind::Fp8, p.fp8),
        (MmaKind::Fp4BlockScale, p.fp4_nvfp4),
    ] {
        if listed(kind) != (peak > 0.0) {
            return Err(field(format!(
                "`{}` is {} in native_mma but its dense peak is {peak}",
                kind.name(),
                if listed(kind) { "true" } else { "false" }
            )));
        }
    }
    let nonzero = |v: f64| (v > 0.0).then_some(v);
    Ok(Device {
        peaks: Peaks {
            bf16: p.bf16,
            fp8: nonzero(p.fp8),
            fp4_block_scale: nonzero(p.fp4_nvfp4),
        },
        memory_bytes: d.memory_gib_visible.unwrap_or(d.memory_gb_datasheet) * (1u64 << 30) as f64,
        id: d.id,
        name: d.name,
        class: d.class,
        arch: d.arch,
        compute_capability: d.compute_capability,
        sms: d.sm_count,
        memory_type: d.memory_type,
        bandwidth_gbps: d.memory_bandwidth_gbps,
        l2_mb: d.l2_mb,
        smem_per_sm_kb: d.smem_per_sm_kb,
        cluster_max: d.max_cluster,
        tmem_per_sm_kb: d.tmem_per_sm_kb,
        mma_family,
        native,
        measured: d.measured,
        usable_fraction: d.usable_fraction,
        usable_why: d.usable_why,
        derived: d.derived,
        sources: d.sources,
    })
}
