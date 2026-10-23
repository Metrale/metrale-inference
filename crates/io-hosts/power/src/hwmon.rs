// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: [`HwmonProbe`]: Linux hwmon power and energy inputs, read-only. Under an explicit
//! sysfs root (normally `/sys/class/hwmon`) it discovers every `power<N>_input` and
//! `power<N>_average` (µW) and `energy<N>_input` (µJ, cumulative), with the chip `name` and the
//! channel `*_label`. Which rail a channel is comes only from the caller's mapping table (PCND:
//! amdgpu `PPT` is the APU socket, `Rail::Soc`); an unmapped channel is listed in the declared
//! version string and never reported. A root with no power or energy input at all (GB10: acpitz,
//! nvme, mlx5 temperatures only) does not open.
//!
//! Owner: metrale-io-host-power.
//! Invariants: nothing under the root is ever written.

use std::fs;
use std::path::{Path, PathBuf};

use metrale_abstract_io::MonoNs;
use metrale_abstract_io::power::{
    CounterSpec, EnergyCount, Field, PowerCapabilities, PowerProbe, PowerSample, Rail, RailCaps,
    SourceId, SourceKind,
};
use metrale_error::{ErrorCode, error};

/// 2026-10-22: hwmon's energy unit: one count is a µJ.
pub const ENERGY_UNIT_J: f64 = 1e-6;

/// 2026-10-22: One channel's rail: chip `name`, channel label (`None` matches a channel with no
/// label file), and the cadence the caller states for it (hwmon declares none).
#[derive(Clone, Debug, PartialEq)]
pub struct HwmonMapping {
    pub chip: String,
    pub label: Option<String>,
    pub rail: Rail,
    pub update_period_ms: f64,
}

/// 2026-10-22: What one input file is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputKind {
    Power,
    PowerAverage,
    Energy,
}

/// 2026-10-22: One discovered input file.
#[derive(Clone, Debug, PartialEq)]
pub struct Input {
    pub path: PathBuf,
    pub chip: String,
    pub label: Option<String>,
    pub kind: InputKind,
}

fn read_trim(p: &Path) -> Option<String> {
    fs::read_to_string(p).ok().map(|s| s.trim().to_string())
}

/// 2026-10-22: Every power and energy input under `root`, sorted by path.
pub fn discover(root: &Path) -> std::io::Result<Vec<Input>> {
    let mut out = Vec::new();
    for dev in fs::read_dir(root)? {
        let dir = dev?.path();
        let chip = read_trim(&dir.join("name")).unwrap_or_default();
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for e in entries.flatten() {
            let file = e.file_name().to_string_lossy().into_owned();
            let (stem, kind) = if let Some(s) = file.strip_suffix("_average") {
                (s, InputKind::PowerAverage)
            } else if let Some(s) = file.strip_suffix("_input") {
                match s {
                    _ if s.starts_with("power") => (s, InputKind::Power),
                    _ if s.starts_with("energy") => (s, InputKind::Energy),
                    _ => continue,
                }
            } else {
                continue;
            };
            let prefix = if kind == InputKind::Energy {
                "energy"
            } else {
                "power"
            };
            if !stem.starts_with(prefix)
                || !stem[prefix.len()..].chars().all(|c| c.is_ascii_digit())
            {
                continue;
            }
            out.push(Input {
                path: e.path(),
                chip: chip.clone(),
                label: read_trim(&dir.join(format!("{stem}_label"))),
                kind,
            });
        }
    }
    out.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(out)
}

/// 2026-10-22: The mapped inputs of one root.
pub struct HwmonProbe {
    caps: PowerCapabilities,
    reads: Vec<(Field, PathBuf)>,
}

fn field(kind: InputKind, rail: Rail) -> Field {
    match kind {
        InputKind::Power => Field::InstantPower(rail),
        InputKind::PowerAverage => Field::AveragePower(rail),
        InputKind::Energy => Field::Energy(rail),
    }
}

impl HwmonProbe {
    /// 2026-10-22: Discover `root` and keep the inputs `mapping` names.
    pub fn open(root: &Path, mapping: &[HwmonMapping]) -> metrale_abstract_io::Result<Self> {
        let name = format!("hwmon:{}", root.display());
        let no = |why: String| error!(ErrorCode::PowerSourceUnavailable, name.clone(), why);
        let inputs = discover(root).map_err(|e| no(format!("cannot list it: {e}")))?;
        if inputs.is_empty() {
            return Err(no("no hwmon power or energy inputs".to_string()));
        }
        let mut mapped: Vec<(&Input, &HwmonMapping, Field)> = Vec::new();
        let mut unmapped = Vec::new();
        for i in &inputs {
            let Some(m) = mapping
                .iter()
                .find(|m| m.chip == i.chip && m.label == i.label)
            else {
                unmapped.push(format!(
                    "{}/{}/{:?}",
                    i.chip,
                    i.label.as_deref().unwrap_or("-"),
                    i.kind
                ));
                continue;
            };
            let f = field(i.kind, m.rail);
            if mapped.iter().any(|(_, _, g)| *g == f) {
                return Err(no(format!("two inputs map to {}", f.name())));
            }
            mapped.push((i, m, f));
        }
        if mapped.is_empty() {
            return Err(no(format!(
                "power inputs found but none mapped: {}",
                unmapped.join(", ")
            )));
        }
        let version = format!(
            "hwmon {}; unmapped (not reported): {}",
            root.display(),
            if unmapped.is_empty() {
                "none".to_string()
            } else {
                unmapped.join(", ")
            }
        );
        let mut b = PowerCapabilities::builder(SourceId(name), SourceKind::Hwmon, version);
        for (i, m, f) in &mapped {
            b = match i.kind {
                InputKind::Power => b.sensor(
                    m.rail,
                    RailCaps {
                        update_period_ms: m.update_period_ms,
                        average_window_ms: None,
                    },
                ),
                InputKind::PowerAverage => b.field(*f),
                InputKind::Energy => b.counter(
                    m.rail,
                    CounterSpec {
                        unit_j: ENERGY_UNIT_J,
                        update_period_ms: m.update_period_ms,
                    },
                ),
            };
        }
        let caps = b
            .rest_unavailable(
                &Field::universe(),
                "no mapped hwmon input reports this field",
            )
            .build(&Field::universe())?;
        Ok(Self {
            caps,
            reads: mapped
                .into_iter()
                .map(|(i, _, f)| (f, i.path.clone()))
                .collect(),
        })
    }
}

impl PowerProbe for HwmonProbe {
    fn capabilities(&self) -> &PowerCapabilities {
        &self.caps
    }

    fn read(&mut self, at: MonoNs) -> metrale_abstract_io::Result<PowerSample> {
        let mut s = PowerSample::at(at);
        for (f, path) in &self.reads {
            let fail = |why: String| {
                error!(
                    ErrorCode::PowerReadFailed,
                    self.caps.source().0.clone(),
                    format!("{}: {why}", path.display())
                )
            };
            let text = fs::read_to_string(path).map_err(|e| fail(e.to_string()))?;
            let v: u64 = text
                .trim()
                .parse()
                .map_err(|_| fail(format!("{:?} is not a count", text.trim())))?;
            match *f {
                Field::Energy(r) => {
                    s.energy.insert(r, EnergyCount(v));
                }
                Field::InstantPower(r) => {
                    s.rails.entry(r).or_default().instant_w = Some(micro(v));
                }
                Field::AveragePower(r) => {
                    s.rails.entry(r).or_default().average_w = Some(micro(v));
                }
                _ => unreachable!("only power and energy inputs are mapped"),
            }
        }
        Ok(s)
    }
}

#[allow(clippy::cast_precision_loss)]
fn micro(v: u64) -> f64 {
    // 2026-10-22: µW counts are far below 2^53, so the f64 is exact.
    v as f64 * 1e-6
}

#[cfg(test)]
#[path = "hwmon_tests.rs"]
mod tests;
