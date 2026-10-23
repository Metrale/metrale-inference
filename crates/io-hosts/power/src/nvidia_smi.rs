// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: [`NvidiaSmiProbe`]: one long-lived `nvidia-smi -lms <period>` child for one GPU,
//! one line per read. `power.draw.average` is the driver's moving average of the GPU rail, so it
//! is reported as `AveragePower(GpuCore)`, never as an instant reading; the bench's existing
//! `gpu_rail_*` keys are integrals of the same column. A column that reads `[N/A]` when the probe
//! opens is declared unavailable with that reason; one that reads `[N/A]` later is a failed read.
//!
//! Owner: metrale-io-host-power.
//! Invariants: the child is killed and reaped on drop.

use std::io::{BufRead, BufReader};
use std::process::{Child, ChildStdout, Command, Stdio};

use metrale_abstract_io::MonoNs;
use metrale_abstract_io::power::{
    Field, PowerCapabilities, PowerProbe, PowerSample, Rail, RailReading, SourceId, SourceKind,
};
use metrale_error::{ErrorCode, error};

/// 2026-10-22: The streamed columns, in the order [`parse_line`] reads them.
pub const QUERY: &str = "--query-gpu=power.draw.average,clocks.sm,temperature.gpu,\
                         clocks_event_reasons.active,driver_version";

/// 2026-10-22: `clocks_event_reasons` bits that do not hold the clocks down: GpuIdle (0x1).
/// "None" is 0. Cleared so a non-zero mask means throttled (the port's `throttle` contract).
pub const IDLE_REASON_BITS: u64 = 0x1;

/// 2026-10-22: One parsed line; `None` where the driver printed `[N/A]` / `[Not Supported]`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SmiLine {
    pub power_avg_w: Option<f64>,
    pub sm_mhz: Option<u32>,
    pub temp_c: Option<f64>,
    pub reasons: Option<u64>,
    pub driver: Option<String>,
}

fn absent(v: &str) -> bool {
    matches!(v, "[N/A]" | "N/A" | "[Not Supported]" | "[Unknown Error]")
}

/// 2026-10-22: One `--format=csv,noheader,nounits` line of [`QUERY`]; why it does not parse
/// otherwise.
pub fn parse_line(line: &str) -> Result<SmiLine, String> {
    let cols: Vec<&str> = line.trim().split(',').map(str::trim).collect();
    let [p, sm, t, r, d] = cols.as_slice() else {
        return Err(format!("expected 5 columns, got {}: {line:?}", cols.len()));
    };
    fn num<T: std::str::FromStr>(name: &str, v: &str) -> Result<Option<T>, String> {
        if absent(v) {
            return Ok(None);
        }
        v.parse()
            .map(Some)
            .map_err(|_| format!("{name} = {v:?} is not a number"))
    }
    let reasons = if absent(r) {
        None
    } else {
        let hex = r
            .strip_prefix("0x")
            .ok_or_else(|| format!("reasons {r:?} is not 0x-hex"))?;
        Some(u64::from_str_radix(hex, 16).map_err(|_| format!("reasons {r:?} is not hex"))?)
    };
    Ok(SmiLine {
        power_avg_w: num("power.draw.average", p)?,
        sm_mhz: num("clocks.sm", sm)?,
        temp_c: num("temperature.gpu", t)?,
        reasons,
        driver: (!absent(d) && !d.is_empty()).then(|| (*d).to_string()),
    })
}

/// 2026-10-22: The streaming child for GPU `index`.
pub struct NvidiaSmiProbe {
    caps: PowerCapabilities,
    child: Child,
    lines: BufReader<ChildStdout>,
}

fn unavailable(name: &str, why: String) -> metrale_error::Error {
    error!(ErrorCode::PowerSourceUnavailable, name.to_string(), why)
}

impl NvidiaSmiProbe {
    /// 2026-10-22: Start `nvidia-smi -i <index> -lms <period_ms>` (both explicit), read its first
    /// line, and declare from it what it reports.
    pub fn open(program: &str, index: u32, period_ms: u32) -> metrale_abstract_io::Result<Self> {
        let name = format!("nvidia-smi{index}");
        let mut child = Command::new(program)
            .arg(QUERY)
            .arg("--format=csv,noheader,nounits")
            .arg("-i")
            .arg(index.to_string())
            .arg("-lms")
            .arg(period_ms.to_string())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| unavailable(&name, format!("cannot spawn {program}: {e}")))?;
        let stdout = child.stdout.take().expect("stdout is piped");
        let mut lines = BufReader::new(stdout);
        let mut first = String::new();
        let n = lines.read_line(&mut first).unwrap_or(0);
        let parsed = if n == 0 {
            Err("exited before printing a reading".to_string())
        } else {
            parse_line(&first)
        };
        let caps = parsed
            .map_err(|why| unavailable(&name, why))
            .and_then(|first| declare(&name, &first, period_ms));
        match caps {
            Ok(caps) => Ok(Self { caps, child, lines }),
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                Err(e)
            }
        }
    }
}

/// 2026-10-22: The capabilities a first line shows.
pub fn declare(
    name: &str,
    first: &SmiLine,
    period_ms: u32,
) -> metrale_abstract_io::Result<PowerCapabilities> {
    const NA: &str = "nvidia-smi printed [N/A] for this column when the probe opened";
    let mut b = PowerCapabilities::builder(
        SourceId(name.to_string()),
        SourceKind::NvidiaSmi,
        format!(
            "nvidia-smi -lms {period_ms}; driver {}",
            first.driver.as_deref().unwrap_or("[N/A]")
        ),
    )
    .unavailable(
        Field::InstantPower(Rail::GpuCore),
        "power.draw.average is the driver's moving average, not an instant reading",
    );
    let cols = [
        (
            first.power_avg_w.is_some(),
            Field::AveragePower(Rail::GpuCore),
        ),
        (first.sm_mhz.is_some(), Field::SmClock),
        (first.temp_c.is_some(), Field::GpuTemp),
        (first.reasons.is_some(), Field::Throttle),
    ];
    for (present, f) in cols {
        b = if present {
            b.field(f)
        } else {
            b.unavailable(f, NA)
        };
    }
    b.rest_unavailable(
        &Field::universe(),
        "not in the nvidia-smi query this probe streams",
    )
    .build(&Field::universe())
}

/// 2026-10-22: `line` as a sample at `at`, refusing a column the capabilities declare available
/// that now reads `[N/A]`.
pub fn to_sample(
    caps: &PowerCapabilities,
    line: &SmiLine,
    at: MonoNs,
) -> metrale_abstract_io::Result<PowerSample> {
    let name = || caps.source().0.clone();
    let want = |f: Field| caps.available().contains(&f);
    let mut s = PowerSample::at(at);
    let missing = |f: Field| {
        error!(
            ErrorCode::PowerReadFailed,
            name(),
            format!("{} read [N/A]", f.name())
        )
    };
    if want(Field::AveragePower(Rail::GpuCore)) {
        let w = line
            .power_avg_w
            .ok_or_else(|| missing(Field::AveragePower(Rail::GpuCore)))?;
        s.rails.insert(
            Rail::GpuCore,
            RailReading {
                average_w: Some(w),
                ..RailReading::default()
            },
        );
    }
    if want(Field::SmClock) {
        s.clocks.sm_mhz = Some(line.sm_mhz.ok_or_else(|| missing(Field::SmClock))?);
    }
    if want(Field::GpuTemp) {
        s.temps.gpu_c = Some(line.temp_c.ok_or_else(|| missing(Field::GpuTemp))?);
    }
    if want(Field::Throttle) {
        let r = line.reasons.ok_or_else(|| missing(Field::Throttle))?;
        s.throttle = Some(r & !IDLE_REASON_BITS);
    }
    Ok(s)
}

impl PowerProbe for NvidiaSmiProbe {
    fn capabilities(&self) -> &PowerCapabilities {
        &self.caps
    }

    fn read(&mut self, at: MonoNs) -> metrale_abstract_io::Result<PowerSample> {
        let name = || self.caps.source().0.clone();
        let mut buf = String::new();
        match self.lines.read_line(&mut buf) {
            Ok(0) => Err(error!(
                ErrorCode::PowerReadFailed,
                name(),
                "nvidia-smi exited"
            )),
            Ok(_) => {
                let line = parse_line(&buf)
                    .map_err(|why| error!(ErrorCode::PowerReadFailed, name(), why))?;
                to_sample(&self.caps, &line, at)
            }
            Err(e) => Err(error!(ErrorCode::PowerReadFailed, name(), e.to_string())),
        }
    }
}

impl Drop for NvidiaSmiProbe {
    fn drop(&mut self) {
        // 2026-10-22: A child that already exited needs only reaping.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[cfg(test)]
#[path = "nvidia_smi_tests.rs"]
mod tests;
