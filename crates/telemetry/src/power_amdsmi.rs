// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: [`AmdSmiProbe`]: the AMD SMI energy accumulator of one AMD processor as a
//! `PowerProbe` (gfx1151 / Strix Halo, MI-series). `libamd_smi.so` is opened at run time
//! (dlopen), so a host without ROCm gets `PowerSourceUnavailable`, never a load failure. Built
//! only with the `amdsmi` feature.
//!
//! What it reports: `amdsmi_get_energy_count` (a cumulative accumulator whose unit is the
//! `counter_resolution` the library returns at open, µJ per count, declared as the counter's
//! `unit_j`) on the rail the caller declares (`Rail::Soc` on an APU: the socket counter covers
//! CPU and GPU, which are not separable, so `GpuCore` alone is declared unavailable), and the
//! edge temperature (`amdsmi_get_temp_metric`, °C) when it reads in range at open.
//!
//! Declared unavailable, with reasons: socket power (`amdsmi_get_power_info` fills a struct whose
//! layout changed across amdsmi releases; reading it blind would guess an ABI — use the hwmon
//! `PPT` channel, `metrale-io-host-power::hwmon`), clocks (the same struct-ABI reason), throttle,
//! utilization, the library timestamp (its unit differs across releases).
//!
//! Owner: telemetry.
//! Invariants: `amdsmi_shut_down` runs once on drop, balancing a successful init.
#![cfg(feature = "amdsmi")]

use std::ffi::{c_float, c_int, c_void};

use libloading::Library;
use metrale_abstract_io::MonoNs;
use metrale_abstract_io::power::{
    CounterSpec, EnergyCount, Field, PowerCapabilities, PowerProbe, PowerSample, Rail, SourceId,
    SourceKind,
};
use metrale_error::{ErrorCode, error};

/// 2026-10-22: `amdsmi_status_t` success.
const AMDSMI_STATUS_SUCCESS: u32 = 0;
/// 2026-10-22: `AMDSMI_INIT_AMD_GPUS` (`1 << 1`).
const AMDSMI_INIT_AMD_GPUS: u64 = 1 << 1;
/// 2026-10-22: `AMDSMI_TEMPERATURE_TYPE_EDGE` and `AMDSMI_TEMP_CURRENT`.
const TEMP_EDGE: c_int = 0;
const TEMP_CURRENT: c_int = 0;
/// 2026-10-22: The library names tried, in order.
pub const LIBRARY_NAMES: [&str; 2] = ["libamd_smi.so", "/opt/rocm/lib/libamd_smi.so"];

type Handle = *mut c_void;
type FnInit = unsafe extern "C" fn(u64) -> u32;
type FnVoid = unsafe extern "C" fn() -> u32;
type FnHandles = unsafe extern "C" fn(*mut u32, *mut Handle) -> u32;
type FnProcHandles = unsafe extern "C" fn(Handle, *mut u32, *mut Handle) -> u32;
type FnEnergy = unsafe extern "C" fn(Handle, *mut u64, *mut c_float, *mut u64) -> u32;
type FnTemp = unsafe extern "C" fn(Handle, c_int, c_int, *mut i64) -> u32;

/// 2026-10-22: One AMD processor's energy accumulator.
pub struct AmdSmiProbe {
    caps: PowerCapabilities,
    rail: Rail,
    handle: Handle,
    energy: FnEnergy,
    temp: Option<FnTemp>,
    shut_down: FnVoid,
    _lib: Library,
}

// 2026-10-22: SAFETY: the handle is an opaque id owned by the library, which documents its
// queries as thread-safe; the probe is moved to one reader thread and used only there.
unsafe impl Send for AmdSmiProbe {}

fn unavailable(name: &str, why: String) -> metrale_error::Error {
    error!(ErrorCode::PowerSourceUnavailable, name.to_string(), why)
}

fn sym<T: Copy>(lib: &Library, name: &str) -> Option<T> {
    let mut s = name.as_bytes().to_vec();
    s.push(0);
    // 2026-10-22: SAFETY: every call site names the symbol with its amdsmi.h signature.
    unsafe { lib.get::<T>(s.as_slice()).ok().map(|f| *f) }
}

impl AmdSmiProbe {
    /// 2026-10-22: Open processor `index` (in socket order) and declare its counter on `rail`
    /// (explicit: `Rail::Soc` on an APU).
    pub fn open(index: u32, rail: Rail) -> metrale_abstract_io::Result<Self> {
        let name = format!("amdsmi{index}");
        let mut last = String::from("no candidate names");
        let lib = LIBRARY_NAMES
            .iter()
            // 2026-10-22: SAFETY: loading runs the library's initialisers; amdsmi itself is
            // initialised only by `amdsmi_init` below.
            .find_map(|n| match unsafe { Library::new(*n) } {
                Ok(l) => Some(l),
                Err(e) => {
                    last = format!("{n}: {e}");
                    None
                }
            })
            .ok_or_else(|| unavailable(&name, last))?;
        let need = |s: &str| unavailable(&name, format!("libamd_smi lacks {s}"));
        let init: FnInit = sym(&lib, "amdsmi_init").ok_or_else(|| need("amdsmi_init"))?;
        let shut_down: FnVoid =
            sym(&lib, "amdsmi_shut_down").ok_or_else(|| need("amdsmi_shut_down"))?;
        let sockets: FnHandles = sym(&lib, "amdsmi_get_socket_handles")
            .ok_or_else(|| need("amdsmi_get_socket_handles"))?;
        let procs: FnProcHandles = sym(&lib, "amdsmi_get_processor_handles")
            .ok_or_else(|| need("amdsmi_get_processor_handles"))?;
        let energy: FnEnergy =
            sym(&lib, "amdsmi_get_energy_count").ok_or_else(|| need("amdsmi_get_energy_count"))?;
        let temp: Option<FnTemp> = sym(&lib, "amdsmi_get_temp_metric");
        // 2026-10-22: SAFETY: amdsmi_init(uint64_t) per amdsmi.h.
        let rc = unsafe { init(AMDSMI_INIT_AMD_GPUS) };
        if rc != AMDSMI_STATUS_SUCCESS {
            return Err(unavailable(&name, format!("amdsmi_init returned {rc}")));
        }
        let fail = |why: String| {
            // 2026-10-22: SAFETY: balances the successful init above.
            unsafe { shut_down() };
            unavailable(&name, why)
        };
        let handle = match processor(sockets, procs, index) {
            Ok(h) => h,
            Err(why) => return Err(fail(why)),
        };
        let (mut acc, mut res, mut ts) = (0u64, 0.0f32, 0u64);
        // 2026-10-22: SAFETY: valid out-pointers; handle from the library.
        let rc = unsafe { energy(handle, &mut acc, &mut res, &mut ts) };
        if rc != AMDSMI_STATUS_SUCCESS || res.is_nan() || res <= 0.0 {
            return Err(fail(format!(
                "amdsmi_get_energy_count returned {rc}, resolution {res}"
            )));
        }
        let mut b = PowerCapabilities::builder(
            SourceId(name.clone()),
            SourceKind::AmdSmi,
            format!("libamd_smi; processor {index}"),
        )
        .counter(
            rail,
            CounterSpec {
                unit_j: f64::from(res) * 1e-6,
                // 2026-10-22: The SMU refreshes the accumulator every 1 ms (amdsmi docs).
                update_period_ms: 1.0,
            },
        );
        if rail == Rail::Soc {
            b = b.unavailable(
                Field::Energy(Rail::GpuCore),
                "APU socket: CPU and GPU not separable",
            );
        }
        let temp =
            temp.filter(|t| read_temp(*t, handle).is_some_and(|c| (-50.0..150.0).contains(&c)));
        b = match temp {
            Some(_) => b.field(Field::GpuTemp),
            None => b.unavailable(
                Field::GpuTemp,
                "amdsmi_get_temp_metric absent or out of the °C range at open",
            ),
        };
        let caps = b
            .unavailable(
                Field::AveragePower(rail),
                "amdsmi_power_info_t's layout differs across amdsmi releases; use hwmon PPT",
            )
            .unavailable(
                Field::DriverTimestamp,
                "the amdsmi energy timestamp unit differs across releases",
            )
            .rest_unavailable(&Field::universe(), "not read by this amdsmi probe")
            .build(&Field::universe());
        let caps = match caps {
            Ok(c) => c,
            Err(e) => {
                // 2026-10-22: SAFETY: balances the successful init above.
                unsafe { shut_down() };
                return Err(e);
            }
        };
        Ok(Self {
            caps,
            rail,
            handle,
            energy,
            temp,
            shut_down,
            _lib: lib,
        })
    }
}

fn processor(sockets: FnHandles, procs: FnProcHandles, index: u32) -> Result<Handle, String> {
    let mut n = 0u32;
    // 2026-10-22: SAFETY: a null array asks for the count.
    if unsafe { sockets(&mut n, std::ptr::null_mut()) } != AMDSMI_STATUS_SUCCESS || n == 0 {
        return Err("no AMD socket".to_string());
    }
    let mut socks = vec![std::ptr::null_mut(); n as usize];
    // 2026-10-22: SAFETY: `socks` holds `n` handles.
    if unsafe { sockets(&mut n, socks.as_mut_ptr()) } != AMDSMI_STATUS_SUCCESS {
        return Err("amdsmi_get_socket_handles failed".to_string());
    }
    let mut all = Vec::new();
    for s in socks.into_iter().take(n as usize) {
        let mut m = 0u32;
        // 2026-10-22: SAFETY: as above, count first then fill.
        if unsafe { procs(s, &mut m, std::ptr::null_mut()) } != AMDSMI_STATUS_SUCCESS {
            continue;
        }
        let mut hs = vec![std::ptr::null_mut(); m as usize];
        // 2026-10-22: SAFETY: `hs` holds `m` handles.
        if unsafe { procs(s, &mut m, hs.as_mut_ptr()) } == AMDSMI_STATUS_SUCCESS {
            all.extend(hs.into_iter().take(m as usize));
        }
    }
    all.get(index as usize)
        .copied()
        .ok_or_else(|| format!("no AMD processor {index} ({} found)", all.len()))
}

fn read_temp(f: FnTemp, h: Handle) -> Option<f64> {
    let mut t = 0i64;
    // 2026-10-22: SAFETY: valid out-pointer, handle from the library.
    (unsafe { f(h, TEMP_EDGE, TEMP_CURRENT, &mut t) } == AMDSMI_STATUS_SUCCESS).then(|| {
        #[allow(clippy::cast_precision_loss)]
        let c = t as f64;
        c
    })
}

impl PowerProbe for AmdSmiProbe {
    fn capabilities(&self) -> &PowerCapabilities {
        &self.caps
    }

    fn read(&mut self, at: MonoNs) -> metrale_abstract_io::Result<PowerSample> {
        let (mut acc, mut res, mut ts) = (0u64, 0.0f32, 0u64);
        // 2026-10-22: SAFETY: valid out-pointers; handle from the library.
        let rc = unsafe { (self.energy)(self.handle, &mut acc, &mut res, &mut ts) };
        if rc != AMDSMI_STATUS_SUCCESS {
            return Err(error!(
                ErrorCode::PowerReadFailed,
                self.caps.source().0.clone(),
                format!("amdsmi_get_energy_count returned {rc}")
            ));
        }
        let mut s = PowerSample::at(at);
        s.energy.insert(self.rail, EnergyCount(acc));
        if let Some(t) = self.temp {
            s.temps.gpu_c = read_temp(t, self.handle);
        }
        Ok(s)
    }
}

impl Drop for AmdSmiProbe {
    fn drop(&mut self) {
        // 2026-10-22: SAFETY: balances the successful init in `open`; a failed shutdown leaves
        // nothing this side can release.
        let _ = unsafe { (self.shut_down)() };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-10-22: On a host without ROCm the probe is unavailable, never a load failure.
    #[test]
    fn a_host_without_amdsmi_reports_unavailable() {
        if let Err(e) = AmdSmiProbe::open(0, Rail::Soc) {
            assert_eq!(e.code(), ErrorCode::PowerSourceUnavailable);
        }
    }

    /// 2026-10-22: Read-only on an AMD host: `cargo test -p metrale-telemetry --features amdsmi
    /// -- --ignored live_amdsmi --nocapture`.
    #[test]
    #[ignore]
    fn live_amdsmi() {
        let mut p = AmdSmiProbe::open(0, Rail::Soc).unwrap();
        println!("caps {:?}", p.capabilities());
        let a = p.read(MonoNs(0)).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(500));
        let b = p.read(MonoNs(1)).unwrap();
        let unit = p.capabilities().counter(Rail::Soc).unwrap().unit_j;
        #[allow(clippy::cast_precision_loss)]
        let w = (b.energy[&Rail::Soc].0 - a.energy[&Rail::Soc].0) as f64 * unit / 0.5;
        println!("{a:?}\n{b:?}\n~{w:.2} W");
    }
}
