// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: [`IoReportProbe`]: Apple silicon energy from IOReport's "Energy Model" group,
//! read without root (`powermetrics` needs root and is not used; declared). Each read takes a
//! sample, forms the delta against the previous one, and adds each mapped channel's energy into
//! a per-rail running total in nJ, exposed as a cumulative counter (`unit_j` 1e-9) so the
//! attribution's counter rules apply unchanged.
//!
//! Channel to rail is a declared table ([`CHANNELS`]): `CPU Energy` is `CpuPackage`, `GPU Energy`
//! is `GpuCore`, `DRAM*` is `Dram`; `ANE*` has no rail in the port and is skipped, declared.
//! Rails whose channel is absent at open are declared unavailable.
//!
//! `unsafe` is allowed in this module only: IOReport and CoreFoundation are C APIs with no safe
//! binding. Every block states why it is sound.
//!
//! Owner: metrale-io-host-power.
//! Invariants: every CF object this probe creates is released exactly once (drop releases the
//! subscription's channel dictionary and the previous sample).
#![allow(unsafe_code)]

use std::collections::BTreeMap;
use std::ffi::{CString, c_char, c_void};

use metrale_abstract_io::MonoNs;
use metrale_abstract_io::power::{
    CounterSpec, EnergyCount, Field, PowerCapabilities, PowerProbe, PowerSample, Rail, SourceId,
    SourceKind,
};
use metrale_error::{ErrorCode, error};

type CFTypeRef = *const c_void;
type CFStringRef = *const c_void;
type CFDictionaryRef = *const c_void;
type CFMutableDictionaryRef = *mut c_void;
type CFArrayRef = *const c_void;
type CFIndex = isize;

const UTF8: u32 = 0x0800_0100;

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFStringCreateWithCString(alloc: CFTypeRef, s: *const c_char, enc: u32) -> CFStringRef;
    fn CFStringGetCString(s: CFStringRef, buf: *mut c_char, len: CFIndex, enc: u32) -> bool;
    fn CFDictionaryGetValue(d: CFDictionaryRef, key: CFTypeRef) -> CFTypeRef;
    fn CFDictionaryCreateMutableCopy(
        alloc: CFTypeRef,
        cap: CFIndex,
        d: CFDictionaryRef,
    ) -> CFMutableDictionaryRef;
    fn CFArrayGetCount(a: CFArrayRef) -> CFIndex;
    fn CFArrayGetValueAtIndex(a: CFArrayRef, i: CFIndex) -> CFTypeRef;
    fn CFRelease(o: CFTypeRef);
}

#[link(name = "IOReport")]
unsafe extern "C" {
    fn IOReportCopyChannelsInGroup(
        group: CFStringRef,
        sub: CFStringRef,
        a: u64,
        b: u64,
        c: u64,
    ) -> CFDictionaryRef;
    fn IOReportCreateSubscription(
        a: *const c_void,
        desired: CFMutableDictionaryRef,
        subbed: *mut CFMutableDictionaryRef,
        id: u64,
        b: CFTypeRef,
    ) -> *const c_void;
    fn IOReportCreateSamples(
        sub: *const c_void,
        subbed: CFMutableDictionaryRef,
        a: CFTypeRef,
    ) -> CFDictionaryRef;
    fn IOReportCreateSamplesDelta(
        prev: CFDictionaryRef,
        cur: CFDictionaryRef,
        a: CFTypeRef,
    ) -> CFDictionaryRef;
    fn IOReportChannelGetChannelName(ch: CFDictionaryRef) -> CFStringRef;
    fn IOReportChannelGetUnitLabel(ch: CFDictionaryRef) -> CFStringRef;
    fn IOReportSimpleGetIntegerValue(ch: CFDictionaryRef, idx: i32) -> i64;
}

/// 2026-10-22: Channel-name rule to rail: (prefix, exact?, rail or `None` = skipped, declared).
pub const CHANNELS: [(&str, bool, Option<Rail>); 4] = [
    ("CPU Energy", true, Some(Rail::CpuPackage)),
    ("GPU Energy", true, Some(Rail::GpuCore)),
    ("DRAM", false, Some(Rail::Dram)),
    ("ANE", false, None),
];

/// 2026-10-22: The rail a channel name maps to; `None` when unmapped or skipped.
#[must_use]
pub fn rail_of(name: &str) -> Option<Rail> {
    CHANNELS
        .iter()
        .find(|(p, exact, _)| {
            if *exact {
                name == *p
            } else {
                name.starts_with(p)
            }
        })
        .and_then(|(_, _, r)| *r)
}

/// 2026-10-22: nJ per one unit of a label IOReport prints; `None` for a label it does not.
#[must_use]
pub fn nj_per(unit: &str) -> Option<i64> {
    match unit.trim() {
        "nJ" => Some(1),
        "uJ" | "µJ" => Some(1_000),
        "mJ" => Some(1_000_000),
        "J" => Some(1_000_000_000),
        _ => None,
    }
}

fn cfstr(s: &str) -> CFStringRef {
    let c = CString::new(s).expect("no NUL in a literal");
    // 2026-10-22: SAFETY: a valid NUL-terminated UTF-8 string; the caller releases the result.
    unsafe { CFStringCreateWithCString(std::ptr::null(), c.as_ptr(), UTF8) }
}

fn string(s: CFStringRef) -> String {
    if s.is_null() {
        return String::new();
    }
    let mut buf = [0 as c_char; 256];
    // 2026-10-22: SAFETY: `s` is a CFString owned by the sample (get rule: not released here);
    // the buffer length is passed.
    let ok = unsafe { CFStringGetCString(s, buf.as_mut_ptr(), 256, UTF8) };
    if !ok {
        return String::new();
    }
    // 2026-10-22: SAFETY: CFStringGetCString wrote a NUL-terminated string into `buf`.
    unsafe { std::ffi::CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

/// 2026-10-22: (channel name, unit label, value) of every channel in `samples`.
fn channels(samples: CFDictionaryRef) -> Vec<(String, String, i64)> {
    let key = cfstr("IOReportChannels");
    // 2026-10-22: SAFETY: `samples` is a live dictionary; the get rule leaves the array owned by
    // it; the key is released after.
    let arr = unsafe { CFDictionaryGetValue(samples, key) };
    // 2026-10-22: SAFETY: created above, released once.
    unsafe { CFRelease(key) };
    if arr.is_null() {
        return Vec::new();
    }
    // 2026-10-22: SAFETY: `arr` is a CFArray of channel dictionaries owned by `samples`.
    let n = unsafe { CFArrayGetCount(arr) };
    (0..n)
        .map(|i| {
            // 2026-10-22: SAFETY: `i` < count; the channel and its strings are owned by `arr`.
            unsafe {
                let ch = CFArrayGetValueAtIndex(arr, i);
                (
                    string(IOReportChannelGetChannelName(ch)),
                    string(IOReportChannelGetUnitLabel(ch)),
                    IOReportSimpleGetIntegerValue(ch, 0),
                )
            }
        })
        .collect()
}

/// 2026-10-22: An Energy Model subscription and its running per-rail totals.
pub struct IoReportProbe {
    caps: PowerCapabilities,
    sub: *const c_void,
    subbed: CFMutableDictionaryRef,
    prev: CFDictionaryRef,
    totals_nj: BTreeMap<Rail, u64>,
}

// 2026-10-22: SAFETY: the CF objects are owned by this probe alone and used on one thread at a
// time (it moves to its reader thread once).
unsafe impl Send for IoReportProbe {}

impl IoReportProbe {
    /// 2026-10-22: Subscribe to the "Energy Model" group; `update_period_ms` is the caller's read
    /// cadence (IOReport integrates between samples, so the counter updates at each read).
    pub fn open(update_period_ms: f64) -> metrale_abstract_io::Result<Self> {
        let name = "ioreport";
        let no = |why: &str| error!(ErrorCode::PowerSourceUnavailable, name, why.to_string());
        let group = cfstr("Energy Model");
        // 2026-10-22: SAFETY: documented-by-use IOReport calls (no public header); the copy is
        // released after the mutable copy is made; the group string is released once.
        let (sub, subbed) = unsafe {
            let chans = IOReportCopyChannelsInGroup(group, std::ptr::null(), 0, 0, 0);
            CFRelease(group);
            if chans.is_null() {
                return Err(no("IOReport has no Energy Model group"));
            }
            let desired = CFDictionaryCreateMutableCopy(std::ptr::null(), 0, chans);
            CFRelease(chans);
            let mut subbed: CFMutableDictionaryRef = std::ptr::null_mut();
            let sub = IOReportCreateSubscription(
                std::ptr::null(),
                desired,
                &mut subbed,
                0,
                std::ptr::null(),
            );
            CFRelease(desired);
            if sub.is_null() || subbed.is_null() {
                return Err(no("IOReportCreateSubscription failed"));
            }
            (sub, subbed)
        };
        // 2026-10-22: SAFETY: a live subscription and its channel dictionary.
        let prev = unsafe { IOReportCreateSamples(sub, subbed, std::ptr::null()) };
        if prev.is_null() {
            return Err(no("IOReportCreateSamples failed"));
        }
        let seen: Vec<Rail> = channels(prev)
            .iter()
            .filter_map(|(n, _, _)| rail_of(n))
            .collect();
        let mut b = PowerCapabilities::builder(
            SourceId(name.to_string()),
            SourceKind::AppleIoReport,
            "IOReport Energy Model (no root; powermetrics not used)".to_string(),
        );
        for r in [Rail::CpuPackage, Rail::GpuCore, Rail::Dram] {
            b = if seen.contains(&r) {
                b.counter(
                    r,
                    CounterSpec {
                        unit_j: 1e-9,
                        update_period_ms,
                    },
                )
            } else {
                b.unavailable(
                    Field::Energy(r),
                    "no Energy Model channel for this rail on this Mac",
                )
            };
        }
        let caps = b
            .rest_unavailable(
                &Field::universe(),
                "IOReport Energy Model reports channel energy only (ANE has no rail; clocks, \
                 temperatures and instant power need powermetrics, which needs root)",
            )
            .build(&Field::universe())?;
        let totals_nj = seen.into_iter().map(|r| (r, 0)).collect();
        Ok(Self {
            caps,
            sub,
            subbed,
            prev,
            totals_nj,
        })
    }
}

impl PowerProbe for IoReportProbe {
    fn capabilities(&self) -> &PowerCapabilities {
        &self.caps
    }

    fn read(&mut self, at: MonoNs) -> metrale_abstract_io::Result<PowerSample> {
        // 2026-10-22: SAFETY: live subscription; `prev` is ours; the delta is released below.
        let cur = unsafe { IOReportCreateSamples(self.sub, self.subbed, std::ptr::null()) };
        if cur.is_null() {
            return Err(error!(
                ErrorCode::PowerReadFailed,
                "ioreport", "IOReportCreateSamples failed"
            ));
        }
        // 2026-10-22: SAFETY: both samples are live dictionaries we own.
        let delta = unsafe { IOReportCreateSamplesDelta(self.prev, cur, std::ptr::null()) };
        // 2026-10-22: SAFETY: `prev` was created by us and is replaced by `cur`.
        unsafe { CFRelease(self.prev) };
        self.prev = cur;
        if delta.is_null() {
            return Err(error!(
                ErrorCode::PowerReadFailed,
                "ioreport", "IOReportCreateSamplesDelta failed"
            ));
        }
        let chans = channels(delta);
        // 2026-10-22: SAFETY: created above, released once.
        unsafe { CFRelease(delta) };
        for (n, unit, v) in chans {
            let Some(r) = rail_of(&n) else { continue };
            let Some(scale) = nj_per(&unit) else {
                return Err(error!(
                    ErrorCode::PowerReadFailed,
                    "ioreport",
                    format!("channel {n} unit {unit:?}")
                ));
            };
            let add = u64::try_from(v.saturating_mul(scale)).unwrap_or(0);
            if let Some(t) = self.totals_nj.get_mut(&r) {
                *t = t.wrapping_add(add);
            }
        }
        let mut s = PowerSample::at(at);
        for (r, t) in &self.totals_nj {
            s.energy.insert(*r, EnergyCount(*t));
        }
        Ok(s)
    }
}

impl Drop for IoReportProbe {
    fn drop(&mut self) {
        // 2026-10-22: SAFETY: both were created by this probe and are released once. The
        // subscription object has no documented release; it lives until the process ends.
        unsafe {
            CFRelease(self.prev);
            CFRelease(self.subbed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_names_map_to_rails_and_ane_is_skipped() {
        assert_eq!(rail_of("CPU Energy"), Some(Rail::CpuPackage));
        assert_eq!(rail_of("GPU Energy"), Some(Rail::GpuCore));
        assert_eq!(rail_of("DRAM Energy"), Some(Rail::Dram));
        assert_eq!(rail_of("ANE Energy"), None);
        assert_eq!(rail_of("EACC_CPU"), None);
        assert_eq!(nj_per("mJ"), Some(1_000_000));
        assert_eq!(nj_per("W"), None);
    }

    /// 2026-10-22: On a Mac, read-only, no root.
    #[test]
    #[ignore]
    fn live_ioreport() {
        let mut p = IoReportProbe::open(500.0).unwrap();
        println!("caps {:?}", p.capabilities());
        let _ = p.read(MonoNs(0)).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(500));
        println!("{:?}", p.read(MonoNs(1)).unwrap());
    }
}
