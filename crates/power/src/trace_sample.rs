// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: One sample's `key=value;...` form in a power trace. Keys: `e.<rail>` raw counter,
//! `p.<rail>` instant W, `a.<rail>` average W, `l.<rail>` limit W, `sm`, `mem`, `gfx` MHz, `tg`,
//! `tm` °C, `thr` throttle mask, `ug`, `um` percent, `dts` driver µs.
//!
//! Owner: metrale-power.
//! Invariants: `decode_sample(encode_sample(s)) == s` (floats in shortest round-trip form).

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use metrale_abstract_io::MonoNs;
use metrale_abstract_io::power::{EnergyCount, PowerSample, Rail};

use crate::Result;
use crate::trace::{bad, num};

pub(crate) fn encode_sample(s: &PowerSample) -> String {
    let mut kv: Vec<String> = Vec::new();
    for (r, c) in &s.energy {
        kv.push(format!("e.{}={}", r.as_str(), c.0));
    }
    for (r, v) in &s.rails {
        let rail = r.as_str();
        if let Some(w) = v.instant_w {
            kv.push(format!("p.{rail}={w:?}"));
        }
        if let Some(w) = v.average_w {
            kv.push(format!("a.{rail}={w:?}"));
        }
        if let Some(w) = v.limit_w {
            kv.push(format!("l.{rail}={w:?}"));
        }
    }
    let ints = [
        ("sm", s.clocks.sm_mhz.map(u64::from)),
        ("mem", s.clocks.mem_mhz.map(u64::from)),
        ("gfx", s.clocks.gfx_mhz.map(u64::from)),
        ("thr", s.throttle),
        ("ug", s.util.gpu_pct.map(u64::from)),
        ("um", s.util.mem_pct.map(u64::from)),
        ("dts", s.driver_ts_us),
    ];
    for (k, v) in ints {
        if let Some(v) = v {
            kv.push(format!("{k}={v}"));
        }
    }
    for (k, v) in [("tg", s.temps.gpu_c), ("tm", s.temps.mem_c)] {
        if let Some(v) = v {
            kv.push(format!("{k}={v:?}"));
        }
    }
    kv.join(";")
}

pub(crate) fn decode_sample(at: MonoNs, kv: &str, n: usize) -> Result<PowerSample> {
    let mut s = PowerSample::at(at);
    for pair in kv.split(';').filter(|p| !p.is_empty()) {
        let (k, v) = pair
            .split_once('=')
            .ok_or_else(|| bad(n, &format!("`{pair}` is not key=value")))?;
        if let Some((p, rail)) = k.split_once('.') {
            let r = Rail::parse(rail).ok_or_else(|| bad(n, &format!("unknown rail {rail}")))?;
            match p {
                "e" => {
                    s.energy.insert(r, EnergyCount(num(v, n)?));
                }
                "p" => s.rails.entry(r).or_default().instant_w = Some(num(v, n)?),
                "a" => s.rails.entry(r).or_default().average_w = Some(num(v, n)?),
                "l" => s.rails.entry(r).or_default().limit_w = Some(num(v, n)?),
                _ => return Err(bad(n, &format!("unknown sample key {k}"))),
            }
            continue;
        }
        match k {
            "sm" => s.clocks.sm_mhz = Some(num(v, n)?),
            "mem" => s.clocks.mem_mhz = Some(num(v, n)?),
            "gfx" => s.clocks.gfx_mhz = Some(num(v, n)?),
            "thr" => s.throttle = Some(num(v, n)?),
            "ug" => s.util.gpu_pct = Some(num(v, n)?),
            "um" => s.util.mem_pct = Some(num(v, n)?),
            "dts" => s.driver_ts_us = Some(num(v, n)?),
            "tg" => s.temps.gpu_c = Some(num(v, n)?),
            "tm" => s.temps.mem_c = Some(num(v, n)?),
            _ => return Err(bad(n, &format!("unknown sample key {k}"))),
        }
    }
    Ok(s)
}
