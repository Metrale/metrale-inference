// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: `kernel_energy`: the GPU-free half of per-kernel energy (DESIGN-kernel-energy.md).
//!
//! - `kernel_energy recompute <trace>`: the row's energy lines from the trace alone (HPAR: the
//!   same trace, the same bytes);
//! - `kernel_energy write --summary <harness JSON> --measurements <toml> --families
//!   <KERNEL_FAMILIES.toml> --kernel --file --hardware --model --regime --source --notes
//!   --peak-class --energy-floor-class`: recompute the harness's trace (its hash must match the
//!   summary's), add the time fields and floors, and append the `[[m]]` row (a duplicate
//!   `<kernel> @ <regime>` is refused);
//! - `kernel_energy cost --j-per-tok <mean,se,df> --tok-per-s <mean,se,df>
//!   --electricity-usd-per-kwh --hardware-usd --lifetime-hours --utilization --level --reps
//!   --seed`: $/Mtok, electricity and amortized hardware, delta method and bootstrap. Every input
//!   is required (PCND: no default price, lifetime or utilization); cost is printed, never stored.
//!
//! Owner: bench, kernel-energy.
//! Invariants: none beyond the library functions it calls.

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};
use metrale_bench::benchmarks::kernel_energy::{recompute, writer};
use metrale_power::cost::{CostInputs, cost};
use metrale_stats::interval::Summary;

fn flags(args: &[String]) -> Result<BTreeMap<String, String>> {
    let mut out = BTreeMap::new();
    let mut it = args.iter();
    while let Some(k) = it.next() {
        let key = k
            .strip_prefix("--")
            .with_context(|| format!("expected --flag, got {k}"))?;
        let v = it
            .next()
            .with_context(|| format!("--{key} needs a value"))?;
        out.insert(key.to_string(), v.clone());
    }
    Ok(out)
}

fn need<'a>(f: &'a BTreeMap<String, String>, k: &str) -> Result<&'a str> {
    f.get(k)
        .map(String::as_str)
        .with_context(|| format!("--{k} is required (no default)"))
}

fn summary(s: &str) -> Result<Summary> {
    let v: Vec<f64> = s
        .split(',')
        .map(str::parse)
        .collect::<Result<_, _>>()
        .context("expected mean,se,df")?;
    let [mean, se, df] = v.as_slice() else {
        bail!("expected mean,se,df")
    };
    Ok(Summary {
        mean: *mean,
        se: *se,
        df: *df,
    })
}

fn write(f: &BTreeMap<String, String>) -> Result<()> {
    let summary: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(need(f, "summary")?)?)?;
    let trace = summary["trace"].as_str().context("summary has no trace")?;
    let (sha, _) = recompute(trace)?;
    if summary["trace_sha256"].as_str() != Some(sha.as_str()) {
        bail!(
            "trace {trace} hashes to {sha}, not the summary's {}",
            summary["trace_sha256"]
        );
    }
    let text = std::fs::read_to_string(trace)?;
    let t = metrale_power::trace::PowerTrace::decode(&text).map_err(|e| anyhow::anyhow!("{e}"))?;
    let row = metrale_power::derive::derive(&t, &sha)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .row;
    let num = |k: &str| {
        summary[k]
            .as_f64()
            .with_context(|| format!("summary has no {k}"))
    };
    let base = writer::BaseRow {
        kernel: need(f, "kernel")?.to_string(),
        file: need(f, "file")?.to_string(),
        hardware: need(f, "hardware")?.to_string(),
        model: need(f, "model")?.to_string(),
        regime: need(f, "regime")?.to_string(),
        time_us: num("time_us")?,
        bytes: num("bytes")? as u64,
        flops: num("flops")? as u64,
        source: need(f, "source")?.to_string(),
        notes: need(f, "notes")?.to_string(),
    };
    let class = need(f, "energy-floor-class")?;
    let roof = writer::roofline(
        &std::fs::read_to_string(need(f, "families")?)?,
        need(f, "peak-class")?,
        class,
    )?;
    let block = writer::render(&base, &roof, &row, Some(class))?;
    let path = need(f, "measurements")?;
    let out = writer::append(&std::fs::read_to_string(path)?, &block)?;
    std::fs::write(path, out)?;
    print!("{block}");
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some((cmd, rest)) = args.split_first() else {
        bail!(
            "usage: kernel_energy recompute <trace> | write --summary ... | cost --j-per-tok ..."
        );
    };
    match cmd.as_str() {
        "recompute" => {
            let [trace] = rest else {
                bail!("usage: kernel_energy recompute <trace>")
            };
            let (sha, lines) = recompute(trace)?;
            println!("# trace_sha256 {sha}");
            for l in lines {
                println!("{l}");
            }
        }
        "write" => write(&flags(rest)?)?,
        "cost" => {
            let f = flags(rest)?;
            let opt = |k: &str| {
                f.get(k)
                    .map(|v| v.parse::<f64>())
                    .transpose()
                    .with_context(|| format!("--{k}"))
            };
            let inputs = CostInputs {
                electricity_usd_per_kwh: opt("electricity-usd-per-kwh")?,
                hardware_usd: opt("hardware-usd")?,
                lifetime_hours: opt("lifetime-hours")?,
                utilization: opt("utilization")?,
            };
            let c = cost(
                summary(need(&f, "j-per-tok")?)?,
                summary(need(&f, "tok-per-s")?)?,
                &inputs,
                need(&f, "level")?.parse()?,
                need(&f, "reps")?.parse()?,
                need(&f, "seed")?.parse()?,
            )
            .map_err(|e| anyhow::anyhow!("{e}"))?;
            println!("{c:#?}");
        }
        other => bail!("unknown command {other}"),
    }
    Ok(())
}
