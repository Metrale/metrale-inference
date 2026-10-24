// SPDX-License-Identifier: MIT OR Apache-2.0

//! The benchmark: kzx against the installed baseline compressors, per content class.
//!
//! ```text
//! cargo run --release --example bench -- [--corpus DIR] [--out results.csv] [--no-external] [--scale N]
//! ```
//!
//! Without `--corpus` it generates the synthetic corpus (`synth.rs`); with it, every file under
//! `DIR/{ptx,cubin,body,kbp}/` is a corpus member of that class. Each class is measured twice:
//! `per-file` (each file its own stream, sums reported) and `solid` (all files concatenated, one
//! stream), except `kbp`, where kzx compresses the container's sections as separate blocks
//! (`sections`) and the baselines see the whole file. Every measurement is a verified round trip.

mod synth;
mod tools;

use std::path::{Path, PathBuf};
use std::time::Instant;
use tools::{Row, Tool};

const CLASSES: [&str; 4] = ["ptx", "cubin", "body", "kbp"];

struct Args {
    corpus: Option<PathBuf>,
    out: Option<PathBuf>,
    external: bool,
    scale: usize,
}

fn parse_args() -> Args {
    let mut a = Args {
        corpus: None,
        out: None,
        external: true,
        scale: 4,
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        match k.as_str() {
            "--corpus" => a.corpus = it.next().map(PathBuf::from),
            "--out" => a.out = it.next().map(PathBuf::from),
            "--no-external" => a.external = false,
            "--scale" => a.scale = it.next().and_then(|s| s.parse().ok()).unwrap_or(4),
            _ => {
                eprintln!("usage: bench [--corpus DIR] [--out CSV] [--no-external] [--scale N]");
                std::process::exit(2);
            }
        }
    }
    a
}

fn load_corpus(dir: &Path) -> Vec<(&'static str, Vec<Vec<u8>>)> {
    let mut out = Vec::new();
    for class in CLASSES {
        let sub = dir.join(class);
        let Ok(rd) = std::fs::read_dir(&sub) else {
            continue;
        };
        let mut paths: Vec<PathBuf> = rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.is_file())
            .collect();
        paths.sort();
        let files: Vec<Vec<u8>> = paths.iter().filter_map(|p| std::fs::read(p).ok()).collect();
        if !files.is_empty() {
            out.push((class, files));
        }
    }
    out
}

/// The `.kbp` sections (header, body, locator, payload, signature), if `data` is a container.
fn kbp_sections(data: &[u8]) -> Option<Vec<&[u8]>> {
    if data.len() < 72 || &data[..4] != b"MKBP" {
        return None;
    }
    let u64_at = |at: usize| {
        u64::from_le_bytes(data[at..at + 8].try_into().ok()?)
            .try_into()
            .ok()
    };
    let body: usize = u64_at(8)?;
    let locator: usize = u64_at(48)?;
    let payload: usize = u64_at(56)?;
    let sig: usize = u64_at(64)?;
    let mut at = 72usize;
    let mut parts = vec![&data[..72]];
    for len in [body, locator, payload, sig] {
        let end = at.checked_add(len)?;
        parts.push(data.get(at..end)?);
        at = end;
    }
    (at == data.len()).then_some(parts)
}

fn kzx_measure(inputs: &[&[u8]], sections: bool) -> (u64, f64, f64) {
    let limits = kzx::Limits {
        max_output: 1 << 40,
        max_memory: 8 << 30,
    };
    let mut packed_total = 0u64;
    let (mut c_secs, mut d_secs) = (0.0, 0.0);
    for &input in inputs {
        let t0 = Instant::now();
        let packed = match kbp_sections(input).filter(|_| sections) {
            Some(parts) => kzx::compress_blocks(&parts),
            None => kzx::compress(input),
        };
        c_secs += t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let back = kzx::decompress(&packed, &limits).expect("kzx decompress");
        d_secs += t1.elapsed().as_secs_f64();
        assert!(back == input, "kzx round trip failed");
        packed_total += packed.len() as u64;
    }
    (packed_total, c_secs, d_secs)
}

fn rows_for(
    class: &str,
    mode: &str,
    inputs: &[&[u8]],
    tools: &[Tool],
    scratch: &Path,
    rows: &mut Vec<Row>,
) {
    let raw: u64 = inputs.iter().map(|i| i.len() as u64).sum();
    let (packed, c, d) = kzx_measure(inputs, mode == "sections");
    rows.push(Row {
        class: class.into(),
        mode: mode.into(),
        tool: "kzx".into(),
        files: inputs.len(),
        raw,
        packed,
        c_secs: c,
        d_secs: d,
    });
    eprintln!(
        "  {class}/{mode} kzx: {raw} -> {packed} ({:.2}x) c {:.2}s d {:.2}s",
        raw as f64 / packed.max(1) as f64,
        c,
        d
    );
    for t in tools {
        let (mut p, mut cs, mut ds) = (0u64, 0.0, 0.0);
        let mut ok = true;
        for (i, input) in inputs.iter().enumerate() {
            match tools::measure(t, input, scratch, &format!("{class}-{mode}-{i}")) {
                Ok((size, c, d)) => {
                    p += size;
                    cs += c;
                    ds += d;
                }
                Err(why) => {
                    eprintln!("  {class}/{mode} {} file {i}: {why}", t.name);
                    ok = false;
                }
            }
        }
        if ok {
            rows.push(Row {
                class: class.into(),
                mode: mode.into(),
                tool: t.name.into(),
                files: inputs.len(),
                raw,
                packed: p,
                c_secs: cs,
                d_secs: ds,
            });
            eprintln!(
                "  {class}/{mode} {}: {raw} -> {p} ({:.2}x) c {:.2}s d {:.2}s",
                t.name,
                raw as f64 / p.max(1) as f64,
                cs,
                ds
            );
        } else {
            eprintln!("  {class}/{mode} {}: failed", t.name);
        }
    }
}

fn main() {
    let args = parse_args();
    let corpus = match &args.corpus {
        Some(dir) => load_corpus(dir),
        None => synth::corpus(0x006b_7a78, args.scale),
    };
    let tools = if args.external {
        tools::available()
    } else {
        Vec::new()
    };
    eprintln!(
        "baselines: {:?}",
        tools.iter().map(|t| t.name).collect::<Vec<_>>()
    );
    let scratch = tools::scratch_dir();
    let mut rows = Vec::new();
    for (class, files) in &corpus {
        let refs: Vec<&[u8]> = files.iter().map(|f| f.as_slice()).collect();
        let total: usize = files.iter().map(|f| f.len()).sum();
        eprintln!("class {class}: {} files, {total} bytes", files.len());
        if *class == "kbp" {
            rows_for(class, "sections", &refs, &tools, &scratch, &mut rows);
            continue;
        }
        rows_for(class, "per-file", &refs, &tools, &scratch, &mut rows);
        if files.len() > 1 {
            let solid: Vec<u8> = files.concat();
            rows_for(
                class,
                "solid",
                &[solid.as_slice()],
                &tools,
                &scratch,
                &mut rows,
            );
        }
    }
    let _ = std::fs::remove_dir_all(&scratch);
    tools::print_table(&rows);
    if let Some(out) = &args.out {
        let mut csv = String::from(Row::csv_header());
        csv.push('\n');
        for r in &rows {
            csv.push_str(&r.csv());
            csv.push('\n');
        }
        std::fs::write(out, csv).expect("write csv");
        eprintln!("wrote {}", out.display());
    }
}
