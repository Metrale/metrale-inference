// SPDX-License-Identifier: MIT OR Apache-2.0

//! External baseline compressors driven through their CLIs (when installed), with wall-clock
//! timing and a round-trip check, plus the CSV row type shared with the kzx measurements.

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

/// One measured (class, mode, tool) cell.
#[derive(Clone, Debug)]
pub struct Row {
    pub class: String,
    pub mode: String,
    pub tool: String,
    pub files: usize,
    pub raw: u64,
    pub packed: u64,
    pub c_secs: f64,
    pub d_secs: f64,
}

impl Row {
    pub fn ratio(&self) -> f64 {
        self.raw as f64 / self.packed.max(1) as f64
    }
    pub fn c_mbps(&self) -> f64 {
        mbps(self.raw, self.c_secs)
    }
    pub fn d_mbps(&self) -> f64 {
        mbps(self.raw, self.d_secs)
    }
    pub fn csv_header() -> &'static str {
        "class,mode,tool,files,raw_bytes,packed_bytes,ratio,c_secs,d_secs,c_mbps,d_mbps"
    }
    pub fn csv(&self) -> String {
        format!(
            "{},{},{},{},{},{},{:.4},{:.4},{:.4},{:.3},{:.3}",
            self.class,
            self.mode,
            self.tool,
            self.files,
            self.raw,
            self.packed,
            self.ratio(),
            self.c_secs,
            self.d_secs,
            self.c_mbps(),
            self.d_mbps()
        )
    }
}

pub fn mbps(bytes: u64, secs: f64) -> f64 {
    if secs > 0.0 {
        bytes as f64 / 1e6 / secs
    } else {
        0.0
    }
}

/// A baseline: name, compress argv, decompress argv (both read stdin, write stdout).
pub struct Tool {
    pub name: &'static str,
    pub compress: Vec<String>,
    pub decompress: Vec<String>,
}

fn argv(parts: &[&str]) -> Vec<String> {
    parts.iter().map(|s| s.to_string()).collect()
}

fn have(program: &str) -> bool {
    Command::new(program)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok()
}

fn have_python_brotli() -> bool {
    Command::new("python3")
        .args(["-c", "import brotli"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// The baselines installed on this host.
pub fn available() -> Vec<Tool> {
    let mut out = Vec::new();
    if have("gzip") {
        out.push(Tool {
            name: "deflate-9",
            compress: argv(&["gzip", "-9", "-c"]),
            decompress: argv(&["gzip", "-dc"]),
        });
    }
    if have("bzip2") {
        out.push(Tool {
            name: "bzip2-9",
            compress: argv(&["bzip2", "-9", "-c"]),
            decompress: argv(&["bzip2", "-dc"]),
        });
    }
    if have("zstd") {
        out.push(Tool {
            name: "zstd-19-long",
            compress: argv(&["zstd", "-q", "-19", "--long", "-T1", "-c"]),
            decompress: argv(&["zstd", "-q", "-d", "--long=27", "-c"]),
        });
        out.push(Tool {
            name: "zstd-22-ultra-long27",
            compress: argv(&["zstd", "-q", "--ultra", "-22", "--long=27", "-T1", "-c"]),
            decompress: argv(&["zstd", "-q", "-d", "--long=27", "-c"]),
        });
    }
    if have("xz") {
        out.push(Tool {
            name: "xz-9e",
            compress: argv(&["xz", "-9e", "-T1", "-c"]),
            decompress: argv(&["xz", "-dc"]),
        });
    }
    if have("brotli") {
        out.push(Tool {
            name: "brotli-11",
            compress: argv(&["brotli", "-q", "11", "--large_window=24", "-c"]),
            decompress: argv(&["brotli", "-dc"]),
        });
    } else if have_python_brotli() {
        out.push(Tool {
            name: "brotli-11",
            compress: argv(&["python3", "-c", "import brotli,sys;sys.stdout.buffer.write(brotli.compress(sys.stdin.buffer.read(),quality=11,lgwin=24))"]),
            decompress: argv(&["python3", "-c", "import brotli,sys;sys.stdout.buffer.write(brotli.decompress(sys.stdin.buffer.read()))"]),
        });
    }
    out
}

fn run_pipe(argv: &[String], input: &Path, output: &Path) -> Option<f64> {
    let inf = std::fs::File::open(input).ok()?;
    let outf = std::fs::File::create(output).ok()?;
    let t0 = Instant::now();
    let status = Command::new(&argv[0])
        .args(&argv[1..])
        .stdin(inf)
        .stdout(outf)
        .stderr(Stdio::null())
        .status()
        .ok()?;
    let secs = t0.elapsed().as_secs_f64();
    status.success().then_some(secs)
}

/// Run `tool` over one input written to `scratch`; `None` if it failed or did not round-trip.
pub fn measure(tool: &Tool, data: &[u8], scratch: &Path, tag: &str) -> Option<(u64, f64, f64)> {
    let input = scratch.join(format!("{tag}.in"));
    let packed = scratch.join(format!("{tag}.{}", tool.name));
    let back = scratch.join(format!("{tag}.{}.rt", tool.name));
    std::fs::write(&input, data).ok()?;
    let c = run_pipe(&tool.compress, &input, &packed)?;
    let d = run_pipe(&tool.decompress, &packed, &back)?;
    let size = std::fs::metadata(&packed).ok()?.len();
    let rt = std::fs::read(&back).ok()?;
    let _ = std::fs::remove_file(&back);
    let _ = std::fs::remove_file(&packed);
    let _ = std::fs::remove_file(&input);
    (rt == data).then_some((size, c, d))
}

/// A scratch directory for the baseline files.
pub fn scratch_dir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("kzx-bench-{}", std::process::id()));
    let _ = std::fs::create_dir_all(&dir);
    dir
}

/// Print rows as a Markdown table.
pub fn print_table(rows: &[Row]) {
    println!(
        "| class | mode | tool | files | raw bytes | packed bytes | ratio | comp MB/s | decomp MB/s |"
    );
    println!("|---|---|---|---:|---:|---:|---:|---:|---:|");
    for r in rows {
        println!(
            "| {} | {} | {} | {} | {} | {} | {:.2} | {:.2} | {:.2} |",
            r.class,
            r.mode,
            r.tool,
            r.files,
            r.raw,
            r.packed,
            r.ratio(),
            r.c_mbps(),
            r.d_mbps()
        );
    }
}
