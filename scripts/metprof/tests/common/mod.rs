// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: shared test helpers: an in-memory capture behind the
//! `TraceSource` port, an in-memory `Files` port, and a driver for the real
//! `cli::run`. Times are given in microseconds and stored as ns, as the
//! profiler stores them.

#![allow(dead_code)]

use std::collections::HashMap;

use metprof::cli::{Io, Outcome, run};
use metprof::core::capture::{CopyRow, KernelRow};
use metprof::core::error::{Error, Result};
use metprof::core::model::{ApiCall, Kernel, Launch};
use metprof::ports::{Files, Trace, TraceSource};

pub const US: f64 = 1_000.0;

/// The launch configuration of every synthetic kernel unless a test sets one.
pub const LAUNCH0: Launch = Launch {
    block: (128, 1, 1),
    registers_per_thread: 32,
    static_shared_bytes: 0,
    dynamic_shared_bytes: 0,
};

pub const FAMILIES_TOML: &str = r#"
schema = 1
coverage_min = 0.99
[api]
host_sync_regex = "^(cu|cuda)(Stream|Ctx|Event|Device)Synchronize"
[[rule]]
family = "comm"
group = "comm"
class = "comm"
name_regex = "^nccl"
[[rule]]
family = "gemm"
group = "dense"
class = "compute"
name_regex = "^gemm"
[[rule]]
family = "norm"
group = "misc"
class = "compute"
name_regex = "^norm"
"#;

pub const SHIPPED_FAMILIES: &[u8] = include_bytes!("../../config/families.toml");

fn ns(us: f64) -> i64 {
    (us * US).round() as i64
}

/// Optional fields of one synthetic kernel launch.
#[derive(Debug, Clone, Copy)]
pub struct K {
    pub stream: i64,
    pub device: i64,
    pub corr: Option<i64>,
    pub node: Option<i64>,
    pub grid: (i64, i64, i64),
    pub launch: Launch,
    /// The demangled name; `None` = the short name.
    pub demangled: Option<&'static str>,
}

pub const K0: K = K {
    stream: 7,
    device: 0,
    corr: None,
    node: None,
    grid: (1, 1, 1),
    launch: LAUNCH0,
    demangled: None,
};

#[derive(Debug, Clone)]
pub struct Capture {
    kernels: Vec<KernelRow>,
    copies: Vec<CopyRow>,
    calls: Vec<ApiCall>,
    corr: i64,
    node_column: bool,
}

impl Default for Capture {
    fn default() -> Self {
        Capture {
            kernels: Vec::new(),
            copies: Vec::new(),
            calls: Vec::new(),
            corr: 1000,
            node_column: true,
        }
    }
}

impl Capture {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn kernel(self, name: &str, start_us: f64, end_us: f64) -> Self {
        self.kernel_with(name, start_us, end_us, K0)
    }

    pub fn kernel_with(mut self, name: &str, start_us: f64, end_us: f64, k: K) -> Self {
        let corr = match k.corr {
            Some(c) => c,
            None => {
                self.corr += 1;
                self.corr
            }
        };
        self.kernels.push(KernelRow {
            device: k.device,
            kernel: Kernel {
                start: ns(start_us),
                end: ns(end_us),
                stream: k.stream,
                correlation: corr,
                graph_node: k.node,
                name: name.to_string(),
                demangled: k.demangled.unwrap_or(name).to_string(),
                grid: k.grid,
                launch: k.launch,
            },
        });
        self
    }

    pub fn copy(self, kind: i64, start_us: f64, end_us: f64) -> Self {
        self.copy_on(kind, start_us, end_us, 0)
    }

    pub fn copy_on(mut self, kind: i64, start_us: f64, end_us: f64, device: i64) -> Self {
        self.copies.push(CopyRow {
            device,
            start: ns(start_us),
            end: ns(end_us),
            kind,
            nbytes: 64,
        });
        self
    }

    pub fn call(self, name: &str, start_us: f64, end_us: f64) -> Self {
        self.call_corr(name, start_us, end_us, 0)
    }

    pub fn call_corr(mut self, name: &str, start_us: f64, end_us: f64, corr: i64) -> Self {
        self.calls.push(ApiCall {
            start: ns(start_us),
            end: ns(end_us),
            name: name.to_string(),
            correlation: corr,
        });
        self
    }

    /// An export taken without the graph-node column.
    pub fn without_node_column(mut self) -> Self {
        self.node_column = false;
        self
    }
}

impl Trace for Capture {
    fn kernel_rows(&self) -> Result<Vec<KernelRow>> {
        let mut rows = self.kernels.clone();
        if !self.node_column {
            for r in &mut rows {
                r.kernel.graph_node = None;
            }
        }
        Ok(rows)
    }

    fn copy_rows(&self) -> Result<Vec<CopyRow>> {
        Ok(self.copies.clone())
    }

    fn call_rows(&self) -> Result<Vec<ApiCall>> {
        Ok(self.calls.clone())
    }
}

/// Captures and files by path, in memory.
#[derive(Default)]
pub struct Mem {
    captures: HashMap<String, Capture>,
    files: HashMap<String, Vec<u8>>,
}

impl Mem {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn capture(mut self, path: &str, c: Capture) -> Self {
        self.captures.insert(path.to_string(), c);
        self
    }

    pub fn file(mut self, path: &str, text: &str) -> Self {
        self.files
            .insert(path.to_string(), text.as_bytes().to_vec());
        self
    }

    pub fn bytes(mut self, path: &str, b: &[u8]) -> Self {
        self.files.insert(path.to_string(), b.to_vec());
        self
    }

    /// Run the real CLI over these ports.
    pub fn run(&self, argv: &[&str]) -> Outcome {
        let argv: Vec<String> = argv.iter().map(|s| s.to_string()).collect();
        run(
            &argv,
            &Io {
                traces: self,
                files: self,
            },
        )
    }

    /// Run with `--json`, require success, and parse stdout.
    pub fn json(&self, argv: &[&str]) -> serde_json::Value {
        let mut full = vec!["--json"];
        full.extend_from_slice(argv);
        let out = self.run(&full);
        assert_eq!(out.code, 0, "metprof {argv:?} failed: {}", out.stderr);
        serde_json::from_str(&out.stdout).expect("stdout is JSON")
    }

    /// Run, require exit status 2, and return stderr.
    pub fn refused(&self, argv: &[&str]) -> String {
        let out = self.run(argv);
        assert_eq!(
            out.code, 2,
            "metprof {argv:?} was expected to fail; stdout: {}",
            out.stdout
        );
        assert!(out.stdout.is_empty());
        out.stderr
    }
}

impl TraceSource for Mem {
    fn open(&self, path: &str) -> Result<Box<dyn Trace>> {
        match self.captures.get(path) {
            Some(c) => Ok(Box::new(c.clone())),
            None => Err(Error(format!("capture not found: {path}"))),
        }
    }
}

impl Files for Mem {
    fn read(&self, path: &str) -> std::io::Result<Vec<u8>> {
        match self.files.get(path) {
            Some(b) => Ok(b.clone()),
            None => Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "No such file or directory",
            )),
        }
    }
}

/// Assert `text` matches the regex `pattern` (a search, not a full match).
pub fn assert_matches(text: &str, pattern: &str) {
    let re = regex::Regex::new(pattern).expect("test pattern is a valid regex");
    assert!(re.is_match(text), "{text:?} does not match {pattern:?}");
}

/// Assert a core result is an error whose message matches `pattern`.
pub fn assert_err<T: std::fmt::Debug>(r: Result<T>, pattern: &str) {
    match r {
        Ok(v) => panic!("expected an error matching {pattern:?}, got Ok({v:?})"),
        Err(e) => assert_matches(&e.0, pattern),
    }
}

pub fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9 * (1.0 + a.abs().max(b.abs()))
}

pub fn assert_close(a: f64, b: f64) {
    assert!(close(a, b), "{a} != {b}");
}

pub fn f(v: &serde_json::Value) -> f64 {
    v.as_f64().unwrap_or_else(|| panic!("{v} is not a number"))
}
