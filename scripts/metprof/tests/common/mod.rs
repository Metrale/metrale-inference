// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: shared test helpers: an in-memory capture behind the
//! `TraceSource` port, an in-memory `Files` port, a fake KB behind the
//! `KbOpen` port with a fixed clock, and a driver for the real `cli::run`.
//! Times are given in microseconds and stored as ns, as the profiler stores
//! them.

#![allow(dead_code)]

pub mod kb;
pub mod tree;
pub mod ttft;

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, VecDeque};

use metprof::cli::{Io, Outcome, run};
use metprof::core::capture::{CopyRow, KernelRow};
use metprof::core::error::{Error, Result};
use metprof::core::model::{ApiCall, Kernel, Launch};
use metprof::core::ttft::client::Endpoint;
use metprof::core::ttft::sse::Read;
use metprof::ports::{ChatTransport, Files, SaltSource, Trace, TraceSource};
use metprof_kb::port::{Arg, RunRecord};

pub use kb::{EMBEDDING_TOML, HASHED, KB_TOML, Sink, kb_toml};

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

/// Captures and files by path, in memory, and the runs the fake KB holds.
pub struct Mem {
    captures: HashMap<String, Capture>,
    files: HashMap<String, Vec<u8>>,
    replies: RefCell<VecDeque<Result<Vec<Read>>>>,
    /// Every exchange the CLI made: (endpoint, request bytes, timeout s).
    pub sent: RefCell<Vec<(Endpoint, Vec<u8>, f64)>>,
    salts: Cell<u64>,
    sink: Sink,
    records: RefCell<Vec<RunRecord>>,
}

impl Default for Mem {
    fn default() -> Self {
        Mem {
            captures: HashMap::new(),
            files: HashMap::from([
                (KB_TOML.to_string(), kb_toml("seal = \"off\"").into_bytes()),
                (EMBEDDING_TOML.to_string(), HASHED.as_bytes().to_vec()),
            ]),
            sink: Sink::Materialises,
            records: RefCell::new(Vec::new()),
            replies: RefCell::new(VecDeque::new()),
            sent: RefCell::new(Vec::new()),
            salts: Cell::new(0),
        }
    }
}

impl Mem {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn sink(mut self, sink: Sink) -> Self {
        self.sink = sink;
        self
    }

    /// The runs recorded so far, in order.
    pub fn records(&self) -> Vec<RunRecord> {
        self.records.borrow().clone()
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

    /// The next exchange answers with these reads.
    pub fn reply(self, reads: Vec<Read>) -> Self {
        self.replies.borrow_mut().push_back(Ok(reads));
        self
    }

    /// The next exchange fails with this transport error.
    pub fn reply_err(self, msg: &str) -> Self {
        self.replies
            .borrow_mut()
            .push_back(Err(Error(msg.to_string())));
        self
    }

    /// Run the real CLI over these ports, recording through `KB_TOML`
    /// unless `argv` names a `--kb`. A run that succeeds records exactly one
    /// run, a refused one none, and no recorded word or result string is a
    /// path this `Mem` holds.
    pub fn run(&self, argv: &[&str]) -> Outcome {
        self.run_with(argv, self)
    }

    /// `run`, with the files read through `files`.
    pub fn run_with(&self, argv: &[&str], files: &dyn Files) -> Outcome {
        let names_kb = argv.iter().any(|a| *a == "--kb" || a.starts_with("--kb="));
        let help = argv.iter().any(|a| *a == "--help" || *a == "-h");
        let mut full = if names_kb {
            vec![]
        } else {
            vec!["--kb", KB_TOML]
        };
        full.extend_from_slice(argv);
        let before = self.records.borrow().len();
        let out = self.run_raw(&full, files);
        let new = self.records.borrow()[before..].to_vec();
        // A spooled run is recorded even when it is not materialised (exit 3).
        let spooled = out.code == 0 || (out.code == 3 && self.sink == Sink::SpoolsOnly);
        let recorded = !names_kb && !help && spooled;
        assert_eq!(
            new.len(),
            usize::from(recorded),
            "metprof {argv:?} (exit {}) recorded {} runs",
            out.code,
            new.len()
        );
        for r in &new {
            self.assert_no_paths(r);
        }
        out
    }

    /// Run the real CLI with exactly `argv`.
    pub fn run_raw(&self, argv: &[&str], files: &dyn Files) -> Outcome {
        let argv: Vec<String> = argv.iter().map(|s| s.to_string()).collect();
        run(
            &argv,
            &Io {
                traces: self,
                files,
                chat: self,
                salt: self,
                kb: self,
                clock: self,
            },
        )
    }

    fn assert_no_paths(&self, r: &RunRecord) {
        let held = |s: &str| self.files.contains_key(s) || self.captures.contains_key(s);
        for a in &r.args {
            if let Arg::Word(w) = a {
                assert!(!held(w), "the record keeps the path {w:?}: {:?}", r.args);
            }
        }
        fn strings(v: &serde_json::Value, out: &mut Vec<String>) {
            match v {
                serde_json::Value::String(s) => out.push(s.clone()),
                serde_json::Value::Array(xs) => xs.iter().for_each(|x| strings(x, out)),
                serde_json::Value::Object(m) => m.values().for_each(|x| strings(x, out)),
                _ => {}
            }
        }
        let mut all = Vec::new();
        r.result.values().for_each(|v| strings(v, &mut all));
        for s in all {
            assert!(!held(&s), "the record's result keeps the path {s:?}");
        }
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

    /// A file's digest; a capture held only as rows digests its rows.
    fn sha256(&self, path: &str) -> std::io::Result<String> {
        match (self.files.get(path), self.captures.get(path)) {
            (Some(b), _) => Ok(metprof::core::provenance::sha256_hex(b)),
            (None, Some(c)) => Ok(metprof::core::provenance::sha256_hex(
                format!("{c:?}").as_bytes(),
            )),
            (None, None) => Err(std::io::Error::new(
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

impl ChatTransport for Mem {
    fn exchange(&self, ep: &Endpoint, request: &[u8], timeout_s: f64) -> Result<Vec<Read>> {
        self.sent
            .borrow_mut()
            .push((ep.clone(), request.to_vec(), timeout_s));
        match self.replies.borrow_mut().pop_front() {
            Some(r) => r,
            None => Err(Error("test transport: no scripted reply left".to_string())),
        }
    }
}

/// Salts 1, 2, 3, ... in call order.
impl SaltSource for Mem {
    fn fresh(&self) -> u64 {
        self.salts.set(self.salts.get() + 1);
        self.salts.get()
    }
}
