// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the I/O ports. The pure core never reads a file; the CLI asks
//! these traits for bytes and rows and hands plain values to the core. The
//! real implementations live in `adapters`; tests implement them in memory.

use crate::core::capture::{CopyRow, KernelRow};
use crate::core::error::Result;
use crate::core::model::ApiCall;

/// Opens a capture by path. A missing path is `capture not found: PATH` and
/// nothing is ever created or written.
pub trait TraceSource {
    fn open(&self, path: &str) -> Result<Box<dyn Trace>>;
}

/// One opened capture: every row of every device, unordered.
pub trait Trace {
    /// The kernel table joined with its names. `graph_node` is `None` when
    /// the export has no graph-node column.
    fn kernel_rows(&self) -> Result<Vec<KernelRow>>;
    fn copy_rows(&self) -> Result<Vec<CopyRow>>;
    /// Host-side CUDA runtime API calls with their names.
    fn call_rows(&self) -> Result<Vec<ApiCall>>;
}

/// Whole-file reads for configs, records and environments.
pub trait Files {
    fn read(&self, path: &str) -> std::io::Result<Vec<u8>>;

    /// The SHA-256 (lowercase hex) of the file at `path`: how the KB records
    /// a file argument.
    fn sha256(&self, path: &str) -> std::io::Result<String> {
        self.read(path)
            .map(|b| crate::core::provenance::sha256_hex(&b))
    }
}

/// The texts a circuit-backed kernel tree is planned from, by name. Each
/// method returns the file's text or an error that names the file.
pub trait CircuitTexts {
    /// The instance list (recipes, shapes, policies).
    fn instances(&self) -> Result<String>;
    /// The circuit of one arch.
    fn circuit(&self, arch: &str) -> Result<String>;
    /// One block library a circuit includes.
    fn block(&self, name: &str) -> Result<String>;
    /// A stated precision table.
    fn precision_table(&self, name: &str) -> Result<String>;
    /// A checkpoint plan fixture (the policy precision's input).
    fn checkpoint_plan(&self, name: &str) -> Result<String>;
    /// The fusion rules of one hardware class.
    fn fusions(&self, hw: &str) -> Result<String>;
}

/// One HTTP exchange over a fresh connection: write `request` whole, then read
/// until the server closes. Each read is returned with its time since just
/// before the write. A connect, write or read failure, and a response not
/// complete within `timeout_s` of the write, are errors that name the endpoint.
pub trait ChatTransport {
    fn exchange(
        &self,
        ep: &crate::core::ttft::client::Endpoint,
        request: &[u8],
        timeout_s: f64,
    ) -> Result<Vec<crate::core::ttft::sse::Read>>;
}

/// A fresh salt per call (never the same twice in one process).
pub trait SaltSource {
    fn fresh(&self) -> u64;
}
