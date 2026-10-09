// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The boundary between the pure harness and a device (SBIO): a runner launches the
//! case's entry point on the case's operands and returns the output bytes. The GPU runner lives
//! with the CLI; the crate's tests use a CPU runner over the conforming emulation.
//!
//! Owner: metrale-accuracy.
//! Invariants:
//! - A runner returns the whole output (`case.out` dims, in `case.out` encoding), shards placed
//!   at their `out_at` columns, and columns no shard writes left at the sentinel it was
//!   initialised with — so a missing column is a wrong value, never a stale-but-plausible one.

use crate::case::{Case, Enc, Tensor};

/// 2026-10-09: The byte every output buffer is initialised to before a launch (bf16 0x7f7f is
/// 3.4e38, f32 0x7f7f7f7f likewise: far outside every bound).
pub const SENTINEL: u8 = 0x7f;

/// 2026-10-09: Launches entry points.
pub trait KernelRunner {
    /// 2026-10-09: Run `case.kernel` on `case`; the output bytes.
    fn run(&mut self, case: &Case) -> Result<Vec<u8>, String>;
    /// 2026-10-09: What runs: the target closure hash of the binary, or `cpu-emulation`.
    fn closure(&self) -> String;
    /// 2026-10-09: The device, for records.
    fn device(&self) -> String;
}

/// 2026-10-09: The values of output bytes at flat indices `idx`.
pub fn decode(case: &Case, bytes: &[u8], idx: &[usize]) -> Result<Vec<f64>, String> {
    let (dims, enc) = (&case.out.0, case.out.1);
    let n: usize = dims.iter().product();
    if bytes.len() != enc.bytes_for(n) {
        return Err(format!(
            "{} output bytes for {n} {enc:?} values",
            bytes.len()
        ));
    }
    if enc == Enc::I32 {
        return Err("an i32 output has no bounded comparison".into());
    }
    let t = Tensor {
        enc,
        dims: dims.clone(),
        bytes: bytes.to_vec(),
    };
    idx.iter()
        .map(|&i| {
            if i < n {
                Ok(t.get(i))
            } else {
                Err(format!("index {i} beyond {n} outputs"))
            }
        })
        .collect()
}
