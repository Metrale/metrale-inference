// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: A W4A16 projection launch run from a plan group (`glm_w4a16`): the one-row NVFP4
//! GEMV or the pipelined GEMV over 1..=6 projections sharing their input, as a typed step over
//! plan kernel ids ([`W4a16Step`]), on the weights the dense tier registered at load
//! (`glm5next_w4a16_dense::register`) and the launchers its own dispatch uses.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Reads no dense lever: the kernel is the plan's, the pipe's block cap the kernel's own rule
//!   ([`w4a16_gemv_pipe_cap`]) over the launch's shapes and the SM count bound at installation.
//! - Every member must be registered at the shape it is launched with; an unregistered member is
//!   an error, never a BF16 or FP8 launch.

use anyhow::{Context, Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_layers::layers::ops::{
    W4A16_GEMV_PIPE_MODULE, W4a16Seg, w4a16_gemv, w4a16_gemv_pipe_cap, w4a16_gemv_pipe_entry,
    w4a16_gemv_pipe_with,
};

/// 2026-10-23: The entries a W4A16 step launches.
pub mod entry {
    pub type Entry = (&'static str, &'static str);
    /// 2026-10-23: The one-row NVFP4 GEMV.
    pub const GEMV: Entry = ("w4a16_gemv", "w4a16_gemv");

    /// 2026-10-23: The pipelined GEMV of `rows` rows, `None` where it has no entry.
    pub fn pipe(rows: u64) -> Option<Entry> {
        let m = u32::try_from(rows).ok()?;
        super::w4a16_gemv_pipe_entry(m).map(|f| (super::W4A16_GEMV_PIPE_MODULE, f))
    }
}

/// 2026-10-23: One W4A16 launch of a plan group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum W4a16Step<K> {
    /// 2026-10-23: One row of one projection on `w4a16_gemv`.
    Gemv(K),
    /// 2026-10-23: The group's rows over all its projections in one `w4a16_gemv_pipe` launch.
    Pipe(K),
}

impl<K> W4a16Step<K> {
    pub fn map<L>(self, f: impl FnOnce(K) -> Result<L>) -> Result<W4a16Step<L>> {
        Ok(match self {
            Self::Gemv(k) => W4a16Step::Gemv(f(k)?),
            Self::Pipe(k) => W4a16Step::Pipe(f(k)?),
        })
    }
}

/// 2026-10-23: `output_i [m, n_i] = a [m, k] @ W_i^T` for `members[i] = (weight key, output,
/// n_i)` by `step`, on `sms` SMs (the pipe's block cap).
#[allow(clippy::too_many_arguments)]
pub fn launch(
    gpu: &dyn GpuBackend,
    step: W4a16Step<KernelHandle>,
    members: &[(DevicePtr, DevicePtr, usize)],
    a: DevicePtr,
    m: usize,
    k: usize,
    sms: u32,
    stream: u64,
) -> Result<()> {
    let mut segs = Vec::with_capacity(members.len());
    for &(b, output, n) in members {
        let e = crate::glm5next_w4a16_dense::lookup(b, n, k)?.with_context(|| {
            format!("W4A16 plan: weight {b} [{n}, {k}] is not registered with the dense tier")
        })?;
        segs.push(W4a16Seg {
            weight: e.w,
            output,
            n: u32::try_from(n)?,
        });
    }
    let (m32, k32) = (u32::try_from(m)?, u32::try_from(k)?);
    match step {
        W4a16Step::Gemv(kernel) => {
            ensure!(
                m == 1 && segs.len() == 1,
                "W4A16 plan: the GEMV runs one row of one projection ({m} rows, {} members)",
                segs.len()
            );
            let s = &segs[0];
            w4a16_gemv(gpu, kernel, a, &s.weight, s.output, s.n, k32, stream)
        }
        W4a16Step::Pipe(kernel) => {
            let ns: Vec<u32> = segs.iter().map(|s| s.n).collect();
            let cap = w4a16_gemv_pipe_cap(&ns, k32, m32, sms);
            w4a16_gemv_pipe_with(gpu, kernel, a, &segs, m32, k32, cap, stream)
        }
    }
}
