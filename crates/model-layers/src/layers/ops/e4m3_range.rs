// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: Debug-build range check on the activations an unscaled E4M3 cast reads.
//!
//! Owner: model-layers ops.
//! Invariants: release builds compile the check out; it never changes a launch.
//!
//! Several W4A16 and FP8 prefill kernels cast their BF16 A operand to E4M3 with no scale
//! (`bf16x4_to_e4m3x4`, `cvt.rn.satfinite`), and the `bf16_to_fp8` op casts a buffer the
//! same way (the FP8 activation path, and some weights at load). E4M3's largest finite value is 448, so a larger
//! activation saturates without an error. A kernel declares the cast with
//! `<entry>_a_e4m3` (`GpuBackend::kernel_casts_a_to_e4m3`). Measured on the certified
//! dense 27B models (Qwen3.8-27B-NVFP4 and Qwen3.6-27B-NVFP4; short, 5k, 27.5k-token,
//! tool-call and code prompts), the largest activation at these sites was 134, a margin of
//! 3.3x. In debug builds every such launch, and every `bf16_to_fp8`, checks its source and
//! panics past 448.
//!
//! 2026-09-30: Two more things.
//! - The grouped MoE launchers check the A rows their kernel reads (the rows the expert
//!   offsets span, through `sorted_token_ids` when one is passed), the same way.
//! - A saturation that is disclosed or known does not panic. The call site opens an
//!   [`allow_e4m3_saturation`] scope naming one of the [`E4m3Saturation`] constants below,
//!   which are the complete list. Inside it, a violation up to that entry's bound is logged
//!   once per entry and counted ([`e4m3_saturations`]); past the bound, or outside any
//!   scope, it still panics.

use std::cell::RefCell;
use std::collections::BTreeMap;

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use parking_lot::Mutex;

/// 2026-09-29: E4M3's largest finite magnitude; an unscaled cast saturates above it.
pub const E4M3_MAX: f32 = 448.0;

/// 2026-09-30: An unscaled E4M3 cast whose saturation is disclosed or known, with the
/// largest `|x|` the disclosure covers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct E4m3Saturation {
    /// 2026-09-30: Where it happens; the key its count is kept under.
    pub site: &'static str,
    /// 2026-09-30: Why it is not a defect to panic on.
    pub why: &'static str,
    /// 2026-09-30: The largest `|x|` covered. Past it the check panics as for an undeclared
    /// site.
    pub bound: f32,
}

/// 2026-09-30: `--nemotron-shared-expert-e4m3`: the Nemotron-H shared expert's prefill tile
/// GEMM casts its activations to E4M3 unscaled. The flag's help discloses the saturation
/// (1-30% local error measured on Nemotron-3-Nano), so any magnitude is covered.
pub const NEMOTRON_SHARED_EXPERT_E4M3: E4m3Saturation = E4m3Saturation {
    site: "Nemotron-H shared expert, --nemotron-shared-expert-e4m3",
    why: "opted in on the command line; the flag discloses the saturating cast",
    bound: f32::INFINITY,
};

/// 2026-09-30: The routed experts' down projection reads the SiLU product. On
/// Qwen3.6-35B-A3B-NVFP4 it reaches 510 at position 0 of the last layer (measured
/// 2026-09-29, no effect on the output); every other certified model stays under 448. The
/// scaled cast that removes it is follow-up work, so this covers exactly the known value.
pub const MOE_ROUTED_DOWN_KNOWN: E4m3Saturation = E4m3Saturation {
    site: "MoE routed-down input",
    why: "known: Qwen3.6-35B-A3B-NVFP4 reaches 510 at position 0 of its last layer",
    bound: 512.0,
};

thread_local! {
    static SCOPES: RefCell<Vec<E4m3Saturation>> = const { RefCell::new(Vec::new()) };
}

/// 2026-09-30: Saturating launches seen per site, and whether the site has been logged.
static SEEN: Mutex<BTreeMap<&'static str, u64>> = Mutex::new(BTreeMap::new());

/// 2026-09-30: While the returned scope lives, a saturation on this thread up to
/// `saturation.bound` is logged once and counted instead of panicking. Scopes nest; the
/// innermost one applies.
#[must_use = "the allowance ends when the scope is dropped"]
pub fn allow_e4m3_saturation(saturation: E4m3Saturation) -> E4m3SaturationScope {
    SCOPES.with(|s| s.borrow_mut().push(saturation));
    E4m3SaturationScope { _private: () }
}

/// 2026-09-30: See [`allow_e4m3_saturation`].
pub struct E4m3SaturationScope {
    _private: (),
}

impl Drop for E4m3SaturationScope {
    fn drop(&mut self) {
        SCOPES.with(|s| s.borrow_mut().pop());
    }
}

/// 2026-09-30: Saturating launches counted under `site` in this process (debug builds only).
pub fn e4m3_saturations(site: &str) -> u64 {
    SEEN.lock().get(site).copied().unwrap_or(0)
}

/// 2026-09-30: The verdict on one launch's largest activation magnitude.
fn judge(amax: f32) {
    if amax <= E4M3_MAX {
        return;
    }
    match SCOPES.with(|s| s.borrow().last().copied()) {
        Some(allowed) if amax <= allowed.bound => {
            let mut seen = SEEN.lock();
            let count = seen.entry(allowed.site).or_insert(0);
            if *count == 0 {
                tracing::warn!(
                    site = allowed.site,
                    why = allowed.why,
                    amax,
                    "unscaled E4M3 cast saturates (|x| = {amax} > {E4M3_MAX}) at a disclosed \
                     site; counted, logged once"
                );
            }
            *count += 1;
        }
        Some(allowed) => panic!(
            "unscaled E4M3 cast would saturate: activation |x| = {amax} > {E4M3_MAX}, past the \
             {} that `{}` discloses",
            allowed.bound, allowed.site
        ),
        None => panic!("unscaled E4M3 cast would saturate: activation |x| = {amax} > {E4M3_MAX}"),
    }
}

/// 2026-09-30: The largest finite-or-infinite magnitude among BF16 `bytes`; NaN is skipped.
fn bf16_amax<'a>(rows: impl Iterator<Item = &'a [u8]>) -> f32 {
    rows.flat_map(|r| r.chunks_exact(2))
        .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16).abs())
        .filter(|v| !v.is_nan())
        .fold(0.0f32, f32::max)
}

/// 2026-09-29: When `kernel` casts its A operand to E4M3 with no scale, check the
/// `[m, k]` BF16 `input` it reads (see [`check_e4m3_range`]). A no-op otherwise, and in
/// release builds.
pub fn check_e4m3_activation_range(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    m: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    if cfg!(debug_assertions) && gpu.kernel_casts_a_to_e4m3(kernel) {
        check_e4m3_range(gpu, input, m as usize * k as usize, stream)?;
    }
    Ok(())
}

/// 2026-09-29: Debug builds: copy the `elements` BF16 values at `input` to the host on
/// `stream` and panic when one exceeds [`E4M3_MAX`] in magnitude, unless an
/// [`allow_e4m3_saturation`] scope covers it. NaN is not checked here. Skipped under graph
/// capture, where a copy to the host cannot run. Release builds: a no-op.
pub fn check_e4m3_range(
    gpu: &dyn GpuBackend,
    input: DevicePtr,
    elements: usize,
    stream: u64,
) -> Result<()> {
    if !cfg!(debug_assertions) || elements == 0 || gpu.stream_is_capturing(stream) {
        return Ok(());
    }
    let mut buf = vec![0u8; elements * 2];
    gpu.copy_d2h_on_stream(input, &mut buf, stream)?;
    judge(bf16_amax(std::iter::once(buf.as_slice())));
    Ok(())
}

/// 2026-09-30: The grouped (pointer-table) MoE form of [`check_e4m3_activation_range`]. The
/// kernel reads A row `sorted_token_ids[p]`, or row `p` when `sorted_token_ids` is null, for
/// every position `p` in `expert_offsets[e]..expert_offsets[e + 1]`, `e < num_experts`; only
/// those rows are checked, each `k` BF16 values wide. A no-op for a kernel that keeps A in
/// BF16, in release builds, and under graph capture.
#[allow(clippy::too_many_arguments)]
pub fn check_e4m3_grouped(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: DevicePtr,
    expert_offsets: DevicePtr,
    sorted_token_ids: DevicePtr,
    num_experts: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    if !cfg!(debug_assertions)
        || !gpu.kernel_casts_a_to_e4m3(kernel)
        || gpu.stream_is_capturing(stream)
    {
        return Ok(());
    }
    let offsets = read_i32s(gpu, expert_offsets, num_experts as usize + 1, stream)?;
    let positions: Vec<usize> = offsets
        .windows(2)
        .flat_map(|w| w[0].max(0) as usize..w[1].max(0) as usize)
        .collect();
    let Some(&last) = positions.iter().max() else {
        return Ok(());
    };
    let rows: Vec<usize> = if sorted_token_ids.is_null() {
        positions
    } else {
        let ids = read_i32s(gpu, sorted_token_ids, last + 1, stream)?;
        positions
            .iter()
            .map(|&p| usize::try_from(ids[p]))
            .collect::<Result<_, _>>()
            .map_err(|_| {
                anyhow::anyhow!("a negative sorted token id at a position the kernel reads")
            })?
    };
    let Some(&top) = rows.iter().max() else {
        return Ok(());
    };
    let width = k as usize * 2;
    let mut buf = vec![0u8; (top + 1) * width];
    gpu.copy_d2h_on_stream(a, &mut buf, stream)?;
    judge(bf16_amax(
        rows.iter().map(|&r| &buf[r * width..(r + 1) * width]),
    ));
    Ok(())
}

fn read_i32s(gpu: &dyn GpuBackend, ptr: DevicePtr, n: usize, stream: u64) -> Result<Vec<i32>> {
    let mut buf = vec![0u8; n * 4];
    gpu.copy_d2h_on_stream(ptr, &mut buf, stream)?;
    Ok(buf
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

#[cfg(test)]
#[path = "e4m3_range_tests.rs"]
mod tests;
