// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: `METRALE_GLM_AR_L2_PREFETCH=<MB>`: while a layer's MLP all-reduce is on the wire,
//! a side branch reads the next layer's first dense projection weights through L2
//! (`l2_warm_ldcg`), so the projections that follow the reduce find them there. At one decode
//! row the GPU waits ~10-30 us per reduce with DRAM idle; the next layer's q/k/v W4A16
//! projections (~6.5 MB each at TP 3) then stream from DRAM at ~220 GB/s. GB10 microbench
//! (`l2bench`): 8 MB reads in ~22 us from DRAM and ~7 us from L2.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - The warm only reads (`l2_warm.cu`): every output byte is unchanged.
//! - Every fork is joined before the pass ends: the fork at layer i is joined at the next fork
//!   site (layer i+1's MLP reduce, by when its projections have long finished) or by the last
//!   layer; the first layer drops a pending fork left by a pass that failed midway, whose
//!   capture was abandoned with it.
//! - Only decode-width launches fork (`rows <= L2_WARM_MAX_ROWS`).
//! - The ranges come from `glm5next_w4a16_dense::footprint`, so only W4A16-registered
//!   projections are warmed, in the order the mixer lists them (`dense_projections`), up to the
//!   byte budget.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, bail, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::Glm5NextMixer;

/// 2026-10-10: The lever; unset or `0` is off, `N` warms up to N MB per fork.
pub const L2_WARM_ENV: &str = "METRALE_GLM_AR_L2_PREFETCH";

/// 2026-10-10: Widest launch that forks a warm: the decode and verify widths. A prefill
/// sub-chunk reads each weight once for hundreds of rows and is not waiting on the wire.
pub const L2_WARM_MAX_ROWS: usize = 16;

/// 2026-10-10: Blocks per SM of one warm launch; a few resident warps per SM keep enough loads
/// in flight to stream at DRAM speed.
const BLOCKS_PER_SM: u32 = 2;

/// 2026-10-10: One byte range to warm.
pub type Range = (DevicePtr, usize);

/// 2026-10-10: The lever's budget in bytes, `None` when off. Errors on a value that is not a
/// whole number of MB. 2026-10-17: Read from the startup settings (`GlmLevers::l2_warm_budget`).
pub fn budget(s: &metrale_settings::InferenceServerSettings) -> Result<Option<usize>> {
    let Some(v) = s.kernels.glm_ar_l2_prefetch else {
        return Ok(None);
    };
    let mb = usize::try_from(v)
        .map_err(|_| anyhow::anyhow!("{L2_WARM_ENV}={v}: expected a whole number of MB"))?;
    Ok((mb > 0).then_some(mb << 20))
}

/// 2026-10-10: The first `budget` bytes of `ranges`, in order, each cut to whole 16-byte words
/// (the kernel's load width); empty ranges dropped.
pub fn within_budget(ranges: &[Range], budget: usize) -> Vec<Range> {
    let mut left = budget;
    let mut out = Vec::new();
    for &(p, bytes) in ranges {
        let take = bytes.min(left) & !15;
        if take == 0 {
            break;
        }
        out.push((p, take));
        left -= take;
    }
    out
}

/// 2026-10-10: Per layer position i, what its fork warms: layer i+1's ranges within `budget`;
/// the last layer warms nothing.
pub fn plan_next(per_layer: &[Vec<Range>], budget: usize) -> Vec<Vec<Range>> {
    (0..per_layer.len())
        .map(|i| {
            per_layer
                .get(i + 1)
                .map(|r| within_budget(r, budget))
                .unwrap_or_default()
        })
        .collect()
}

/// 2026-10-10: A mixer's W4A16-registered projection weights, in its `dense_projections` order
/// (the order the forward launches them), codes then scales for each.
pub fn mixer_ranges(mixer: &Glm5NextMixer) -> Vec<Range> {
    let projs = match mixer {
        Glm5NextMixer::Kda { layer, .. } => layer.dense_projections(),
        Glm5NextMixer::Dsa(layer) => layer.dense_projections(),
    };
    projs
        .into_iter()
        .filter_map(|(w, ..)| crate::glm5next_w4a16_dense::footprint(w))
        .flatten()
        .collect()
}

/// 2026-10-10: The side stream, its fork and join events, the kernel and the per-layer plan,
/// shared by every layer of the stack.
pub struct L2Warm {
    kernel: KernelHandle,
    side: u64,
    fork_ev: u64,
    join_ev: u64,
    sink: DevicePtr,
    blocks: u32,
    budget: usize,
    pending: AtomicBool,
    plan: OnceLock<Vec<Vec<Range>>>,
}

impl L2Warm {
    /// 2026-10-10: Errors when the target lacks `l2_warm_ldcg` or a stream, event or the sink
    /// cannot be created.
    pub fn new(gpu: &dyn GpuBackend, budget: usize) -> Result<Self> {
        let kernel = gpu
            .kernel("l2_warm", "l2_warm_ldcg")
            .with_context(|| format!("{L2_WARM_ENV} needs l2_warm::l2_warm_ldcg"))?;
        Ok(Self {
            kernel,
            side: gpu.create_stream()?,
            fork_ev: gpu.create_event()?,
            join_ev: gpu.create_event()?,
            sink: gpu.alloc(4)?,
            blocks: gpu.sm_count()? * BLOCKS_PER_SM,
            budget,
            pending: AtomicBool::new(false),
            plan: OnceLock::new(),
        })
    }

    pub fn budget(&self) -> usize {
        self.budget
    }

    /// 2026-10-10: Install the plan (`plan_next`), once, after every layer is built.
    pub fn set_plan(&self, plan: Vec<Vec<Range>>) -> Result<()> {
        if self.plan.set(plan).is_err() {
            bail!("{L2_WARM_ENV}: the warm plan was already set");
        }
        Ok(())
    }

    /// 2026-10-10: At layer `layer_idx`'s MLP reduce, before it is enqueued on `stream`: join a
    /// pending warm, then, for a decode-width launch with something planned, fork the side
    /// stream from `stream` and warm the next layer's ranges there.
    pub fn fork(
        &self,
        gpu: &dyn GpuBackend,
        layer_idx: usize,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        self.join(gpu, stream)?;
        let Some(plan) = self.plan.get() else {
            bail!("{L2_WARM_ENV}: a layer forked before the warm plan was set");
        };
        let ranges = plan.get(layer_idx).map(Vec::as_slice).unwrap_or_default();
        if rows > L2_WARM_MAX_ROWS || ranges.is_empty() {
            return Ok(());
        }
        gpu.record_event(self.fork_ev, stream)?;
        gpu.stream_wait_event(self.side, self.fork_ev)?;
        for &(p, bytes) in ranges {
            KernelLaunch::new(gpu, self.kernel)
                .grid([self.blocks, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(p)
                .arg_u64((bytes / 16) as u64)
                .arg_ptr(self.sink)
                .launch(self.side)?;
        }
        gpu.record_event(self.join_ev, self.side)?;
        self.pending.store(true, Ordering::Relaxed);
        Ok(())
    }

    /// 2026-10-10: Make `stream` wait for a pending warm; a no-op without one.
    pub fn join(&self, gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
        if self.pending.swap(false, Ordering::Relaxed) {
            gpu.stream_wait_event(stream, self.join_ev)?;
        }
        Ok(())
    }

    /// 2026-10-10: Forget a fork a failed pass left pending (the first layer, see the module
    /// invariants).
    pub fn reset(&self) {
        self.pending.store(false, Ordering::Relaxed);
    }

    #[cfg(test)]
    fn is_pending(&self) -> bool {
        self.pending.load(Ordering::Relaxed)
    }
}

/// 2026-10-10: Check a plan against the stack: one entry per layer.
pub fn check_plan(plan: &[Vec<Range>], layers: usize) -> Result<()> {
    ensure!(
        plan.len() == layers,
        "{L2_WARM_ENV}: plan for {} layers, stack has {layers}",
        plan.len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend};

    /// 2026-10-17: The replaced `parse_budget`, unchanged.
    fn old_budget(v: Option<&str>) -> Result<Option<usize>> {
        let Some(v) = v else { return Ok(None) };
        let mb: usize = v
            .trim()
            .parse()
            .with_context(|| format!("{L2_WARM_ENV}={v}: expected a whole number of MB"))?;
        Ok((mb > 0).then_some(mb << 20))
    }

    /// 2026-10-17: The budget at `probe` through the settings; `None` when refused.
    fn new_budget(probe: Option<&str>) -> Option<Option<usize>> {
        let pairs: Vec<(&str, &str)> = probe.map(|v| (L2_WARM_ENV, v)).into_iter().collect();
        metrale_model_layers::lever_equivalence::settings_from(&pairs)
            .ok()
            .and_then(|s| budget(&s).ok())
    }

    /// 2026-10-17: MB in, bytes out, 0 off, garbage refused, as the replaced reader, but for
    /// the listed probes: `-0` now reads as off and a count past `i64::MAX` is refused.
    #[test]
    fn the_budget_reads_as_before_except_signed_zero_and_past_i64() {
        for probe in [
            None,
            Some("0"),
            Some("6"),
            Some(" 6 "),
            Some("+6"),
            Some("6.5"),
            Some("lots"),
            Some("-1"),
            Some(""),
            Some("1"),
            Some("true"),
            Some("off"),
            Some("garbage"),
        ] {
            assert_eq!(new_budget(probe), old_budget(probe).ok(), "{probe:?}");
        }
        let past_i64 = (1u64 << 63).to_string();
        for (probe, was, now) in [
            ("-0", None, Some(None)),
            // 2026-10-17: The old shift dropped every bit of 2^63 MB: a zero-byte budget.
            (past_i64.as_str(), Some(Some(0)), None),
        ] {
            assert_eq!(
                (old_budget(Some(probe)).ok(), new_budget(Some(probe))),
                (was, now),
                "{probe:?}"
            );
        }
    }

    #[test]
    fn the_budget_cuts_in_order_to_whole_words() {
        let r = [(DevicePtr(0x1000), 100), (DevicePtr(0x2000), 64)];
        assert_eq!(
            within_budget(&r, 1 << 20),
            vec![(DevicePtr(0x1000), 96), (DevicePtr(0x2000), 64)]
        );
        assert_eq!(
            within_budget(&r, 120),
            vec![(DevicePtr(0x1000), 96), (DevicePtr(0x2000), 16)]
        );
        assert_eq!(within_budget(&r, 40), vec![(DevicePtr(0x1000), 32)]);
        assert!(within_budget(&r, 15).is_empty());
    }

    #[test]
    fn each_layer_warms_the_next_and_the_last_nothing() {
        let per = vec![
            vec![(DevicePtr(0x10), 64)],
            vec![(DevicePtr(0x20), 64)],
            vec![(DevicePtr(0x30), 64)],
        ];
        let plan = plan_next(&per, 1 << 20);
        assert_eq!(
            plan,
            vec![
                vec![(DevicePtr(0x20), 64)],
                vec![(DevicePtr(0x30), 64)],
                vec![]
            ]
        );
        assert!(check_plan(&plan, 3).is_ok());
        assert!(check_plan(&plan, 4).is_err());
    }

    /// 2026-10-10: A fork launches one warm per range on the side stream with the word count,
    /// leaves a join pending, and the next fork joins it first; a wide launch or an empty plan
    /// forks nothing.
    #[test]
    fn a_fork_warms_on_the_side_stream_and_is_joined_once() {
        let gpu = MockGpuBackend::new();
        let w = L2Warm {
            kernel: KernelHandle(77),
            side: 900,
            fork_ev: 1,
            join_ev: 2,
            sink: DevicePtr(0x50),
            blocks: 96,
            budget: 1 << 20,
            pending: AtomicBool::new(false),
            plan: OnceLock::new(),
        };
        assert!(w.fork(&gpu, 0, 1, 7).is_err(), "no plan yet");
        w.set_plan(vec![
            vec![(DevicePtr(0x1000), 4096), (DevicePtr(0x9000), 512)],
            vec![],
        ])
        .unwrap();
        assert!(w.set_plan(vec![]).is_err());
        w.fork(&gpu, 0, 17, 7).unwrap();
        assert!(!w.is_pending(), "17 rows is not a decode width");
        w.fork(&gpu, 0, 1, 7).unwrap();
        assert!(w.is_pending());
        let l = gpu.launches_snapshot();
        assert_eq!(l.len(), 2);
        assert!(
            l.iter()
                .all(|x| x.func == 77 && x.stream == 900 && x.grid == [96, 1, 1])
        );
        assert_eq!(
            l[0].args,
            vec![
                MockArg::Buffer(DevicePtr(0x1000)),
                MockArg::Bytes(256u64.to_le_bytes().to_vec()),
                MockArg::Buffer(DevicePtr(0x50)),
            ]
        );
        assert_eq!(l[1].args[1], MockArg::Bytes(32u64.to_le_bytes().to_vec()));
        w.fork(&gpu, 1, 1, 7).unwrap();
        assert!(
            !w.is_pending(),
            "layer 1 joined layer 0's warm and has nothing to warm"
        );
        assert_eq!(gpu.launches_snapshot().len(), 2);
        w.fork(&gpu, 0, 1, 7).unwrap();
        w.reset();
        assert!(!w.is_pending());
    }
}
