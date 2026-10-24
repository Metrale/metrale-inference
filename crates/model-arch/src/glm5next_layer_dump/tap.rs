// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: The I/O side of the layer dump (`glm5next_layer_dump`): a [`Tap`] opened per
//! layer call copies the selected rows of a buffer to the host on the launch's stream and hands
//! them to a [`Sink`]; [`FileSink`] writes one file per row and appends its manifest line.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - [`begin`] returns `None` with the lever off, on a capturing stream, and when the launch
//!   holds no wanted row; the hooks ([`put`], [`site`], [`topk`], [`moe_ids`]) are then no-ops.
//! - A row is read at `base + row * row_bytes`: every dumped buffer is row-contiguous.

use std::collections::{BTreeSet, HashMap};
use std::io::Write as _;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::layer::ForwardContext;

pub use super::{Call, Path, Point, Window};
use super::{DumpConfig, DumpRow, Entry, select};

/// 2026-10-13: Where dumped rows go.
pub trait Sink: Send + Sync {
    fn write(&self, entry: &Entry, bytes: &[u8]) -> Result<()>;
}

/// 2026-10-13: One file per row in `dir`, and a line per row appended to the rank's
/// `manifest_r{rank}.jsonl`, under one lock so a rank's lines keep its write order.
pub struct FileSink {
    dir: PathBuf,
    lock: Mutex<()>,
}

impl FileSink {
    /// 2026-10-13: Creates `dir` (and its parents).
    pub fn create(dir: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("layer dump: creating {}", dir.display()))?;
        Ok(Self {
            dir,
            lock: Mutex::new(()),
        })
    }
}

impl Sink for FileSink {
    fn write(&self, entry: &Entry, bytes: &[u8]) -> Result<()> {
        let _held = self
            .lock
            .lock()
            .map_err(|_| anyhow::anyhow!("layer dump: the sink lock is poisoned"))?;
        let file = self.dir.join(entry.file_name());
        std::fs::write(&file, bytes)
            .with_context(|| format!("layer dump: writing {}", file.display()))?;
        let manifest = self.dir.join(format!("manifest_r{}.jsonl", entry.rank));
        let mut m = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&manifest)
            .with_context(|| format!("layer dump: opening {}", manifest.display()))?;
        writeln!(m, "{}", entry.manifest_line())
            .with_context(|| format!("layer dump: appending to {}", manifest.display()))
    }
}

/// 2026-10-13: The layer being dumped: its index, hidden size, stream count, and its DSA
/// selection output (base, row width) when its mixer is DSA and the launch's rows own it.
/// 2026-10-22: `moe_ids` is the MoE router's top-k ids (base, `top_k`) when the MLP is routed;
/// `None` on a dense MLP layer.
#[derive(Clone, Copy, Debug)]
pub struct Layer {
    pub idx: usize,
    pub hidden: usize,
    pub hc: usize,
    pub topk: Option<(DevicePtr, usize)>,
    pub moe_ids: Option<(DevicePtr, usize)>,
}

/// 2026-10-13: A layer call's dump: the rows to copy and where they go. `own` is this rank's
/// row shard under sequence-parallel mHC (`Point::own_rows_only` points skip other rows).
pub struct Tap<'a> {
    gpu: &'a dyn GpuBackend,
    stream: u64,
    sink: &'a dyn Sink,
    layer: Layer,
    rank: usize,
    call: Call,
    ordinal: usize,
    rows: Vec<DumpRow>,
    own: Option<(usize, usize)>,
}

/// 2026-10-13: Where and what a [`Tap`] is opened for, besides the rows.
#[derive(Clone, Copy, Debug)]
pub struct Launch {
    pub layer: Layer,
    pub call: Call,
    pub rank: usize,
    pub own: Option<(usize, usize)>,
}

/// 2026-10-13: The tap of a layer call on `stream`, or `None` (module invariants). `windows`
/// runs only with the lever on. `own` as in [`Tap`]. 2026-10-17: `cfg` is the layer's
/// resolved lever (`Glm5NextLayer::dump`), `None` when off.
pub fn begin<'a>(
    ctx: &ForwardContext<'a>,
    cfg: Option<&DumpConfig>,
    stream: u64,
    layer: Layer,
    call: Call,
    own: Option<(usize, usize)>,
    windows: impl FnOnce() -> Vec<Window>,
) -> Result<Option<Tap<'a>>> {
    let Some(cfg) = cfg else {
        return Ok(None);
    };
    if ctx.graph_capture {
        static WARNED: AtomicBool = AtomicBool::new(false);
        if !WARNED.swap(true, Ordering::Relaxed) {
            tracing::warn!(
                "{}: a captured launch is not dumped; run the decode eagerly to dump it",
                super::DIR_ENV
            );
        }
        return Ok(None);
    }
    static SINK: OnceLock<Result<FileSink, String>> = OnceLock::new();
    let sink = SINK
        .get_or_init(|| FileSink::create(cfg.dir.clone()).map_err(|e| format!("{e:#}")))
        .as_ref()
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let launch = Launch {
        layer,
        call,
        rank: ctx.comm.map_or(0, |c| c.rank()),
        own,
    };
    Ok(open(
        ctx.gpu,
        stream,
        sink,
        cfg.rows.as_ref(),
        launch,
        &windows(),
    ))
}

/// 2026-10-13: [`begin`] past the lever and capture checks: takes the layer's next call
/// ordinal on this rank and returns a tap over the selected rows, `None` when there are none.
pub fn open<'a>(
    gpu: &'a dyn GpuBackend,
    stream: u64,
    sink: &'a dyn Sink,
    wanted: Option<&BTreeSet<usize>>,
    launch: Launch,
    windows: &[Window],
) -> Option<Tap<'a>> {
    static CALLS: OnceLock<Mutex<HashMap<(usize, usize), usize>>> = OnceLock::new();
    let ordinal = {
        let mut calls = CALLS
            .get_or_init(Default::default)
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let n = calls.entry((launch.rank, launch.layer.idx)).or_default();
        *n += 1;
        *n - 1
    };
    let rows = select(wanted, windows);
    (!rows.is_empty()).then_some(Tap {
        gpu,
        stream,
        sink,
        layer: launch.layer,
        rank: launch.rank,
        call: launch.call,
        ordinal,
        rows,
        own: launch.own,
    })
}

impl Tap<'_> {
    /// 2026-10-13: Copy each selected row of `point`'s buffer at `base` (blocking, ordered
    /// after `stream`'s work) to the sink.
    pub fn put(&self, point: Point, base: DevicePtr) -> Result<()> {
        self.put_rows(point, base, 0..usize::MAX)
    }

    /// 2026-10-13: [`Self::put`] of the selected rows inside `rows` only (a piece of the
    /// launch whose buffer is about to be overwritten).
    pub fn put_rows(
        &self,
        point: Point,
        base: DevicePtr,
        rows: std::ops::Range<usize>,
    ) -> Result<()> {
        let topk = self.layer.topk.map_or(0, |t| t.1);
        let moe_topk = self.layer.moe_ids.map_or(0, |t| t.1);
        let (dtype, shape) = point.row_layout(self.layer.hidden, self.layer.hc, topk, moe_topk);
        let mut entry = Entry {
            layer: self.layer.idx,
            point,
            rank: self.rank,
            path: self.call.path,
            call: self.ordinal,
            chunk: self.call.chunk,
            seq: 0,
            pos: 0,
            row: 0,
            dtype,
            shape,
        };
        let mut buf = vec![0u8; entry.nbytes()];
        for r in self.rows.iter().filter(|r| rows.contains(&r.row)) {
            if point.own_rows_only()
                && let Some((lo, hi)) = self.own
                && !(lo..hi).contains(&r.row)
            {
                continue;
            }
            self.gpu
                .copy_d2h_on_stream(base.offset(r.row * buf.len()), &mut buf, self.stream)
                .with_context(|| format!("layer dump: {} row {}", point.as_str(), r.row))?;
            (entry.seq, entry.pos, entry.row) = (r.seq, r.pos, r.row);
            self.sink.write(&entry, &buf)?;
        }
        Ok(())
    }
}

/// 2026-10-13: [`Tap::put`] when there is a tap.
pub fn put(tap: Option<&Tap<'_>>, point: Point, base: DevicePtr) -> Result<()> {
    tap.map_or(Ok(()), |t| t.put(point, base))
}

/// 2026-10-13: The DSA selection rows, when the tap's layer has them.
pub fn topk(tap: Option<&Tap<'_>>) -> Result<()> {
    match tap.and_then(|t| t.layer.topk.map(|(p, _)| (t, p))) {
        Some((t, p)) => t.put(Point::DsaTopk, p),
        None => Ok(()),
    }
}

/// 2026-10-22: The MoE router's top-k ids, when the tap's layer is routed. Called after the MLP
/// on the stream that ran its top-k: nothing after the top-k writes the ids (the expert, union,
/// sort and combine launches only read them), and the D2H is ordered after it on that stream;
/// the shared-expert fork joins before the MLP returns.
pub fn moe_ids(tap: Option<&Tap<'_>>) -> Result<()> {
    match tap.and_then(|t| t.layer.moe_ids.map(|(p, _)| (t, p))) {
        Some((t, p)) => t.put(Point::MoeIds, p),
        None => Ok(()),
    }
}

/// 2026-10-13: An mHC site after its pre-norm: the streams it read (the layer input at the
/// attention site, the post-attention streams at the FFN site), its `post`, `comb` and normed
/// rows.
pub fn site(
    tap: Option<&Tap<'_>>,
    ffn: bool,
    [streams, post, comb, normed]: [DevicePtr; 4],
) -> Result<()> {
    let Some(t) = tap else {
        return Ok(());
    };
    let points = if ffn {
        [
            Point::PostAttn,
            Point::FfnPost,
            Point::FfnComb,
            Point::FfnNormed,
        ]
    } else {
        [
            Point::StreamsIn,
            Point::AttnPost,
            Point::AttnComb,
            Point::AttnNormed,
        ]
    };
    for (p, base) in points.into_iter().zip([streams, post, comb, normed]) {
        t.put(p, base)?;
    }
    Ok(())
}
