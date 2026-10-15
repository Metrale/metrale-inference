// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: `METRALE_GLM_MOE_QUANT_EARLY=1`: under the pipelined SP-mHC body
//! (`METRALE_GLM_SP_OVERLAP=1`, `glm5next_sp_overlap`), each rank quantizes its own shard of
//! the FFN-normed rows for the routed experts' gate/up GEMMs on a side stream as soon as its
//! last section-2 piece is computed (while the comm stream still all-gathers the normed rows),
//! then all-gathers the codes, group scales and global scales on the comm stream after the
//! normed rows. The grouped W4A4 launch (`glm5next_mlp::forward::w4a4_grouped`) waits for the
//! gathered codes instead of quantizing every row itself.
//!
//! Numerics: the quantizer computes each row from that row alone under the static gate/up
//! `input_scale` (`w4a4_quant_rows_static`, one block per row), so a shard launched at its row
//! offset writes the bytes a launch of every row writes there
//! (`glm5next_mlp::forward::GateUpCodes::launch_rows`); the gather moves those bytes to the
//! same rows on every rank. The MoE reads the same bytes as before, bit for bit.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Every rank decides from `k`, the world size, the piece count and the levers alone, so all
//!   ranks enqueue the same gather at the same program point.
//! - Streams are ordered by events only: the side stream starts after the caller's last piece
//!   (and so after every earlier reader of the scratch), the gather after the side stream's
//!   quantization, and the MoE after the gather. Nothing waits on the host.
//! - A gather arms exactly one grouped launch on this thread ([`take`]); [`finish`] refuses a
//!   gather nothing consumed, and a grouped launch of other rows or another scratch is refused.

use std::cell::Cell;
use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result, bail, ensure};
use metrale_comm::CommBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use crate::glm5next_layer::{Glm5NextLayer, Glm5NextMlpSite};
use crate::glm5next_mlp::forward::{GateUpCodes, gate_up_codes};

/// 2026-10-15: The lever's environment variable.
pub const LEVER: &str = "METRALE_GLM_MOE_QUANT_EARLY";

/// 2026-10-15: [`LEVER`], read once: unset or `0` off, `1` on (which needs
/// `METRALE_GLM_SP_OVERLAP=1`, `METRALE_GLM_MOE_W4A4_GROUPED=1` and no
/// `METRALE_GLM_SP_PIPELINE=1`); another value is an error.
pub fn enabled() -> Result<bool> {
    static V: OnceLock<Result<bool, String>> = OnceLock::new();
    V.get_or_init(|| {
        let s = |r: Result<bool>| r.map_err(|e| e.to_string());
        parse(
            std::env::var(LEVER).ok().as_deref(),
            s(crate::glm5next_sp_overlap::enabled())?,
            s(crate::glm5next_sp_pipeline::enabled())?,
            s(crate::glm5next_mlp::forward::moe_w4a4_grouped())?,
        )
    })
    .clone()
    .map_err(anyhow::Error::msg)
}

/// 2026-10-15: [`enabled`]'s parse, pure: `overlap`, `pipeline` and `grouped` are the values of
/// `METRALE_GLM_SP_OVERLAP`, `METRALE_GLM_SP_PIPELINE` and `METRALE_GLM_MOE_W4A4_GROUPED`.
pub(crate) fn parse(
    v: Option<&str>,
    overlap: bool,
    pipeline: bool,
    grouped: bool,
) -> Result<bool, String> {
    let on = match v {
        None | Some("0") => false,
        Some("1") => true,
        Some(other) => return Err(format!("{LEVER}={other:?}: expected 0 or 1")),
    };
    if !on {
        return Ok(false);
    }
    if !overlap {
        return Err(format!("{LEVER}=1 needs METRALE_GLM_SP_OVERLAP=1"));
    }
    if pipeline {
        return Err(format!(
            "{LEVER}=1 is not supported with METRALE_GLM_SP_PIPELINE=1"
        ));
    }
    if !grouped {
        return Err(format!("{LEVER}=1 needs METRALE_GLM_MOE_W4A4_GROUPED=1"));
    }
    Ok(true)
}

/// 2026-10-15: The load-time readback line.
pub fn readback(on: bool) -> String {
    if on {
        format!(
            "{LEVER}=1: a pipelined SP-mHC sub-chunk quantizes each rank's shard of the routed \
             gate/up rows on a side stream after its last section-2 piece and all-gathers the \
             codes on the comm stream; the grouped W4A4 launch waits for them (same bytes)"
        )
    } else {
        format!("{LEVER} off: the grouped W4A4 launch quantizes every gate/up row itself")
    }
}

/// 2026-10-15: Log [`readback`] once and refuse a lever setting this load cannot run.
pub fn check_at_load() -> Result<()> {
    static LOGGED: std::sync::Once = std::sync::Once::new();
    let on = enabled()?;
    LOGGED.call_once(|| tracing::info!("{}", readback(on)));
    Ok(())
}

/// 2026-10-15: The side stream and the three events: the caller's last piece, the side
/// stream's quantization, the comm stream's gather.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Side {
    stream: u64,
    last: u64,
    quant: u64,
    gathered: u64,
}

/// 2026-10-15: [`Side`], created on first use (never under a graph capture: SP-mHC is not
/// taken there).
pub(crate) fn side(gpu: &dyn GpuBackend) -> Result<Side> {
    static S: Mutex<Option<Side>> = Mutex::new(None);
    let mut g = S.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(s) = *g {
        return Ok(s);
    }
    let ev = || gpu.create_event().context("GLM MoE early quant: event");
    let s = Side {
        stream: gpu
            .create_stream()
            .context("GLM MoE early quant: side stream")?,
        last: ev()?,
        quant: ev()?,
        gathered: ev()?,
    };
    *g = Some(s);
    Ok(s)
}

/// 2026-10-15: The gathered codes one grouped launch on this thread waits for.
#[derive(Clone, Copy, Debug)]
struct Armed {
    x: DevicePtr,
    rows: usize,
    codes: GateUpCodes,
    ready: u64,
}

thread_local! {
    static ARMED: Cell<Option<Armed>> = const { Cell::new(None) };
}

/// 2026-10-15: For the grouped launch of `rows` rows at `x` quantized into `codes`: the event
/// its gathered codes are complete at, once (`None` when nothing is armed). Armed codes of
/// other rows or another scratch are an error.
pub fn take(x: DevicePtr, rows: usize, codes: &GateUpCodes) -> Result<Option<u64>> {
    let Some(a) = ARMED.with(Cell::take) else {
        return Ok(None);
    };
    ensure!(
        a.x == x && a.rows == rows && a.codes.same(codes),
        "GLM MoE early quant: codes of {} rows at {:#x} were gathered, but the grouped launch \
         reads {rows} rows at {:#x}",
        a.rows,
        a.x.0,
        x.0
    );
    Ok(Some(a.ready))
}

/// 2026-10-15: One sub-chunk's early quantization: the codes, the `rows` normed rows at `x`,
/// every rank's shard, this rank's, and the comm stream.
#[derive(Debug)]
pub struct Early {
    codes: GateUpCodes,
    x: DevicePtr,
    rows: usize,
    shards: Vec<(usize, usize)>,
    own: (usize, usize),
    comm_stream: u64,
    side: Side,
    quantized: bool,
}

impl Glm5NextLayer {
    /// 2026-10-15: The early quantization of this layer's `k` normed rows at `x` under the
    /// lever, when the sub-chunk is pipelined (`comm_stream`, the overlap's comm stream) and
    /// its MoE runs the grouped W4A4 GEMM; `None` otherwise.
    pub(crate) fn moe_quant_early(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        k: usize,
        shards: &[(usize, usize)],
        rank: usize,
        comm_stream: Option<u64>,
    ) -> Result<Option<Early>> {
        let (true, Some(comm_stream), Glm5NextMlpSite::Moe(w)) =
            (enabled()?, comm_stream, &self.mlp)
        else {
            return Ok(None);
        };
        let Some(codes) = gate_up_codes(&self.mlp_kernels, &self.mlp_cfg, w, &self.mlp_ws, k)?
        else {
            return Ok(None);
        };
        plan(codes, x, k, shards, rank, comm_stream, side(gpu)?).map(Some)
    }
}

/// 2026-10-15: [`Early`] of `codes` for `k` rows at `x` cut into `shards`, this rank's at
/// `rank`.
pub(crate) fn plan(
    codes: GateUpCodes,
    x: DevicePtr,
    k: usize,
    shards: &[(usize, usize)],
    rank: usize,
    comm_stream: u64,
    side: Side,
) -> Result<Early> {
    let own = *shards
        .get(rank)
        .context("GLM MoE early quant: rank past the shards")?;
    ensure!(
        shards.first().map(|s| s.0) == Some(0) && shards.last().map(|s| s.1) == Some(k),
        "GLM MoE early quant: shards {shards:?} do not cover {k} rows"
    );
    ensure!(
        own.0 < own.1 && k <= codes.rows_cap,
        "GLM MoE early quant: shard {own:?} of {k} rows (scratch {})",
        codes.rows_cap
    );
    Ok(Early {
        codes,
        x,
        rows: k,
        shards: shards.to_vec(),
        own,
        comm_stream,
        side,
        quantized: false,
    })
}

/// 2026-10-15: After this rank's section-2 piece ending at row `b` on `main`: on the last one
/// (the end of this rank's shard), the side stream waits for it and quantizes the shard.
pub fn after_piece(e: Option<&mut Early>, gpu: &dyn GpuBackend, b: usize, main: u64) -> Result<()> {
    let Some(e) = e else { return Ok(()) };
    if b != e.own.1 {
        return Ok(());
    }
    ensure!(
        !e.quantized,
        "GLM MoE early quant: the shard was quantized twice"
    );
    let s = e.side;
    gpu.record_event(s.last, main)?;
    gpu.stream_wait_event(s.stream, s.last)?;
    e.codes.launch_rows(gpu, e.x, e.own.0, e.own.1, s.stream)?;
    gpu.record_event(s.quant, s.stream)?;
    e.quantized = true;
    Ok(())
}

/// 2026-10-15: After section 2: the comm stream waits for this rank's quantization and
/// all-gathers the codes, group scales and global scales of every shard in place, then the
/// grouped launch on this thread is armed with the gather's end.
pub fn gather(e: Option<&mut Early>, gpu: &dyn GpuBackend, comm: &dyn CommBackend) -> Result<()> {
    let Some(e) = e else { return Ok(()) };
    ensure!(
        e.quantized,
        "GLM MoE early quant: no section-2 piece ended this rank's shard at row {}",
        e.own.1
    );
    // 2026-10-15: Refuse before enqueueing anything: codes still armed were never consumed.
    if let Some(stale) = ARMED.with(Cell::get) {
        bail!(
            "GLM MoE early quant: codes of {} rows were still armed at the next gather",
            stale.rows
        );
    }
    let c = e.comm_stream;
    gpu.stream_wait_event(c, e.side.quant)?;
    let bytes = e.codes.row_bytes();
    for (out, row_bytes) in e.codes.out.into_iter().zip(bytes) {
        crate::glm5next_sp_mhc::all_gather_rows(comm, out, &e.shards, row_bytes, c)?;
    }
    gpu.record_event(e.side.gathered, c)?;
    let armed = Armed {
        x: e.x,
        rows: e.rows,
        codes: e.codes,
        ready: e.side.gathered,
    };
    ARMED.with(|a| a.set(Some(armed)));
    Ok(())
}

/// 2026-10-15: After the MLP: the gathered codes must have been consumed by its grouped launch.
pub fn finish(e: Option<Early>) -> Result<()> {
    if let (Some(e), Some(left)) = (e, ARMED.with(Cell::take)) {
        bail!(
            "GLM MoE early quant: the MLP of {} rows did not run the grouped W4A4 launch the \
             codes of {} rows were gathered for",
            e.rows,
            left.rows
        );
    }
    Ok(())
}

#[cfg(test)]
#[path = "glm5next_moe_quant_early_tests.rs"]
mod tests;
