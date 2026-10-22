// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: `METRALE_GLM_LAYER_DUMP=<dir>`: an opt-in per-layer dump of a GLM forward, to
//! find the first layer and point where two runs' bits part (a diagnostic, never a serve
//! setting). Each mHC layer copies chosen rows of its intermediates to `<dir>` on every rank:
//! the layer input, each site's streams, `post`, `comb` and normed rows, the mixer output before
//! and after its all-reduce, the DSA selection, the MoE router's expert ids, the MLP output
//! before and after its reduce, the layer's output streams and, on the last layer, the hidden row before the final norm.
//!
//! Rows: the last row of every prefill pass (the prompt's last row is the last pass's), plus
//! every row whose absolute position is in `METRALE_GLM_LAYER_DUMP_ROWS` (a comma list), on
//! every path (prefill, single and batched decode, verify). A row is found from its position
//! inside the launch's windows ([`select`]), so the sub-chunk ("chunk") a position fell in and
//! its row there are recorded, never assumed.
//!
//! Each dumped row is one file named by [`Entry::file_name`], plus one line of
//! `manifest_r{rank}.jsonl` ([`Entry::manifest_line`]) in write order, which is execution
//! order on that rank.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Unset, nothing is launched, copied or synchronised: every hook is a `None` check
//!   ([`tap::begin`] returns `None`), so the bits and the launch sequence are those without it.
//! - Set, each copy is a blocking D2H on the launch's stream after the launches that wrote the
//!   buffer: it adds host syncs, never a launch, so the dumped bits are the undumped run's.
//! - A capturing stream is never dumped (a D2H wait is illegal inside a capture): decode dumps
//!   need an eager decode.
//! - 2026-10-17: The lever is read from the startup settings ([`config`]) once per load and
//!   held by each layer (`Glm5NextLayer::dump`), never from the process environment.

use std::collections::BTreeSet;
use std::path::PathBuf;

use anyhow::Result;

pub mod tap;

/// 2026-10-13: The output directory; unset is off.
pub const DIR_ENV: &str = "METRALE_GLM_LAYER_DUMP";
/// 2026-10-13: The absolute positions to dump besides each prefill pass's last row.
pub const ROWS_ENV: &str = "METRALE_GLM_LAYER_DUMP_ROWS";

/// 2026-10-13: The parsed lever.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DumpConfig {
    pub dir: PathBuf,
    /// 2026-10-13: `None` when [`ROWS_ENV`] is unset: only each prefill pass's last row.
    pub rows: Option<BTreeSet<usize>>,
}

/// 2026-10-17: The lever of the startup settings; `None` when [`DIR_ENV`] is unset. A malformed
/// value is an error.
pub fn config(s: &metrale_settings::InferenceServerSettings) -> Result<Option<DumpConfig>> {
    let t = &s.telemetry;
    parse(
        t.glm_layer_dump.as_deref(),
        t.glm_layer_dump_rows.as_deref(),
    )
    .map_err(anyhow::Error::msg)
}

/// 2026-10-13: The line the lever logs once, printed by the text-stack loader.
pub fn announce(c: &DumpConfig) {
    match &c.rows {
        None => tracing::warn!(
            "{DIR_ENV}={}: {ROWS_ENV} unset, dumping only the last row of each prefill pass",
            c.dir.display()
        ),
        Some(r) => tracing::warn!(
            "{DIR_ENV}={}: dumping each prefill pass's last row and positions {r:?}",
            c.dir.display()
        ),
    }
}

/// 2026-10-13: [`config`]'s parse, pure. An empty directory, an empty or malformed row list, or
/// a row list without a directory is an error.
pub(crate) fn parse(dir: Option<&str>, rows: Option<&str>) -> Result<Option<DumpConfig>, String> {
    let rows = rows
        .map(|r| {
            let set = r
                .split(',')
                .map(|p| {
                    p.trim()
                        .parse::<usize>()
                        .map_err(|e| format!("{ROWS_ENV}={r:?}: {p:?} is not a position: {e}"))
                })
                .collect::<Result<BTreeSet<usize>, String>>()?;
            Ok::<_, String>(set)
        })
        .transpose()?;
    match (dir, rows) {
        (None, None) => Ok(None),
        (None, Some(_)) => Err(format!("{ROWS_ENV} is set without {DIR_ENV}")),
        (Some(""), _) => Err(format!("{DIR_ENV} is set but empty")),
        (Some(d), rows) => Ok(Some(DumpConfig {
            dir: PathBuf::from(d),
            rows,
        })),
    }
}

/// 2026-10-13: Which forward a launch belongs to; part of every file name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Path {
    /// 2026-10-13: A prefill sub-chunk of one sequence (`forward_k` / `forward_one`).
    Prefill,
    /// 2026-10-13: One decode token (`forward_one`).
    Decode,
    /// 2026-10-13: One sequence's speculative verify (`forward_k`).
    Verify,
    /// 2026-10-13: A row group of the batched decode (`forward_spans`, one row per sequence).
    MultiDecode,
    /// 2026-10-13: A row group of the batched verify (`forward_spans`).
    MultiVerify,
    /// 2026-10-13: A row group of the multi-sequence prefill (`forward_prefill_spans`).
    SpanPrefill,
}

impl Path {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Prefill => "prefill",
            Self::Decode => "decode",
            Self::Verify => "verify",
            Self::MultiDecode => "mdecode",
            Self::MultiVerify => "mverify",
            Self::SpanPrefill => "sprefill",
        }
    }
}

/// 2026-10-13: A dump point of a layer, in launch order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Point {
    /// 2026-10-13: The BF16 hidden row the first layer expands.
    HiddenIn,
    StreamsIn,
    AttnPost,
    AttnComb,
    AttnNormed,
    /// 2026-10-13: The mixer's output, this rank's partial before the all-reduce.
    MixerOut,
    /// 2026-10-13: The DSA selection row (i32 token ids).
    DsaTopk,
    /// 2026-10-13: The mixer's output after the all-reduce (multi-rank only).
    MixerRed,
    /// 2026-10-13: The streams after the attention site's post-mix.
    PostAttn,
    FfnPost,
    FfnComb,
    FfnNormed,
    /// 2026-10-13: The MLP's output, this rank's partial before the all-reduce.
    MlpOut,
    /// 2026-10-22: A routed MoE layer's router top-k (i32 expert ids, `top_k` per row, in the
    /// top-k kernel's slot order). Global ids in `0..num_experts` under every expert layout:
    /// each rank routes over the full logits row and writes the same ids, so no rank's slice
    /// shows. Absent on dense MLP layers.
    MoeIds,
    /// 2026-10-13: The MLP's output after the all-reduce (when the MLP reduces).
    MlpRed,
    /// 2026-10-13: The layer's output streams.
    StreamsOut,
    /// 2026-10-13: The last layer's BF16 hidden row, before the final norm and the head.
    FinalHidden,
}

/// 2026-10-13: A dumped row's element type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    Bf16,
    F32,
    I32,
}

impl Dtype {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::F32 => "f32",
            Self::I32 => "i32",
        }
    }

    pub fn size(self) -> usize {
        match self {
            Self::Bf16 => 2,
            Self::F32 | Self::I32 => 4,
        }
    }
}

impl Point {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::HiddenIn => "hidden_in",
            Self::StreamsIn => "streams_in",
            Self::AttnPost => "attn_post",
            Self::AttnComb => "attn_comb",
            Self::AttnNormed => "attn_normed",
            Self::MixerOut => "mixer_out",
            Self::DsaTopk => "dsa_topk",
            Self::MixerRed => "mixer_red",
            Self::PostAttn => "post_attn",
            Self::FfnPost => "ffn_post",
            Self::FfnComb => "ffn_comb",
            Self::FfnNormed => "ffn_normed",
            Self::MlpOut => "mlp_out",
            Self::MoeIds => "moe_ids",
            Self::MlpRed => "mlp_red",
            Self::StreamsOut => "streams_out",
            Self::FinalHidden => "final_hidden",
        }
    }

    /// 2026-10-13: One row's element type and shape at hidden size `h`, `hc` streams and a
    /// DSA selection `topk` wide. Rows are contiguous in their buffer, so the row stride is the
    /// row's byte size. 2026-10-22: `moe_topk` is the MoE router's experts per row.
    pub fn row_layout(
        self,
        h: usize,
        hc: usize,
        topk: usize,
        moe_topk: usize,
    ) -> (Dtype, Vec<usize>) {
        match self {
            Self::StreamsIn | Self::PostAttn | Self::StreamsOut => (Dtype::F32, vec![hc, h]),
            Self::AttnPost | Self::FfnPost => (Dtype::F32, vec![hc]),
            Self::AttnComb | Self::FfnComb => (Dtype::F32, vec![hc, hc]),
            Self::DsaTopk => (Dtype::I32, vec![topk]),
            Self::MoeIds => (Dtype::I32, vec![moe_topk]),
            Self::HiddenIn
            | Self::AttnNormed
            | Self::MixerOut
            | Self::MixerRed
            | Self::FfnNormed
            | Self::MlpOut
            | Self::MlpRed
            | Self::FinalHidden => (Dtype::Bf16, vec![h]),
        }
    }

    /// 2026-10-13: Whether, under sequence-parallel mHC, a rank holds this point only for its
    /// own shard of rows (the mHC state and the reduced sums); the rest (normed rows after the
    /// all-gather, the mixer and MLP outputs, the final hidden) exist on every rank for every
    /// row. 2026-10-22: The MoE ids too: every rank routes every row of the launch.
    pub fn own_rows_only(self) -> bool {
        matches!(
            self,
            Self::StreamsIn
                | Self::AttnPost
                | Self::AttnComb
                | Self::MixerRed
                | Self::PostAttn
                | Self::FfnPost
                | Self::FfnComb
                | Self::MlpRed
                | Self::StreamsOut
        )
    }
}

/// 2026-10-13: What a layer call's caller knows of the dump: the path, the sub-chunk index
/// within the pass (the row group for the batched paths) and, for a prefill pass, the absolute
/// position of the pass's last row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Call {
    pub path: Path,
    pub chunk: usize,
    pub last_pos: Option<usize>,
}

impl Call {
    /// 2026-10-13: A call outside a prefill pass: no last row of its own.
    pub fn step(path: Path) -> Self {
        Self {
            path,
            chunk: 0,
            last_pos: None,
        }
    }
}

/// 2026-10-13: Launch rows `row0..row0 + rows` are sequence `seq`'s positions
/// `first_pos..first_pos + rows`; `last_pos` is that sequence's prefill-pass last position.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Window {
    pub row0: usize,
    pub rows: usize,
    pub seq: usize,
    pub first_pos: usize,
    pub last_pos: Option<usize>,
}

impl Window {
    /// 2026-10-13: A launch of one sequence's `rows` positions from `first_pos`, at row 0.
    pub fn single(rows: usize, first_pos: usize, last_pos: Option<usize>) -> Self {
        Self {
            row0: 0,
            rows,
            seq: 0,
            first_pos,
            last_pos,
        }
    }
}

/// 2026-10-13: A launch row to dump and the sequence position it holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DumpRow {
    pub row: usize,
    pub seq: usize,
    pub pos: usize,
}

/// 2026-10-13: The rows of a launch to dump, in launch-row order: each window's positions that
/// are in `wanted` or are its `last_pos`, each at launch row `row0 + pos - first_pos`.
pub fn select(wanted: Option<&BTreeSet<usize>>, windows: &[Window]) -> Vec<DumpRow> {
    let mut out = Vec::new();
    for w in windows {
        let span = w.first_pos..w.first_pos + w.rows;
        let mut pos: BTreeSet<usize> = wanted
            .map(|s| s.range(span.clone()).copied().collect())
            .unwrap_or_default();
        pos.extend(w.last_pos.filter(|p| span.contains(p)));
        out.extend(pos.into_iter().map(|p| DumpRow {
            row: w.row0 + p - w.first_pos,
            seq: w.seq,
            pos: p,
        }));
    }
    out.sort_by_key(|r| r.row);
    out
}

/// 2026-10-13: One dumped row: where it came from and what it holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    pub layer: usize,
    pub point: Point,
    pub rank: usize,
    pub path: Path,
    /// 2026-10-13: This layer's call ordinal on this rank (every call while the lever is on).
    pub call: usize,
    pub chunk: usize,
    pub seq: usize,
    pub pos: usize,
    pub row: usize,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
}

impl Entry {
    pub fn nbytes(&self) -> usize {
        self.dtype.size() * self.shape.iter().product::<usize>()
    }

    pub fn file_name(&self) -> String {
        format!(
            "L{:02}_{}_r{}_{}_n{:05}_c{}_s{}_pos{}.bin",
            self.layer,
            self.point.as_str(),
            self.rank,
            self.path.as_str(),
            self.call,
            self.chunk,
            self.seq,
            self.pos
        )
    }

    /// 2026-10-13: The manifest line (JSON, no newline).
    pub fn manifest_line(&self) -> String {
        serde_json::json!({
            "layer": self.layer,
            "point": self.point.as_str(),
            "rank": self.rank,
            "path": self.path.as_str(),
            "call": self.call,
            "chunk": self.chunk,
            "seq": self.seq,
            "pos": self.pos,
            "row": self.row,
            "dtype": self.dtype.as_str(),
            "nbytes": self.nbytes(),
            "shape": self.shape,
            "file": self.file_name(),
        })
        .to_string()
    }
}

#[cfg(test)]
mod tests;
