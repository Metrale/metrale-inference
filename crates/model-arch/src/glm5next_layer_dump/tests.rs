// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: The layer dump's pure parts (lever parse, row selection, naming, layouts) and
//! the tap's copy against the mock backend: which rows are copied from where, and the
//! sequence-parallel shard filter.

use std::sync::Mutex;

use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::tap::{Launch, Layer, Sink, open};
use super::*;

fn set(v: &[usize]) -> BTreeSet<usize> {
    v.iter().copied().collect()
}

/// 2026-10-17: [`config`] over the settings reads every pair of directory and row-list probes
/// as the replaced environment reader did (`parse` of the two variables' text): both are text,
/// passed verbatim.
#[test]
fn config_reads_the_settings_as_the_environment_reader_did() {
    use metrale_model_layers::lever_equivalence::{PROBES, settings_from};
    let dirs = PROBES
        .iter()
        .copied()
        .chain([Some("out"), Some(""), Some(" d ")]);
    for dir in dirs {
        let rows = PROBES
            .iter()
            .copied()
            .chain([Some(" 7, 3,7 "), Some(""), Some("4,x")]);
        for rows in rows {
            let mut pairs = Vec::new();
            pairs.extend(dir.map(|v| (DIR_ENV, v)));
            pairs.extend(rows.map(|v| (ROWS_ENV, v)));
            let s = settings_from(&pairs).expect("text levers resolve");
            assert_eq!(config(&s).ok(), parse(dir, rows).ok(), "{dir:?} {rows:?}");
        }
    }
}

#[test]
fn parse_accepts_dir_with_and_without_rows() {
    assert_eq!(parse(None, None), Ok(None));
    assert_eq!(
        parse(Some("out"), None),
        Ok(Some(DumpConfig {
            dir: PathBuf::from("out"),
            rows: None
        }))
    );
    assert_eq!(
        parse(Some("out"), Some(" 7, 3,7 ")),
        Ok(Some(DumpConfig {
            dir: PathBuf::from("out"),
            rows: Some(set(&[3, 7]))
        }))
    );
}

#[test]
fn parse_rejects_malformed_values() {
    for (dir, rows) in [
        (Some(""), None),
        (None, Some("3")),
        (Some("out"), Some("")),
        (Some("out"), Some("3,,4")),
        (Some("out"), Some("-1")),
        (Some("out"), Some("x")),
    ] {
        assert!(parse(dir, rows).is_err(), "{dir:?} {rows:?} parsed");
    }
}

fn win(row0: usize, rows: usize, seq: usize, first_pos: usize, last: Option<usize>) -> Window {
    Window {
        row0,
        rows,
        seq,
        first_pos,
        last_pos: last,
    }
}

fn dr(row: usize, seq: usize, pos: usize) -> DumpRow {
    DumpRow { row, seq, pos }
}

#[test]
fn select_maps_positions_to_launch_rows_across_windows() {
    // 2026-10-13: Two sequences packed in one launch: seq 4 at rows 0..3 holding positions
    // 10..13, seq 2 at rows 3..5 holding 100..102. 13 is one past seq 4's window and 50 is in
    // none, so only 11 (row 1) and 101 (row 4) are this launch's.
    let wanted = set(&[11, 50, 101, 13]);
    let got = select(
        Some(&wanted),
        &[win(0, 3, 4, 10, None), win(3, 2, 2, 100, None)],
    );
    assert_eq!(got, vec![dr(1, 4, 11), dr(4, 2, 101)]);
}

#[test]
fn select_adds_last_pos_once_and_sorts_by_row() {
    let wanted = set(&[5, 7]);
    // 2026-10-13: Windows given out of row order; the last position 7 is also wanted (one row,
    // not two), and seq 1's last position 21 is dumped although not wanted.
    let got = select(
        Some(&wanted),
        &[win(4, 4, 1, 18, Some(21)), win(0, 4, 0, 4, Some(7))],
    );
    assert_eq!(
        got,
        vec![dr(1, 0, 5), dr(3, 0, 7), dr(7, 1, 21)],
        "dedup, sort, last row of each window"
    );
}

#[test]
fn select_without_rows_takes_only_in_window_last_positions() {
    // 2026-10-13: A last position outside its window (a sub-chunk before the pass's last) is
    // not this launch's row.
    let got = select(None, &[win(0, 4, 0, 0, Some(9)), win(4, 1, 1, 9, Some(9))]);
    assert_eq!(got, vec![dr(4, 1, 9)]);
    assert!(select(None, &[win(0, 8, 0, 0, None)]).is_empty());
}

fn entry(point: Point) -> Entry {
    let (dtype, shape) = point.row_layout(6, 4, 16);
    Entry {
        layer: 3,
        point,
        rank: 1,
        path: Path::SpanPrefill,
        call: 42,
        chunk: 2,
        seq: 5,
        pos: 1234,
        row: 9,
        dtype,
        shape,
    }
}

#[test]
fn file_name_and_manifest_line_carry_every_key() {
    let e = entry(Point::AttnComb);
    assert_eq!(
        e.file_name(),
        "L03_attn_comb_r1_sprefill_n00042_c2_s5_pos1234.bin"
    );
    let v: serde_json::Value = serde_json::from_str(&e.manifest_line()).unwrap();
    assert_eq!(v["layer"], 3);
    assert_eq!(v["point"], "attn_comb");
    assert_eq!(v["rank"], 1);
    assert_eq!(v["path"], "sprefill");
    assert_eq!(v["call"], 42);
    assert_eq!(v["chunk"], 2);
    assert_eq!(v["seq"], 5);
    assert_eq!(v["pos"], 1234);
    assert_eq!(v["row"], 9);
    assert_eq!(v["dtype"], "f32");
    assert_eq!(v["shape"], serde_json::json!([4, 4]));
    assert_eq!(v["nbytes"], 64);
    assert_eq!(v["file"], e.file_name());
}

#[test]
fn row_layouts_match_the_buffers() {
    let (h, hc, topk) = (6, 4, 16);
    let cases = [
        (Point::HiddenIn, Dtype::Bf16, h * 2),
        (Point::StreamsIn, Dtype::F32, hc * h * 4),
        (Point::AttnPost, Dtype::F32, hc * 4),
        (Point::AttnComb, Dtype::F32, hc * hc * 4),
        (Point::AttnNormed, Dtype::Bf16, h * 2),
        (Point::MixerOut, Dtype::Bf16, h * 2),
        (Point::DsaTopk, Dtype::I32, topk * 4),
        (Point::MixerRed, Dtype::Bf16, h * 2),
        (Point::PostAttn, Dtype::F32, hc * h * 4),
        (Point::FfnPost, Dtype::F32, hc * 4),
        (Point::FfnComb, Dtype::F32, hc * hc * 4),
        (Point::FfnNormed, Dtype::Bf16, h * 2),
        (Point::MlpOut, Dtype::Bf16, h * 2),
        (Point::MlpRed, Dtype::Bf16, h * 2),
        (Point::StreamsOut, Dtype::F32, hc * h * 4),
        (Point::FinalHidden, Dtype::Bf16, h * 2),
    ];
    for (p, dtype, bytes) in cases {
        let e = entry(p);
        assert_eq!((e.dtype, e.nbytes()), (dtype, bytes), "{}", p.as_str());
    }
}

#[test]
fn own_rows_only_is_the_mhc_state_and_the_reduced_sums() {
    let own = [
        Point::StreamsIn,
        Point::AttnPost,
        Point::AttnComb,
        Point::MixerRed,
        Point::PostAttn,
        Point::FfnPost,
        Point::FfnComb,
        Point::MlpRed,
        Point::StreamsOut,
    ];
    let all = [
        Point::HiddenIn,
        Point::AttnNormed,
        Point::MixerOut,
        Point::DsaTopk,
        Point::FfnNormed,
        Point::MlpOut,
        Point::FinalHidden,
    ];
    assert!(own.iter().all(|p| p.own_rows_only()));
    assert!(all.iter().all(|p| !p.own_rows_only()));
}

/// 2026-10-13: Keeps every written row in memory.
#[derive(Default)]
struct MemSink(Mutex<Vec<(Entry, Vec<u8>)>>);

impl Sink for MemSink {
    fn write(&self, entry: &Entry, bytes: &[u8]) -> anyhow::Result<()> {
        self.0.lock().unwrap().push((entry.clone(), bytes.to_vec()));
        Ok(())
    }
}

/// 2026-10-13: A device buffer of `rows` BF16 rows of `h` elements whose every byte names its
/// row (`row + 1`), so a copy from the wrong row is visible.
fn rows_buffer(gpu: &MockGpuBackend, rows: usize, h: usize) -> DevicePtr {
    let p = gpu.alloc(rows * h * 2).unwrap();
    let bytes: Vec<u8> = (0..rows)
        .flat_map(|r| std::iter::repeat_n(r as u8 + 1, h * 2))
        .collect();
    gpu.copy_h2d(&bytes, p).unwrap();
    p
}

fn launch(layer_idx: usize, own: Option<(usize, usize)>) -> Launch {
    Launch {
        layer: Layer {
            idx: layer_idx,
            hidden: 8,
            hc: 4,
            topk: None,
        },
        call: Call {
            path: Path::Prefill,
            chunk: 1,
            last_pos: Some(13),
        },
        rank: 0,
        own,
    }
}

#[test]
fn tap_copies_each_selected_row_from_its_offset() {
    let gpu = MockGpuBackend::new();
    let sink = MemSink::default();
    let base = rows_buffer(&gpu, 6, 8);
    let wanted = set(&[9]);
    // 2026-10-13: Rows 0..6 hold positions 8..14: position 9 is row 1, the last (13) row 5.
    let tap = open(
        &gpu,
        7,
        &sink,
        Some(&wanted),
        launch(900, None),
        &[win(0, 6, 0, 8, Some(13))],
    )
    .expect("rows selected");
    tap.put(Point::MlpOut, base).unwrap();
    let got = sink.0.into_inner().unwrap();
    assert_eq!(got.len(), 2);
    for ((e, bytes), (row, pos)) in got.iter().zip([(1usize, 9usize), (5, 13)]) {
        assert_eq!((e.row, e.pos, e.layer, e.chunk), (row, pos, 900, 1));
        assert_eq!(bytes.len(), 16);
        assert!(bytes.iter().all(|&b| b == row as u8 + 1), "row {row}");
    }
}

#[test]
fn tap_skips_rows_outside_the_shard_for_own_points_only() {
    let gpu = MockGpuBackend::new();
    let sink = MemSink::default();
    let base = rows_buffer(&gpu, 6, 8);
    let wanted = set(&[8, 12]);
    // 2026-10-13: This rank owns rows 3..6: row 0 (position 8) is someone else's for the
    // reduced sum, but every rank holds it for the MLP's partial output.
    let tap = open(
        &gpu,
        7,
        &sink,
        Some(&wanted),
        launch(901, Some((3, 6))),
        &[win(0, 6, 0, 8, None)],
    )
    .expect("rows selected");
    tap.put(Point::MlpRed, base).unwrap();
    tap.put(Point::MlpOut, base).unwrap();
    let got: Vec<_> = sink
        .0
        .into_inner()
        .unwrap()
        .into_iter()
        .map(|(e, b)| (e.point, e.row, b[0]))
        .collect();
    assert_eq!(
        got,
        vec![
            (Point::MlpRed, 4, 5),
            (Point::MlpOut, 0, 1),
            (Point::MlpOut, 4, 5)
        ]
    );
}

#[test]
fn tap_put_rows_limits_to_the_piece_and_open_counts_calls() {
    let gpu = MockGpuBackend::new();
    let sink = MemSink::default();
    let base = rows_buffer(&gpu, 6, 8);
    let wanted = set(&[0, 1, 2, 3, 4, 5]);
    let windows = [win(0, 6, 0, 0, None)];
    // 2026-10-13: The first call selects nothing and still takes ordinal 0.
    assert!(open(&gpu, 7, &sink, None, launch(902, None), &windows).is_none());
    let tap = open(&gpu, 7, &sink, Some(&wanted), launch(902, None), &windows).unwrap();
    tap.put_rows(Point::MixerOut, base, 2..4).unwrap();
    let got: Vec<_> = sink
        .0
        .into_inner()
        .unwrap()
        .into_iter()
        .map(|(e, b)| (e.call, e.row, b[0]))
        .collect();
    assert_eq!(got, vec![(1, 2, 3), (1, 3, 4)]);
}
