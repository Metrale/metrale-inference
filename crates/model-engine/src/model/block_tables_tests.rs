// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: Tests for the incremental block-table upload: after every step the device rows
//! equal the full rows the replaced upload wrote, and a steady step's copies follow the live
//! context, not the stride (`--max-seq-len`).

use super::{BlockTables, Region, RowSpec, Span, TableMirror};
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;

const DUMMY: i32 = 7;

/// 2026-10-13: The full row the replaced upload wrote: `blocks` cut at the stride, then fill.
fn full_row(spec: &RowSpec<'_>, stride: usize) -> Vec<i32> {
    (0..stride).map(|i| spec.at(i)).collect()
}

fn apply(device: &mut [i32], spans: &[Span]) {
    for s in spans {
        device[s.start..s.start + s.values.len()].copy_from_slice(&s.values);
    }
}

fn entries(spans: &[Span]) -> usize {
    spans.iter().map(|s| s.values.len()).sum()
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: u64) -> usize {
        (self.next() % n) as usize
    }
}

/// 2026-10-13: Random steps (rows grow, shrink, swap sequences, change fill, the row count
/// changes, the mirror is forgotten over a scribbled device) against a device that starts as
/// garbage: every described row must read back as its full row after every step.
#[test]
fn every_step_leaves_the_described_rows_equal_to_their_full_rows() {
    let (stride, cap) = (24, 6);
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut device = vec![-99i32; stride * cap];
    let mut mirror = TableMirror::new(stride, cap);
    let mut tables: Vec<Vec<u32>> = (0..cap).map(|r| vec![r as u32 + 100]).collect();
    let mut fills = vec![DUMMY; cap];
    for step in 0..4000 {
        for r in 0..cap {
            match rng.below(10) {
                0..=4 => {}
                5 | 6 => tables[r].push(rng.below(50) as u32),
                7 => {
                    let keep = rng.below(tables[r].len() as u64 + 1);
                    tables[r].truncate(keep);
                }
                8 => tables[r] = tables[rng.below(cap as u64)].clone(),
                _ => fills[r] = if fills[r] == DUMMY { 0 } else { DUMMY },
            }
            // 2026-10-13: Entries past the stride exist (cut off) and entries equal to the
            // fill sit inside the live prefix, both edge cases of the diff.
            if tables[r].len() > stride + 3 {
                tables[r].truncate(rng.below(stride as u64 + 3));
            }
        }
        if step % 97 == 0 {
            device.iter_mut().for_each(|v| *v = -5);
            mirror.forget();
        }
        let n = 1 + rng.below(cap as u64);
        let specs: Vec<RowSpec<'_>> = (0..n)
            .map(|r| RowSpec {
                blocks: &tables[r],
                fill: fills[r],
            })
            .collect();
        let spans = mirror.plan(&specs).unwrap();
        apply(&mut device, &spans);
        for (r, spec) in specs.iter().enumerate() {
            assert_eq!(
                device[r * stride..(r + 1) * stride],
                full_row(spec, stride)[..],
                "step {step} row {r}"
            );
        }
    }
}

/// 2026-10-13: 16 rows at 1000 tokens (block 16) decoding 400 steps: after the first step the
/// copies are the new block entries only, the same count at a 64k, 256k and 512k stride.
#[test]
fn a_steady_decode_copies_the_same_entries_at_every_max_seq_len() {
    let per_stride: Vec<(usize, usize)> = [65536usize, 262144, 524288]
        .iter()
        .map(|&msl| {
            let stride = msl / 16 + 1;
            let mut mirror = TableMirror::new(stride, 16);
            let mut tables: Vec<Vec<u32>> = (0..16u32)
                .map(|r| (0..63).map(|b| r * 1000 + b).collect())
                .collect();
            let mut lens = [1000usize; 16];
            let mut first = 0;
            let mut later = 0;
            for step in 0..400 {
                for (r, t) in tables.iter_mut().enumerate() {
                    lens[r] += 1;
                    if lens[r].div_ceil(16) > t.len() {
                        t.push(r as u32 * 1000 + t.len() as u32);
                    }
                }
                let specs: Vec<RowSpec<'_>> = tables
                    .iter()
                    .map(|t| RowSpec {
                        blocks: t,
                        fill: DUMMY,
                    })
                    .collect();
                let n = entries(&mirror.plan(&specs).unwrap());
                if step == 0 { first = n } else { later += n }
            }
            (first, later)
        })
        .collect();
    assert_eq!(
        per_stride[0].0,
        16 * (65536 / 16 + 1),
        "the first step writes whole rows"
    );
    let new_blocks: usize = 16 * ((1400usize.div_ceil(16)) - (1001usize.div_ceil(16)));
    for &(_, later) in &per_stride {
        assert_eq!(later, new_blocks, "{per_stride:?}");
    }
}

#[test]
fn an_unchanged_step_copies_nothing_and_whole_unknown_rows_merge_into_one_copy() {
    let stride = 9;
    let mut mirror = TableMirror::new(stride, 4);
    let t = [vec![1u32, 2], vec![3], vec![]];
    let specs: Vec<RowSpec<'_>> = t
        .iter()
        .map(|b| RowSpec {
            blocks: b,
            fill: DUMMY,
        })
        .collect();
    let first = mirror.plan(&specs).unwrap();
    assert_eq!(first.len(), 1);
    assert_eq!((first[0].start, first[0].values.len()), (0, 3 * stride));
    assert!(mirror.plan(&specs).unwrap().is_empty());
    // 2026-10-13: Row 1 drops to no blocks: one entry back to the fill, at row 1 entry 0.
    let shrunk = [
        specs[0],
        RowSpec {
            blocks: &[],
            fill: DUMMY,
        },
    ];
    assert_eq!(
        mirror.plan(&shrunk).unwrap(),
        vec![Span {
            start: stride,
            values: vec![DUMMY]
        }]
    );
}

#[test]
fn more_rows_than_the_region_holds_is_refused() {
    let mut mirror = TableMirror::new(4, 2);
    let specs = [RowSpec {
        blocks: &[],
        fill: 0,
    }; 3];
    assert!(mirror.plan(&specs).is_err());
}

/// 2026-10-13: Through the GPU trait: the two regions hold their own rows (a verify upload
/// never touches decode rows), and each reads back as its full rows across steps.
#[test]
fn the_decode_and_verify_regions_read_back_as_full_rows() {
    let gpu = MockGpuBackend::new();
    let stride = 5;
    let bt = BlockTables::with_rows(&gpu, stride, 2, 3).unwrap();
    let read = |region: Region, rows: usize| -> Vec<i32> {
        let mut buf = vec![0u8; rows * stride * 4];
        gpu.copy_d2h(bt.ptr(region), &mut buf).unwrap();
        buf.chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    let decode = [vec![10u32, 11], vec![12]];
    let verify = [vec![20u32], vec![20], vec![DUMMY as u32]];
    for grow in 0..3u32 {
        let d: Vec<Vec<u32>> = decode
            .iter()
            .map(|t| [&t[..], &[30 + grow]].concat())
            .collect();
        let d_specs: Vec<RowSpec<'_>> = d
            .iter()
            .map(|b| RowSpec {
                blocks: b,
                fill: DUMMY,
            })
            .collect();
        let v_specs: Vec<RowSpec<'_>> = verify
            .iter()
            .map(|b| RowSpec { blocks: b, fill: 0 })
            .collect();
        bt.upload(Region::Decode, &gpu, &d_specs, 0).unwrap();
        bt.upload(Region::Verify, &gpu, &v_specs, 0).unwrap();
        let want_d: Vec<i32> = d_specs.iter().flat_map(|s| full_row(s, stride)).collect();
        let want_v: Vec<i32> = v_specs.iter().flat_map(|s| full_row(s, stride)).collect();
        assert_eq!(read(Region::Decode, 2), want_d, "grow {grow}");
        assert_eq!(read(Region::Verify, 3), want_v, "grow {grow}");
    }
}
