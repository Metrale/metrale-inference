// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: Value statistics: classes, reads, counting, the text round trip and sampling.
//!
//! Owner: metrale-ml-utils.
//! Invariants: the sampling check counts decoded draws against the class counts with a
//! tolerance derived from the sample size; determinism is checked element by element across the
//! threaded chunk boundaries.

use super::*;
use crate::index::TensorEntry;
use crate::routing::total_variation;
use crate::testkit;

fn schedule() -> LayerSchedule {
    metrale_circuit::layer_schedule(&testkit::moe_fp8().0).unwrap()
}

#[test]
fn a_class_stars_the_layer_and_every_digit_segment_and_names_the_dtype() {
    let s = schedule();
    assert_eq!(
        class_key(
            &s,
            "model.language_model.layers.3.mlp.experts.17.down_proj.weight",
            Dtype::F8E4m3
        ),
        "L.mlp.experts.*.down_proj.weight|F8_E4M3"
    );
    assert_eq!(
        class_key(
            &s,
            "model.language_model.layers.0.mlp.experts.2.down_proj.weight",
            Dtype::F8E4m3
        ),
        class_key(
            &s,
            "model.language_model.layers.7.mlp.experts.5.down_proj.weight",
            Dtype::F8E4m3
        ),
    );
    assert_eq!(
        class_key(&s, "model.language_model.embed_tokens.weight", Dtype::Bf16),
        "model.language_model.embed_tokens.weight|BF16"
    );
    assert_ne!(
        class_key(
            &s,
            "model.language_model.layers.3.mlp.gate.weight",
            Dtype::Bf16
        ),
        class_key(
            &s,
            "model.language_model.layers.3.mlp.gate.weight",
            Dtype::F32
        ),
    );
}

fn entry(name: &str, dtype: Dtype, shape: Vec<u64>, offset: u64) -> TensorEntry {
    TensorEntry {
        name: name.into(),
        dtype,
        shape,
        shard: "a.safetensors".into(),
        offset,
    }
}

#[test]
fn reads_cover_small_tensors_whole_and_large_ones_in_aligned_chunks_inside_the_tensor() {
    let s = schedule();
    let mut list = Vec::new();
    for l in 0..8u64 {
        let p = format!("model.language_model.layers.{l}.mlp.experts.0");
        list.push(entry(
            &format!("{p}.down_proj.weight"),
            Dtype::Bf16,
            vec![2048, 1024],
            1000 + l * (8 << 20),
        ));
        list.push(entry(
            &format!("{p}.up_proj.weight"),
            Dtype::F32,
            vec![4, 4],
            7 + l * 64,
        ));
    }
    list.push(entry(
        "model.language_model.layers.0.idx",
        Dtype::I32,
        vec![4],
        0,
    ));
    let index = TensorIndex::from_entries(list).unwrap();
    let reads = plan_reads(&s, &index);
    let large: Vec<_> = reads.iter().filter(|r| r.kind == Kind::Halves).collect();
    let small: Vec<_> = reads.iter().filter(|r| r.kind == Kind::Words).collect();
    assert!(
        reads.iter().all(|r| r.kind != Kind::Nibbles),
        "the I32 tensor is not counted"
    );
    assert_eq!(
        small.len(),
        TENSORS_PER_CLASS,
        "at most {TENSORS_PER_CLASS} tensors per class"
    );
    assert!(small.iter().all(|r| r.len == 64));
    assert_eq!(large.len(), TENSORS_PER_CLASS * CHUNKS as usize);
    let bytes = 2048 * 1024 * 2;
    for r in &large {
        let base = index
            .iter()
            .find(|e| e.dtype == Dtype::Bf16 && e.offset <= r.offset && r.offset < e.offset + bytes)
            .unwrap();
        assert!(
            r.offset + r.len <= base.offset + bytes,
            "a chunk past its tensor"
        );
        assert_eq!((r.offset - base.offset) % 2, 0, "a chunk splits an element");
    }
    let first = large.iter().map(|r| r.offset).min().unwrap();
    let last = large
        .iter()
        .map(|r| r.offset + r.len)
        .filter(|&e| e <= 1000 + bytes)
        .max()
        .unwrap();
    assert_eq!(
        (first, last),
        (1000, 1000 + bytes),
        "the first and last chunks reach both ends"
    );
}

fn read(kind: Kind) -> Read {
    Read {
        class: "c|X".into(),
        kind,
        shard: "a".into(),
        offset: 0,
        len: 0,
    }
}

#[test]
fn counting_splits_nibbles_low_first_and_reads_little_endian_elements() {
    let mut acc = Accumulator::default();
    acc.add(&read(Kind::Nibbles), &[0x21, 0x11]);
    let (s, _) = acc.finish("x");
    let want: BTreeMap<u32, u64> = [(1, 3), (2, 1)].into();
    assert_eq!(s.classes["c|X"].counts, want);

    let mut acc = Accumulator::default();
    acc.add(&read(Kind::Halves), &[0x80, 0x3F, 0x80, 0x3F, 0x00, 0x40]);
    let (s, _) = acc.finish("x");
    let want: BTreeMap<u32, u64> = [(0x3F80, 2), (0x4000, 1)].into();
    assert_eq!(s.classes["c|X"].counts, want);
    // 2026-10-04: values 1, 1, 2: sqrt((1 + 1 + 4) / 3).
    assert!((s.classes["c|X"].bf16_rms().unwrap() - 2f32.sqrt()).abs() < 1e-6);

    let mut acc = Accumulator::default();
    acc.add(&read(Kind::Words), &1.5f32.to_le_bytes());
    let (s, _) = acc.finish("x");
    assert_eq!(
        s.classes["c|X"].counts.keys().copied().collect::<Vec<_>>(),
        [1.5f32.to_bits()]
    );
    assert_eq!(
        s.classes["c|X"].bf16_rms(),
        None,
        "RMS is defined for BF16 classes only"
    );
}

#[test]
fn the_text_round_trips_and_its_digest_is_the_texts() {
    let (config, index) = testkit::moe_fp8();
    let schedule = metrale_circuit::layer_schedule(&config).unwrap();
    let mut acc = Accumulator::default();
    for r in plan_reads(&schedule, &index) {
        acc.add(&r, &testkit::toy_read_bytes(&r));
    }
    let (stats, text) = acc.finish("toy/model");
    let back = ValueStats::parse(&text).unwrap();
    assert_eq!(back, stats);
    assert_eq!(
        back.digest,
        hex(&<sha2::Sha256 as sha2::Digest>::digest(text.as_bytes()))
    );
    assert!(
        back.classes
            .contains_key("L.mlp.experts.*.down_proj.weight|F8_E4M3")
    );
}

#[test]
fn malformed_statistics_are_refused() {
    let ok = r#"{"schema":1,"source":"s","classes":{"c|BF16":{"kind":"halves","counts":[[1,2]]}}}"#;
    ValueStats::parse(ok).unwrap();
    for (bad, why) in [
        (ok.replace("\"schema\":1", "\"schema\":2"), "schema 2"),
        (ok.replace("halves", "quarters"), "kind"),
        (ok.replace("[[1,2]]", "[[1,0]]"), "malformed"),
        (ok.replace("[[1,2]]", "[[4294967296,2]]"), "malformed"),
        (ok.replace("[[1,2]]", "[]"), "no counts"),
    ] {
        let err = ValueStats::parse(&bad).unwrap_err().to_string();
        assert!(err.contains(why), "{bad}: {err}");
    }
    let s = ValueStats::parse(ok).unwrap();
    let err = s.sampler("other|BF16").unwrap_err().to_string();
    assert!(err.contains("no class `other|BF16`"), "{err}");
}

fn stats_of(kind: Kind, counts: &[(u32, u64)]) -> ValueStats {
    let mut classes = BTreeMap::new();
    classes.insert(
        "c".to_string(),
        ClassStats {
            kind,
            counts: counts.iter().copied().collect(),
        },
    );
    ValueStats {
        source: "x".into(),
        classes,
        digest: String::new(),
    }
}

#[test]
fn samples_follow_the_class_distribution_for_every_kind() {
    let s = Stream::for_tensor(3, "t", "x", &[1]);
    let cases: [(Kind, &[(u32, u64)]); 4] = [
        (Kind::Nibbles, &[(0, 50), (1, 30), (7, 15), (15, 5)]),
        (Kind::Bytes, &[(0x00, 10), (0x38, 60), (0xB8, 30)]),
        (Kind::Halves, &[(0x3F80, 1), (0xBC00, 3), (0x0000, 6)]),
        (Kind::Words, &[(1.0f32.to_bits(), 9), (0.5f32.to_bits(), 1)]),
    ];
    for (kind, counts) in cases {
        let sampler = stats_of(kind, counts).sampler("c").unwrap();
        let bytes = sampler.bytes(s, 1 << 18);
        let mut acc = Accumulator::default();
        acc.add(&read(kind), &bytes);
        let fin = acc.finish("x").0;
        let got = &fin.classes["c|X"].counts;
        let want: Vec<u64> = counts.iter().map(|c| c.1).collect();
        let seen: Vec<u64> = counts
            .iter()
            .map(|c| got.get(&c.0).copied().unwrap_or(0))
            .collect();
        assert_eq!(
            seen.iter().sum::<u64>(),
            got.values().sum::<u64>(),
            "{kind:?}: a pattern outside the class"
        );
        // 2026-10-04: >= 64k draws per kind; the multinomial TV is below 0.004 at that size.
        let tv = total_variation(&want, &seen);
        assert!(tv < 0.01, "{kind:?}: tv {tv} ({seen:?} vs {want:?})");
    }
}

#[test]
fn an_element_depends_only_on_the_stream_and_its_position() {
    let s = Stream::for_tensor(3, "t", "x", &[1]);
    for kind in [Kind::Nibbles, Kind::Bytes, Kind::Halves, Kind::Words] {
        let counts: Vec<(u32, u64)> = (0..16).map(|p| (p, 1 + u64::from(p))).collect();
        let sampler = stats_of(kind, &counts).sampler("c").unwrap();
        // 2026-10-04: 5 MiB spans several threaded chunks (each >= 1 MiB) on any box with >= 2
        // threads; every chunk's first and last elements are checked against direct draws.
        let n = 5 << 20;
        let big = sampler.bytes(s, n);
        assert_eq!(
            &big[..4096],
            &sampler.bytes(s, 4096)[..],
            "{kind:?}: a prefix differs"
        );
        let w = kind.width();
        for at in (0..n).step_by(1 << 18).chain([n - w]) {
            let at = at / w * w;
            let mut one = vec![0u8; w];
            sampler.fill(s, at, &mut one);
            assert_eq!(&big[at..at + w], &one[..], "{kind:?}: byte {at}");
        }
        assert_ne!(
            big,
            sampler.bytes(s.derive(1), n),
            "{kind:?}: another stream draws the same"
        );
    }
}
