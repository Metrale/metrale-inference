// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-10-19: The rank shares from the real config, the per-tensor cut table, and the read plan
//! against a byte matrix whose every byte names its position.

use metrale_config::{ModelConfig, TpSupport, parse_config};

use super::*;

pub(crate) const CONFIG: &str =
    include_str!("../../../../config/src/tests/fixtures/deepseek_v41_flash/config.json");

pub(crate) fn v41_rank(world: usize, rank: usize) -> ModelConfig {
    let mut c = parse_config(CONFIG).expect("the V4.1 config parses");
    c.tp_world_size = world;
    c.tp_rank = rank;
    c.shard_heads_for_tp(TpSupport::SharedKvHead).unwrap();
    c
}

#[test]
fn three_ranks_keep_24_24_16_heads_in_whole_groups_and_768_wide_experts() {
    let got: Vec<Local> = (0..3)
        .map(|r| Local::of(&v41_rank(3, r)).unwrap())
        .collect();
    let heads: Vec<_> = got.iter().map(|l| l.heads.clone()).collect();
    assert_eq!(heads, vec![0..24, 24..48, 48..64]);
    let groups: Vec<_> = got.iter().map(|l| l.groups.clone()).collect();
    assert_eq!(groups, vec![0..3, 3..6, 6..8]);
    let inter: Vec<_> = got.iter().map(|l| l.inter.clone()).collect();
    assert_eq!(inter, vec![0..768, 768..1536, 1536..2304]);
    assert!(
        got.iter()
            .all(|l| l.world == 3 && l.head_dim == 512 && l.o_rank == 1024)
    );
}

#[test]
fn one_rank_keeps_everything_and_cuts_nothing() {
    let l = Local::of(&v41_rank(1, 0)).unwrap();
    assert_eq!(
        (l.heads.clone(), l.groups.clone(), l.inter.clone()),
        (0..64, 0..8, 0..2304)
    );
    assert_eq!(Local::whole(&v41_rank(1, 0)), l);
    for t in [T::WqB, T::WoA, T::WoB, T::Sink, T::SharedW1, T::SharedW2] {
        assert_eq!(cut_of(t, &l), Cut::Whole, "{t:?}");
    }
}

#[test]
fn local_refuses_a_config_whose_heads_were_not_sharded_to_match() {
    let mut c = v41_rank(3, 2);
    c.num_attention_heads = 24;
    let e = Local::of(&c).unwrap_err().to_string();
    assert!(e.contains("keeps heads 48..64"), "{e}");
}

#[test]
fn the_cut_table_slices_exactly_the_head_and_intermediate_tensors() {
    let l = Local::of(&v41_rank(3, 1)).unwrap();
    assert_eq!(cut_of(T::Sink, &l), Cut::Rows(24..48));
    assert_eq!(cut_of(T::WqB, &l), Cut::Rows(24 * 512..48 * 512));
    assert_eq!(cut_of(T::WoA, &l), Cut::Rows(3 * 1024..6 * 1024));
    assert_eq!(cut_of(T::WoB, &l), Cut::Cols(3 * 1024..6 * 1024));
    assert_eq!(cut_of(T::SharedW1, &l), Cut::Rows(768..1536));
    assert_eq!(cut_of(T::SharedW3, &l), Cut::Rows(768..1536));
    assert_eq!(cut_of(T::SharedW2, &l), Cut::Cols(768..1536));
    for t in [
        T::WqA,
        T::QNorm,
        T::Wkv,
        T::KvNorm,
        T::CompWkv,
        T::CompWgate,
        T::CompNorm,
        T::IdxWqB,
        T::IdxProj,
        T::IdxWk,
        T::IdxKNorm,
        T::GateW,
        T::GateBias,
        T::AttnNorm,
        T::FfnNorm,
        T::EngramQ,
        T::EngramK,
        T::EngramWkv,
    ] {
        assert_eq!(cut_of(t, &l), Cut::Whole, "{t:?}");
    }
}

#[test]
fn div_scales_a_cut_and_refuses_a_partial_unit() {
    assert_eq!(Cut::Rows(64..128).div(32).unwrap(), Cut::Rows(2..4));
    assert_eq!(Cut::Cols(768..1536).div(2).unwrap(), Cut::Cols(384..768));
    assert_eq!(Cut::Whole.div(32).unwrap(), Cut::Whole);
    assert!(Cut::Rows(16..64).div(32).is_err());
    assert!(Cut::Cols(0..48).div(32).is_err());
}

/// 2026-10-19: A `[rows, cols]` matrix of `elem`-byte values whose bytes encode their position.
fn matrix(rows: usize, cols: usize, elem: usize) -> Vec<u8> {
    (0..rows * cols * elem)
        .map(|i| (i * 7 + i / 251) as u8)
        .collect()
}

fn run(m: &[u8], shape: &[usize], cut: &Cut) -> Vec<u8> {
    let p = read_plan(shape, m.len(), cut).unwrap();
    finish(&p, m[p.start..p.start + p.len].to_vec()).unwrap()
}

#[test]
fn a_row_cut_reads_only_its_rows_and_a_column_cut_keeps_whole_values() {
    let (rows, cols, elem) = (6, 5, 4);
    let m = matrix(rows, cols, elem);
    let rb = cols * elem;
    let p = read_plan(&[rows, cols], m.len(), &Cut::Rows(2..5)).unwrap();
    assert_eq!((p.start, p.len), (2 * rb, 3 * rb));
    assert_eq!(run(&m, &[rows, cols], &Cut::Rows(2..5)), m[2 * rb..5 * rb]);

    let got = run(&m, &[rows, cols], &Cut::Cols(1..3));
    let want: Vec<u8> = (0..rows)
        .flat_map(|r| m[r * rb + elem..r * rb + 3 * elem].to_vec())
        .collect();
    assert_eq!(got, want);
    assert_eq!(run(&m, &[rows, cols], &Cut::Whole), m);
}

#[test]
fn a_vector_cuts_by_rows_and_refuses_columns() {
    let m = matrix(64, 1, 4);
    assert_eq!(run(&m, &[64], &Cut::Rows(24..48)), m[24 * 4..48 * 4]);
    assert!(read_plan(&[64], m.len(), &Cut::Cols(0..1)).is_err());
}

#[test]
fn the_plan_refuses_ranges_outside_the_tensor_and_ragged_sizes() {
    assert!(read_plan(&[4, 4], 16, &Cut::Rows(3..5)).is_err());
    assert!(read_plan(&[4, 4], 16, &Cut::Rows(2..2)).is_err());
    assert!(read_plan(&[4, 4], 16, &Cut::Cols(2..5)).is_err());
    assert!(read_plan(&[4, 4], 17, &Cut::Whole).is_err());
    assert!(read_plan(&[2, 2, 2], 8, &Cut::Whole).is_err());
    let p = read_plan(&[4, 4], 16, &Cut::Whole).unwrap();
    assert!(finish(&p, vec![0; 15]).is_err());
}

#[test]
fn expert_tables_keep_the_ranks_rows_of_w1_w3_and_bytes_of_w2() {
    let l = Local::of(&v41_rank(3, 1)).unwrap();
    let got: Vec<Cut> = (0..6).map(|t| expert_cut(t, &l).unwrap()).collect();
    let rows = Cut::Rows(768..1536);
    assert_eq!(
        got,
        vec![
            rows.clone(),
            rows.clone(),
            rows.clone(),
            rows,
            Cut::Cols(384..768),
            Cut::Cols(24..48)
        ]
    );
    assert!(expert_cut(6, &l).is_err());
    let one = Local::of(&v41_rank(1, 0)).unwrap();
    assert!((0..6).all(|t| expert_cut(t, &one).unwrap() == Cut::Whole));
}
