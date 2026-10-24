// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: The rotation against the Sylvester Hadamard matrix, its inverse, and the sign
//! tables against the kernel header they must equal.

mod common;

use common::Rng;
use metrale_kv_tq128::rotation::{INV_SQRT_128, SIGNS1, SIGNS2, fwht128, rotate, unrotate};

fn hadamard(i: usize, j: usize) -> f64 {
    if (i & j).count_ones().is_multiple_of(2) {
        1.0
    } else {
        -1.0
    }
}

#[test]
fn the_fwht_equals_the_sylvester_hadamard_matrix() {
    let mut rng = Rng::new(10);
    for _ in 0..50 {
        let x: Vec<f32> = rng.normals(128, 1.0);
        let mut v: [f32; 128] = x.clone().try_into().unwrap();
        fwht128(&mut v);
        for (i, &got) in v.iter().enumerate() {
            let want: f64 = (0..128).map(|j| hadamard(i, j) * f64::from(x[j])).sum();
            assert!(
                (f64::from(got) - want).abs() < 1e-4,
                "row {i}: {got} vs {want}"
            );
        }
    }
    // Integer inputs are exact: e_j maps to column j.
    for j in [0, 1, 77, 127] {
        let mut v = [0.0_f32; 128];
        v[j] = 1.0;
        fwht128(&mut v);
        for (i, &got) in v.iter().enumerate() {
            assert_eq!(f64::from(got), hadamard(i, j));
        }
    }
}

#[test]
fn rotate_is_orthogonal_and_unrotate_inverts_it() {
    let mut rng = Rng::new(11);
    for _ in 0..200 {
        let x: [f32; 128] = rng.normals(128, 0.3).try_into().unwrap();
        let mut v = x;
        rotate(&mut v);
        let n0: f64 = x.iter().map(|a| f64::from(*a).powi(2)).sum();
        let n1: f64 = v.iter().map(|a| f64::from(*a).powi(2)).sum();
        assert!((n0 - n1).abs() < 1e-5 * n0, "norm {n0} -> {n1}");
        unrotate(&mut v);
        for (a, b) in x.iter().zip(&v) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }
    // rotate(e_0) = signs1[0] * signs2 / sqrt(128): the closed form the encode KAT relies on.
    let mut e0 = [0.0_f32; 128];
    e0[0] = 1.0;
    rotate(&mut e0);
    for (i, &v) in e0.iter().enumerate() {
        assert_eq!(v, f32::from(SIGNS1[0] * SIGNS2[i]) * INV_SQRT_128);
    }
}

#[test]
fn inv_sqrt_128_is_correctly_rounded() {
    let exact = 1.0 / 128f64.sqrt();
    let got = f64::from(INV_SQRT_128);
    let ulp = f64::from(INV_SQRT_128.next_up()) - got;
    assert!((got - exact).abs() <= ulp / 2.0, "{got} vs {exact}");
    assert_eq!(INV_SQRT_128, (1.0 / 128f64.sqrt()) as f32);
}

fn header_table(src: &str, name: &str) -> Vec<i8> {
    let start = src
        .find(&format!("{name}[128] = {{"))
        .expect("table present");
    let body = &src[start..];
    let body = &body[body.find('{').unwrap() + 1..body.find('}').unwrap()];
    body.split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(|t| match t {
            "1.0f" => 1,
            "-1.0f" => -1,
            other => panic!("unexpected sign token {other:?}"),
        })
        .collect()
}

#[test]
fn the_sign_tables_equal_the_kernel_header() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../kernels/gb10/common/tq_plus_signs.cuh"
    );
    let src = std::fs::read_to_string(path).expect("kernel sign header");
    assert_eq!(header_table(&src, "TQP_SIGNS1_128"), SIGNS1.to_vec());
    assert_eq!(header_table(&src, "TQP_SIGNS2_128"), SIGNS2.to_vec());
}
