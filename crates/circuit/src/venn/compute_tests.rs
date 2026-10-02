// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: `ComputeUnit` parsing and a group's unit from its kernels'.
//!
//! Owner: metrale-circuit (venn).
//! Invariants: none beyond the types.

use super::*;

fn tc(atom: &str) -> ComputeUnit {
    ComputeUnit::TensorCore { atom: atom.into() }
}

#[test]
fn a_unit_parses_only_with_the_atom_it_needs() {
    assert_eq!(
        ComputeUnit::parse("tensor_core", Some("mma.sync.m16n8k32.e4m3")),
        Ok(tc("mma.sync.m16n8k32.e4m3"))
    );
    assert_eq!(
        ComputeUnit::parse("cuda_core", None),
        Ok(ComputeUnit::CudaCore)
    );
    assert_eq!(ComputeUnit::parse("memory", None), Ok(ComputeUnit::Memory));
    assert!(ComputeUnit::parse("tensor_core", None).is_err());
    assert!(ComputeUnit::parse("tensor_core", Some(" ")).is_err());
    assert!(ComputeUnit::parse("cuda_core", Some("mma.sync.m16n8k16.bf16")).is_err());
    assert!(ComputeUnit::parse("simt", None).is_err());
}

#[test]
fn a_group_runs_on_tensor_cores_when_any_kernel_issues_mmas() {
    let quant = ComputeUnit::Memory;
    let mma = tc("mma.sync.m16n8k64.mxf4nvf4.block_scale");
    assert_eq!(group_unit([&quant, &mma, &quant]), Some(mma.clone()));
    assert_eq!(
        group_unit([&quant, &ComputeUnit::CudaCore]),
        Some(ComputeUnit::CudaCore)
    );
    assert_eq!(group_unit([&quant]), Some(ComputeUnit::Memory));
    assert_eq!(group_unit([]), None);
}

#[test]
fn a_kernel_override_wins_over_the_family_unit() {
    let k = |f: &str| KernelId {
        module: "m".into(),
        func: f.into(),
    };
    let fc = FamilyCompute {
        unit: tc("mma.sync.m16n8k16.bf16"),
        kernels: [(k("gemv"), ComputeUnit::CudaCore)].into_iter().collect(),
    };
    assert_eq!(fc.of(&k("gemv")), &ComputeUnit::CudaCore);
    assert!(fc.of(&k("gemm")).is_tensor_core());
}
