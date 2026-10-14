// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: RAII pieces of the BF16 cuBLASLt GEMMs: the descriptors of one
//! `out[M,N] = act[M,K] @ weight[N,K]ᵀ`, a matmul preference, and the heuristic result record,
//! shared by the per-call heuristic path (`cublaslt.rs`) and the pinned one (`pinned.rs`).
//!
//! Owner: gpu-runtime (cuBLASLt wrapper).
//! Invariants: every descriptor created here is destroyed exactly once, on drop, also when a
//! later creation step fails.

use anyhow::Result;
use std::ffi::c_void;

use super::*;

/// 2026-10-14: One `cublasLtMatmulHeuristicResult_t` as the CUDA 13 cublasLt.h lays it out: the
/// 64-byte algo, then workspaceSize, state, wavesCount and reserved[4].
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct HeuristicResult {
    pub(super) algo: [u64; 8],
    pub(super) workspace_size: usize,
    // 2026-10-14: Read by no caller; present so the record has the header's 96-byte layout.
    #[allow(dead_code)]
    state: i32,
    #[allow(dead_code)]
    waves_count: f32,
    #[allow(dead_code)]
    reserved: [i32; 4],
}
const _: () = assert!(std::mem::size_of::<HeuristicResult>() == 96);

impl HeuristicResult {
    pub(super) const ZERO: Self = Self {
        algo: [0; 8],
        workspace_size: 0,
        state: 0,
        waves_count: 0.0,
        reserved: [0; 4],
    };
}

/// 2026-10-14: A matmul preference, destroyed on drop.
pub(super) struct Pref(cublasLtMatmulPreference_t);

impl Pref {
    pub(super) fn new() -> Result<Self> {
        let mut pref: cublasLtMatmulPreference_t = std::ptr::null_mut();
        chk(
            unsafe { cublasLtMatmulPreferenceCreate(&mut pref) },
            "PrefCreate",
        )?;
        Ok(Self(pref))
    }

    /// 2026-10-14: Set `attr` to `value`, whose type must be the attribute's (cublasLt.h).
    pub(super) fn set<T: Copy>(&self, attr: u32, value: &T) -> Result<()> {
        let size = std::mem::size_of::<T>();
        let buf = value as *const T as *const c_void;
        chk(
            unsafe { cublasLtMatmulPreferenceSetAttribute(self.0, attr, buf, size) },
            "PrefSetAttribute",
        )
    }
}

impl Drop for Pref {
    fn drop(&mut self) {
        unsafe { cublasLtMatmulPreferenceDestroy(self.0) };
    }
}

/// 2026-10-14: The operation and layout descriptors of one row-major
/// `out[M,N] = act[M,K] @ weight[N,K]ᵀ` with BF16 operands, destroyed on drop.
pub(super) struct Bf16Gemm {
    desc: cublasLtMatmulDesc_t,
    la: cublasLtMatrixLayout_t,
    lb: cublasLtMatrixLayout_t,
    ld: cublasLtMatrixLayout_t,
}

impl Bf16Gemm {
    pub(super) fn new(m: u32, n: u32, k: u32, out_dtype: i32) -> Result<Self> {
        let mut g = Self {
            desc: std::ptr::null_mut(),
            la: std::ptr::null_mut(),
            lb: std::ptr::null_mut(),
            ld: std::ptr::null_mut(),
        };
        unsafe {
            chk(
                cublasLtMatmulDescCreate(&mut g.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F),
                "DescCreate",
            )?;
            for (attr, op, what) in [
                (DESC_TRANSA, CUBLAS_OP_T, "TRANSA"),
                (DESC_TRANSB, CUBLAS_OP_N, "TRANSB"),
            ] {
                let op_ptr = &op as *const i32 as *const c_void;
                chk(
                    cublasLtMatmulDescSetAttribute(g.desc, attr, op_ptr, 4),
                    what,
                )?;
            }
            // 2026-09-25: A = weight, row-major [N,K] = col-major [K,N], ld K, opT.
            // B = act, row-major [M,K] = col-major [K,M], ld K, opN.
            // D = out, row-major [M,N] = col-major [N,M], ld N.
            let (m, n, k) = (m as u64, n as u64, k as u64);
            chk(
                cublasLtMatrixLayoutCreate(&mut g.la, CUDA_R_16BF, k, n, k as i64),
                "LayoutA",
            )?;
            chk(
                cublasLtMatrixLayoutCreate(&mut g.lb, CUDA_R_16BF, k, m, k as i64),
                "LayoutB",
            )?;
            chk(
                cublasLtMatrixLayoutCreate(&mut g.ld, out_dtype, n, m, n as i64),
                "LayoutD",
            )?;
        }
        Ok(g)
    }

    /// 2026-10-14: Fill `results` with up to its length of algorithms for this GEMM under
    /// `pref`, best first; returns how many were written.
    pub(super) fn heuristic(
        &self,
        ctx: &Ctx,
        pref: &Pref,
        results: &mut [HeuristicResult],
    ) -> Result<usize> {
        let mut returned: i32 = 0;
        chk(
            unsafe {
                cublasLtMatmulAlgoGetHeuristic(
                    ctx.handle,
                    self.desc,
                    self.la,
                    self.lb,
                    self.ld,
                    self.ld,
                    pref.0,
                    i32::try_from(results.len())?,
                    results.as_mut_ptr() as *mut c_void,
                    &mut returned,
                )
            },
            "AlgoGetHeuristic",
        )?;
        Ok(usize::try_from(returned)?.min(results.len()))
    }

    /// 2026-10-14: `cublasLtMatmulAlgoCheck` of `algo` on this GEMM: the workspace bytes it
    /// needs, or an error when it cannot run these descriptors on this device.
    pub(super) fn check(&self, ctx: &Ctx, algo: &[u64; 8]) -> Result<usize> {
        let mut result = HeuristicResult::ZERO;
        chk(
            unsafe {
                cublasLtMatmulAlgoCheck(
                    ctx.handle,
                    self.desc,
                    self.la,
                    self.lb,
                    self.ld,
                    self.ld,
                    algo.as_ptr() as *const c_void,
                    &mut result as *mut HeuristicResult as *mut c_void,
                )
            },
            "AlgoCheck",
        )?;
        Ok(result.workspace_size)
    }

    /// 2026-10-14: Run `algo` on `(act, weight, out)` with `workspace` = (pointer, bytes).
    pub(super) fn matmul(
        &self,
        ctx: &Ctx,
        algo: &[u64; 8],
        (act, weight, out): (u64, u64, u64),
        alpha: f32,
        workspace: (u64, usize),
        stream: u64,
    ) -> Result<()> {
        let beta: f32 = 0.0;
        let status = unsafe {
            cublasLtMatmul(
                ctx.handle,
                self.desc,
                &alpha as *const f32 as *const c_void,
                weight as *const c_void,
                self.la,
                act as *const c_void,
                self.lb,
                &beta as *const f32 as *const c_void,
                out as *const c_void,
                self.ld,
                out as *mut c_void,
                self.ld,
                algo.as_ptr() as *const c_void,
                workspace.0 as *mut c_void,
                workspace.1,
                stream as *mut c_void,
            )
        };
        chk(status, "Matmul")
    }
}

impl Drop for Bf16Gemm {
    fn drop(&mut self) {
        unsafe {
            for layout in [self.la, self.lb, self.ld] {
                if !layout.is_null() {
                    cublasLtMatrixLayoutDestroy(layout);
                }
            }
            if !self.desc.is_null() {
                cublasLtMatmulDescDestroy(self.desc);
            }
        }
    }
}
