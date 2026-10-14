// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: cuBLASLt FFI for `out[M,N] = act[M,K] @ weight[N,K]ᵀ`: BF16 GEMMs
//! with BF16 or FP32 output here, FP8 E4M3 GEMMs in `fp8`, and the FP8
//! scale-tensor layout math in [`scale_layout`].
//!
//! Owner: gpu-runtime (cuBLASLt wrapper).
//! Invariants:
//! - `CTX` holds a single cuBLASLt handle and 64 MiB workspace for the
//!   process, allocated with `cuMemAlloc_v2`, outside the backend's
//!   allocation ledger.
//! - Every GEMM here passes the whole workspace to `cublasLtMatmul`, so two
//!   GEMMs must not run concurrently.

use anyhow::{Result, bail};
use std::ffi::c_void;
use std::sync::OnceLock;

mod fp8;
pub use fp8::{
    fp8_gemm_act_weight_t_blkscaled, fp8_gemm_act_weight_t_blkscaled_ldc,
    fp8_gemm_act_weight_t_rowwise, fp8_gemm_act_weight_t_rowwise_ldc,
};

pub mod scale_layout;

mod bf16_gemm;
use bf16_gemm::{Bf16Gemm, HeuristicResult, Pref};
mod pinned;
pub use pinned::bf16_gemm_act_weight_t_pinned;

#[allow(non_camel_case_types)]
type cublasLtHandle_t = *mut c_void;
#[allow(non_camel_case_types)]
type cublasLtMatmulDesc_t = *mut c_void;
#[allow(non_camel_case_types)]
type cublasLtMatrixLayout_t = *mut c_void;
#[allow(non_camel_case_types)]
type cublasLtMatmulPreference_t = *mut c_void;

const CUDA_R_16BF: i32 = 14;
const CUDA_R_32F: i32 = 0;
const CUDA_R_8F_E4M3: i32 = 28;
const CUBLAS_COMPUTE_32F: i32 = 68;
const CUBLAS_OP_N: i32 = 0;
const CUBLAS_OP_T: i32 = 1;
const DESC_TRANSA: u32 = 3;
const DESC_TRANSB: u32 = 4;
const DESC_A_SCALE_POINTER: u32 = 17;
const DESC_B_SCALE_POINTER: u32 = 18;
const DESC_A_SCALE_MODE: u32 = 31;
const DESC_B_SCALE_MODE: u32 = 32;
const SCALE_MODE_OUTER_VEC_32F: i32 = 3;
const SCALE_MODE_VEC128_32F: i32 = 4;
const SCALE_MODE_BLK128X128_32F: i32 = 5;
const PREF_MAX_WORKSPACE_BYTES: u32 = 1;
const PREF_REDUCTION_SCHEME_MASK: u32 = 3;
const PREF_MIN_ALIGNMENT_A_BYTES: u32 = 5;
const PREF_MIN_ALIGNMENT_B_BYTES: u32 = 6;
const PREF_MIN_ALIGNMENT_C_BYTES: u32 = 7;
const PREF_MIN_ALIGNMENT_D_BYTES: u32 = 8;
const ALGO_CONFIG_ID: u32 = 0;
const ALGO_CONFIG_TILE_ID: u32 = 1;
const ALGO_CONFIG_SPLITK_NUM: u32 = 2;
const ALGO_CONFIG_REDUCTION_SCHEME: u32 = 3;
const ALGO_CONFIG_STAGES_ID: u32 = 6;

unsafe extern "C" {
    fn cublasLtCreate(handle: *mut cublasLtHandle_t) -> i32;
    fn cublasLtMatmulDescCreate(
        desc: *mut cublasLtMatmulDesc_t,
        compute_type: i32,
        scale_type: i32,
    ) -> i32;
    fn cublasLtMatmulDescSetAttribute(
        desc: cublasLtMatmulDesc_t,
        attr: u32,
        buf: *const c_void,
        size: usize,
    ) -> i32;
    fn cublasLtMatmulDescDestroy(desc: cublasLtMatmulDesc_t) -> i32;
    fn cublasLtMatrixLayoutCreate(
        layout: *mut cublasLtMatrixLayout_t,
        dtype: i32,
        rows: u64,
        cols: u64,
        ld: i64,
    ) -> i32;
    fn cublasLtMatrixLayoutDestroy(layout: cublasLtMatrixLayout_t) -> i32;
    fn cublasLtMatmulPreferenceCreate(pref: *mut cublasLtMatmulPreference_t) -> i32;
    fn cublasLtMatmulPreferenceSetAttribute(
        pref: cublasLtMatmulPreference_t,
        attr: u32,
        buf: *const c_void,
        size: usize,
    ) -> i32;
    fn cublasLtMatmulPreferenceDestroy(pref: cublasLtMatmulPreference_t) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn cublasLtMatmulAlgoGetHeuristic(
        handle: cublasLtHandle_t,
        desc: cublasLtMatmulDesc_t,
        a: cublasLtMatrixLayout_t,
        b: cublasLtMatrixLayout_t,
        c: cublasLtMatrixLayout_t,
        d: cublasLtMatrixLayout_t,
        pref: cublasLtMatmulPreference_t,
        requested: i32,
        results: *mut c_void,
        returned: *mut i32,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn cublasLtMatmul(
        handle: cublasLtHandle_t,
        desc: cublasLtMatmulDesc_t,
        alpha: *const c_void,
        a: *const c_void,
        layout_a: cublasLtMatrixLayout_t,
        b: *const c_void,
        layout_b: cublasLtMatrixLayout_t,
        beta: *const c_void,
        c: *const c_void,
        layout_c: cublasLtMatrixLayout_t,
        d: *mut c_void,
        layout_d: cublasLtMatrixLayout_t,
        algo: *const c_void,
        workspace: *mut c_void,
        workspace_size: usize,
        stream: *mut c_void,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn cublasLtMatmulAlgoCheck(
        handle: cublasLtHandle_t,
        desc: cublasLtMatmulDesc_t,
        a: cublasLtMatrixLayout_t,
        b: cublasLtMatrixLayout_t,
        c: cublasLtMatrixLayout_t,
        d: cublasLtMatrixLayout_t,
        algo: *const c_void,
        result: *mut c_void,
    ) -> i32;
    fn cublasLtMatmulAlgoConfigGetAttribute(
        algo: *const c_void,
        attr: u32,
        buf: *mut c_void,
        size: usize,
        written: *mut usize,
    ) -> i32;
    fn cuMemAlloc_v2(dptr: *mut u64, bytesize: usize) -> i32;
    fn cuMemFree_v2(dptr: u64) -> i32;
    fn cuStreamSynchronize(stream: u64) -> i32;
}

struct Ctx {
    handle: cublasLtHandle_t,
    workspace: u64,
    ws_size: usize,
}
// 2026-09-25: The handle and workspace are process-global and shared by
// every caller; see the module invariants.
unsafe impl Send for Ctx {}
unsafe impl Sync for Ctx {}

/// 2026-09-25: The process-wide cuBLASLt handle and workspace, created on
/// first use by `ctx`. Its size is fixed, not derived from a model, so it
/// stays valid across a model swap, like the process CUDA context
/// (`crate::cuda_host`).
static CTX: OnceLock<Ctx> = OnceLock::new();

fn ctx() -> Result<&'static Ctx> {
    if let Some(c) = CTX.get() {
        return Ok(c);
    }
    let mut handle: cublasLtHandle_t = std::ptr::null_mut();
    let st = unsafe { cublasLtCreate(&mut handle) };
    if st != 0 {
        bail!("cublasLtCreate failed: {st}");
    }
    let ws_size = 64 * 1024 * 1024;
    let mut ws: u64 = 0;
    let st = unsafe { cuMemAlloc_v2(&mut ws, ws_size) };
    if st != 0 {
        bail!("cuMemAlloc cuBLASLt workspace failed: {st}");
    }
    let _ = CTX.set(Ctx {
        handle,
        workspace: ws,
        ws_size,
    });
    Ok(CTX.get().unwrap())
}

/// 2026-09-25: Run one 64x64x64 BF16 GEMM on `stream` at model load
/// (`serve_load.rs`), so the first cuBLASLt call, including `ctx`'s
/// `cublasLtCreate` and workspace allocation, happens before the first
/// request.
///
/// Returns nothing: a failure is logged as a warning, and the first GEMM of
/// a request then pays the same costs.
pub fn prewarm(stream: u64) {
    let r = (|| -> Result<()> {
        let bytes = 64usize * 64 * 2;
        let mut a = 0u64;
        let mut b = 0u64;
        let mut d = 0u64;
        unsafe {
            chk(cuMemAlloc_v2(&mut a, bytes), "prewarm alloc a")?;
            chk(cuMemAlloc_v2(&mut b, bytes), "prewarm alloc b")?;
            chk(cuMemAlloc_v2(&mut d, bytes), "prewarm alloc d")?;
        }
        let res = bf16_gemm_act_weight_t(a, b, d, 64, 64, 64, stream);
        unsafe {
            chk(cuStreamSynchronize(stream), "prewarm sync")?;
            let _ = cuMemFree_v2(a);
            let _ = cuMemFree_v2(b);
            let _ = cuMemFree_v2(d);
        }
        res
    })();
    match r {
        Ok(()) => tracing::info!("cuBLASLt pre-warmed (handle + workspace + kernel images)"),
        Err(e) => tracing::warn!("cuBLASLt pre-warm failed (request 1 pays lazy init): {e}"),
    }
}

fn chk(status: i32, what: &str) -> Result<()> {
    if status != 0 {
        bail!("cuBLASLt {what} failed: status {status}");
    }
    Ok(())
}

/// 2026-09-25: Row-major `out[M,N] = act[M,K] @ weight[N,K]ᵀ`, all BF16, FP32
/// accumulation. In cuBLASLt's column-major terms:
/// `D[N,M] = opT(weightᶜ[K,N]) · opN(actᶜ[K,M])`.
pub fn bf16_gemm_act_weight_t(
    act: u64,
    weight: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    gemm_act_weight_t_out(act, weight, out, m, n, k, CUDA_R_16BF, 1.0, stream)
}

/// 2026-10-09: [`bf16_gemm_act_weight_t`] times `alpha`: `out = alpha * act @ weightᵀ`, the
/// scale applied to the FP32 sum before the BF16 store (a weight's global scale, so its BF16
/// copy can stay exact).
#[allow(clippy::too_many_arguments)]
pub fn bf16_gemm_act_weight_t_alpha(
    act: u64,
    weight: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    alpha: f32,
    stream: u64,
) -> Result<()> {
    gemm_act_weight_t_out(act, weight, out, m, n, k, CUDA_R_16BF, alpha, stream)
}

/// 2026-09-25: [`bf16_gemm_act_weight_t`] with an FP32 `out`. Only the D
/// layout's type differs; both use `CUBLAS_COMPUTE_32F`.
pub fn bf16_gemm_act_weight_t_f32_out(
    act: u64,
    weight: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    gemm_act_weight_t_out(act, weight, out, m, n, k, CUDA_R_32F, 1.0, stream)
}

/// 2026-09-25: Shared body of the wrappers above; `out_dtype` is the D
/// layout's type. 2026-10-09: `alpha` scales the product (1.0 for the plain forms).
/// 2026-10-14: The heuristic is asked per call with only a workspace cap, so the algorithm (and
/// with it the bits of a row) may change with `m`; [`bf16_gemm_act_weight_t_pinned`] does not.
#[allow(clippy::too_many_arguments)]
fn gemm_act_weight_t_out(
    act: u64,
    weight: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    out_dtype: i32,
    alpha: f32,
    stream: u64,
) -> Result<()> {
    let ctx = ctx()?;
    let gemm = Bf16Gemm::new(m, n, k, out_dtype)?;
    let pref = Pref::new()?;
    pref.set(PREF_MAX_WORKSPACE_BYTES, &(ctx.ws_size as u64))?;
    let mut result = [HeuristicResult::ZERO];
    if gemm.heuristic(ctx, &pref, &mut result)? < 1 {
        bail!("cuBLASLt: no algorithm for {m}x{n}x{k}");
    }
    let workspace = (ctx.workspace, ctx.ws_size);
    gemm.matmul(
        ctx,
        &result[0].algo,
        (act, weight, out),
        alpha,
        workspace,
        stream,
    )
}
