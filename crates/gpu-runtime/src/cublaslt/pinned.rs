// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: The pinned BF16 cuBLASLt GEMM, row-major `out[M,N] = act[M,K] @ weight[N,K]ᵀ`
//! with one algorithm per (N, K), chosen once and reused at every M, so the bits of a row do not
//! depend on how many rows share the call (the per-call heuristic of
//! [`super::bf16_gemm_act_weight_t`] may pick another algorithm, or a split-K one, at another M).
//!
//! Owner: gpu-runtime (cuBLASLt wrapper).
//! Invariants:
//! - The algorithm of an (N, K) is the heuristic's first candidate at [`PINNED_REFERENCE_M`] rows,
//!   asked with reduction scheme NONE, at most [`PINNED_WORKSPACE_CAP`] workspace bytes and 16-byte
//!   operand alignment, that reports split-K 1 and reduction NONE and passes
//!   `cublasLtMatmulAlgoCheck` at every `PINNED_PROBE_M` within the cap. It depends on the device,
//!   the library and (N, K), never on the M of the call that first asks for it.
//! - 2026-10-14: Its workspace is fixed per (N, K): the largest need over the reference and probe
//!   rows (H100 reports 4 bytes at 32,768 rows for every candidate, 0 below). Every call passes
//!   that same size from the process workspace, so the size cuBLASLt sees never varies with M.
//! - Every call checks the algorithm again at its own M and errors when it fails or needs more
//!   than the fixed workspace, never a silent re-pick.
//! - Whether the algorithm is in fact M-invariant is the GPU test's question
//!   (model-layers `ops/dense_cublas_pinned_tests.rs`).

use anyhow::{Context, Result, anyhow, bail, ensure};
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

use super::*;

/// 2026-10-14: Rows of the heuristic query that picks an (N, K)'s algorithm: one prefill slab.
pub const PINNED_REFERENCE_M: u32 = 2048;
/// 2026-10-14: Row counts the chosen algorithm must also pass the check at: one row, 16 and 20
/// (the decode width and the prefill tail chunk the M-invariance test covers), and a 32k prompt.
const PINNED_PROBE_M: [u32; 4] = [1, 16, 20, 32_768];
/// 2026-10-14: Candidates the heuristic is asked for.
const PINNED_CANDIDATES: usize = 8;
/// 2026-10-14: Operand alignment in bytes the algorithm may assume, and so every call must meet.
/// cuBLASLt otherwise assumes 256, which an activation slab at a row offset need not have.
const PINNED_ALIGNMENT: u32 = 16;
const BF16_BYTES: u64 = 2;

/// 2026-10-14: The opaque `cublasLtMatmulAlgo_t` (64 bytes).
type Algo = [u64; 8];
/// 2026-10-14: Largest workspace in bytes a pinned algorithm may need at any probed M.
const PINNED_WORKSPACE_CAP: usize = 1 << 20;

/// 2026-10-14: An (N, K)'s pinned algorithm and the fixed workspace bytes every call passes.
#[derive(Clone, Copy)]
struct Pinned {
    algo: Algo,
    workspace: usize,
}

/// 2026-10-14: The algorithm pinned for each (N, K), filled on first use.
static PINNED: OnceLock<Mutex<HashMap<(u32, u32), Pinned>>> = OnceLock::new();

/// 2026-10-14: Whether a call meets the alignment the pinned algorithm assumes: every pointer and
/// both row pitches (N and K BF16s) a multiple of 16 bytes.
fn pinned_operands_admitted(act: u64, weight: u64, out: u64, n: u32, k: u32) -> bool {
    let align = u64::from(PINNED_ALIGNMENT);
    [
        act,
        weight,
        out,
        u64::from(n) * BF16_BYTES,
        u64::from(k) * BF16_BYTES,
    ]
    .iter()
    .all(|v| v % align == 0)
}

/// 2026-10-14: Whether a configuration computes each output in one pass over K: one K split and
/// no reduction scheme.
fn single_pass(split_k: i32, reduction: u32) -> bool {
    split_k == 1 && reduction == 0
}

/// 2026-10-14: The fixed workspace of an algorithm from its needs at the reference and probe
/// rows: the largest, or an error when one exceeds [`PINNED_WORKSPACE_CAP`].
fn fixed_workspace(needs: &[(u32, usize)]) -> Result<usize> {
    let mut fixed = 0;
    for &(m, need) in needs {
        ensure!(
            need <= PINNED_WORKSPACE_CAP,
            "needs {need} workspace bytes at M={m}, over the {PINNED_WORKSPACE_CAP}-byte cap"
        );
        fixed = fixed.max(need);
    }
    Ok(fixed)
}

/// 2026-10-14: [`super::bf16_gemm_act_weight_t`] on the algorithm pinned for (N, K). Errors when
/// the operands are not admitted ([`pinned_operands_admitted`]), when no candidate qualifies, or
/// when the pinned algorithm fails its check at this M. Zero rows launch nothing.
pub fn bf16_gemm_act_weight_t_pinned(
    act: u64,
    weight: u64,
    out: u64,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        pinned_operands_admitted(act, weight, out, n, k),
        "pinned cuBLASLt BF16: operands not {PINNED_ALIGNMENT}-byte aligned \
         (act {act:#x}, weight {weight:#x}, out {out:#x}, N {n}, K {k})"
    );
    if m == 0 {
        return Ok(());
    }
    let ctx = ctx()?;
    let pinned = pinned_algo(ctx, n, k)?;
    let gemm = Bf16Gemm::new(m, n, k, CUDA_R_16BF)?;
    let need = gemm
        .check(ctx, &pinned.algo)
        .with_context(|| format!("pinned cuBLASLt algorithm for N={n} K={k} rejected at M={m}"))?;
    ensure!(
        need <= pinned.workspace,
        "pinned cuBLASLt algorithm for N={n} K={k} needs {need} workspace bytes at M={m}, over \
         its fixed {}",
        pinned.workspace
    );
    let workspace = (ctx.workspace, pinned.workspace);
    gemm.matmul(
        ctx,
        &pinned.algo,
        (act, weight, out),
        1.0,
        workspace,
        stream,
    )
}

fn pinned_algo(ctx: &Ctx, n: u32, k: u32) -> Result<Pinned> {
    let table = PINNED.get_or_init(|| Mutex::new(HashMap::new()));
    let mut table = table
        .lock()
        .map_err(|_| anyhow!("pinned cuBLASLt table poisoned by an earlier panic"))?;
    if let Some(algo) = table.get(&(n, k)) {
        return Ok(*algo);
    }
    let pinned = choose(ctx, n, k)?;
    table.insert((n, k), pinned);
    Ok(pinned)
}

fn choose(ctx: &Ctx, n: u32, k: u32) -> Result<Pinned> {
    ensure!(
        PINNED_WORKSPACE_CAP <= ctx.ws_size,
        "pinned cuBLASLt workspace cap {PINNED_WORKSPACE_CAP} over the {}-byte process workspace",
        ctx.ws_size
    );
    let gemm = Bf16Gemm::new(PINNED_REFERENCE_M, n, k, CUDA_R_16BF)?;
    let pref = Pref::new()?;
    pref.set(PREF_MAX_WORKSPACE_BYTES, &(PINNED_WORKSPACE_CAP as u64))?;
    pref.set(PREF_REDUCTION_SCHEME_MASK, &0u32)?;
    for attr in [
        PREF_MIN_ALIGNMENT_A_BYTES,
        PREF_MIN_ALIGNMENT_B_BYTES,
        PREF_MIN_ALIGNMENT_C_BYTES,
        PREF_MIN_ALIGNMENT_D_BYTES,
    ] {
        pref.set(attr, &PINNED_ALIGNMENT)?;
    }
    let mut results = [HeuristicResult::ZERO; PINNED_CANDIDATES];
    let found = gemm.heuristic(ctx, &pref, &mut results)?;
    let mut rejected = Vec::new();
    for result in &results[..found] {
        let config = AlgoConfig::read(&result.algo)?;
        match qualify(ctx, n, k, result, &config) {
            Ok(workspace) => {
                tracing::info!(
                    "cuBLASLt BF16 N={n} K={k} pinned to {config}, workspace {workspace} bytes"
                );
                return Ok(Pinned {
                    algo: result.algo,
                    workspace,
                });
            }
            Err(e) => rejected.push(format!("{config}: {e:#}")),
        }
    }
    bail!(
        "pinned cuBLASLt BF16: no single-pass algorithm for N={n} K={k} among {found} \
         candidates at M={PINNED_REFERENCE_M} [{}]",
        rejected.join("; ")
    )
}

/// 2026-10-14: The fixed workspace of a single-pass candidate that passes the check at every
/// probed M within [`PINNED_WORKSPACE_CAP`]; an error names why another is rejected.
fn qualify(ctx: &Ctx, n: u32, k: u32, result: &HeuristicResult, c: &AlgoConfig) -> Result<usize> {
    ensure!(
        single_pass(c.split_k, c.reduction),
        "not single-pass (split-K {}, reduction {})",
        c.split_k,
        c.reduction
    );
    let mut needs = vec![(PINNED_REFERENCE_M, result.workspace_size)];
    for m in PINNED_PROBE_M {
        let need = Bf16Gemm::new(m, n, k, CUDA_R_16BF)?
            .check(ctx, &result.algo)
            .with_context(|| format!("check at M={m}"))?;
        needs.push((m, need));
    }
    fixed_workspace(&needs)
}

/// 2026-10-14: The configuration attributes of an algorithm that name it in the log and decide
/// whether it is single-pass.
struct AlgoConfig {
    id: i32,
    tile: u32,
    stages: u32,
    split_k: i32,
    reduction: u32,
}

impl AlgoConfig {
    fn read(algo: &Algo) -> Result<Self> {
        Ok(Self {
            id: config_attr(algo, ALGO_CONFIG_ID)?,
            tile: config_attr(algo, ALGO_CONFIG_TILE_ID)?,
            stages: config_attr(algo, ALGO_CONFIG_STAGES_ID)?,
            split_k: config_attr(algo, ALGO_CONFIG_SPLITK_NUM)?,
            reduction: config_attr(algo, ALGO_CONFIG_REDUCTION_SCHEME)?,
        })
    }
}

impl std::fmt::Display for AlgoConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "algo {} tile {} stages {} split-K {} reduction {}",
            self.id, self.tile, self.stages, self.split_k, self.reduction
        )
    }
}

/// 2026-10-14: One configuration attribute of `algo`; `T` must be the attribute's type in
/// cublasLt.h.
fn config_attr<T: Copy + Default>(algo: &Algo, attr: u32) -> Result<T> {
    let mut value = T::default();
    let size = std::mem::size_of::<T>();
    let mut written = 0usize;
    chk(
        unsafe {
            cublasLtMatmulAlgoConfigGetAttribute(
                algo.as_ptr() as *const c_void,
                attr,
                &mut value as *mut T as *mut c_void,
                size,
                &mut written,
            )
        },
        "AlgoConfigGetAttribute",
    )?;
    ensure!(
        written == size,
        "cuBLASLt algo attribute {attr}: {written} bytes written, {size} expected"
    );
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operands_must_be_sixteen_byte_aligned_in_pointers_and_pitches() {
        let base = 0x7f00_0000_0000u64;
        let at = |a: u64, w: u64, o: u64| pinned_operands_admitted(a, w, o, 2560, 2560);
        assert!(at(base, base + 256, base + 4096));
        assert!(pinned_operands_admitted(
            base + 16,
            base + 32,
            base + 48,
            8,
            8
        ));
        // 2026-10-14: One row into an activation of K = 2,563 BF16s is 6 bytes off.
        assert!(!at(base + 2 * 2563, base, base));
        for off in [2u64, 8] {
            assert!(!at(base + off, base, base));
            assert!(!at(base, base + off, base));
            assert!(!at(base, base, base + off));
        }
        // 2026-10-14: Pitches of 4 BF16s (8 bytes) in N or K.
        assert!(!pinned_operands_admitted(base, base, base, 2564, 2560));
        assert!(!pinned_operands_admitted(base, base, base, 2560, 2564));
    }

    #[test]
    fn the_fixed_workspace_is_the_largest_need_within_the_cap() {
        // 2026-10-14: The H100 report: 0 bytes at every probe but 4 at 32,768 rows.
        let h100 = [(2048, 0), (1, 0), (16, 0), (20, 0), (32_768, 4)];
        assert_eq!(fixed_workspace(&h100).unwrap(), 4);
        assert_eq!(fixed_workspace(&[(2048, 0), (1, 0)]).unwrap(), 0);
        assert_eq!(
            fixed_workspace(&[(1, PINNED_WORKSPACE_CAP), (16, 4)]).unwrap(),
            PINNED_WORKSPACE_CAP
        );
        let over = fixed_workspace(&[(1, 0), (32_768, PINNED_WORKSPACE_CAP + 1), (16, 4)]);
        assert!(over.unwrap_err().to_string().contains("M=32768"));
    }

    #[test]
    fn only_one_split_without_reduction_is_single_pass() {
        assert!(single_pass(1, 0));
        assert!(!single_pass(2, 0));
        assert!(!single_pass(0, 0));
        for reduction in [1, 2, 4] {
            assert!(!single_pass(1, reduction));
        }
    }
}
