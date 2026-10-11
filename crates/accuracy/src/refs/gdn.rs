// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The gated delta rule decode reference (`gdn_recurrence`): one token's step of
//! every (sequence, value head), as the decode kernels compute it:
//!
//! ```text
//! g  = clamp(decay, 1e-6, 1 - 1e-6)
//! u  = (v - g * (S^T k)) * beta              per value column
//! S' = g * S + k (outer) u                   the new state, stored
//! o  = (S'^T q) * rsqrt(k_dim)               from the stored S', before any clamp
//! if ||S'||_F^2 > M^2:  S' *= M * rsqrt(||S'||_F^2)    M = the contract's `state_max_norm`
//! ```
//!
//! The clamp is a kernel constant the contract declares (`constants.state_max_norm`): `1000`
//! where the kernel compiles `SSM_STATE_MAX_NORM`'s clamp, `inf` where it has none. The same
//! symbol differs across the targets' sources (the shadows of gated_delta_rule.cu), so the
//! contract states which computation it holds the symbol to.
//!
//! Value head `vh` reads key head `vh / (v_heads / k_heads)`. q/k/v are NOT normalized inside
//! the kernel (the conv kernel's L2 norm precedes it).
//!
//! Tensors of a case: `q`, `k` `[rows, k_heads*k_dim]`, `v` `[rows, v_heads*v_dim]`, `gate`,
//! `beta` `[rows, v_heads]`, `state` `[rows, v_heads, k_dim, v_dim]` (the state handed to the
//! kernel) and `state_prev` (the state one token earlier, from which `state` was advanced).
//! Scalars `k_heads`, `k_dim`, `v_heads`, `v_dim`. Output `[rows, v_heads*v_dim +
//! v_heads*k_dim*v_dim]`: each row is the step's output `o` followed by the new state, so the
//! one-output comparison judges both.
//!
//! Owner: metrale-accuracy.
//! Invariants:
//! - The head geometry comes from the point's runtime values (the family's params read the
//!   circuit dims) and must reproduce the node's concatenated widths; nothing is assumed.
//! - `state` is `state_prev` advanced by one seeded token (in f64, stored), so the `state_stale`
//!   mutation hands the kernel a state a model really held one step earlier.

use metrale_circuit::format::Format;
use metrale_circuit::pipeline::{StepKind, Value};
use metrale_circuit::state::StateDtype;

use crate::case::{Case, Enc, Tensor};
use crate::elem::{self, Elem, F32_FTZ};
use crate::inputs::{InputClass, SplitMix64, tensor};
use crate::mutation::Mutation;
use crate::plan::Plan;
use crate::points::Shape;

/// 2026-10-09: Distinct (state, previous token) head slots drawn before tiling over (row,
/// head): prime, so neither a head pitch nor a row pitch maps a slot onto its neighbour's.
pub const STATE_PERIOD: usize = 127;

/// 2026-10-09: Standard deviation of a drawn state element. A served state accumulates
/// `k (outer) u` with unit-norm keys and O(1) values under decays below one; 0.1 gives a head
/// norm near 13, far below the clamp, as such states are.
pub const STATE_SCALE: f64 = 0.1;

/// 2026-10-09: The range a drawn decay is uniform in (the gates kernel's `exp(-softplus(..)*A)`
/// spans it for the Qwen3.5/3.6 GDN layers).
pub const DECAY: (f64, f64) = (0.25, 1.0);

/// 2026-10-09: The range a drawn write strength is uniform in (a sigmoid's useful range).
pub const BETA: (f64, f64) = (0.05, 0.95);

/// 2026-10-09: The kernel's lower and upper decay clamp, as the f32 constants it compiles
/// (`fminf(fmaxf(g, 1e-6f), 1.0f - 1e-6f)`, gated_delta_rule.cu:730).
pub fn decay_clamp() -> (f64, f64) {
    let lo = 1e-6f32;
    (f64::from(lo), f64::from(1.0f32 - lo))
}

/// 2026-10-09: The head geometry of one launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Geom {
    /// 2026-10-09: Sequences (batch rows).
    pub rows: usize,
    /// 2026-10-09: Key heads.
    pub k_heads: usize,
    /// 2026-10-09: Key head width.
    pub k_dim: usize,
    /// 2026-10-09: Value heads.
    pub v_heads: usize,
    /// 2026-10-09: Value head width.
    pub v_dim: usize,
}

impl Geom {
    /// 2026-10-09: The geometry of a swept shape: its runtime `k_heads`, `k_dim`, `v_heads`,
    /// `v_dim`, checked against the node's widths (`in_dim` is q|k|v concatenated, `out_dim`
    /// the value heads).
    pub fn of_shape(shape: &Shape) -> Result<Geom, String> {
        let get = |name: &str| -> Result<usize, String> {
            shape
                .runtime
                .get(name)
                .ok_or_else(|| {
                    format!(
                        "the point carries no runtime `{name}` (the family declares no such param)"
                    )
                })?
                .parse()
                .map_err(|e| format!("runtime `{name}`: {e}"))
        };
        let g = Geom {
            rows: shape.rows as usize,
            k_heads: get("k_heads")?,
            k_dim: get("k_dim")?,
            v_heads: get("v_heads")?,
            v_dim: get("v_dim")?,
        };
        let qkv = 2 * g.k_heads * g.k_dim + g.v_heads * g.v_dim;
        if shape.in_dim as usize != qkv || shape.out_dim as usize != g.o_len() {
            return Err(format!(
                "heads {g:?} give in {qkv} out {}, the node has in {} out {}",
                g.o_len(),
                shape.in_dim,
                shape.out_dim
            ));
        }
        g.check()?;
        Ok(g)
    }

    /// 2026-10-09: The geometry a filled case records.
    pub fn of_case(case: &Case) -> Result<Geom, String> {
        let s = |n: &str| case.scalar(n).map(|v| v as usize);
        Ok(Geom {
            rows: case.out.0[0],
            k_heads: s("k_heads")?,
            k_dim: s("k_dim")?,
            v_heads: s("v_heads")?,
            v_dim: s("v_dim")?,
        })
    }

    /// 2026-10-09: The kernels' preconditions: a value head reads one key head; `k_dim == 128`
    /// and `v_dim == 128` (the qwen3.6-35b-a3b sources fix both, `K_DIM` and `V_DIM_DECODE`,
    /// :22-26; the block is 128 threads, one value column each, and a clamp's reduction reads
    /// four warps' partials).
    pub fn check(&self) -> Result<(), String> {
        if self.k_heads == 0 || !self.v_heads.is_multiple_of(self.k_heads) {
            return Err(format!(
                "{self:?}: value heads are not a multiple of key heads"
            ));
        }
        if self.k_dim != 128 {
            return Err(format!("{self:?}: the decode kernels run k_dim = 128"));
        }
        if self.v_dim != 128 {
            return Err(format!(
                "{self:?}: the decode kernels run v_dim = 128 (the block)"
            ));
        }
        Ok(())
    }

    /// 2026-10-09: Output values per row.
    pub fn o_len(&self) -> usize {
        self.v_heads * self.v_dim
    }

    /// 2026-10-09: State values per row.
    pub fn s_len(&self) -> usize {
        self.v_heads * self.k_dim * self.v_dim
    }

    /// 2026-10-09: Output columns per row (`o` then the state).
    pub fn cols(&self) -> usize {
        self.o_len() + self.s_len()
    }
}

/// 2026-10-09: The formats of a `gdn_recurrence` plan.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Formats {
    /// 2026-10-09: q, k, v as read.
    pub qkv: Elem,
    /// 2026-10-09: The decay as read.
    pub gate: Elem,
    /// 2026-10-09: The write strength as read.
    pub beta: Elem,
    /// 2026-10-09: The state as stored and reread.
    pub state: Elem,
    /// 2026-10-09: Products, sums and the scale arithmetic.
    pub compute: Elem,
    /// 2026-10-09: `o` as written.
    pub out: Elem,
}

/// 2026-10-09: The rounding model of a real-valued operand format.
fn real(f: Format, ftz: bool) -> Result<Elem, String> {
    match f {
        Format::F32 if ftz => Ok(F32_FTZ),
        Format::F32 => Ok(elem::F32),
        Format::Bf16 => Ok(elem::BF16),
        o => Err(format!("a gdn_recurrence operand in {}", o.name())),
    }
}

/// 2026-10-09: The formats `plan` declares (inputs q|k|v, decay, beta; steps state, compute;
/// output `o`).
pub fn formats(plan: &Plan) -> Result<Formats, String> {
    let p = &plan.pipeline;
    let input = |i: usize, what: &str| {
        p.inputs
            .get(i)
            .copied()
            .ok_or_else(|| format!("the pipeline names no {what} input"))
            .and_then(|f| real(f, plan.ftz))
    };
    let state = match plan.step(StepKind::State) {
        Some(Value::State(StateDtype::F32)) => real(Format::F32, plan.ftz)?,
        Some(Value::State(d)) => {
            return Err(format!(
                "a {} state: the reference stores f32 states",
                d.name()
            ));
        }
        _ => return Err("the pipeline has no `state` step".into()),
    };
    let compute = match plan.step(StepKind::Compute) {
        Some(Value::Num(n)) => {
            let e = elem::of_num(*n);
            if plan.ftz && e == elem::F32 {
                F32_FTZ
            } else {
                e
            }
        }
        _ => return Err("the pipeline has no `compute` step".into()),
    };
    let out = real(
        *p.outputs.first().ok_or("the pipeline names no output")?,
        plan.ftz,
    )?;
    if out != state {
        return Err(format!(
            "o in {} and the state in {}: the case writes both as one output",
            out.name, state.name
        ));
    }
    Ok(Formats {
        qkv: input(0, "q|k|v")?,
        gate: input(1, "decay")?,
        beta: input(2, "beta")?,
        state,
        compute,
        out,
    })
}

/// 2026-10-09: The tensor encoding of a rounding model.
fn enc(e: Elem) -> Result<Enc, String> {
    match e.name {
        "f32" => Ok(Enc::F32),
        "bf16" => Ok(Enc::Bf16),
        o => Err(format!("no tensor encoding for {o}")),
    }
}

/// 2026-10-09: `n` values uniform in `[lo, hi)`, rounded into `fmt`.
fn uniform(r: &mut SplitMix64, n: usize, (lo, hi): (f64, f64), fmt: Elem) -> Vec<f64> {
    (0..n)
        .map(|_| {
            fmt.round_saturating(lo + (hi - lo) * r.unit())
                .unwrap_or(lo)
        })
        .collect()
}

/// 2026-10-09: One head slot: the state a model held one token earlier, and that state advanced
/// in f64 by a seeded gaussian token and stored in `fmt`. Returns (earlier, current).
fn slot(r: &mut SplitMix64, g: &Geom, fmt: Elem) -> (Vec<f64>, Vec<f64>) {
    let (kd, vd) = (g.k_dim, g.v_dim);
    let s0 = tensor(r, InputClass::Gaussian, kd, vd, STATE_SCALE, fmt);
    let k = tensor(
        r,
        InputClass::Gaussian,
        1,
        kd,
        1.0 / (kd as f64).sqrt(),
        fmt,
    );
    let v = tensor(r, InputClass::Gaussian, 1, vd, 1.0, fmt);
    let (lo, hi) = decay_clamp();
    let gate = uniform(r, 1, DECAY, fmt)[0].clamp(lo, hi);
    let beta = uniform(r, 1, BETA, fmt)[0];
    let mut s1 = vec![0.0; kd * vd];
    for c in 0..vd {
        let hk: f64 = (0..kd).map(|j| s0[j * vd + c] * k[j]).sum();
        let u = (v[c] - gate * hk) * beta;
        for j in 0..kd {
            let x = gate * s0[j * vd + c] + k[j] * u;
            s1[j * vd + c] = fmt.round_saturating(x).unwrap_or(0.0);
        }
    }
    (s0, s1)
}

/// 2026-10-09: The `[rows, v_heads, k_dim, v_dim]` state of every (row, head), slot
/// `(row * v_heads + head) % slots` each.
fn tiled(slots: &[Vec<f64>], g: &Geom, e: Enc) -> Result<Tensor, String> {
    let per = g.k_dim * g.v_dim;
    let encoded: Vec<Tensor> = slots
        .iter()
        .map(|s| Tensor::encode(e, vec![per], s))
        .collect::<Result<_, _>>()?;
    let heads = g.rows * g.v_heads;
    let mut bytes = Vec::with_capacity(e.bytes_for(heads * per));
    for h in 0..heads {
        bytes.extend_from_slice(&encoded[h % encoded.len()].bytes);
    }
    Ok(Tensor {
        enc: e,
        dims: vec![g.rows, g.v_heads, g.k_dim, g.v_dim],
        bytes: std::sync::Arc::new(bytes),
    })
}

/// 2026-10-09: Fill `case` with one decode step of `g` drawn for `class`: the class shapes the
/// token's q/k/v (q and k at the unit-norm scale the preceding L2 norm gives, v at 1); decay,
/// beta and the state are drawn as the model produces and holds them.
pub fn fill(
    case: &mut Case,
    plan: &Plan,
    g: Geom,
    class: InputClass,
    stream: &dyn Fn(&str) -> SplitMix64,
) -> Result<(), String> {
    g.check()?;
    let f = formats(plan)?;
    let qk = g.k_heads * g.k_dim;
    let key_scale = 1.0 / (g.k_dim as f64).sqrt();
    let act = |name: &str, cols: usize, scale: f64| -> Result<Tensor, String> {
        let vals = tensor(&mut stream(name), class, g.rows, cols, scale, f.qkv);
        Tensor::encode(enc(f.qkv)?, vec![g.rows, cols], &vals)
    };
    case.tensors.insert("q".into(), act("q", qk, key_scale)?);
    case.tensors.insert("k".into(), act("k", qk, key_scale)?);
    case.tensors.insert("v".into(), act("v", g.o_len(), 1.0)?);
    let n = g.rows * g.v_heads;
    let gate = uniform(&mut stream("gate"), n, DECAY, f.gate);
    let beta = uniform(&mut stream("beta"), n, BETA, f.beta);
    case.tensors.insert(
        "gate".into(),
        Tensor::encode(enc(f.gate)?, vec![g.rows, g.v_heads], &gate)?,
    );
    case.tensors.insert(
        "beta".into(),
        Tensor::encode(enc(f.beta)?, vec![g.rows, g.v_heads], &beta)?,
    );
    let mut rs = stream("state");
    let (prev, cur): (Vec<_>, Vec<_>) = (0..STATE_PERIOD.min(n))
        .map(|_| slot(&mut rs, &g, f.state))
        .unzip();
    let se = enc(f.state)?;
    case.tensors.insert("state".into(), tiled(&cur, &g, se)?);
    case.tensors
        .insert("state_prev".into(), tiled(&prev, &g, se)?);
    for (k, v) in [
        ("k_heads", g.k_heads),
        ("k_dim", g.k_dim),
        ("v_heads", g.v_heads),
        ("v_dim", g.v_dim),
    ] {
        case.scalars.insert(k.into(), v as f64);
    }
    case.out = (vec![g.rows, g.cols()], enc(f.out)?);
    Ok(())
}

/// 2026-10-09: Apply `m` to a gdn case. `state_stale` hands the kernel the state one token
/// earlier; it reaches every output.
pub fn mutate(case: &mut Case, m: &Mutation) -> Result<Vec<usize>, String> {
    match m {
        Mutation::StateStale => {
            let prev = case.tensor("state_prev")?.clone();
            case.tensors.insert("state".into(), prev);
            Ok(Vec::new())
        }
        other => Err(format!(
            "`{}` does not apply to a gated delta rule step",
            other.name()
        )),
    }
}
