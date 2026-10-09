// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The GDN conv step reference (`conv1d_l2norm`): the causal depthwise conv1d
//! update with SiLU and the per-head L2 norm of the q/k channels, as
//! causal_conv1d_update_l2norm_f32(_strided) compute it (causal_conv1d.cu:413-472, 496-557):
//!
//! ```text
//! window = window[1..] ++ [x]                       per channel, stored back (f32)
//! a      = sum_t window[t] * w[ch, t]               no bias (the GDN conv has none)
//! s      = a * (1 / (1 + __expf(-a)))
//! out    = s * rsqrt(sum_head s^2 + eps)            q/k channels (0..qk_channels)
//! out    = s                                        v channels
//! ```
//!
//! Tensors of a case: `x` `[rows, dim]` (the token's q|k|v projection, bf16), `w`
//! `[dim, d_conv]` (bf16), `window` `[rows, dim, d_conv]` (f32, oldest first: the previous
//! `d_conv` inputs) and `window_prev` (the window one token earlier). Scalars `dim`, `d_conv`,
//! `qk_channels`, `head_dim`, `eps`. Output `[rows, dim + dim * d_conv]`: each row's conv
//! output followed by its updated window.
//!
//! Owner: metrale-accuracy.
//! Invariants:
//! - The geometry and the launch scalars (`d_conv`, `eps`) come from the point's runtime
//!   values; nothing is assumed.
//! - The fused L2 norm is taken at the conv op's compute and output precisions; the family
//!   declares the same for its `l2_norm` op on these kernels, which the CPU test asserts.

use metrale_circuit::format::Format;
use metrale_circuit::pipeline::{StepKind, Value};
use metrale_circuit::state::StateDtype;

use crate::case::{Case, Enc, Tensor};
use crate::elem::{self, Elem, F32_FTZ};
use crate::inputs::{InputClass, SplitMix64, tensor};
use crate::mutation::Mutation;
use crate::plan::Plan;
use crate::points::Shape;

/// 2026-10-09: The geometry and launch scalars of one conv step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Geom {
    /// 2026-10-09: Sequences.
    pub rows: usize,
    /// 2026-10-09: Channels (q|k|v).
    pub dim: usize,
    /// 2026-10-09: Taps.
    pub d_conv: usize,
    /// 2026-10-09: The channels the L2 norm covers (q and k).
    pub qk_channels: usize,
    /// 2026-10-09: Channels per normalized head.
    pub head_dim: usize,
    /// 2026-10-09: The L2 norm's epsilon, as the f32 the launch passes.
    pub eps: f64,
}

/// 2026-10-09: Threads per block of the launchers (ops/ssm_mamba.rs), two heads of 128 each.
pub const BLOCK: usize = 256;

impl Geom {
    /// 2026-10-09: The geometry of a swept shape: `dim` is the node's width; the runtime
    /// values `k_heads`, `k_dim`, `d_conv` and `l2_eps` give the rest.
    pub fn of_shape(shape: &Shape) -> Result<Geom, String> {
        let get = |name: &str| -> Result<f64, String> {
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
        let (k_heads, k_dim) = (get("k_heads")? as usize, get("k_dim")? as usize);
        let g = Geom {
            rows: shape.rows as usize,
            dim: shape.in_dim as usize,
            d_conv: get("d_conv")? as usize,
            qk_channels: 2 * k_heads * k_dim,
            head_dim: k_dim,
            eps: f64::from(get("l2_eps")? as f32),
        };
        if shape.out_dim != shape.in_dim {
            return Err(format!(
                "a conv node of in {} out {}: the step keeps the width",
                shape.in_dim, shape.out_dim
            ));
        }
        g.check()?;
        Ok(g)
    }

    /// 2026-10-09: The geometry a filled case records.
    pub fn of_case(case: &Case) -> Result<Geom, String> {
        let s = |n: &str| case.scalar(n);
        Ok(Geom {
            rows: case.out.0[0],
            dim: s("dim")? as usize,
            d_conv: s("d_conv")? as usize,
            qk_channels: s("qk_channels")? as usize,
            head_dim: s("head_dim")? as usize,
            eps: s("eps")?,
        })
    }

    /// 2026-10-09: The kernels' preconditions (causal_conv1d.cu:513-546): a head is four warps
    /// (`head_dim == 128`, the `base_warp .. + 3` sum), a block holds whole heads and the q/k
    /// channels end on a block edge (`block_needs_l2` is per block), at least one tap.
    pub fn check(&self) -> Result<(), String> {
        if self.head_dim != 128 {
            return Err(format!(
                "{self:?}: the L2 norm sums four warps, head_dim must be 128"
            ));
        }
        if !self.qk_channels.is_multiple_of(BLOCK) || self.qk_channels > self.dim {
            return Err(format!(
                "{self:?}: the q/k channels must end on a {BLOCK}-channel block edge"
            ));
        }
        if self.d_conv == 0 || self.eps.is_nan() || self.eps <= 0.0 {
            return Err(format!("{self:?}: no taps or a non-positive epsilon"));
        }
        Ok(())
    }

    /// 2026-10-09: Output columns per row (conv output, then the window).
    pub fn cols(&self) -> usize {
        self.dim * (1 + self.d_conv)
    }
}

/// 2026-10-09: The formats of a `conv1d_update` plan.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Formats {
    /// 2026-10-09: The token's input as read (and the weight's storage).
    pub input: Elem,
    /// 2026-10-09: The window as stored.
    pub state: Elem,
    /// 2026-10-09: Products, sums, the sigmoid and the norm.
    pub compute: Elem,
    /// 2026-10-09: The output as written.
    pub out: Elem,
}

/// 2026-10-09: The rounding model of a real-valued format.
fn real(f: Format, ftz: bool) -> Result<Elem, String> {
    match f {
        Format::F32 if ftz => Ok(F32_FTZ),
        Format::F32 => Ok(elem::F32),
        Format::Bf16 => Ok(elem::BF16),
        o => Err(format!("a conv operand in {}", o.name())),
    }
}

/// 2026-10-09: The formats `plan` declares (input; steps state, compute; output).
pub fn formats(plan: &Plan) -> Result<Formats, String> {
    let p = &plan.pipeline;
    let state = match plan.step(StepKind::State) {
        Some(Value::State(StateDtype::F32)) => real(Format::F32, plan.ftz)?,
        Some(Value::State(d)) => {
            return Err(format!("a {} window: the reference stores f32", d.name()));
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
    let input = real(
        *p.inputs.first().ok_or("the pipeline names no input")?,
        plan.ftz,
    )?;
    let out = real(
        *p.outputs.first().ok_or("the pipeline names no output")?,
        plan.ftz,
    )?;
    if out != state {
        return Err(format!(
            "the output in {} and the window in {}: the case writes both as one output",
            out.name, state.name
        ));
    }
    Ok(Formats {
        input,
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

/// 2026-10-09: Fill `case` with one conv step of `g` drawn for `class`: the class shapes the
/// token's input and the previous inputs the window holds (a window is the last inputs, in the
/// input's format, widened to f32); the weights are drawn as a checkpoint holds them.
pub fn fill(
    case: &mut Case,
    plan: &Plan,
    g: Geom,
    class: InputClass,
    stream: &dyn Fn(&str) -> SplitMix64,
) -> Result<(), String> {
    g.check()?;
    let f = formats(plan)?;
    let x = tensor(&mut stream("x"), class, g.rows, g.dim, 1.0, f.input);
    let w = tensor(
        &mut stream("w"),
        InputClass::Gaussian,
        g.dim,
        g.d_conv,
        1.0 / (g.d_conv as f64).sqrt(),
        f.input,
    );
    // 2026-10-09: The d_conv + 1 previous inputs of every channel, oldest first.
    let d1 = g.d_conv + 1;
    let hist = tensor(
        &mut stream("history"),
        class,
        g.rows,
        g.dim * d1,
        1.0,
        f.input,
    );
    let hist = &hist;
    let window = |from: usize| -> Vec<f64> {
        (0..g.rows * g.dim)
            .flat_map(|rc| (from..from + g.d_conv).map(move |t| hist[rc * d1 + t]))
            .collect()
    };
    let (ie, se) = (enc(f.input)?, enc(f.state)?);
    case.tensors
        .insert("x".into(), Tensor::encode(ie, vec![g.rows, g.dim], &x)?);
    case.tensors
        .insert("w".into(), Tensor::encode(ie, vec![g.dim, g.d_conv], &w)?);
    let dims = vec![g.rows, g.dim, g.d_conv];
    case.tensors.insert(
        "window".into(),
        Tensor::encode(se, dims.clone(), &window(1))?,
    );
    case.tensors
        .insert("window_prev".into(), Tensor::encode(se, dims, &window(0))?);
    for (k, v) in [
        ("dim", g.dim as f64),
        ("d_conv", g.d_conv as f64),
        ("qk_channels", g.qk_channels as f64),
        ("head_dim", g.head_dim as f64),
        ("eps", g.eps),
    ] {
        case.scalars.insert(k.into(), v);
    }
    case.out = (vec![g.rows, g.cols()], enc(f.out)?);
    Ok(())
}

/// 2026-10-09: Apply `m` to a conv case. `state_stale` hands the kernel the window one token
/// earlier; it reaches every output.
pub fn mutate(case: &mut Case, m: &Mutation) -> Result<Vec<usize>, String> {
    match m {
        Mutation::StateStale => {
            let prev = case.tensor("window_prev")?.clone();
            case.tensors.insert("window".into(), prev);
            Ok(Vec::new())
        }
        other => Err(format!("`{}` does not apply to a conv step", other.name())),
    }
}
