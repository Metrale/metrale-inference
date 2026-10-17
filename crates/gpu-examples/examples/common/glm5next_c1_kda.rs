// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: Part 3 of `glm5next_c1_fuse_microtest`: the KDA recurrence of ONE decode row at
//! the TP=3 per-rank shape (22 heads of 128), `kda_recurrent_decode_bf16_smem` (the decode's
//! kernel) against `kda_recurrent_decode_bf16_seq_reg` at T = 1 for several block widths
//! (the state column in registers, all of its loads in flight at once), and against
//! `kda_recurrent_decode_bf16_rows_reg` at one row. Sixteen consecutive tokens step one state;
//! every token's output and the final state must be byte-identical.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants: as `glm5next_c1_fuse_microtest`'s.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::fixture::{Lcg, same, up_bf16, up_f32};
use super::timing::report;
use super::{read, tg, zeros};

const H: usize = 22;
const D: usize = 128;
/// 2026-10-10: Tokens stepped on one state, and the decode's KDA layers.
const TOKENS: usize = 16;
const KDA_LAYERS: usize = 34;

/// 2026-10-10: How one token's recurrence is launched.
#[derive(Clone, Copy)]
enum Arm {
    Smem,
    SeqReg(usize),
    RowsReg,
}

impl Arm {
    fn name(self) -> String {
        match self {
            Arm::Smem => "smem".into(),
            Arm::SeqReg(vb) => format!("seq_reg vb={vb}"),
            Arm::RowsReg => "rows_reg".into(),
        }
    }
}

/// 2026-10-10: One token's inputs: q|k|v BF16 rows (`3 * H * D`), the log-decay gate and beta.
struct Token {
    qkv: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
}

#[allow(clippy::too_many_arguments)]
fn step(
    g: &dyn GpuBackend,
    k: [KernelHandle; 3],
    arm: Arm,
    t: &Token,
    state: DevicePtr,
    out: DevicePtr,
    s: u64,
) -> Result<()> {
    let scale = 1.0 / (D as f32).sqrt();
    fn inputs<'a>(l: KernelLaunch<'a>, t: &Token) -> KernelLaunch<'a> {
        let qkv = H * D;
        l.arg_ptr(t.qkv)
            .arg_ptr(t.qkv.offset(qkv * 2))
            .arg_ptr(t.qkv.offset(qkv * 4))
            .arg_ptr(t.gate)
            .arg_ptr(t.beta)
    }
    let head = |l| inputs(l, t);
    match arm {
        Arm::Smem => head(
            KernelLaunch::new(g, k[0])
                .grid([H as u32, (D / 32) as u32, 1])
                .block([32, 1, 1])
                .shared_mem(((3 * D + 32 * (D + 1)) * 4) as u32),
        )
        .arg_ptr(state)
        .arg_ptr(out)
        .arg_u32(H as u32)
        .arg_u32(D as u32)
        .arg_f32(scale)
        .arg_u32(32)
        .launch(s),
        Arm::SeqReg(vb) => head(
            KernelLaunch::new(g, k[1])
                .grid([H as u32, (D / vb) as u32, 1])
                .block([vb as u32, 1, 1])
                .shared_mem((3 * D * 4) as u32),
        )
        .arg_ptr(state)
        .arg_ptr(out)
        .arg_u32(H as u32)
        .arg_f32(scale)
        .arg_u32(1)
        .arg_u32(0)
        .arg_u32(0)
        .arg_u32(0)
        .arg_u32(0)
        .launch(s),
        Arm::RowsReg => {
            let mut l = head(
                KernelLaunch::new(g, k[2])
                    .grid([H as u32, 1, 1])
                    .block([D as u32, 1, 1])
                    .shared_mem((3 * D * 4) as u32),
            )
            .arg_ptr(out)
            .arg_u32(H as u32)
            .arg_f32(scale)
            .arg_u32(0)
            .arg_u32(0)
            .arg_u32(0)
            .arg_u32(0)
            .arg_u64(state.0);
            for _ in 1..16 {
                l = l.arg_u64(0);
            }
            for _ in 0..16 {
                l = l.arg_u32(0);
            }
            l.launch(s)
        }
    }
}

pub(crate) fn kda_one_row(g: &dyn GpuBackend, s: u64, rng: &mut Lcg) -> Result<()> {
    let k = [
        g.kernel("kda_recurrent", "kda_recurrent_decode_bf16_smem")?,
        g.kernel("kda_recurrent", "kda_recurrent_decode_bf16_seq_reg")?,
        g.kernel("kda_recurrent", "kda_recurrent_decode_bf16_rows_reg")?,
    ];
    let qkv = H * D;
    let tokens: Vec<Token> = (0..TOKENS)
        .map(|_| {
            Ok(Token {
                qkv: up_bf16(g, &rng.vec(3 * qkv, 0.3))?,
                gate: up_f32(
                    g,
                    &(0..qkv)
                        .map(|_| -0.05 - rng.next().abs() * 0.1)
                        .collect::<Vec<_>>(),
                )?,
                beta: up_f32(
                    g,
                    &(0..H).map(|_| 0.5 + rng.next() * 0.2).collect::<Vec<_>>(),
                )?,
            })
        })
        .collect::<Result<_>>()?;
    let init = rng.vec(H * D * D, 0.1);
    let (sb, ob) = (H * D * D * 4, TOKENS * qkv * 4);
    let run = |arm: Arm| -> Result<(DevicePtr, DevicePtr)> {
        let (state, out) = (up_f32(g, &init)?, zeros(g, ob)?);
        for (i, t) in tokens.iter().enumerate() {
            step(g, k, arm, t, state, out.offset(i * qkv * 4), s)?;
        }
        Ok((state, out))
    };
    let (ref_state, ref_out) = run(Arm::Smem)?;
    let states = (0..KDA_LAYERS)
        .map(|_| up_f32(g, &init))
        .collect::<Result<Vec<_>>>()?;
    let (want_state, want_out) = (read(g, s, ref_state, sb)?, read(g, s, ref_out, ob)?);
    for arm in [
        Arm::SeqReg(32),
        Arm::SeqReg(16),
        Arm::SeqReg(64),
        Arm::SeqReg(128),
        Arm::RowsReg,
    ] {
        let (state, out) = run(arm)?;
        let what = format!("KDA one row {} ({TOKENS} tokens)", arm.name());
        same(&format!("{what} outputs"), &want_out, &read(g, s, out, ob)?)?;
        same(
            &format!("{what} state"),
            &want_state,
            &read(g, s, state, sb)?,
        )?;
        // 2026-10-10: Timed over one state per KDA layer, as the decode steps them: 34 states of
        // 1.4 MB do not stay in L2 between replays, one would.
        let layer = std::cell::Cell::new(0usize);
        let time = |arm: Arm| {
            tg(g, s, KDA_LAYERS, &|s| {
                let l = layer.get();
                layer.set((l + 1) % KDA_LAYERS);
                step(g, k, arm, &tokens[0], states[l], out, s)
            })
        };
        report(
            &format!("KDA 1-row {}", arm.name()),
            1,
            time(Arm::Smem)?,
            time(arm)?,
        );
        g.free(state)?;
        g.free(out)?;
    }
    for p in states
        .iter()
        .copied()
        .chain([ref_state, ref_out])
        .chain(tokens.iter().flat_map(|t| [t.qkv, t.gate, t.beta]))
    {
        g.free(p)?;
    }
    Ok(())
}
