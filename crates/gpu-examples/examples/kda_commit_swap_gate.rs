// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: Byte gate for the SSM commit swap (`METRALE_SSM_COMMIT_SWAP`,
//! `ssm_reserve::commit_swap`) on the GLM-5.3-Flash KDA layer at the serving geometry.
//!
//! Two worlds run one random program of operations from one start state and the same hidden
//! rows:
//! - COPY (today): one live buffer and an intermediate 0; a K=2 verify (`decode_k` with the
//!   intermediate as its snapshot) runs eagerly, and a reject copies intermediate 0 over the
//!   live state.
//! - SWAP: two buffers whose roles a `SlotSwaps` bit decides (`resolve_pair`, as the pool's
//!   accessors resolve them) and an `SsmLayerState` whose pointers the kernels read (as
//!   `mix.rs` does); the verify replays a CUDA graph captured per role (as `SlotGraphKey`
//!   keys it), and a reject toggles the bit and swaps the pointers.
//! Operations: verify-accept, verify-reject, one-row decode, snapshot save / restore of the live
//! state through the resolved address (prefix cache, Marconi, a rewind), and the batched-step
//! return home (`unswap_for_batch`: live copied into intermediate 0, then the roles swapped
//! back). After every operation the gate compares the layer output and the live h and conv
//! bytes of the two worlds, and that the sequence's pointers equal the resolved addresses.
//!
//! `--sabotage stale-graph|no-pointer-swap|no-toggle` breaks the SWAP world's protocol, to show
//! the gate goes red.
//!
//!   cargo run -p metrale-gpu-examples --release --example kda_commit_swap_gate \
//!       --features cuda,gpu-examples [-- --sabotage stale-graph]
//!
//! Owner: gpu-examples (GLM-5.3 KDA layer).
//! Invariants: exits 1 when any comparison is not bit-exact.

use std::collections::HashMap;

use anyhow::{Result, bail};
use metrale_gpu_examples::kda_fixture::{KdaFixture, Lcg, down, kda_fixture_with_inputs};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle};
use metrale_model_arch::glm5next_kda::KdaSeqState;
use metrale_model_layers::layer::SsmLayerState;
use metrale_model_layers::ssm_reserve::{SlotSwaps, resolve_pair};

const STEPS: usize = 600;
/// 2026-10-21: Distinct hidden rows drawn up front; operation `n` reads rows from `n % ROWS`.
const ROWS: usize = 64;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Sabotage {
    None,
    StaleGraph,
    NoPointerSwap,
    NoToggle,
}

#[derive(Clone, Copy, Debug)]
enum Op {
    Accept,
    Reject,
    Decode,
    Save,
    Restore,
    Home,
}

/// 2026-10-21: The operation mix, close to serving (P(accept) ~0.63 of verifies).
fn draw(rng: &mut Lcg) -> Op {
    match ((rng.u() + 1.0) * 50.0) as u32 {
        0..=44 => Op::Accept,
        45..=71 => Op::Reject,
        72..=83 => Op::Decode,
        84..=89 => Op::Save,
        90..=95 => Op::Restore,
        _ => Op::Home,
    }
}

struct Buf {
    h: DevicePtr,
    c: DevicePtr,
}

fn buf(g: &dyn GpuBackend, f: &KdaFixture) -> Result<Buf> {
    let (hb, cb) = f.state_bytes();
    let b = Buf {
        h: g.alloc(hb)?,
        c: g.alloc(cb)?,
    };
    g.copy_h2d(&f.s0.0, b.h)?;
    g.copy_h2d(&f.s0.1, b.c)?;
    Ok(b)
}

fn copy_state(
    g: &dyn GpuBackend,
    f: &KdaFixture,
    from: (DevicePtr, DevicePtr),
    to: (DevicePtr, DevicePtr),
    s: u64,
) -> Result<()> {
    let (hb, cb) = f.state_bytes();
    g.copy_d2d_async(from.0, to.0, hb, s)?;
    g.copy_d2d_async(from.1, to.1, cb, s)
}

struct Swap {
    home: Buf,
    inter: Buf,
    swaps: SlotSwaps,
    st: SsmLayerState,
    graphs: HashMap<bool, GraphHandle>,
}

impl Swap {
    /// 2026-10-21: (live, intermediate 0) as (h, conv) pairs, as the pool resolves them.
    fn resolved(&self) -> ((DevicePtr, DevicePtr), (DevicePtr, DevicePtr)) {
        let sw = self.swaps.is_swapped(0);
        let (lh, ih) = resolve_pair(self.home.h, self.inter.h, sw);
        let (lc, ic) = resolve_pair(self.home.c, self.inter.c, sw);
        ((lh, lc), (ih, ic))
    }

    fn pointers_agree(&self) -> bool {
        let ((lh, lc), (ih, ic)) = self.resolved();
        self.st.h_state == lh
            && self.st.conv_state == lc
            && self.st.h_state_intermediates == [ih]
            && self.st.conv_state_intermediates == [ic]
    }

    /// 2026-10-21: The commit swap; the sabotage drops one half of it.
    fn swap_roles(&mut self, sab: Sabotage) -> Result<()> {
        if sab != Sabotage::NoToggle {
            self.swaps.toggle(0);
        }
        if sab != Sabotage::NoPointerSwap {
            self.st.swap_live_with_inter0()?;
        }
        Ok(())
    }
}

/// 2026-10-21: One head count; returns the number of failed checks.
fn run(g: &dyn GpuBackend, heads: usize, sab: Sabotage) -> Result<usize> {
    let (f, hidden) = kda_fixture_with_inputs(g, heads, 2, ROWS + 1, 0xC0DE_0000 + heads as u64)?;
    let s = g.default_stream();
    let hid_row = f.cfg.hidden * 2;
    let out2 = 2 * hid_row;
    let (hb, cb) = f.state_bytes();
    // 2026-10-21: The verify reads its two rows from one fixed buffer, filled before each launch.
    let vin = g.alloc(out2)?;

    let live = buf(g, &f)?;
    let inter = buf(g, &f)?;
    let snap_copy = buf(g, &f)?;
    let mut sw = Swap {
        home: buf(g, &f)?,
        inter: buf(g, &f)?,
        swaps: SlotSwaps::new(1),
        st: SsmLayerState {
            h_state: DevicePtr::NULL,
            conv_state: DevicePtr::NULL,
            h_state_checkpoint: None,
            conv_state_checkpoint: None,
            h_state_intermediates: Vec::new(),
            conv_state_intermediates: Vec::new(),
            replay_ring: None,
            h_is_f16: false,
            h_prefill_stage: None,
            ple: None,
        },
        graphs: HashMap::new(),
    };
    sw.st.h_state = sw.home.h;
    sw.st.conv_state = sw.home.c;
    sw.st.h_state_intermediates = vec![sw.inter.h];
    sw.st.conv_state_intermediates = vec![sw.inter.c];
    let snap_swap = buf(g, &f)?;
    g.synchronize(s)?;

    let mut rng = Lcg(0x5A_B0_7A_6E ^ heads as u64);
    let mut fails = 0usize;
    let mut counts: HashMap<String, usize> = HashMap::new();
    for n in 0..STEPS {
        let op = draw(&mut rng);
        let row = hidden.offset((n % ROWS) * hid_row);
        // 2026-10-21: COPY world.
        let ref_live = KdaSeqState {
            conv: live.c,
            recurrent: live.h,
        };
        let mut ref_out = None;
        match op {
            Op::Accept | Op::Reject => {
                g.copy_d2d_async(row, vin, out2, s)?;
                f.layer
                    .decode_k(g, vin, 2, &ref_live, &f.ws, &[(inter.h, inter.c)], s)?;
                ref_out = Some(out2);
                if matches!(op, Op::Reject) {
                    copy_state(g, &f, (inter.h, inter.c), (live.h, live.c), s)?;
                }
            }
            Op::Decode => {
                f.layer.decode(g, row, &ref_live, &f.ws, s)?;
                ref_out = Some(hid_row);
            }
            Op::Save => copy_state(g, &f, (live.h, live.c), (snap_copy.h, snap_copy.c), s)?,
            Op::Restore => copy_state(g, &f, (snap_copy.h, snap_copy.c), (live.h, live.c), s)?,
            // 2026-10-21: A home slot needs nothing; the COPY world is always home.
            Op::Home => {}
        }
        g.synchronize(s)?;
        let want_out = ref_out.map(|b| down(g, f.ws.final_out, b)).transpose()?;
        let want = (down(g, live.h, hb)?, down(g, live.c, cb)?);

        // 2026-10-21: SWAP world.
        let kda = KdaSeqState {
            conv: sw.st.conv_state,
            recurrent: sw.st.h_state,
        };
        match op {
            Op::Accept | Op::Reject => {
                g.copy_d2d_async(row, vin, out2, s)?;
                let key = sab != Sabotage::StaleGraph && sw.swaps.is_swapped(0);
                let graph = match sw.graphs.get(&key) {
                    Some(&gr) => gr,
                    None => {
                        let snaps = [(
                            sw.st.h_state_intermediates[0],
                            sw.st.conv_state_intermediates[0],
                        )];
                        g.begin_capture(s)?;
                        f.layer.decode_k(g, vin, 2, &kda, &f.ws, &snaps, s)?;
                        let gr = g.end_capture(s)?;
                        if gr.0 == 0 {
                            bail!("the K=2 verify did not capture");
                        }
                        sw.graphs.insert(key, gr);
                        gr
                    }
                };
                g.launch_graph(graph, s)?;
                if matches!(op, Op::Reject) {
                    sw.swap_roles(sab)?;
                }
            }
            Op::Decode => f.layer.decode(g, row, &kda, &f.ws, s)?,
            Op::Save => copy_state(g, &f, sw.resolved().0, (snap_swap.h, snap_swap.c), s)?,
            Op::Restore => copy_state(g, &f, (snap_swap.h, snap_swap.c), sw.resolved().0, s)?,
            Op::Home => {
                if sw.swaps.is_swapped(0) {
                    let (l, i) = sw.resolved();
                    copy_state(g, &f, l, i, s)?;
                    sw.swap_roles(sab)?;
                }
            }
        }
        g.synchronize(s)?;
        let got_out = ref_out.map(|b| down(g, f.ws.final_out, b)).transpose()?;
        let (lh, lc) = sw.resolved().0;
        let got = (down(g, lh, hb)?, down(g, lc, cb)?);
        *counts.entry(format!("{op:?}")).or_default() += 1;

        let mut fail = |what: &str| {
            if fails < 8 {
                println!("  [FAIL] heads={heads} step {n} {op:?}: {what}");
            }
            fails += 1;
        };
        if got_out != want_out {
            fail("layer output differs");
        }
        if got.0 != want.0 {
            fail("live h state differs");
        }
        if got.1 != want.1 {
            fail("live conv state differs");
        }
        if !sw.pointers_agree() {
            fail("sequence pointers disagree with the resolved addresses");
        }
    }
    for gr in sw.graphs.values() {
        g.destroy_graph(*gr)?;
    }
    let mut c: Vec<_> = counts.into_iter().collect();
    c.sort();
    println!(
        "  heads={heads}: {STEPS} ops {c:?}, swapped at end: {}, graphs: {}, failed checks: {fails}",
        sw.swaps.is_swapped(0),
        sw.graphs.len()
    );
    Ok(fails)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let sab = match args
        .iter()
        .position(|a| a == "--sabotage")
        .map(|i| args.get(i + 1))
    {
        None => Sabotage::None,
        Some(Some(v)) if v == "stale-graph" => Sabotage::StaleGraph,
        Some(Some(v)) if v == "no-pointer-swap" => Sabotage::NoPointerSwap,
        Some(Some(v)) if v == "no-toggle" => Sabotage::NoToggle,
        Some(v) => bail!("--sabotage {v:?}: stale-graph | no-pointer-swap | no-toggle"),
    };
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    println!("SSM commit swap byte gate (GLM-5.3 KDA, K=2), sabotage {sab:?}");
    // 2026-10-21: 22 heads (the largest TP=3 share of 64, the serving rank-0 shape) and 64 (TP=1).
    let fails = run(g, 22, sab)? + run(g, 64, sab)?;
    if fails > 0 {
        println!("RED: {fails} check(s) failed");
        std::process::exit(1);
    }
    println!("GREEN: all checks bit-exact");
    Ok(())
}
