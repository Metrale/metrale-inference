// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-10-19: Tensor-parallel attention on one GPU against the CPU reference. Each emulated
//! rank holds its heads' `wq_b` rows and `sink`, its groups' `wo_a` rows and `wo_b` columns, and
//! its own window rings, compressor state, caches and shared slots; a local gather stands in for
//! the communicator. Across prefill and both decode steps every layer's index selection must be
//! exact, each rank's `q` and `sa_o` must be the reference's head slice within bf16, and every
//! rank's summed output must be the same bytes and the reference's `attn_out` within bf16.
//!
//! Owner: model-arch, DeepSeek-V4.1.
//! Invariants: none beyond the types.

use super::*;
use crate::tp_reduce_v41::CombineRows;
use crate::tp_slice_v41::rank_share;
use metrale_comm::CommBackend;
use metrale_gpu_runtime::gpu::GpuBackend;

/// 2026-10-19: A one-GPU stand-in for the communicator: the gather copies every rank's partial.
struct LocalGather<'a> {
    gpu: &'a dyn GpuBackend,
    partials: Vec<DevicePtr>,
}

impl CommBackend for LocalGather<'_> {
    fn all_reduce(&self, _p: u64, _b: usize) -> anyhow::Result<()> {
        anyhow::bail!("not used")
    }
    fn all_gather(&self, _send: u64, recv: u64, bytes: usize) -> anyhow::Result<()> {
        let s = self.gpu.default_stream();
        for (r, p) in self.partials.iter().enumerate() {
            self.gpu
                .copy_d2d_async(*p, DevicePtr(recv + (r * bytes) as u64), bytes, s)?;
        }
        self.gpu.synchronize(s)
    }
    fn reduce_scatter(&self, _s: u64, _r: u64, _b: usize) -> anyhow::Result<()> {
        anyhow::bail!("not used")
    }
    fn broadcast(&self, _p: u64, _b: usize, _root: usize) -> anyhow::Result<()> {
        anyhow::bail!("not used")
    }
    fn send_to(&self, _p: u64, _b: usize, _peer: usize, _s: u64) -> anyhow::Result<()> {
        anyhow::bail!("not used")
    }
    fn recv_from(&self, _p: u64, _b: usize, _peer: usize, _s: u64) -> anyhow::Result<()> {
        anyhow::bail!("not used")
    }
    fn barrier(&self) -> anyhow::Result<()> {
        Ok(())
    }
    fn rank(&self) -> usize {
        0
    }
    fn world_size(&self) -> usize {
        self.partials.len()
    }
}

struct Rank {
    shard: Shard,
    attn: AttnV41,
    weights: Vec<AttnV41LayerWeights>,
    states: Vec<AttnV41LayerState>,
    shared: SharedV41,
}

/// 2026-10-19: The heads `keep` of a `[tokens, heads, hd]` activation.
fn heads_of(v: &[f32], heads: usize, hd: usize, keep: &std::ops::Range<usize>) -> Vec<f32> {
    v.chunks_exact(heads * hd)
        .flat_map(|t| t[keep.start * hd..keep.end * hd].iter().copied())
        .collect()
}

#[test]
#[ignore = "requires a CUDA GB10 + the deepseek-v4-flash kernel target"]
fn two_head_sliced_ranks_sum_to_the_reference_attention_on_every_layer_and_regime() {
    let golden = Golden::load();
    let c = ModelCfg::from_golden(&golden);
    let t = EngramTables::from_golden(&golden);
    let traces = reference_traces(&golden, &c, &t);
    let rw = ModelWeights::from_golden(&golden, &c, &t);
    let gpu = backend();
    let g: &dyn GpuBackend = &gpu;
    let s = g.default_stream();
    let world = 2;
    let unit = c.n_heads / c.groups;
    let mut ranks: Vec<Rank> = (0..world)
        .map(|r| {
            let heads = rank_share(c.n_heads, unit, world, r).unwrap();
            let shard = Shard {
                groups: heads.start / unit..heads.end / unit,
                heads,
            };
            let mut cfg = cfg_of(&c);
            cfg.n_heads = shard.heads.len();
            cfg.groups = shard.groups.len();
            cfg.tp_world = world;
            let attn = AttnV41::new(g, cfg.clone(), at_defaults()).unwrap();
            let weights: Vec<_> = (0..c.n_layers)
                .map(|l| layer_weights(g, &c, &rw, l, &shard))
                .collect();
            let states = (0..c.n_layers)
                .map(|l| AttnV41LayerState::new(g, &cfg, role_of(&c, l)).unwrap())
                .collect();
            Rank {
                shard,
                attn,
                weights,
                states,
                shared: SharedV41::default(),
            }
        })
        .collect();
    assert!(ranks.iter().all(|r| !r.shard.heads.is_empty()));
    let comm = LocalGather {
        gpu: g,
        partials: ranks
            .iter()
            .map(|r| r.attn.tp.as_ref().unwrap().reduce.partial())
            .collect(),
    };
    let hd = c.head_dim;
    for (regime, start_pos, trace) in &traces {
        let m = trace.tokens;
        for l in 0..c.n_layers {
            let lt = &trace.layers[l];
            let what = format!("{regime}.L{l}");
            let x = up_bf16(g, &lt.attn_in);
            let mut outs = Vec::new();
            for (r, rank) in ranks.iter_mut().enumerate() {
                assert!(!rank.attn.decode_capturable(&rank.weights[l]));
                let run = rank
                    .attn
                    .forward(
                        g,
                        &rank.weights[l],
                        &mut rank.states[l],
                        &mut rank.shared,
                        x,
                        m,
                        *start_pos,
                        Some(&comm),
                        s,
                    )
                    .unwrap();
                assert_eq!(run.idx, lt.attn.idx, "{what}.r{r}: sa_topk_idxs");
                let nl = rank.shard.heads.len();
                let keep = &rank.shard.heads;
                let q = down_bf16(g, run.q, m * nl * hd);
                let want_q = heads_of(&lt.attn.q, c.n_heads, hd, keep);
                assert_within_bf16(&format!("{what}.r{r}.sa_q"), &q, &want_q);
                let o = down_bf16(g, run.o, m * nl * hd);
                let want_o = heads_of(&lt.attn.o, c.n_heads, hd, keep);
                assert_within_bf16(&format!("{what}.r{r}.sa_o"), &o, &want_o);
                outs.push(run.out);
            }
            // 2026-10-19: The gather is sequential here, so the earlier ranks summed partials
            // the later ranks had not written yet; with every partial now in place they sum again.
            let rows = CombineRows {
                rows: m,
                pos: DevicePtr::NULL,
                topk: 0,
                shared_base: 0,
                m,
            };
            for rank in &ranks[..world - 1] {
                let tp = rank.attn.tp.as_ref().unwrap();
                tp.reduce
                    .reduce(g, Some(&comm), rows, rank.attn.out_ptr(), s)
                    .unwrap();
            }
            g.synchronize(s).unwrap();
            g.free(x).unwrap();
            let bytes: Vec<Vec<u8>> = outs
                .iter()
                .map(|&p| {
                    let mut b = vec![0u8; m * c.dim * 2];
                    g.copy_d2h(p, &mut b).unwrap();
                    b
                })
                .collect();
            assert!(
                bytes.windows(2).all(|w| w[0] == w[1]),
                "{what}: the ranks' summed outputs differ"
            );
            let out = down_bf16(g, outs[0], m * c.dim);
            assert_within_bf16(&format!("{what}.attn_out"), &out, &lt.attn.out);
            println!("  {what}: {world} ranks, idx exact, q/o slices and attn_out within bf16");
        }
    }
    for rank in ranks {
        for mut st in rank.states {
            st.free(g).unwrap();
        }
        rank.attn.free(g).unwrap();
    }
}
