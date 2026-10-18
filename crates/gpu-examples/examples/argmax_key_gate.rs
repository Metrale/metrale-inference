// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: Bit-identity gate: the sharded feed argmax (`argmax_key_shard_bf16` over each
//! slice of a row, then `argmax_key_resolve`) against `argmax_bf16_batch_feed` over the whole
//! row, for every split tried.
//!
//! Owner: model-arch examples (sampling kernels).
//! Invariants:
//! - The run fails (panics) unless every split gives the whole-row kernel's token, through
//!   both stand-ins for the ranks: one key buffer that every slice raises with `atomicMax`,
//!   and one key pair per slice combined by the resolve kernel's peer max, as the one-shot
//!   exchange hands each rank its peers' keys.
//!
//! Rows cover planted ties across slice boundaries and inside one thread's stride, `±0`, the
//! `-1e30` floor, `-inf`, NaN, masks that hit the plain argmax, dense random ties, and random
//! BF16 rows of the GLM-5.3 vocabulary under its three-rank split.
//!
//!   cargo run -p metrale-gpu-examples --release --example argmax_key_gate \
//!       --features cuda,gpu-examples

use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::layers::ops::{
    argmax_bf16_batch_feed, argmax_key_resolve, argmax_key_shard_bf16,
};

const NO_ID: u32 = u32::MAX;
/// 2026-10-10: Stride of the stand-in exchange slots (a one-shot slot is wider than 16 bytes).
const PEER_SLOT: usize = 64;

fn bf16(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

struct Rig<'a> {
    g: &'a dyn GpuBackend,
    row: DevicePtr,
    masks: DevicePtr,
    keys: DevicePtr,
    /// 2026-10-10: Per-slice key pairs at `PEER_SLOT` bytes apart: the one-shot exchange's
    /// slots, slice 0 standing in for this rank.
    peers: DevicePtr,
    cell: DevicePtr,
    k_feed: metrale_gpu_runtime::gpu::KernelHandle,
    k_shard: metrale_gpu_runtime::gpu::KernelHandle,
    k_resolve: metrale_gpu_runtime::gpu::KernelHandle,
}

impl Rig<'_> {
    fn read_cell(&self) -> u32 {
        let mut b = [0u8; 4];
        self.g.copy_d2h(self.cell, &mut b).unwrap();
        u32::from_le_bytes(b)
    }

    /// 2026-10-10: The tokens for `row` cut at `cuts`: the whole-row kernel, the slices keyed
    /// into one buffer, and the slices keyed apart and combined by the resolve's peer max.
    fn tokens(&self, row: &[u16], mask: [u32; 2], cuts: &[usize]) -> (u32, u32, u32) {
        let g = self.g;
        let n = row.len();
        let bytes: Vec<u8> = row.iter().flat_map(|v| v.to_le_bytes()).collect();
        g.copy_h2d(&bytes, self.row).unwrap();
        let m: Vec<u8> = mask.iter().flat_map(|v| v.to_le_bytes()).collect();
        g.copy_h2d(&m, self.masks).unwrap();
        let s = g.default_stream();
        argmax_bf16_batch_feed(
            g,
            self.k_feed,
            self.row,
            self.masks,
            self.cell,
            n as u32,
            1,
            n as u32,
            s,
        )
        .unwrap();
        g.synchronize(s).unwrap();
        let whole = self.read_cell();
        g.memset(self.keys, 0, 16).unwrap();
        let mut starts = vec![0];
        starts.extend(cuts.iter().copied().filter(|&c| c > 0 && c < n));
        starts.push(n);
        for w in starts.windows(2) {
            argmax_key_shard_bf16(
                g,
                self.k_shard,
                self.row.offset(w[0] * 2),
                self.masks,
                self.keys,
                w[0] as u32,
                (w[1] - w[0]) as u32,
                s,
            )
            .unwrap();
        }
        argmax_key_resolve(
            g,
            self.k_resolve,
            self.keys,
            DevicePtr::NULL,
            0,
            0,
            self.masks,
            self.cell,
            s,
        )
        .unwrap();
        g.synchronize(s).unwrap();
        let one_buffer = self.read_cell();
        let slices = starts.len() - 1;
        g.memset(self.peers, 0, PEER_SLOT * slices).unwrap();
        for (k, w) in starts.windows(2).enumerate() {
            argmax_key_shard_bf16(
                g,
                self.k_shard,
                self.row.offset(w[0] * 2),
                self.masks,
                self.peers.offset(k * PEER_SLOT),
                w[0] as u32,
                (w[1] - w[0]) as u32,
                s,
            )
            .unwrap();
        }
        argmax_key_resolve(
            g,
            self.k_resolve,
            self.peers,
            self.peers.offset(PEER_SLOT),
            PEER_SLOT as u64,
            (slices - 1) as u32,
            self.masks,
            self.cell,
            s,
        )
        .unwrap();
        g.synchronize(s).unwrap();
        (whole, one_buffer, self.read_cell())
    }

    fn check(&self, row: &[u16], mask: [u32; 2], what: &str) {
        let n = row.len();
        for cuts in [
            vec![],
            vec![n / 2],
            vec![n / 3, 2 * n / 3],
            vec![1, n.saturating_sub(1)],
            vec![n / 4, n / 2, 3 * n / 4],
        ] {
            let (whole, one_buffer, peers) = self.tokens(row, mask, &cuts);
            assert_eq!(
                one_buffer, whole,
                "{what}: n={n} cuts={cuts:?} mask={mask:?}"
            );
            assert_eq!(
                peers, whole,
                "{what} (peers): n={n} cuts={cuts:?} mask={mask:?}"
            );
        }
    }
}

fn main() -> anyhow::Result<()> {
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &gpu;
    const V: usize = 154_856;
    let rig = Rig {
        g,
        row: g.alloc(V * 2).unwrap(),
        masks: g.alloc(8).unwrap(),
        keys: g.alloc(16).unwrap(),
        peers: g.alloc(PEER_SLOT * 4).unwrap(),
        cell: g.alloc(4).unwrap(),
        k_feed: g.kernel("argmax_feed", "argmax_bf16_batch_feed").unwrap(),
        k_shard: g.kernel("argmax_key", "argmax_key_shard_bf16").unwrap(),
        k_resolve: g.kernel("argmax_key", "argmax_key_resolve").unwrap(),
    };

    // 2026-10-10: Planted ties: threads 1 and 2, one thread's stride, a three-rank boundary.
    let mut row = vec![bf16(0.0); 2048];
    row[1] = bf16(5.0);
    row[2] = bf16(5.0);
    rig.check(&row, [NO_ID, NO_ID], "threads 1 and 2");
    let mut row = vec![bf16(0.0); 3072];
    row[1024 + 7] = bf16(5.0);
    row[2048 + 7] = bf16(5.0);
    rig.check(&row, [NO_ID, NO_ID], "one stride");
    let mut row = vec![bf16(-1.0); V];
    for i in [51_647, 51_648, 154_855] {
        row[i] = bf16(9.0);
    }
    rig.check(&row, [NO_ID, NO_ID], "rank boundary");
    rig.check(&row, [51_647, NO_ID], "rank boundary, masked p");

    // 2026-10-10: Signed zeros, the floor, -inf, NaN.
    let mut row = vec![0x8000u16; 3000];
    row[2500] = 0;
    rig.check(&row, [NO_ID, NO_ID], "-0 and +0");
    rig.check(&row, [0, NO_ID], "-0 and +0, mask on 0");
    rig.check(&[bf16(-1e30); 70], [NO_ID, NO_ID], "floor");
    rig.check(
        &[bf16(f32::NEG_INFINITY); 70],
        [0, NO_ID],
        "all -inf, mask on 0",
    );
    let mut row = vec![bf16(1.0); 5000];
    row[17] = 0x7FC0;
    row[4999] = bf16(2.0);
    rig.check(&row, [4999, NO_ID], "NaN and a masked max");

    // 2026-10-10: Random rows over a small value set (dense ties) and random BF16 rows.
    let mut state = 0x1234_5678u32;
    let mut next = || {
        state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        state >> 8
    };
    let set = [
        -1e31f32,
        -1e30,
        -3.5,
        -0.0,
        0.0,
        1.25,
        7.0,
        f32::NEG_INFINITY,
        7.0,
    ];
    for case in 0..300 {
        let n = [37, 1500, 4100, 9000][case % 4];
        let row: Vec<u16> = (0..n)
            .map(|_| bf16(set[next() as usize % set.len()]))
            .collect();
        let pick = |x: u32| match x % 4 {
            0 => NO_ID,
            1 => (n + 3) as u32,
            _ => x % n as u32,
        };
        let mask = [pick(next()), pick(next())];
        rig.check(&row, mask, "dense ties");
    }
    for _ in 0..20 {
        let row: Vec<u16> = (0..V).map(|_| 0x3F00 + (next() % 0x280) as u16).collect();
        let (whole, one_buffer, peers) = rig.tokens(&row, [NO_ID, NO_ID], &[51_648, 103_296]);
        assert_eq!((one_buffer, peers), (whole, whole), "random GLM row");
    }
    println!("argmax_key_gate: PASS (every split, both combine paths, byte-equal tokens)");
    Ok(())
}
