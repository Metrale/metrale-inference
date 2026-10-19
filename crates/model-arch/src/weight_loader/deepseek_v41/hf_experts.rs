// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-10-19: Every layer's routed experts of the safetensors checkpoint, resident on the device
//! as this rank's slice ([`ResidentExperts`]), and the engram row tables. Reader threads read the
//! slices past the page cache; the calling thread uploads each into its pointer-table slot, so
//! the device is driven from one thread. A bounded queue keeps at most a few slices in host
//! memory.
//!
//! Owner: model-arch weight loader.
//! Invariants:
//! - Every expert table is admitted only with the dtype and the full-width shape the config
//!   implies, and every slice fills exactly its slot.
//! - On any error the readers stop and the error is returned; the device tables already
//!   allocated are the caller's to free with the rest of a failed load.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::sync_channel;

use anyhow::{Context, Result, ensure};
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_weights::weights::{Fp8EngramRows, WeightDtype, WeightStore};

use super::hf::read_cut;
use super::source::{EXPERT_TABLES, engram_rows_name, expert_name, scale_of};
use super::tp_cut::{BLOCK, Local, expert_cut};
use crate::moe_v41::resident::ResidentExperts;

/// 2026-10-19: The full-width `[rows, cols]` and dtype of expert table `table` under `config`.
pub fn expert_table_shape(config: &ModelConfig, table: usize) -> (WeightDtype, [usize; 2]) {
    let (dim, inter) = (config.hidden_size, config.moe_intermediate_size);
    let dtype = match table % 2 {
        0 => WeightDtype::UInt8,
        _ => WeightDtype::FP8E8M0,
    };
    let shape = match table {
        0 | 2 => [inter, dim / 2],
        1 | 3 => [inter, dim / BLOCK],
        4 => [dim, inter / 2],
        _ => [dim, inter / BLOCK],
    };
    (dtype, shape)
}

fn alloc_layer(
    gpu: &dyn GpuBackend,
    config: &ModelConfig,
    local: &Local,
    l: usize,
) -> Result<ResidentExperts> {
    let (n, inter, dim) = (config.num_experts, local.inter.len(), config.hidden_size);
    let sizes = ResidentExperts {
        n_experts: n,
        inter,
        dim,
        w1: DevicePtr(0),
        w1_s: DevicePtr(0),
        w3: DevicePtr(0),
        w3_s: DevicePtr(0),
        w2: DevicePtr(0),
        w2_s: DevicePtr(0),
    };
    let (packed, scale) = (n * sizes.packed_bytes(), n * sizes.scale_bytes());
    let table = |w: &str, bytes: usize, dtype: &str| -> Result<DevicePtr> {
        let p = gpu.alloc(bytes)?;
        let what = format!("layers.{l}.ffn.experts.*.{w} x{n}");
        gpu.label_alloc(p, &format!("{what} resident={dtype} declared={dtype}"));
        Ok(p)
    };
    Ok(ResidentExperts {
        w1: table("w1.weight", packed, "e2m1")?,
        w1_s: table("w1.scale", scale, "e8m0")?,
        w3: table("w3.weight", packed, "e2m1")?,
        w3_s: table("w3.scale", scale, "e8m0")?,
        w2: table("w2.weight", packed, "e2m1")?,
        w2_s: table("w2.scale", scale, "e8m0")?,
        ..sizes
    })
}

/// 2026-10-19: Table `table` of expert `e` of layer `l`, this rank's slice, checked against the
/// slot it fills.
fn read_slice(
    store: &WeightStore,
    config: &ModelConfig,
    local: &Local,
    ex: &ResidentExperts,
    (l, e, table): (usize, usize, usize),
) -> Result<Vec<u8>> {
    let n = expert_name(l, e, table);
    let d = store
        .deferred(&n)
        .with_context(|| format!("deepseek-v4.1: {n} was not deferred by the loader"))?;
    let (dtype, shape) = expert_table_shape(config, table);
    ensure!(
        d.dtype == dtype && d.shape == shape,
        "{n}: {:?} {:?}, expected {dtype:?} {shape:?}",
        d.dtype,
        d.shape
    );
    let bytes = read_cut(d, &expert_cut(table, local)?)?;
    let want = match table % 2 {
        0 => ex.packed_bytes(),
        _ => ex.scale_bytes(),
    };
    ensure!(
        bytes.len() == want,
        "{n}: {} slice bytes for a {want}-byte slot",
        bytes.len()
    );
    Ok(bytes)
}

/// 2026-10-19: Every layer's routed experts, this rank's slice of each, read on `threads`
/// reader threads.
pub fn upload_experts(
    gpu: &dyn GpuBackend,
    store: &WeightStore,
    config: &ModelConfig,
    local: &Local,
    threads: usize,
) -> Result<Vec<ResidentExperts>> {
    let n_layers = config.num_hidden_layers;
    let layers: Vec<ResidentExperts> = (0..n_layers)
        .map(|l| alloc_layer(gpu, config, local, l))
        .collect::<Result<_>>()?;
    let per_layer = config.num_experts * EXPERT_TABLES.len();
    let total = n_layers * per_layer;
    let job = |i: usize| (i / per_layer, (i % per_layer) / 6, i % 6);
    let next = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    let threads = threads.max(1);
    let started = std::time::Instant::now();
    std::thread::scope(|s| -> Result<()> {
        let (tx, rx) = sync_channel::<Result<(usize, Vec<u8>)>>(2 * threads);
        for _ in 0..threads {
            let tx = tx.clone();
            let (next, stop, layers) = (&next, &stop, &layers);
            s.spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= total {
                        break;
                    }
                    let (l, e, t) = job(i);
                    let r = read_slice(store, config, local, &layers[l], (l, e, t)).map(|b| (i, b));
                    if tx.send(r).is_err() {
                        break;
                    }
                }
            });
        }
        drop(tx);
        let mut done = 0usize;
        let result = (|| -> Result<()> {
            for r in rx.iter() {
                let (i, bytes) = r?;
                let (l, e, t) = job(i);
                gpu.copy_h2d(&bytes, DevicePtr(layers[l].addr(t, e as u32)))?;
                done += 1;
                if done.is_multiple_of(per_layer) {
                    tracing::info!(
                        "DeepSeek-V4.1: routed experts {}/{n_layers} layers resident ({:.0} s)",
                        done / per_layer,
                        started.elapsed().as_secs_f64()
                    );
                }
            }
            ensure!(done == total, "uploaded {done} of {total} expert slices");
            Ok(())
        })();
        stop.store(result.is_err(), Ordering::Relaxed);
        result
    })?;
    Ok(layers)
}

/// 2026-10-19: Every engram layer's row table and its scales, read in place.
pub fn engram_rows(store: &WeightStore, config: &ModelConfig) -> Result<Fp8EngramRows> {
    let tables = config
        .engram_layer_ids
        .iter()
        .map(|&l| {
            let n = engram_rows_name(l);
            let get = |n: &str| {
                store
                    .deferred(n)
                    .cloned()
                    .with_context(|| format!("deepseek-v4.1: {n} was not deferred by the loader"))
            };
            Ok((l, get(&n)?, get(&scale_of(&n)?)?))
        })
        .collect::<Result<Vec<_>>>()?;
    Fp8EngramRows::new(tables)
}

#[cfg(test)]
#[path = "hf_experts_tests.rs"]
mod tests;
