// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: Typed borrowed visual-transformer weights, without GPU loading or I/O.
//! The caller supplies component-local names and borrowed storage handles. The binder
//! checks every name, dtype and shape before returning any usable component.

mod config;
pub use config::Config;

use crate::weights::WeightDtype;
use anyhow::{Result, anyhow, ensure};
use std::collections::BTreeMap;

/// 2026-10-07: A borrowed tensor's metadata and storage handle; no payload copy.
pub struct Tensor<'a, T> {
    /// 2026-10-07: Storage retained by the caller (e.g. a mapped tensor or device handle).
    pub storage: &'a T,
    /// 2026-10-07: Physical tensor dimensions before any packing or conversion.
    pub shape: &'a [usize],
    /// 2026-10-07: Actual stored dtype; implicit conversion is refused.
    pub dtype: WeightDtype,
}

/// 2026-10-07: Exact bindings for one single-stream transformer block.
pub struct Block<'a, T> {
    pub q: &'a T,
    pub k: &'a T,
    pub v: &'a T,
    pub o: &'a T,
    pub q_norm: &'a T,
    pub k_norm: &'a T,
    pub gate: &'a T,
    pub up: &'a T,
    pub down: &'a T,
}

/// 2026-10-07: Complete visual transformer binding, not an executable model.
pub struct Weights<'a, T> {
    pub image_in: &'a T,
    pub text_norm: &'a T,
    pub text_in: &'a T,
    pub text_out: &'a T,
    pub time_in: &'a T,
    pub time_out: &'a T,
    pub modulation: &'a T,
    pub final_scale: &'a T,
    pub image_out: &'a T,
    pub blocks: Vec<Block<'a, T>>,
}

/// 2026-10-07: Validate the exact component inventory and bind borrowed storage.
/// All checkpoint tensors are BF16, including Q/K and zero-centered text norm
/// weights. This does not change the required FP32 normalization arithmetic.
pub fn bind<'a, T>(
    config: &Config,
    tensors: &BTreeMap<String, Tensor<'a, T>>,
) -> Result<Weights<'a, T>> {
    ensure!(
        tensors.len() == 9 + 9 * config.layers(),
        "unexpected transformer tensor count"
    );
    let get = |name: &str, shape: &[usize]| -> Result<&'a T> {
        let tensor = tensors
            .get(name)
            .ok_or_else(|| anyhow!("missing transformer tensor {name}"))?;
        ensure!(
            tensor.dtype == WeightDtype::BF16,
            "wrong dtype for {name}: expected BF16"
        );
        ensure!(
            tensor.shape == shape,
            "wrong shape for {name}: {:?}, expected {shape:?}",
            tensor.shape
        );
        Ok(tensor.storage)
    };
    let h = config.hidden();
    let d = config.head_dim();
    let m = config.intermediate();
    let c = config.channels();
    let mut blocks = Vec::with_capacity(config.layers());
    for layer in 0..config.layers() {
        let get_layer = |suffix: &str, shape: &[usize]| {
            get(
                &format!("transformer_blocks.{layer}.{suffix}.weight"),
                shape,
            )
        };
        blocks.push(Block {
            q: get_layer("attn.to_q", &[h, h])?,
            k: get_layer("attn.to_k", &[h, h])?,
            v: get_layer("attn.to_v", &[h, h])?,
            o: get_layer("attn.to_out.0", &[h, h])?,
            q_norm: get_layer("attn.norm_q", &[d])?,
            k_norm: get_layer("attn.norm_k", &[d])?,
            gate: get_layer("img_mlp.gate_layer", &[m, h])?,
            up: get_layer("img_mlp.proj", &[m, h])?,
            down: get_layer("img_mlp.out", &[h, m])?,
        });
    }
    Ok(Weights {
        image_in: get("img_in.weight", &[h, c])?,
        text_norm: get("txt_in.text_norm.weight", &[h])?,
        text_in: get("txt_in.in_layer.weight", &[h, h])?,
        text_out: get("txt_in.out_layer.weight", &[h, h])?,
        time_in: get(
            "time_text_embed.timestep_embedder.linear_1.weight",
            &[h, 256],
        )?,
        time_out: get("time_text_embed.timestep_embedder.linear_2.weight", &[h, h])?,
        modulation: get("modulation.1.weight", &[4 * h, h])?,
        final_scale: get("norm_out.linear.weight", &[h, h])?,
        image_out: get("proj_out.weight", &[c, h])?,
        blocks,
    })
}

/// 2026-10-07: Bind an already loaded, transformer-only store without copying
/// tensor storage. Callers remain responsible for checkpoint integrity and I/O.
pub fn bind_store<'a>(
    config: &Config,
    store: &'a crate::weights::WeightStore,
) -> Result<Weights<'a, crate::weights::WeightTensor>> {
    let mut tensors = BTreeMap::new();
    for name in store.names() {
        let tensor = store.get(name)?;
        tensors.insert(
            name.to_owned(),
            Tensor {
                storage: tensor,
                shape: &tensor.shape,
                dtype: tensor.dtype,
            },
        );
    }
    bind(config, &tensors)
}

#[cfg(test)]
mod tests;
