// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The whole-expert bind of the default (EP) layout, moved verbatim out of
//! `glm5_next_load.rs` when it gained the layer slab for decoded csf scales.
//!
//! Owner: model-arch weight loader.
//! Invariants: a whole expert's only new allocation is its decoded csf scales, in the layer's
//! slab ([`expert_scales::place_block_scales`]).

use super::*;

/// 2026-09-25: One routed expert's gate/up/down as NVFP4 triples. Each
/// projection takes the first arm that applies:
///
/// 1. deferred by [`Glm5NextWeightLoader::defer_predicate`]: read from disk,
///    quantised, uploaded;
/// 2. resident U8: bound zero-copy with its `weight_scale` and scalar
///    `weight_scale_2`, and its static activation scale when the checkpoint has one
///    (2026-10-08, [`act_scale::input_scale`]); 2026-10-17: a csf-encoded scale plane is
///    decoded on the host and placed in the layer's `slab` ([`expert_scales`]);
/// 3. resident BF16: read back and quantised; `prune_after_load` frees the
///    source.
///
/// Any other dtype is an error.
pub(super) fn bind_expert(
    gpu: &dyn GpuBackend,
    store: &WeightStore,
    layer: usize,
    id: usize,
    slab: &expert_slab::ExpertSlab,
) -> Result<Glm5NextExpertWeights> {
    use expert_scales::BlockScales;
    let proj = |p: &str| -> Result<(Nvfp4Proj, BlockScales)> {
        let base = format!("mlp.experts.{id}.{p}");
        let wname = qualify(layer, &format!("{base}.weight"));
        // 2026-09-25: Checked first: a deferred tensor has no store entry.
        if let Some(d) = store.deferred(&wname) {
            let q = quantize_deferred_expert_proj(gpu, store, d, &base)?;
            return Ok((q, BlockScales::Resident(q.scale)));
        }
        let w = store.get(&wname)?;
        match w.dtype {
            WeightDtype::UInt8 => {
                let scale = expert_scales::block_scales(gpu, store, layer, &base, w)?;
                let s2 = store.get(&qualify(layer, &format!("{base}.weight_scale_2")))?;
                let s2 = host_f32(gpu, s2, &format!("{base}.weight_scale_2"))?;
                let [s2] = s2[..] else {
                    bail!("{base}.weight_scale_2 is not a scalar");
                };
                // 2026-10-17: The pointer is set by `place_block_scales` below.
                let p = Nvfp4Proj {
                    packed: w.ptr,
                    scale: DevicePtr::NULL,
                    scale_2: s2,
                    input_scale: input_scale(gpu, store, layer, &base)?,
                };
                Ok((p, scale))
            }
            WeightDtype::BF16 => {
                let q = quantize_expert_proj(gpu, store, w, &base)?;
                Ok((q, BlockScales::Resident(q.scale)))
            }
            other => bail!("{base}.weight is {other:?}, expected packed U8 NVFP4 or BF16"),
        }
    };
    let ((mut g, gs), (mut u, us), (mut d, ds)) =
        (proj("gate_proj")?, proj("up_proj")?, proj("down_proj")?);
    [g.scale, u.scale, d.scale] =
        expert_scales::place_block_scales(gpu, store, slab, id, [gs, us, ds])?;
    Ok(Glm5NextExpertWeights {
        gate_proj: g,
        up_proj: u,
        down_proj: d,
    })
}
