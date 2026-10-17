// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: `METRALE_GLM_LAYER_STREAM=1`: keep the GLM-5.3 text layers' non-expert tensors
//! out of the device weight store and read each layer's from its shard when the layer is
//! collected (`LayerSource::collect`), instead of uploading all of them at load and freeing them
//! in `prune_after_load` once every layer is built.
//!
//! Those are the tensors `is_reuploaded` matches: the binders already build their own buffers
//! from host copies, so the store's device copies only ever served as the source of that host
//! copy. Holding all of them until the last layer is built puts their total (every layer's
//! attention, dense and shared-expert weights) on top of the built layers at the load peak; on a
//! unified-memory box that transient is what decides whether a load fits. Streaming bounds the
//! transient by one layer's host copy.
//!
//! Numerics: none change. A tensor's host bytes are its shard bytes either way (the resident
//! path uploads those bytes and reads them back); `layer_stream_tests` proves the collected
//! layer identical under both.
//!
//! Owner: model-arch weight loader (GLM-5.3).
//! Invariants:
//! - Unset or `0` defers nothing new (today's load); `1` defers exactly the `is_reuploaded`
//!   tensors that are not activation scales (those are already deferred, and read by
//!   `act_scale::input_scale`). Any other value is refused when the layers are loaded.

use std::sync::OnceLock;

use anyhow::{Result, bail};
use metrale_model_weights::weights::WeightDtype;

use super::{is_activation_scale, is_reuploaded};

/// 2026-10-16: The lever's name.
pub(super) const ENV: &str = "METRALE_GLM_LAYER_STREAM";

/// 2026-10-16: Parse a lever value; `None` is unset.
pub(super) fn parse(v: Option<&str>) -> Result<bool> {
    match v {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => bail!("{ENV}={other:?}: expected 0 or 1"),
    }
}

/// 2026-10-16: `METRALE_GLM_LAYER_STREAM`, read once.
pub(super) fn enabled() -> Result<bool> {
    static V: OnceLock<Result<bool, String>> = OnceLock::new();
    V.get_or_init(|| parse(std::env::var(ENV).ok().as_deref()).map_err(|e| format!("{e:#}")))
        .clone()
        .map_err(anyhow::Error::msg)
}

/// 2026-10-16: Whether the defer hook keeps `name` off the device under the lever.
pub(super) fn streams(name: &str, dtype: WeightDtype, num_layers: usize) -> bool {
    is_reuploaded(name, num_layers) && !is_activation_scale(name, dtype)
}

/// 2026-10-16: Whether `LayerSource::collect` reads a deferred tensor of its layer (`rel` is
/// the layer-relative name of `name`) from the shard. Routed experts are bound by name and
/// activation scales read on their own, as when they are deferred without the lever; for a text
/// layer this is exactly [`streams`], so with the lever off no deferred tensor qualifies.
pub(super) fn collected_from_shard(rel: &str, name: &str, dtype: WeightDtype) -> bool {
    !rel.starts_with("mlp.experts.") && !is_activation_scale(name, dtype)
}
