// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The synthesis units of a mock: each kept tensor's value class, quantized weights
//! grouped with their scales, routers fitted to the profile.
//!
//! Owner: metrale-ml-utils.
//! Invariants:
//! - Units follow the source index's name order; a group's tensors are contiguous.
//! - Histogram routing applies to main-layer routers only (the draft head's router stays
//!   random), and zeroes the bias-channel row of main-layer residual writers only.

use std::collections::{BTreeMap, BTreeSet};

use metrale_circuit::LayerSchedule;

use super::{OutTensor, RouterFit, Unit};
use crate::error::{MlError, Result};
use crate::index::{TensorEntry, TensorIndex};
use crate::rename::Renamer;
use crate::rng::Stream;
use crate::routing::{BiasChannel, RoutingCalibration, RoutingProfile, bias_channel, fit};
use crate::scheme::{QuantGroup, find_groups};
use crate::spec::{MockSpec, RoutingMode};
use crate::synth::GroupDtypes;
use crate::values::{ValueClass, classify_plain, is_residual_writer, residual_gain};

/// 2026-10-03: What units are built from.
pub(super) struct Ctx<'a> {
    pub index: &'a TensorIndex,
    pub renamer: &'a Renamer<'a>,
    pub schedule: &'a LayerSchedule,
    pub dims: &'a BTreeMap<String, u64>,
    pub spec: &'a MockSpec,
    pub routing: Option<&'a RoutingProfile>,
    pub block: Option<(u64, u64)>,
    pub calibration: Option<&'a RoutingCalibration>,
}

/// 2026-10-03: The units and their tensors.
pub(super) struct Built {
    pub tensors: Vec<OutTensor>,
    pub units: Vec<Unit>,
    pub routers: Vec<RouterFit>,
    pub bias_channel: Option<BiasChannel>,
}

/// 2026-10-03: The routing state of one build.
pub(super) struct Routing<'a> {
    pub profile: &'a RoutingProfile,
    pub channel: BiasChannel,
    pub hidden: u64,
    pub experts: u64,
    /// 2026-10-03: Source layer -> its row in the profile (MoE layers in order).
    pub ordinal: BTreeMap<usize, usize>,
}

fn dim(dims: &BTreeMap<String, u64>, d: &str) -> Result<u64> {
    dims.get(d).copied().ok_or_else(|| {
        MlError::Checkpoint(format!(
            "the arch shape has no `{d}` dim (histogram routing needs it)"
        ))
    })
}

pub(super) fn is_router(e: &TensorEntry) -> bool {
    e.name.ends_with(".mlp.gate.weight")
}

pub(super) fn routing<'a>(c: &Ctx<'a>) -> Result<Option<Routing<'a>>> {
    let profile = match (&c.spec.routing, c.routing) {
        (RoutingMode::Uniform, None) => return Ok(None),
        (RoutingMode::Histogram { .. }, Some(p)) => p,
        (RoutingMode::Uniform, Some(_)) => {
            return Err(MlError::Routing(
                "a profile was given for uniform routing".into(),
            ));
        }
        (RoutingMode::Histogram { path, .. }, None) => {
            return Err(MlError::Routing(format!(
                "the spec names {path}; it was not supplied"
            )));
        }
    };
    let (hidden, experts, top_k) = (
        dim(c.dims, "hidden")?,
        dim(c.dims, "experts")?,
        dim(c.dims, "top_k")?,
    );
    if profile.experts as u64 != experts || profile.top_k as u64 != top_k {
        return Err(MlError::Routing(format!(
            "the profile has {} experts top-{}; the checkpoint {experts} top-{top_k}",
            profile.experts, profile.top_k
        )));
    }
    let layers: BTreeSet<usize> = c
        .index
        .iter()
        .filter(|e| is_router(e))
        .filter_map(|e| c.schedule.layer_of(&e.name))
        .collect();
    if profile.layers.len() < layers.len() {
        return Err(MlError::Routing(format!(
            "the profile has {} MoE layers; the checkpoint {}",
            profile.layers.len(),
            layers.len()
        )));
    }
    Ok(Some(Routing {
        profile,
        channel: bias_channel(hidden),
        hidden,
        experts,
        ordinal: layers
            .into_iter()
            .enumerate()
            .map(|(o, l)| (l, o))
            .collect(),
    }))
}

fn out_tensor(c: &Ctx<'_>, name: &str) -> Result<OutTensor> {
    let e = c.index.get(name).ok_or_else(|| MlError::Tensor {
        name: name.to_string(),
        why: "named by a quantized group but absent".into(),
    })?;
    let mock = c.renamer.name(name).ok_or_else(|| MlError::Tensor {
        name: name.to_string(),
        why: "its weight is kept but it is not".into(),
    })?;
    Ok(OutTensor {
        source: name.to_string(),
        name: mock,
        dtype: e.dtype,
        shape: e.shape.clone(),
    })
}

/// 2026-10-03: The zero row of a residual writer under histogram routing.
pub(super) fn zero_row(
    r: &Option<Routing<'_>>,
    residual: bool,
    rows: u64,
    name: &str,
) -> Result<Option<u64>> {
    match r {
        Some(r) if residual => {
            if rows != r.hidden {
                return Err(MlError::Tensor {
                    name: name.to_string(),
                    why: format!(
                        "a residual writer with {rows} rows, not hidden {}",
                        r.hidden
                    ),
                });
            }
            Ok(Some(r.channel.channel))
        }
        _ => Ok(None),
    }
}

pub(super) fn build(c: &Ctx<'_>) -> Result<Built> {
    let groups = find_groups(c.index, c.block)?;
    let by_weight: BTreeMap<&str, &QuantGroup> =
        groups.iter().map(|g| (g.weight.as_str(), g)).collect();
    let members: BTreeSet<&str> = groups.iter().flat_map(|g| g.tensors()).collect();
    let r = routing(c)?;
    let gain_res = residual_gain(c.schedule.layer_kinds.len());
    let mut b = Built {
        tensors: Vec::new(),
        units: Vec::new(),
        routers: Vec::new(),
        bias_channel: r.as_ref().map(|r| r.channel),
    };
    for e in c.index.iter() {
        let Some(mock) = c.renamer.name(&e.name) else {
            continue;
        };
        let main = c.schedule.layer_of(&e.name);
        let module = e.name.rsplit_once('.').map_or("", |(m, _)| m);
        let residual = main.is_some() && is_residual_writer(module);
        let gain = if residual { gain_res } else { 1.0 };
        if let Some(g) = by_weight.get(e.name.as_str()) {
            let first = b.tensors.len();
            for name in g.tensors() {
                b.tensors.push(out_tensor(c, name)?);
            }
            let dt = |n: Option<&String>| n.and_then(|n| c.index.get(n)).map(|x| x.dtype);
            b.units.push(Unit::Group {
                group: (*g).clone(),
                tensors: (first..b.tensors.len()).collect(),
                class: ValueClass::Normal {
                    std: gain / (g.cols as f32).sqrt(),
                    row_len: g.cols,
                    zero_row: zero_row(&r, residual, g.rows, &e.name)?,
                },
                dtypes: GroupDtypes {
                    weight: e.dtype,
                    scale: dt(Some(&g.scale)).ok_or_else(|| MlError::Tensor {
                        name: g.scale.clone(),
                        why: "a group scale absent from the index".into(),
                    })?,
                    global: dt(g.global.as_ref()),
                    input: dt(g.input.as_ref()),
                },
            });
            continue;
        }
        if members.contains(e.name.as_str()) {
            continue;
        }
        let class = match (&r, main) {
            (Some(rt), Some(layer)) if is_router(e) => {
                router_class(c, rt, e, layer, &mock, &mut b)?
            }
            _ if e.name.ends_with("embed_tokens.weight") => ValueClass::Embedding {
                row_len: *e.shape.get(1).ok_or_else(|| MlError::Tensor {
                    name: e.name.clone(),
                    why: "an embedding of rank < 2".into(),
                })?,
                bias: r.as_ref().map(|r| (r.channel.channel, r.channel.k)),
            },
            _ => match classify_plain(e, gain)? {
                ValueClass::Normal { std, row_len, .. } => ValueClass::Normal {
                    std,
                    row_len,
                    zero_row: zero_row(&r, residual, e.shape[0], &e.name)?,
                },
                other => other,
            },
        };
        b.tensors.push(OutTensor {
            source: e.name.clone(),
            name: mock,
            dtype: e.dtype,
            shape: e.shape.clone(),
        });
        b.units.push(Unit::Plain {
            tensor: b.tensors.len() - 1,
            class,
        });
    }
    Ok(b)
}

fn router_class(
    c: &Ctx<'_>,
    r: &Routing<'_>,
    e: &TensorEntry,
    layer: usize,
    mock: &str,
    b: &mut Built,
) -> Result<ValueClass> {
    let column = fit_router(c, r, e, layer, mock, b, 1.0 / r.channel.s)?;
    Ok(ValueClass::Router {
        row_len: r.hidden,
        channel: r.channel.channel,
        sigma: r.channel.sigma,
        column,
    })
}

/// 2026-10-04: Fit router `e` of source layer `layer` to its profile row and return its bias
/// column: the unit-noise bias times `scale` times the layer's calibration gain (1 without a
/// calibration). Records the fit in `b.routers`.
pub(super) fn fit_router(
    c: &Ctx<'_>,
    r: &Routing<'_>,
    e: &TensorEntry,
    layer: usize,
    mock: &str,
    b: &mut Built,
    scale: f32,
) -> Result<Vec<f32>> {
    if e.shape != [r.experts, r.hidden] {
        return Err(MlError::Tensor {
            name: e.name.clone(),
            why: format!(
                "a router of shape {:?}, not [{}, {}]",
                e.shape, r.experts, r.hidden
            ),
        });
    }
    let row = r.ordinal[&layer];
    let stream = Stream::for_tensor(c.spec.seed, &e.name, e.dtype.name(), &e.shape).derive(1);
    let f = fit(&r.profile.layers[row], r.profile.top_k, stream)?;
    let gain = match c.calibration {
        None => 1.0,
        Some(cal) => *cal.gains.get(&layer).ok_or_else(|| {
            MlError::Routing(format!(
                "the calibration has no gain for source layer {layer}"
            ))
        })?,
    };
    let column = f.bias.iter().map(|x| x * scale * gain).collect();
    b.routers.push(RouterFit {
        tensor: mock.to_string(),
        source_layer: layer,
        profile_row: row,
        tv: f.tv,
        floored: f.floored,
        bias: f.bias,
        gain,
    });
    Ok(column)
}
