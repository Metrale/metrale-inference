// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The deployment a serve over N devices runs, derived from the circuit's one
//! `[parallel]` spec: the engine's tensor parallelism splits every declared part over all N
//! ranks (`--tp-size N`; a part the engine does not split is not declared), so each part runs at
//! degree N, and N = 1 is no deployment (the whole model). A part that cannot run at N, by its
//! degree bounds (a kernel family's limit), unit or remainder policy, refuses N with
//! `ErrorCode::CircuitRanksRefused` naming the part and the dim.
//!
//! Owner: metrale-circuit.
//! Invariants:
//! - One resolver: the executor and the checks derive the deployment here, for any N; an
//!   instance's stated `[instance.parallel]` is the recipe's record of the same derivation at its
//!   N (tested equal), not a second source.
//! - Nothing is defaulted: every part is placed by [`crate::parallel::place`], which checks it.

use crate::prelude::*;

use alloc::collections::BTreeMap;

use metrale_error::{Error, ErrorCode, error};

use crate::instances::Deployment;
use crate::parallel::{self, Chosen, ParallelError, ParallelSpec};

/// 2026-10-23: Where the derived deployment comes from, as its `why`.
pub const WHY: &str = "the serve's --tp-size N: the engine splits every declared part over all N ranks (crates/circuit/src/deploy.rs)";

/// 2026-10-23: The deployment over `ranks` devices of a circuit whose spec is `spec` at `dims`:
/// `None` at one device, else every part at degree `ranks`, placed and checked.
pub fn tensor_parallel(
    spec: &ParallelSpec,
    dims: &BTreeMap<String, u64>,
    ranks: u64,
) -> Result<Option<Deployment>, Error> {
    if ranks == 1 {
        return Ok(None);
    }
    let degree: BTreeMap<String, u64> = parallel::parts(spec).map(|p| (p.clone(), ranks)).collect();
    parallel::place(spec, dims, ranks, &degree, Chosen::Instance)
        .map_err(|e| refused(spec, ranks, e))?;
    Ok(Some(Deployment {
        ranks,
        degree,
        why: WHY.into(),
    }))
}

fn refused(spec: &ParallelSpec, ranks: u64, e: ParallelError) -> Error {
    let (dim, why) = match e {
        ParallelError::Split { dim, detail, .. } => (dim, detail),
        ParallelError::UnknownDim(dim) => (dim, "the shape does not define it".into()),
        ParallelError::Rank { rank, ranks } => {
            (String::from("-"), format!("rank {rank} of {ranks}"))
        }
    };
    let part = parallel::part_of(spec, &dim).to_string();
    error!(ErrorCode::CircuitRanksRefused, ranks, part, dim, why)
}

/// 2026-10-23: `instance` over `ranks` devices: its `parallel` replaced by the derived deployment
/// of the circuit in `src` ([`tensor_parallel`]). The one resolver the executor and the checks
/// load a deployment from.
pub fn deployed(
    instance: &crate::Instance,
    src: crate::Sources<'_>,
    ranks: u64,
) -> Result<crate::Instance, Error> {
    let spec = crate::instantiate::parallel_spec(src.circuit, src.blocks).map_err(|e| {
        let path = format!("kernels/circuits/{}.toml", instance.arch);
        error!(ErrorCode::CircuitSource, path, e.to_string())
    })?;
    let mut out = instance.clone();
    out.parallel = tensor_parallel(&spec, &instance.shape.dims, ranks)?;
    Ok(out)
}
