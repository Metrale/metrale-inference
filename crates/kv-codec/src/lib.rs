// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: The KV-cache codec registry (xPEAR over SBIO). One table row per codec
//! ([`KvCodecSpec`]) carries everything the engine needs to know about it: bits, side tensors,
//! rotation, launch-argument family ([`AbiId`]), host dequantization and citation. Byte sizes,
//! labels, the CLI spellings and the pair aliases are all derived from that table, and the
//! resolved [`KvCachePlan`] and VBR ladder are typed values built from it.
//!
//! Owner: kv-codec.
//! Invariants:
//! - `no_std` + `alloc`, no `unsafe`; builds for both wasm32 targets (NSCC).
//! - The only dependency is `metrale-error`; errors are registry codes (CERF).
//! - No wildcard arm over an enum of this crate's own, so adding a codec, side element or ABI
//!   fails to compile until every derivation handles it.

#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]
#![deny(warnings)]
#![deny(clippy::all)]
#![deny(clippy::wildcard_enum_match_arm)]
#![deny(
    clippy::cast_possible_truncation,
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

extern crate alloc;

mod contiguous_table;
mod indexer;
mod kernel_key;
mod kernel_table;
mod label;
mod pair;
mod plan;
mod registry;
mod resolve;
mod rows;
mod spec;
mod spill;

pub use contiguous_table::{contiguous_kv_kernel_names, contiguous_kv_pairs};
pub use indexer::IndexerPlan;
pub use kernel_key::{HeadDimClass, KvKernelKey, KvKernelSet, KvOp, ResolvedKvKernels};
pub use kernel_table::{KernelName, kv_kernel_names};
pub use label::KvPlanLabel;
pub use pair::{KvPair, accepted};
pub use plan::{
    KvCachePlan, KvLayoutEpoch, KvSide, KvSidePlan, KvTransition, LayerIx, LayerKvPlan,
    PressureTrigger, RungIx, VbrLadder,
};
pub use registry::{KvCodecId, PAIR_ALIASES, PairAlias, SPEC, by_name, spec};
pub use resolve::{KvPlanInputs, layer_entry};
pub use spec::{
    AbiId, AutoBoundary, BlockGeometry, CodecRole, HostLutId, KvCodecSpec, Rotation, SideElem,
    SideKind, SideTensor, TrellisSpec,
};
pub use spill::{SPILL_SALT_VERSION, SpillSalt};
