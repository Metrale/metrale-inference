// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The metrale pack's vocabulary (DESIGN-latlas-ontology.md §4.3): the 19 labels and
//! 24 relations it appends to the core's, plus the core's, under one name each ([`labels`],
//! [`rels`]) so an adapter writes `labels::LedgerEntry` and `labels::Claim` alike. The order of
//! [`PACK_LABELS`] and [`PACK_RELS`] is the LatticeDB label bit / relation id after the core's:
//! append-only.
//!
//! Owner: metrale-latlas-pack.
//! Invariants: no label or relation repeats a core name or prefix (`id::vocab_is_valid`, tested).

pub use latlas_core::id::{Label, NodeId, Rel};

/// 2026-10-23: Every label an adapter of this pack names.
pub mod labels {
    pub use latlas_core::id::core_labels::*;
    latlas_core::labels! {
        pub Model = "model";
        pub Instance = "instance";
        pub Hardware = "hw", implicit;
        pub Box = "box", implicit;
        pub Op = "op", implicit;
        pub ParallelDim = "pdim";
        pub KernelFamily = "family";
        pub FamilyParam = "fparam";
        pub FamilyPoint = "point";
        pub Kernel = "kernel", implicit;
        pub FusionRule = "rule";
        pub RuntimeRoute = "route";
        pub CriticalPair = "cpair";
        pub Plan = "plan";
        pub Measurement = "m";
        pub HeadroomSite = "hsite";
        pub Profile = "profile", implicit;
        pub Lever = "lever", implicit;
        pub LedgerEntry = "ledger";
    }
}

/// 2026-10-23: Every relation an adapter of this pack names.
pub mod rels {
    pub use latlas_core::id::core_rels::*;
    latlas_core::rels! {
        pub BlockedBy = "BLOCKED_BY";
        pub Pins = "PINS";
        pub UsesProfile = "USES_PROFILE";
        pub Toggles = "TOGGLES";
        pub PredictedBy = "PREDICTED_BY";
        pub Instantiates = "INSTANTIATES";
        pub HasParam = "HAS_PARAM";
        pub CoversOp = "COVERS_OP";
        pub LowersTo = "LOWERS_TO";
        pub EmitsVia = "EMITS_VIA";
        pub GuardedBy = "GUARDED_BY";
        pub MicrotestedBy = "MICROTESTED_BY";
        pub Overlaps = "OVERLAPS";
        pub UsesRule = "USES_RULE";
        pub Launches = "LAUNCHES";
        pub Plans = "PLANS";
        pub Targets = "TARGETS";
        pub ForModel = "FOR_MODEL";
        pub KeyedTo = "KEYED_TO";
        pub AtCommit = "AT_COMMIT";
        pub SiteOf = "SITE_OF";
        pub RunsKernel = "RUNS_KERNEL";
        pub Splits = "SPLITS";
        pub Allowlisted = "ALLOWLISTED";
    }
}

/// 2026-10-23: The pack's labels in bit order after the core's.
pub const PACK_LABELS: &[Label] = {
    use labels::*;
    &[
        Model,
        Instance,
        Hardware,
        Box,
        Op,
        ParallelDim,
        KernelFamily,
        FamilyParam,
        FamilyPoint,
        Kernel,
        FusionRule,
        RuntimeRoute,
        CriticalPair,
        Plan,
        Measurement,
        HeadroomSite,
        Profile,
        Lever,
        LedgerEntry,
    ]
};

/// 2026-10-23: The pack's relations in id order after the core's.
pub const PACK_RELS: &[Rel] = {
    use rels::*;
    &[
        BlockedBy,
        Pins,
        UsesProfile,
        Toggles,
        PredictedBy,
        Instantiates,
        HasParam,
        CoversOp,
        LowersTo,
        EmitsVia,
        GuardedBy,
        MicrotestedBy,
        Overlaps,
        UsesRule,
        Launches,
        Plans,
        Targets,
        ForModel,
        KeyedTo,
        AtCommit,
        SiteOf,
        RunsKernel,
        Splits,
        Allowlisted,
    ]
};
