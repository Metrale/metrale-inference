// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: Typed ids (TIDE): `<label-prefix>:<key>` strings stop at [`KgId::parse`]; inside,
//! every node is a [`KgId`] variant holding its label's newtype. [`Label`] and [`Rel`] are the
//! ontology's fixed vocabularies (DESIGN-kernel-knowledge-graph-ontology.md §2, §3); their order
//! is their LatticeDB label bit / relation id, so it is append-only.
//!
//! Owner: metrale-kg-derive.
//! Invariants:
//! - A key is non-empty and holds no newline or tab.
//! - `Ord` on [`KgId`] is the order of its displayed string (prefixes are lowercase letters and
//!   `:` sorts below them, so (prefix, key) order and string order agree).

use crate::prelude::*;
use core::cmp::Ordering;
use core::fmt;
use metrale_error::{ErrorCode, Result, error};

macro_rules! labels {
    ($( $(#[$m:meta])* $label:ident ($newtype:ident) = $prefix:literal; )*) => {
        /// 2026-10-23: A node label; the discriminant is its LatticeDB label bit.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub enum Label { $( $(#[$m])* $label, )* }

        impl Label {
            /// 2026-10-23: Every label, in bit order.
            pub const ALL: &'static [Label] = &[$(Label::$label),*];

            /// 2026-10-23: The label's name as Cypher and the bundle spell it.
            pub const fn name(self) -> &'static str {
                match self { $(Label::$label => stringify!($label),)* }
            }

            /// 2026-10-23: The id prefix (`ledger` in `ledger:l2-prefetch-6`).
            pub const fn prefix(self) -> &'static str {
                match self { $(Label::$label => $prefix,)* }
            }

            /// 2026-10-23: The label whose id prefix is `p`.
            pub fn from_prefix(p: &str) -> Option<Label> {
                Self::ALL.iter().copied().find(|l| l.prefix() == p)
            }

            /// 2026-10-23: The label named `n` (as [`Label::name`] spells it).
            pub fn from_name(n: &str) -> Option<Label> {
                Self::ALL.iter().copied().find(|l| l.name() == n)
            }
        }

        $(
            /// 2026-10-23: The key of a node of the label of the same name.
            #[derive(Clone, Debug, PartialEq, Eq, Hash)]
            #[repr(transparent)]
            pub struct $newtype(pub String);
        )*

        /// 2026-10-23: A node id: its label's newtype around the key.
        #[derive(Clone, Debug, PartialEq, Eq, Hash)]
        pub enum KgId { $( $label($newtype), )* }

        impl KgId {
            /// 2026-10-23: The node's label.
            pub fn label(&self) -> Label {
                match self { $(KgId::$label(_) => Label::$label,)* }
            }

            /// 2026-10-23: The key after the prefix.
            pub fn key(&self) -> &str {
                match self { $(KgId::$label(k) => k.0.as_str(),)* }
            }

            fn wrap(label: Label, key: String) -> KgId {
                match label { $(Label::$label => KgId::$label($newtype(key)),)* }
            }
        }
    };
}

labels! {
    // core
    Evidence(EvidenceId) = "ev";
    Run(RunId) = "run";
    Claim(ClaimId) = "claim";
    Gate(GateId) = "gate";
    Decision(DecisionId) = "decision";
    Principle(PrincipleId) = "principle";
    /// 2026-10-23: A feedback memory (a behavioural rule).
    Rule(FeedbackId) = "feedback";
    Memory(MemoryId) = "memory";
    Tool(ToolId) = "tool";
    Friction(FrictionId) = "friction";
    Idea(IdeaId) = "idea";
    Paper(PaperId) = "paper";
    Design(DesignId) = "design";
    Commit(CommitId) = "commit";
    /// 2026-10-23: A project config file (owner requirement 2026-10-23: the project layer of the
    /// settings ingests as a node); key = its root-relative path.
    Config(ConfigId) = "config";
    /// 2026-10-23: A reference no source defines; key `<label>:<key>` of what was referenced.
    Unresolved(UnresolvedId) = "unresolved";
    // metrale
    Model(ModelId) = "model";
    Instance(InstanceId) = "instance";
    Hardware(HardwareId) = "hw";
    Box(BoxId) = "box";
    Op(OpId) = "op";
    ParallelDim(ParallelDimId) = "pdim";
    KernelFamily(FamilyId) = "family";
    FamilyParam(FamilyParamId) = "fparam";
    FamilyPoint(FamilyPointId) = "point";
    Kernel(KernelId) = "kernel";
    FusionRule(RuleId) = "rule";
    RuntimeRoute(RouteId) = "route";
    CriticalPair(CriticalPairId) = "cpair";
    Plan(PlanId) = "plan";
    Measurement(MeasurementId) = "m";
    HeadroomSite(HeadroomSiteId) = "hsite";
    Profile(ProfileId) = "profile";
    Lever(LeverId) = "lever";
    LedgerEntry(LedgerId) = "ledger";
}

/// 2026-10-23: The most labels one LatticeDB collection holds (a `u64` label bitmap).
pub const LABEL_LIMIT: usize = 64;

impl Label {
    /// 2026-10-23: Every id prefix, comma-separated (for refusals that must not dead-end).
    pub fn prefixes() -> String {
        let all: Vec<&str> = Label::ALL.iter().map(|l| l.prefix()).collect();
        all.join(", ")
    }
}

impl KgId {
    /// 2026-10-23: The id of `label` with `key`; refuses an empty key, a tab or a newline.
    pub fn new(label: Label, key: impl Into<String>) -> Result<KgId> {
        let key = key.into();
        if key.is_empty() || key.contains(['\n', '\t', '\r']) {
            return Err(error!(
                ErrorCode::KgBadId,
                format!("{}:{key}", label.prefix()),
                "the key is empty or holds a tab or newline"
            ));
        }
        Ok(KgId::wrap(label, key))
    }

    /// 2026-10-23: Parse `<prefix>:<key>` (the wire form; TIDE: strings stop here).
    pub fn parse(s: &str) -> Result<KgId> {
        let Some((p, k)) = s.split_once(':') else {
            return Err(error!(
                ErrorCode::KgBadId,
                s.to_string(),
                format!("no `<prefix>:` part; prefixes: {}", Label::prefixes())
            ));
        };
        let Some(label) = Label::from_prefix(p) else {
            return Err(error!(
                ErrorCode::KgBadId,
                s.to_string(),
                format!(
                    "unknown label prefix `{p}`; prefixes: {}",
                    Label::prefixes()
                )
            ));
        };
        KgId::new(label, k)
    }

    /// 2026-10-23: The `unresolved:` id standing for a dangling reference to `target`.
    pub fn unresolved(target: &KgId) -> KgId {
        KgId::wrap(Label::Unresolved, target.to_string())
    }
}

impl fmt::Display for KgId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.label().prefix(), self.key())
    }
}

impl Ord for KgId {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.label().prefix(), self.key()).cmp(&(other.label().prefix(), other.key()))
    }
}

impl PartialOrd for KgId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

macro_rules! rels {
    ($( $rel:ident = $name:literal; )*) => {
        /// 2026-10-23: An edge relation; the discriminant is its LatticeDB relation id.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub enum Rel { $( $rel, )* }

        impl Rel {
            /// 2026-10-23: Every relation, in id order.
            pub const ALL: &'static [Rel] = &[$(Rel::$rel),*];

            /// 2026-10-23: The relation's name as Cypher spells it.
            pub const fn name(self) -> &'static str {
                match self { $(Rel::$rel => $name,)* }
            }

            /// 2026-10-23: The relation named `n`.
            pub fn from_name(n: &str) -> Option<Rel> {
                Self::ALL.iter().copied().find(|r| r.name() == n)
            }
        }
    };
}

rels! {
    // core
    DependsOn = "DEPENDS_ON";
    Retracts = "RETRACTS";
    Claims = "CLAIMS";
    GatedBy = "GATED_BY";
    ImplementedBy = "IMPLEMENTED_BY";
    MeasuredIn = "MEASURED_IN";
    Cites = "CITES";
    Measures = "MEASURES";
    EvidenceFor = "EVIDENCE_FOR";
    DecidedBy = "DECIDED_BY";
    Supersedes = "SUPERSEDES";
    Touches = "TOUCHES";
    Applies = "APPLIES";
    Violates = "VIOLATES";
    Proposes = "PROPOSES";
    ResolvedBy = "RESOLVED_BY";
    Recurs = "RECURS";
    Links = "LINKS";
    LandsAs = "LANDS_AS";
    RestsOn = "RESTS_ON";
    StatedIn = "STATED_IN";
    // metrale
    BlockedBy = "BLOCKED_BY";
    Pins = "PINS";
    UsesProfile = "USES_PROFILE";
    Toggles = "TOGGLES";
    PredictedBy = "PREDICTED_BY";
    Instantiates = "INSTANTIATES";
    HasParam = "HAS_PARAM";
    CoversOp = "COVERS_OP";
    LowersTo = "LOWERS_TO";
    EmitsVia = "EMITS_VIA";
    GuardedBy = "GUARDED_BY";
    MicrotestedBy = "MICROTESTED_BY";
    Overlaps = "OVERLAPS";
    UsesRule = "USES_RULE";
    Launches = "LAUNCHES";
    Plans = "PLANS";
    Targets = "TARGETS";
    ForModel = "FOR_MODEL";
    KeyedTo = "KEYED_TO";
    AtCommit = "AT_COMMIT";
    SiteOf = "SITE_OF";
    RunsKernel = "RUNS_KERNEL";
    Splits = "SPLITS";
    Allowlisted = "ALLOWLISTED";
}

impl Rel {
    /// 2026-10-23: The LatticeDB relation id (the position in [`Rel::ALL`]).
    pub const fn id(self) -> u16 {
        self as u16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_round_trip_through_their_wire_form() {
        for s in [
            "ledger:l2-prefetch-6",
            "m:glm::x @ decode",
            "unresolved:ledger:nope",
        ] {
            assert_eq!(KgId::parse(s).unwrap().to_string(), s);
        }
        assert_eq!(KgId::parse("feedback:x").unwrap().label(), Label::Rule);
    }

    #[test]
    fn malformed_ids_are_refused_with_the_bad_id_code() {
        for s in ["nolabel", "bogus:x", "ledger:", "ledger:a\tb"] {
            assert_eq!(
                KgId::parse(s).unwrap_err().code(),
                ErrorCode::KgBadId,
                "{s}"
            );
        }
    }

    #[test]
    fn id_order_is_string_order() {
        let mut ids: Vec<KgId> = ["memory:a", "m:z", "model:b", "m:a"]
            .iter()
            .map(|s| KgId::parse(s).unwrap())
            .collect();
        ids.sort();
        let got: Vec<String> = ids.iter().map(ToString::to_string).collect();
        let mut want = got.clone();
        want.sort();
        assert_eq!(got, want);
    }

    #[test]
    fn labels_fit_one_collection_and_prefixes_are_unique() {
        assert!(Label::ALL.len() <= LABEL_LIMIT);
        let p: BTreeSet<&str> = Label::ALL.iter().map(|l| l.prefix()).collect();
        assert_eq!(p.len(), Label::ALL.len());
        let r: BTreeSet<&str> = Rel::ALL.iter().map(|r| r.name()).collect();
        assert_eq!(r.len(), Rel::ALL.len());
    }
}
