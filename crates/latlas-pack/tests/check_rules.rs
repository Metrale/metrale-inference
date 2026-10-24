// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The metrale pack's score rules on the campaign fixture: BLOCKED rows are not
//! outcomes, gated rows are; C equals the share of outcome weight whose claim is gated, and the
//! score is the one `claims_without_gate` implies.

mod common;

use latlas_check::dims::Dim;
use latlas_check::weights::weights;
use latlas_check::{CheckConfig, CheckFacts, Checks, score};
use latlas_core::NodeId;
use metrale_latlas_pack::queries::claims_without_gate;
use metrale_latlas_pack::{Metrale, derive};

const CHECK: &str = r#"
schema = "latlas-check/v1"
[weights]
churn = "off"
importance = [{ prefix = "", value = 1.0 }]
pagerank = { damping = 0.85, iterations = 30 }
[window]
days = 30
[prose]
weight = 0.5
[ratchet.epsilon]
F = 0.0
R = 0.0
C = 0.0
G = 0.0
P = 0.0
S = 0.0
H = 0.0
T = 0.0
X = 0.0
O = 0.0
"#;

#[test]
fn outcomes_are_the_rows_that_need_a_claim_and_c_matches_claims_without_gate() {
    let b = derive(&common::config(), &common::sources(), None).unwrap();
    let id = |s: &str| NodeId::parse::<Metrale>(s).unwrap();
    let node = |s: &str| b.node(&id(s)).unwrap();
    assert!(!Metrale::is_outcome(node("ledger:p0-x")), "BLOCKED is not an outcome");
    assert!(Metrale::is_outcome(node("ledger:l2-pf")));
    let cfg = CheckConfig::parse(CHECK).unwrap();
    let w = weights(&b, &cfg, &CheckFacts::default()).unwrap();
    let outcomes: Vec<_> = b.nodes.iter().filter(|n| Metrale::is_outcome(n)).collect();
    let ungated: Vec<NodeId> = claims_without_gate(&b).into_iter().map(|r| r.0).collect();
    let all: f64 = outcomes.iter().map(|n| w[&n.id]).sum();
    let gated: f64 = outcomes
        .iter()
        .filter(|n| !ungated.contains(&n.id))
        .map(|n| w[&n.id] * Metrale::claim_credit(&b, &id(&format!("claim:{}/1", n.id.key()))))
        .sum();
    let s = score::<Metrale>(&b, None, &cfg, &CheckFacts::default()).unwrap();
    let c = s.get(Dim::C).unwrap().value.unwrap();
    assert!((c - gated / all).abs() < 1e-12, "C {c} vs {}", gated / all);
}
