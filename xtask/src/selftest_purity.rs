// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The purity gates' selftest. Each gate runs on its fixture workspace under
//! `xtask/fixtures/purity-*`; it must report exactly the listed problems (kind and key), each
//! for its named reason, and nothing else. A fixture breaks its gate in several shapes and also
//! holds offenders the allowlist excuses and clean code that must pass, so a gate that misses a
//! shape, refuses clean code or ignores its allowlist fails here. The gates that run cargo get a
//! throwaway copy under `target/` so cargo never writes into the source tree.

use crate::allow::Kind::{self, Grown, Refused, Shrunk, Stale};
use crate::allow::Problem;
use crate::gates::{self, Gate};
use crate::purity_io;
use crate::shell;
use std::path::Path;

/// 2026-10-17: A problem the gate must report: its kind, key and a phrase from its reason.
struct Want {
    kind: Kind,
    key: &'static str,
    reason: &'static str,
}

const fn want(kind: Kind, key: &'static str, reason: &'static str) -> Want {
    Want { kind, key, reason }
}

struct Probe {
    fixture: &'static str,
    gate: Gate,
    wants: &'static [Want],
}

const PROBES: [Probe; 7] = [
    Probe {
        fixture: "purity-roles",
        gate: Gate::Roles,
        wants: &[
            want(
                Refused,
                "r-none",
                "declares no [package.metadata.metrale] role",
            ),
            want(Refused, "r-bad", "role `kore` is not one of"),
            want(Stale, "r-stale", "now declares role `device`"),
            want(Refused, "r-mismatch", "declares role `core` but"),
            want(Stale, "r-gone", "is not a workspace member"),
        ],
    },
    Probe {
        fixture: "purity-filesize",
        gate: Gate::FileSize,
        wants: &[
            want(Refused, "big.rs", "501 lines, over the 500-line cap"),
            want(Grown, "grown.cu", "grew from 502 to 503"),
            want(
                Shrunk,
                "shrunk.py",
                "tighten xtask/file-size-allow.txt to `shrunk.py 501`",
            ),
            want(Stale, "gone.rs", "no longer offends"),
        ],
    },
    Probe {
        fixture: "purity-vendor",
        gate: Gate::VendorBan,
        wants: &[
            want(
                Refused,
                "crates/v-bin/src/main.rs",
                "2 GPU vendor identifier(s) in the `bin` crate `v-bin` (first: line 3 `cudarc`)",
            ),
            want(Grown, "crates/v-core/src/grown.rs", "grew from 1 to 2"),
            want(Stale, "crates/v-core/src/gone.rs", "no longer offends"),
        ],
    },
    Probe {
        fixture: "purity-kvcodec",
        gate: Gate::KvCodecBan,
        wants: &[
            want(
                Refused,
                "crates/k-app/src/main.rs",
                "6 KV codec decision(s) outside the registry (first: line 3 `KvCacheDtype`)",
            ),
            want(Grown, "crates/k-app/src/grown.rs", "grew from 1 to 2"),
            want(Stale, "crates/k-app/src/gone.rs", "no longer offends"),
        ],
    },
    Probe {
        fixture: "purity-lints",
        gate: Gate::Lints,
        wants: &[
            want(Refused, "l-sign", "#![deny(clippy::cast_sign_loss)]"),
            want(
                Refused,
                "l-comment",
                "#![deny(clippy::cast_possible_truncation)]",
            ),
            want(
                Refused,
                "l-warn",
                "clippy::cast_possible_truncation, clippy::cast_lossless, clippy::cast_possible_wrap, clippy::cast_sign_loss",
            ),
            want(Stale, "l-stale", "no longer offends"),
        ],
    },
    Probe {
        fixture: "purity-depgraph",
        gate: Gate::Depgraph,
        wants: &[
            want(
                Refused,
                "core-a->tokio",
                "`tokio`, which is on PURE_DENYLIST (core-a -> tokio",
            ),
            want(
                Refused,
                "core-b->dev-x",
                "may not depend on the `device` crate `dev-x`",
            ),
            want(
                Refused,
                "core-b->hstd",
                "may not dev-depend on the `host-std` crate `hstd`",
            ),
            want(
                Refused,
                "core-c->leftpad",
                "not on PURE_ALLOWLIST (core-c -> serde -> leftpad",
            ),
            want(
                Refused,
                "core-d->rayon",
                "which is on PURE_DENYLIST (core-d -> rayon",
            ),
            want(
                Refused,
                "dev-x->hstd",
                "a `device` crate may not depend on the `host-std`",
            ),
            want(
                Refused,
                "app->cudarc",
                "VBOP: only `device` crates name the GPU vendor crate",
            ),
            want(
                Refused,
                "metrale-error->cfg-if",
                "is a leaf and may have no normal or build dependency; `cfg-if` is a normal",
            ),
            want(
                Refused,
                "metrale-error->autocfg",
                "is a leaf and may have no normal or build dependency; `autocfg` is a build",
            ),
            want(Stale, "core-c->rayon", "no longer offends"),
        ],
    },
    Probe {
        fixture: "purity-wasm",
        gate: Gate::Wasm,
        wants: &[
            want(Refused, "w-std:wasm32v1-none", "can't find crate for `std`"),
            want(Refused, "w-std:wasm32-unknown-unknown", "unix"),
            want(Stale, "w-stale:wasm32v1-none", "no longer offends"),
        ],
    },
];

/// 2026-10-17: Gates that run cargo on their fixture, which therefore runs on a copy.
fn runs_cargo(gate: Gate) -> bool {
    matches!(gate, Gate::Depgraph | Gate::Wasm)
}

/// 2026-10-17: Run the probes (the wasm one only when `with_builds`: it builds ten crates);
/// returns how many did not end as intended.
pub fn run(fixtures: &Path, with_builds: bool) -> Result<usize, String> {
    let scratch = shell::repo_root()?.join("target").join("xtask-selftest");
    let mut failures: usize = 0;
    for probe in PROBES
        .iter()
        .filter(|p| with_builds || p.gate != Gate::Wasm)
    {
        let source = fixtures.join(probe.fixture);
        let root = if runs_cargo(probe.gate) {
            let copy = scratch.join("ws").join(probe.fixture);
            purity_io::copy_tree(&source, &copy)?;
            copy
        } else {
            source
        };
        let target_dir = scratch.join("target");
        let outcome = gates::run(probe.gate, &root, Some(&target_dir))?;
        let (ok, why) = judge(probe, &outcome.problems);
        let mark = if ok { "PASS" } else { "FAIL" };
        println!(
            "[{mark}] selftest {} ({}): {why}",
            probe.fixture,
            probe.gate.name()
        );
        if !ok {
            failures = failures.saturating_add(1);
        }
    }
    Ok(failures)
}

fn judge(probe: &Probe, got: &[Problem]) -> (bool, String) {
    let listing: String = got.iter().map(|p| format!("\n    {p}")).collect();
    let mut want: Vec<(Kind, &str)> = probe.wants.iter().map(|w| (w.kind, w.key)).collect();
    let mut have: Vec<(Kind, &str)> = got.iter().map(|p| (p.kind, p.key.as_str())).collect();
    want.sort_unstable();
    have.sort_unstable();
    if want != have {
        let expected: String = probe
            .wants
            .iter()
            .map(|w| format!("\n    [{}] {}", w.kind, w.key))
            .collect();
        return (
            false,
            format!("expected exactly:{expected}\n  got:{listing}"),
        );
    }
    for w in probe.wants {
        let named = got
            .iter()
            .any(|p| p.kind == w.kind && p.key == w.key && p.detail.contains(w.reason));
        if !named {
            return (
                false,
                format!(
                    "[{}] {} was not reported for `{}`:{listing}",
                    w.kind, w.key, w.reason
                ),
            );
        }
    }
    (
        true,
        format!(
            "{} problem(s), each for its named reason{listing}",
            got.len()
        ),
    )
}

#[cfg(test)]
mod tests {
    // 2026-10-17: The probes `cargo xtask selftest` runs, minus the wasm one (it builds ten
    // crates for two targets; CI's purity job runs it through `cargo xtask selftest`).
    #[test]
    fn every_purity_fixture_is_judged_for_its_reason() {
        assert_eq!(super::run(&crate::shell::fixtures_dir(), false), Ok(0));
    }
}
