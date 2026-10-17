// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The `METRALE_*` environment a gate's server is measured under.
//!
//! A serve lever is a `METRALE_*` variable that is not a harness variable.
//! Levers are not recipe keys, so they do not appear in the argv a leased
//! server is fingerprinted by. Instead the recipe declares the levers it is
//! measured under (its `env:` block, with the gate's BENCH.toml
//! `[benchmarks.serve_env]` pin on top), and [`reconcile`] refuses a lever in
//! the harness's own environment that nothing declared, and a declared lever
//! it carries at another value. The set is fingerprinted into the serve
//! identity (`identity`) and disclosed on the gate record
//! (`gate::GateRecord::serve_env`).
//!
//! It lives in this crate so both ends share one definition: the server
//! fingerprints its own environment for `GET /serve-config`, and the harness
//! computes what that fingerprint must be.
//!
//! 2026-10-17: Moved from metrale-bench (`serve_env`). Reading this process's
//! environment moved to `metrale-host-probe::serve_identity::process_levers`;
//! refusals are registry errors (`ErrorCode::ServeEnv*`), whose `Display` is the text `bail!`
//! printed.
//!
//! Owner: serve identity (serve environment).
//! Invariants:
//! - [`declared`] returns only `Runtime`-class levers with non-empty values.
//! - [`reconcile`] succeeds only when every present lever is declared at the
//!   declared value; its `env` is then the declaration, whole.

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use metrale_error::{Error, ErrorCode, error};

use metrale_settings::groups::lookup;
use metrale_settings::spec::Class;
use sha2::{Digest, Sha256};

/// 2026-09-26: The prefix every lever carries.
pub use metrale_settings::spec::PREFIX;

/// 2026-09-26: Whether `name` is a serve lever: any `METRALE_*` name that is
/// not a harness variable.
///
/// The harness variables are the table's `Class::Harness` entries
/// (`metrale_settings::groups`): they place, log or build for the harness.
/// Every other name under the prefix counts, including one the table does
/// not declare, so the harness names it in a refusal; `met` itself refuses
/// to start with an undeclared one set (`levers::check` in `main.rs`).
#[must_use]
pub fn is_lever(name: &str) -> bool {
    name.starts_with(PREFIX) && !lookup(name).is_some_and(|l| l.class == Class::Harness)
}

/// 2026-09-26: The levers in an environment, from any `(name, value)`
/// iterator.
pub fn levers<I, K, V>(env: I) -> BTreeMap<String, String>
where
    I: IntoIterator<Item = (K, V)>,
    K: AsRef<str>,
    V: AsRef<str>,
{
    env.into_iter()
        .filter(|(k, _)| is_lever(k.as_ref()))
        .map(|(k, v)| (k.as_ref().to_string(), v.as_ref().to_string()))
        .collect()
}

/// 2026-09-26: The SHA-256 of a lever set: `name=value` in key order, each
/// NUL-terminated, so `A=1 B=2` and `A=1B=2` differ and the empty set has one
/// fixed digest.
#[must_use]
pub fn fingerprint(levers: &BTreeMap<String, String>) -> String {
    let mut h = Sha256::new();
    for (k, v) in levers {
        h.update(k.as_bytes());
        h.update(b"=");
        h.update(v.as_bytes());
        h.update([0u8]);
    }
    crate::hex_lower(&h.finalize())
}

/// 2026-09-26: Validate one declaration (a recipe's `env:` block or a
/// BENCH.toml `[benchmarks.serve_env]` table) into the lever set it means.
///
/// `owner` names the declaring file in every refusal.
///
/// # Errors
/// A key that is not a `Runtime` lever (no prefix; a harness, build, dev or
/// tool variable; or a name the lever table does not declare), or a value
/// that is empty after trimming (how a shell spells "unset").
pub fn declared(
    owner: &str,
    raw: &BTreeMap<String, String>,
) -> Result<BTreeMap<String, String>, Error> {
    let mut out = BTreeMap::new();
    for (name, value) in raw {
        let (owner, name) = (owner.to_string(), name.clone());
        if !name.starts_with(PREFIX) {
            return Err(error!(ErrorCode::ServeEnvNotALever, owner, name));
        }
        match lookup(&name).map(|l| l.class) {
            Some(Class::Runtime) => {}
            Some(Class::Harness) => {
                return Err(error!(
                    ErrorCode::ServeEnvHarness,
                    owner,
                    name.clone(),
                    name
                ));
            }
            Some(class) => {
                return Err(error!(
                    ErrorCode::ServeEnvNotRead,
                    owner,
                    name,
                    class.as_str()
                ));
            }
            None => return Err(error!(ErrorCode::ServeEnvUndeclared, owner, name)),
        }
        if value.trim().is_empty() {
            return Err(error!(ErrorCode::ServeEnvEmpty, owner, name));
        }
        out.insert(name, value.clone());
    }
    Ok(out)
}

/// 2026-09-26: The recipe's declaration under the gate entry's pin; the pin
/// wins on a clash.
#[must_use]
pub fn merge_declared(
    recipe: BTreeMap<String, String>,
    baseline: BTreeMap<String, String>,
) -> BTreeMap<String, String> {
    let mut out = recipe;
    out.extend(baseline);
    out
}

/// 2026-09-26: What the harness must do to serve `declared` from an
/// environment that already holds `present`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reconciled {
    /// 2026-09-26: The lever set the server runs under: the declaration,
    /// whole.
    pub env: BTreeMap<String, String>,
    /// 2026-09-26: The declared levers the harness does not already carry:
    /// what a child serve is given on top of the inherited environment, and
    /// what an in-process serve refuses to start without.
    pub missing: BTreeMap<String, String>,
}

/// 2026-09-26: Reconcile the recipe's declaration with the harness's own
/// levers (`metrale-host-probe::serve_identity::process_levers`).
///
/// # Errors
/// A present lever that `declared` does not name: refused, not inherited and
/// not stripped, because either would change what the gate measures without
/// a trace. A declared lever present at another value: refused, because
/// neither side may win silently.
pub fn reconcile(
    owner: &str,
    declared: &BTreeMap<String, String>,
    present: &BTreeMap<String, String>,
) -> Result<Reconciled, Error> {
    let undeclared: Vec<String> = present
        .iter()
        .filter(|(k, _)| !declared.contains_key(*k))
        .map(|(k, v)| alloc::format!("{k}={v}"))
        .collect();
    if !undeclared.is_empty() {
        return Err(error!(
            ErrorCode::ServeEnvInherited,
            undeclared.len(),
            owner.to_string(),
            undeclared.join(", ")
        ));
    }
    let mut missing = BTreeMap::new();
    let mut contradicted = Vec::new();
    for (k, want) in declared {
        match present.get(k) {
            Some(got) if got == want => {}
            Some(got) => contradicted.push(alloc::format!(
                "{k}: harness {got:?}, {owner} declares {want:?}"
            )),
            None => {
                missing.insert(k.clone(), want.clone());
            }
        }
    }
    if !contradicted.is_empty() {
        return Err(error!(
            ErrorCode::ServeEnvContradicted,
            owner.to_string(),
            contradicted.len(),
            contradicted.join("; ")
        ));
    }
    Ok(Reconciled {
        env: declared.clone(),
        missing,
    })
}

#[cfg(test)]
#[path = "env_tests.rs"]
mod tests;
