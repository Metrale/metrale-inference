// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: The recipe-fully-explicit check's interface, and the single stub
//! implementation that exists until PR #124 lands on main.
//!
//! PR #124, "recipe: require every recipe-settable serve flag to be explicit"
//! (`feat/recipes-fully-explicit`), adds `crates/server/src/recipe/explicit.rs`:
//! `required_keys()` (derived from `cli::manifest::build()`, i.e. from `ServeArgs` itself)
//! and `missing_keys(&required, &recipe)`. It is not on main as of this branch, and it
//! touches ~30 recipe files this branch has no reason to carry, so this preflight check
//! calls it through [`check`] — an interface, not a reimplementation — and reports
//! [`super::core::RecipeExplicitFacts::Unavailable`] loudly rather than silently passing.
//!
//! Owner: server CLI (`met bench preflight`).
//! Invariants:
//! - [`check`] never returns anything that `core::evaluate` would read as `Pass`: there is
//!   exactly one variant of [`super::core::RecipeExplicitFacts`] today, and it is
//!   `Unavailable`.
//!
//! **When PR #124 lands**, replace this file's body with:
//! ```ignore
//! let required = crate::recipe::explicit::required_keys();
//! let missing = crate::recipe::explicit::missing_keys(&required, &recipe);
//! if missing.is_empty() {
//!     RecipeExplicitFacts::Checked { missing: Vec::new() }
//! } else {
//!     RecipeExplicitFacts::Checked { missing: missing.into_iter().collect() }
//! }
//! ```
//! and add the `Checked` variant (and its `core::evaluate` arm) alongside `Unavailable`.

use super::core::RecipeExplicitFacts;

/// 2026-10-05: Why this check cannot run yet. Printed verbatim as the row's detail.
pub const UNAVAILABLE_REASON: &str = "PR #124 (\"recipe: require every recipe-settable serve flag to be explicit\", branch \
     feat/recipes-fully-explicit) is not yet on main — this preflight cannot verify that \
     the recipe sets every recipe-settable serve flag explicitly. Land #124, then wire its \
     recipe::explicit::{required_keys, missing_keys} into this check (see this file's doc).";

/// 2026-10-05: The interface `met bench preflight --recipe <ref>` calls. Takes the recipe
/// reference only (not a loaded [`crate::recipe::Recipe`]): the real implementation will
/// need to load it the same way `recipe::explicit_tests::load_real_recipes` does, and
/// until then there is nothing for this stub to do with it either way.
pub fn check(_recipe_ref: &str) -> RecipeExplicitFacts {
    RecipeExplicitFacts::Unavailable {
        reason: UNAVAILABLE_REASON,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-10-05: The stub always reports `Unavailable`, never `Checked` (which does not
    /// exist yet) and never anything `core::evaluate` could read as a pass.
    #[test]
    fn the_stub_always_reports_unavailable() {
        match check("qwen3.6/qwen3.6-35b-a3b-nvfp4-declared") {
            RecipeExplicitFacts::Unavailable { reason } => {
                assert!(reason.contains("#124"), "{reason}");
            }
        }
    }
}
