// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-06: Device-free receipts for investor-mvp #44 and #58.
//!
//! The chat handler and the `qci-dump` binary both call [`gate`] and [`render_dump`].
//! Neither function reads a weight file or a GPU. A missing measurement is a named
//! blocker in the text, and the word that would claim a model is served is never written.
//!
//! Owner: qci-dump.
//! Invariants:
//! - [`render_dump`] is a pure function of its arguments. Two calls with the same
//!   arguments return the same string.
//! - [`gate`] returns [`Gate::Pass`] for every model that is not one of the two pinned
//!   campaigns, so other chat traffic keeps the existing handler.

mod admit;
mod dump;
mod recipe;

pub use dump::DumpInput;

/// 2026-10-06: What the chat handler does with a request after JSON has been read.
#[derive(Debug, Clone, PartialEq)]
pub enum Gate {
    /// The model is not a QCI campaign id. The handler continues.
    Pass,
    /// An HTTP status and a JSON body. Errors and admissions both use this.
    Respond {
        status: u16,
        body: serde_json::Value,
    },
}

/// 2026-10-06: Campaign admission for `model`, or [`Gate::Pass`] when `model` is not one.
///
/// `body` is the raw chat-completions JSON. A campaign model never falls through:
/// a body this function cannot admit becomes an API error.
pub fn gate(model: &str, body: &[u8]) -> Gate {
    let Some(recipe) = recipe::for_model(model) else {
        return Gate::Pass;
    };
    match admit::admit(&recipe, model, body) {
        Ok(body) => Gate::Respond { status: 200, body },
        Err(err) => Gate::Respond {
            status: err.status,
            body: err.json(),
        },
    }
}

/// 2026-10-06: The dump text for one case and one topology.
pub fn render_dump(input: &DumpInput<'_>) -> String {
    dump::render(input)
}

/// 2026-10-06: Whether `model` is one of the two pinned campaign ids.
pub fn is_campaign_model(model: &str) -> bool {
    recipe::for_model(model).is_some()
}

/// 2026-10-06: The recipe `model` for a campaign id, when the id is one of the two.
pub fn model_for_case(case_id: &str) -> Option<String> {
    recipe::for_case(case_id).map(|recipe| recipe.model)
}
