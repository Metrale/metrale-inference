// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Exercise the production routing checks without initializing a GPU.
mod layers {
    pub use metrale_model_layers::layers::ops;
}
#[path = "../src/layers/moe/init_validation.rs"]
mod validation;
