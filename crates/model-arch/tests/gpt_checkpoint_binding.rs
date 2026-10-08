// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Run metadata binding controls without unrelated CUDA-only unit modules.
use metrale_config::ModelConfig;
use metrale_model_arch::weight_loader::gpt_oss::*;
#[path = "../src/weight_loader/gpt_oss/tests.rs"]
mod binding;
