// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Isolate launch admission tests from unrelated CUDA-only unit tests.
#[allow(dead_code)]
#[path = "../src/layers/ops/dense_batchm_fp32.rs"]
mod implementation;
