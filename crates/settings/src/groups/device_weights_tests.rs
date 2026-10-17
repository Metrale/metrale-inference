// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `WeightsSettings` properties that need no old reader. The equivalence with the
//! readers it replaces, the RDMA fallback chains included, is proven where those readers live,
//! against their `legacy_env_*` form (model-weights `legacy_env_tests.rs`).
//!
//! Owner: settings.
//! Invariants: never reads or writes the process environment.

use super::{
    EVICT_RANDOM_PCT_BUILT_IN, RDMA_DEV_BUILT_IN, RDMA_GID_BUILT_IN, Rail, WeightsSettings,
};

#[test]
fn declared_settings_load_bf16_through_direct_reads_on_the_built_in_rails() {
    let rail0 = Rail {
        dev: RDMA_DEV_BUILT_IN.into(),
        gid: RDMA_GID_BUILT_IN,
    };
    assert_eq!(
        WeightsSettings::declared().expect("declared defaults resolve"),
        WeightsSettings {
            gguf_native_q2: false,
            q2_group: 128,
            gguf_force_cpu: false,
            keep_page_cache: false,
            direct_reads: true,
            kquant_resident: true,
            q6k_head: true,
            evict_random_pct: EVICT_RANDOM_PCT_BUILT_IN,
            weight_rail0: rail0.clone(),
            weight_rail1: None,
            lora_rail: rail0,
        }
    );
}
