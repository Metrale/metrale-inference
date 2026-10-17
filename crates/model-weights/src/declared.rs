// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The model-weights constructors at the declared `WeightsSettings`, for tests and
//! tools. A serve resolves the settings once at startup and calls the `*_with`/`with_settings`
//! forms, which take them.
//!
//! Owner: model-weights.
//! Invariants: never reads the process environment.

use std::path::Path;

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_settings::groups::device_weights::WeightsSettings;

use crate::weights::GgufLoader;
use crate::weights::expert_stream::{ExpertLru, ShardFiles, SlotLayout};

impl GgufLoader {
    /// 2026-10-17: [`GgufLoader::with_settings`] with no expert parallelism, at the declared
    /// settings.
    pub fn new() -> Self {
        let declared = WeightsSettings::declared().expect("the declared weights table projects");
        Self::with_settings(0, 1, 0, declared)
    }
}

impl Default for GgufLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl ShardFiles {
    /// 2026-10-17: [`ShardFiles::open_dir_with`] at the declared settings.
    ///
    /// # Errors
    /// As [`ShardFiles::open_dir_with`].
    pub fn open_dir(dir: &Path) -> Result<Self> {
        Self::open_dir_with(dir, &WeightsSettings::declared()?)
    }
}

impl ExpertLru {
    /// 2026-10-17: [`ExpertLru::with_settings`] at the declared settings.
    ///
    /// # Errors
    /// As [`ExpertLru::with_settings`].
    pub fn new(host: *mut u8, dev: DevicePtr, bytes: usize, layout: SlotLayout) -> Result<Self> {
        Self::with_settings(host, dev, bytes, layout, &WeightsSettings::declared()?)
    }
}
