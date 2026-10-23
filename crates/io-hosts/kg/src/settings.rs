// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The native host of the two-layer settings (owner requirement 2026-10-23): read the
//! global layer from [`GLOBAL_CONFIG`] under `$HOME`, merge it under the project layer (pure, in
//! `metrale_kg_derive::settings`) and resolve once at startup.
//!
//! Owner: metrale-io-host-kg.
//! Invariants:
//! - The one place the global file's location is named; renaming the directory changes only
//!   [`GLOBAL_CONFIG`].
//! - An absent global file is an empty layer and is reported as such; an unreadable or invalid one
//!   is `KgConfig` (fail fast, never silently skipped).

use std::path::PathBuf;

use metrale_error::{ErrorCode, Result, error};
use metrale_kg_derive::settings::{EmbedLayer, Embedder, Provenance, merge, parse_global, resolve};

/// 2026-10-23: The global layer, relative to `$HOME` (DESIGN-latlas.md: `~/.latlas/config.toml`).
pub const GLOBAL_CONFIG: &str = ".latlas/config.toml";

/// 2026-10-23: What the global layer was.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GlobalLayer {
    Read(PathBuf),
    Absent(PathBuf),
    NoHome,
}

impl GlobalLayer {
    /// 2026-10-23: The report form.
    pub fn describe(&self) -> String {
        match self {
            Self::Read(p) => format!("read {}", p.display()),
            Self::Absent(p) => format!("absent ({} does not exist)", p.display()),
            Self::NoHome => "absent ($HOME is unset)".into(),
        }
    }
}

/// 2026-10-23: The resolved settings with where they came from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Resolved {
    pub embedder: Embedder,
    pub provenance: Provenance,
    pub global: GlobalLayer,
}

/// 2026-10-23: Read the global layer and resolve it under `project`.
pub fn resolve_settings(project: &EmbedLayer) -> Result<Resolved> {
    let (layer, global) = match std::env::var_os("HOME") {
        None => (EmbedLayer::default(), GlobalLayer::NoHome),
        Some(home) => {
            let path = PathBuf::from(home).join(GLOBAL_CONFIG);
            match std::fs::read_to_string(&path) {
                Ok(t) => (parse_global(&t)?, GlobalLayer::Read(path)),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    (EmbedLayer::default(), GlobalLayer::Absent(path))
                }
                Err(e) => {
                    return Err(error!(
                        ErrorCode::KgConfig,
                        path.display().to_string(),
                        format!("unreadable: {}", e.kind())
                    ));
                }
            }
        }
    };
    let (merged, provenance) = merge(&layer, project);
    Ok(Resolved {
        embedder: resolve(&merged)?,
        provenance,
        global,
    })
}
