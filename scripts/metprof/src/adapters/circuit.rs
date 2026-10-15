// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the circuit texts from a `kernels/` directory, laid out as the
//! engine keeps them:
//!
//! - `circuits/INSTANCES.toml`
//! - `circuits/<arch>.toml`
//! - `circuits/blocks/<name>.toml`
//! - `circuits/precision/<name>.toml`
//! - `circuits/checkpoints/<name>.toml`
//! - `<hw>/common/FUSIONS.toml`
//!
//! Every read goes through the `Files` port; a name that would leave the
//! directory is refused before anything is read.

use crate::core::error::{Error, Result};
use crate::core::tree::circuit::{PlanTexts, PrecisionFile, includes, needs};
use crate::fail;
use crate::ports::{CircuitTexts, Files};

pub struct KernelsDir<'a> {
    pub files: &'a dyn Files,
    /// The `kernels/` directory.
    pub root: &'a str,
}

impl KernelsDir<'_> {
    fn read(&self, rel: &str) -> Result<String> {
        let path = format!("{}/{rel}", self.root.trim_end_matches('/'));
        let bytes = self
            .files
            .read(&path)
            .map_err(|e| Error(format!("cannot read {path}: {e}")))?;
        String::from_utf8(bytes).map_err(|_| Error(format!("{path}: not UTF-8")))
    }

    fn named(&self, dir: &str, name: &str) -> Result<String> {
        let ok = !name.is_empty()
            && name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
            && !name.contains("..");
        if !ok {
            fail!("circuit file name {name:?} is not a plain name");
        }
        self.read(&format!("{dir}{name}.toml"))
    }
}

impl CircuitTexts for KernelsDir<'_> {
    fn instances(&self) -> Result<String> {
        self.read("circuits/INSTANCES.toml")
    }

    fn circuit(&self, arch: &str) -> Result<String> {
        self.named("circuits/", arch)
    }

    fn block(&self, name: &str) -> Result<String> {
        self.named("circuits/blocks/", name)
    }

    fn precision_table(&self, name: &str) -> Result<String> {
        self.named("circuits/precision/", name)
    }

    fn checkpoint_plan(&self, name: &str) -> Result<String> {
        self.named("circuits/checkpoints/", name)
    }

    fn fusions(&self, hw: &str) -> Result<String> {
        let ok = !hw.is_empty() && hw.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
        if !ok {
            fail!("hardware class {hw:?} is not a plain name");
        }
        self.read(&format!("{hw}/common/FUSIONS.toml"))
    }
}

/// Every text the plan of `recipe` (an instance of `arch`) is built from.
pub fn plan_texts(t: &dyn CircuitTexts, recipe: &str, arch: &str) -> Result<PlanTexts> {
    let instances = t.instances()?;
    let n = needs(&instances, recipe, arch)?;
    let circuit = t.circuit(&n.arch)?;
    let mut blocks = Vec::new();
    for b in includes(&circuit)? {
        let text = t.block(&b)?;
        blocks.push((b, text));
    }
    let precision = match &n.precision {
        PrecisionFile::Table(p) => t.precision_table(p)?,
        PrecisionFile::CheckpointPlan(p) => t.checkpoint_plan(p)?,
    };
    let fusions = t.fusions(&n.hw)?;
    Ok(PlanTexts {
        instances,
        circuit,
        blocks,
        precision,
        fusions,
    })
}
