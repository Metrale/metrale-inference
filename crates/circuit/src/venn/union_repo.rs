// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The whole union over a repository: every instance on a hardware class is the
//! target of a Venn report against the golden instances (all but itself) at C1, C16 and C128,
//! folded by [`super::union::build_union`]. One assembly for the checked-in pre-sweep report
//! (`crates/circuit/tests/venn_union.rs`) and the post-sweep one (`met accuracy envelope union`).
//!
//! Owner: metrale-circuit (venn).
//! Invariants: business logic only; every byte arrives through [`Repo`] (SBIO).

use super::union::{SweptCell, UnionInput, UnionReport, build_union};
use super::{Repo, VennArgs, build, load_instance, parse_families, parse_measurements};
use crate::instances::parse_instances;
use crate::rules::Mode;
use crate::venn::report::{Side, VennInputs};

/// 2026-10-10: The rows every union report covers: C1, C16 and C128.
pub const UNION_ROWS: [u64; 3] = [1, 16, 128];

/// 2026-10-10: The union of every instance on `hardware` in `repo`.
pub fn union_of_repo(
    repo: &dyn Repo,
    hardware: &str,
    swept: &dyn Fn(SweptCell<'_>) -> bool,
) -> Result<UnionReport, String> {
    let all = parse_instances(&repo.read("kernels/circuits/INSTANCES.toml")?)
        .map_err(|e| e.to_string())?;
    let all: Vec<_> = all
        .into_iter()
        .filter(|i| i.target.split('/').next() == Some(hardware))
        .collect();
    let loaded = all
        .iter()
        .map(|i| load_instance(repo, i).map_err(|e| e.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    let fams =
        parse_families(&repo.read(&format!("kernels/{hardware}/common/KERNEL_FAMILIES.toml"))?)
            .map_err(|e| e.to_string())?;
    let meas = parse_measurements(&repo.read("docs/kernel-perf/measurements.toml")?)?;
    let mut reports = Vec::with_capacity(all.len());
    for (ti, t) in all.iter().enumerate() {
        let against: Vec<Side<'_>> = all
            .iter()
            .zip(&loaded)
            .enumerate()
            .filter(|(i, (inst, _))| inst.golden && *i != ti)
            .map(|(_, (instance, loaded))| Side { instance, loaded })
            .collect();
        let args = VennArgs {
            target: t.recipe.clone(),
            against: against.iter().map(|s| s.instance.recipe.clone()).collect(),
            modes: vec![Mode::Decode, Mode::MultiSeq],
            rows: UNION_ROWS.to_vec(),
            verify_rows: vec![2],
            out: String::new(),
        };
        let r = build(&VennInputs {
            target: Side {
                instance: t,
                loaded: &loaded[ti],
            },
            against,
            families: &fams,
            measurements: &meas,
            runs: args.runs().map_err(|e| e.to_string())?,
            command: args.command(),
        })
        .map_err(|e| format!("{}: {e}", t.recipe))?;
        reports.push(r);
    }
    let inputs: Vec<UnionInput<'_>> = all
        .iter()
        .zip(&loaded)
        .zip(&reports)
        .map(|((instance, l), report)| UnionInput {
            instance,
            circuit: &l.circuit,
            report,
        })
        .collect();
    build_union(&inputs, swept)
}
