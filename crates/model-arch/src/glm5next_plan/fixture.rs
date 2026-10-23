// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The served GLM-5.3 TP=3 configuration for CPU equivalence gates: each rank's
//! circuit and fused plans from the checked-in instance (kernels/circuits/INSTANCES.toml), the
//! layer levers from the served recipe's `env:` (recipes/glm-5.3/), and each rank's MLP config
//! from the checked-in checkpoint config.json.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: reads the repository only; every value comes from a checked-in file, none is
//! restated here.

use std::path::PathBuf;

use metrale_circuit::{AvailableKernels, Circuit, FusionPlan, Mode};
use metrale_model_layers::circuit_exec::sources;

use crate::glm5next_layer::GlmLevers;
use crate::glm5next_mlp::Glm5NextMlpConfig;

/// 2026-10-23: The served recipe and its instance.
pub(crate) const RECIPE: &str = "glm-5.3/glm-5.3-flash-nvfp4-tp3-512k";
/// 2026-10-23: The checkpoint config the recipe serves.
const CONFIG: &str =
    "crates/circuit/tests/fixtures/checkpoints/nvidia--GLM-5.3-Flash-NVFP4/config.json";

pub(crate) fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// 2026-10-23: Each rank's circuit and the rules of the served instance.
pub(crate) fn ranks() -> (
    Vec<Circuit>,
    Vec<metrale_circuit::Rule>,
    metrale_circuit::Policy,
) {
    let inst = sources::instance(RECIPE).unwrap();
    let loaded = metrale_circuit::load(&inst, sources::sources(&inst).unwrap()).unwrap();
    assert_eq!(
        loaded.ranks.len(),
        3,
        "the served instance plans three ranks"
    );
    (loaded.ranks, loaded.rules, inst.policy)
}

/// 2026-10-23: Rank `rank`'s plan at `mode` and `rows`, with every kernel a rule names available.
pub(crate) fn plan(
    circuits: &[Circuit],
    rules: &[metrale_circuit::Rule],
    policy: &metrale_circuit::Policy,
    rank: usize,
    mode: Mode,
    rows: u64,
) -> FusionPlan {
    let avail = AvailableKernels::all_named_by(rules);
    metrale_circuit::fuse(&circuits[rank], rules, &avail, policy, mode, rows)
        .unwrap_or_else(|e| panic!("rank {rank} {mode:?} {rows}: {e}"))
}

/// 2026-10-23: The recipe's top-level section `section:` (`env`, `defaults`) as `(key, value)`
/// pairs, comments and blank lines skipped, quotes stripped.
pub(crate) fn recipe_section(section: &str) -> Vec<(String, String)> {
    let path = root().join(format!("recipes/{RECIPE}.yaml"));
    let text = std::fs::read_to_string(&path).unwrap();
    let mut out = Vec::new();
    let mut inside = false;
    for line in text.lines() {
        if line == format!("{section}:") {
            inside = true;
            continue;
        }
        if !inside {
            continue;
        }
        let t = line.trim();
        if t.is_empty() || t.starts_with('#') {
            continue;
        }
        if !line.starts_with("  ") {
            break;
        }
        let (k, v) = t
            .split_once(':')
            .unwrap_or_else(|| panic!("{section} line `{line}`"));
        out.push((k.trim().to_string(), v.trim().trim_matches('"').to_string()));
    }
    out
}

/// 2026-10-23: The recipe's `env:` as `(name, value)` pairs.
pub(crate) fn recipe_env() -> Vec<(String, String)> {
    let out = recipe_section("env");
    assert!(
        out.len() > 30,
        "the recipe's env: section was not read ({} pairs)",
        out.len()
    );
    out
}

/// 2026-10-23: The recipe's `defaults:` value of `key`.
pub(crate) fn recipe_default(key: &str) -> String {
    recipe_section("defaults")
        .into_iter()
        .find(|(k, _)| k == key)
        .unwrap_or_else(|| panic!("the recipe's defaults: has no `{key}`"))
        .1
}

/// 2026-10-23: The served startup settings: the recipe's `env:` over the declared defaults.
pub(crate) fn served_settings() -> metrale_settings::InferenceServerSettings {
    let env = recipe_env();
    let pairs: Vec<(&str, &str)> = env.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
    metrale_model_layers::lever_equivalence::settings_from(&pairs).unwrap()
}

/// 2026-10-23: Run `body` in a child process of this test binary that runs only the test `path`
/// (its full path in the binary), after `publish` set the process-wide tiers the body reads; the
/// parent asserts the child ran exactly that test and passed. The tiers are published once per
/// process, and the rest of this binary runs without them; nothing on the paths is mocked.
pub(crate) fn in_child_process(path: &str, publish: impl FnOnce(), body: impl FnOnce()) {
    const CHILD: &str = "METRALE_TEST_GLM_PLAN_CHILD";
    if std::env::var_os(CHILD).is_some() {
        publish();
        body();
        return;
    }
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args([path, "--exact", "--nocapture", "--test-threads=1"])
        .env(CHILD, "1")
        .output()
        .unwrap();
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(out.status.success(), "child `{path}` failed:\n{text}");
    assert!(
        text.contains("1 passed"),
        "child `{path}` did not run the test:\n{text}"
    );
}

/// 2026-10-23: The served layer levers: the recipe's `env:` over the declared defaults.
pub(crate) fn served_levers() -> GlmLevers {
    GlmLevers::from_settings(&served_settings()).unwrap()
}

/// 2026-10-23: Rank `rank`'s MLP config under the recipe's `tensor_parallel: 3`, `ep_size: 1`,
/// `moe_expert_layout: tp`.
pub(crate) fn mlp_config(rank: usize) -> Glm5NextMlpConfig {
    Glm5NextMlpConfig::from_config(&model_config(rank)).unwrap()
}

/// 2026-10-23: Rank `rank`'s model config under the recipe's parallel layout.
pub(crate) fn model_config(rank: usize) -> metrale_config::ModelConfig {
    let json = std::fs::read_to_string(root().join(CONFIG)).unwrap();
    let mut c = metrale_config::parse_config(&json).unwrap();
    c.tp_world_size = 3;
    c.tp_rank = rank;
    c.ep_world_size = 1;
    c.moe_expert_layout = metrale_config::MoeExpertLayout::Tp;
    c
}

/// 2026-10-23: The routed MoE layers of `c` (every layer with a `moe` block).
pub(crate) fn moe_layers(c: &Circuit) -> Vec<usize> {
    let mut v: Vec<usize> = c
        .nodes
        .iter()
        .filter(|n| n.block == "moe")
        .filter_map(|n| n.layer)
        .collect();
    v.dedup();
    v
}

/// 2026-10-23: The runs a speculative TP=3 step makes, and the non-speculative decode: the shapes
/// plan execution installs at.
pub(crate) const RUNS: [(Mode, u64); 2] = super::install::GATED;
