// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: An in-memory kernel tree for the hardware tests: three classes (`base` with a
//! native FP4 MMA device, `child` inheriting `base` and compiling the FP4 region out, `lonely`
//! with no rules), the toy circuit's rules and families, and a device registry in the research
//! schema of kernels/DEVICES.toml.
//!
//! Owner: metrale-circuit tests.
//! Invariants: none beyond the types.

use std::collections::{BTreeMap, BTreeSet};

use super::sources::{ClassSources, KernelTree, Module};
use super::{ModelUnderPlan, Registry, parse_devices};
use crate::fuser::Policy;
use crate::venn::repo::Repo;

/// 2026-09-30: One device profile in the research schema.
pub struct Dev<'a> {
    pub id: &'a str,
    pub class: &'a str,
    pub family: &'a str,
    pub bw: f64,
    pub bf16: f64,
    pub fp8: f64,
    pub fp4: f64,
    pub gib: f64,
}

impl Dev<'_> {
    fn text(&self) -> String {
        format!(
            r#"
[[device]]
id = "{id}"
class = "{class}"
name = "{id}"
arch = "sm_{class}"
compute_capability = "0.0"
sm_count = 1
memory_type = "HBM"
memory_gb_datasheet = {gib}
memory_bandwidth_gbps = {bw}
smem_per_sm_kb = 100
tmem_per_sm_kb = 0
mma_family = "{family}"
max_cluster = 1
derived = []
sources = ["S1"]
usable_fraction = 0.9
usable_why = "test"
[device.peak_tflops]
bf16 = {bf16}
fp8 = {fp8}
fp4_nvfp4 = {fp4}
[device.native_mma]
bf16_bf16 = true
fp8_fp8 = {has8}
fp4_fp4_nvfp4_block16 = {has4}
"#,
            id = self.id,
            class = self.class,
            gib = self.gib,
            bw = self.bw,
            family = self.family,
            bf16 = self.bf16,
            fp8 = self.fp8,
            fp4 = self.fp4,
            has8 = self.fp8 > 0.0,
            has4 = self.fp4 > 0.0,
        )
    }
}

/// 2026-09-30: The default devices: `fp4dev` (base, native FP4), `nofp4` and `nofp4big` (child,
/// same class, different roofline and memory), `slowmath` (child, compute-starved), `nofp8`
/// (child, BF16 only) and `alone` (lonely).
pub fn devices() -> Vec<Dev<'static>> {
    let d = |id, class, family, bw, bf16, fp8, fp4, gib| Dev {
        id,
        class,
        family,
        bw,
        bf16,
        fp8,
        fp4,
        gib,
    };
    vec![
        d("fp4dev", "base", "mma_sync", 1000.0, 100.0, 200.0, 400.0, 64.0),
        d("nofp4", "child", "wgmma", 1000.0, 100.0, 200.0, 0.0, 64.0),
        d("nofp4big", "child", "wgmma", 2000.0, 100.0, 200.0, 0.0, 128.0),
        d("slowmath", "child", "wgmma", 1e6, 1e-6, 2e-6, 0.0, 64.0),
        d("nofp8", "child", "wgmma", 1000.0, 100.0, 0.0, 0.0, 64.0),
        d("alone", "lonely", "wgmma", 1000.0, 100.0, 200.0, 0.0, 64.0),
    ]
}

/// 2026-09-30: The registry text for `devs`.
pub fn registry_text(devs: &[Dev<'_>]) -> String {
    let mut s = String::from(
        r#"
schema_version = 1
compiled = "test"
[sources]
S1 = "test"
[[guard]]
macro = "METRALE_NO_WARP_BLOCKSCALE_MMA"
polarity = "ifndef"
requires = "mma_sync.fp4_block_scale"
"#,
    );
    for d in devs {
        s.push_str(&d.text());
    }
    s
}

/// 2026-09-30: The parsed default registry.
pub fn registry() -> Registry {
    parse_devices(&registry_text(&devices())).unwrap_or_else(|e| panic!("fixture registry: {e}"))
}

fn rule(id: &str, pattern: &str, func: &str, priority: i64) -> String {
    format!(
        r#"
[[rule]]
id = "{id}"
pattern = [{pattern}]
kernels = [{{ module = "m", func = "{func}" }}]
repeat = "once"
emitter = "{id}"
rows = [1, 128]
modes = ["decode", "multi_seq", "verify"]
numerics = "reference"
priority = {priority}
cite = "test"
"#
    )
}

/// 2026-09-30: The base class's rules: one per toy op, plus an FP4-MMA gate_up at a higher
/// priority.
pub fn base_rules() -> String {
    let mut s = String::from("schema = 1\n");
    for (id, pattern) in [
        ("embed", r#"{ op = "embed" }"#),
        ("norm", r#"{ op = "rms_norm" }"#),
        ("up", r#"{ op = "linear", role = "gate_up" }"#),
        ("act", r#"{ op = "silu_mul" }"#),
        ("down", r#"{ op = "linear", role = "down" }"#),
        ("add", r#"{ op = "residual_add" }"#),
        ("final_norm", r#"{ op = "final_norm" }"#),
        ("lm_head", r#"{ op = "lm_head" }"#),
    ] {
        s.push_str(&rule(id, pattern, id, 10));
    }
    s.push_str(&rule(
        "up_fp4",
        r#"{ op = "linear", role = "gate_up" }"#,
        "up_fp4",
        50,
    ));
    s
}

/// 2026-09-30: The kernel module every class compiles: `up_fp4` sits in the FP4 guard.
pub const MODULE: &str = r#"
__global__ void embed(int x) {}
__global__ void norm(int x) {}
__global__ void up(int x) {}
__global__ void act(int x) {}
__global__ void act_child(int x) {}
__global__ void down(int x) {}
__global__ void add(int x) {}
__global__ void final_norm(int x) {}
__global__ void lm_head(int x) {}
#ifndef METRALE_NO_WARP_BLOCKSCALE_MMA
extern "C" __global__ void __launch_bounds__(256)
up_fp4(int x) {}
#endif
"#;

fn families() -> String {
    let mut s = String::from(
        r#"
schema = 1
hardware = "base"
[roofline]
dram_gbps = 900.0
bf16_tflops = 90.0
fp8_tflops = 180.0
nvfp4_tflops = 360.0
context_tokens = 4096
"#,
    );
    for (id, op, func) in [
        ("f_embed", r#"{ op = "embed" }"#, "embed"),
        ("f_norm", r#"{ op = "rms_norm" }, { op = "final_norm" }"#, "norm"),
        ("f_up", r#"{ op = "linear", roles = ["gate_up"] }"#, "up"),
        ("f_up_fp4", r#"{ op = "linear", roles = ["gate_up"] }"#, "up_fp4"),
        ("f_act", r#"{ op = "silu_mul" }"#, "act"),
        ("f_down", r#"{ op = "linear", roles = ["down"] }"#, "down"),
        ("f_add", r#"{ op = "residual_add" }"#, "add"),
        ("f_head", r#"{ op = "lm_head" }"#, "lm_head"),
    ] {
        let extra = if func == "norm" {
            r#", "m::final_norm""#
        } else if func == "act" {
            r#", "m::act_child""#
        } else {
            ""
        };
        s.push_str(&format!(
            r#"
[[family]]
id = "{id}"
description = "test"
kernels = ["m::{func}"{extra}]
rows = [1, 128]
op = [{op}]
[[family.point]]
values = {{}}
how = "instantiation"
files = ["kernels/base/common/m.cu"]
[[family.evidence]]
point = {{}}
rows = [1]
microbench = "test"
"#
        ));
    }
    s
}

/// 2026-09-30: The in-memory tree.
pub struct Tree {
    pub files: BTreeMap<String, String>,
}

/// 2026-09-30: The default tree.
pub fn tree() -> Tree {
    let mut files = BTreeMap::new();
    let hw = |arch: &str, extra: &str| format!("[hardware]\narch = \"{arch}\"\n{extra}");
    let flags = "[build]\nextra_nvcc_flags = [\"-DMETRALE_NO_WARP_BLOCKSCALE_MMA\"]\n";
    files.insert("kernels/base/HARDWARE.toml".into(), hw("sm_base", ""));
    files.insert(
        "kernels/child/HARDWARE.toml".into(),
        hw("sm_child", &format!("inherits = \"base\"\n{flags}")),
    );
    files.insert("kernels/lonely/HARDWARE.toml".into(), hw("sm_lonely", flags));
    files.insert("kernels/base/common/FUSIONS.toml".into(), base_rules());
    files.insert(
        "kernels/child/common/FUSIONS.toml".into(),
        format!(
            "schema = 1\ninherits = \"base\"\n{}",
            rule("act", r#"{ op = "silu_mul" }"#, "act_child", 10)
        ),
    );
    files.insert("kernels/base/common/KERNEL_FAMILIES.toml".into(), families());
    files.insert(
        "kernels/lonely/common/KERNEL_FAMILIES.toml".into(),
        families().replace("hardware = \"base\"", "hardware = \"lonely\""),
    );
    Tree { files }
}

impl Repo for Tree {
    fn read(&self, rel: &str) -> Result<String, String> {
        self.files
            .get(rel)
            .cloned()
            .ok_or_else(|| format!("{rel}: not in the fixture"))
    }

    fn list(&self, rel: &str) -> Result<Vec<String>, String> {
        Ok(self
            .files
            .keys()
            .filter(|k| k.starts_with(rel))
            .cloned()
            .collect())
    }
}

impl KernelTree for Tree {
    fn class_sources(&self, class: &str, _model: &str, _quant: &str) -> Result<ClassSources, String> {
        let path = "kernels/base/common/m.cu".to_string();
        Ok(ClassSources {
            class: class.to_string(),
            target: Some(format!("{class}/toy/q")),
            modules: BTreeMap::from([(
                "m".to_string(),
                Module {
                    path: path.clone(),
                    text: MODULE.to_string(),
                },
            )]),
            files: BTreeSet::from([path]),
            expected_absent: BTreeMap::new(),
        })
    }

    fn kernel_target(
        &self,
        _model_type: &str,
        _hidden: u64,
        _refs: &[&str],
    ) -> Result<Option<String>, String> {
        Ok(Some("toy".into()))
    }

    fn as_repo(&self) -> &dyn Repo {
        self
    }
}

/// 2026-09-30: The toy model (2 layers) with `precision`'s formats.
pub fn model(precision: &str) -> ModelUnderPlan {
    let table = crate::precision::PrecisionTable::parse(precision).unwrap();
    let circuit = crate::instantiate(crate::test_toy::CIRCUIT, &[], &crate::test_toy::shape(2), &table)
        .unwrap_or_else(|e| panic!("toy: {e}"));
    ModelUnderPlan {
        label: "toy".into(),
        checkpoint: "toy".into(),
        kernel_model: "toy".into(),
        kernel_quant: "q".into(),
        circuit,
        policy: Policy {
            opt_in_levers: BTreeSet::new(),
            settings: BTreeMap::from([
                ("kv_cache_dtype".to_string(), "bf16".to_string()),
                ("ssm_h_dtype".to_string(), "f32".to_string()),
            ]),
        },
        header: vec![("target".into(), "base/toy/q".into())],
        precision: "test".into(),
        precision_choice: super::PrecisionChoice::Declared,
    }
}
