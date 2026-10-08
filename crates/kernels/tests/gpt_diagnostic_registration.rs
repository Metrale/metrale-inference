// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Diagnostic expert linkage must preserve the common module.
use std::path::Path;

#[test]
fn diagnostic_experts_do_not_shadow_the_common_experts() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap();
    let target = metrale_closure::layout::Target {
        hardware: "gb10".into(),
        model: "gpt-oss-20b".into(),
        quant: "mxfp4".into(),
    };
    let layout = metrale_closure::layout::discover(&root, &target).unwrap();
    let modules = layout.modules();
    let source = |name: &str| {
        modules
            .iter()
            .find(|(stem, _)| stem == name)
            .unwrap()
            .1
            .source
            .clone()
    };
    assert_eq!(
        source("moe_w4a16_grouped_gemm"),
        root.join("kernels/gb10/common/moe_w4a16_grouped_gemm.cu")
    );
    let diagnostic = source("gpt_oss_mxfp4_mma");
    let body = std::fs::read_to_string(diagnostic).unwrap();
    assert!(body.contains("../../deepseek-v4-flash/nvfp4/moe_w4a16_grouped_gemm.cu"));
    assert!(
        !body.contains("__global__"),
        "shared implementation must not be copied"
    );
    let mut common_alias = None;
    for config in layout.configs() {
        let manifest: toml::Value =
            toml::from_str(&std::fs::read_to_string(config).unwrap()).unwrap();
        if let Some(alias) = manifest
            .get("modules")
            .and_then(|m| m.get("moe_w4a16_grouped_gemm"))
        {
            common_alias = alias.as_str().map(str::to_owned);
        }
    }
    assert_eq!(common_alias.as_deref(), Some("moe_w4a16"));
}

// 2026-10-07: The thin module's shared implementation is outside GPT's layout inputs.
#[test]
fn diagnostic_rebuild_inputs_include_the_shared_implementation() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap();
    let thin = root.join("kernels/gb10/gpt-oss-20b/mxfp4/gpt_oss_mxfp4_mma.cu");
    let shared = root.join("kernels/gb10/deepseek-v4-flash/nvfp4/moe_w4a16_grouped_gemm.cu");
    let closure = metrale_closure::hash_with_report(
        &root,
        &metrale_closure::ClosureInputs {
            sources: vec![thin.clone()],
            configs: vec![],
            flags: vec![],
            arch: "sm_121a".into(),
            compiler: "dependency-only fixture, no compiler invoked".into(),
        },
    )
    .unwrap();
    assert!(closure.files.contains(&thin));
    assert!(closure.files.contains(&shared));
}
