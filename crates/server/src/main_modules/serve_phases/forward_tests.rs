// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: `--forward` resolution against the embedded INSTANCES.toml (no stub: the lookup
//! the server boots with). Each test was proven RED by sabotaging `resolve_forward`.

use super::*;

const GLM: &str = "nvidia/GLM-5.3-Flash-NVFP4";
const GLM_TARGET: &str = "gb10/glm-5.3-flash/nvfp4";
// 2026-10-23: A checkpoint and target no circuit instance names.
const OTHER: &str = "meta-llama/Llama-3.1-8B-Instruct";
const OTHER_TARGET: &str = "gb10/llama-3.1-8b/nvfp4";
// 2026-10-23: The rank count of the GLM recipe the campaign serves (`--tp-size 3`).

#[test]
fn an_unset_forward_on_glm_resolves_to_circuit_with_its_instance() {
    let r = resolve_forward(None, Some(GLM), Ok(GLM_TARGET), 3).unwrap();
    assert_eq!(r.mode, ForwardArg::Circuit, "{}", r.reason);
    assert_eq!(r.instance.unwrap().checkpoint, GLM);
    assert!(r.reason.contains("circuit instance"), "{}", r.reason);
    // 2026-10-23: The rank count picks the deployment: no GLM recipe deploys 2 ranks.
    let r = resolve_forward(None, Some(GLM), Ok(GLM_TARGET), 2).unwrap();
    assert_eq!(r.mode, ForwardArg::Legacy, "{}", r.reason);
}

#[test]
fn an_unset_forward_without_an_instance_resolves_to_legacy_and_says_why() {
    let r = resolve_forward(None, Some(OTHER), Ok(OTHER_TARGET), 3).unwrap();
    assert_eq!(r.mode, ForwardArg::Legacy);
    assert!(r.instance.is_none());
    assert!(r.reason.contains("no circuit instance serves"), "{}", r.reason);
    let r = resolve_forward(None, None, Ok(GLM_TARGET), 3).unwrap();
    assert_eq!(r.mode, ForwardArg::Legacy);
    assert!(r.reason.contains("no checkpoint id"), "{}", r.reason);
    let r = resolve_forward(None, Some(GLM), Err("kernel arch `gfx1151` is not an sm_ arch"), 3).unwrap();
    assert_eq!(r.mode, ForwardArg::Legacy);
    assert!(r.reason.contains("gfx1151"), "{}", r.reason);
}

#[test]
fn an_explicit_circuit_without_an_instance_is_refused() {
    for mode in [ForwardArg::Circuit, ForwardArg::CircuitReference] {
        let e = resolve_forward(Some(mode), Some(OTHER), Ok(OTHER_TARGET), 3).unwrap_err();
        assert!(format!("{e:#}").contains("no circuit instance serves"), "{e:#}");
        let e = resolve_forward(Some(mode), None, Ok(GLM_TARGET), 3).unwrap_err();
        assert!(format!("{e:#}").contains("needs the checkpoint id"), "{e:#}");
        assert!(resolve_forward(Some(mode), Some(GLM), Err("no circuit target"), 3).is_err());
    }
}

#[test]
fn an_explicit_legacy_wins_where_an_instance_exists() {
    let r = resolve_forward(Some(ForwardArg::Legacy), Some(GLM), Ok(GLM_TARGET), 3).unwrap();
    assert_eq!(r.mode, ForwardArg::Legacy);
    assert!(r.instance.is_none());
    let r = resolve_forward(Some(ForwardArg::Circuit), Some(GLM), Ok(GLM_TARGET), 3).unwrap();
    assert_eq!(r.mode, ForwardArg::Circuit);
}

#[test]
fn the_cli_leaves_forward_unset_by_default() {
    use clap::Parser;
    let a = cli::Cli::try_parse_from(["met", "serve", GLM]).unwrap();
    let cli::Command::Serve(s) = a.command else {
        panic!("not a serve")
    };
    assert_eq!(s.forward, None);
}
