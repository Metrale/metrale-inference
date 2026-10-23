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
    // 2026-10-23: Any rank count the circuit admits resolves to circuit, its deployment derived
    // for that N; one rank takes the one-device recipe; a count a part cannot split at (GLM's
    // vocab past four) runs legacy, naming the part.
    for n in [2, 3, 4] {
        let r = resolve_forward(None, Some(GLM), Ok(GLM_TARGET), n).unwrap();
        assert_eq!(r.mode, ForwardArg::Circuit, "{n}: {}", r.reason);
        assert_eq!(r.instance.unwrap().parallel.unwrap().ranks, n);
    }
    let one = resolve_forward(None, Some(GLM), Ok(GLM_TARGET), 1).unwrap();
    assert_eq!(one.mode, ForwardArg::Circuit, "{}", one.reason);
    assert!(one.instance.unwrap().parallel.is_none());
    for n in 5..=8 {
        let r = resolve_forward(None, Some(GLM), Ok(GLM_TARGET), n).unwrap();
        assert_eq!(r.mode, ForwardArg::Legacy, "{n}: {}", r.reason);
        assert!(r.reason.contains("part `vocab`"), "{}", r.reason);
        let e = resolve_forward(Some(ForwardArg::Circuit), Some(GLM), Ok(GLM_TARGET), n);
        assert!(format!("{:#}", e.unwrap_err()).contains("part `vocab`"));
    }
}

#[test]
fn an_unset_forward_without_an_instance_resolves_to_legacy_and_says_why() {
    let r = resolve_forward(None, Some(OTHER), Ok(OTHER_TARGET), 3).unwrap();
    assert_eq!(r.mode, ForwardArg::Legacy);
    assert!(r.instance.is_none());
    assert!(
        r.reason.contains("no circuit instance serves"),
        "{}",
        r.reason
    );
    let r = resolve_forward(None, None, Ok(GLM_TARGET), 3).unwrap();
    assert_eq!(r.mode, ForwardArg::Legacy);
    assert!(r.reason.contains("no checkpoint id"), "{}", r.reason);
    let r = resolve_forward(
        None,
        Some(GLM),
        Err("kernel arch `gfx1151` is not an sm_ arch"),
        3,
    )
    .unwrap();
    assert_eq!(r.mode, ForwardArg::Legacy);
    assert!(r.reason.contains("gfx1151"), "{}", r.reason);
}

#[test]
fn an_explicit_circuit_without_an_instance_is_refused() {
    for mode in [ForwardArg::Circuit, ForwardArg::CircuitReference] {
        let e = resolve_forward(Some(mode), Some(OTHER), Ok(OTHER_TARGET), 3).unwrap_err();
        assert!(
            format!("{e:#}").contains("no circuit instance serves"),
            "{e:#}"
        );
        let e = resolve_forward(Some(mode), None, Ok(GLM_TARGET), 3).unwrap_err();
        assert!(
            format!("{e:#}").contains("needs the checkpoint id"),
            "{e:#}"
        );
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
