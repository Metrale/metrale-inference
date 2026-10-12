// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Explicit precision declarations reject common-only fallback while legacy models retain it.
use super::*;

#[test]
fn declared_precision_is_not_a_common_kernel_fallback() {
    let fx = Fx::new("quant-allowlist");
    fx.file(
        "gb10/modelB/MODEL.toml",
        "[model]\nsupported_quants = [\"mxfp4\"]\n",
    );
    assert!(fx.discover("gb10", "modelB", "mxfp4").is_ok());
    assert!(matches!(
        fx.discover("gb10", "modelB", "nvfp4"),
        Err(LayoutError::Manifest { .. })
    ));
    assert!(matches!(
        fx.discover("gb10", "modelB", "phantom"),
        Err(LayoutError::Manifest { .. })
    ));
    fx.file("gb10/modelB/MODEL.toml", "[model]\n");
    assert!(fx.discover("gb10", "modelB", "phantom").is_ok());
}

#[test]
fn malformed_allowlist_fails_closed() {
    let fx = Fx::new("quant-malformed");
    for declaration in ["[]", "[1]", "[\"\"]", "\"mxfp4\""] {
        fx.file(
            "gb10/modelB/MODEL.toml",
            &format!("[model]\nsupported_quants = {declaration}\n"),
        );
        assert!(matches!(
            fx.discover("gb10", "modelB", "mxfp4"),
            Err(LayoutError::Manifest { .. })
        ));
    }
}

#[test]
fn redirect_cannot_escape_source_precision_restriction() {
    let fx = Fx::new("quant-redirect");
    fx.file(
        "gb10/modelA/MODEL.toml",
        "[model]\nsupported_quants = [\"nvfp4\"]\n",
    );
    fx.file(
        "gb10/modelB/MODEL.toml",
        "[model]\nkernel_source = \"modelA\"\n",
    );
    assert!(fx.discover("gb10", "modelB", "nvfp4").is_ok());
    assert!(matches!(
        fx.discover("gb10", "modelB", "mxfp4"),
        Err(LayoutError::Manifest { .. })
    ));
}
