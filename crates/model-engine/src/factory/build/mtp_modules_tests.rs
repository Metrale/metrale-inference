// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The MTP-head ignore probe asks module paths, so an exact module entry
//! (`mtp.fc`, compressed-tensors) and a `re:` entry name the head, as a glob (`mtp*`) does.

use metrale_model_layers::quant_format::{CompressedTensorsFormat, ModeloptFormat};

use super::mtp_head_ignored;

fn ct(entries: &[&str]) -> CompressedTensorsFormat {
    let e: Vec<String> = entries.iter().map(|s| s.to_string()).collect();
    CompressedTensorsFormat::new(String::new(), &e).expect("ignore list")
}

#[test]
fn an_exact_or_re_mtp_entry_marks_the_head_unquantized() {
    // 2026-09-30: Sehyo/Qwen3.5-35B-A3B-NVFP4 lists `mtp.fc` and a `re:` for the layers.
    assert!(mtp_head_ignored(&ct(&["mtp.fc"])));
    assert!(mtp_head_ignored(&ct(&[r"re:mtp\.layers\.\d+\."])));
    // 2026-09-30: unsloth/Qwen3.8-27B-NVFP4.
    assert!(mtp_head_ignored(&ct(&["re:^mtp.*"])));
    let glob = ModeloptFormat::new(String::new(), &["mtp*".to_string()]).unwrap();
    assert!(mtp_head_ignored(&glob));
    assert!(!mtp_head_ignored(&ct(&[
        "lm_head",
        r"re:model\.visual\..*"
    ])));
}
