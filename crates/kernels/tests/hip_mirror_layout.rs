// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: The HIP mirror keeps the stage layout, so a leaf source's
//! `#include "../../common/x.cu"` reaches the mirrored, mask-widened common/
//! exactly as it reaches the staged one (build_stage.rs).
//!
//! Owner: metrale-kernels tests.
//! Usage: `METRALE_SKIP_BUILD=1 cargo test -p metrale-kernels --test hip_mirror_layout`.

#![allow(dead_code)]

/// 2026-10-07: Stand-in for build.rs's `content_hash`, which build_hip.rs
/// imports from its parent module. Only distinctness matters here.
fn content_hash(s: &str) -> String {
    format!(
        "h{:016x}",
        s.bytes()
            .fold(0u64, |h, b| h.wrapping_mul(31).wrapping_add(b as u64))
    )
}

#[path = "../build_hip.rs"]
mod build_hip;

use std::path::Path;

fn write(path: &Path, text: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

#[test]
fn leaf_common_include_resolves_in_the_mirror() {
    let tmp = std::env::temp_dir().join(format!("metrale-hip-mirror-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    let stage = tmp.join("stage");
    let mirror = tmp.join("hip_mirror");
    write(
        &stage.join("common/attn.cu"),
        "__global__ void k(float* v){ v[0] = __shfl_xor_sync(0xffffffff, v[0], 1); }\n",
    );
    write(&stage.join("common/rms.cu"), "__global__ void r(){}\n");
    let wrapper = stage.join("gpt-oss-20b/mxfp4/attn.cu");
    write(
        &wrapper,
        "#define HDIM 64\n#include \"../../common/attn.cu\"\n",
    );

    // 2026-10-07: The leaf is mirrored before any common source, the order
    // that broke the flat mirror: the include must still resolve.
    let compiled = build_hip::hip_mirror_source(&wrapper, &mirror, &stage, "cu");
    assert_eq!(compiled, mirror.join("stage/gpt-oss-20b/mxfp4/attn.cu"));
    let included = compiled.parent().unwrap().join("../../common/attn.cu");
    let text = std::fs::read_to_string(&included)
        .unwrap_or_else(|e| panic!("leaf include {} unresolved: {e}", included.display()));
    assert!(text.contains("__shfl_xor_sync(0xffffffffULL"), "{text}");

    let common = build_hip::hip_mirror_source(&stage.join("common/rms.cu"), &mirror, &stage, "cu");
    assert_eq!(common, mirror.join("stage/common/rms.cu"));

    // 2026-10-07: A source outside the stage keeps a hashed subdir.
    let loose = tmp.join("elsewhere/x.cu");
    write(&loose, "__global__ void x(){}\n");
    let out = build_hip::hip_mirror_source(&loose, &mirror, &stage, "cu");
    assert!(out.starts_with(&mirror) && !out.starts_with(mirror.join("stage")));
    assert!(out.exists());
    let _ = std::fs::remove_dir_all(&tmp);
}
