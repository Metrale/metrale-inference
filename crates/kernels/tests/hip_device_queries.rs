// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: Host-compiled production query bodies; no HIP runtime claim.
#![cfg(unix)]

#[test]
fn physical_queries_forward_and_compatibility_policy_refuses_bad_devices() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(std::path::Path::parent)
        .expect("workspace root");
    let output = std::process::Command::new("python3")
        .arg(root.join("scripts/test_hip_device_query.py"))
        .current_dir(root)
        .output()
        .expect("python3 and a host C++17 compiler are required");
    assert!(
        output.status.success(),
        "host shim contract failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
