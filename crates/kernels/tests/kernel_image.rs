// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-24: Tests `build_image.rs`: `METRALE_KERNEL_IMAGE` selects a fatbin (unset or
//! `fatbin`) or PTX only (`ptx`), refuses anything else, and maps to the nvcc arguments that
//! produce it.
//!
//! Owner: metrale-kernels tests.
//! Invariants: none beyond the types.

#[allow(dead_code)] // 2026-10-24: `from_env` reads the process environment; the build script calls it.
#[path = "../build_image.rs"]
mod build_image;

use build_image::KernelImage;

#[test]
fn unset_and_fatbin_select_the_fatbin_and_ptx_stays_selectable() {
    assert_eq!(KernelImage::parse(None), Ok(KernelImage::Fatbin));
    assert_eq!(KernelImage::parse(Some("fatbin")), Ok(KernelImage::Fatbin));
    assert_eq!(KernelImage::parse(Some("ptx")), Ok(KernelImage::Ptx));
}

#[test]
fn any_other_value_fails_instead_of_falling_back() {
    for bad in ["", "PTX", "Fatbin", "cubin", "0", "1", " ptx"] {
        let err = KernelImage::parse(Some(bad)).unwrap_err();
        assert!(err.contains("METRALE_KERNEL_IMAGE"), "{err}");
        assert!(err.contains(&format!("{bad:?}")), "{err}");
    }
}

#[test]
fn the_fatbin_carries_sass_and_uncompressed_ptx_of_the_declared_arch() {
    assert_eq!(
        KernelImage::Fatbin.nvcc_mode_args("sm_121f").unwrap(),
        [
            "--fatbin",
            "--no-compress",
            "-Xptxas",
            "--enable-extended-smem=true",
            "-gencode",
            "arch=compute_121f,code=[sm_121f,compute_121f]"
        ]
    );
    assert_eq!(
        KernelImage::Fatbin.nvcc_mode_args("sm_90a").unwrap(),
        [
            "--fatbin",
            "--no-compress",
            "-Xptxas",
            "--enable-extended-smem=true",
            "-gencode",
            "arch=compute_90a,code=[sm_90a,compute_90a]"
        ]
    );
    assert!(KernelImage::Fatbin.nvcc_mode_args("gfx1151").is_err());
}

#[test]
fn the_ptx_image_keeps_the_pre_lever_argv_and_compiler_identity() {
    assert_eq!(
        KernelImage::Ptx.nvcc_mode_args("sm_121f").unwrap(),
        ["--ptx", "-arch=sm_121f"]
    );
    let v = "Cuda compilation tools, release 13.0, V13.0.88".to_string();
    assert_eq!(KernelImage::Ptx.compiler_id(v.clone()), v);
    assert_ne!(KernelImage::Fatbin.compiler_id(v.clone()), v);
}

#[test]
fn each_image_names_its_files_by_its_own_extension() {
    assert_eq!(KernelImage::Fatbin.extension(), "fatbin");
    assert_eq!(KernelImage::Ptx.extension(), "ptx");
}
