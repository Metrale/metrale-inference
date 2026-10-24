// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-24: The image an NVIDIA kernel compile emits, chosen by
//! `METRALE_KERNEL_IMAGE` (DEV-LOOP-SPEED #5).
//!
//! Owner: metrale-kernels build.
//! Invariants:
//! - `fatbin` (unset means `fatbin`) carries SASS for the declared arch plus the
//!   same PTX, uncompressed: the driver loads the SASS with no JIT, and the
//!   registry still reads the PTX text for its globals and PDL entries.
//! - `ptx` is PTX text only, JIT-compiled by the driver at module load: the
//!   pre-2026-10-24 image, kept as an explicit value, never a fallback.
//! - Any other value fails the build.
//!
//! Included via `#[path = "build_image.rs"] mod build_image;`; no `super::`
//! dependencies, so `tests/kernel_image.rs` compiles the same code.

/// 2026-10-24: The lever. Unset selects [`KernelImage::Fatbin`].
pub(crate) const IMAGE_ENV: &str = "METRALE_KERNEL_IMAGE";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum KernelImage {
    Fatbin,
    Ptx,
}

impl KernelImage {
    /// 2026-10-24: Parse the lever value; `None` is unset.
    pub(crate) fn parse(raw: Option<&str>) -> Result<Self, String> {
        match raw {
            None | Some("fatbin") => Ok(Self::Fatbin),
            Some("ptx") => Ok(Self::Ptx),
            Some(other) => Err(format!(
                "{IMAGE_ENV} must be unset, `fatbin` or `ptx`, got {other:?}"
            )),
        }
    }

    /// 2026-10-24: Read the lever once, for the whole build. Panics on an invalid value.
    pub(crate) fn from_env() -> Self {
        let raw = match std::env::var(IMAGE_ENV) {
            Ok(v) => Some(v),
            Err(std::env::VarError::NotPresent) => None,
            Err(std::env::VarError::NotUnicode(v)) => panic!("{IMAGE_ENV} is not UTF-8: {v:?}"),
        };
        Self::parse(raw.as_deref()).unwrap_or_else(|e| panic!("{e}"))
    }

    pub(crate) fn extension(self) -> &'static str {
        match self {
            Self::Fatbin => "fatbin",
            Self::Ptx => "ptx",
        }
    }

    /// 2026-10-24: The nvcc arguments that select the output and its architectures, for the
    /// HARDWARE.toml `arch` (`sm_121f`, `sm_90a`). `fatbin` embeds SASS for that arch and the
    /// PTX of its virtual arch (`compute_121f`), the PTX uncompressed (`--no-compress`).
    ///
    /// `--enable-extended-smem=true` (an undocumented ptxas switch): offline ptxas caps an
    /// entry's static `__shared__` at 48 KiB (`uses too much shared data (..., 0xc000 max)`),
    /// while the driver JIT that loads the same PTX accepts up to the device limit (99 KiB on
    /// GB10), and ~30 GB10 entries (`attn_prefill*`, `int8_gemm_mmq*`, `w4a16_gemm_t_m128_v3`, ...)
    /// declare more. With the switch, ptxas emits for them what the JIT emits.
    ///
    /// 2026-10-24: Checked by `cuobjdump -sass` against the driver's JIT cache (driver
    /// 580.126.09, whose JIT is ptxas V13.0.97; the toolkit's is V13.0.88): a 64 KiB probe and
    /// 887 of the 889 functions of the holo-3.1-0.8b target are identical; the two
    /// `gated_delta_rule_prefill_split*` functions differ in register allocation and order only
    /// (same instruction count). The SASS now follows the toolkit, not the host's driver.
    pub(crate) fn nvcc_mode_args(self, arch: &str) -> Result<Vec<String>, String> {
        match self {
            Self::Ptx => Ok(vec!["--ptx".into(), format!("-arch={arch}")]),
            Self::Fatbin => {
                let digits = arch.strip_prefix("sm_").ok_or_else(|| {
                    format!("{IMAGE_ENV}=fatbin needs an `sm_*` arch, got {arch:?}")
                })?;
                Ok(vec![
                    "--fatbin".into(),
                    "--no-compress".into(),
                    "-Xptxas".into(),
                    "--enable-extended-smem=true".into(),
                    "-gencode".into(),
                    format!("arch=compute_{digits},code=[sm_{digits},compute_{digits}]"),
                ])
            }
        }
    }

    /// 2026-10-24: The device-code compiler identity for the closure attestation, given the
    /// `nvcc --version` line. A fatbin's SASS comes from this toolkit's ptxas, a PTX image's
    /// from the driver's JIT, so the two images are different compilers; the PTX image keeps
    /// the bare version, the identity every record before this lever carries.
    pub(crate) fn compiler_id(self, nvcc_version: String) -> String {
        match self {
            Self::Ptx => nvcc_version,
            Self::Fatbin => format!("{nvcc_version}; image fatbin"),
        }
    }
}
