// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-09: The DFlash drafter's row-scaled FP8 GEMMs (fp8_gemm_t_row_scaled, _p4, _k64, _m16)
// for the GLM-5.3 Flash target, in their own module, `dflash_fp8_gemm`. They ship in the
// qwen3.6-27b leaf's w4a16 shadow; this target's `w4a16` module stays common's, so its own
// w4a16 dispatch is unchanged, and the drafter falls back to this module when `w4a16` lacks
// them (dflash_head/from_weights/kernel_handles.rs). The source is that leaf's file, included
// whole so the kernels are compiled from one copy; the build passes this file's layer directory
// as -I (CompileJob::include_dir), so the path resolves as kquant_moe.cu's vendored include does.
// The module also carries the leaf's other w4a16 entry points; nothing looks them up here.

#include "../../../gb10/qwen3.6-27b/nvfp4/w4a16_gemm.cu"
