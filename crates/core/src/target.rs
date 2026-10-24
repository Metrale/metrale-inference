// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Kernel target descriptor: the `(arch, model, quant)` key of a compiled kernel set.
//!
//! 2026-10-22: Defined in the no_std `metrale-kernel-target` (so the blueprint body carries the
//! same type) and re-exported here at its old path.
//!
//! Owner: metrale-core.
//! Invariants: none beyond the types.

pub use metrale_kernel_target::KernelTarget;
