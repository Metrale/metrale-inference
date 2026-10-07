# HIP device reporting contract

2026-10-07: The CUDA compatibility shim obtains device names and physical
attributes from HIP instead of reporting a fixed Strix name and capacities.
This change does not establish native HIP execution or model support.

The mapping is deliberately narrow:

| CUDA attribute number | HIP query | Units / meaning |
| --- | --- | --- |
| 1 | `hipDeviceAttributeMaxThreadsPerBlock` | threads |
| 8 | `hipDeviceAttributeMaxSharedMemoryPerBlock` | bytes |
| 10 | `hipDeviceAttributeWarpSize` | threads |
| 16 | `hipDeviceAttributeMultiprocessorCount` | processor count |
| 18 | `hipDeviceAttributeIntegrated` | integrated-device flag |
| 19 | `hipDeviceAttributeCanMapHostMemory` | host mapping capability |
| 36 | `hipDeviceAttributeMemoryClockRate` | kHz |
| 41 | `hipDeviceAttributeUnifiedAddressing` | unified addressing capability |

Names use `hipDeviceGetName`; attributes use `hipDeviceGetAttribute` with
named HIP enums, never casts of CUDA numeric values. HIP errors propagate.
In particular, a HIP implementation that does not support an attribute can
refuse it; the shim does not fabricate a successful zero. Unknown CUDA
attributes return `hipErrorInvalidValue` without writing the output.

Two compatibility policies remain synthetic after validating the requested
device with `hipDeviceGet`:

- 75/76 return CUDA compute capability 12.1, preserving the existing selection
  contract. This is **not** the physical AMD ISA or a compatibility proof.
- 115 returns zero because the shim does not implement CUDA asynchronous
  pool allocation. `vendor/cudarc/src/driver/safe/core.rs` probes this when
  constructing a context; forwarding underlying HIP pool support would enable
  an unsupported allocation path.

The current consumers in `cuda_backend/arch_preflight.rs` query 75/76 and 16;
`gpu_impl_graph.rs` uses 16 for grid sizing; `memory.rs` uses 18 for integrated
memory budgeting. Real values can therefore change scheduling and memory
budgets. An AMD execution gate must check these outcomes before deployment.
No model manifest, kernel arithmetic, or architecture guard changes here.

## Evidence and remaining gate

`python3 scripts/test_hip_device_query.py` extracts exactly one definition of
each production query function, compiles those bodies against independently
authored host call witnesses, and executes them. Two distinct devices and
HIP enum numbers deliberately different from CUDA values catch hardcoded
identity, ignored device IDs, and accidental enum casts. Tests cover all eight
mapped attributes, name truncation, nulls, unknown attributes, propagated
not-initialized/invalid-device/unsupported errors, and synthetic policy device
validation. A compiled wrong-case mutation maps processor count to warp size;
the same checks must reject it. Separate fixed-name and fixed-processor-count
mutations reproduce the original reporting faults and must fail. Missing/duplicate source definitions also
refuse extraction. The Rust integration test runs this host check on Unix.

This tests production forwarding logic, **not** HIP ABI/header compatibility,
the entire shim, actual GPU properties, native WSL execution, or performance.
The next gate is compiling against the installed HIP SDK and comparing both
query surfaces on the actual device, including an invalid ordinal and driver
error behavior. Do not infer a physical GPU architecture from synthetic
CUDA capability 12.1.

Implementation inputs are this repository and the official
[HIP 6.4.2 API declarations and attribute definitions](https://rocm.docs.amd.com/projects/HIP/en/docs-6.4.2/doxygen/html/hip__runtime__api_8h_source.html).
The tests were authored for this contract; no external reference implementation
was copied.
