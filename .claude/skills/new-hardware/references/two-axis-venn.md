# The two-axis Venn

A new hardware + model combination differs from what the engine already serves along two
axes at once. Classify every op along both before writing a kernel; most of the work turns out
to be on one axis only.

## Axis 1: the class against its nearest supported class

The nearest class is usually its `inherits` parent (Hopper inherits GB10). The questions:
- Which kernels does the class compile (its own, plus inherited), and which are absent (guarded
  out by `[build] extra_nvcc_flags`, or `[expected_absent]` in a MODEL.toml)? `met circuit plan`
  lists "Rule kernels this device cannot run" per model.
- Which formats does the device run natively? (`DEVICES.toml` `native_mma`; the plan's
  "Declared formats on this device" table says how each declared pair executes: native, an
  exact conversion such as E2M1 to E4M3 on the FP8 MMA, or no path.)
- Does the planner see the class's own kernels? A kernel the class compiles that no
  `FUSIONS.toml` rule names never appears in a plan, so the plans, the tensor-core policy and
  the Venn all describe the parent class's routing. Model the class's own kernels in its
  `common/FUSIONS.toml` and `common/KERNEL_FAMILIES.toml` overlays (rules override by id; a
  family override replaces by id) before reading any hardware-axis report as the truth.

Commands:
```
met circuit plan --checkpoint <id> --hardware <device> --precision declared   # report: gaps by class
met circuit plan --checkpoint <id> --hardware <device> --precision declared \
    --format plan --mode multi_seq --rows 16                                     # one plan as text
met circuit show --recipe <recipe> --hardware <device> --mode multi_seq --rows 16
met circuit venn --target <recipe> --against <recipe>[,<recipe>] --hardware <device> \
    --out kernels/circuits/venn/<target>-vs-<against>--<device>.md
met circuit precision --checkpoint <id> --hardware <device> --node '<glob>'
```

With `--hardware`, the Venn plans every compared model with the device class's rules and the
kernels it can run, costs nodes with the device's roofline, and counts only the class's own
microbench evidence: on a class with no records, everything shared is "shared, unmeasured".

## Axis 2: the model against its nearest supported model

Exactly `/new-model` steps 1-3 (read the checkpoint, choose the comparison set, write the
circuit and run the Venn), with `--hardware <device>` on every plan. A model the nearest class
already serves needs no model-axis work beyond its precision table on this device.

## Classify every op

Run both axes at C1, C16 and C128 (and the MTP verify width where the model speculates). Rank
the ops by estimated step share (roofline first, nsys as soon as a serve runs) and give each
one class:

| Class | Meaning on a new hardware class | Action |
|---|---|---|
| **shared-measured** | the class runs a kernel for the op at this parameter point AND has its own microbench record there | reuse |
| **shared-unmeasured** | the kernel runs here (inherited, or the class's own) but no record exists on this class | reuse, then microbench; an inherited kernel is a candidate for retuning, not proof of speed |
| **parameterize** | the family exists; the point (head_dim, group size, tile, format, rows) is missing here or exists only as a copy | parameterize, behind the stability gate |
| **policy variant** | only a policy differs (weight format, scale layout, activation quantization, epilogue, routing) | a policy of an existing template, or a split |
| **novel** | no family runs the op at this format and row count on this class (including a format the device has no MMA for) | build it, last, and only after the four above |

Then rank the work: the largest step share first, and within a share the cheapest class first
(shared-unmeasured before parameterize before policy variant before novel). Record the ranked
table in the campaign PR and update it as records arrive.

## What a gap on axis 1 usually is

- An inherited **CUDA-core** kernel where the device has a faster tensor-core path (Hopper:
  wgmma, FP8 MMA): a `[tensor_core_policy]` backlog entry, closed by a parameterization of the
  tensor-core family (a new MMA atom as a policy) rather than a new kernel.
- A **format with no native MMA**: the exact conversion path, planned and costed as such, never
  a silent upcast and never a silent precision drop.
- A **rule threshold tuned to the parent's SM count**: a parameterization (hardware facts as
  data), not a class-specific rule copy.
- A **planner gap**: the op has no rule on this class at some row count (the plan shows "no rule
  of this class covers it"). The runtime may still run something; find out what (nsys), then
  write the rule with its routing citation.
