# Parameterization: the standing objective

Every bring-up session carries one objective beside "make it run and make it fast": **leave the
tree with less code per supported (hardware, model) pair than it found.** A new class is when
this is cheapest, because the second value of every hidden constant is now in view. Treat each
of the patterns below as a search you run at the start of the campaign and again whenever a
kernel is touched.

## 1. Hardware facts are data

A fact about the device belongs in exactly one place, and code reads it from there:
- **The SKU** (SM count, shared memory per SM, L2, memory size and bandwidth, the tensor-core
  formats it runs natively, cluster / TMA / wgmma / TMEM availability, peaks) lives in
  `kernels/DEVICES.toml`, or is probed at boot from the driver. Several SKUs share one class
  (H100 SXM, H100 NVL and H200 all build `hopper`), so a class file cannot hold a SKU fact.
- **The class** (arch, inheritance, build flags, serving defaults, policies, limits) lives in
  `kernels/<class>/HARDWARE.toml`.
- **Search for:** a number in a kernel, a rule or a `[defaults]` row that is really a device
  fact: an SM count baked into a launch floor, a 96-row threshold that is "two CTAs per SM on a
  48-SM part", a GB10 bandwidth in a roofline formula, a tile chosen for one shared-memory size.
  Replace it with the fact and a derivation (`MIN_CTAS_PER_SM * sm_count`, `clamp(ceil(2 *
  sm_count / heads), 1, 16)` are the existing good examples), and the class-specific copy
  disappears.
- **Known duplicates to remove** (verify they still exist): `[hardware] sm_count` and the
  `memory_*` keys of a class's HARDWARE.toml repeat DEVICES.toml (the `memory_*` keys have no
  reader at all, per `docs/HARDWARE.md`); the roofline floors in `KERNEL-PERF.md` and
  `docs/kernel-perf/measurements.toml` are stated in GB10 constants, so a new class needs its
  peaks taken from DEVICES.toml before its "% of floor" means anything.

## 2. Template and policy parameters, not duplicated kernels

- **Compile-time template parameters** for what sizes registers, shared memory or unrolling:
  head_dim, group size, tile shape, rows per CTA. Instantiate only the points the union of
  MODEL.toml values needs; build.rs emits the instantiations and the lookup names encode the
  point. Example in the tree: the paged decode attention exists as one file per head_dim and KV
  format (`paged_decode_attn_*_128.cu`, `*_512.cu`, plus per-model `*_512.cu` copies), more
  than twenty files that differ in constants.
- **Policy templates** (the WxAy engine, `kernels/gb10/common/wxay_engine.cuh`: load_w /
  prep_w / mma / store) for weight format, scale layout, activation quantization, epilogue and
  routing scoring. BF16 is a W16A16 policy of the same engine, not a separate GEMV family.
- **Class tuning as a policy.** A class that "replaces a gb10 file of the same name" (a
  `[shadow]` entry) usually changes a tile, a vector width or a pipeline depth. Make that a
  tuning policy the class selects in data, and the shadow copy goes away. A class's own
  addition that is the parent's math with a different launch shape is the same pattern.
- **Runtime arguments** for anything that does not size registers or shared memory (strides,
  counts, eps, scale pointers, flags that do not change the inner loop). A difference in a
  runtime parameter is never a reason for a second kernel.
- **Split** when a parameter would change the algorithm, explode the instantiation count, or
  regress an existing point; let the auto-fuser recombine the pieces with `bit_identical`
  rules, each with a microtest.

## 3. The stability gate

Every parameterization of a kernel an existing (class, model) already runs must pass all three
before it merges:
1. **Byte-identical output at every existing point**: a microtest comparing old and new on
   adversarial inputs over the full row range, on the class that runs it today.
2. **No microbench regression at any existing point**: interleaved reps, same box, within the
   noise band.
3. **Existing gates unchanged**: greedy byte-identity on the existing recipes, and their
   certified gates at the next campaign.

Only then does the new point get its own microbench and tuning. If the existing point lives on a
class you cannot run (the parameterization touches GB10 code and you are on an H100), the PR
says so and asks for the gate on that class; it does not merge on the new class's evidence.

## 4. The code-deleted tally

Every bring-up PR body carries:

```
Code deleted: <n> lines in <f> files   Code added: <n> lines   Net: <+/-n>
Parameterizations: <kernel or table> -> <parameter> (<points before> -> <points after>)
Copies removed: <file> (now <template>::<point>)
```

A PR that adds a per-class or per-model copy of a kernel says which parameterization it
declined and why (it changes the algorithm, it regresses a point, the compile time explodes).
