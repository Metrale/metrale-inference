# Bit parity first, then performance

Iterating on performance is useless without bit parity, on the mock or on the real model: a
faster kernel that computes something else is not a win, and every timing taken before parity
measures an unknown program. So:

- **The improvement loop may not start until bit parity is achieved and recorded** (in the
  campaign's target file log and in the bring-up ledger, `ledger/<class>.toml`).
- **Every loop iteration keeps parity.** A lever that breaks a Tier 1 or Tier 2 check is
  discarded, or kept only behind a flag (default off, disclosed on records) with the Tier 3
  accuracy bar.

Exact identity with another hardware class is generally impossible: tile shapes, split counts
derived from the SM count, MMA atoms and vendor libraries change the order of floating-point
accumulation. Parity is therefore defined in three tiers, and each states what is required
where.

## Tier 1: self-consistency on the target, bit-exact

All on the new device, all byte-for-byte:
- **circuit vs legacy**: `met circuit diff` (decode logits under the legacy forward and the
  circuit forward; eager and graphed; every padded width; with its detection and repeat
  controls);
- **eager vs graphed**: the same request with and without CUDA graphs
  (`METRALE_DEBUG_NO_GRAPH=1`) produces the same bytes;
- **run-to-run determinism**: the same request on a fresh serve twice, identical transcripts and
  identical strict prefill bits;
- **batch-row invariance** where the class promises it (the canonical-tier / row-invariant MoE
  paths): a sequence's output does not depend on which other sequences share its step;
- **mock bytes deterministic**: `met ml-utils mockify` twice (and `met serve --mock`) give the
  same weight bytes and the same digest.

## Tier 2: correctness against a reference

- **Logits vs the reference implementation** (Hugging Face transformers on fixed prompts),
  within the tolerance declared per format (BF16, FP8, NVFP4 W4A16, and an exact-conversion path
  such as E2M1 to E4M3 each carry their own), on the real model.
- **Greedy transcripts vs the certified reference box**: the same prompts, the same recipe, a
  match rate (identical transcripts, and the first divergence position of the rest), always
  beside a **same-box control** (the reference box against itself on a second fresh serve), so a
  near-tie flip is not mistaken for a defect.
- **Where cross-hardware bit-exactness IS achievable, require it.** A kernel that runs the same
  source with the same launch shape and reduction order on both classes, with the same compile
  flags (no FMA contraction differences) and no vendor library call, must produce
  byte-identical outputs to the reference box on a fixed input: a microtest whose output hash is
  compared across the two boxes. This covers most of the memory-bound ops (norms, RoPE,
  residual adds, KV writes, argmax) and any inherited GEMV whose split and tile do not depend on
  the device. It does not cover split-K counts derived from the SM count, class-tuned tiles,
  different MMA atoms or cuBLASLt calls: those get the tolerance and the transcript match rate.

## Tier 3: the accuracy bar

Before certification, on the real model at its declared precision: BFCL with its draw recorded
(N, category sample pct, SHA-256 of the ordered sample ids) and agentic-webserver with a
same-night control. A lever that changes precision needs this bar before it is used in any
published number, and stays behind a flag.

## What "parity achieved" means for the loop

- **Mock**: Tier 1 passing on the mock (circuit vs legacy where the arch has a circuit, eager vs
  graphed, determinism, mock-byte determinism). The mock has no reference logits; it cannot pass
  Tier 2.
- **Real model**: Tier 1 and Tier 2 passing.
- Record each with its date, the commit, the binary's digest and the evidence (logs, hashes, the
  match-rate table) in the target file log, and set `ttbp_mock` / `ttbp_real` in the ledger.

## Metrics: TTBP and TTPV

Measured per hardware + model combination, so later campaigns can learn what was slow:
- **TTBP (time to bit parity)**: campaign start to Tier 1 + Tier 2 passing; recorded separately
  for the mock (Tier 1 only) and for the real model.
- **TTPV (time to performance/energy victory)**: campaign start to the improvement loop's exit
  criterion met on a same-box scoreboard with a PARITY-PASS from the PARITY-O.R.A.C.L.E.

Both live in `ledger/<class>.toml`, beside the iteration count, the levers kept and discarded,
and the lines added and removed by parameterization. Update the ledger at every milestone, and
read every previous ledger at the start of a new campaign to reuse what worked.
