# Measurement discipline for this beachhead

These are the rules this branch enforces in `qci-dump`. They follow the
discipline in metrale-inference #126. A number that breaks one of them is not
a result.

- Bit parity is recorded, or the receipt names `PARITY_UNMEASURED`, before any
  tok/s or J/tok comparison. The dump prints `tok_s: withheld until bit parity`
  and `j_per_tok: withheld until bit parity` on every row until that record
  exists.
- tok/s and J/tok are separate fields. A faster run is not an energy win.
- Reference-runtime rows and native Metrale rows are both present. One does
  not stand in for the other.
- Each row carries concurrency, context, precision, and MTP. The runtime commit
  is a real sha or `BLOCKER ENGINE_NOT_BUILT`.
- Disk size is not VRAM. `weights_file_bytes` is a file length. Memory stays
  `BLOCKER VRAM_UNMEASURED` until a serve records it.
- A topology this host did not measure stays a blocker. Passing `--hardware
  strix-halo` does not invent a Strix receipt.
- The receipt names `support_label: withheld`. Model size, a download count,
  or a file that exists is not a support claim.
- An absent weight path and a different absent weight path produce the same
  receipt. The path is not part of the text.
- Contracts from investor-mvp #41, #42, and #46 that this issue requires stay
  `BLOCKER UNFROZEN` until those issues freeze them. This branch does not
  invent the schema.
