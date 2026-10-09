# The Latent Kernel Blueprint (LKB)

This page moved to the book: **`book/src/architecture/lkb.md`** (The Latent Kernel Blueprint),
with the circuit compiler's vocabulary in `book/src/architecture/circuit-compiler.md` and the
architecture side in `book/src/architecture/lab.md`. The book is the single copy; this stub keeps
the skill's links working.

For a campaign, the parts to apply:
- every new kernel is a point of a family or a named LKB residual entry (never a silent copy);
- read LKB coverage, the residual (with copy points and the single-class bucket) and the ledger
  fields from `met circuit lkb --checkpoint <id> --hardware <device> --precision <tier>
  --format toml`;
- every exit report states "Promoted into the LKB", "Residual delta", "Promoted into the LAB"
  and "LAB residual delta".
