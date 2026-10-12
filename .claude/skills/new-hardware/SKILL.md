# QCI hardware and model beachhead

This file is the index for the Gemma 4 26B-A4B and Nemotron 3.5 Lightning
beachhead on this branch. The method it follows is the Hardware Beachhead
Campaign written in
[Metrale/metrale-inference#126](https://github.com/Metrale/metrale-inference/pull/126).
That pull request is still open. This branch does not vendor it, and it does
not copy another class's measured limits.

Read, in order:

1. `targets/qci-gemma-nemotron.md` — the audit, the pins, the checklist.
2. `ledger/qci-gemma-nemotron.toml` — one row per hardware and model. Empty
   milestone fields are unmeasured.
3. `references/measurement-discipline.md` — the rules the dump enforces.
4. `docs/qci/gemma-nemotron-working-log.md` — commands, pins, and blockers.

Bit parity is recorded before any tok/s or J/tok number. Those two metrics stay
separate. A missing artifact or an unmeasured topology is a named blocker. A
substitution is a recorded product decision. RTX 3090 numbers supplement Strix
acceptance. They are not that acceptance. Nemotron 3.5 Lightning does not block
the four-model P0 exit and is not the Nemotron-3-Nano benchmark.

The dump entry is `qci-dump` (`crates/qci-dump`). The chat path for the two
pinned model ids is `metrale_qci_dump::gate`, called from
`crates/server/src/api/chat/mod.rs`. Video decoding stays outside that path.
