# Changelog

All notable changes to `kzx` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/). A change to the compressed bytes of any fixture
(the golden tests) is a **format change** and is called out as such.

## [Unreleased]

## [0.1.0] - 2026-10-22

Initial release. Format version 1.

### Added
- Container format: `kzx` magic, version byte, block count, per-block codec id, model-memory
  parameter and both lengths; streams of independent blocks (`compress_blocks`).
- `cm-generic` codec: order-1/2/3/4/6 and word contexts over one-byte bit-history states with
  learned state maps, a long-range match model, a gated logistic mixer and two SSE stages, coded
  by a 32-bit binary arithmetic coder.
- `cm-ptx` codec: a reversible PTX transform that replaces the digit runs of `%`/`$`
  identifiers with a marker and codes each number inline through an inner delta-stream model,
  plus opcode/field and previous-line column contexts; a "match continues" flag codes a byte
  predicted by a long match as one binary decision.
- `cm-elf` codec: 16-byte record (SASS instruction word) column contexts for cubins and other
  ELF objects.
- Content sniffing (`sniff`): ELF magic, PTX banner or directives, printable text, else generic.
- Bomb-proof decoding: declared output and model memory are checked against caller `Limits`
  before any allocation; every loop is bounded by a declared length; typed `Error`.
- `no_std` + `alloc`, `#![forbid(unsafe_code)]`, no dependencies; builds for
  `wasm32-unknown-unknown`, `wasm32v1-none` and `thumbv7em-none-eabi`.
- Tests: byte-exact round trips over a synthetic corpus, typed negative cases, seeded
  fuzz-style mutation tests, golden byte pins; `examples/bench` with a synthetic public corpus
  and optional CLI baselines; `examples/quickstart`; `examples/file`.
