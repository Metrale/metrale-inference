# kzx design: models, format, and how it plugs into `.kbp`

`kzx` is a context-mixing compressor for the three kinds of bytes a Metrale blueprint
(`.kbp`) carries: compiled kernels as PTX text, compiled kernels as ELF objects (cubin/SASS,
HIP code objects), and the postcard-encoded signed body. It is written from scratch in
`no_std` Rust with no dependencies and no `unsafe`, so the same code verifies and unpacks a
blueprint in `met`, in CI and in the browser (the store runs the verifier on `wasm32`).

The design goal is ratio, not speed: a blueprint is decoded once at load, so a few MB/s is
enough, while every byte saved is a byte less to sign, mirror, fetch and paste. Where a general
LZ coder sees PTX as text with a lot of near-repeats, kzx sees instruction skeletons whose only
variation is register numbering, and models the two separately.

## 1. Model architecture

```
                          ┌──────────── per block (one Predictor) ────────────┐
 input bytes ─► transform ─► skeleton byte ─► LZP flag? ──hit──► byte emitted  │
 (PTX only)    (markers,        │              │miss                           │
                numbers out)    │              ▼                               │
                                │   8 binary decisions, each:                  │
                                │   contexts ─► buckets ─► bit-history state   │
                                │   ─► StateMap p ─► stretch ─► Mixer ─► APM×2 │
                                │   ─► arithmetic coder                        │
                                │                                              │
                                └─► marker? ─► NumberModel (inner Predictor    │
                                               over the delta byte stream)     │
                                └──────────────────────────────────────────────┘
```

**Contexts** (`context.rs`). Every byte boundary emits up to ten 32-bit hashes. All profiles
share orders 1, 2, 3, 4 and 6 and a word context (hash of the current alphanumeric token and
the previous one). `cm-generic` adds two sparse contexts (bytes −2..−3; the high nibbles of
bytes −1..−6) that catch the structure of varint streams. `cm-ptx` adds a token-in-field context
(hash of the line's opcode, the operand index and the token prefix) and a column context (the
same field of the previous line). `cm-elf` adds 16-byte record contexts for SASS instruction
words: (column, this word's opcode bytes, order-1) and (column, bytes at −16 and −32).

**Statistics** (`slots.rs`, `states.rs`). Each context hash selects a 16-byte bucket: one
check byte and 15 one-byte bit-history states, one per node of the binary tree over a nibble,
so a byte costs two bucket lookups per context. Buckets are probed three deep inside one cache
line and the least-used is recycled. A state is a bounded pair of counts `(n0, n1)` where
growing one count discounts the other (nonstationary), generated once by rules in `states.rs`.
A `StateMap` per context learns the state → probability mapping online (22-bit probability,
count-scaled step). Order-1 and order-2 tables are sized to their small context spaces; the
rest take `2^mem_log2` buckets.

**Match model** (`matcher.rs`). A hash of the last `min_len` bytes (7–10, per profile) finds
the most recent earlier occurrence, verified backwards up to 48 bytes; while the following bytes
agree, the byte after the match is predicted bit by bit with a confidence learned per match
length. The window is the whole block, which is what captures duplication across kernels in a
solid payload.

**Mixing** (`mixer.rs`). Inputs are the stretched state-map probabilities, one deterministic-
context input per context (the sign and strength of a run of identical bits), two match-model
inputs, one exclusion input (after a failed LZP flag the predicted byte is known wrong) and a
bias. A gated linear mixer in the logistic domain holds 4096 weight sets selected by the partial
byte and the match-length bucket, trained online by gradient descent on coding cost. Two APM
(SSE) stages refine the result under the partial byte and under order-1.

**LZP flag** (`engine.rs`, `predictor.rs`). When the match model has followed a match for at
least `lzp_min` bytes (32 for PTX, 24 otherwise), the next byte is first coded as one binary
decision "the match continues", under contexts of match length, recent flag history and the
expected byte. A hit emits the byte for well under a bit; a miss falls back to the eight-bit
path with the exclusion input set. This is where the speed comes from on PTX: after the
transform the skeleton is mostly long exact repeats.

**PTX transform** (`ptx.rs`). PTX bodies repeat almost exactly except for the numbers inside
register and label names (`%r123`, `%f4`, `$L__BB6_219`): sequential allocation makes every
unrolled line differ from its twin by a few digits, which breaks byte matches and dilutes every
context. The encoder replaces each canonical digit run inside a `%`/`$` identifier (maximal,
no leading zero, ≤ 18 digits, identifier ≤ 96 bytes back to its sigil) with the marker byte
`0x01`; raw `0x01`/`0x02` bytes are escaped with `0x02`, so the transform is exact for arbitrary
input and needs no side table: the decoder re-inserts a number wherever it decodes a marker.
The number is coded right after the marker as a zigzag varint of its delta against the previous
number of the same identifier class (hash of the identifier bytes before the digits), through
an **inner** `cm-generic` predictor whose history is the delta byte stream itself. Before each
delta the inner history receives two bytes neither side has to code (both know them): a class
tag and a hash of the skeleton line, so the inner orders see "which register class, on which
instruction shape" and the inner match model finds the repeating delta sequences of unrolled
loops. Measured on the GB10 set the transform took the 3 MB sample from 157,973 to 74,794 bytes
(xz −9e: 210,244) and raised throughput from 1.2 to ~8 MB/s, because 60–90 % of skeleton bytes
are then LZP hits.

**Arithmetic coder** (`coder.rs`). A carry-less 32-bit binary coder with 12-bit probabilities
(the lpaq coder). The decoder reads zeros past the end of its input: a truncated or corrupt
stream yields wrong bytes of the declared length, never an out-of-bounds read or a hang.

**Determinism.** Every operation is integer; `usize` only indexes tables whose sizes are
powers of two below 2^31; the state table, the stretch table and all hashes are computed
identically on every target. The golden tests pin the bytes.

## 2. Format

```
stream  = "kzx" u8:version=1  u32le:block_count  block*
block   = u8:codec  u8:mem_log2  u16le:reserved=0  u64le:raw_len  u64le:packed_len  bytes[packed_len]
codec   = 0 stored | 1 cm-generic | 2 cm-ptx | 3 cm-elf
```

- Blocks are independent (own model state, own window). A stream of N blocks is how a
  container compresses N sections or N artifacts in one object (`compress_blocks`).
- `mem_log2` (10..=26) sizes the high-order context tables (`2^mem_log2` 16-byte buckets each)
  and the match table; `memory_bytes(codec, mem_log2)` is the exact decoder allocation. The
  encoder picks it from the input length (`default_mem_log2`: tables about as large as the
  input, capped at 2^22 buckets).
- `raw_len` is the declared output; `packed_len` tiles the stream exactly (no trailing bytes).
- Inputs shorter than 24 bytes are stored.
- Version 1 is frozen by the golden tests; a model change that alters bytes is a format
  change and bumps the version byte so old readers refuse new streams.

## 3. Safety on untrusted input

Decompression may run before the payload hash is checked, so the bounds are hard:

| Hazard | Control |
|---|---|
| Output bomb | `inspect` sums `raw_len` over all blocks with checked arithmetic and refuses `> Limits::max_output` before any allocation; the PTX decode loop stops at `raw_len` even if a corrupt marker would emit more digits. |
| Memory bomb | `mem_log2` outside `10..=26` is refused; `memory_bytes` must be `≤ Limits::max_memory` per block; the PTX skeleton buffer is at most `2 × raw_len`, the inner delta history at most `11 × (markers)`, both bounded by `raw_len`. |
| Hang | Every loop is bounded by a header field: `MAX_BLOCKS` header reads, `raw_len` byte iterations, ≤ 10 varint bytes per number, ≤ 48-byte match verification, 3 bucket probes. The coder never waits for input. |
| Panic | No indexing without a power-of-two mask or a bounds check, no `unwrap` on input-derived values, no arithmetic that can overflow outside wrapping ops; `tests/fuzz.rs` drives random, mutated and forged streams with a fixed seed; `#![forbid(unsafe_code)]`. |
| Wrong bytes | Not a kzx concern: the container hashes the raw bytes (`ArtifactId`, `body_sha256`); a corrupt stream decodes to the declared length and fails that check. |

Errors are a closed `enum Error` with the offending values (`OutputCapExceeded { declared,
cap }` …). The crate has no dependencies, so it cannot name Metrale's `ErrorCode`; the adapter
maps each variant to a registry code at the edge (§4.4).

## 4. Plugging into `.kbp`

The Phase-1 container (`metrale-blueprint`, `codec.rs`) has one compression seam: `Codec` with
`compress(raw) -> Vec<u8>` and `decompress(what, stored, raw_len, cap)`, a one-byte id per
section, and today's ids `0 = stored`, `1 = deflate`. kzx slots in as `2 = kzx`:

```rust
Codec::Kzx => kzx::compress(raw),                       // sniffs PTX / ELF / generic per section
Codec::Kzx => kzx::decompress(stored, &kzx::Limits { max_output: cap, max_memory: KZX_MEM_CAP })
```
`decompress` returns the declared length or a typed refusal; the container then checks the
length equals `raw_len` and the hash equals the pinned id, exactly as for deflate.

### 4.1 Per payload or as a whole?

Compress **per section, and the payload as one solid block**:

- The **body** is its own block (`cm-generic`, ~90 KB from ~830 KB for a GLM-scale body (45 layers, 843 levers, ~950 kernel entries)). It must
  stay separately addressable: the signed bytes are the raw body, and lite ↔ full conversion
  rewrites the other sections without touching it.
- The **payload** (all inline artifacts) should be one block, not one block per artifact: the
  match model's window is the block, and cross-module duplication (shared prologues, template
  instantiations, the same device functions in many modules) is only exploitable inside one
  window. The locator table already gives each artifact its offset in the *decompressed*
  payload, so per-artifact addressing is unchanged. Sniffing is per block, so a payload that
  mixes PTX and cubins should be grouped by kind (two blocks) or the container should pass the
  kind explicitly (`compress_with(.., Profile::Ptx, mem)`), which is also what makes the choice
  explicit rather than heuristic (PCND). The benchmark table reports both `per-file` and
  `solid` so the cost of per-artifact blocks is visible.
- The **locator** and **signature** sections are small and mostly random (signatures, hashes):
  store them.
- Whole-file compression (one block over header + body + locator + payload + sig) is what the
  baselines do in the benchmark; it is *not* recommended for the container because the header
  and signature must stay readable before any decode (DESIGN.md §4, verification order), and
  the body hash must be checkable before the payload is touched.

Decode time: Metrale's full GB10 kernel set (254 PTX modules, 71 MB) decodes in ~7 s single-threaded on
the GB10 Grace CPU as one solid block. If that budget tightens, cut the payload into a few
blocks (e.g. 4 × 18 MB) and decode them on separate threads; blocks are independent, so the
loader can parallelise without a format change, at a small ratio cost.

### 4.2 Signature and hash

Nothing kzx does touches identity:

- `ArtifactId = sha256(raw module bytes)`, `body_sha256 = sha256(raw body)`,
  `BlueprintId = sha256("kbp-body\0" ‖ raw body)`: all over **uncompressed** bytes. A blueprint
  re-encoded with another codec keeps every id and its signature.
- The signed message is `"kbp-sig-v1\0" ‖ KBP_FORMAT ‖ body_len ‖ body_sha256 ‖ body` with the
  raw body; the codec id of the body section is **not** signed, so it can be re-encoded (the
  attacker gains nothing: a wrong decode fails `body_sha256`).
- Because the hash check comes *after* decompression, the decoder is the only thing that runs on
  unverified bytes; §3 is what makes that acceptable. The container should pass `cap =
  body_len` from the header (hard refusal at 4 MiB body, 1 GiB full, already in the design) and
  a memory cap chosen by the host (`KZX_MEM_CAP`, e.g. 1 GiB on a server, 64 MiB in the store's
  wasm verifier, which only ever decodes the lite body).

### 4.3 Lite ↔ full identity

DESIGN.md's rule: lite and full forms of one build share `BlueprintId` and one signature,
because only the unsigned locator and payload sections differ. Compression preserves this
trivially as long as the body block is encoded identically in both forms, which holds because
kzx is deterministic: the same raw body gives the same compressed bytes, so even the *compressed*
body bytes of lite and full are identical, and `met blueprint lite|full` can copy the body
section verbatim without decoding it. The locator's `Inline { offset }` fields refer to offsets
in the decompressed payload and are unchanged by the codec.

### 4.4 Errors

Map at the adapter (the crate itself stays dependency-free):

| `kzx::Error` | registry code (range `0x0600–0x06FF` blueprint, or a new codec sub-block) |
|---|---|
| `BadMagic`, `UnsupportedVersion`, `UnknownCodec`, `ReservedNonZero`, `TrailingBytes`, `Truncated`, `StoredLengthMismatch`, `TooManyBlocks`, `LengthOverflow` | `BlueprintFrame` (malformed section) |
| `OutputCapExceeded`, `MemoryCapExceeded`, `BadMemoryParameter` | `BlueprintTooLarge` (refused before work) |
| `InvalidParameter` | never reaches the adapter (encoder-side programming error) |

### 4.5 Transport

`kbp1.` + padded standard base64 of the file bytes is unchanged. The lite form is what gets
pasted; its size is header + body block + locator + sig, so the body ratio sets the transport
size: a GLM-scale body of ~830 KB becomes ~90 KB compressed → ~120 KB of base64, inside the
256 KB lite target (deflate would give ~170 KB → ~230 KB).

### 4.6 Consuming the crate

`kzx` lives in its own repository (`github.com/tbraun96/kzx`, dual MIT/Apache-2.0). Metrale can
take it as a path dependency while it sits in `crates/kzx`, or as a git/crates.io dependency
once published, declared once in the workspace table (`kzx = { version = "0.1", default-features
= false }`) and added to `PURE_ALLOWLIST` with the reason "no_std, no deps, builds for both
wasm32 targets". It has no features, so there is nothing to name on the member side.

## 5. What was evaluated and rejected

- **Byte-level LZP without the PTX transform**: hit rate 93 % at ≥ 24-byte matches, 38 % of
  bytes flagged, 6 % worse ratio than the plain CM: register numbers break every long match.
- **Direct-probability counters instead of bit-history states**: 15–20 % worse at equal memory;
  states are 4× denser and capture determinism.
- **Mixer selected by partial byte only**: 3.5 % worse than selecting by match length as well.
- **Structured per-field number coder (flags "same delta as last time in this line/class",
  then explicit bit length and mantissa)**: 5.9 bits per number; replaced by the inner byte CM
  over the delta stream at ~2 bits per number, because the sequence of deltas across unrolled
  instructions is itself the strongest context.
- **A second ELF record context on order 2** instead of order 6: worse.
- **Larger inner model memory** (`2^18`): no measurable gain.
