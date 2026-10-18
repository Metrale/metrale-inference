# Decoupling principles

Adopted 2026-10-17. SBIO is one kind of decoupling;
these fifteen principles name the others, so a
violation can be called out by name and refused by a gate. Seven further proposals were folded into existing
principles as clauses (see the end).

Each principle has: the **rule**, the **violation response** (what to do when you find one), and the
**enforcement** (the mechanical check that stops a regression). A principle without a gate is a wish: when you
introduce a principle into a codebase, land its gate with a shrinking allowlist in the same change.

Crate/module **roles** used below: `core` (pure logic), `abstract-io` (port traits), `host-*` (port
implementations: mem, std, worker), `device` (vendor/FFI backends), `bin` (composition root), `tool`.

---

## ROTI: Resolve Once, Then Immutable
**Rule.** Every decision input (setting, flag, env value, checkpoint field, probed capability) is resolved once,
at a named phase, into a typed immutable value owned by its consumer. No static is written after startup; no
value is derived a second time somewhere else. Core code holds no mutable `static`/`OnceLock`/`thread_local!`;
accumulators belong to the owning value and are published outward.
**Violation response.** Delete the second resolver; thread the resolved value; turn the static into an owned field.
**Enforcement.** An env-ban gate (env reads only in the host adapter); a `statics` scan with a ratcheted
allowlist; `publish_*`/`set_*_from_cli` style setters banned outside role `bin`; one construction site per
resolved config type.
**Folds in:** NGIC (No Globals In Core).

## VBOP: Vendor Behind One Port
**Rule.** Vendor APIs (CUDA, cuBLAS, CUTLASS, NCCL, NVML, Metal, HIP, …) are named only in role-`device`
crates. Everyone else sees a backend trait whose surface names no vendor concept. The device backends (GPU,
comm, weight store) are the only I/O that is not an abstract-io port, and no pure crate depends on a `device`
crate.
**Violation response.** Move the binding behind a backend-trait method, or delete it.
**Enforcement.** depgraph: vendor crates (cudarc, nvml, metal, libc, memmap2) only under role `device`, and a
`core → device` edge is refused; a `vendor-ban` text scan; `cfg(feature = "cuda")` count = 0 outside `device`;
a CI build against the mock backend.
**Folds in:** FFIX (the FFI exception trio).

## FACE: Families Are Capabilities to the Engine
**Rule.** Code outside a model family's own module never names a family. It asks a capability question
("has an indexer?", "MTP heads?") answered by a descriptor the family registers once.
**Violation response.** Replace the name branch (`if model_type == "…"`) with a capability; move the
family-specific lever into the family's group.
**Enforcement.** A `family-ban` scan over engine/bin roles; an exhaustive `enum ModelType` (no string matching,
no wildcard arm); a test that capability names are trait methods, not strings.

## PLEX: PLan, then EXecute
**Rule.** Every selection (kernel, tile, precision, accumulate type, K, eviction victim) is a typed field of a
plan built before execution. Executors `match` on the plan and never read a lever, a string or a shape
threshold. Numerics are one plan (a single `PrecisionPlan`), not scattered per-site decisions.
**Violation response.** Lift the choice into the planner; the executor gets a `match` on the plan field.
**Enforcement.** A `lever-at-launch` file gate (no setting/env read in launch/executor files); a dtype-string
gate; `clippy::wildcard_enum_match_arm` on plan enums; a coverage test that every kernel family appears in
the planner.

## TIDE: Typed Ids, Domain Edges
**Rule.** Wire and persisted records stop at the adapter. Every looked-up engine concept (slot, sequence,
block, expert, kernel, dtype, model type, lever) is a newtype or enum, never a `String` or bare integer.
Time is `MonoNs(u64)`/`DurNs(u64)`, never float milliseconds in core state.
**Violation response.** Introduce the newtype/enum at the edge and convert once; split records that mix wire
and domain fields.
**Enforcement.** depgraph (the engine does not depend on the protocol crate); a raw-id signature scan;
generated kernel ids with a ban on string kernel lookups; ratchets on `serde(skip)`/`serde(default)`.

## ACNB: Async Core, Never Blocks
**Rule.** The core is `async` wherever an operation can take time: receiving bytes or requests, blob reads
and writes, HTTP, a process wait, a timer, a device completion. It awaits; it never blocks. No
`thread::sleep`, no blocking lock, no busy-wait, no synchronous wait on I/O, no `block_on`. The core names no
runtime and spawns nothing: the host supplies the executor (tokio on native, `spawn_local` in the web worker,
a manual single-thread executor in tests) and any spawning goes through a port. Port methods whose operation
can take time are `async fn`; port futures carry no `Send` bound (as in mfs). Methods that answer
instantly (clock read, entropy fill, env lookup) stay plain `fn`. Policies stay stepped structs, so one core
runs on an OS thread or in a web worker. (Owner correction 2026-10-17: this replaces SCAH, "Sync Core, Async
Host"; the core is async, but it must never block.)
**Violation response.** Turn the blocking call into an awaited port call; move spawn/runtime code to the
host; atomics become struct fields; deadlines come from the Clock port and waits from the Timer port.
**Enforcement.** depgraph denylist for core roles (tokio, async-std, rayon, parking_lot, crossbeam,
futures-executor, std threads); a `blocking-ban` scan over core roles (`thread::sleep`, `block_on`,
`std::sync::Mutex`, blocking `recv()`, `spin_loop`); the `wasm32v1-none` build (NSCC); core futures driven by
the mem host's manual executor in the HPAR test.
**Folds in:** the "await what takes time" port rule (it replaces SPBD, Sync Ports By Default, which is
withdrawn).

## FILL: Fill Idle Latency with Live work
**Rule.** A wait boundary is a scheduling point, not a stop. Where an operation would wait (a device
completion, a readback, a peer's collective, a blob or network read, the next request), the work is split
into *issue* and *complete*: the core issues it, runs other ready work in between (another lane's prefill,
sampling and grammar-mask preparation, admission, KV planning, the next step's host-side setup) and consumes
the result when it is ready. Pipelines are double-buffered so the device and the host each have the next
item queued before the current one finishes. Waiting happens only when no ready work remains, and that idle
time is a measured, reported quantity with a recorded reason. Fill items are bounded and chunked, and read
only state frozen at the last plan point, so every host produces the same events (HPAR). ACNB says the
core never blocks; FILL says a yield must have something to yield to. A poll loop with nothing between its
polls is a FILL violation even when it does not block.
**Violation response.** Split the operation into issue/complete; move independent work between the two; give
the device its next item before reading back the current one. If nothing can go in between, record the
idle reason so the gap shows up in the trace instead of hiding in a spin.
**Enforcement.** An idle-time golden in the trace harness (idle ns per tick, with its reason, on the simulated
device); a `wait-site` scan (`event_synchronize`, stream/device sync, condvar waits, `park`, poll loops) over
non-host roles whose allowlist holds only declared load-only functions, the `sync-profile` feature, the
completion-thread module and the device runtime's primitives, plus a closed debt list that can never grow and
must be empty when the migration ends (no ratchet); the bench fails a run whose steady-state host syncs per
step is above zero; the TTFT grid and decode ladder gate any change to a wait site.

## CLER: CLosed ERrors at crate edges
**Rule.** Every crate boundary returns the registry error (CERF), never `anyhow`, `Box<dyn Error>` or a
crate-private string error. `anyhow` only in bin/tool/host roles. Never branch on error text. Exactly one place maps
error codes to HTTP statuses. Hosts log the detail through a diagnostics port before returning the code.
**Violation response.** Add the code (CERF), `match` it, delete the `.contains("…")`.
**Enforcement.** A per-role `anyhow` ratchet in depgraph; an `error-sniff` scan (`to_string().contains`,
`format!("{e}")` comparisons); a status-mapping test; `clippy::unwrap_used` in core.
**Folds in:** CEEC (Closed-Enum Error Contract).

## CERF: Coded Errors, Registered Forms
**Rule.** Errors come from one registry: a fieldless `#[repr(u16)]` `ErrorCode` enum. Its discriminant is
the stable code: append-only, never renumbered, with a reserved range per crate or domain. Each variant
declares its message once, as a form template with positional `{}` placeholders. One pointer-sized error
value (`Box` of code + call-site `file:line` origin + the form's arguments) is rendered lazily on `Display`.
Text is flexible: any argument can fill a placeholder, and `Generic = "{}"` exists for the rare unclassified
case. Text is never identity: the code alone decides equality, matching, the HTTP status and the wire form
(`u16` code + rendered text). Errors are built only with `error!(ErrorCode::X, args…)`, which checks the
argument count against the form at compile time. Modelled on Citadel-Protocol's `citadel_io::error`, with
three engine changes: a declarative `error_registry!` macro instead of a proc-macro derive (compile time);
arguments are a closed `ErrArg` enum, not `Box<dyn …>` (NDIC); and the registry crate is pure,
`no_std` + `alloc` (NSCC).
**Violation response.** Add a variant with its form; replace `Other(String)`, `anyhow!` or a crate-private
string error with `error!`; replace text inspection with a `match` on the code.
**Enforcement.** A committed `codes.lock` (code → name) that CI refuses to renumber or shrink; the
compile-time argument-count assert; a `Generic` use-count ratchet; a size test (`size_of::<Error>() == 8` and
`Result<(), Error>` == 8); a test that every form's placeholders are only `{}`, `{{` or `}}`.

## OWAY: One-WAY observability
**Rule.** Telemetry, metrics and benchmarking flow out of the core and never back in: no decision reads a
metric or gauge. Measurement code depends on serving code, never the reverse. Verbosity never changes device
behaviour (no debug-only syncs). Libraries never print.
**Violation response.** Replace a metric read with a boot-time probed value; move a domain trait out of the
telemetry crate; put bench-only items behind the bench feature.
**Enforcement.** depgraph (telemetry via a port only; bench only under its feature); a no-domain-trait-in-
telemetry test; metric/gauge reads banned outside `bin`; `clippy::print_stdout`/`print_stderr`; a debug-sync
scan.

## NSCC: No-Std Core Compiles
**Rule.** Every `core`/`abstract-io` crate is `#![no_std]` + `alloc` with `#![forbid(unsafe_code)]`, and
builds for `wasm32v1-none` and `wasm32-unknown-unknown`. Never `cfg(feature = "std")` the core.
**Violation response.** Remove the std item or push it into a host.
**Enforcement.** An `xtask wasm` gate over every pure crate (both targets); pure unit tests also run on a
32-bit wasm target.

## NDIC: No Dyn Inside the Core
**Rule.** Core crates hold no trait objects: ports and the device are generic parameters chosen once in the
binary. Generics stop at the core boundary (compile time).
**Violation response.** Replace `dyn X` with a type parameter or a closed enum; keep any unavoidable trait
object in the `device` adapter.
**Enforcement.** A `dyn-ban` scan over role `core` (allowlist: `dyn core::fmt::Write`, `dyn Any` in tests).

## BU64: Bytes are U64
**Rule.** Byte sizes, offsets and counts in persisted or wire data are `u64`/`u32`; `usize` only for
in-process indexing, converted with `try_from`, never `as`. No `usize`/`isize` in wire formats, persisted
records or hashed inputs. (Keeps the core correct on 32-bit wasm and cheap to port to wasm64.)
**Violation response.** Change the field type; convert at the use site; add the lint deny to the crate.
**Enforcement.** `clippy::cast_possible_truncation`, `cast_lossless`, `cast_possible_wrap`, `cast_sign_loss`
denied in every pure `lib.rs` (a `lints` gate checks the attribute is present).

## BPPT: Bounded Port calls Per Tick
**Rule.** The hot loop calls ports O(1) per tick and zero per token: entropy once per request at admission,
the clock once per tick phase, telemetry behind an `enabled()` guard.
**Violation response.** Hoist the call to the tick or admission boundary; cache the value in core state.
**Enforcement.** A port-call-counter golden in the trace harness (calls per tick per port); build/TTFT gates.

## HPAR: Host PARity
**Rule.** The same request through any two hosts, with the simulated device, the same seed and the same
virtual clock, yields a byte-identical event sequence. A host is complete only when the parity test passes.
**Violation response.** Fix the host. A differing event sequence is a host bug, never a tolerance.
**Enforcement.** A `host_parity` test (mem vs std vs worker) in the CI purity job.

---

## Folded clauses (extend existing principles)
- **SBIO** gains HOIO and TTPO. *Hosts Only do I/O:* every clock, random, env, file, process, network, log and
  message effect goes through a port trait; only a `host-*` crate implements a port and only a `bin` crate
  picks a host (enforced by depgraph roles + a pure denylist + env-ban). *Time Through the Port Only:* no
  `Instant`/`SystemTime`/`performance.now` outside a host (a `time-ban` scan with a shrinking allowlist).
- **PCND** gains DDEG. *Declared DEGradation:* when a host reports a capability `Unavailable`, the core switches
  to a mode named in settings (or the init message) and reports it in the served config; it never guesses a
  fallback. Host constructors return `Result`; the serve fingerprint includes the degradation set.
- **ROTI** absorbs NGIC, **VBOP** absorbs FFIX, **CLER** absorbs CEEC (text above). SPBD is withdrawn: ACNB
  replaces it with "await what takes time".

## Choosing a principle
| Smell | Principle |
|---|---|
| Same setting read in two places, or a global written after startup | ROTI |
| `cudarc`/`cublas`/`nccl` named outside a device crate | VBOP |
| `if model_type == "…"` outside the family's module | FACE |
| An executor or kernel launcher reads a lever, string or shape threshold | PLEX |
| A `String`/`usize` id crossing an engine API; float-ms in core state | TIDE |
| `tokio::spawn`, `thread::sleep`, `block_on`, a lock or an atomic inside core logic; a sync port method that waits | ACNB |
| A spin or synchronize with no other work between issue and wait; the device idle while the host prepares its next step | FILL |
| `anyhow` in a library API; `err.to_string().contains(…)` | CLER |
| `Other(String)`, a stringly error, an error type without a stable `u16` code | CERF |
| A decision reads a metric; serve code imports bench code; a library prints | OWAY |
| A core crate needs `std` | NSCC |
| `&dyn Trait` / `Box<dyn …>` inside the core | NDIC |
| `as usize` on a size/offset; `usize` in a persisted record | BU64 |
| A port call per token in the hot loop | BPPT |
| Two hosts disagree on the event stream | HPAR |
