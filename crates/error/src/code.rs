// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The registry itself: every [`ErrorCode`], its stable number and its form,
//! grouped by the owner-confirmed range blocks (table in the crate docs).
//!
//! Owner: error.
//! Invariants:
//! - Append-only: a code is never renumbered or removed. A new code takes the next free number
//!   in its block and gets a line appended to `crates/error/codes.lock`.
//! - Prefer a specific code to [`ErrorCode::Generic`]; its uses are ratcheted
//!   (`tests/generic_ratchet.rs`).

use crate::registry::error_registry;

error_registry! {
    /// 2026-10-17: `0x0000–0x00FF`: universal codes any crate may use.
    General = 0x0000..=0x00FF {
        /// 2026-10-17: Unclassified; the one argument is the whole message. Ratcheted.
        Generic = 0x0000 => "{}",
        /// 2026-10-17: An i/o failure a host reports; the argument describes it.
        Io = 0x0001 => "i/o error: {}",
        /// 2026-10-17: A caller passed a value the callee refuses: (argument, reason).
        InvalidArgument = 0x0002 => "invalid argument {}: {}",
        /// 2026-10-17: A declared, reported gap: (operation, backend or host).
        NotImplemented = 0x0003 => "{} is not yet implemented on {}",
        /// 2026-10-17: A capability that exists but cannot serve now: (what).
        Unavailable = 0x0004 => "{} is unavailable",
    }
    /// 2026-10-17: `0x0100–0x01FF`: abstract-io ports and hosts (abstract-io-2.md D17).
    /// Sub-blocks: time `0x0100` (reserved; cannot fail), entropy `0x0110`, env `0x0118`,
    /// blob `0x0120–0x013F`, proc `0x0140`, http `0x0150`, transport `0x0170`,
    /// host construction `0x0180`.
    AbstractIo = 0x0100..=0x01FF {
        /// 2026-10-17: Entropy source unavailable.
        EntropyUnavailable = 0x0110 => "entropy source unavailable: {}",
        /// 2026-10-17: An environment variable is not valid unicode.
        EnvNotUnicode = 0x0118 => "environment variable {} is not valid unicode",
        /// 2026-10-17: A blob namespace is unavailable.
        BlobUnavailable = 0x0120 => "blob namespace {} unavailable",
        /// 2026-10-17: A blob was not found.
        BlobNotFound = 0x0121 => "blob {} not found",
        /// 2026-10-17: A blob key is invalid.
        BlobBadKey = 0x0122 => "invalid blob key {}",
        /// 2026-10-17: A blob namespace is full.
        BlobFull = 0x0123 => "blob namespace {} is full",
        /// 2026-10-17: A blob is corrupt.
        BlobCorrupt = 0x0124 => "blob {} is corrupt",
        /// 2026-10-17: A blob operation failed with an OS error.
        BlobOs = 0x0125 => "blob {}: os error {}",
        /// 2026-10-17: Process spawning is unavailable.
        ProcUnavailable = 0x0140 => "process spawning unavailable",
        /// 2026-10-17: A program was not found.
        ProcNotFound = 0x0141 => "program {} not found",
        /// 2026-10-17: Permission was denied running a program.
        ProcPermission = 0x0142 => "permission denied running {}",
        /// 2026-10-17: A program timed out.
        ProcTimeout = 0x0143 => "{} timed out after {} ms",
        /// 2026-10-17: A program wrote too much output.
        ProcOutputTooLarge = 0x0144 => "{} wrote more than {} bytes",
        /// 2026-10-17: A process operation failed with an OS error.
        ProcOs = 0x0145 => "{}: os error {}",
        /// 2026-10-17: A host is unreachable.
        HttpUnreachable = 0x0150 => "{} is unreachable",
        /// 2026-10-17: An HTTP request timed out.
        HttpTimeout = 0x0151 => "{} timed out after {} ms",
        /// 2026-10-17: An HTTP response exceeded a size limit.
        HttpTooLarge = 0x0152 => "{} exceeded {} bytes",
        /// 2026-10-17: An HTTP protocol error occurred.
        HttpProtocol = 0x0153 => "http protocol error from {}: {}",
        /// 2026-10-17: An HTTP response was truncated.
        HttpTruncated = 0x0154 => "response from {} was truncated",
        /// 2026-10-17: An HTTP operation failed with an OS error.
        HttpOs = 0x0155 => "{}: os error {}",
        /// 2026-10-17: A stream was closed.
        SinkClosed = 0x0170 => "stream {} closed",
        /// 2026-10-17: A stream is full.
        SinkFull = 0x0171 => "stream {} full",
        /// 2026-10-17: A host configuration error.
        HostConfig = 0x0180 => "host config: {}",
        /// 2026-10-17: A host capability is unavailable.
        HostUnavailable = 0x0181 => "host capability {} unavailable",
    }
    /// 2026-10-17: `0x0200–0x02FF`: sched-core (the `Device` trait, scheduler, lora) and the
    /// device backends. Sub-blocks: sched-core `0x0200–0x027F`, device backends `0x0280–0x02FF`.
    SchedDevice = 0x0200..=0x02FF {
        /// 2026-10-17: A device driver call returned a failure status: (call, status).
        DeviceDriver = 0x0280 => "{} failed: driver status {}",
    }
    /// 2026-10-17: `0x0300–0x03FF`: serve-identity and host-probe (bench support). Sub-blocks:
    /// serve env `0x0300–0x030F`.
    ServeIdentity = 0x0300..=0x03FF {
        /// 2026-10-17: A declared env key outside the `METRALE_` prefix (owner, name).
        ServeEnvNotALever = 0x0300 => "{} declares env {}, which is not a METRALE_* serve lever. \
            The gate applies exactly the levers a recipe declares and nothing else, so a variable \
            the server does not read cannot be declared here.",
        /// 2026-10-17: A declared harness variable (owner, name, name).
        ServeEnvHarness = 0x0301 => "{} declares env {}, which places, logs or builds rather \
            than measuring ({} is a harness variable, not a serve lever) — it is the box's to \
            set, never the recipe's.",
        /// 2026-10-17: A declared build, dev or tool variable (owner, name, class).
        ServeEnvNotRead = 0x0302 => "{} declares env {}, which the server never reads (it is a \
            {} variable), so declaring it here would change nothing that is measured.",
        /// 2026-10-17: A declared name the lever table does not declare (owner, name).
        ServeEnvUndeclared = 0x0303 => "{} declares env {}, which is not a declared lever: \
            nothing reads it, so it is a typo or a lever that no longer exists, and `met serve` \
            refuses to start with it set. `met dump-serve-options` lists every declared lever.",
        /// 2026-10-17: A declared lever whose value is empty after trimming (owner, name).
        ServeEnvEmpty = 0x0304 => "{} declares env {} with an empty value. An exported-but-empty \
            variable is how a shell spells \"unset\": declare a value or drop the key.",
        /// 2026-10-17: Levers the harness carries that the owner does not declare (count, owner,
        /// `NAME=value` joined by ", ").
        ServeEnvInherited = 0x0305 => "the harness environment carries {} serve lever(s) that {} \
            does not declare: {}. A lever the recipe did not ask for changes what the gate \
            measures (METRALE_FP8_ROWWISE=1 builds FP8 twin weights that a 0.70-util serve \
            cannot boot with — #1242), so it is refused rather than inherited. Unset it — on a \
            node agent, remove it from bench.yaml `env:`, which reaches EVERY gate on the box — \
            or declare it in the recipe's `env:` (or the gate's [benchmarks.serve_env]) if the \
            gate is measured under it.",
        /// 2026-10-17: Declared levers the harness carries at another value (owner, count, one
        /// line each joined by "; ").
        ServeEnvContradicted = 0x0306 => "the harness environment contradicts {} on {} serve \
            lever(s): {}. Neither value wins silently: unset the variable so the declaration \
            applies, or change the declaration.",
    }
    /// 2026-10-17: `0x0400–0x04FF`: the device settings projections (`metrale-settings`
    /// `groups::device*`: what gpu-runtime, kernels, cache, speculative and model-weights read).
    DeviceProjection = 0x0400..=0x04FF {
        /// 2026-10-17: A resolved lever value the device-side type cannot hold (env, value,
        /// accepted range).
        ProjectionOutOfRange = 0x0400 => "{}: {} is outside {}",
    }
    /// 2026-10-18: `0x0500–0x05FF`: the KV-cache codec registry and plan (`metrale-kv-codec`).
    KvCodec = 0x0500..=0x05FF {
        /// 2026-10-18: A codec name no registry row or pair alias spells: (name, every accepted
        /// spelling).
        KvCodecUnknown = 0x0500 => "unknown KV-cache codec `{}`; accepted: {}",
        /// 2026-10-18: A (K, V) codec pair with no fused kernels: (k, v, head_dim).
        KvPairUnsupported = 0x0501 => "KV-cache pair k={} v={} is not supported at head_dim {}",
        /// 2026-10-18: A VBR transition the ladder refuses: (layer, side, from rung, to rung,
        /// reason).
        KvTransitionNotPermitted = 0x0502 => "KV transition on layer {} side {} from rung {} to \
            rung {} is not permitted: {}",
        /// 2026-10-18: The model does not declare the capability a codec needs: (codec, what).
        KvCapabilityMissing = 0x0503 => "KV-cache codec {} needs a capability the model does \
            not declare: {}",
        /// 2026-10-18: A spilled block read back under another layout: (written codec, written
        /// digest, reading codec, reading digest).
        KvSpillCodecMismatch = 0x0504 => "spilled KV block was written as {} (layout {}) but \
            is read as {} (layout {})",
        /// 2026-10-18: A KV plan input or entry that must be explicit is absent: (what).
        KvPlanMissing = 0x0505 => "the KV-cache plan has no entry for {}",
        /// 2026-10-18: A VBR ladder that breaks one of its construction rules: (rule).
        KvLadderInvalid = 0x0506 => "the VBR ladder is invalid: {}",
        /// 2026-10-18: An ABI family the registry declares but this build has no launcher for:
        /// (ABI, operation).
        KvAbiUnimplemented = 0x0507 => "the {} KV ABI has no {} launcher in this build",
        /// 2026-10-18: An indexer codec plan or block the codec's layout cannot hold: (codec,
        /// reason).
        KvIndexerGeometry = 0x0508 => "indexer cache {}: {}",
        /// 2026-10-18: A codec with no host dequant (its registry row's `host_dequant` is `None`) asked
        /// for by a host path: (codec, path).
        KvHostDequantMissing = 0x0509 => "KV-cache codec {} has no host dequant, so {} cannot \
            run",
        /// 2026-10-18: A codec asked for a mode its kernels do not implement: (codec, mode).
        KvCodecFeatureUnsupported = 0x050A => "KV-cache codec {} does not support {}",
        /// 2026-10-19: A compressed-latent attention cache plan or row its codec cannot hold:
        /// (cache, reason).
        KvLatentGeometry = 0x050B => "latent cache {}: {}",
    }
    /// 2026-10-22: `0x0600–0x06FF`: blueprints (`.kbp`): framing, trust, signatures, kernel
    /// resolution and the recipe projection (`metrale-blueprint`, `metrale-blueprint-trust`).
    Blueprint = 0x0600..=0x06FF {
        /// 2026-10-22: A `.kbp` whose framing is malformed: (what).
        BlueprintFrame = 0x0600 => "blueprint frame: {}",
        /// 2026-10-22: A section larger than its cap, refused before allocation: (section,
        /// length, cap).
        BlueprintTooLarge = 0x0601 => "blueprint {} is {} bytes, over the {}-byte cap",
        /// 2026-10-22: A format or schema this reader does not know: (what, found, supported).
        BlueprintFormat = 0x0602 => "blueprint {} {} is not supported (this reader knows {})",
        /// 2026-10-22: The body's SHA-256 differs from the header's.
        BlueprintBodyHash = 0x0603 => "blueprint body hash mismatch: header {} body {}",
        /// 2026-10-22: A signature that does not verify: (what, algorithm, kid).
        BlueprintSignature = 0x0604 => "blueprint {}: the {} signature of {} does not verify",
        /// 2026-10-22: A key no trust anchor vouches for: (kid, reason).
        BlueprintUntrusted = 0x0605 => "key {} is not trusted: {}",
        /// 2026-10-22: A key on the revocation list: (kid, revocation serial).
        BlueprintRevoked = 0x0606 => "key {} is revoked (revocations serial {})",
        /// 2026-10-22: A certificate outside its validity window: (kid, not_before ms,
        /// not_after ms, now ms).
        BlueprintCertExpired = 0x0607 => "certificate of {} is valid {}..{} ms, now is {} ms",
        /// 2026-10-22: A serial lower than the floor this reader holds: (what, found, floor).
        BlueprintRollback = 0x0608 => "{} serial {} is below the floor {}: refused as a rollback",
        /// 2026-10-22: A certificate whose scope does not allow the use: (kid, scope, needed).
        BlueprintScope = 0x0609 => "key {} has scope {} but {} is needed",
        /// 2026-10-22: A verified section that does not decode: (section, reason).
        BlueprintDecode = 0x060A => "blueprint {} does not decode: {}",
        /// 2026-10-22: An artifact whose bytes hash to another id: (module, expected, found).
        ArtifactHashMismatch = 0x060B => "kernel module {} must hash to {} but hashes to {}",
        /// 2026-10-22: A kernel module the chosen source cannot supply: (module, artifact, source).
        KernelMissing = 0x060C => "kernel module {} ({}) is not available from {}",
        /// 2026-10-22: A kernel source this build parses but cannot resolve yet: (source, module,
        /// phase).
        KernelSourceNotInPhase = 0x060D => "kernel source {} of {} is resolved from phase {}",
        /// 2026-10-22: A recipe the blueprint projection refuses: (recipe, reason).
        BlueprintRecipe = 0x060E => "recipe {} cannot be projected to a blueprint: {}",
        /// 2026-10-22: A key file or seed that is malformed: (what, reason).
        BlueprintKey = 0x060F => "key {}: {}",
        /// 2026-10-22: A stored section that does not decompress to its declared raw length:
        /// (section, codec, reason).
        BlueprintCodec = 0x0610 => "blueprint {} does not decompress under {}: {}",
    }
    /// 2026-10-22: `0x0700–0x07FF`: precision declarations (`metrale-precision`) and the
    /// architecture circuits' sources (`metrale-circuit`). Sub-blocks: precision `0x0700–0x070F`,
    /// circuit sources `0x0710–0x071F`.
    Precision = 0x0700..=0x07FF {
        /// 2026-10-22: A `quantization_config` that does not parse into a declared precision
        /// plan: (the field path and why, as one sentence).
        QuantConfigInvalid = 0x0700 => "{}",
        /// 2026-10-22: A declared weight whose on-disk storage is not modelled or not valid:
        /// (module, reason).
        WeightStorageUnmodelled = 0x0701 => "{}: {}",
        /// 2026-10-22: A `--weight-quantization` tier and W4A4 lever that do not combine:
        /// (reason).
        WeightQuantTierInvalid = 0x0702 => "{}",
        /// 2026-10-22: An `--activation-quantization` value that does not parse: (where and why).
        ActQuantInvalid = 0x0703 => "{}",
        /// 2026-10-22: A circuit source (a repository file or listing) its host could not
        /// supply: (repo-relative path, reason).
        CircuitSource = 0x0710 => "{}: {}",
    }
}
