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
    /// resolution, the recipe projection and sealed kernels (`metrale-blueprint`,
    /// `metrale-blueprint-trust`, `metrale-seal`).
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
        /// 2026-10-22: A sealed section that is malformed, bound to another body, or names a
        /// module whose source is `internal`: (what).
        SealedManifest = 0x0611 => "sealed manifest: {}",
        /// 2026-10-22: A recipient key whose fingerprint the sealed manifest does not list:
        /// (fingerprint).
        SealedNotRecipient = 0x0612 => "recipient key {} is not a recipient of this sealed \
            blueprint",
        /// 2026-10-22: The content key wrapped for a listed recipient does not unwrap (wrong key,
        /// altered wrap or another body; one opaque failure): (fingerprint).
        SealedUnwrap = 0x0613 => "the content key wrapped for recipient {} does not unwrap",
        /// 2026-10-22: A sealed module's ciphertext differs from the signed length or hash,
        /// refused before any decryption: (module, expected, found).
        SealedCiphertextHash = 0x0614 => "sealed module {} ciphertext must be {} but is {}",
        /// 2026-10-22: A sealed module's ciphertext does not authenticate (opaque): (module).
        SealedOpen = 0x0615 => "sealed module {} does not open",
        /// 2026-10-22: A sealed module resolved without a recipient key: (module).
        SealedKeyRequired = 0x0616 => "sealed module {} needs a recipient key \
            (--recipient-key)",
        /// 2026-10-22: A new revision sealed under the content key already running: (revision,
        /// running revision).
        SealedCekReused = 0x0617 => "revision {} reuses the content key of running revision {}",
    }
    /// 2026-10-22: `0x0700–0x07FF`: precision declarations (`metrale-precision`) and the
    /// architecture circuits' sources (`metrale-circuit`). Sub-blocks: precision `0x0700–0x070F`,
    /// circuit sources `0x0710–0x071F`, circuit plans `0x0720–0x072F`.
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
        /// 2026-10-22: A kernel-tree resolution (a class's sources, a checkpoint's kernel
        /// target) its host could not complete: (the host's reason, verbatim, since reports
        /// quote it).
        KernelTreeResolve = 0x0711 => "{}",
        /// 2026-10-23: A shape the serve runs that no plan-executed site of this rank covers
        /// under `--forward circuit` on a tensor-parallel rank: (mode, rows, why).
        CircuitPlanUncovered = 0x0720 => "{} at {} rows has no plan coverage: {}",
        /// 2026-10-23: A device count the circuit cannot be instantiated at: the engine splits
        /// every declared part over all N ranks, and this part cannot run at N: (N, the part, the
        /// dim, why: its degree bounds, unit or remainder policy).
        CircuitRanksRefused = 0x0721 => "{} ranks: part `{}` (dim `{}`) cannot split: {}",
    }
    /// 2026-10-23: `0x0800–0x08FF`: the kernel knowledge graph (`metrale-kg-derive`,
    /// `metrale-io-host-kg`, `metrale-kg-mcp`; DESIGN-kernel-knowledge-graph.md). Confirmed
    /// 2026-10-23 by delegated decision (the owner's reserved block). Sub-blocks: bundle
    /// integrity `0x0800–0x080F`, sources and records `0x0810–0x081F`, embedder `0x0820–0x082F`,
    /// store and tools `0x0830–0x083F`.
    KnowledgeGraph = 0x0800..=0x08FF {
        /// 2026-10-23: A record names a node no source defines: (where, the reference).
        KgDanglingRef = 0x0800 => "{} references {}, which no source defines",
        /// 2026-10-23: Two records derive the same id: (id, first source, second source).
        KgDuplicateId = 0x0801 => "id {} is derived twice ({} and {})",
        /// 2026-10-23: Two typed ids hash to one u64 point id: (id, id, point id).
        KgIdCollision = 0x0802 => "ids {} and {} collide on point id {}",
        /// 2026-10-23: A LEDGER.tsv row its checker refuses: (where, reason).
        KgBadLedgerRow = 0x0803 => "{}: {}",
        /// 2026-10-23: A record that is not in its file's row format: (where, reason).
        KgBadRow = 0x0804 => "{}: {}",
        /// 2026-10-23: More node labels than a collection holds: (needed, limit).
        KgTooManyLabels = 0x0805 => "the ontology needs {} labels; a collection holds {}",
        /// 2026-10-23: A field matched the secrets denylist and was dropped: (where, pattern).
        KgSecretRedacted = 0x0806 => "{}: a field matched the secrets denylist ({}) and was dropped",
        /// 2026-10-23: A LEDGER `gate` claim outside the grammar: (where, claim, reason).
        KgBadClaim = 0x0807 => "{}: claim `{}`: {}",
        /// 2026-10-23: An id that does not parse as `<label-prefix>:<key>`: (id, reason).
        KgBadId = 0x0808 => "id `{}`: {}",
        /// 2026-10-23: A memory `[[wikilink]]` to a memory not written yet: a deliberate forward
        /// reference (the memory rules), reported apart from integrity errors: (at, target).
        KgForwardLink = 0x0809 => "{} links to {}, which is not written yet",
        /// 2026-10-23: The LKB's own inventory script refused to build (its model is strict):
        /// (script, its first refusal line and count). The graph keeps the rest of the LKB.
        KgInventoryRefused = 0x080A => "{} refused to build the kernel inventory: {}",
        /// 2026-10-23: The inventory script built the inventory but reported a problem other than
        /// an unclassified kernel: (script, `kind: text`). Not fatal; the inventory is kept.
        KgInventoryProblem = 0x080B => "{} reports an inventory problem: {}",
        /// 2026-10-23: A config field missing or invalid; there is no default: (field, reason).
        KgConfig = 0x0810 => "kg config {}: {}",
        /// 2026-10-23: A source its host could not read or stat: (path, reason).
        KgSource = 0x0811 => "kg source {}: {}",
        /// 2026-10-23: A file's own validator refused a record; the file is unchanged:
        /// (validator, its output).
        KgValidatorRejected = 0x0812 => "validator {} refused the record: {}",
        /// 2026-10-23: A record write that could not complete: (path, reason).
        KgRecordWrite = 0x0813 => "kg record {}: {}",
        /// 2026-10-23: The bundle could not be re-derived; results are stale: (reason).
        KgStaleBundle = 0x0814 => "the bundle is stale: {}",
        /// 2026-10-23: The embedder is in its declared unavailable mode or failed: (reason).
        KgEmbedderUnavailable = 0x0820 => "embedder unavailable: {}",
        /// 2026-10-23: The embedder kept rate-limiting: (attempts).
        KgEmbedRateLimited = 0x0821 => "embedder still rate-limited or failing (HTTP 429/5xx) after {} attempts",
        /// 2026-10-23: An embedder response outside its protocol: (reason).
        KgEmbedProtocol = 0x0822 => "embedder response: {}",
        /// 2026-10-23: An embedding of another width than the config declares: (got, declared).
        KgEmbedDim = 0x0823 => "embedder returned dimension {}; the config declares {}",
        /// 2026-10-23: A Cypher text `kg_query` refuses: (reason).
        KgCypherRejected = 0x0830 => "cypher refused: {}",
        /// 2026-10-23: A graph store operation failed: (operation, reason).
        KgStore = 0x0831 => "kg store {}: {}",
        /// 2026-10-23: A tool argument outside its schema: (argument, reason).
        KgToolArgument = 0x0832 => "kg tool argument {}: {}",
        /// 2026-10-23: An id that names no node: (id).
        KgUnknownId = 0x0833 => "no node {}",
    }
    /// 2026-10-22: `0x0900–0x09FF`: power and energy (DESIGN-kernel-energy.md): the power port
    /// and its meters `0x0900–0x090F`, energy attribution `0x0910–0x091F`, power traces
    /// `0x0920–0x092F`, statistics (`metrale-stats`) `0x0930–0x093F`, cost `0x0940–0x094F`,
    /// measurement rows `0x0950–0x095F`, external meters `0x0960–0x096F`.
    Energy = 0x0900..=0x09FF {
        /// 2026-10-22: A power source that cannot be opened here: (source, reason).
        PowerSourceUnavailable = 0x0900 => "power source {} is unavailable: {}",
        /// 2026-10-22: A meter reported a field its capabilities do not declare available:
        /// (source, field).
        PowerFieldUndeclared = 0x0901 => "power source {} reported {}, which its capabilities \
            do not declare available",
        /// 2026-10-22: A ticket the meter did not issue or already completed: (source, ticket).
        PowerTicketUnknown = 0x0902 => "power source {}: ticket {} was not issued or is done",
        /// 2026-10-22: One read of a meter failed: (source, reason).
        PowerReadFailed = 0x0903 => "power source {}: read failed: {}",
        /// 2026-10-22: A capability set that leaves a field neither available nor declared
        /// unavailable with a reason (PCND): (source, field).
        PowerCapabilitiesIncomplete = 0x0904 => "power source {}: {} is neither available nor \
            declared unavailable",
        /// 2026-10-22: A meter's issue queue is full; complete a ticket first: (source, depth).
        PowerQueueFull = 0x0905 => "power source {}: {} reads already in flight",
        /// 2026-10-22: A measurement window the attribution refuses to integrate: (window,
        /// reason).
        EnergyWindowRefused = 0x0910 => "energy window {} refused: {}",
        /// 2026-10-22: Dynamic energy without both interleaved idle baselines: (repeat, which).
        EnergyBaselineMissing = 0x0911 => "repeat {}: idle baseline {} is missing; dynamic \
            energy needs the baselines on both sides of the load",
        /// 2026-10-22: The two idle baselines of a repeat drift apart: (repeat, percent, limit).
        EnergyBaselineDrift = 0x0912 => "repeat {}: idle baselines differ by {}% (limit {}%)",
        /// 2026-10-22: A run whose SM clock is below the floor: (MHz, floor MHz).
        EnergyClockFloor = 0x0913 => "SM clock {} MHz is below the {} MHz floor",
        /// 2026-10-22: Too many samples carry a throttle reason: (percent, limit).
        EnergyThrottled = 0x0914 => "{}% of load samples carry a throttle reason (limit {}%)",
        /// 2026-10-22: An in-serve attribution whose design matrix is ill-conditioned:
        /// (condition number, limit, worst VIF).
        EnergyIllConditioned = 0x0915 => "attribution refused: condition number {} exceeds {} \
            (worst VIF {})",
        /// 2026-10-22: Attribution inputs that do not fit together: (what).
        EnergyInputInvalid = 0x0916 => "energy attribution input invalid: {}",
        /// 2026-10-22: No source the platform's declared preference names is present: (scope,
        /// order).
        EnergyNoPreferredSource = 0x0917 => "no {} power source is present; the platform \
            declares the order {}",
        /// 2026-10-22: A power trace line or header that does not parse: (line, reason).
        PowerTraceFormat = 0x0920 => "power trace line {}: {}",
        /// 2026-10-22: A trace whose bytes differ from the hash its row names: (expected, found).
        PowerTraceHash = 0x0921 => "power trace sha256 must be {} but is {}",
        /// 2026-10-22: Too few observations for a statistic: (statistic, needed, got).
        StatsInsufficient = 0x0930 => "{} needs at least {} observations, got {}",
        /// 2026-10-22: Statistic inputs that are not valid: (statistic, reason).
        StatsInvalid = 0x0931 => "{}: {}",
        /// 2026-10-22: An iterative estimator that did not converge: (estimator, iterations).
        StatsNoConvergence = 0x0932 => "{} did not converge in {} iterations",
        /// 2026-10-22: A cost input that was not given; there is no default (PCND): (input).
        CostInputMissing = 0x0940 => "cost: {} is required; there is no default",
        /// 2026-10-22: A cost input out of range: (input, reason).
        CostInputInvalid = 0x0941 => "cost: {} {}",
        /// 2026-10-22: A measurement row key already present: (key).
        MeasurementDuplicate = 0x0950 => "measurements.toml already has a row {}",
        /// 2026-10-22: A measurement row the writer refuses: (key, reason).
        MeasurementInvalid = 0x0951 => "measurement row {}: {}",
        /// 2026-10-22: An external meter frame that does not parse: (meter, reason).
        ExternalMeterProtocol = 0x0960 => "external meter {}: {}",
        /// 2026-10-22: An external meter transport failure: (meter, reason).
        ExternalMeterTransport = 0x0961 => "external meter {} transport: {}",
    }
}
