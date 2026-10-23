// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: `met blueprint` arguments.
//!
//! Owner: server CLI.
//! Invariants: no defaults for a decision (PCND): the weights pin, the blueprint revision, the
//! minimum device memory and every key path are required where they are read.

use std::path::PathBuf;

/// `met blueprint`: build, sign, verify and convert signed blueprints (`.kbp`).
#[derive(clap::Args, Debug)]
pub struct BlueprintArgs {
    #[command(subcommand)]
    pub action: BlueprintAction,
}

/// 2026-10-22: What to verify against. Without `--trust-anchor` only the anchors compiled into
/// this binary are trusted.
#[derive(clap::Args, Debug, Clone)]
pub struct TrustArgs {
    /// A root public key file (`met blueprint key anchor`). Trusting it is the declared,
    /// reported mode `trust = dev-anchor`.
    #[arg(long)]
    pub trust_anchor: Option<PathBuf>,
    /// A signed revocation list (`met blueprint key revoke`), checked against the anchors and
    /// refused below this binary's revocation floor.
    #[arg(long)]
    pub revocations: Option<PathBuf>,
}

/// 2026-10-22: How a sealed file's modules are reached. Without `--recipient-key` a sealed
/// file is checked by ciphertext hash only, and that is reported.
#[derive(clap::Args, Debug, Clone)]
pub struct SealedArgs {
    /// A recipient key file (`kbp-recipient-key v1`, mode 600): unwrap the content key and
    /// decrypt and check every sealed module.
    #[arg(long)]
    pub recipient_key: Option<PathBuf>,
    /// A directory of ciphertexts named by their SHA-256 hex (a local copy of the
    /// content-addressed store), for sealed modules whose source is remote.
    #[arg(long)]
    pub ciphertexts: Option<PathBuf>,
}

/// 2026-10-22: `met blueprint seal` inputs.
#[derive(clap::Args, Debug)]
pub struct SealArgs {
    /// The plain, signed blueprint (lite or full).
    pub input: PathBuf,
    #[command(flatten)]
    pub trust: TrustArgs,
    /// A recipient public key file (`kbp-recipient-public v1`); repeat for several keys of
    /// one customer. There is no escrow recipient.
    #[arg(long, required = true)]
    pub recipient_public: Vec<PathBuf>,
    /// The customer id; only its opaque tag `sha256("kbp-customer\0" || lv(id))` is written.
    #[arg(long)]
    pub customer_id: String,
    /// The signing key file for the sealed manifest (`kbp-key v1`, scope blueprints; mode 600).
    #[arg(long)]
    pub key: PathBuf,
    /// The signing key's certificate.
    #[arg(long)]
    pub cert: PathBuf,
    /// Seal for the public content-addressed store under this base (ciphertexts at
    /// `<base>/<ct_sha256>`) and write a sealed lite file; without it the ciphertexts are inline
    /// (a single-file sealed full).
    #[arg(long, requires = "ciphertext_dir")]
    pub remote_base: Option<String>,
    /// Where the ciphertexts of `--remote-base` are written, one file per hash (must not exist).
    #[arg(long, requires = "remote_base")]
    pub ciphertext_dir: Option<PathBuf>,
    #[command(flatten)]
    pub out: OutArgs,
}

/// 2026-10-22: Where a converted file goes.
#[derive(clap::Args, Debug, Clone)]
pub struct OutArgs {
    /// Output path.
    #[arg(short = 'o', long)]
    pub output: PathBuf,
    /// Write the `kbp1.<base64>` text transport instead of the binary file.
    #[arg(long)]
    pub text: bool,
}

/// 2026-10-22: `from-recipe` / `build` inputs.
#[derive(clap::Args, Debug, Clone)]
pub struct BuildArgs {
    /// The recipe file, `recipes/<family>/<stem>.yaml`; its id is `<family>/<stem>`.
    pub recipe: PathBuf,
    /// The checkpoint directory (`config.json`, `hf_quant_config.json`, the shards).
    #[arg(long)]
    pub checkpoint: PathBuf,
    /// The checkpoint's git revision (the recipe states it only in a comment).
    #[arg(long)]
    pub checkpoint_revision: String,
    /// `full` hashes every `*.safetensors` shard; `skip` records the weights as not pinned
    /// (a declared mode the card reports).
    #[arg(long, value_parser = ["full", "skip"])]
    pub weights_digest: String,
    /// This blueprint's revision within its lineage.
    #[arg(long)]
    pub blueprint_revision: u64,
    /// The least device memory each rank must see, MiB.
    #[arg(long)]
    pub min_visible_mib: u64,
    #[command(flatten)]
    pub out: OutArgs,
}

/// 2026-10-22: The `met blueprint` subcommands.
#[derive(clap::Subcommand, Debug)]
pub enum BlueprintAction {
    /// Build an unsigned lite blueprint from a recipe, its circuit instance and the kernels
    /// embedded in this binary (every kernel source `internal`).
    FromRecipe(Box<BuildArgs>),
    /// `from-recipe`, optionally writing every kernel inline (`--full`).
    Build {
        #[command(flatten)]
        build: Box<BuildArgs>,
        /// Inline every module (a full file).
        #[arg(long)]
        full: bool,
    },
    /// Sign a blueprint with a seed file and its certificate.
    Sign {
        /// The blueprint.
        input: PathBuf,
        /// The key file (`kbp-key v1`; mode 600).
        #[arg(long)]
        key: PathBuf,
        /// The key's certificate (`met blueprint key certify`).
        #[arg(long)]
        cert: PathBuf,
        #[command(flatten)]
        out: OutArgs,
    },
    /// Verify a blueprint: header, body hash, certificate chain, both signatures, decode, and
    /// every kernel's bytes against its pinned hash (for a sealed file: the sealed signature,
    /// every ciphertext's hash, and with `--recipient-key` every decrypted module).
    Verify {
        /// The blueprint (binary or `kbp1.` text).
        input: PathBuf,
        #[command(flatten)]
        trust: TrustArgs,
        #[command(flatten)]
        sealed: SealedArgs,
    },
    /// Seal a signed blueprint for its recipients: a fresh content key per run, every module
    /// compressed then encrypted, the key wrapped to each recipient, the sealed manifest signed.
    /// One process; the content key never leaves it.
    Seal(Box<SealArgs>),
    /// Verify a sealed blueprint and write its decrypted modules into a new directory.
    Open {
        /// The sealed blueprint.
        input: PathBuf,
        #[command(flatten)]
        trust: TrustArgs,
        /// The recipient key file (`kbp-recipient-key v1`, mode 600).
        #[arg(long)]
        recipient_key: PathBuf,
        /// Local ciphertexts by hash, for remote sealed modules.
        #[arg(long)]
        ciphertexts: Option<PathBuf>,
        /// The directory to create for the modules (must not exist).
        #[arg(long)]
        out_dir: PathBuf,
    },
    /// Verify, then print a blueprint.
    Show {
        /// The blueprint.
        input: PathBuf,
        #[command(flatten)]
        trust: TrustArgs,
        /// `text`, `json` (the card) or `recipe`.
        #[arg(long, value_parser = ["text", "json", "recipe"])]
        format: String,
    },
    /// Verify, then print the recipe the blueprint projects to.
    ToRecipe {
        /// The blueprint.
        input: PathBuf,
        #[command(flatten)]
        trust: TrustArgs,
    },
    /// Rewrite as lite: no payload, every module `internal` (a sealed module `remote` under
    /// `--remote-base`, required for a sealed file). Body and signatures unchanged.
    Lite {
        /// The blueprint.
        input: PathBuf,
        #[command(flatten)]
        trust: TrustArgs,
        /// The content-addressed store base of a sealed file's ciphertexts.
        #[arg(long)]
        remote_base: Option<String>,
        #[command(flatten)]
        out: OutArgs,
    },
    /// Rewrite as full: every module inline (a sealed module's ciphertext, never its
    /// plaintext). Body and signatures unchanged.
    Full {
        /// The blueprint.
        input: PathBuf,
        #[command(flatten)]
        trust: TrustArgs,
        /// Local ciphertexts by hash, for remote sealed modules.
        #[arg(long)]
        ciphertexts: Option<PathBuf>,
        #[command(flatten)]
        out: OutArgs,
    },
    /// Export the store's card or graph JSON, or a JSON Schema.
    Export(Box<ExportArgs>),
    /// Key helpers: dev-anchor ceremony (anchor, certify, revoke) and sealed-kernel recipient
    /// keys (gen-recipient, recipient-fingerprint).
    #[command(subcommand)]
    Key(KeyAction),
}

/// 2026-10-22: `met blueprint export`.
#[derive(clap::Args, Debug)]
#[command(group(clap::ArgGroup::new("what").required(true).args(["card", "graph", "schema"])))]
pub struct ExportArgs {
    /// The blueprint (not with `--schema`).
    #[arg(required_unless_present = "schema")]
    pub input: Option<PathBuf>,
    #[command(flatten)]
    pub trust: TrustArgs,
    /// Export `kbp-card/v1`.
    #[arg(long)]
    pub card: bool,
    /// Export `kbp-graph/v1` for `--arm`.
    #[arg(long, requires = "arm")]
    pub graph: bool,
    /// The plan arm of a graph, `mode:rows` (e.g. `decode:1`).
    #[arg(long)]
    pub arm: Option<String>,
    /// Write the generated JSON Schema, `card` or `graph`.
    #[arg(long, value_parser = ["card", "graph"])]
    pub schema: Option<String>,
    /// Output path.
    #[arg(short = 'o', long)]
    pub output: PathBuf,
}

/// 2026-10-22: `met blueprint key`.
#[derive(clap::Subcommand, Debug)]
pub enum KeyAction {
    /// Write the public anchor of a root key file.
    Anchor {
        /// The root key file.
        #[arg(long)]
        key: PathBuf,
        /// Output path.
        #[arg(short = 'o', long)]
        output: PathBuf,
    },
    /// Certify a signing key with a root key.
    Certify {
        /// The root key file.
        #[arg(long)]
        root: PathBuf,
        /// The key file to certify.
        #[arg(long)]
        key: PathBuf,
        /// `blueprints`, `artifacts` or `index`.
        #[arg(long, value_parser = ["blueprints", "artifacts", "index"])]
        scope: String,
        /// First valid instant, Unix ms.
        #[arg(long)]
        not_before_ms: i64,
        /// End of validity (exclusive), Unix ms.
        #[arg(long)]
        not_after_ms: i64,
        /// Certificate serial.
        #[arg(long)]
        serial: u64,
        /// Output path.
        #[arg(short = 'o', long)]
        output: PathBuf,
    },
    /// Generate a sealed-kernel recipient key on this machine: a new mode-600 key file and its
    /// public half to send to the sealer. The seed is the only copy.
    GenRecipient {
        /// The key file to create (must not exist).
        #[arg(long)]
        out: PathBuf,
        /// The public key file to create (must not exist).
        #[arg(long)]
        public: PathBuf,
    },
    /// Print the fingerprint of a recipient public key file.
    RecipientFingerprint {
        /// The public key file (`kbp-recipient-public v1`).
        public: PathBuf,
    },
    /// Sign a revocation list with a root key.
    Revoke {
        /// The root key file.
        #[arg(long)]
        root: PathBuf,
        /// The list's serial (monotonic).
        #[arg(long)]
        serial: u64,
        /// Revoked kids.
        #[arg(long, value_delimiter = ',')]
        kid: Vec<String>,
        /// Output path.
        #[arg(short = 'o', long)]
        output: PathBuf,
    },
}
