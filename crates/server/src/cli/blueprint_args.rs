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
    /// every kernel's bytes against its pinned hash.
    Verify {
        /// The blueprint (binary or `kbp1.` text).
        input: PathBuf,
        #[command(flatten)]
        trust: TrustArgs,
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
    /// Rewrite as lite: no payload, every module `internal`. Body and signature unchanged.
    Lite {
        /// The blueprint.
        input: PathBuf,
        #[command(flatten)]
        trust: TrustArgs,
        #[command(flatten)]
        out: OutArgs,
    },
    /// Rewrite as full: every module inline. Body and signature unchanged.
    Full {
        /// The blueprint.
        input: PathBuf,
        #[command(flatten)]
        trust: TrustArgs,
        #[command(flatten)]
        out: OutArgs,
    },
    /// Export the store's card or graph JSON, or a JSON Schema.
    Export(Box<ExportArgs>),
    /// Key ceremony helpers for dev anchors: derive an anchor, certify a key, sign revocations.
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
