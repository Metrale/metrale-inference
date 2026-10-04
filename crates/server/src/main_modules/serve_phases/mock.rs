// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: `met serve --mock <spec>`: serve a mock (rehearsal) of MODEL without its weights.
//!
//! 1. [`materialize`] plans the mock from the source's metadata and safetensors headers (local,
//!    or read from the Hub), and writes its *skeleton* (config, metadata, tokenizer files, the
//!    resolved spec and shard files whose headers are real and whose data is sparse) under the
//!    cache root's `metrale-mocks/<digest>/`. The serve then runs on that directory, so every
//!    reader of config, metadata and headers sees the mock.
//! 2. [`load_store`] synthesizes the weights instead of reading the data region.
//! 3. [`disclosure`] names any mock a model directory holds (by the resolved spec, or by a
//!    shard's `metrale_mock` metadata), with or without `--mock`.
//!
//! Owner: server startup (`met serve`).
//! Invariants:
//! - The bytes uploaded are `metrale_ml_utils::synthesize`'s, the bytes `met ml-utils mockify`
//!   writes for the same spec and source.
//! - Tensor and expert parallelism are refused under `--mock` (M1 synthesizes whole tensors).

use std::path::Path;

use anyhow::{Context, Result, bail};
use metrale_config::ModelConfig;
use metrale_ml_utils::{MOCK_METADATA_KEY, MockPlan, RESOLVED_FILE};

use crate::cli;
use crate::cli::ml_utils_io::{FsCheckpoint, FsSink, open_source};

/// 2026-10-03: A planned mock and its skeleton directory.
pub(crate) struct MockServe {
    pub plan: MockPlan,
}

/// 2026-10-03: Plan `--mock`, write its skeleton if this cache does not hold it yet, and point
/// `args.model_from_path` at it. `Ok(None)` without `--mock`.
pub(crate) fn materialize(args: &mut cli::ServeArgs) -> Result<Option<MockServe>> {
    let Some(spec_path) = args.mock.clone() else {
        return Ok(None);
    };
    if args.tp_size > 1 || args.ep_size > 1 {
        bail!("--mock serves at --tp-size 1 --ep-size 1 (sharded mocks are a later milestone)");
    }
    let source = match (&args.model_from_path, &args.model) {
        (Some(p), _) => p.to_string_lossy().into_owned(),
        (None, Some(m)) => m.clone(),
        (None, None) => bail!("--mock needs MODEL or --model-from-path: the checkpoint to mock"),
    };
    let (spec, profile) = cli::ml_utils::read_spec(&spec_path)?;
    let src = open_source(&source, args.cache_dir.as_deref(), true)?;
    let plan = cli::ml_utils::plan_of(&src, &spec, profile.as_ref())?;
    let root =
        crate::model_resolver::resolve_cache_root(args.cache_dir.as_deref())?.join("metrale-mocks");
    std::fs::create_dir_all(&root).with_context(|| format!("creating {}", root.display()))?;
    let dir = root.join(&plan.digest);
    match std::fs::read_to_string(dir.join(RESOLVED_FILE)) {
        Ok(text) if text == plan.resolved => {
            tracing::info!("Mock skeleton {} (cached)", dir.display());
        }
        Ok(_) => bail!(
            "{} holds a different resolved spec than this plan; remove it and serve again",
            dir.display()
        ),
        Err(_) => {
            let mut sink = FsSink::create(&dir)?;
            metrale_ml_utils::write_skeleton(&plan, src.source.as_ref(), &mut sink)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            sink.commit()?;
            tracing::info!("Mock skeleton written to {}", dir.display());
        }
    }
    tracing::info!(
        "MOCK: serving a rehearsal of {} (digest {}): layers {:?} of {}, {} tensors, {:.2} GB \
         synthesized; no weight file is read",
        src.id,
        plan.digest,
        plan.selection.kept_layers(),
        plan.layers_full,
        plan.tensors.len(),
        plan.bytes() as f64 / 1e9
    );
    args.model_from_path = Some(dir);
    Ok(Some(MockServe { plan }))
}

/// 2026-10-03: The weight store of a mock: synthesized, keeping exactly what the fast loader
/// would keep.
pub(crate) fn load_store(
    m: &MockServe,
    config: &ModelConfig,
    gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
    ep_rank: usize,
    ep_size: usize,
) -> Result<metrale_model_weights::weights::WeightStore> {
    #[cfg(unix)]
    {
        let loader = super::weights::fast_loader(config, ep_rank, ep_size, None);
        if loader.defer.is_some() {
            bail!("--mock does not serve a model whose loader defers tensors to disk");
        }
        let threads = std::thread::available_parallelism().map_or(1, |t| t.get());
        metrale_model_weights::synthetic::load_synthetic(
            &m.plan,
            gpu,
            &|name| loader.should_skip_tensor(name),
            threads,
        )
    }
    #[cfg(not(unix))]
    {
        let _ = (m, config, gpu, ep_rank, ep_size);
        bail!("--mock needs a Unix host (it shares the fast loader's skip rules)")
    }
}

/// 2026-10-03: The digest of the mock `model_dir` holds, if any: sha256 of its resolved spec, or
/// the `metrale_mock` metadata of its shards. A resolved spec and shard metadata that disagree,
/// or shards that name different mocks, are an error.
pub(crate) fn disclosure(model_dir: &Path) -> Result<Option<String>> {
    let from_file = match std::fs::read(model_dir.join(RESOLVED_FILE)) {
        Ok(b) => Some(metrale_ml_utils::resolved_digest(&b)),
        Err(_) => None,
    };
    let mut from_shards: Option<String> = None;
    let src = FsCheckpoint::new(model_dir);
    let headers = metrale_ml_utils::io::CheckpointSource::shard_headers(&src)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    for (name, h) in headers {
        let v: serde_json::Value = serde_json::from_slice(&h).with_context(|| name.clone())?;
        if let Some(d) = v["__metadata__"][MOCK_METADATA_KEY].as_str() {
            match &from_shards {
                Some(prev) if prev != d => bail!("{name} names mock {d}, another shard {prev}"),
                _ => from_shards = Some(d.to_string()),
            }
        }
    }
    match (from_file, from_shards) {
        (Some(f), Some(s)) if f != s => {
            bail!(
                "{} digests to {f} but its shards name mock {s}",
                RESOLVED_FILE
            )
        }
        (f, s) => Ok(f.or(s)),
    }
}

#[cfg(test)]
#[path = "mock_tests.rs"]
mod mock_tests;
