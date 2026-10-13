// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: Tests of the host-resident input embedding: which tensor the hook defers,
//! that a backend unable to map host memory fails naming the lever, and (GPU, `#[ignore]`)
//! that the production gather kernel reads the checkpoint's exact bytes through the mapped
//! table.
//!
//! GPU test: run with
//! ```text
//! cargo test -p metrale-model-arch --release host_embed -- --ignored --nocapture
//! ```
//! on a host whose build has the `qwen3.8-flash-next/nvfp4` kernel set.

use super::{EMBED_ON_HOST_LEVER, defer_hook, load};
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_weights::weights::{DeferredTensor, WeightDtype};

const VOCAB: usize = 37;
const HIDDEN: usize = 24;
const HEADER: u64 = 13;

fn config(tie_word_embeddings: bool) -> ModelConfig {
    ModelConfig {
        tie_word_embeddings,
        ..ModelConfig::qwen3_next_80b_nvfp4()
    }
}

fn untied() -> ModelConfig {
    config(false)
}

/// 2026-10-13: A shard whose table starts at a non-zero offset, every byte distinct
/// modulo 251 so a misplaced row or a wrong offset cannot read back equal.
fn shard() -> (tempfile::TempDir, DeferredTensor, Vec<u8>) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("shard.safetensors");
    let table: Vec<u8> = (0..VOCAB * HIDDEN * 2).map(|i| (i % 251) as u8).collect();
    let mut file = vec![0xEEu8; HEADER as usize];
    file.extend_from_slice(&table);
    file.extend_from_slice(&[0xEE; 7]);
    std::fs::write(&path, &file).unwrap();
    let d = DeferredTensor {
        path,
        offset: HEADER,
        shape: vec![VOCAB, HIDDEN],
        dtype: WeightDtype::BF16,
    };
    (dir, d, table)
}

#[test]
fn the_hook_defers_exactly_the_untied_bf16_input_embedding() {
    let config = untied();
    let name = super::super::embed_name(&config);
    let hook = defer_hook(&config, true).expect("lever set, untied: a hook");
    assert!(hook(&name, WeightDtype::BF16));
    assert!(
        !hook(&name, WeightDtype::FP32),
        "only the BF16 table is gathered in place"
    );
    for other in [
        "lm_head.weight",
        "model.ngram_embeddings.embedders.0.weight",
        "mtp.embed_tokens.weight",
        &format!("{name}_scale"),
        &format!("x.{name}"),
    ] {
        assert!(
            !hook(other, WeightDtype::BF16),
            "{other} must stay on device"
        );
    }
}

#[test]
fn no_hook_without_the_lever_or_on_a_tied_checkpoint() {
    assert!(defer_hook(&untied(), false).is_none());
    assert!(
        defer_hook(&config(true), true).is_none(),
        "a tied lm_head is a dense GEMM over the table: it must stay on device"
    );
}

#[test]
fn a_backend_that_cannot_map_host_memory_fails_naming_the_lever() {
    let (_dir, d, _) = shard();
    let gpu = metrale_gpu_runtime::gpu::mock::MockGpuBackend::new();
    let err = load(&d, &gpu).expect_err("the mock backend maps no host memory");
    assert!(format!("{err:#}").contains(EMBED_ON_HOST_LEVER), "{err:#}");
}

#[test]
fn a_truncated_shard_is_an_error_not_a_short_table() {
    let (_dir, mut d, _) = shard();
    d.shape = vec![VOCAB + 1, HIDDEN];
    let gpu = metrale_gpu_runtime::gpu::mock::MockGpuBackend::new();
    assert!(load(&d, &gpu).is_err());
}

#[test]
#[ignore]
fn the_gather_kernel_reads_the_checkpoint_bytes_through_the_mapped_table() {
    let (_dir, d, table) = shard();
    let set = metrale_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("qwen3.8-flash-next/nvfp4 not in this build");
    let gpu = metrale_gpu_runtime::cuda_backend::MetraleCudaBackend::new(0, &set.modules)
        .expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let embed = load(&d, g).unwrap();
    let ids: [u32; 6] = [36, 0, 17, 17, 5, 36];
    let id_bytes: Vec<u8> = ids.iter().flat_map(|t| t.to_le_bytes()).collect();
    let ids_dev = g.alloc(id_bytes.len()).unwrap();
    g.copy_h2d(&id_bytes, ids_dev).unwrap();
    let row = HIDDEN * 2;
    let out = g.alloc(ids.len() * row).unwrap();
    let stream = g.default_stream();
    let kernel = g.kernel("embed_from_argmax", "batched_embed").unwrap();
    metrale_model_layers::layers::ops::batched_embed(
        g,
        kernel,
        ids_dev,
        embed.weight,
        out,
        ids.len() as u32,
        HIDDEN as u32,
        stream,
    )
    .unwrap();
    // 2026-10-13: The single-row path (`embed_row`) is a device-to-device copy.
    let single = g.alloc(row).unwrap();
    g.copy_d2d_async(embed.weight.offset(17 * row), single, row, stream)
        .unwrap();
    g.synchronize(stream).unwrap();
    let mut got = vec![0u8; ids.len() * row];
    g.copy_d2h(out, &mut got).unwrap();
    for (i, &t) in ids.iter().enumerate() {
        let want = &table[t as usize * row..(t as usize + 1) * row];
        assert_eq!(&got[i * row..(i + 1) * row], want, "row {i} (token {t})");
    }
    let mut one = vec![0u8; row];
    g.copy_d2h(single, &mut one).unwrap();
    assert_eq!(one, table[17 * row..18 * row]);
}
