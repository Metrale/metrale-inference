// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: The embedding check counts an embedding a model's loader deferred (read from
//! disk at bind time, never uploaded) and still refuses a store with none at all.

use super::check_embedding_and_head;
use crate::weights::{DeferredTensor, WeightDtype, WeightStore};

fn deferred(name: &str) -> WeightStore {
    let mut store = WeightStore::empty();
    store.defer(
        name.to_string(),
        DeferredTensor {
            path: std::path::PathBuf::from("shard.safetensors"),
            offset: 0,
            shape: vec![8, 4],
            dtype: WeightDtype::BF16,
        },
    );
    store
}

#[test]
fn a_deferred_embedding_counts_as_present() {
    check_embedding_and_head(&deferred("model.embed_tokens.weight")).unwrap();
}

#[test]
fn a_store_with_no_embedding_anywhere_is_refused() {
    let err = check_embedding_and_head(&deferred("model.ngram_embeddings.embedders.0.weight"))
        .expect_err("an n-gram table is not the input embedding");
    assert!(err.to_string().contains("no embedding tensor"), "{err}");
}
