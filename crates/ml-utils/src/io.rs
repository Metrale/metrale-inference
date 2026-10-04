// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The I/O boundary (SBIO). This crate reads a checkpoint only through
//! [`CheckpointSource`] and writes one only through [`CheckpointSink`]; the server implements
//! both over the filesystem and the Hub, tests over memory ([`mem`]).
//!
//! Owner: metrale-ml-utils.
//! Invariants:
//! - A source returns file text and safetensors header bytes; it never interprets them.
//! - A sink appends bytes in the order it is given them; the container format is decided by
//!   `st_format`, not by the sink.

use crate::error::Result;

/// 2026-10-03: Read access to one checkpoint.
pub trait CheckpointSource {
    /// 2026-10-03: The file names the checkpoint holds (relative, `/`-separated).
    fn files(&self) -> Result<Vec<String>>;
    /// 2026-10-03: A small file's bytes (config, tokenizer, templates).
    fn read(&self, rel: &str) -> Result<Vec<u8>>;
    /// 2026-10-03: Each safetensors shard's header JSON (the bytes after the 8-byte length),
    /// with the shard's file name.
    fn shard_headers(&self) -> Result<Vec<(String, Vec<u8>)>>;
}

/// 2026-10-03: Write access for one output checkpoint.
pub trait CheckpointSink {
    /// 2026-10-03: Write a whole small file.
    fn write_file(&mut self, rel: &str, bytes: &[u8]) -> Result<()>;
    /// 2026-10-03: Start a shard file with its header bytes (length prefix included).
    fn begin_shard(&mut self, rel: &str, header: &[u8]) -> Result<()>;
    /// 2026-10-03: Append tensor bytes to the open shard.
    fn append(&mut self, bytes: &[u8]) -> Result<()>;
    /// 2026-10-03: Extend the open shard by `len` bytes that are never read (a skeleton
    /// checkpoint's data region; a filesystem sink leaves it sparse).
    fn skip(&mut self, len: u64) -> Result<()>;
    /// 2026-10-03: Finish the open shard.
    fn end_shard(&mut self) -> Result<()>;
}

/// 2026-10-03: The files besides the weights and the two rewritten JSON files that a mock
/// copies unchanged: tokenizer, chat template, generation and preprocessor configs. A README or
/// model card describes the source, not the mock, and is not copied.
pub fn is_aux_file(rel: &str) -> bool {
    matches!(
        rel,
        "tokenizer.json"
            | "tokenizer_config.json"
            | "tokenizer.model"
            | "vocab.json"
            | "merges.txt"
            | "special_tokens_map.json"
            | "added_tokens.json"
            | "chat_template.jinja"
            | "chat_template.json"
            | "generation_config.json"
            | "preprocessor_config.json"
            | "video_preprocessor_config.json"
    )
}

/// 2026-10-03: In-memory source and sink, for tests of this crate and its callers.
#[cfg(any(test, feature = "test-utils"))]
pub mod mem {
    use std::collections::BTreeMap;

    use super::{CheckpointSink, CheckpointSource};
    use crate::error::{MlError, Result};

    /// 2026-10-03: A checkpoint held in memory: files by name.
    #[derive(Debug, Clone, Default)]
    pub struct MemCheckpoint {
        /// 2026-10-03: Every file, by relative name.
        pub files: BTreeMap<String, Vec<u8>>,
    }

    impl CheckpointSource for MemCheckpoint {
        fn files(&self) -> Result<Vec<String>> {
            Ok(self.files.keys().cloned().collect())
        }
        fn read(&self, rel: &str) -> Result<Vec<u8>> {
            self.files
                .get(rel)
                .cloned()
                .ok_or_else(|| MlError::Io(format!("{rel}: no such file")))
        }
        fn shard_headers(&self) -> Result<Vec<(String, Vec<u8>)>> {
            let mut out = Vec::new();
            for (name, bytes) in &self.files {
                if !name.ends_with(".safetensors") {
                    continue;
                }
                let len = bytes
                    .get(..8)
                    .map(|b| u64::from_le_bytes(b.try_into().expect("8 bytes")) as usize)
                    .ok_or_else(|| MlError::Io(format!("{name}: shorter than 8 bytes")))?;
                let header = bytes
                    .get(8..8 + len)
                    .ok_or_else(|| MlError::Io(format!("{name}: truncated header")))?;
                out.push((name.clone(), header.to_vec()));
            }
            Ok(out)
        }
    }

    /// 2026-10-03: A sink that collects files in memory; it is itself a readable checkpoint.
    #[derive(Debug, Default)]
    pub struct MemSink {
        /// 2026-10-03: What was written.
        pub out: MemCheckpoint,
        open: Option<(String, Vec<u8>)>,
    }

    impl CheckpointSink for MemSink {
        fn write_file(&mut self, rel: &str, bytes: &[u8]) -> Result<()> {
            self.out.files.insert(rel.to_string(), bytes.to_vec());
            Ok(())
        }
        fn begin_shard(&mut self, rel: &str, header: &[u8]) -> Result<()> {
            if self.open.is_some() {
                return Err(MlError::Io("a shard is already open".into()));
            }
            self.open = Some((rel.to_string(), header.to_vec()));
            Ok(())
        }
        fn append(&mut self, bytes: &[u8]) -> Result<()> {
            let (_, buf) = self
                .open
                .as_mut()
                .ok_or_else(|| MlError::Io("append with no open shard".into()))?;
            buf.extend_from_slice(bytes);
            Ok(())
        }
        fn skip(&mut self, len: u64) -> Result<()> {
            let (_, buf) = self
                .open
                .as_mut()
                .ok_or_else(|| MlError::Io("skip with no open shard".into()))?;
            buf.resize(buf.len() + len as usize, 0);
            Ok(())
        }
        fn end_shard(&mut self) -> Result<()> {
            let (rel, buf) = self
                .open
                .take()
                .ok_or_else(|| MlError::Io("end_shard with no open shard".into()))?;
            self.out.files.insert(rel, buf);
            Ok(())
        }
    }
}
