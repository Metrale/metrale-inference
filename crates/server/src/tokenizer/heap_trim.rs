// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Returning the tokenizer's freed heap to the OS after a long prompt
//! (`--tokenizer-trim-min-tokens`).
//!
//! Encoding a prompt builds a `tokenizers::Encoding` holding offsets, type ids, masks and
//! word ids for every token beside the ids. The ids are copied out and the `Encoding` is
//! dropped, but glibc keeps most of the freed pages in its arenas, so a very long prompt
//! leaves hundreds of bytes per token resident in host memory that, on a unified-memory
//! device, the GPU pools share. A [`HeapTrimPolicy`] with a threshold asks the allocator to
//! give those pages back (`malloc_trim(0)`) once the prompt that freed them reached the
//! threshold. The trim runs after the ids are copied, so it cannot change them.
//!
//! Owner: server (tokenizer).
//! Invariants: a policy whose threshold is 0 never calls its trimmer; one whose threshold
//! is `N > 0` calls it once per encode of `N` or more tokens and never for fewer.

use std::sync::Arc;

/// 2026-10-17: Gives freed heap memory back to the operating system.
pub trait HeapTrim: Send + Sync {
    fn trim(&self);
}

/// 2026-10-17: glibc's `malloc_trim(0)`: every arena releases its free pages.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
struct GlibcTrim;

#[cfg(all(target_os = "linux", target_env = "gnu"))]
impl HeapTrim for GlibcTrim {
    fn trim(&self) {
        // SAFETY: `malloc_trim` takes no pointers; it only walks glibc's own arenas under
        // their locks, so it is sound from any thread at any time.
        unsafe {
            libc::malloc_trim(0);
        }
    }
}

/// 2026-10-17: Whether, and from how many prompt tokens, `ChatTokenizer::encode` trims the
/// process heap after an encode.
#[derive(Clone)]
pub struct HeapTrimPolicy {
    min_tokens: usize,
    trimmer: Option<Arc<dyn HeapTrim>>,
}

impl HeapTrimPolicy {
    /// 2026-10-17: Never trim.
    pub fn off() -> Self {
        Self {
            min_tokens: 0,
            trimmer: None,
        }
    }

    /// 2026-10-17: The policy `--tokenizer-trim-min-tokens N` asks for: off at 0, else
    /// glibc's trim after every encode of `N` or more tokens.
    ///
    /// # Errors
    /// `N > 0` on a target whose allocator is not glibc's, where there is no trim to run
    /// and the flag would do nothing.
    pub fn process_heap(min_tokens: usize) -> Result<Self, String> {
        if min_tokens == 0 {
            return Ok(Self::off());
        }
        Self::glibc_trimmer().map(|trimmer| Self::with_trimmer(min_tokens, trimmer))
    }

    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    fn glibc_trimmer() -> Result<Arc<dyn HeapTrim>, String> {
        Ok(Arc::new(GlibcTrim))
    }

    #[cfg(not(all(target_os = "linux", target_env = "gnu")))]
    fn glibc_trimmer() -> Result<Arc<dyn HeapTrim>, String> {
        Err("this build's allocator is not glibc's, so there is no malloc_trim to run".into())
    }

    /// 2026-10-17: `trimmer` after every encode of `min_tokens` or more tokens; 0 never.
    pub fn with_trimmer(min_tokens: usize, trimmer: Arc<dyn HeapTrim>) -> Self {
        Self {
            min_tokens,
            trimmer: Some(trimmer),
        }
    }

    /// 2026-10-17: The threshold; 0 is off.
    pub fn min_tokens(&self) -> usize {
        self.min_tokens
    }

    /// 2026-10-17: Run the trim when an encode of `tokens` tokens reaches the threshold.
    pub(super) fn after_encode(&self, tokens: usize) {
        match &self.trimmer {
            Some(trimmer) if self.min_tokens > 0 && tokens >= self.min_tokens => {
                let started = std::time::Instant::now();
                trimmer.trim();
                tracing::debug!(
                    tokens,
                    trim_ms = started.elapsed().as_secs_f64() * 1e3,
                    "tokenizer heap trimmed"
                );
            }
            _ => {}
        }
    }
}
