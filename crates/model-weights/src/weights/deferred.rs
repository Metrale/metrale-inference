// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tensors the checkpoint loader deliberately does NOT upload, and the hook a
//! model's own weight loader uses to say so.
//!
//! # Two ways a tensor becomes deferred
//!
//! 1. **`is_ngram_table`** — a NAME rule baked into the runtime, for the
//!    LongCat / Qwen3.8-Flash-Next embedding tables. Always on, because those
//!    tables (63 GB to 102 GB) cannot be uploaded on any box we serve.
//! 2. **[`DeferHook`]** — a predicate the MODEL's loader supplies, keyed on
//!    `(name, on-disk dtype)`. Off unless a loader asks for it, and plumbed
//!    exactly like `skip_vision`: `serve_phases/weights.rs` asks
//!    `ModelWeightLoader::defer_predicate` once and hands the answer to
//!    whichever checkpoint loader runs.
//!
//! Both end in the same place — a [`DeferredTensor`] in the store's
//! `deferred` map — so a consumer never has to know which rule withheld it.
//!
//! # Why a loader would ask for this
//!
//! The rule is NOT "this tensor is huge". It is "the bytes that reach the
//! device are not the bytes on disk, and the model's loader is the only thing
//! that knows how to make them".
//!
//! `nvidia/GLM-5.3-Flash-NVFP4` ships the MTP block's 288 routed experts as
//! full-width BF16 while the forward is w4a16 end to end. Every one of those
//! tensors is read once, by `glm5_next_load::bind_expert`, quantised, and
//! replaced by an NVFP4 triple ~3.6x smaller. Uploading the BF16 first is
//! ~7.25 GB per EP=2 rank of device residency that lasts from the sweep until
//! `prune_after_load` — i.e. across the WHOLE 45-layer build, which on a
//! unified-memory box is where the host-memory guard fires. Measured
//! 2026-09-21 on a 2-rank boot: survived at a 2048-token shape (host floor
//! 1.74 GB), killed rank 0 at layer 38 at the 32K shape (935 MB against a
//! 1200 MB floor) BEFORE the MTP bind could free anything.
//!
//! Deferring makes that transient structurally impossible: the BF16 never
//! leaves the page cache, and the only expert bytes that reach the device are
//! the NVFP4 ones, for the experts this rank owns.
//!
//! # Contract, both halves of it
//!
//! * The loader that defers a tensor MUST read it back — nothing else will.
//!   A predicate that over-matches withholds a tensor a binder expects, and
//!   `WeightStore::get` then fails by NAME at bind time rather than producing
//!   a subtly wrong model. That is the intended failure mode; it is still the
//!   loader author's job not to reach it.
//! * The OOM pre-flight MUST agree with the predicate. A deferred tensor is
//!   never part of the sweep peak, so counting it at its on-disk size refuses
//!   loads that fit — the same reasoning that already excludes the n-gram
//!   tables. The pre-flight and the retain loop are handed the SAME closure
//!   for exactly this reason.
//!
//! # 🪤 Keyed on the ON-DISK dtype, and F16 is never offered
//!
//! A deferred tensor is served from its own bytes in the shard, so the
//! predicate is asked about the width those bytes actually have. F16 is the
//! one width both disk loaders REWRITE on the way to the store
//! (`f16_to_bf16_bytes`), so a (path, offset) locator would hand its reader
//! F16 where the rest of the engine expects BF16. F16 tensors are therefore
//! never deferred — by the retain loop and by the pre-flight alike, which is
//! the only way the two can stay the same rule.

use anyhow::{Context, Result};

use super::WeightDtype;

/// A model loader's answer to "which tensors will I read from disk myself?".
///
/// `(tensor name, store dtype) -> defer`. `Arc` because both checkpoint
/// loaders hold it for the length of a load and the fast loader's pre-flight
/// and retain loop both call it; `Send + Sync` because the fast loader's
/// pipeline is threaded.
pub type DeferHook = std::sync::Arc<dyn Fn(&str, WeightDtype) -> bool + Send + Sync>;

/// 2026-10-17: The device bytes a model's loader will allocate itself for a tensor it DEFERS,
/// for the OOM pre-flight: `(tensor name, store dtype, shape) -> bytes`. Asked only about
/// tensors the [`DeferHook`] claimed; a loader that decodes a deferred tensor into a device
/// buffer of another size (a compressed scale plane decoded to its native bytes) states that
/// size here, so the pre-flight does not count the buffer as zero. `None` on a loader means
/// every deferred tensor costs the pre-flight nothing, today's rule.
pub type DeferCharge = std::sync::Arc<dyn Fn(&str, WeightDtype, &[usize]) -> usize + Send + Sync>;

/// Where a skipped tensor lives, so a consumer can read it in place.
///
/// 2026-10-17: The byte length is stored, not recomputed, and is private: a
/// header's `shape` is untrusted, so a tensor is only ever built by a
/// constructor that checks `shape × element size` without overflow and, where
/// the header states a span, refuses a shape that disagrees with it. Before
/// this a shape larger than its span read the neighbouring tensor's bytes.
#[derive(Clone, Debug)]
pub struct DeferredTensor {
    /// Shard file containing the tensor.
    pub path: std::path::PathBuf,
    /// ABSOLUTE byte offset of the tensor's first element in that file
    /// (safetensors header length + the tensor's `data_offsets[0]`).
    pub offset: u64,
    pub shape: Vec<usize>,
    pub dtype: WeightDtype,
    len: usize,
}

impl DeferredTensor {
    /// 2026-10-17: A tensor whose container states its byte span (a
    /// safetensors header's `data_offsets`). Refuses a shape whose byte size
    /// overflows or differs from `span`.
    pub fn at_span(
        path: std::path::PathBuf,
        offset: u64,
        span: u64,
        shape: Vec<usize>,
        dtype: WeightDtype,
    ) -> Result<Self> {
        let d = Self::from_shape(path, offset, shape, dtype)?;
        if u64::try_from(d.len).ok() != Some(span) {
            anyhow::bail!(
                "deferred tensor at offset {offset} of {}: shape {:?} × {} B is {} B but its \
                 span is {span} B",
                d.path.display(),
                d.shape,
                d.dtype.byte_size(),
                d.len
            );
        }
        Ok(d)
    }

    /// 2026-10-17: A tensor whose container states no span (a GGUF entry, a
    /// test fixture): its length is `shape × element size`, refused on overflow.
    pub fn from_shape(
        path: std::path::PathBuf,
        offset: u64,
        shape: Vec<usize>,
        dtype: WeightDtype,
    ) -> Result<Self> {
        let len = shape
            .iter()
            .try_fold(dtype.byte_size(), |acc, &d| acc.checked_mul(d))
            .with_context(|| {
                format!(
                    "deferred tensor at offset {offset} of {}: shape {shape:?} overflows",
                    path.display()
                )
            })?;
        Ok(DeferredTensor {
            path,
            offset,
            shape,
            dtype,
            len,
        })
    }

    /// The tensor's on-disk footprint.
    ///
    /// Only the fixed-width dtypes a safetensors header can carry reach here,
    /// so `numel * byte_size` is exact — the block-based GGUF dtypes
    /// ([`WeightDtype::PackedQ2_0`] and the K-quants) report 0 bytes/element
    /// and are never deferred by either rule.
    pub fn byte_size(&self) -> usize {
        self.len
    }

    /// Read the whole tensor's raw bytes into host memory.
    ///
    /// The consumer for a deferred tensor is usually a row cache that faults
    /// single rows off NVMe; this is the other shape of consumer — a loader
    /// that needs the tensor ONCE, at bind time, to derive the buffer it will
    /// actually upload.
    ///
    /// 🪤 `read_exact` rather than `read_to_end`: a short read means the shard
    /// is truncated or the offset is wrong, and both must be an error here
    /// rather than a shorter tensor whose element count is rejected later with
    /// a message pointing at the wrong thing. Seek-then-read rather than
    /// `pread`, because `WeightStore` builds on Windows too.
    pub fn read_host_bytes(&self) -> Result<Vec<u8>> {
        self.read_host_range(0, self.byte_size())
    }

    /// 2026-10-09: Read `len` raw bytes starting `start` bytes into the tensor, for a loader
    /// that needs a contiguous part of it (the rows one tensor-parallel rank keeps). Errors when
    /// the range runs past the tensor, and on a short read, as [`Self::read_host_bytes`].
    pub fn read_host_range(&self, start: usize, len: usize) -> Result<Vec<u8>> {
        use std::io::{Read, Seek, SeekFrom};
        let at = self.file_offset(start, len)?;
        let mut f = std::fs::File::open(&self.path)
            .with_context(|| format!("deferred tensor: opening {}", self.path.display()))?;
        f.seek(SeekFrom::Start(at))?;
        let mut buf = vec![0u8; len];
        f.read_exact(&mut buf).with_context(|| {
            format!(
                "deferred tensor: reading {len} B at offset {at} of {}",
                self.path.display()
            )
        })?;
        Ok(buf)
    }

    /// 2026-10-19: [`Self::read_host_range`] by positional read, dropping the range from the
    /// page cache after it (`expert_stream::pread_uncached`), for a loader that reads most of a
    /// checkpoint once: on a unified-memory device the cached copy competes with the weights
    /// for the same memory. Positional reads need a unix host.
    pub fn read_host_range_uncached(&self, start: usize, len: usize) -> Result<Vec<u8>> {
        let at = self.file_offset(start, len)?;
        let f = std::fs::File::open(&self.path)
            .with_context(|| format!("deferred tensor: opening {}", self.path.display()))?;
        let mut buf = vec![0u8; len];
        super::expert_stream::pread_uncached(&f, at, &mut buf, false).with_context(|| {
            format!(
                "deferred tensor: reading {len} B at offset {at} of {}",
                self.path.display()
            )
        })?;
        Ok(buf)
    }

    /// 2026-10-19: The file offset of bytes `start..start + len` of the tensor. Errors when the
    /// range runs past the tensor or the offset overflows.
    fn file_offset(&self, start: usize, len: usize) -> Result<u64> {
        let total = self.byte_size();
        start
            .checked_add(len)
            .filter(|&e| e <= total)
            .with_context(|| {
                format!(
                    "deferred tensor: bytes {start}..+{len} run past its {total} B in {}",
                    self.path.display()
                )
            })?;
        self.offset.checked_add(start as u64).with_context(|| {
            format!(
                "deferred tensor: offset {} + {start} overflows",
                self.offset
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_shard(bytes: &[u8]) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("shard.safetensors");
        std::fs::write(&path, bytes).unwrap();
        (dir, path)
    }

    #[test]
    fn byte_size_is_the_on_disk_footprint() {
        let d = DeferredTensor::from_shape(
            std::path::PathBuf::from("x"),
            0,
            vec![2048, 4096],
            WeightDtype::BF16,
        )
        .unwrap();
        assert_eq!(d.byte_size(), 2048 * 4096 * 2);
    }

    #[test]
    fn host_read_returns_exactly_the_tensor_at_its_offset() {
        // 8 filler bytes stand in for a header, then four BF16 values.
        let payload: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let (_d, path) = write_shard(&[vec![0xFFu8; 8], payload.clone()].concat());
        let d = DeferredTensor::from_shape(path, 8, vec![2, 2], WeightDtype::BF16).unwrap();
        assert_eq!(d.read_host_bytes().unwrap(), payload);
    }

    /// 2026-10-09: A range read returns exactly those bytes of the tensor (not of the file),
    /// and a range past the tensor's end is refused even when the file has more bytes after it.
    /// 2026-10-19: The uncached reader answers the same, refusals included.
    #[test]
    fn range_read_is_relative_to_the_tensor_and_bounded_by_it() {
        let payload: Vec<u8> = (10u8..26).collect();
        let (_d, path) = write_shard(&[vec![0xFFu8; 8], payload.clone(), vec![0xEE; 8]].concat());
        let d = DeferredTensor::from_shape(path, 8, vec![4, 4], WeightDtype::UInt8).unwrap();
        type Reader = fn(&DeferredTensor, usize, usize) -> Result<Vec<u8>>;
        let readers: [Reader; 2] = [
            DeferredTensor::read_host_range,
            DeferredTensor::read_host_range_uncached,
        ];
        for read in readers {
            assert_eq!(read(&d, 4, 8).unwrap(), payload[4..12]);
            assert_eq!(read(&d, 0, 16).unwrap(), payload);
            assert_eq!(read(&d, 16, 0).unwrap(), Vec::<u8>::new());
            let e = read(&d, 12, 5).unwrap_err().to_string();
            assert!(e.contains("run past"), "{e}");
            assert!(read(&d, usize::MAX, 2).is_err());
        }
    }

    /// 2026-10-17: The pre-flight charge of a loader that sets no [`DeferCharge`].
    fn no_charge(_: &str, _: WeightDtype, _: &[usize]) -> usize {
        0
    }

    /// Write a one-tensor-per-entry safetensors file and return its path.
    fn write_safetensors(
        dir: &std::path::Path,
        entries: &[(&str, &str, Vec<usize>, usize)],
    ) -> std::path::PathBuf {
        let mut header = serde_json::Map::new();
        let mut end = 0usize;
        for (name, dtype, shape, len) in entries {
            header.insert(
                (*name).to_string(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [end, end + len],
                }),
            );
            end += len;
        }
        let header = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut blob = (header.len() as u64).to_le_bytes().to_vec();
        blob.extend_from_slice(&header);
        blob.extend_from_slice(&vec![0u8; end]);
        let path = dir.join("model.safetensors");
        std::fs::write(&path, blob).unwrap();
        path
    }

    /// 🔴 The other half of the contract: a deferred tensor is never swept, so
    /// the OOM pre-flight must not count it. Before this, the official GLM-5.3
    /// export's ~7.25 GB of full-width MTP experts were charged to a peak they
    /// never joined.
    #[test]
    fn the_preflight_estimate_drops_deferred_tensors() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_safetensors(
            dir.path(),
            &[
                ("big.weight", "BF16", vec![8, 8], 128),
                ("small.weight", "BF16", vec![4], 8),
            ],
        );
        let files = [path];

        let all = super::super::estimate_load_bytes(&files, &|_| false, &|_, _| false, &no_charge)
            .unwrap();
        assert_eq!(all, 136);

        let without = super::super::estimate_load_bytes(
            &files,
            &|_| false,
            &|n, d| n == "big.weight" && d == WeightDtype::BF16,
            &no_charge,
        )
        .unwrap();
        assert_eq!(without, 8, "the deferred tensor must leave the peak");

        // 🪤 Keyed on the dtype too: the same NAME at a width the predicate
        // does not claim is still counted, because it is still uploaded.
        let wrong_dtype = super::super::estimate_load_bytes(
            &files,
            &|_| false,
            &|n, d| n == "big.weight" && d == WeightDtype::UInt8,
            &no_charge,
        )
        .unwrap();
        assert_eq!(wrong_dtype, 136);
    }

    /// 2026-10-17: A deferred tensor adds what the loader's charge says it will allocate for it,
    /// asked with its name, store dtype and shape; a skipped tensor is never charged, and a kept
    /// one is never asked.
    #[test]
    fn the_preflight_estimate_adds_the_charge_of_deferred_tensors() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_safetensors(
            dir.path(),
            &[
                ("plane.codes", "U8", vec![40], 40),
                ("other.codes", "U8", vec![6], 6),
                ("kept.weight", "BF16", vec![4], 8),
            ],
        );
        let files = [path];
        let asked = std::sync::Mutex::new(Vec::new());
        let charge = |n: &str, d: WeightDtype, s: &[usize]| {
            asked.lock().unwrap().push((n.to_string(), d, s.to_vec()));
            s[0] * 2
        };
        let got = super::super::estimate_load_bytes(
            &files,
            &|n| n == "other.codes",
            &|n, _| n.ends_with(".codes"),
            &charge,
        )
        .unwrap();
        assert_eq!(got, 8 + 80, "kept tensor at its size plus the charge");
        assert_eq!(
            *asked.lock().unwrap(),
            vec![("plane.codes".to_string(), WeightDtype::UInt8, vec![40])]
        );
    }

    /// F16 is rewritten on the way to the store, so it can never be served
    /// from its on-disk bytes — the estimate counts it however loudly the
    /// predicate claims it, which is exactly what the loaders then do.
    #[test]
    fn an_f16_tensor_is_counted_even_when_the_predicate_claims_everything() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_safetensors(dir.path(), &[("h.weight", "F16", vec![8, 8], 128)]);
        let got = super::super::estimate_load_bytes(&[path], &|_| false, &|_, _| true, &no_charge)
            .unwrap();
        assert_eq!(got, 128);
    }

    #[test]
    fn a_truncated_shard_is_an_error_not_a_short_tensor() {
        let (_d, path) = write_shard(&[0u8; 8]);
        let d = DeferredTensor::from_shape(path, 0, vec![64, 64], WeightDtype::BF16).unwrap();
        assert!(d.read_host_bytes().is_err());
    }

    /// 2026-10-17: A header span that disagrees with `shape × element size` is refused in
    /// both directions; a shape larger than its span used to read the next tensor's bytes.
    #[test]
    fn a_span_that_disagrees_with_the_shape_is_refused() {
        let p = std::path::PathBuf::from("x");
        let ok = DeferredTensor::at_span(p.clone(), 8, 32, vec![4, 4], WeightDtype::BF16).unwrap();
        assert_eq!(ok.byte_size(), 32);
        for span in [31u64, 33, 0, u64::MAX] {
            let e = DeferredTensor::at_span(p.clone(), 8, span, vec![4, 4], WeightDtype::BF16)
                .unwrap_err()
                .to_string();
            assert!(e.contains("span"), "span {span}: {e}");
        }
    }

    /// 2026-10-17: A shape whose byte size overflows is refused by both constructors, and an
    /// empty shape is one element.
    #[test]
    fn an_overflowing_shape_is_refused() {
        let p = std::path::PathBuf::from("x");
        let big = vec![usize::MAX / 2 + 1, 1];
        let e = DeferredTensor::from_shape(p.clone(), 0, big.clone(), WeightDtype::BF16)
            .unwrap_err()
            .to_string();
        assert!(e.contains("overflows"), "{e}");
        assert!(DeferredTensor::at_span(p.clone(), 0, 0, big, WeightDtype::BF16).is_err());
        let scalar = DeferredTensor::from_shape(p, 0, vec![], WeightDtype::FP32).unwrap();
        assert_eq!(scalar.byte_size(), 4);
    }

    /// 2026-10-17: An offset near `u64::MAX` is an error, not a wrapped seek.
    #[test]
    fn a_range_read_past_u64_offsets_is_an_error() {
        let d = DeferredTensor::from_shape(
            std::path::PathBuf::from("x"),
            u64::MAX,
            vec![4],
            WeightDtype::UInt8,
        )
        .unwrap();
        let e = d.read_host_range(2, 1).unwrap_err().to_string();
        assert!(e.contains("overflows"), "{e}");
    }
}
