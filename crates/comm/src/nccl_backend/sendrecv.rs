// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: The send/recv all-reduces: the 2-rank exchange (moved here unchanged from
//! `nccl_backend.rs`) and the one-shot exchange at `world_size >= 2` (2026-10-16: was `>= 3`).
//!
//! One-shot: each rank sends its BF16 partial to every peer and receives every peer's partial into
//! its own slot of a registered buffer (one grouped NCCL call, one network hop), then
//! `bf16_add_rank_sum` sums the partials in rank order in FP32 and rounds once, so every rank
//! writes the same bytes. It is for latency-bound payloads (decode steps); a payload above the
//! enabled bound goes to `ncclAllReduce`, whose ring moves fewer bytes per rank.
//!
//! Owner: metrale-comm.
//! Invariants:
//! - The one-shot path never receives more than `max_bytes` per peer: `applies` admits only
//!   payloads of at most `max_bytes`, and the buffer holds `world_size - 1` slots of `max_bytes`.
//! - Enabled only at `world_size >= 2`. 2026-10-16: At `world_size == 2` a plain `all_reduce`
//!   still takes the 2-rank exchange (`bf16_add_inplace`, checked first in `comm_impl`), so its
//!   bytes do not move; the one-shot there serves `exchange_partials`, whose consumers sum the
//!   two partials as `bf16_add_rank_sum` does. That FP32 sum rounded once equals `__hadd`'s
//!   correctly rounded BF16 sum for every pair except two negative zeros (+0 from the FP32
//!   accumulator's +0 start, -0 from `__hadd`): FP32 keeps 24 bits >= 2 * 8 + 1, so rounding
//!   twice is innocuous for a sum (`world2_sum_tests`). The two paths' bytes are therefore NOT
//!   identical: they differ in the sign of the exact-zero sum of two -0 partials.
//! - The path an unplanned all-reduce takes is `ar_path::ar_path`; a planned one names its own
//!   (`CommBackend::all_reduce_via`), refused where `ar_path::runnable` refuses it.

use anyhow::Result;
use std::ffi::c_void;
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};

use metrale_gpu_sys::nccl::{self, NcclComm, NcclDataType, NcclResult};

use crate::PeerPartials;

use crate::ar_path::{PathState, oneshot_applies};
use metrale_circuit::CollectivePath;
use super::{
    ALL_REDUCE_DTYPE_BYTES, NcclBackend, cuLaunchKernel, cuMemAlloc_v2, cuMemFree_v2,
    ensure_payload_fits,
};

/// 2026-10-08: State of the one-shot all-reduce. `max_bytes == 0` means disabled.
pub(super) struct OneShot {
    max_bytes: usize,
    /// 2026-10-08: `world_size - 1` slots of `max_bytes`, peer r in slot `r < rank ? r : r - 1`.
    recv: u64,
    /// 2026-10-08: `bf16_add_rank_sum` handle from `set_rank_sum_kernel`; 0 until set.
    sum_kernel: AtomicU64,
}

/// 2026-10-08: The receive slot of `peer` on `rank`, or `None` for the rank itself. A free function
/// so the tests check it without a communicator.
pub(super) fn peer_slot(rank: usize, peer: usize) -> Option<usize> {
    match peer.cmp(&rank) {
        std::cmp::Ordering::Less => Some(peer),
        std::cmp::Ordering::Equal => None,
        std::cmp::Ordering::Greater => Some(peer - 1),
    }
}

/// 2026-10-16: Refuse a one-shot bound the communicator cannot use: a nonzero bound on a single
/// rank (nothing to exchange) or one that is not a whole number of BF16 elements. A free function
/// for the tests.
pub(super) fn validate_oneshot(world_size: usize, max_bytes: usize) -> Result<()> {
    if max_bytes == 0 {
        return Ok(());
    }
    if world_size < 2 {
        anyhow::bail!(
            "the one-shot all-reduce needs world_size >= 2 (world_size {world_size}); \
             a single rank has nothing to exchange"
        );
    }
    if !max_bytes.is_multiple_of(ALL_REDUCE_DTYPE_BYTES) {
        anyhow::bail!(
            "one-shot all-reduce bound {max_bytes} B is not a whole number of BF16 elements"
        );
    }
    Ok(())
}

impl OneShot {
    /// 2026-10-08: Allocate and register the receive slots when `max_bytes > 0`.
    ///
    /// # Errors
    /// What [`validate_oneshot`] refuses, or a failed allocation. A failed registration is only
    /// logged, as for the 2-rank buffer.
    pub(super) fn new(comm: NcclComm, world_size: usize, max_bytes: usize) -> Result<Self> {
        validate_oneshot(world_size, max_bytes)?;
        if max_bytes == 0 {
            return Ok(Self {
                max_bytes: 0,
                recv: 0,
                sum_kernel: AtomicU64::new(0),
            });
        }
        let total = max_bytes
            .checked_mul(world_size - 1)
            .ok_or_else(|| anyhow::anyhow!("one-shot receive buffer size overflows"))?;
        let mut recv: u64 = 0;
        let status = unsafe { cuMemAlloc_v2(&mut recv, total) };
        if status != 0 {
            anyhow::bail!(
                "cuMemAlloc_v2 for the one-shot receive slots ({total} bytes) failed: status {status}"
            );
        }
        let mut handle: *mut c_void = ptr::null_mut();
        let result =
            unsafe { nccl::ncclCommRegister(comm, recv as *mut c_void, total, &mut handle) };
        if result != NcclResult::Success {
            tracing::warn!(
                "ncclCommRegister for the one-shot receive slots failed (non-fatal): {result:?}"
            );
        }
        tracing::info!(
            "one-shot all-reduce enabled: payloads <= {max_bytes} B, {} receive slots",
            world_size - 1
        );
        Ok(Self {
            max_bytes,
            recv,
            sum_kernel: AtomicU64::new(0),
        })
    }

    pub(super) fn set_sum_kernel(&self, handle: u64) {
        self.sum_kernel.store(handle, Ordering::Relaxed);
    }

    pub(super) fn free(&self) {
        if self.recv != 0 {
            unsafe { cuMemFree_v2(self.recv) };
        }
    }
}

impl NcclBackend {
    /// 2026-10-08: Whether `bytes` takes the one-shot path on this communicator.
    pub(super) fn oneshot_applies(&self, bytes: usize) -> bool {
        oneshot_applies(self.world_size, self.oneshot.max_bytes, bytes)
    }

    /// 2026-10-11: The largest payload the one-shot path takes here: its buffer size when it
    /// applies on this communicator (two or more ranks, a nonzero buffer), else 0.
    pub(super) fn oneshot_capacity(&self) -> usize {
        if self.oneshot_applies(self.oneshot.max_bytes) {
            self.oneshot.max_bytes
        } else {
            0
        }
    }

    /// 2026-10-23: The state the path of an all-reduce depends on (`ar_path`).
    pub(super) fn path_state(&self) -> PathState {
        PathState {
            world_size: self.world_size,
            add_kernel_set: self.add_kernel.load(Ordering::Relaxed) != 0,
            sum_kernel_set: self.oneshot.sum_kernel.load(Ordering::Relaxed) != 0,
            oneshot_max_bytes: self.oneshot.max_bytes,
            recv_capacity: self.recv_capacity,
        }
    }

    /// 2026-10-16: The path an unplanned all-reduce of `bytes` takes on this communicator.
    pub(super) fn ar_path(&self, bytes: usize) -> CollectivePath {
        crate::ar_path::ar_path_of(&self.path_state(), bytes)
    }

    pub(super) fn set_oneshot_sum_kernel(&self, handle: u64) {
        self.oneshot.set_sum_kernel(handle);
    }

    /// 2026-10-08: One-shot all-reduce of `bytes` at `ptr` on `stream` (see the module header).
    /// Every rank must call it with the same `bytes`. 2026-10-10: The exchange is
    /// `exchange_oneshot`; this adds the `bf16_add_rank_sum` launch after it, the calls in the
    /// order they were made before.
    ///
    /// # Errors
    /// A payload over the bound, an NCCL failure, the sum kernel not set, or a failed launch.
    pub(super) fn all_reduce_oneshot(&self, ptr: u64, bytes: usize, stream: u64) -> Result<()> {
        ensure_payload_fits(bytes, self.oneshot.max_bytes, self.rank, self.world_size)?;
        let kernel = self.oneshot.sum_kernel.load(Ordering::Relaxed);
        if kernel == 0 {
            anyhow::bail!("bf16_add_rank_sum kernel not set — call set_rank_sum_kernel() first");
        }
        let partials = self.exchange_oneshot(ptr, bytes, stream)?;
        let count = bytes / ALL_REDUCE_DTYPE_BYTES;
        let threads: u32 = 256;
        let blocks: u32 = (count as u32).div_ceil(threads);
        let mut p_dst = ptr;
        let mut p_peers = partials.recv;
        let mut p_slot = partials.slot_elems as i64;
        let mut p_ranks = self.world_size as i32;
        let mut p_rank = self.rank as i32;
        let mut p_n = count as i32;
        let mut params: [*mut c_void; 6] = [
            &mut p_dst as *mut u64 as *mut c_void,
            &mut p_peers as *mut u64 as *mut c_void,
            &mut p_slot as *mut i64 as *mut c_void,
            &mut p_ranks as *mut i32 as *mut c_void,
            &mut p_rank as *mut i32 as *mut c_void,
            &mut p_n as *mut i32 as *mut c_void,
        ];
        let status = unsafe {
            cuLaunchKernel(
                kernel,
                blocks,
                1,
                1,
                threads,
                1,
                1,
                0,
                stream,
                params.as_mut_ptr(),
                ptr::null_mut(),
            )
        };
        if status != 0 {
            anyhow::bail!("cuLaunchKernel (bf16_add_rank_sum) failed: status {status}");
        }
        Ok(())
    }

    /// 2026-10-10: The exchange half of the one-shot all-reduce: send the `bytes` at `ptr` to every
    /// peer and receive every peer's partial into its slot, one grouped NCCL call on `stream`.
    /// Returns where the partials land; the caller sums them in rank order (`bf16_add_rank_sum`,
    /// or a consumer kernel with the same arithmetic). Every rank must call it with the same
    /// `bytes`.
    ///
    /// # Errors
    /// A payload over the bound or an NCCL failure.
    pub(super) fn exchange_oneshot(
        &self,
        ptr: u64,
        bytes: usize,
        stream: u64,
    ) -> Result<PeerPartials> {
        ensure_payload_fits(bytes, self.oneshot.max_bytes, self.rank, self.world_size)?;
        let count = bytes / ALL_REDUCE_DTYPE_BYTES;
        let comm = *self.comm.lock();
        let result = unsafe { nccl::ncclGroupStart() };
        nccl::check_nccl(result, "ncclGroupStart")?;
        for peer in 0..self.world_size {
            let Some(slot) = peer_slot(self.rank, peer) else {
                continue;
            };
            let dst = self.oneshot.recv + (slot * self.oneshot.max_bytes) as u64;
            let result = unsafe {
                nccl::ncclSend(
                    ptr as *const c_void,
                    count,
                    NcclDataType::Bfloat16,
                    peer as i32,
                    comm,
                    stream,
                )
            };
            nccl::check_nccl(result, "ncclSend")?;
            let result = unsafe {
                nccl::ncclRecv(
                    dst as *mut c_void,
                    count,
                    NcclDataType::Bfloat16,
                    peer as i32,
                    comm,
                    stream,
                )
            };
            nccl::check_nccl(result, "ncclRecv")?;
        }
        let result = unsafe { nccl::ncclGroupEnd() };
        nccl::check_nccl(result, "ncclGroupEnd")?;
        self.check_async_error(comm);
        Ok(PeerPartials {
            recv: self.oneshot.recv,
            slot_elems: self.oneshot.max_bytes / ALL_REDUCE_DTYPE_BYTES,
            world_size: self.world_size,
            rank: self.rank,
        })
    }

    /// 2026-09-26: 2-rank all-reduce: refuse a payload larger than
    /// `recv_capacity`, then a grouped `ncclSend`/`ncclRecv` with the partner
    /// rank, then `ptr[i] += recv_buffer[i]` with the BF16 add kernel on
    /// `stream`. Errors when the add kernel is not set.
    pub(super) fn all_reduce_2rank(&self, ptr: u64, bytes: usize, stream: u64) -> Result<()> {
        ensure_payload_fits(bytes, self.recv_capacity, self.rank, self.world_size)?;

        // 2026-09-26: Nothing to reduce, and a zero-block launch is invalid.
        // Both ranks skip the send/recv, provided both pass the same `bytes`.
        if bytes == 0 {
            return Ok(());
        }

        let count = bytes / ALL_REDUCE_DTYPE_BYTES;
        let partner = (1 - self.rank) as i32;
        let comm = *self.comm.lock();

        let result = unsafe { nccl::ncclGroupStart() };
        nccl::check_nccl(result, "ncclGroupStart")?;

        let result = unsafe {
            nccl::ncclSend(
                ptr as *const c_void,
                count,
                NcclDataType::Bfloat16,
                partner,
                comm,
                stream,
            )
        };
        nccl::check_nccl(result, "ncclSend")?;

        let result = unsafe {
            nccl::ncclRecv(
                self.recv_buffer as *mut c_void,
                count,
                NcclDataType::Bfloat16,
                partner,
                comm,
                stream,
            )
        };
        nccl::check_nccl(result, "ncclRecv")?;

        let result = unsafe { nccl::ncclGroupEnd() };
        nccl::check_nccl(result, "ncclGroupEnd")?;

        self.check_async_error(comm);

        let kernel = self.add_kernel.load(Ordering::Relaxed);
        if kernel != 0 {
            let threads: u32 = 256;
            let blocks: u32 = (count as u32).div_ceil(threads);
            let mut p_dst = ptr;
            let mut p_src = self.recv_buffer;
            let mut p_n = count as i32;
            let mut params: [*mut c_void; 3] = [
                &mut p_dst as *mut u64 as *mut c_void,
                &mut p_src as *mut u64 as *mut c_void,
                &mut p_n as *mut i32 as *mut c_void,
            ];
            let status = unsafe {
                cuLaunchKernel(
                    kernel,
                    blocks,
                    1,
                    1,
                    threads,
                    1,
                    1,
                    0,
                    stream,
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                )
            };
            if status != 0 {
                anyhow::bail!("cuLaunchKernel (bf16_add_inplace) failed: status {status}");
            }
        } else {
            anyhow::bail!("bf16_add_inplace kernel not set — call set_add_kernel() first");
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{oneshot_applies, peer_slot, validate_oneshot};
    // 2026-10-23: `oneshot_applies` is `crate::ar_path`'s (pure), shared with the fuser.

    #[test]
    fn every_peer_gets_a_distinct_slot_and_the_rank_none() {
        for world in 2..=8 {
            for rank in 0..world {
                let mut slots: Vec<usize> = (0..world).filter_map(|p| peer_slot(rank, p)).collect();
                assert_eq!(peer_slot(rank, rank), None);
                assert_eq!(slots.len(), world - 1);
                slots.sort_unstable();
                assert_eq!(
                    slots,
                    (0..world - 1).collect::<Vec<_>>(),
                    "world {world} rank {rank}"
                );
            }
        }
    }

    #[test]
    fn the_kernel_and_the_host_agree_on_slots() {
        // 2026-10-08: bf16_add_rank_sum reads peer r at slot (r < my_rank ? r : r - 1).
        for rank in 0..5 {
            for peer in 0..5usize {
                let kernel_slot = if peer < rank {
                    peer
                } else {
                    peer.wrapping_sub(1)
                };
                if peer != rank {
                    assert_eq!(peer_slot(rank, peer), Some(kernel_slot));
                }
            }
        }
    }

    /// 2026-10-16: Admitted from two ranks up (was three), each world at and over the bound.
    #[test]
    fn oneshot_admits_only_small_payloads_at_two_or_more_ranks() {
        for world in [2, 3, 8] {
            assert!(
                oneshot_applies(world, 1024, 1024),
                "world {world} at the bound"
            );
            assert!(
                oneshot_applies(world, 1024, 2),
                "world {world}, one element"
            );
            assert!(
                !oneshot_applies(world, 1024, 1026),
                "world {world}: over the bound goes to the 2-rank exchange or NCCL"
            );
            assert!(
                !oneshot_applies(world, 1024, 0),
                "world {world}: an empty payload sends nothing"
            );
            assert!(!oneshot_applies(world, 0, 512), "world {world}: disabled");
        }
        assert!(
            !oneshot_applies(1, 1024, 512),
            "a single rank has no peer to exchange with"
        );
    }

    /// 2026-10-16: A nonzero bound is refused on one rank and accepted from two up; a bound of 0
    /// is accepted everywhere; a bound that splits a BF16 element is refused at every world.
    #[test]
    fn a_oneshot_bound_is_refused_only_where_it_cannot_run() {
        assert!(validate_oneshot(1, 0).is_ok(), "disabled on one rank");
        let one = validate_oneshot(1, 1024).expect_err("one rank has no peer");
        assert!(one.to_string().contains("world_size >= 2"), "{one}");
        for world in [2, 3, 8] {
            assert!(validate_oneshot(world, 0).is_ok(), "world {world} disabled");
            assert!(
                validate_oneshot(world, 2).is_ok(),
                "world {world}, one element"
            );
            assert!(
                validate_oneshot(world, 1 << 20).is_ok(),
                "world {world}, 1 MiB"
            );
            let odd = validate_oneshot(world, 1025).expect_err("half an element");
            assert!(odd.to_string().contains("whole number"), "{odd}");
        }
    }
}

#[cfg(test)]
#[path = "world2_sum_tests.rs"]
mod world2_sum_tests;
