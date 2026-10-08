// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: The send/recv all-reduces: the 2-rank exchange (moved here unchanged from
//! `nccl_backend.rs`) and the one-shot exchange at `world_size >= 3`.
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
//! - Enabled only at `world_size >= 3`; `world_size == 2` keeps its own exchange.

use anyhow::Result;
use std::ffi::c_void;
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};

use metrale_gpu_sys::nccl::{self, NcclComm, NcclDataType, NcclResult};

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

/// 2026-10-08: Whether a payload of `bytes` takes the one-shot path. A free function for the tests.
pub(super) fn oneshot_applies(world_size: usize, max_bytes: usize, bytes: usize) -> bool {
    world_size >= 3 && max_bytes > 0 && bytes > 0 && bytes <= max_bytes
}

impl OneShot {
    /// 2026-10-08: Allocate and register the receive slots when `max_bytes > 0`.
    ///
    /// # Errors
    /// `max_bytes > 0` at `world_size < 3`, a `max_bytes` that is not a whole number of BF16
    /// elements, or a failed allocation. A failed registration is only logged, as for the 2-rank
    /// buffer.
    pub(super) fn new(comm: NcclComm, world_size: usize, max_bytes: usize) -> Result<Self> {
        if max_bytes == 0 {
            return Ok(Self {
                max_bytes: 0,
                recv: 0,
                sum_kernel: AtomicU64::new(0),
            });
        }
        if world_size < 3 {
            anyhow::bail!(
                "the one-shot all-reduce is for world_size >= 3 (world_size {world_size}); \
                 world_size 2 has its own send/recv exchange"
            );
        }
        if !max_bytes.is_multiple_of(ALL_REDUCE_DTYPE_BYTES) {
            anyhow::bail!(
                "one-shot all-reduce bound {max_bytes} B is not a whole number of BF16 elements"
            );
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

    pub(super) fn set_oneshot_sum_kernel(&self, handle: u64) {
        self.oneshot.set_sum_kernel(handle);
    }

    /// 2026-10-08: One-shot all-reduce of `bytes` at `ptr` on `stream` (see the module header).
    /// Every rank must call it with the same `bytes`.
    ///
    /// # Errors
    /// A payload over the bound, an NCCL failure, the sum kernel not set, or a failed launch.
    pub(super) fn all_reduce_oneshot(&self, ptr: u64, bytes: usize, stream: u64) -> Result<()> {
        ensure_payload_fits(bytes, self.oneshot.max_bytes, self.rank, self.world_size)?;
        let kernel = self.oneshot.sum_kernel.load(Ordering::Relaxed);
        if kernel == 0 {
            anyhow::bail!("bf16_add_rank_sum kernel not set — call set_rank_sum_kernel() first");
        }
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

        let threads: u32 = 256;
        let blocks: u32 = (count as u32).div_ceil(threads);
        let mut p_dst = ptr;
        let mut p_peers = self.oneshot.recv;
        let mut p_slot = (self.oneshot.max_bytes / ALL_REDUCE_DTYPE_BYTES) as i64;
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
    use super::{oneshot_applies, peer_slot};

    #[test]
    fn every_peer_gets_a_distinct_slot_and_the_rank_none() {
        for world in 3..=8 {
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

    #[test]
    fn oneshot_admits_only_small_payloads_at_three_or_more_ranks() {
        assert!(oneshot_applies(3, 1024, 1024));
        assert!(oneshot_applies(3, 1024, 2));
        assert!(
            !oneshot_applies(3, 1024, 1026),
            "over the bound goes to NCCL"
        );
        assert!(
            !oneshot_applies(3, 1024, 0),
            "an empty payload sends nothing"
        );
        assert!(
            !oneshot_applies(2, 1024, 512),
            "world 2 keeps its own exchange"
        );
        assert!(!oneshot_applies(3, 0, 512), "disabled");
    }
}
