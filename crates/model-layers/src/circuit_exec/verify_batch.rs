// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The batched MTP verify's programs (`Mode::VerifyBatch`). A batched verify's
//! plan depends on its row table (the runs of equal `k` and their contiguity), and a serve
//! meets many tables, so each program is compiled the first time its table is verified and kept
//! in a bounded cache. The legacy forward's graphs are keyed by the batch's `(slot, k)` pairs,
//! which fix the table, so a graph captured around a program replays that program's launches.
//!
//! Owner: model-layers circuit executor.
//! Invariants:
//! - Every program is laid out from the shared workspace base, and the build sized the workspace
//!   for the widest table it admits (`max_rows`); a larger table is refused, never placed past
//!   the workspace.
//! - The cache holds at most [`CACHE_CAP`] programs; a miss compiles, an eviction drops the least
//!   recently used.

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use metrale_circuit::planner::plan_buffers_with;
use metrale_circuit::{AvailableKernels, Circuit, FusionPlan, Policy, RowTable, Rule, fuse_table};
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use parking_lot::Mutex;

use super::bindings::{CircuitLayer, HeadBinding};
use super::compile::{self, Fixed};
use super::kernels::KernelTable;
use super::program::Program;

/// 2026-09-30: Programs kept per executor.
pub const CACHE_CAP: usize = 256;

/// 2026-09-30: A compiled batched-verify program and its plan.
pub struct VerifyBatchProgram {
    pub program: Program,
    pub plan: FusionPlan,
}

/// 2026-09-30: What compiling a batched-verify program reads, kept from the build.
pub struct VerifyBatch {
    circuit: Circuit,
    rules: Vec<Rule>,
    available: AvailableKernels,
    policy: Policy,
    kernels: KernelTable,
    fixed: Fixed,
    layers: Vec<CircuitLayer>,
    head: HeadBinding,
    config: ModelConfig,
    workspace: DevicePtr,
    workspace_bytes: u64,
    /// 2026-09-30: The widest table admitted, in rows.
    pub max_rows: u64,
    cache: Mutex<(u64, BTreeMap<RowTable, (u64, Arc<VerifyBatchProgram>)>)>,
}

/// 2026-09-30: The tables the workspace is sized over: for every row count `4..=max_rows`, one
/// table of that many rows (runs of 2, with one run of 3 for an odd count).
pub fn sizing_tables(max_rows: u64) -> Vec<RowTable> {
    (4..=max_rows)
        .map(|r| {
            let text = if r % 2 == 0 {
                format!("2x{}", r / 2)
            } else {
                // 2026-10-01: A run of one sequence is never batched, so it is marked `!`.
                let tail = (r - 3) / 2;
                format!("3x1! 2x{tail}{}", if tail == 1 { "!" } else { "" })
            };
            RowTable::parse(&text).expect("a well-formed sizing table")
        })
        .collect()
}

/// 2026-09-30: The arena bytes of `table`'s plan.
pub fn arena_bytes(
    circuit: &Circuit,
    rules: &[Rule],
    available: &AvailableKernels,
    policy: &Policy,
    table: &RowTable,
) -> Result<u64> {
    let plan = fuse_table(circuit, rules, available, policy, table)
        .with_context(|| format!("fusing the batched verify of `{table}`"))?;
    let layout = compile::layout(circuit, &plan)?;
    Ok(plan_buffers_with(circuit, &plan, plan.rows, &layout)?.arena_bytes)
}

/// 2026-09-30: The build's state a batched-verify compile reads.
pub struct Parts {
    pub circuit: Circuit,
    pub rules: Vec<Rule>,
    pub available: AvailableKernels,
    pub policy: Policy,
    pub kernels: KernelTable,
    pub fixed: Fixed,
    pub layers: Vec<CircuitLayer>,
    pub head: HeadBinding,
    pub config: ModelConfig,
}

impl VerifyBatch {
    /// 2026-09-30: The compiler over `parts`, in a workspace of `workspace_bytes` at
    /// `workspace`, admitting tables of up to `max_rows` rows.
    pub fn new(parts: Parts, workspace: DevicePtr, workspace_bytes: u64, max_rows: u64) -> Self {
        let Parts {
            circuit,
            rules,
            available,
            policy,
            kernels,
            fixed,
            layers,
            head,
            config,
        } = parts;
        VerifyBatch {
            circuit,
            rules,
            available,
            policy,
            kernels,
            fixed,
            layers,
            head,
            config,
            workspace,
            workspace_bytes,
            max_rows,
            cache: Mutex::new((0, BTreeMap::new())),
        }
    }

    /// 2026-09-30: The program for `table`, compiled on a miss.
    pub fn program(
        &self,
        gpu: &dyn GpuBackend,
        table: &RowTable,
    ) -> Result<Arc<VerifyBatchProgram>> {
        ensure!(
            table.rows() <= self.max_rows,
            "a batched verify of {} rows is past the {} this executor was sized for",
            table.rows(),
            self.max_rows
        );
        {
            let mut c = self.cache.lock();
            c.0 += 1;
            let tick = c.0;
            if let Some(e) = c.1.get_mut(table) {
                e.0 = tick;
                return Ok(e.1.clone());
            }
        }
        let plan = fuse_table(
            &self.circuit,
            &self.rules,
            &self.available,
            &self.policy,
            table,
        )
        .with_context(|| format!("fusing the batched verify of `{table}`"))?;
        let layout = compile::layout(&self.circuit, &plan)?;
        let buffers = plan_buffers_with(&self.circuit, &plan, plan.rows, &layout)?;
        ensure!(
            buffers.arena_bytes <= self.workspace_bytes,
            "the batched verify of `{table}` needs {} workspace bytes; the build sized {}",
            buffers.arena_bytes,
            self.workspace_bytes
        );
        let inputs = compile::Inputs {
            gpu,
            config: &self.config,
            kernels: &self.kernels,
            fixed: &self.fixed,
            layers: &self.layers,
            head: &self.head,
            draft: None,
            arena: None,
        };
        let program = compile::compile(
            &self.circuit,
            &plan,
            &layout,
            &buffers,
            self.workspace,
            &inputs,
        )
        .with_context(|| format!("compiling the batched verify of `{table}`"))?;
        let entry = Arc::new(VerifyBatchProgram { program, plan });
        let mut c = self.cache.lock();
        let tick = c.0;
        if c.1.len() >= CACHE_CAP
            && let Some(oldest) =
                c.1.iter()
                    .min_by_key(|(_, (t, _))| *t)
                    .map(|(k, _)| k.clone())
        {
            c.1.remove(&oldest);
        }
        c.1.insert(table.clone(), (tick, entry.clone()));
        Ok(entry)
    }

    /// 2026-09-30: The metadata the programs read (`Fixed::verify_batch_meta`).
    pub fn fixed_meta(&self) -> &crate::layer::AttnMetadataDev {
        &self.fixed.verify_batch_meta
    }

    /// 2026-09-30: Where the programs write each row's token (`Fixed::verify_batch_tokens`).
    pub fn fixed_tokens(&self) -> DevicePtr {
        self.fixed.verify_batch_tokens
    }

    /// 2026-09-30: Programs cached now.
    pub fn cached(&self) -> usize {
        self.cache.lock().1.len()
    }
}
