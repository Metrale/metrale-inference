// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: `ModelCircuit`, the supertrait of `Model` that selects which forward decode runs:
//! the hand-written layer loops (`legacy`) or a program compiled from the model's circuit plan
//! (`circuit`), and discloses the choice for records.
//!
//! Owner: model-engine.
//! Invariants:
//! - The default is the legacy forward, and a model without a circuit executor refuses
//!   `circuit` rather than running legacy under that name.

use anyhow::{Result, bail};
use metrale_circuit::Instance;
use metrale_model_layers::circuit_exec::{Fusions, TargetModules};

/// 2026-09-28: The forward a model's decode runs.
#[derive(Debug, Clone)]
pub enum ForwardSelect {
    /// 2026-09-28: The hand-written layer loops.
    Legacy,
    /// 2026-09-28: The program compiled from `instance`'s circuit.
    Circuit {
        /// 2026-09-28: The circuit instance serving this checkpoint on this target.
        instance: Box<Instance>,
        /// 2026-09-28: Which rules the plan may select.
        fusions: Fusions,
        /// 2026-09-28: The served target's compiled modules: which kernels exist.
        modules: TargetModules,
        /// 2026-09-30: The checkpoint's `config.json` as the server read it; the executor maps it
        /// to the served shape through the circuit's config map.
        config_json: String,
    },
}

/// 2026-09-28: What a record states about the forward.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForwardDisclosure {
    /// 2026-09-28: `legacy`, `circuit`, or `circuit-reference` (no `bit_identical` fusions).
    pub forward: &'static str,
    /// 2026-09-28: The digest over every compiled plan (decode, then each multi-sequence
    /// width; `CircuitExec::plans_digest`); `None` under legacy.
    pub plan_digest: Option<String>,
    /// 2026-09-28: Launches per decode step of the plan; `None` under legacy.
    pub launches_per_step: Option<usize>,
}

impl ForwardDisclosure {
    /// 2026-09-28: The legacy forward.
    pub fn legacy() -> Self {
        ForwardDisclosure {
            forward: "legacy",
            plan_digest: None,
            launches_per_step: None,
        }
    }
}

/// 2026-09-28: Forward selection; see the module header.
pub trait ModelCircuit {
    /// 2026-09-28: Make decode run `sel` from the next step. Captured decode graphs are dropped.
    /// Default: only `Legacy` is accepted.
    fn set_forward(&self, sel: &ForwardSelect) -> Result<()> {
        match sel {
            ForwardSelect::Legacy => Ok(()),
            ForwardSelect::Circuit { .. } => bail!("this model has no circuit executor"),
        }
    }

    /// 2026-09-28: The forward decode runs now. Default: legacy.
    fn forward_disclosure(&self) -> ForwardDisclosure {
        ForwardDisclosure::legacy()
    }

    /// 2026-10-03: A digest of `seq`'s state, after the work queued so far completes: per
    /// recurrent layer its h and conv bytes, per attention layer the K and V bytes of the
    /// sequence's blocks in block-table order. The parity instruments compare two runs' state
    /// with it. Default: refused.
    fn state_digest(&self, _seq: &crate::traits::SequenceState) -> Result<Vec<(String, u64)>> {
        bail!("this model does not digest its state")
    }
}

/// 2026-10-03: 64-bit FNV-1a of `bytes`, continuing `acc` (start from [`FNV_START`]).
pub fn fnv1a(acc: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(acc, |a, &b| (a ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3))
}

/// 2026-10-03: The FNV-1a offset basis.
pub const FNV_START: u64 = 0xcbf2_9ce4_8422_2325;
