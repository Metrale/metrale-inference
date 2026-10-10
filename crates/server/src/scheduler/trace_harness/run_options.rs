// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: [`RunOptions`], the scheduler settings one scenario runs with (moved out of
//! `runner.rs` unchanged, to keep that file under the size cap).
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

#[derive(Clone, Debug)]
pub(super) struct RunOptions {
    pub max_batch_size: usize,
    pub use_speculative: bool,
    pub dflash: bool,
    pub num_drafts: usize,
    pub max_prefill_tokens: usize,
    pub max_batch_tokens: usize,
    pub self_speculative: bool,
    pub ngram_speculative: bool,
    pub prompt_lookup: Option<metrale_speculative::prompt_lookup::PromptLookupConfig>,
    /// 2026-10-04: The cross-request prompt-lookup cache; `None` when off.
    pub shared_lookup: Option<crate::scheduler::shared_lookup_step::SharedLookupSetup>,
    pub swap_space_gb: usize,
    pub slai_policy: bool,
    pub mtp_gate_force: bool,
    /// 2026-10-09: The `--prefill-codispatch` and `METRALE_EP_PREFILL_BATCH` levers.
    pub prefill_codispatch: bool,
    pub ep_prefill_batch: bool,
    pub loop_watchdog: bool,
    /// 2026-09-25: Token ids the watchdog rollback treats as boundaries.
    pub boundary_tokens: Vec<u32>,
    pub think_end_token: Option<u32>,
    pub think_start_token: Option<u32>,
    /// 2026-09-25: A LoRA rotation queued before the loop starts; the scheduler applies it
    /// once nothing is in flight.
    pub lora_rotation: Option<String>,
    /// 2026-10-03: Close the request channel while the loop is parked at the first model call
    /// whose trace line starts with this prefix, then let it go on (the D15 race, forced).
    pub close_inbox_at: Option<&'static str>,
    /// 2026-09-25: The instrument set the run feeds. The goldens are recorded with a
    /// never-configured (`Off`) one.
    pub telemetry: &'static metrale_telemetry::Telemetry,
    /// 2026-09-25: Fault injection for the pipelined lane (the negative controls).
    pub pipeline_faults: crate::scheduler::PipelineFaults,
}

/// 2026-09-25: Never configured: every feed into it takes the `Off` branch.
static TELEMETRY_OFF: metrale_telemetry::Telemetry =
    metrale_telemetry::Telemetry::new(&metrale_telemetry::clock::MonotonicClock);

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            use_speculative: false,
            dflash: false,
            num_drafts: 1,
            max_prefill_tokens: 0,
            max_batch_tokens: 8192,
            self_speculative: false,
            ngram_speculative: false,
            prompt_lookup: None,
            shared_lookup: None,
            swap_space_gb: 0,
            slai_policy: false,
            mtp_gate_force: true,
            prefill_codispatch: false,
            ep_prefill_batch: false,
            loop_watchdog: false,
            boundary_tokens: Vec::new(),
            think_end_token: None,
            think_start_token: None,
            lora_rotation: None,
            close_inbox_at: None,
            telemetry: &TELEMETRY_OFF,
            pipeline_faults: crate::scheduler::PipelineFaults::NONE,
        }
    }
}
