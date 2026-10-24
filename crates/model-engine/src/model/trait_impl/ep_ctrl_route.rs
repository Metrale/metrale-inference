// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: The model's side of the control channel (`METRALE_GLM_EP_CTRL_COMM`,
//! `model/ep_ctrl.rs`): attaching it, and rank 0's sends and a worker's receives of the command
//! blocks and K=2 verdicts it carries. Each helper routes by [`ep_ctrl::route`]; with the lever
//! off its caller sends or receives on the main communicator as before.
//!
//! Owner: model-engine (EP worker protocol).
//! Invariants: the ones in `model/ep_ctrl.rs`.

use std::sync::Arc;

use anyhow::{Context, Result, anyhow, ensure};
use metrale_comm::CommBackend;
use metrale_model_layers::ep_wire::{self, CmdBlock, EpWire};
use metrale_model_layers::speculative::{K2_VERDICT_WORDS, K2Verdict};

use super::super::ep_ctrl::{self, CtrlChannel, DeviceArg, Kind, Route};
use super::super::types::TransformerModel;

impl TransformerModel {
    /// 2026-10-21: The control channel, when the multi-rank protocol is live and the lever
    /// routes control messages to it; `None` sends them on the main communicator.
    pub(in crate::model) fn ctrl_channel(&self) -> Result<Option<&CtrlChannel>> {
        if !self.multi_rank_protocol_active() {
            return Ok(None);
        }
        let attached = self.ctrl.get();
        Ok(
            match ep_ctrl::route(self.settings.runtime.glm_ep_ctrl_comm, attached.is_some())? {
                Route::Main => None,
                Route::Ctrl => attached,
            },
        )
    }

    /// 2026-10-21: Attach the control channel: a greatest-priority stream, the communicator the
    /// host bootstraps on it (`bootstrap(stream)`), and the channel's buffers. Every rank calls
    /// it once, after the rank agreement and before the first command.
    pub(in crate::model) fn attach_ctrl_comm_impl(
        &self,
        bootstrap: &mut dyn FnMut(u64) -> Result<Arc<dyn CommBackend>>,
    ) -> Result<()> {
        ensure!(
            self.settings.runtime.glm_ep_ctrl_comm,
            "attach_ctrl_comm with {} off",
            ep_ctrl::ENV
        );
        ep_ctrl::check_config(true, EpWire::get())?;
        ensure!(
            self.multi_rank_protocol_active() && self.ep_protocol_v2,
            "{} needs the multi-rank protocol on the v3 wire",
            ep_ctrl::ENV
        );
        let main = self
            .comm
            .clone()
            .context("a control channel without a communicator")?;
        let stream = self.gpu.create_stream_greatest_priority()?;
        let comm = bootstrap(stream).context("bootstrapping the control communicator")?;
        let channel = CtrlChannel::new(self.gpu.as_ref(), comm, main, stream)?;
        self.ctrl
            .set(channel)
            .map_err(|_| anyhow!("the control channel is already attached"))?;
        tracing::info!(
            "{}: control channel attached (stream {stream:#x})",
            ep_ctrl::ENV
        );
        Ok(())
    }

    /// 2026-10-21: Rank 0: a command block on the control channel. `Ok(false)` when the route
    /// is the main communicator: nothing was sent, and the caller sends as before.
    pub(in crate::model) fn ctrl_send_block(
        &self,
        block: &[u8; ep_wire::CMD_BLOCK_BYTES],
        arg: Option<DeviceArg>,
    ) -> Result<bool> {
        let Some(c) = self.ctrl_channel()? else {
            return Ok(false);
        };
        c.send(self.gpu.as_ref(), Kind::Block, block, arg)?;
        Ok(true)
    }

    /// 2026-10-21: Rank 0: the K=2 verdict on the control channel; `Ok(false)` as
    /// [`Self::ctrl_send_block`].
    pub(in crate::model) fn ctrl_send_verdict(&self, verdict: K2Verdict) -> Result<bool> {
        let Some(c) = self.ctrl_channel()? else {
            return Ok(false);
        };
        let bytes: Vec<u8> = verdict
            .to_words()
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        c.send(self.gpu.as_ref(), Kind::Verdict, &bytes, None)?;
        Ok(true)
    }

    /// 2026-10-21: A worker: the next command block from the control channel, `None` when the
    /// route is the main communicator. A block whose argument a kernel reads from the command
    /// buffer ([`ep_ctrl::needs_device_mirror`]) is copied there from the host, on the compute
    /// stream.
    pub(in crate::model) fn ctrl_recv_block(&self) -> Result<Option<CmdBlock>> {
        let Some(c) = self.ctrl_channel()? else {
            return Ok(None);
        };
        let payload = c.recv(self.gpu.as_ref(), Kind::Block)?;
        let bytes: [u8; ep_wire::CMD_BLOCK_BYTES] = payload
            .as_slice()
            .try_into()
            .context("control block payload size")?;
        let block = ep_wire::parse_block(&bytes);
        if ep_ctrl::needs_device_mirror(&block) {
            self.gpu
                .copy_h2d_async(&bytes, self.ep_cmd_buf, self.gpu.default_stream())?;
        }
        Ok(Some(block))
    }

    /// 2026-10-21: A worker: rank 0's K=2 verdict words, from the channel rank 0 sent them on.
    pub(in crate::model) fn ep_recv_k2_verdict_words(&self) -> Result<Vec<u32>> {
        let Some(c) = self.ctrl_channel()? else {
            return self.ep_broadcast_tokens(&[0u32; K2_VERDICT_WORDS]);
        };
        let payload = c.recv(self.gpu.as_ref(), Kind::Verdict)?;
        Ok(payload
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect())
    }

    /// 2026-10-21: Whether the communicators are usable: both when a control channel is
    /// attached. When one of the two is not, the other is poisoned too, so every rank stops.
    pub(in crate::model) fn comms_healthy(&self) -> bool {
        let main = self.comm.as_ref().is_none_or(|c| c.is_healthy());
        match self.ctrl.get() {
            None => main,
            Some(c) => {
                let both = main && c.is_healthy();
                if !both {
                    c.poison_both();
                }
                both
            }
        }
    }
}
