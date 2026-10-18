// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: Inert `Starter`s for tests. Neither spawns a thread, opens a
//! connection or touches the disk; each records the repo it was asked for, so
//! a test can prove `DownloadState::start` went through the port.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use std::cell::RefCell;
use std::path::PathBuf;
use std::sync::mpsc::Sender;

use super::DownloadState;
use crate::model_download::{DownloadError, DownloadMsg, Handle};

thread_local! {
    static STARTED: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
    /// 2026-10-18: The senders of `pending` jobs, held so their channels stay open.
    static OPEN: RefCell<Vec<Sender<DownloadMsg>>> = const { RefCell::new(Vec::new()) };
}

/// 2026-10-18: A job that never settles: `pump` sees an open, empty channel.
pub(crate) fn pending(repo: &str, _cache_root: PathBuf) -> Handle {
    let (handle, tx, _cancel) = Handle::open(repo);
    STARTED.with(|s| s.borrow_mut().push(repo.to_string()));
    OPEN.with(|o| o.borrow_mut().push(tx));
    handle
}

/// 2026-10-18: A job that fails at once as `Offline`, as an unreachable Hub does.
pub(crate) fn offline(repo: &str, _cache_root: PathBuf) -> Handle {
    let (handle, tx, _cancel) = Handle::open(repo);
    STARTED.with(|s| s.borrow_mut().push(repo.to_string()));
    tx.send(DownloadMsg::Failed(DownloadError::Offline(
        "inert test starter".into(),
    )))
    .expect("the handle holds the receiver");
    handle
}

/// 2026-10-18: The repos this thread's inert starters were asked for, oldest
/// first; the list is emptied.
pub(crate) fn take_started() -> Vec<String> {
    STARTED.with(|s| std::mem::take(&mut *s.borrow_mut()))
}

/// 2026-10-18: A `DownloadState` whose jobs never settle.
pub(crate) fn state() -> DownloadState {
    DownloadState::new(pending)
}
