// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: one writer per durable directory (DESIGN §3.1 step 2).
//!
//! The lease is an exclusive advisory lock (`File::try_lock`: `flock(LOCK_EX |
//! LOCK_NB)` on Linux and macOS, `LockFileEx` on Windows) on the file
//! `<durable_dir>/.lock`. The operating system decides who holds it: a holder
//! keeps it however long it runs, and a holder that dies (even by `SIGKILL`)
//! loses it when its descriptor closes. There is no age-based staleness and
//! no fallback: on a platform without file locks `acquire` fails.
//!
//! The holder writes `pid=<pid>\nsince=<unix s>\n` into the lock file and
//! empties it on release. That statement is informational only (it names the
//! holder in a busy message); it never decides anything, and a crashed
//! holder's leftover statement is simply overwritten.
//!
//! The lock file is never deleted, by this code or by anyone else: deleting
//! it while it is held lets a second writer lock a new file of the same name.
//!
//! A held lock is waited on for `WAIT_S`, then the caller gets `Busy`: the
//! run stays in the spool, not materialised (exit 3; `metprof kb replay`
//! catches up).

use std::fs::{self, File, OpenOptions, TryLockError};
use std::io::{Seek, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::{Error, Result, atomic, bail};

pub const LOCK_FILE: &str = ".lock";
pub const WAIT_S: u64 = 30;
pub const POLL: Duration = Duration::from_millis(500);

/// What the lease needs from the machine (a port: tests drive a fake clock).
pub trait LeaseEnv {
    fn pid(&self) -> u32;
    /// Seconds since the Unix epoch.
    fn now_s(&self) -> Result<u64>;
    fn sleep(&self, d: Duration);
}

/// The real machine.
pub struct SystemLeaseEnv;

impl LeaseEnv for SystemLeaseEnv {
    fn pid(&self) -> u32 {
        std::process::id()
    }

    fn now_s(&self) -> Result<u64> {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .map_err(|e| Error(format!("lease: the clock is before the Unix epoch: {e}")))
    }

    fn sleep(&self, d: Duration) {
        std::thread::sleep(d);
    }
}

/// A lease holder as its lock file states it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Holder {
    pub pid: u32,
    pub since: u64,
}

impl Holder {
    pub fn encode(&self) -> String {
        format!("pid={}\nsince={}\n", self.pid, self.since)
    }

    pub fn parse(text: &str) -> Result<Holder> {
        let mut pid = None;
        let mut since = None;
        for line in text.lines() {
            match line.split_once('=') {
                Some(("pid", v)) => pid = v.parse::<u32>().ok(),
                Some(("since", v)) => since = v.parse::<u64>().ok(),
                _ => bail!("lease holder: unexpected line {line:?}"),
            }
        }
        match (pid, since) {
            (Some(pid), Some(since)) => Ok(Holder { pid, since }),
            _ => bail!("lease holder: pid and since are required"),
        }
    }
}

/// The outcome of `acquire`.
#[derive(Debug)]
pub enum Acquired {
    Held(Lease),
    /// Another writer held the lock for the whole wait (as it states itself,
    /// when its statement is readable).
    Busy(Option<Holder>),
}

/// A held lease; `release` gives it back (dropping it does the same).
#[derive(Debug)]
pub struct Lease {
    file: File,
    path: PathBuf,
    holder: Holder,
    released: bool,
}

fn io(what: &str, path: &Path, e: std::io::Error) -> Error {
    Error(format!("lease: {what} {}: {e}", path.display()))
}

/// The holder a lock file states; anything unreadable states nobody.
pub fn read_holder(lock_path: &Path) -> Option<Holder> {
    fs::read_to_string(lock_path)
        .ok()
        .and_then(|t| Holder::parse(&t).ok())
}

/// Take the lease on `durable_dir`, waiting up to `WAIT_S` for its holder.
pub fn acquire(durable_dir: &Path, env: &dyn LeaseEnv) -> Result<Acquired> {
    atomic::create_dir(durable_dir)?;
    let path = durable_dir.join(LOCK_FILE);
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&path)
        .map_err(|e| io("opening", &path, e))?;
    let deadline = env.now_s()? + WAIT_S;
    loop {
        match file.try_lock() {
            Ok(()) => break,
            Err(TryLockError::WouldBlock) => {}
            Err(TryLockError::Error(e)) => return Err(io("locking", &path, e)),
        }
        if env.now_s()? >= deadline {
            return Ok(Acquired::Busy(read_holder(&path)));
        }
        env.sleep(POLL);
    }
    let holder = Holder {
        pid: env.pid(),
        since: env.now_s()?,
    };
    let lease = Lease {
        file,
        path,
        holder,
        released: false,
    };
    lease.state(holder.encode().as_bytes())?;
    Ok(Acquired::Held(lease))
}

impl Lease {
    pub fn holder(&self) -> Holder {
        self.holder
    }

    /// Replace the lock file's statement (the lock is on the descriptor, so
    /// rewriting the content never drops it).
    fn state(&self, text: &[u8]) -> Result<()> {
        let mut f = &self.file;
        f.set_len(0)
            .and_then(|()| f.rewind())
            .and_then(|()| f.write_all(text))
            .and_then(|()| f.sync_data())
            .map_err(|e| io("writing", &self.path, e))
    }

    /// Give the lease back: empty the statement; the lock goes when the
    /// descriptor closes as `self` drops, never before the statement is
    /// emptied (so a next holder's statement is never touched). The lock is
    /// released even when emptying fails (the error is still returned).
    pub fn release(mut self) -> Result<()> {
        self.released = true;
        self.state(b"")
    }
}

impl Drop for Lease {
    /// Closing the descriptor releases the lock; the statement is emptied
    /// first (`release` already did, and reported any error).
    fn drop(&mut self) {
        if !self.released
            && let Err(e) = self.state(b"")
        {
            eprintln!("metprof-kb: {e}");
        }
    }
}
