// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: one writer per durable directory (DESIGN §3.1 step 2).
//!
//! The lease is the directory `<durable_dir>/.lease`, made with `create_dir`
//! (atomic on POSIX), holding `holder` = `pid=<pid>\nsince=<unix s>\n`. A
//! lease is stale when its pid is dead or it is older than `STALE_AFTER_S`;
//! a stale lease is taken by renaming it aside (checked afterwards: a lease
//! that changed hands in between is put back, never broken). A live lease is
//! waited on for `WAIT_S`, then the caller gets `Busy`: the run stays in the
//! spool, not materialised (exit 3; `metprof kb replay` catches up).

use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::{Error, Result, atomic, bail};

pub const LEASE_DIR: &str = ".lease";
pub const HOLDER_FILE: &str = "holder";
pub const STALE_AFTER_S: u64 = 600;
pub const WAIT_S: u64 = 30;
pub const POLL: Duration = Duration::from_millis(500);

/// What the lease needs from the machine (a port: tests drive a fake clock).
pub trait LeaseEnv {
    fn pid(&self) -> u32;
    fn alive(&self, pid: u32) -> bool;
    /// Seconds since the Unix epoch.
    fn now_s(&self) -> u64;
    fn sleep(&self, d: Duration);
}

/// The real machine.
pub struct SystemLeaseEnv;

impl LeaseEnv for SystemLeaseEnv {
    fn pid(&self) -> u32 {
        std::process::id()
    }

    /// Linux answers from `/proc`; elsewhere every pid counts as alive, so a
    /// lease there goes stale by age only.
    fn alive(&self, pid: u32) -> bool {
        !cfg!(target_os = "linux") || Path::new("/proc").join(pid.to_string()).exists()
    }

    fn now_s(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }

    fn sleep(&self, d: Duration) {
        std::thread::sleep(d);
    }
}

/// A lease holder as its file states it.
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
    /// Another live writer held the lease for the whole wait.
    Busy(Option<Holder>),
}

/// A held lease; `release` gives it back (dropping it does the same).
#[derive(Debug)]
pub struct Lease {
    dir: PathBuf,
    holder: Holder,
    released: bool,
}

fn io(what: &str, path: &Path, e: std::io::Error) -> Error {
    Error(format!("lease: {what} {}: {e}", path.display()))
}

fn read_holder(lease_dir: &Path) -> Result<Option<Holder>> {
    match fs::read_to_string(lease_dir.join(HOLDER_FILE)) {
        Ok(text) => Holder::parse(&text).map(Some),
        Err(e) if e.kind() == ErrorKind::NotFound => Ok(None),
        Err(e) => Err(io("reading", lease_dir, e)),
    }
}

/// A lease directory's age by its mtime (a holder not yet written).
fn dir_age_s(lease_dir: &Path, now_s: u64) -> Result<Option<u64>> {
    match fs::metadata(lease_dir).and_then(|m| m.modified()) {
        Ok(t) => {
            let at = t
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            Ok(Some(now_s.saturating_sub(at)))
        }
        Err(e) if e.kind() == ErrorKind::NotFound => Ok(None),
        Err(e) => Err(io("reading", lease_dir, e)),
    }
}

/// Take the lease on `durable_dir`, waiting up to `WAIT_S` for a live holder.
pub fn acquire(durable_dir: &Path, env: &dyn LeaseEnv) -> Result<Acquired> {
    fs::create_dir_all(durable_dir).map_err(|e| io("creating", durable_dir, e))?;
    let lease_dir = durable_dir.join(LEASE_DIR);
    let deadline = env.now_s() + WAIT_S;
    loop {
        match fs::create_dir(&lease_dir) {
            Ok(()) => {
                let holder = Holder {
                    pid: env.pid(),
                    since: env.now_s(),
                };
                atomic::write(&lease_dir, HOLDER_FILE, holder.encode().as_bytes())?;
                return Ok(Acquired::Held(Lease {
                    dir: lease_dir,
                    holder,
                    released: false,
                }));
            }
            Err(e) if e.kind() == ErrorKind::AlreadyExists => {}
            Err(e) => return Err(io("creating", &lease_dir, e)),
        }
        let now = env.now_s();
        let seen = read_holder(&lease_dir)?;
        let stale = match seen {
            Some(h) => !env.alive(h.pid) || now.saturating_sub(h.since) > STALE_AFTER_S,
            None => dir_age_s(&lease_dir, now)?.is_some_and(|age| age > STALE_AFTER_S),
        };
        if stale {
            take_stale(durable_dir, &lease_dir, seen, env)?;
            continue;
        }
        if now >= deadline {
            return Ok(Acquired::Busy(seen));
        }
        env.sleep(POLL);
    }
}

/// Move a stale lease aside and delete it; if what moved is not the lease
/// judged stale (it changed hands in between), move it back.
fn take_stale(
    durable_dir: &Path,
    lease_dir: &Path,
    seen: Option<Holder>,
    env: &dyn LeaseEnv,
) -> Result<()> {
    let aside = durable_dir.join(format!("{LEASE_DIR}.stale-{}-{}", env.pid(), env.now_s()));
    match fs::rename(lease_dir, &aside) {
        Ok(()) => {}
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(io("moving aside", lease_dir, e)),
    }
    if read_holder(&aside)? != seen {
        fs::rename(&aside, lease_dir).map_err(|e| {
            Error(format!(
                "lease: a live lease was moved to {} and could not be put back: {e}",
                aside.display()
            ))
        })?;
        return Ok(());
    }
    fs::remove_dir_all(&aside).map_err(|e| io("removing", &aside, e))
}

impl Lease {
    pub fn holder(&self) -> Holder {
        self.holder
    }

    /// Give the lease back; a lease no longer ours is left alone.
    pub fn release(mut self) -> Result<()> {
        self.released = true;
        self.give_back()
    }

    fn give_back(&self) -> Result<()> {
        if read_holder(&self.dir)? != Some(self.holder) {
            bail!(
                "lease: {} is no longer held by pid {}",
                self.dir.display(),
                self.holder.pid
            );
        }
        let holder = self.dir.join(HOLDER_FILE);
        fs::remove_file(&holder).map_err(|e| io("removing", &holder, e))?;
        fs::remove_dir(&self.dir).map_err(|e| io("removing", &self.dir, e))
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        if !self.released
            && let Err(e) = self.give_back()
        {
            eprintln!("metprof-kb: {e}");
        }
    }
}
