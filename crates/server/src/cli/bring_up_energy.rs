// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: `model-bring-up-bench --energy`: one NVML cumulative-energy reader per listed
//! host (the local box as `localhost`, others over `ssh -o BatchMode=yes`), streaming the GPU's
//! `nvmlDeviceGetTotalEnergyConsumption` counter in millijoules, stamped on arrival with this
//! box's clock so every host's series shares the clock the sweep's windows are read on.
//!
//! Owner: server CLI (`met ml-utils`).
//! Invariants:
//! - The reader is Python's ctypes over `libnvidia-ml.so.1` on the host, fed on stdin, so
//!   nothing is installed or written there; it dies with its pipe when the reader is dropped.
//! - The counter is the GPU rail only (on GB10 module and memory power read N/A), so a J/tok
//!   from it is a lower bound on system energy; the record says "GPU rail".

use std::io::{BufRead as _, BufReader, Write as _};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result, bail};

use super::bring_up_conc::Series;

/// 2026-10-10: The reader: device 0's counter every 50 ms, one integer (mJ) per line.
const READER: &str = r#"
import ctypes, sys, time
nv = ctypes.CDLL("libnvidia-ml.so.1")
assert nv.nvmlInit_v2() == 0
h = ctypes.c_void_p()
assert nv.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(h)) == 0
e = ctypes.c_ulonglong()
while True:
    rc = nv.nvmlDeviceGetTotalEnergyConsumption(h, ctypes.byref(e))
    if rc != 0:
        sys.exit("nvml rc=%d" % rc)
    print(e.value, flush=True)
    time.sleep(0.05)
"#;

/// 2026-10-10: How long a reader may take to deliver its first reading.
const FIRST_READING: Duration = Duration::from_secs(20);

struct Reader {
    host: String,
    child: Child,
    series: Arc<Mutex<Series>>,
}

/// 2026-10-10: The running readers of one section.
pub(crate) struct Meter {
    readers: Vec<Reader>,
}

fn unix_now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64())
}

fn spawn(host: &str) -> Result<Reader> {
    let mut cmd = if host == "localhost" {
        Command::new("python3")
    } else {
        let mut c = Command::new("ssh");
        c.args([
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            host,
            "python3",
        ]);
        c
    };
    let mut child = cmd
        .args(["-u", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .with_context(|| format!("starting the NVML energy reader on {host}"))?;
    let mut stdin = child.stdin.take().context("reader stdin")?;
    stdin.write_all(READER.as_bytes())?;
    drop(stdin);
    let stdout = child.stdout.take().context("reader stdout")?;
    let series = Arc::new(Mutex::new(Series::new()));
    let sink = series.clone();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if let Ok(mj) = line.trim().parse::<f64>() {
                sink.lock().expect("energy series").push((unix_now(), mj));
            }
        }
    });
    Ok(Reader {
        host: host.to_string(),
        child,
        series,
    })
}

impl Meter {
    /// 2026-10-10: Start one reader per host and wait for each one's first reading.
    pub(crate) fn start(hosts: &[String]) -> Result<Self> {
        let meter = Meter {
            readers: hosts.iter().map(|h| spawn(h)).collect::<Result<_>>()?,
        };
        let deadline = std::time::Instant::now() + FIRST_READING;
        for r in &meter.readers {
            while r.series.lock().expect("energy series").is_empty() {
                if std::time::Instant::now() > deadline {
                    bail!(
                        "no NVML energy reading from {} within {FIRST_READING:?} (python3 and \
                         libnvidia-ml.so.1 on the host, key-based ssh)",
                        r.host
                    );
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        }
        Ok(meter)
    }

    /// 2026-10-10: Every host's series so far.
    pub(crate) fn series(&self) -> Vec<Series> {
        self.readers
            .iter()
            .map(|r| r.series.lock().expect("energy series").clone())
            .collect()
    }
}

impl Drop for Meter {
    fn drop(&mut self) {
        for r in &mut self.readers {
            let _ = r.child.kill();
            let _ = r.child.wait();
        }
    }
}
