#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0

# 2026-10-05: WAVE 57 of the concurrency ladder = wave 56 unchanged, plus the NVML cumulative
# energy counter read at each rep's edges (nvml_energy.py).
#
# Owner: bench, concurrency ladder.
# Invariants:
# - Everything w56 records is recorded unchanged (its driver_sha256 names w56); this wrapper only
#   adds keys: per rep gpu_energy_counter_*, at the top `energy_counter` (wrapper sha, source,
#   idle baseline over w56's own idle window).
"""Same CLI as harness_w56_conc_ladder.py. w56 stays the pinned instrument: this file imports it
and wraps `run_rep` so the counter is read immediately before the rep's first request and
immediately after its last, the window `wall_s` is measured over."""

import asyncio
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness_w56_conc_ladder as w56  # noqa: E402
from nvml_energy import EnergyCounter, window_keys  # noqa: E402

COUNTER = EnergyCounter()
_run_rep = w56.run_rep
_idle = w56.measure_idle_baseline
IDLE = {}


async def run_rep(session, url, model, conc, isl, osl, power=None):
    e0 = COUNTER.read_mj()
    out = await _run_rep(session, url, model, conc, isl, osl, power=power)
    e1 = COUNTER.read_mj()
    out.update(window_keys(e0, e1, out["wall_s"], out["completion_tokens"]))
    return out


def measure_idle_baseline(window, *args, **kwargs):
    e0, t0 = COUNTER.read_mj(), time.perf_counter()
    res = _idle(window, *args, **kwargs)
    e1, t1 = COUNTER.read_mj(), time.perf_counter()
    if e0 is not None and e1 is not None:
        IDLE["idle_counter_power_w"] = (e1 - e0) / 1000.0 / (t1 - t0)
        IDLE["idle_window_s"] = t1 - t0
    return res


def main():
    w56.run_rep = run_rep
    w56.measure_idle_baseline = measure_idle_baseline
    rc = asyncio.run(w56.main())
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    if out and os.path.exists(out):
        rec = json.load(open(out))
        me = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
        lib = hashlib.sha256(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                               "nvml_energy.py"), "rb").read()).hexdigest()
        rec["energy_counter"] = {"wrapper_sha256": me, "nvml_energy_sha256": lib,
                                 "source": "nvmlDeviceGetTotalEnergyConsumption",
                                 "unavailable": COUNTER.unavailable, **IDLE}
        json.dump(rec, open(out, "w"), indent=2)
    return rc


if __name__ == "__main__":
    sys.exit(main())
