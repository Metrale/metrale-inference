# SPDX-License-Identifier: MIT OR Apache-2.0

# 2026-10-05: The NVML cumulative energy counter, read at a measured window's edges.
#
# Owner: bench, concurrency ladder.
# Invariants:
# - read_mj() returns the device's cumulative energy in millijoules, or None with the reason in
#   `unavailable`; it never estimates.
"""GPU energy from nvmlDeviceGetTotalEnergyConsumption (millijoules since driver load).

Why a counter beside power_window.py's integral: the integral sums nvidia-smi's
`power.draw.average` samples, which read a few percent high under load on discrete cards and
miss sub-sample transients; the counter is the driver's own integral. On a discrete card it
covers the board (GPU, HBM, VRMs on the board rail), so it is labelled `gpu_energy_counter_j`,
never `gpu_rail_*`. Read it at the edges of the window `wall_s` and `tok_s` come from.
"""

import ctypes


class EnergyCounter:
    def __init__(self, index=0):
        self.unavailable = None
        self._h = ctypes.c_void_p()
        try:
            self._nv = ctypes.CDLL("libnvidia-ml.so.1")
        except OSError as e:
            self.unavailable = f"libnvidia-ml.so.1: {e}"
            return
        rc = self._nv.nvmlInit_v2()
        if rc == 0:
            rc = self._nv.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(self._h))
        if rc != 0:
            self.unavailable = f"NVML init/handle rc={rc}"
            return
        if self.read_mj() is None:
            self.unavailable = self.unavailable or "nvmlDeviceGetTotalEnergyConsumption failed"

    def read_mj(self):
        if self.unavailable:
            return None
        v = ctypes.c_ulonglong()
        rc = self._nv.nvmlDeviceGetTotalEnergyConsumption(self._h, ctypes.byref(v))
        if rc != 0:
            self.unavailable = f"nvmlDeviceGetTotalEnergyConsumption rc={rc}"
            return None
        return v.value


def window_keys(start_mj, end_mj, wall_s, completion_tokens):
    """Record keys for one window; absent (not zero) when the counter gave no reading."""
    if start_mj is None or end_mj is None or end_mj < start_mj:
        return {"gpu_energy_counter_status": "no counter reading"}
    j = (end_mj - start_mj) / 1000.0
    out = {"gpu_energy_counter_j": j, "gpu_energy_counter_status": "ok"}
    if wall_s > 0:
        out["gpu_energy_counter_mean_power_w"] = j / wall_s
    if completion_tokens > 0:
        out["gpu_energy_counter_j_per_token"] = j / completion_tokens
    return out
