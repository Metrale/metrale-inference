# SPDX-License-Identifier: MIT OR Apache-2.0

# 2026-10-05: Tests that harness_w57 reads the NVML energy counter at the edges of each rep and
# adds its keys without changing what w56 records.
#
# Owner: bench, concurrency ladder.
# Invariants: none beyond the types.
"""The counter is read immediately before the rep's requests and immediately after them, and its
joules are the difference of those two reads; a missing or backwards reading is absent, never
zero. Stdlib only: aiohttp is stubbed, the counter and the wrapped rep are fakes.
Run: python3 -m unittest discover -s bench/ladder38 -p 'test_harness_*.py'
"""
import asyncio
import importlib.util
import pathlib
import sys
import types
import unittest

_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(_DIR))
sys.modules.setdefault("aiohttp", types.SimpleNamespace(ClientTimeout=lambda total: None))
_spec = importlib.util.spec_from_file_location("harness_w57", _DIR / "harness_w57_conc_ladder.py")
w57 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w57)
import nvml_energy  # noqa: E402


class _Counter:
    def __init__(self, reads, log):
        self._reads, self._log = list(reads), log

    def read_mj(self):
        self._log.append("read")
        return self._reads.pop(0)


def _fake_rep(log, wall_s=2.0, ctok=100):
    async def rep(session, url, model, conc, isl, osl, power=None):
        log.append("rep")
        return {"wall_s": wall_s, "completion_tokens": ctok, "tok_s": ctok / wall_s}
    return rep


class WindowEdges(unittest.TestCase):
    def run_rep(self, reads):
        log = []
        w57.COUNTER = _Counter(reads, log)
        w57._run_rep = _fake_rep(log)
        out = asyncio.run(w57.run_rep(None, "u", "m", 1, 128, 1024))
        return out, log

    def test_reads_bracket_the_rep_and_give_its_joules(self):
        out, log = self.run_rep([10_000, 410_000])
        self.assertEqual(log, ["read", "rep", "read"])
        self.assertEqual(out["gpu_energy_counter_j"], 400.0)
        self.assertEqual(out["gpu_energy_counter_mean_power_w"], 200.0)
        self.assertEqual(out["gpu_energy_counter_j_per_token"], 4.0)
        self.assertEqual(out["gpu_energy_counter_status"], "ok")

    def test_w56_keys_are_unchanged(self):
        out, _ = self.run_rep([0, 1000])
        self.assertEqual((out["wall_s"], out["completion_tokens"], out["tok_s"]), (2.0, 100, 50.0))

    def test_missing_or_backwards_reading_is_absent_not_zero(self):
        for reads in ([None, 1000], [1000, None], [5000, 1000]):
            out, _ = self.run_rep(reads)
            self.assertNotIn("gpu_energy_counter_j", out, reads)
            self.assertEqual(out["gpu_energy_counter_status"], "no counter reading")

    def test_zero_tokens_give_no_ratio(self):
        self.assertNotIn("gpu_energy_counter_j_per_token",
                         nvml_energy.window_keys(0, 1000, 1.0, 0))


if __name__ == "__main__":
    unittest.main()
