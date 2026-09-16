"""Model-free tests for the macOS power-source layer (power.py): pmset output parsing,
the one-shot probe's failure modes, and the PowerMonitor watcher — its synchronous
initial reading (the startup barrier), event tailing, idempotence-agnostic forwarding,
and clean shutdown without a real `pmset` or a real unplugged Mac."""

from __future__ import annotations

import importlib
import sys
import threading
import time

import pytest

from mlx_dspark.power import PowerMonitor, current_power_source, parse_power_source

power_mod = importlib.import_module("mlx_dspark.power")

_darwin_pmset = sys.platform == "darwin"

_AC = "2026-09-16 19:00:00 +0200\nNow drawing from 'AC Power'\n -InternalBattery-0 ...\n"
_BATT = "Now drawing from 'Battery Power'\n -InternalBattery-0 96%; discharging\n"
_UPS = "Now drawing from 'UPS Power'\n"


# --------------------------------------------------------------------------- parsing


class TestParsePowerSource:
    def test_ac_battery_and_ups_lines(self):
        assert parse_power_source(_AC) == "ac"
        assert parse_power_source(_BATT) == "battery"
        assert parse_power_source(_UPS) == "ac"        # external power, not the battery

    def test_pslog_noise_and_timestamp_lines_are_ignored(self):
        assert parse_power_source("2026-09-16 19:02:59 +0200 IOPSNotification\n") is None
        assert parse_power_source("Reading power source information...\n") is None
        assert parse_power_source("") is None
        assert parse_power_source("\n") is None

    def test_partial_matches_do_not_count(self):
        assert parse_power_source("drawing from 'AC") is None
        assert parse_power_source("Now drawing from 'X Power'") is None


# --------------------------------------------------------------------------- probe


class TestCurrentPowerSource:
    def _run(self, stdout=b"", rc=0, raises=None):
        class R:
            returncode = rc

        if raises:
            def run(*a, **k):
                raise raises
        else:
            def run(*a, **k):
                return type("P", (), {"stdout": stdout, "returncode": rc})()
        return run

    def test_parses_pmset_output(self):
        assert current_power_source(self._run(stdout=_BATT)) == "battery"
        assert current_power_source(self._run(stdout=_AC)) == "ac"

    def test_failure_modes_yield_none(self):
        # pmset missing / crashed / unreadable: None — the caller keeps serving.
        assert current_power_source(self._run(rc=1)) is None
        assert current_power_source(self._run(stdout="garbage")) is None
        assert current_power_source(self._run(raises=FileNotFoundError())) is None
        assert current_power_source(self._run(raises=TimeoutError())) is None


# --------------------------------------------------------------------------- monitor


class TestPowerMonitor:
    def test_start_reads_initial_source_synchronously_and_emits_it(self):
        """The startup barrier: probe() runs BEFORE the thread exists, so a server
        starting on battery has its state set with no admission gap."""
        gate = threading.Event()
        def gated_stream():
            gate.wait(1.0)
            yield _AC
        seen = []
        m = PowerMonitor(seen.append, probe=lambda: "battery",
                         stream=gated_stream(), log=lambda msg: None)
        m.start()
        try:
            assert seen == ["battery"] and m.initial_source == "battery"
            assert m._thread.is_alive()
        finally:
            m.stop()
            gate.set()

    def test_tails_the_stream_and_reports_each_distinct_line(self):
        seen = []
        lines = iter(["noise", "garbage line", _AC, _AC, _BATT, _AC])
        m = PowerMonitor(seen.append, probe=lambda: None, stream=lines,
                         log=lambda msg: None)
        m.start()
        deadline = time.time() + 2
        while len(seen) < 3 and time.time() < deadline:
            time.sleep(0.005)
        m.stop()
        # every power line is forwarded in order (duplicates INCLUDED — the coordinator
        # owns idempotence); noise never produces a callback
        assert seen == ["ac", "ac", "battery", "ac"]

    def test_stream_exhaustion_ends_the_thread_cleanly(self):
        m = PowerMonitor(lambda src: None, probe=lambda: "ac", stream=iter([]),
                         log=lambda msg: None)
        m.start()
        deadline = time.time() + 2
        while m._thread.is_alive() and time.time() < deadline:
            time.sleep(0.005)
        assert not m._thread.is_alive()          # EOF: the loop ended, no hang

    def test_watcher_exception_is_logged_not_raised(self):
        logs = []
        def boom_stream():
            raise RuntimeError("no pmset here")
            yield                                # pragma: no cover — generator shape
        m = PowerMonitor(lambda src: None, probe=lambda: "ac", stream=boom_stream(),
                         log=logs.append)
        m.start()
        deadline = time.time() + 2
        while not logs and m._thread.is_alive() and time.time() < deadline:
            time.sleep(0.005)
        m.stop()
        assert any("watcher stopped" in msg for msg in logs)

    def test_stop_joins_the_thread(self):
        gate = threading.Event()
        def slow_stream():
            gate.wait(1.0)
            yield _AC
        seen = []
        m = PowerMonitor(seen.append, probe=lambda: "battery", stream=slow_stream(),
                         log=lambda msg: None)
        m.start()
        m.stop()                                  # unblocks the wait, joins with timeout
        gate.set()
        deadline = time.time() + 2
        while m._thread is not None and time.time() < deadline:
            time.sleep(0.005)
        assert m._thread is None                  # joined and cleared

    def test_stop_before_subprocess_publication_terminates_new_child(self, monkeypatch):
        """If stop wins the publication race, the reader terminates the child as soon
        as Popen returns and never enters its blocking stdout loop."""
        entered = threading.Event()
        release_popen = threading.Event()
        terminated = []

        class Proc:
            pid = 12345
            stdout = iter(())

            def __init__(self):
                self.killed = False

            def poll(self):
                return 0 if self.killed else None

        proc = Proc()

        def fake_popen(*args, **kwargs):
            entered.set()
            release_popen.wait(2.0)
            return proc

        monkeypatch.setattr(power_mod.subprocess, "Popen", fake_popen)
        monkeypatch.setattr(power_mod.os, "killpg",
                            lambda pid, sig: (proc.__setattr__("killed", True),
                                              terminated.append((pid, sig))))
        m = PowerMonitor(lambda src: None, probe=lambda: "ac", log=lambda msg: None)
        m.start()
        assert entered.wait(1.0)
        stopper = threading.Thread(target=m.stop)
        stopper.start()
        assert m._stop.wait(1.0)
        release_popen.set()
        stopper.join(2.5)
        assert not stopper.is_alive()
        assert proc.killed and terminated
        assert m._thread is None and m._proc is None

    def test_timed_out_join_retains_live_reader_reference(self):
        class StuckThread:
            def __init__(self):
                self.alive = True

            def join(self, timeout):
                pass

            def is_alive(self):
                return self.alive

        m = PowerMonitor(lambda src: None, probe=lambda: "ac", log=lambda msg: None)
        thread = StuckThread()
        m._thread = thread
        m.stop()
        assert m._thread is thread
        thread.alive = False
        m.stop()
        assert m._thread is None


@pytest.mark.skipif(not _darwin_pmset, reason="pmset is a macOS facility")
class TestPowerMonitorRealProcess:
    """The subprocess path (not mocked): only asserts lifecycle, never the actual
    power source — this Mac's charger state is none of a test's business."""

    def test_subprocess_path_starts_and_stops(self):
        seen = []
        m = PowerMonitor(seen.append, probe=lambda: "battery", log=lambda msg: None)
        m.start()
        deadline = time.time() + 2
        while m._proc is None and time.time() < deadline:   # thread spawns the child
            time.sleep(0.005)
        assert m._thread.is_alive() and m._proc is not None
        m.stop()                                   # signals the wrapper's process group
        assert m._thread is None
        assert m._proc is None or m._proc.poll() is not None   # wrapper reaped
        assert seen == ["battery"]                 # initial read still synchronous

    @pytest.mark.skipif(True, reason="informational only — run by hand to inspect real output")
    def test_real_pslog_stream_shapes(self):
        import subprocess
        proc = subprocess.Popen(["/usr/bin/pmset", "-g", "pslog"],
                                stdout=subprocess.PIPE, text=True)
        try:
            lines = [proc.stdout.readline() for _ in range(6)]
            assert all(parse_power_source(l) in (None, "ac", "battery") for l in lines)
        finally:
            proc.terminate()


@pytest.mark.skipif(not _darwin_pmset, reason="exercises the real pmset wrapper")
class TestPowerMonitorOrphanCleanup:
    """A SIGTERMed/SIGKILLed server skips atexit and takes its threads with it, so the
    WRAPPER must do the cleanup: it watches the server's pid and kills pmset. Verified
    for real by spawning the watcher in a child process and SIGKILLing it."""

    @staticmethod
    def _pmset_orphans() -> int:
        import subprocess

        out = subprocess.run(["/usr/bin/pgrep", "-f", r"pmset -g pslog"],
                             capture_output=True, text=True)
        return len([l for l in out.stdout.splitlines() if l.strip()])

    def test_sigkilled_owner_leaves_no_pmset_orphan(self):
        import signal as _signal
        import subprocess
        import sys as _sys

        code = (
            "import sys, time\n"
            "sys.path.insert(0, 'src')\n"
            "from mlx_dspark.power import PowerMonitor\n"
            "PowerMonitor(lambda src: None, probe=lambda: 'ac',\n"
            "             log=lambda msg: None).start()\n"
            "time.sleep(60)\n"
        )
        owner = subprocess.Popen([_sys.executable, "-c", code],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        baseline = self._pmset_count()         # pre-existing orphans are not ours to count
        try:
            # the wrapper starts its pmset child within a couple of polls
            deadline = time.time() + 8.0
            while self._pmset_count() <= baseline and time.time() < deadline:
                time.sleep(0.2)
            assert self._pmset_count() > baseline
            owner.send_signal(_signal.SIGKILL)         # the no-cleanup path
            owner.wait(5)          # reap: a zombie's pid still answers `kill -0`
            deadline = time.time() + 12.0
            while self._pmset_count() > baseline and time.time() < deadline:
                time.sleep(0.3)                        # wrapper notices within WATCH_POLL_S
            assert self._pmset_count() <= baseline     # ours reaped, not leaked
        finally:
            owner.kill()
            owner.wait(5)

    @staticmethod
    def _pmset_count() -> int:
        import subprocess

        out = subprocess.run(["/usr/bin/pgrep", "-f", r"pmset -g pslog"],
                             capture_output=True, text=True)
        return len([l for l in out.stdout.splitlines() if l.strip()])
