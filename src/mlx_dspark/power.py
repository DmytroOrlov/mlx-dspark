"""macOS power-source awareness for ``serve --pause-on-battery``.

The feature: on a MacBook, when the Mac moves from AC to battery, stop admitting new
inference requests, let already-admitted ones finish, then release the target/drafter
weights and allocator-retained memory while preserving reusable prefix state. When AC
power returns, reload the same effective configuration, rebind that state when compatible,
and reopen admission. The point is a 27B-class model stopping a laptop from swapping under
battery pressure while the user is away from the charger.

Mechanism (stdlib only — no new dependencies, matching the server's lean philosophy):

* **State query** — ``pmset -g batt``, whose first line says where the power comes from
  (``Now drawing from 'AC Power'`` / ``'Battery Power'`` / ``'UPS Power'``). Read once,
  synchronously, in :meth:`PowerMonitor.start` so a server started on battery never
  opens admission in the gap before the first event.
* **Change notifications** — ``pmset -g pslog``, which bridges IOKit's
  ``IOPSNotificationCreateRunLoopSource`` to stdout: one power-source snapshot per OS
  notification, printed the moment it changes. A daemon thread tails that stream and
  feeds every distinct source to the coordinator; the OS re-emits periodic same-state
  snapshots (battery level updates), so consumers must treat duplicates as idempotent
  (the coordinator does). A ctypes-IOKit notification bridge was tried and hung in this
  environment; a ``pmset`` child process is the simpler, robust event source.

Both interactions are injectable (:func:`current_power_source` via ``probe``, the pslog
line iterator via ``stream``) so the monitor is unit-testable without unplugging a Mac.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import threading

PMSET = "/usr/bin/pmset"

# How often the pslog wrapper re-checks that this process is still alive (see
# pslog_argv — this is the SIGKILL-orphan cleanup granularity, not a poll of the
# power state, which is event-driven).
WATCH_POLL_S = 2


def pslog_argv() -> list[str]:
    """The command that tails power-source changes.

    ``pmset -g pslog`` bridges IOKit's power notifications to stdout, but a hard-killed
    server (SIGKILL; SIGTERM skips atexit too) cannot clean up its child — the watcher
    THREAD dies with the process, so it cannot terminate the child either. The wrapper
    below is the cleanup: it watches THIS process's pid (``kill -0``) and kills ``pmset``
    within ``WATCH_POLL_S`` of the server disappearing from any shutdown path. Spawned in its
    own session so :meth:`PowerMonitor.stop` can signal the whole pair."""
    pid = os.getpid()
    return ["/bin/sh", "-c",
            f"pmset -g pslog & P=$!; "
            f"while kill -0 {pid} 2>/dev/null; do sleep {WATCH_POLL_S}; done; "
            f"kill $P 2>/dev/null; wait $P 2>/dev/null"]

# The three sources `pmset -g batt` reports. A UPS counts as external power for this
# feature's purposes: the machine is not running on its own battery.
_AC_PHRASES = ("Now drawing from 'AC Power'", "Now drawing from 'UPS Power'")
_BATTERY_PHRASE = "Now drawing from 'Battery Power'"


def parse_power_source(text: str) -> str | None:
    """``"ac"`` / ``"battery"`` / ``None`` from ``pmset`` output (one line or a whole
    snapshot — it scans for the phrase, so either works). Pure, for tests."""
    if not text:
        return None
    if _AC_PHRASES[0] in text or _AC_PHRASES[1] in text:
        return "ac"
    if _BATTERY_PHRASE in text:
        return "battery"
    return None


def current_power_source(run=subprocess.run) -> str | None:
    """The power source right now, or ``None`` when it cannot be determined (non-macOS,
    no pmset, timeout) — the caller treats ``None`` as "unknown: keep serving"."""
    try:
        r = run([PMSET, "-g", "batt"], capture_output=True, text=True, timeout=10)
    except Exception:  # noqa: BLE001 — FileNotFoundError on non-macOS, timeouts, …
        return None
    if r.returncode != 0:
        return None
    return parse_power_source(r.stdout or "")


class PowerMonitor:
    """Watches macOS power-source transitions and reports each one to ``on_change``.

    - ``start()`` reads the INITIAL source synchronously (the startup barrier: a server
      that comes up on battery sees ``on_change("battery")`` before it can admit a
      single request) and then tails ``pmset -g pslog`` on a daemon thread for events.
    - Same-state re-notifications (the OS re-snapshots periodically) are forwarded
      unchanged — the coordinator owns idempotence, the monitor only observes.
    - ``stop()`` terminates the child process and joins the reader thread, so process
      shutdown does not leak a ``pmset`` or block on a blocking ``readline``.
    - ``stream`` (an iterable of pslog lines) replaces the subprocess entirely — the
      test seam; ``probe`` replaces the one-shot query.
    """

    def __init__(self, on_change, *, probe=current_power_source, stream=None,
                 log=lambda msg: None):
        self._on_change = on_change
        self._probe = probe
        self._stream = stream
        self._log = log
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self.initial_source: str | None = None

    def start(self) -> PowerMonitor:
        # The synchronous initial read is the startup barrier (requirement: a server
        # starting on battery must never briefly admit inference while waiting for the
        # first event).
        self.initial_source = self._probe()
        if self.initial_source is None:
            self._log("power: could not determine the power source (pmset unavailable?) "
                      "— --pause-on-battery will not act until a transition is seen")
        else:
            self._log(f"power: initial power source is {self.initial_source}")
            self._on_change(self.initial_source)
        self._thread = threading.Thread(target=self._run, name="power-monitor",
                                        daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            proc = self._proc
            thread = self._thread
        if proc is not None and proc.poll() is None:
            self._terminate(proc)
        if thread is not None:
            thread.join(timeout=2.0)     # the child dying unblocks readline
            if not thread.is_alive():
                with self._lock:
                    if self._thread is thread:
                        self._thread = None

    @staticmethod
    def _terminate(proc) -> None:
        # The pslog child runs in its own session (the wrapper's process group) —
        # signal the GROUP so both the wrapper and pmset die, not just the wrapper.
        with contextlib.suppress(OSError, ProcessLookupError):
            os.killpg(proc.pid, signal.SIGTERM)

    # ------------------------------------------------------------------ event tailing

    def _run(self) -> None:
        try:
            if self._stream is not None:
                for line in self._stream:          # the test seam: plain iterator
                    if self._stop.is_set():
                        break
                    source = parse_power_source(line)
                    if source is not None:
                        self._on_change(source)
            else:
                proc = subprocess.Popen(
                    pslog_argv(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    text=True, bufsize=1, start_new_session=True)
                with self._lock:
                    if self._stop.is_set():
                        terminate = True
                    else:
                        self._proc = proc
                        terminate = False
                if terminate:
                    self._terminate(proc)
                    return
                if self._stop.is_set():
                    return
                for line in proc.stdout:     # blocks; stop() terminates the process group
                    if self._stop.is_set():
                        break
                    source = parse_power_source(line)
                    if source is not None:
                        self._on_change(source)
        except Exception as e:  # noqa: BLE001 — the watcher must never take the server down
            self._log(f"power: power-source watcher stopped: {type(e).__name__}: {e}")
        finally:
            with self._lock:
                owned = proc if 'proc' in locals() and self._proc is proc else None
                if owned is not None:
                    self._proc = None
            if owned is not None and owned.poll() is None:
                self._terminate(owned)
