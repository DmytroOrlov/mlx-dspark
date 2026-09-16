"""Model-free tests for the battery-pause coordinator (BatteryPause): the atomic
admission boundary and drain, the Condition-based reconciliation worker (started for
real in the concurrency tests — no manual _step() progress), every documented race
(AC during a pending/in-flight unload, battery during reload, duplicates, flapping),
unload-failure truthfulness, reload fidelity through EngineHolder.reload/swap, and
shutdown. No power events, no weights, no sleeps beyond bounded waits."""

from __future__ import annotations

import importlib
import threading
import time
import types

from mlx_dspark import server as S
from mlx_dspark.server import BatteryPause, EngineHolder


class FakeEngine:
    def __init__(self, model_id="org/Target"):
        self.model_id = model_id
        self.closed = False
        self.suspended = False

    def close(self):
        self.closed = True

    def suspend(self):
        self.suspended = True
        return None


class FakeHolder:
    """Mimics the EngineHolder surface BatteryPause touches: current / suspend /
    reload_kwargs / reload(spec). `unload_gate` holds an unload in place so a test
    can race a power event against a busy action."""

    def __init__(self, engine=None, reload_spec=None):
        self._engine = engine
        self.reload_spec_value = reload_spec if reload_spec is not None else \
            {"model": "org/Target"}
        self.swaps = []                       # reload(spec=...) calls, in order
        self.suspends = 0
        self.unloads = 0                       # destructive lifecycle, never battery-used
        self.preserved_prefix = None
        self.restored_prefixes = 0
        self.swap_error = None
        self.unload_gate = threading.Event()  # set -> unload() returns
        self.unload_gate.set()

    @property
    def current(self):
        return self._engine

    def reload_kwargs(self):
        return dict(self.reload_spec_value)

    def unload(self):
        self.unloads += 1
        self.unload_gate.wait(1.0)
        if self._engine is not None:
            self._engine.closed = True
            self._engine = None

    def suspend(self):
        self.suspends += 1
        self.unload_gate.wait(1.0)
        if self._engine is not None:
            self._engine.suspended = True
            self.preserved_prefix = object()
            self._engine = None

    def reload(self, spec=None):
        spec = dict(spec or {})
        self.swaps.append(spec)
        if self.swap_error is not None:
            raise self.swap_error
        if self.preserved_prefix is not None:
            self.restored_prefixes += 1
            self.preserved_prefix = None
        self._engine = FakeEngine(spec.get("model") or "reloaded")


def _pause(holder, **kw) -> BatteryPause:
    logs = []
    p = BatteryPause(holder, log=logs.append, **kw)
    p.logs = logs
    return p


def _wait(predicate, timeout=2.0):
    deadline = time.time() + timeout
    while not predicate() and time.time() < deadline:
        time.sleep(0.005)
    return predicate()


# --------------------------------------------------------------------------- AC / initial


class TestInitialAC:
    def test_serves_normally(self):
        p = _pause(FakeHolder(FakeEngine()))
        assert p.state == "ready"
        assert p.admit() is True                    # open before any power event
        assert p.health() == {"enabled": True, "source": None, "state": "ready",
                              "admission_open": True, "reason": None,
                              "model_loaded": True, "active_requests": 1,
                              "error": None}
        p.release()
        assert p.health()["active_requests"] == 0

    def test_ac_observation_is_a_noop_while_ready(self):
        p = _pause(FakeHolder(FakeEngine()))
        p.on_power_source("ac")
        assert p.state == "ready" and p.admit() is True
        assert p.logs == []                         # no noise for the initial AC state

    def test_explicit_destructive_unload_remains_separate(self):
        holder = FakeHolder(FakeEngine())
        holder.unload()
        assert holder.unloads == 1 and holder.suspends == 0


class TestInitialBattery:
    def test_coordinator_starts_paused_and_model_less(self):
        """serve --pause-on-battery launched while already on battery: the
        coordinator is constructed paused (before any load), so admission is closed
        from the first moment and the startup model NEVER loads."""
        holder = FakeHolder(None)
        p = _pause(holder, initial="battery")
        assert p.state == "paused" and p.health()["admission_open"] is False
        assert p.admit() is False
        # the AC event loads the startup spec (the first real load)
        p.on_power_source("ac")
        assert p.state == "reloading"
        p._step()
        assert p.state == "ready" and p.health()["model_loaded"] is True
        assert holder.swaps == [{}]                 # spec=None -> the startup config

    def test_no_model_start_stays_model_less_across_a_cycle(self):
        """--no-model intent: battery pauses (nothing to unload), AC reopens
        admission WITHOUT resurrecting a model nobody asked for."""
        holder = FakeHolder(None)
        p = _pause(holder, initial="battery", load_on_ac=False)
        p.on_power_source("ac")
        p._step()
        assert p.state == "ready" and p.health()["model_loaded"] is False
        assert holder.swaps == []                   # no load was started


# --------------------------------------------------------------------------- admission


class TestAdmissionBoundary:
    def test_battery_closes_admission_immediately(self):
        p = _pause(FakeHolder(FakeEngine()))
        assert p.admit() is True                    # admitted just before the boundary
        p.on_power_source("battery")
        assert p.state == "draining"
        assert p.admit() is False                   # refused, never queued
        p.release()                                 # the admitted request finishes
        p._step()                                   # -> unload -> paused
        assert p.state == "paused" and p.admit() is False

    def test_battery_with_no_requests_unloads_without_a_drain(self):
        p = _pause(FakeHolder(FakeEngine()))
        p.on_power_source("battery")
        assert p.state == "unloading"               # nothing to wait for
        assert p.admit() is False

    def test_release_underflow_is_logged_not_silently_clamped(self):
        p = _pause(FakeHolder(FakeEngine()))
        p.release()                                 # no admitted request: loud, no crash
        assert p.state == "ready"
        assert any("release() without a matching" in m for m in p.logs)


# --------------------------------------------------------------------------- real worker


class TestWorkerDrain:
    """The REAL started worker must perform the drain -> unload transition when the
    last admitted request releases — no test-side _step() nudging."""

    def test_release_to_zero_wakes_the_worker_and_unloads(self):
        holder = FakeHolder(FakeEngine())
        p = _pause(holder).start()
        try:
            assert p.admit() is True
            p.on_power_source("battery")            # -> draining, worker waits
            assert _wait(lambda: p.state == "draining")
            p.release()                             # last admitted request finished
            # the worker unloads on its own — this is the lost-wakeup regression guard
            assert _wait(lambda: p.state == "paused")
            assert holder.suspends == 1 and holder.unloads == 0 and holder._engine is None
        finally:
            p.stop()

    def test_battery_with_no_requests_unloads_via_the_worker(self):
        holder = FakeHolder(FakeEngine())
        p = _pause(holder).start()
        try:
            p.on_power_source("battery")
            assert _wait(lambda: p.state == "paused")
            assert holder.suspends == 1 and holder.unloads == 0
            p.on_power_source("ac")                 # -> reloading via the worker
            assert _wait(lambda: p.state == "ready")
            assert len(holder.swaps) == 1
        finally:
            p.stop()

    def test_battery_cycle_uses_suspend_and_offers_preserved_cache_handoff(self):
        holder = FakeHolder(FakeEngine())
        p = _pause(holder)
        p.on_power_source("battery")
        p._step()
        assert holder.suspends == 1 and holder.unloads == 0
        assert holder.current is None and holder.preserved_prefix is not None
        p.on_power_source("ac")
        p._step()
        assert p.state == "ready" and holder.restored_prefixes == 1

    def test_flapping_battery_ac_battery_ends_consistent_with_the_last_event(self):
        """AC -> battery -> AC -> battery in quick succession: extra churn is fine;
        the settled state must match the FINAL observed source (battery: unloaded
        and paused)."""
        holder = FakeHolder(FakeEngine())
        p = _pause(holder).start()
        try:
            for _ in range(3):
                p.on_power_source("battery")
                p.on_power_source("ac")
                p.on_power_source("battery")
                p.on_power_source("ac")
            p.on_power_source("battery")            # final event: battery
            assert _wait(lambda: p.state in ("paused", "unloading"))
            # and back to AC for the symmetric check
            p.on_power_source("ac")
            assert _wait(lambda: p.state == "ready")
        finally:
            p.stop()

    def test_flap_during_drain_ends_battery_paused(self):
        """Battery while a request runs, AC, battery again — the final battery wins:
        drain -> unload -> paused."""
        holder = FakeHolder(FakeEngine())
        p = _pause(holder).start()
        try:
            assert p.admit() is True
            p.on_power_source("battery")            # draining
            p.on_power_source("ac")                 # cancelled mid-drain -> ready
            assert _wait(lambda: p.state == "ready")
            p.on_power_source("battery")            # battery again, request still out
            assert _wait(lambda: p.state == "draining")
            p.release()
            assert _wait(lambda: p.state == "paused")
            assert holder.suspends == 1
        finally:
            p.stop()


# --------------------------------------------------------------------------- races


class TestRaceACDuringDrain:
    def test_ac_before_unload_cancels_the_pause_and_keeps_the_model(self):
        holder = FakeHolder(FakeEngine())
        p = _pause(holder)
        assert p.admit() is True
        p.on_power_source("battery")
        p.on_power_source("ac")                     # back before the drain needed it
        assert p.state == "ready"                   # no unload scheduled
        p.release()                                 # the request finishes normally
        assert p.state == "ready" and holder.suspends == 0
        assert p.admit() is True
        p.release()

    def test_ac_after_unload_was_scheduled_but_not_started_cancels_the_unload(self):
        """The unload was scheduled (zero requests) but the worker has not picked it
        up when AC returns: cancel entirely — the model never has to leave."""
        holder = FakeHolder(FakeEngine())
        p = _pause(holder)
        p.on_power_source("battery")                # -> unloading, unload scheduled
        assert p.state == "unloading"
        holder.unload_gate.clear()                  # would hold the unload if it ran
        p.on_power_source("ac")                     # lands BEFORE the worker acts
        assert p.state == "ready"                   # cancelled, no churn
        assert holder.suspends == 0 and p.admit() is True
        p.release()
        holder.unload_gate.set()

    def test_started_worker_never_runs_a_cancelled_pending_unload(self):
        """The stale-action regression: AC lands while the unload is QUEUED (before any
        worker began it), the cancel flips state back to ready — and then a REAL,
        STARTED worker gets every chance to run. A cancel that only changed `_state`
        (leaving `_action == "unload"` on the queue) would have the worker evict the
        model right after admission reopened."""
        engine = FakeEngine()
        holder = FakeHolder(engine)
        p = _pause(holder)
        p.on_power_source("battery")                # zero requests: unload PENDING
        assert p.state == "unloading" and holder.suspends == 0
        p.on_power_source("ac")                     # AC before any worker began it
        assert p.state == "ready" and p.admit() is True
        p.release()
        p.start()                                   # the worker starts AFTER the cancel
        try:                                        # (deterministic: no wake race)
            deadline = time.time() + 0.3            # a generous chance to run any
            while time.time() < deadline:           # stale queued action
                assert holder.suspends == 0          # never began the cancelled suspend
                assert engine.closed is False       # model stayed resident
                assert holder.swaps == []           # no pointless reload churn either
                assert p.state == "ready"
                assert p.health()["admission_open"] is True
                time.sleep(0.02)
        finally:
            p.stop()

    def test_ac_during_an_in_flight_unload_finishes_into_reload(self):
        """The literal 'AC returns while unload is occurring' race: unload() is held
        on the worker, AC lands, the held unload completes -> reload is scheduled."""
        holder = FakeHolder(FakeEngine())
        p = _pause(holder)
        p.on_power_source("battery")                # no requests -> unload scheduled
        holder.unload_gate.clear()                  # hold the unload open
        worker = threading.Thread(target=p._step, daemon=True)
        worker.start()
        try:
            assert _wait(lambda: p.state == "unloading" and holder.suspends == 1)
            p.on_power_source("ac")                 # AC during the held unload
            holder.unload_gate.set()                # let the unload finish
            worker.join(2.0)
            assert p.state == "reloading"           # the completion point re-read AC
            p._step()
            assert p.state == "ready" and len(holder.swaps) == 1
        finally:
            holder.unload_gate.set()
            worker.join(2.0)


class TestRaceBatteryDuringReload:
    def test_battery_during_reload_pauses_again_after_the_load(self):
        holder = FakeHolder(FakeEngine())
        p = _pause(holder)
        p.on_power_source("battery")
        p._step()                                   # unload
        assert p.state == "paused"
        p.on_power_source("ac")
        p._step()                                   # reload runs...
        assert p.state == "ready"
        p.on_power_source("battery")                # ...and battery is back right after
        p._step()
        assert p.state == "paused"                  # loaded once more, then unloaded again
        assert len(holder.swaps) == 1 and holder.suspends == 2

    def test_battery_observed_mid_reload_uses_the_reload_completion_point(self):
        """battery while state == 'reloading' defers to the completion point (no
        concurrent unload racing the swap lock)."""
        holder = FakeHolder(FakeEngine())
        p = _pause(holder)
        p.on_power_source("battery")
        p._step()
        p.on_power_source("ac")
        p.on_power_source("battery")                # battery lands mid-reload
        assert p.state == "reloading" and holder.suspends == 1
        p._step()                                   # reload completes -> pause again
        assert p.state == "unloading" and holder.suspends == 1
        p._step()
        assert p.state == "paused" and holder.suspends == 2 and len(holder.swaps) == 1


class TestDuplicates:
    def test_repeated_battery_and_ac_reports_change_nothing(self):
        holder = FakeHolder(FakeEngine())
        p = _pause(holder)
        assert p.admit() is True
        p.on_power_source("battery")
        p.on_power_source("battery")                # pslog re-snapshots periodically
        p.on_power_source("battery")
        assert p.state == "draining" and p.health()["active_requests"] == 1
        p.on_power_source("ac")
        p.on_power_source("ac")
        assert p.state == "ready"
        p.release()
        assert holder.suspends == 0 and holder.swaps == []


# --------------------------------------------------------------------------- reload


class TestReload:
    def test_reopen_only_after_the_reload_succeeds(self):
        holder = FakeHolder(FakeEngine(),
                            reload_spec={"model": "org/Target", "mode": "dspark",
                                         "drafter": "org/Drafter", "kv_bits": 4})
        p = _pause(holder)
        p.on_power_source("battery")
        p._step()
        assert p.admit() is False                   # closed while unloaded
        p.on_power_source("ac")
        assert p.admit() is False                   # still closed while reloading
        assert p.state == "reloading"
        p._step()
        assert p.state == "ready" and p.admit() is True
        p.release()
        assert holder.swaps == [{"model": "org/Target", "mode": "dspark",
                                 "drafter": "org/Drafter", "kv_bits": 4}]

    def test_the_running_drafter_round_trips_the_battery_cycle(self):
        """The fidelity guard: an EXPLICIT drafter captured from the RUNNING pair must
        come back on the AC reload — dropped, the reload would fall back to the
        server's startup --drafter (or fail auto-resolving an unregistered target)."""
        holder = FakeHolder(FakeEngine(),
                            reload_spec={"model": "org/T", "drafter": "org/CurrentDrafter",
                                         "mode": "dspark", "lookup_drafts": False,
                                         "confidence_threshold": 0.3, "kv_bits": 8,
                                         "max_draft_tokens": "auto",
                                         "enable_thinking": False,
                                         "reasoning_effort": "medium"})
        p = _pause(holder)
        p.on_power_source("battery")                # captures the RUNNING config
        p._step()
        p.on_power_source("ac")
        p._step()
        swap = holder.swaps[-1]
        assert swap["drafter"] == "org/CurrentDrafter"      # NOT lost, NOT the startup one
        assert swap["model"] == "org/T" and swap["mode"] == "dspark"
        # the CAPTURED spec rides through unchanged (the real swap-key mapping is
        # asserted against the real EngineHolder in TestEndToEndReloadFidelity)
        assert swap["max_draft_tokens"] == "auto" and swap["lookup_drafts"] is False
        assert swap["confidence_threshold"] == 0.3 and swap["kv_bits"] == 8
        assert swap["enable_thinking"] is False and swap["reasoning_effort"] == "medium"

    def test_reload_failure_leaves_admission_closed_with_the_error_exposed(self):
        holder = FakeHolder(FakeEngine(), reload_spec={"model": "org/Target"})
        holder.swap_error = ValueError("model repo gone")
        p = _pause(holder)
        p.on_power_source("battery")
        p._step()
        p.on_power_source("ac")
        p._step()
        assert p.state == "paused"                  # NOT ready — no silent admission
        assert p.admit() is False
        assert "ValueError" in p.health()["error"] and "model repo" in p.health()["error"]
        # a later power cycle (or explicit recovery) retries the reload — no spin
        p.on_power_source("battery")
        p.on_power_source("ac")
        holder.swap_error = None
        p._step()
        assert p.state == "ready" and p.health()["error"] is None


class TestUnloadFailure:
    def test_failed_unload_never_claims_the_model_was_freed(self):
        """holder.suspend() raising must NOT transition to 'paused' / 'model
        unloaded': admission stays closed, the error is exposed, and health says the
        model is still loaded."""
        holder = FakeHolder(FakeEngine())
        def boom():
            holder.suspends += 1
            raise RuntimeError("executor shutdown failed")
        holder.suspend = boom
        p = _pause(holder)
        p.on_power_source("battery")                # no requests -> unload scheduled
        p._step()
        assert holder.current is not None           # the model is still resident
        assert p.state == "unloading"               # NOT 'paused' — not freed
        assert p.admit() is False                   # and still not admitting
        h = p.health()
        assert "suspend failed" in h["error"] and h["model_loaded"] is True

    def test_ac_after_a_failed_unload_recovers_through_the_reload(self):
        holder = FakeHolder(FakeEngine())
        def boom():
            holder.suspends += 1
            raise RuntimeError("executor shutdown failed")
        holder.suspend = boom
        p = _pause(holder)
        p.on_power_source("battery")
        p._step()
        assert p.state == "unloading" and p.health()["error"]
        holder.suspend = types.MethodType(FakeHolder.suspend, holder)  # restore the real one
        p.on_power_source("ac")                     # model still resident -> cancel+reopen
        assert p.state == "ready" and p.health()["error"] is None
        p._step()                                   # (no reload needed — it never left)
        assert p.state == "ready" and holder.swaps == []

    def test_unload_failure_after_release_reports_resident_model(self):
        """close() can fail AFTER the holder dropped its reference: current is None
        but the memory may not be freed — 'paused' then would claim a freed model.
        The state machine keeps the truthful pairing: released -> paused."""
        holder = FakeHolder(FakeEngine())
        def boom():
            holder.suspends += 1
            holder._engine = None                   # released, then the cleanup failed
            raise RuntimeError("cache clear failed")
        holder.suspend = boom
        p = _pause(holder)
        p.on_power_source("battery")
        p._step()
        assert p.state == "paused"                  # released -> paused is truthful
        assert "suspend failed" in p.health()["error"]
        assert p.health()["model_loaded"] is False
        assert p.admit() is False


# --------------------------------------------------------------------------- worker/shutdown


class TestWorkerAndShutdown:
    def test_worker_drives_full_transitions(self):
        holder = FakeHolder(FakeEngine())
        p = _pause(holder).start()
        try:
            p.on_power_source("battery")            # no requests -> straight to unload
            assert _wait(lambda: p.state == "paused")
            p.on_power_source("ac")
            assert _wait(lambda: p.state == "ready")
            assert holder.suspends == 1 and len(holder.swaps) == 1
        finally:
            p.stop()

    def test_stop_joins_the_worker_and_is_idempotent(self):
        p = _pause(FakeHolder(FakeEngine())).start()
        p.stop()
        assert p._thread is None
        p.stop()                                    # idempotent
        p.stop()

    def test_stop_does_not_claim_a_blocked_worker_is_stopped(self):
        """stop() during a long (here: blocked) model action must be idempotent, must
        not start a new action afterwards, and must keep the thread reference while the
        worker is ALIVE — only clearing it once the join proves the thread is gone."""
        holder = FakeHolder(FakeEngine())
        holder.unload_gate.clear()                  # suspend blocks indefinitely...
        def blocked_suspend():                      # (the 1 s fake gate cap is shorter
            holder.suspends += 1                    #  than stop()'s join: block for real)
            holder.unload_gate.wait(30)
            holder._engine = None
        holder.suspend = blocked_suspend
        p = _pause(holder).start()
        p.on_power_source("battery")                # -> unload -> the worker blocks
        assert _wait(lambda: holder.suspends == 1)  # worker is provably inside it
        thread = p._thread
        p.stop()                                    # bounded join: does not hang
        assert p._thread is thread and thread.is_alive()   # truthful: NOT stopped
        assert any("still finishing" in m for m in p.logs)
        p.stop()                                    # idempotent while alive
        assert p._thread is thread
        holder.unload_gate.set()                    # the in-flight action finishes...
        assert _wait(lambda: not thread.is_alive())  # ...and the worker retires
        p.stop()                                    # (no new action ran after stopping)
        assert p._thread is None
        assert holder.swaps == []                   # it retired, it did not keep working

    def test_action_scheduled_while_an_action_runs_is_not_lost(self):
        """The reconciliation-loop property, against the REAL started worker: an
        action scheduled while the worker is inside the unload must still run on the
        next pass (no lost wakeup)."""
        holder = FakeHolder(FakeEngine())
        p = _pause(holder).start()
        try:
            p.on_power_source("battery")                # unload scheduled
            assert _wait(lambda: holder.suspends == 1)  # the worker is inside it
            holder.unload_gate.clear()                  # hold the unload open (busy)
            deadline = time.time() + 2.0
            while p.health()["state"] != "unloading" and time.time() < deadline:
                time.sleep(0.005)                       # unloading is set pre-close
            p.on_power_source("ac")                     # during the busy unload
            holder.unload_gate.set()                    # let it finish
            # the completion point schedules the reload; the REAL worker picks it up
            assert _wait(lambda: p.state == "ready", timeout=5.0), p.state
            assert len(holder.swaps) == 1
        finally:
            holder.unload_gate.set()
            p.stop()


# --------------------------------------------------------------------------- holder


class TestEngineHolderReloadKwargs:
    """reload_kwargs must reproduce the RUNNING engine's effective configuration, not
    just the server's startup flags (a mid-session /admin/load may have moved past)."""

    class _Engine(FakeEngine):
        target_repo = "org/Target"
        drafter_repo = "org/Drafter"
        mode = "dflash"
        lookup_drafts = False
        confidence_threshold = 0.3
        max_draft_tokens = 7
        cap_controller = None
        cap_pinned = True
        template_defaults = {"enable_thinking": False, "reasoning_effort": "medium"}
        warmup_enabled = True
        memory_guard = None
        kv_bits = 8                                 # the target quantizes its KV

        def __init__(self):
            super().__init__()
            self.target = self                      # health reads engine.target.kv_bits

    def test_reproduces_the_running_engine(self):
        h = EngineHolder(self._Engine(), load_kwargs={"wired_limit": False,
                                                      "warmup": True,
                                                      "context_window": 32768})
        kw = h.reload_kwargs()
        assert kw["model"] == "org/Target" and kw["drafter"] == "org/Drafter"
        assert kw["mode"] == "dflash" and kw["lookup_drafts"] is False
        assert kw["confidence_threshold"] == 0.3
        assert kw["max_draft_tokens"] == 7          # the PINNED cap, not the resolved one
        assert kw["kv_bits"] == 8
        assert kw["enable_thinking"] is False and kw["reasoning_effort"] == "medium"
        assert kw["warmup"] is True and kw["memory_guard"] is False
        # server-machine policies stay as the serve started them (not engine-derived)
        assert kw["wired_limit"] is False and kw["context_window"] == 32768

    def test_auto_cap_and_derived_cap(self):
        class Ctrl:
            cap = 7

        e = self._Engine()
        e.cap_controller = Ctrl()
        e.cap_pinned = False
        kw = EngineHolder(e, load_kwargs={}).reload_kwargs()
        assert kw["max_draft_tokens"] == "auto"     # controller-driven pair reloads auto

        e2 = self._Engine()
        e2.cap_controller = None
        e2.cap_pinned = False                       # machine-derived: re-derive
        assert EngineHolder(e2, load_kwargs={}).reload_kwargs()["max_draft_tokens"] is None

    def test_bf16_kv_round_trips_as_none(self):
        e = self._Engine()
        e.kv_bits = None                            # full-precision KV
        kw = EngineHolder(e, load_kwargs={"kv_bits": None}).reload_kwargs()
        assert kw["kv_bits"] is None                # NOT 0/False — stays "no override"


class TestEndToEndReloadFidelity:
    """The full battery cycle against the REAL EngineHolder: the pause captures
    reload_kwargs from the live engine and the AC reload reproduces it through
    Engine.load — including the CURRENT (possibly hot-swapped) drafter."""

    def test_battery_then_ac_reloads_the_same_configuration(self, monkeypatch):
        captured = {}

        def fake_load(**kw):
            captured.update(kw)
            return TestEngineHolderReloadKwargs._Engine()

        monkeypatch.setattr(S.Engine, "load", staticmethod(fake_load))
        monkeypatch.setattr(S, "maybe_batch_engine", lambda e, b: e)
        holder = EngineHolder(TestEngineHolderReloadKwargs._Engine(),
                              load_kwargs={"mode": "auto", "warmup": True})
        p = BatteryPause(holder, log=lambda msg: None)
        p.on_power_source("battery")
        p._step()                                   # -> unload -> paused
        assert holder.current is None
        p.on_power_source("ac")
        p._step()                                   # -> reload through the swap path
        assert p.state == "ready"
        assert captured["model"] == "org/Target"    # the RUNNING pair, reloaded
        assert captured["drafter"] == "org/Drafter" # the CURRENT drafter, not dropped
        assert captured["mode"] == "dflash" and captured["max_draft_tokens"] == 7
        assert captured["kv_bits"] == 8
        assert captured["warmup"] is True           # warmup reruns on reload
        assert captured["memory_guard"] is False    # engine had no guard; reproduced

    def test_explicit_serve_drafter_survives_when_the_engine_reports_it(self, monkeypatch):
        """serve --drafter X: the running engine's drafter repo is X; the reload must
        pass it EXPLICITLY (an auto-resolve for the same target could differ or fail
        for unregistered targets)."""
        captured = {}

        def fake_load(**kw):
            captured.update(kw)
            return TestEngineHolderReloadKwargs._Engine()

        monkeypatch.setattr(S.Engine, "load", staticmethod(fake_load))
        monkeypatch.setattr(S, "maybe_batch_engine", lambda e, b: e)
        holder = EngineHolder(TestEngineHolderReloadKwargs._Engine(),
                              load_kwargs={"drafter": "org/StartupDrafter"})
        assert holder.reload_kwargs()["drafter"] == "org/Drafter"   # the RUNNING one
        p = BatteryPause(holder, log=lambda msg: None)
        p.on_power_source("battery")
        p._step()
        p.on_power_source("ac")
        p._step()
        assert captured["drafter"] == "org/Drafter"  # current wins over stale startup

    def test_machine_policy_overrides_survive_battery_reload(self, monkeypatch):
        """A successful runtime swap's machine policies are the values the battery
        reload must send to the next Engine.load call, not stale startup flags."""
        calls = []

        def fake_load(**kw):
            calls.append(dict(kw))
            e = TestEngineHolderReloadKwargs._Engine()
            e.small_m = kw.get("small_m")
            e.sdpa_split = kw.get("sdpa_split")
            e.cpu_split = {"min_rows": 1} if kw.get("cpu_split") else None
            return e

        monkeypatch.setattr(S.Engine, "load", staticmethod(fake_load))
        monkeypatch.setattr(S, "maybe_batch_engine", lambda e, b: e)
        initial = TestEngineHolderReloadKwargs._Engine()
        initial.small_m, initial.sdpa_split, initial.cpu_split = True, True, {"min_rows": 1}
        holder = EngineHolder(initial, load_kwargs={
            "model": "org/Startup", "small_m": True, "sdpa_split": True,
            "cpu_split": 0.25, "warmup": False})

        holder.swap(model="org/Runtime", small_m=False, sdpa_split=False, cpu_split=0)
        p = BatteryPause(holder, log=lambda msg: None)
        p.on_power_source("battery")
        p._step()
        p.on_power_source("ac")
        p._step()

        assert calls[-1]["small_m"] is False
        assert calls[-1]["sdpa_split"] is False
        assert calls[-1]["cpu_split"] == 0

    def test_failed_runtime_swap_keeps_last_known_good_machine_policies(self, monkeypatch):
        calls = []

        def fake_load(**kw):
            calls.append(dict(kw))
            if kw.get("model") == "org/Bad":
                raise RuntimeError("load failed")
            return TestEngineHolderReloadKwargs._Engine()

        monkeypatch.setattr(S.Engine, "load", staticmethod(fake_load))
        monkeypatch.setattr(S, "maybe_batch_engine", lambda e, b: e)
        holder = EngineHolder(TestEngineHolderReloadKwargs._Engine(), load_kwargs={
            "model": "org/Startup", "small_m": True, "sdpa_split": True,
            "cpu_split": 0.25, "warmup": False})
        try:
            holder.swap(model="org/Bad", small_m=False, sdpa_split=False, cpu_split=0)
        except RuntimeError:
            pass
        assert holder._load_kwargs["small_m"] is True
        assert holder._load_kwargs["sdpa_split"] is True
        assert holder._load_kwargs["cpu_split"] == 0.25
        holder.swap(model="org/Good")
        assert calls[-1]["small_m"] is True
        assert calls[-1]["sdpa_split"] is True
        assert calls[-1]["cpu_split"] == 0.25

    def test_repo_id_pair_reloads_through_resolver_without_online_hub_call(self, monkeypatch):
        """A battery cycle must resolve captured logical repo IDs from complete HF caches."""
        calibrate = importlib.import_module("mlx_dspark.calibrate")
        import mlx_dspark.download as download
        from mlx_dspark import load

        snapshot_calls = []

        def cached_snapshot(repo, **kwargs):
            snapshot_calls.append((repo, kwargs))
            if kwargs.get("local_files_only") is True:
                return "/cached/snapshot"
            raise AssertionError("battery reload attempted an online Hub resolution")

        monkeypatch.setattr(load, "local_dir", lambda repo: None)
        monkeypatch.setattr(load, "snapshot_download", cached_snapshot)
        monkeypatch.setattr(download, "ensure_local", lambda repo: None)

        class Target:
            model = object()

        class Drafter:
            config = types.SimpleNamespace(block_size=8)

            def bind(self, model):
                pass

        def fake_target(repo, **kwargs):
            load._resolve(repo)
            return Target(), object()

        def fake_dflash(repo, **kwargs):
            load._resolve(repo)
            return Drafter(), object()

        monkeypatch.setattr(S, "load_target", fake_target)
        monkeypatch.setattr(S, "load_dflash", fake_dflash)
        monkeypatch.setattr(S, "_generation_defaults", lambda repo: {})
        monkeypatch.setattr(S, "_context_window", lambda repo: 1024)
        monkeypatch.setattr(S, "_target_config", lambda repo: None)
        monkeypatch.setattr(calibrate, "apply_small_m", lambda *a, **k: [])
        monkeypatch.setattr(calibrate, "apply_sdpa_split", lambda *a, **k: None)
        monkeypatch.setattr(calibrate, "apply_wide_gemm", lambda *a, **k: None)
        monkeypatch.setattr(calibrate, "apply_cpu_split", lambda *a, **k: None)

        startup = {
            "model": "org/Target",
            "drafter": "org/Drafter",
            "mode": "dflash",
            "max_draft_tokens": 1,
            "prefix_cache": False,
            "warmup": False,
            "memory_guard": False,
        }
        holder = EngineHolder(None, startup)
        p = BatteryPause(holder, initial="battery", log=lambda msg: None)
        p.on_power_source("ac")
        p._step()                                   # first load from the cached pair
        assert p.state == "ready" and holder.current is not None
        p.on_power_source("battery")
        p._step()                                   # unload
        p.on_power_source("ac")
        p._step()                                   # captured repo IDs reload
        assert p.state == "ready" and holder.current is not None
        assert snapshot_calls and all(
            kwargs == {"local_files_only": True} for _, kwargs in snapshot_calls)


# --------------------------------------------------------------------------- startup load


class _LoadSpy:
    """Monkeypatches Engine.load/maybe_batch_engine and records every call, returning
    a minimal fake engine — the real EngineHolder.swap path runs unmodified."""

    def __init__(self, monkeypatch, engine_factory):
        self.calls = []
        self._factory = engine_factory
        monkeypatch.setattr(S.Engine, "load", staticmethod(self._load))
        monkeypatch.setattr(S, "maybe_batch_engine", lambda e, b: e)

    def _load(self, **kw):
        self.calls.append(dict(kw))
        return self._factory(**kw)


def _startup_engine(**kw):
    e = FakeEngine(kw.get("model") or "unknown")
    e.target_repo = kw.get("model")
    e.drafter_repo = kw.get("drafter")
    e.mode = kw.get("mode") or "baseline"
    return e


class TestStartupOnBatteryFirstLoad:
    """serve --pause-on-battery launched while ALREADY on battery: the startup is
    model-less (no Engine.load), the holder keeps the intended startup configuration,
    and the FIRST AC event performs exactly one real load OF THAT MODEL through the
    real swap path — admission reopens only when it is ready."""

    def test_first_ac_load_uses_the_stored_startup_model(self, monkeypatch):
        spy = _LoadSpy(monkeypatch, _startup_engine)
        startup = {"model": "org/StartupTarget", "mode": "dspark",
                   "drafter": "org/StartupDrafter", "kv_bits": 8,
                   "max_draft_tokens": "auto", "warmup": True}
        holder = EngineHolder(None, dict(startup))          # model-less start
        assert spy.calls == []                              # nothing loaded at startup
        p = _pause(holder, initial="battery").start()
        try:
            assert p.state == "paused" and p.admit() is False   # never admitted
            assert holder.current is None
            p.on_power_source("ac")
            assert _wait(lambda: p.state == "ready"), p.state
            assert len(spy.calls) == 1                      # exactly one load
            kw = spy.calls[0]
            assert kw["model"] == "org/StartupTarget"       # NOT None (the regression)
            assert kw["drafter"] == "org/StartupDrafter"    # the startup pair rides
            assert kw["mode"] == "dspark" and kw["kv_bits"] == 8
            assert kw["max_draft_tokens"] == "auto" and kw["warmup"] is True
            assert holder.current is not None and p.admit() is True
        finally:
            p.stop()


class TestStaleValueReplacement:
    """Exact-reload fidelity: an EXPLICIT None/False captured from the RUNNING engine
    must REPLACE the stale stored startup value across the battery cycle — None there
    is a value (no drafter / full-precision KV), never 'unspecified'."""

    class _Current(FakeEngine):
        target_repo = "org/Target"
        drafter_repo = None                     # the running pair has NO drafter
        mode = "baseline"
        lookup_drafts = False
        confidence_threshold = 0.0
        max_draft_tokens = 4
        cap_controller = None
        cap_pinned = False
        template_defaults = {}
        warmup_enabled = False                  # running with --no-warmup semantics
        memory_guard = None
        kv_bits = None                          # full-precision (BF16) KV

        def __init__(self):
            super().__init__("org/Target")
            self.target = self

    def test_stale_startup_drafter_is_not_resurrected(self, monkeypatch):
        spy = _LoadSpy(monkeypatch, _startup_engine)
        holder = EngineHolder(self._Current(),
                              load_kwargs={"model": "org/Startup",
                                           "drafter": "org/StaleStartupDrafter",
                                           "mode": "auto", "warmup": True})
        p = _pause(holder)
        p.on_power_source("battery")                        # captures drafter=None
        p._step()
        assert holder.current is None
        p.on_power_source("ac")
        p._step()
        assert p.state == "ready"
        kw = spy.calls[-1]
        assert "drafter" in kw and kw["drafter"] is None    # stays NO-drafter
        assert kw["model"] == "org/Target"                  # the running pair returns

    def test_bf16_kv_is_not_quantized_back_to_startup_bits(self, monkeypatch):
        spy = _LoadSpy(monkeypatch, _startup_engine)
        current = self._Current()                           # kv_bits None: BF16
        holder = EngineHolder(current, load_kwargs={"model": "org/Startup",
                                                    "kv_bits": 8, "warmup": True})
        p = _pause(holder)
        p.on_power_source("battery")
        p._step()
        p.on_power_source("ac")
        p._step()
        assert p.state == "ready"
        kw = spy.calls[-1]
        assert "kv_bits" in kw and kw["kv_bits"] is None    # full precision survives
        assert kw["warmup"] is False                        # running --no-warmup wins
        assert kw["lookup_drafts"] is False                 # explicit False survives


# --------------------------------------------------------------------------- metrics


class TestPauseMetrics:
    """/metrics power block: state gauges, the admission-gate rejection counter,
    transition accounting (duplicates must not inflate it) and the real-holder-call
    unload/reload accounting. All against fake holders; no worker sleeps except where
    the REAL started worker is the point."""

    def test_gauges_reflect_state_admission_and_residency(self):
        holder = FakeHolder(FakeEngine())
        p = _pause(holder)
        m = p.metrics()
        assert m["enabled"] is True and m["source"] is None
        assert m["state"] == "ready" and m["admission_open"] is True
        assert m["active_requests"] == 0 and m["model_loaded"] is True
        assert p.admit() is True
        assert p.metrics()["active_requests"] == 1
        p.on_power_source("battery")                    # draining: closed, still resident
        m = p.metrics()
        assert m["state"] == "draining" and m["admission_open"] is False
        assert m["model_loaded"] is True and m["active_requests"] == 1
        p.release()
        p._step()                                       # -> paused, model gone
        m = p.metrics()
        assert m["state"] == "paused" and m["admission_open"] is False
        assert m["model_loaded"] is False and m["active_requests"] == 0

    def test_rejected_counter_increments_exactly_once_per_refusal(self):
        p = _pause(FakeHolder(FakeEngine()))
        assert p.metrics()["rejected_inference_total"] == 0
        p.on_power_source("battery")
        assert p.admit() is False and p.admit() is False and p.admit() is False
        assert p.metrics()["rejected_inference_total"] == 3
        p.on_power_source("ac")                         # AC back -> admits are not rejects
        assert p.admit() is True and p.admit() is True
        assert p.metrics()["rejected_inference_total"] == 3
        assert p.metrics()["rejected_inference_total"] == 3

    def test_duplicate_power_notifications_do_not_inflate_transitions(self):
        holder = FakeHolder(FakeEngine())
        p = _pause(holder)
        assert p.admit() is True                        # holds the drain: never unloads
        for _ in range(3):
            p.on_power_source("battery")
            p.on_power_source("battery")                # re-snapshots of the same state
            p.on_power_source("ac")                     # must not count as anything
            p.on_power_source("ac")
        m = p.metrics()
        assert m["lifecycle_transitions_total"] == 6    # 3x (ready->draining->ready)
        assert m["transitions"] == {"ready->draining": 3, "draining->ready": 3}
        assert m["unload_attempts_total"] == 0 and m["reload_attempts_total"] == 0
        p.release()

    def test_started_worker_records_unload_and_reload_successes_with_durations(self):
        holder = FakeHolder(FakeEngine())
        p = _pause(holder).start()
        try:
            p.on_power_source("battery")
            assert _wait(lambda: p.state == "paused")
            m = p.metrics()
            assert m["unload_attempts_total"] == 1 and m["unload_successes_total"] == 1
            assert m["unload_failures_total"] == 0
            assert m["last_unload_seconds"] is not None and m["last_unload_seconds"] >= 0.0
            p.on_power_source("ac")
            assert _wait(lambda: p.state == "ready")
            m = p.metrics()
            assert m["reload_attempts_total"] == 1 and m["reload_successes_total"] == 1
            assert m["reload_failures_total"] == 0
            assert m["last_reload_seconds"] is not None and m["last_reload_seconds"] >= 0.0
            # every state move went through the counted funnel
            assert m["lifecycle_transitions_total"] == 4   # ready->unloading->paused
                                                           # ->reloading->ready
        finally:
            p.stop()

    def test_cancelled_pending_unload_never_counts_an_attempt(self):
        """Attempt accounting is about the REAL holder call: an unload queued and then
        cancelled by AC must not show up as an attempt (the worker never ran it)."""
        holder = FakeHolder(FakeEngine())
        p = _pause(holder)                              # no worker: unload stays queued
        p.on_power_source("battery")                    # -> unloading, action pending
        p.on_power_source("ac")                         # cancelled before it began
        assert p.state == "ready" and holder.suspends == 0
        m = p.metrics()
        assert m["unload_attempts_total"] == 0 and m["unload_successes_total"] == 0

    def test_unload_failure_is_counted_as_failure_and_still_records_duration(self,
                                                                               monkeypatch):
        holder = FakeHolder(FakeEngine())
        holder.suspend = lambda: (_ for _ in ()).throw(RuntimeError("close failed"))
        ticks = iter([100.0, 104.25])                   # deterministic durations
        monkeypatch.setattr(S.time, "monotonic", lambda: next(ticks))
        p = _pause(holder)
        p.on_power_source("battery")
        p._step()
        m = p.metrics()
        assert m["unload_attempts_total"] == 1 and m["unload_successes_total"] == 0
        assert m["unload_failures_total"] == 1
        assert m["last_unload_seconds"] == 4.25         # measured even on failure

    def test_reload_failure_and_retry_accounting(self):
        holder = FakeHolder(FakeEngine())
        holder.swap_error = ValueError("gone")
        p = _pause(holder)
        p.on_power_source("battery")
        p._step()                                       # unload succeeds
        p.on_power_source("ac")
        p._step()                                       # reload FAILS -> paused
        m = p.metrics()
        assert m["reload_attempts_total"] == 1 and m["reload_successes_total"] == 0
        assert m["reload_failures_total"] == 1
        assert m["last_reload_seconds"] is not None
        assert p.state == "paused"
        holder.swap_error = None                        # a later cycle recovers
        p.on_power_source("battery")                    # nothing resident: no unload churn
        p.on_power_source("ac")
        assert p.state == "reloading"
        p._step()
        m = p.metrics()
        assert p.state == "ready" and m["reload_attempts_total"] == 2
        assert m["reload_successes_total"] == 1 and m["reload_failures_total"] == 1

    def test_model_less_reload_runs_no_holder_call_so_records_no_lifecycle_metrics(self):
        """The trivial (model-less) AC reopen is not a holder.reload(): attempts stay
        at zero — only the state transition is counted."""
        holder = FakeHolder(None)
        p = _pause(holder, load_on_ac=False)
        p.on_power_source("battery")                    # -> unloading (nothing resident)
        p._step()
        p.on_power_source("ac")
        p._step()
        assert p.state == "ready" and holder.swaps == []
        m = p.metrics()
        assert m["reload_attempts_total"] == 0 and m["reload_successes_total"] == 0
        assert m["unload_attempts_total"] == 1          # the unload DID call the holder

    def test_metrics_mutations_are_lock_guarded_snapshots(self):
        """Admitting/rejecting concurrently with metrics() reads: no torn state, and
        rejects + actives add up exactly."""
        holder = FakeHolder(FakeEngine())
        p = _pause(holder)
        p.on_power_source("battery")                    # closed: everything rejects
        results = []
        threads = [threading.Thread(target=lambda: results.append(
            (p.admit(), p.metrics()["rejected_inference_total"]))) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(admitted is False for admitted, _ in results)
        assert p.metrics()["rejected_inference_total"] == 8
