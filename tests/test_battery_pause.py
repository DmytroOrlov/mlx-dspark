"""Holder-owned lifecycle tests with BatteryPause exercised only as a power facade.

The fake holder below overrides raw model operations, but all desired-state,
lease, revision, and worker behavior comes from the real EngineHolder controller.
"""

from __future__ import annotations

import gc
import importlib
import threading
import time
import types
import weakref

import pytest

from mlx_dspark import server as S
from mlx_dspark.server import UNSET, BatteryPause, EngineHolder


class FakeEngine:
    def __init__(self, model_id="org/Target"):
        self.model_id = model_id
        self.mode = "lookup"
        self.target_repo = model_id
        self.drafter_repo = None
        self.prefix = None
        self.memory_guard = None
        self.small_m = False
        self.sdpa_split = False
        self.cpu_split = None
        self.cap_controller = None
        self.sampling_defaults = {}
        self.closed = False
        self.suspended = False

    def close(self):
        self.closed = True

    def suspend(self):
        self.suspended = True
        return _FakePrefix()


class _FakePrefix:
    def __init__(self):
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1


class FakeHolder(EngineHolder):
    """Real lifecycle controller plus deterministic raw-operation fakes."""

    def __init__(self, engine=None, reload_spec=None):
        super().__init__(engine, load_kwargs={})
        self.reload_spec_value = dict(reload_spec or {"model": "org/Target"})
        self.swaps = []
        self.suspends = 0
        self.unloads = 0
        self.preserved_prefix = None
        self.restored_prefixes = 0
        self.swap_error = None
        self.drop_before_swap_error = False
        self.suspend_error = None
        self.unload_error = None
        self.drop_before_suspend_error = False
        self.drop_before_unload_error = False
        self.unload_gate = threading.Event()
        self.unload_gate.set()
        self.swap_gate = threading.Event()
        self.swap_gate.set()
        self.reload_started = threading.Event()

    def reload_kwargs(self):
        return dict(self.reload_spec_value)

    @staticmethod
    def _operation_spec(kwargs):
        fields = ("model", "mode", "drafter", "lookup_drafts",
                  "confidence_threshold", "small_m", "sdpa_split", "cpu_split",
                  "kv_bits", "warmup", "memory_guard", "enable_thinking",
                  "reasoning_effort")
        spec = {key: kwargs[key] for key in fields
                if key in kwargs and kwargs[key] is not None and kwargs[key] is not UNSET}
        max_draft = kwargs.get("max_draft", UNSET)
        if max_draft is not UNSET:
            spec["max_draft_tokens"] = max_draft
        return spec

    def _swap_now(self, **kwargs):
        self.reload_started.set()
        assert self.swap_gate.wait(2.0)
        if self.swap_error is not None:
            if self.drop_before_swap_error:
                self._engine = None
            raise self.swap_error
        spec = self._operation_spec(kwargs)
        self.swaps.append(spec)
        preserved = kwargs.get("preserved_prefix")
        if preserved is not None:
            self.restored_prefixes += 1
            self.preserved_prefix = None
        self._engine = FakeEngine(spec.get("model") or "reloaded")
        self._preserved_prefix = None
        self._suspended = None
        return self.status()

    def _suspend_now(self):
        self.suspends += 1
        assert self.unload_gate.wait(2.0)
        old = self._engine
        if old is None:
            return self.status()
        if self.suspend_error is not None and not self.drop_before_suspend_error:
            raise self.suspend_error
        old.suspend()
        self._engine = None
        self.preserved_prefix = _FakePrefix()
        self._preserved_prefix = self.preserved_prefix
        if self._suspended is None:
            self._suspended = S._SuspendedState(self.reload_kwargs())
        self._suspended.preserved_prefix = self.preserved_prefix
        if self.suspend_error is not None:
            raise self.suspend_error
        return self.status()

    def _unload_now(self):
        self.unloads += 1
        assert self.unload_gate.wait(2.0)
        old = self._engine
        if self.unload_error is not None and not self.drop_before_unload_error:
            raise self.unload_error
        self._engine = None
        self._preserved_prefix = None
        self._suspended = None
        if self.drop_before_unload_error:
            raise self.unload_error or RuntimeError("close failed")
        if old is not None:
            old.close()
        return self.status()


def _pause(holder, **kwargs):
    logs = []
    pause = BatteryPause(holder, log=logs.append, **kwargs)
    pause.logs = logs
    pause._test_leases = []
    return pause


def _step(pause):
    pause._holder._lifecycle._step()


def _thread(pause):
    return pause._holder._lifecycle.thread()


def _admit(pause):
    lease = pause._holder.acquire_generation()
    if lease is not None:
        pause._test_leases.append(lease)
    return lease is not None


def _release(pause):
    if pause._test_leases:
        _release_lease(pause._test_leases.pop(0))
    else:
        pause.logs.append("release() without a matching lease")


def _release_lease(lease):
    lease.release()


def _wait(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.005)
    return predicate()


class TestFacadeAndStartup:
    def test_initial_ac_is_ready_and_power_gate_is_open(self):
        holder = FakeHolder(FakeEngine())
        pause = _pause(holder)
        assert pause.state == "ready"
        assert _admit(pause)
        health = pause.health()
        assert health["admission_open"] is True
        assert health["model_loaded"] is True
        _release(pause)

    def test_startup_on_battery_does_not_load_until_ac(self):
        holder = FakeHolder(None)
        pause = _pause(holder, initial="battery")
        assert pause.state == "paused"
        assert holder.swaps == []
        pause.on_power_source("ac")
        _step(pause)
        assert pause.state == "ready" and holder.current is not None
        assert holder.swaps == [{}]

    def test_first_ac_load_replays_exact_stored_startup_configuration(self, monkeypatch):
        calls = []

        def fake_load(**kwargs):
            calls.append(dict(kwargs))
            engine = FakeEngine(kwargs["model"])
            engine.target_repo = kwargs["model"]
            engine.drafter_repo = kwargs.get("drafter")
            engine.mode = kwargs.get("mode")
            return engine

        monkeypatch.setattr(S.Engine, "load", staticmethod(fake_load))
        monkeypatch.setattr(S, "maybe_batch_engine", lambda engine, _max_batch: engine)
        startup = {"model": "org/StartupTarget", "drafter": "org/StartupDrafter",
                   "mode": "dspark", "kv_bits": 8, "max_draft_tokens": "auto",
                   "warmup": True}
        holder = EngineHolder(None, dict(startup))
        pause = _pause(holder, initial="battery")
        assert calls == []
        pause.on_power_source("ac")
        _step(pause)
        assert len(calls) == 1
        assert calls[0]["model"] == "org/StartupTarget"
        assert calls[0]["drafter"] == "org/StartupDrafter"
        assert calls[0]["mode"] == "dspark"
        assert calls[0]["kv_bits"] == 8
        assert calls[0]["max_draft_tokens"] == "auto"
        assert calls[0]["warmup"] is True

    def test_no_model_intent_reopens_power_gate_without_loading(self):
        holder = FakeHolder(None)
        pause = _pause(holder, initial="battery", load_on_ac=False)
        pause.on_power_source("ac")
        assert pause.state == "ready"
        assert pause.health()["admission_open"] is True
        assert holder.current is None
        assert holder.acquire_generation() is None

    def test_facade_delegates_worker_lifecycle(self):
        holder = FakeHolder(FakeEngine())
        pause = _pause(holder).start()
        try:
            assert _thread(pause) is not None
        finally:
            pause.stop()
        assert _thread(pause) is None


class TestDrainAndPowerRaces:
    def test_generation_drains_before_suspend(self):
        holder = FakeHolder(FakeEngine())
        pause = _pause(holder)
        assert _admit(pause)
        pause.on_power_source("battery")
        assert pause.state == "draining"
        assert not _admit(pause)
        _release(pause)
        _step(pause)
        assert pause.state == "paused"
        assert holder.suspends == 1 and holder.current is None
        assert any("waiting for 1 admitted request" in log for log in pause.logs)
        assert any("last admitted request finished" in log for log in pause.logs)
        assert any("model weights suspended" in log for log in pause.logs)

    def test_battery_without_active_generation_logs_direct_suspend(self):
        holder = FakeHolder(FakeEngine())
        pause = _pause(holder)
        pause.on_power_source("battery")
        assert any("no requests running, suspending model weights" in log
                   for log in pause.logs)
        _step(pause)
        assert any("model weights suspended" in log for log in pause.logs)

    def test_ac_before_drain_completes_cancels_pause_and_snapshot(self):
        holder = FakeHolder(FakeEngine(), reload_spec={"model": "A"})
        pause = _pause(holder)
        assert _admit(pause)
        pause.on_power_source("battery")
        assert holder._suspended.resume_spec["model"] == "A"
        pause.on_power_source("ac")
        assert pause.state == "ready"
        assert pause.health()["reason"] is None
        assert holder._suspended is None
        _release(pause)
        assert holder.suspends == 0

    def test_power_source_and_lifecycle_reason_are_one_snapshot(self):
        holder = FakeHolder(FakeEngine())
        pause = _pause(holder)
        pause.on_power_source("battery")
        battery = pause.health()
        assert battery["source"] == "battery"
        assert battery["reason"] == S.UNAVAILABLE_BATTERY_PAUSE
        assert battery["admission_open"] is False
        pause.on_power_source("ac")
        ac = pause.health()
        assert ac["source"] == "ac"
        assert ac["reason"] is None
        assert ac["admission_open"] is True

    def test_ac_after_queued_suspend_keeps_resident_model(self):
        holder = FakeHolder(FakeEngine())
        pause = _pause(holder)
        pause.on_power_source("battery")
        assert pause.state == "unloading"
        pause.on_power_source("ac")
        assert pause.state == "ready"
        assert pause.health()["reason"] is None
        assert holder.current is not None and holder.suspends == 0
        assert holder._suspended is None
        assert any("pending unload cancelled" in log for log in pause.logs)
        assert not any("model weights suspended" in log for log in pause.logs)

    def test_explicit_unload_owns_boundary_through_power_flap(self):
        holder = FakeHolder(FakeEngine())
        holder.unload_gate.clear()
        pause = _pause(holder)
        lease = holder.acquire_generation()
        errors = []
        unload = threading.Thread(target=lambda: _capture_error(errors, holder.unload))
        unload.start()
        try:
            assert _wait(lambda: holder._lifecycle._explicit_op == "unload")
            assert pause.state == "unloading"
            assert holder._lifecycle._explicit_op == "unload"

            pause.on_power_source("battery")
            pause.on_power_source("ac")
            assert holder.acquire_generation() is None
            assert holder.acquire_read() is None
            assert pause.state == "unloading"
            assert holder.current is not None

            lease.release()
            assert _wait(lambda: holder.unloads == 1)
            assert unload.is_alive()
            holder.unload_gate.set()
            unload.join(2.0)
            assert not unload.is_alive() and errors == []
            assert holder.current is None
            assert pause.state == "ready"
            assert pause.health()["source"] == "ac"
            assert pause.health()["reason"] is None
            assert pause.health()["admission_open"] is True
            assert holder.acquire_generation() is None
            assert holder.acquire_read() is None
        finally:
            lease.release()
            holder.unload_gate.set()
            unload.join(2.0)

    def test_explicit_load_battery_flap_does_not_leave_stale_reason(self, monkeypatch):
        holder = FakeHolder(FakeEngine("old"))
        pause = _pause(holder)
        entered = threading.Event()
        release = threading.Event()

        def fail_after_release(**_kwargs):
            entered.set()
            assert release.wait(2.0)
            holder._engine = None
            raise ValueError("transient load failed")

        monkeypatch.setattr(holder, "_swap_now", fail_after_release)
        errors = []
        load = threading.Thread(target=lambda: _capture_error(
            errors, holder.swap, model="org/Bad"))
        load.start()
        assert entered.wait(1.0)
        assert holder._lifecycle._explicit_op == "swap"

        pause.on_power_source("battery")
        pause.on_power_source("ac")
        assert holder._lifecycle.phase == "reloading"
        assert holder.acquire_generation() is None
        assert holder.acquire_read() is None
        release.set()
        load.join(2.0)

        assert not load.is_alive()
        assert isinstance(errors[0], ValueError)
        snap = pause.health()
        assert snap["source"] == "ac"
        assert snap["reason"] is None
        assert snap["state"] == "ready"
        assert snap["model_loaded"] is False
        assert snap["admission_open"] is True
        assert holder.acquire_generation() is None
        assert holder.acquire_read() is None
        assert holder._lifecycle._intent is False
        assert holder._lifecycle._action is None

    def test_cancelled_snapshot_is_fresh_on_the_next_battery_edge(self):
        holder = FakeHolder(FakeEngine(), reload_spec={"model": "A"})
        pause = _pause(holder)
        pause.on_power_source("battery")
        assert holder._suspended.resume_spec["model"] == "A"
        pause.on_power_source("ac")
        assert holder._suspended is None
        holder.reload_spec_value = {"model": "B"}
        pause.on_power_source("battery")
        assert holder._suspended.resume_spec["model"] == "B"

    def test_duplicate_observations_and_flapping_settle_to_last_source(self):
        holder = FakeHolder(FakeEngine())
        pause = _pause(holder)
        for _ in range(3):
            pause.on_power_source("battery")
            pause.on_power_source("battery")
            pause.on_power_source("ac")
        pause.on_power_source("battery")
        _step(pause)
        assert pause.state == "paused"
        pause.on_power_source("ac")
        _step(pause)
        assert pause.state == "ready"
        # Four real AC->battery edges; duplicate battery observations stay quiet.
        assert sum("AC -> battery — inference admission paused" in log
                   for log in pause.logs) == 4

    def test_facade_source_updates_only_after_observation_succeeds(self, monkeypatch):
        holder = FakeHolder(None)
        pause = _pause(holder)
        original = holder._lifecycle.observe_inhibition

        def fail_observation(**_kwargs):
            raise RuntimeError("observation failed")

        monkeypatch.setattr(holder._lifecycle, "observe_inhibition", fail_observation)
        with pytest.raises(RuntimeError, match="observation failed"):
            pause.on_power_source("battery")
        assert pause._source is None

        monkeypatch.setattr(holder._lifecycle, "observe_inhibition", original)
        pause.on_power_source("battery")
        assert pause._source == "battery"

    def test_real_worker_drain_and_stop_are_idempotent(self):
        holder = FakeHolder(FakeEngine())
        pause = _pause(holder).start()
        pause.on_power_source("battery")
        assert _wait(lambda: pause.state == "paused")
        pause.stop()
        pause.stop()
        assert holder.suspends == 1


class TestAutomaticRecovery:
    def test_queued_resume_is_cancelled_by_a_new_battery_edge(self):
        holder = FakeHolder(None)
        pause = _pause(holder, initial="battery")
        pause.on_power_source("ac")
        assert pause.state == "reloading"
        pause.on_power_source("battery")
        assert pause.state == "paused"
        assert holder._lifecycle._action is None
        assert pause.metrics()["reload_attempts_total"] == 0
        assert not holder.reload_started.is_set()
        assert any("battery returned before automatic reload started" in log
                   for log in pause.logs)

        pause.on_power_source("ac")
        _step(pause)
        assert pause.state == "ready"
        assert pause.metrics()["reload_attempts_total"] == 1

    def test_claimed_resume_settles_before_raw_load_when_battery_returns(self):
        holder = FakeHolder(None)
        pause = _pause(holder, initial="battery")
        pause.on_power_source("ac")
        lifecycle = holder._lifecycle
        assert lifecycle._claim() == "resume"
        pause.on_power_source("battery")
        lifecycle._do_resume()
        lifecycle._finish()
        assert pause.state == "paused"
        assert holder.current is None
        assert not holder.reload_started.is_set()
        assert pause.metrics()["reload_attempts_total"] == 0

        pause.on_power_source("ac")
        _step(pause)
        assert pause.state == "ready"

    def test_successful_auto_resume_is_counted_once(self):
        holder = FakeHolder(None)
        pause = _pause(holder, initial="battery").start()
        try:
            pause.on_power_source("ac")
            assert _wait(lambda: pause.state == "ready")
            metrics = pause.metrics()
            assert metrics["reload_attempts_total"] == 1
            assert metrics["reload_successes_total"] == 1
            assert metrics["reload_failures_total"] == 0
            assert _wait(lambda: any("model reloaded and warm" in log
                                     for log in pause.logs))
        finally:
            pause.stop()

    def test_successful_auto_resume_logs_prefix_handoff_outcome(self):
        for outcome, expected in (
                ("restored", "prefix cache restored"),
                ("discarded", "preserved prefix cache discarded")):
            holder = FakeHolder(None)
            pause = _pause(holder, initial="battery")
            pause.on_power_source("ac")
            holder._last_cache_handoff = outcome
            _step(pause)
            assert any(expected in log for log in pause.logs)

    def test_battery_during_resume_reconciles_after_load(self):
        holder = FakeHolder(None, reload_spec={"model": "T"})
        holder.swap_gate.clear()
        pause = _pause(holder, initial="battery").start()
        try:
            pause.on_power_source("ac")
            assert holder.reload_started.wait(1.0)
            pause.on_power_source("battery")
            holder.swap_gate.set()
            assert _wait(lambda: pause.state == "paused")
            assert holder.current is None and holder.suspends == 1
            assert _wait(lambda: any("battery returned during automatic reload" in log
                                     for log in pause.logs))
        finally:
            holder.swap_gate.set()
            pause.stop()

    def test_ac_during_inflight_suspend_reconciles_to_one_resume(self):
        holder = FakeHolder(FakeEngine())
        holder.unload_gate.clear()
        pause = _pause(holder).start()
        try:
            pause.on_power_source("battery")
            assert _wait(lambda: holder.suspends == 1)
            pause.on_power_source("ac")
            holder.unload_gate.set()
            assert _wait(lambda: pause.state == "ready")
            assert holder.suspends == 1 and len(holder.swaps) == 1
            assert pause.metrics()["unload_attempts_total"] == 1
            assert pause.metrics()["reload_attempts_total"] == 1
            assert _wait(lambda: any("battery -> AC during unload" in log
                                     for log in pause.logs))
            assert any("model reloaded and warm" in log for log in pause.logs)
            assert not any("admission closed until AC power returns" in log
                           for log in pause.logs)
        finally:
            holder.unload_gate.set()
            pause.stop()

    def test_ac_during_failed_suspend_does_not_claim_waiting_for_ac(self):
        holder = FakeHolder(FakeEngine())
        holder.unload_gate.clear()
        holder.suspend_error = RuntimeError("suspend failed")
        pause = _pause(holder).start()
        try:
            pause.on_power_source("battery")
            assert _wait(lambda: holder.suspends == 1)
            pause.on_power_source("ac")
            holder.unload_gate.set()
            assert _wait(lambda: pause.state == "ready")
            assert any("suspend failed" in log for log in pause.logs)
            assert not any("until AC power returns" in log for log in pause.logs)
            health = pause.health()
            assert health["source"] == "ac"
            assert health["state"] == "ready"
            assert health["reason"] is None
            assert "suspend failed" in health["error"]
        finally:
            holder.unload_gate.set()
            pause.stop()

    def test_failed_suspend_recovery_does_not_poison_later_explicit_load_failure(self):
        holder = FakeHolder(FakeEngine())
        holder.unload_gate.clear()
        holder.suspend_error = RuntimeError("suspend failed")
        pause = _pause(holder).start()
        try:
            pause.on_power_source("battery")
            assert _wait(lambda: holder.suspends == 1)
            pause.on_power_source("ac")
            holder.unload_gate.set()
            assert _wait(lambda: pause.state == "ready")
            assert holder.current is not None
            assert pause.health()["reason"] is None

            holder.swap_error = ValueError("explicit load failed")
            holder.drop_before_swap_error = True
            with pytest.raises(ValueError, match="explicit load failed"):
                holder.swap(model="org/Bad")

            health = pause.health()
            assert holder.current is None
            assert health["source"] == "ac"
            assert health["reason"] is None
            assert health["state"] == "ready"
        finally:
            holder.unload_gate.set()
            pause.stop()

    def test_run_server_banner_lease_survives_battery_teardown(self, monkeypatch):
        holder = FakeHolder(FakeEngine())
        pause = _pause(holder).start()

        class StubHTTPServer:
            def __init__(self, _address, _handler):
                self.daemon_threads = False
                self.closed = False

            def serve_forever(self):
                return None

            def server_close(self):
                self.closed = True

        monkeypatch.setattr(S, "ThreadingHTTPServer", StubHTTPServer)
        import builtins
        original_print = builtins.print
        triggered = threading.Event()

        def print_and_teardown(*args, **kwargs):
            if not triggered.is_set():
                triggered.set()
                pause.on_power_source("battery")
                assert _wait(lambda: holder.current is None)
            return original_print(*args, **kwargs)

        monkeypatch.setattr(builtins, "print", print_and_teardown)
        try:
            S.run_server(holder, host="127.0.0.1", port=0, pause=pause)
            assert triggered.is_set()
        finally:
            pause.stop()

    def test_run_server_banner_does_not_retain_prefix_target_during_serve_forever(
            self, monkeypatch):
        class Target:
            def make_cache(self):
                return None

        class Prefix:
            def __init__(self, target):
                self.make_cache = target.make_cache
                self.l2_dir = None
                self.cache_state = object()

            def reset(self):
                self.cache_state = None

        class Engine(FakeEngine):
            def __init__(self, target):
                super().__init__()
                self.target = target
                self.prefix = Prefix(target)

            def close(self):
                self.prefix.reset()
                self.prefix = None
                self.target = None
                self.closed = True

        target = Target()
        target_ref = weakref.ref(target)
        holder = FakeHolder(Engine(target))
        del target

        printed = []
        original_print = __import__("builtins").print

        def recording_print(*args, **kwargs):
            if args:
                printed.append(str(args[0]))
            return original_print(*args, **kwargs)

        class StubHTTPServer:
            def __init__(self, _address, _handler):
                self.daemon_threads = False

            def serve_forever(self):
                assert any("prefix cache: on" in line for line in printed)
                holder.unload()
                gc.collect()
                assert target_ref() is None

            def server_close(self):
                pass

        monkeypatch.setattr(S, "ThreadingHTTPServer", StubHTTPServer)
        monkeypatch.setattr(__import__("builtins"), "print", recording_print)
        S.run_server(holder, host="127.0.0.1", port=0)

    def test_stop_keeps_blocked_worker_visible_until_raw_operation_finishes(self,
                                                                            monkeypatch):
        holder = FakeHolder(FakeEngine())
        release = threading.Event()

        def blocked_suspend():
            holder.suspends += 1
            assert release.wait(30.0)
            holder._engine = None
            return holder.status()

        monkeypatch.setattr(holder, "_suspend_now", blocked_suspend)
        pause = _pause(holder).start()
        try:
            pause.on_power_source("battery")
            assert _wait(lambda: holder.suspends == 1)
            worker = _thread(pause)
            pause.stop()
            assert _thread(pause) is worker and worker.is_alive()
            release.set()
            assert _wait(lambda: not worker.is_alive())
            assert holder.suspends == 1 and holder.swaps == []
            pause.stop()
            assert _thread(pause) is None
        finally:
            release.set()
            pause.stop()

    def test_failed_auto_resume_is_recoverable_by_explicit_load(self):
        holder = FakeHolder(FakeEngine())
        pause = _pause(holder)
        pause.on_power_source("battery")
        _step(pause)
        holder.swap_error = ValueError("automatic load failed")
        pause.on_power_source("ac")
        _step(pause)
        assert pause.state == "paused" and holder.current is None
        assert any("reload failed" in log and "future power transition" in log
                   for log in pause.logs)
        holder.swap_error = None
        holder.swap(model="org/Recovered")
        assert pause.state == "ready" and holder.current is not None

    def test_failed_explicit_recovery_keeps_model_less_pause_unavailable(self):
        holder = FakeHolder(FakeEngine())
        pause = _pause(holder)
        pause.on_power_source("battery")
        _step(pause)
        pause.on_power_source("ac")
        holder.swap_error = ValueError("recovery model gone")
        with pytest.raises(ValueError, match="recovery model gone"):
            holder.swap(model="org/Gone")
        assert holder.current is None
        assert pause.state == "paused"
        assert pause.health()["admission_open"] is False
        assert holder.acquire_generation() is None

    def test_failed_explicit_load_during_battery_queues_resident_suspend(self):
        holder = FakeHolder(FakeEngine("old"))
        pause = _pause(holder)
        holder.swap_gate.clear()
        holder.swap_error = ValueError("explicit load failed")
        error = []
        load = threading.Thread(target=lambda: _capture_error(
            error, holder.swap, model="org/Bad"))
        load.start()
        assert holder.reload_started.wait(1.0)
        pause.on_power_source("battery")
        holder.swap_gate.set()
        load.join(2.0)
        assert isinstance(error[0], ValueError)
        assert pause.state == "unloading" and holder.current is not None
        _step(pause)
        assert pause.state == "paused" and holder.current is None

    def test_suspend_failure_preserves_truthful_recovery_state(self):
        holder = FakeHolder(FakeEngine())
        holder.suspend_error = RuntimeError("suspend failed")
        pause = _pause(holder)
        pause.on_power_source("battery")
        _step(pause)
        assert pause.state == "unloading"
        assert holder.current is not None and "suspend failed" in pause.health()["error"]
        assert any("model still resident" in log for log in pause.logs)

        holder.drop_before_suspend_error = True
        holder.suspend_error = RuntimeError("released during suspend")
        holder._engine = FakeEngine()
        pause.on_power_source("ac")
        pause.on_power_source("battery")
        _step(pause)
        assert holder.current is None and pause.state == "paused"
        assert any("model references released on battery" in log for log in pause.logs)


def _capture_error(out, fn, **kwargs):
    try:
        fn(**kwargs)
    except Exception as exc:  # noqa: BLE001 - test capture
        out.append(exc)


class TestExplicitOperations:
    def test_battery_unload_failure_queues_fallback_suspend(self):
        holder = FakeHolder(FakeEngine())
        pause = _pause(holder)
        pause.on_power_source("battery")
        holder.unload_error = RuntimeError("close failed")
        with pytest.raises(RuntimeError, match="close failed"):
            holder.unload()
        assert pause.state == "unloading"
        assert holder._lifecycle._action == "suspend"
        _step(pause)
        assert pause.state == "paused" and holder.current is None
        assert holder._preserved_prefix is None
        assert holder._suspended is None
        pause.on_power_source("ac")
        assert pause.state == "ready" and holder.current is None
        assert holder.swaps == []

    def test_ac_during_destructive_fallback_suspend_settles_ready_without_resume(self):
        holder = FakeHolder(FakeEngine())
        pause = _pause(holder)
        pause.on_power_source("battery")
        holder.unload_error = RuntimeError("close failed")
        with pytest.raises(RuntimeError, match="close failed"):
            holder.unload()
        assert holder._lifecycle._action == "suspend"
        assert holder._lifecycle._intent is False

        holder.unload_gate.clear()
        pause.start()
        try:
            assert _wait(lambda: holder.suspends == 1)
            assert holder._lifecycle._busy is True

            pause.on_power_source("ac")
            assert pause.health()["source"] == "ac"
            assert pause.state == "unloading"
            assert pause.health()["admission_open"] is False

            holder.unload_gate.set()
            assert _wait(lambda: not holder._lifecycle._busy and pause.state == "ready")

            health = pause.health()
            assert health["source"] == "ac"
            assert health["state"] == "ready"
            assert health["reason"] is None
            assert health["model_loaded"] is False
            assert health["admission_open"] is True
            assert holder._lifecycle._action is None
            assert holder.acquire_generation() is None
            assert holder.acquire_read() is None
            assert holder._suspended is None
            assert holder._preserved_prefix is None
            assert holder.swaps == []
            assert any("released after explicit unload intent" in log for log in pause.logs)
            assert not any("model weights released after explicit unload intent"
                           in log and "prefix cache preserved" in log for log in pause.logs)
        finally:
            holder.unload_gate.set()
            pause.stop()

    def test_unload_drop_then_raise_settles_and_allows_later_load(self):
        holder = FakeHolder(FakeEngine())
        holder.drop_before_unload_error = True
        holder.unload_error = RuntimeError("close failed")
        with pytest.raises(RuntimeError, match="close failed"):
            holder.unload()
        snap = holder.lifecycle_snapshot()
        assert snap["state"] == "ready"
        assert snap["reason"] is None
        assert holder.current is None
        holder.drop_before_unload_error = False
        holder.unload_error = None
        holder.swap(model="org/Recovered")
        assert holder.current is not None and holder.lifecycle_snapshot()["state"] == "ready"

    def test_unload_failure_with_resident_engine_does_not_claim_model_less(self):
        holder = FakeHolder(FakeEngine())
        holder.unload_error = RuntimeError("close failed")
        with pytest.raises(RuntimeError):
            holder.unload()
        assert holder.current is not None
        assert holder.lifecycle_snapshot()["state"] == "ready"

    def test_destructive_fallback_without_generation_does_not_claim_prefix_preserved(self):
        holder = FakeHolder(FakeEngine())
        pause = _pause(holder)
        holder.unload_error = RuntimeError("close failed")
        with pytest.raises(RuntimeError, match="close failed"):
            holder.unload()
        pause.on_power_source("battery")
        assert any("releasing model weights after explicit unload intent" in log
                   for log in pause.logs)
        assert not any("prefix cache preserved" in log for log in pause.logs
                       if "explicit unload intent" in log or "releasing model weights" in log)

    def test_destructive_fallback_after_generation_drain_does_not_claim_prefix_preserved(self):
        holder = FakeHolder(FakeEngine())
        pause = _pause(holder)
        holder.unload_error = RuntimeError("close failed")
        with pytest.raises(RuntimeError, match="close failed"):
            holder.unload()
        lease = holder.acquire_generation()
        assert lease is not None
        pause.on_power_source("battery")
        assert pause.state == "draining"
        lease.release()
        assert any("last admitted request finished — releasing model weights after explicit "
                   "unload intent" in log for log in pause.logs)
        assert not any("prefix cache preserved" in log for log in pause.logs
                       if "explicit unload intent" in log or "releasing model weights" in log)

    def test_queued_resume_is_cancelled_by_explicit_unload(self):
        holder = FakeHolder(None)
        pause = _pause(holder, initial="battery")
        pause.on_power_source("ac")
        assert pause.state == "reloading"
        holder.unload()
        assert holder.current is None and holder._lifecycle._action is None
        assert pause.state == "ready"

    def test_inflight_resume_is_superseded_by_explicit_unload(self):
        holder = FakeHolder(None)
        pause = _pause(holder, initial="battery").start()
        holder.swap_gate.clear()
        try:
            pause.on_power_source("ac")
            assert holder.reload_started.wait(1.0)
            unload = threading.Thread(target=holder.unload)
            unload.start()
            assert _wait(lambda: holder._lifecycle._revision == 1)
            assert holder.acquire_generation() is None
            holder.swap_gate.set()
            unload.join(2.0)
            assert not unload.is_alive()
            assert holder.current is None and pause.state == "ready"
            assert any("automatic reload completed after explicit unload intent" in log
                       for log in pause.logs)
        finally:
            holder.swap_gate.set()
            pause.stop()

    def test_inflight_resume_superseded_by_explicit_swap_is_not_called_unload(self):
        holder = FakeHolder(None)
        pause = _pause(holder, initial="battery").start()
        holder.swap_gate.clear()
        swap = threading.Thread(target=lambda: _capture_error(
            [], holder.swap, model="org/Explicit"))
        try:
            pause.on_power_source("ac")
            assert holder.reload_started.wait(1.0)
            swap.start()
            assert _wait(lambda: holder._lifecycle._explicit_op == "swap")
            holder.swap_gate.set()
            swap.join(2.0)
            assert not swap.is_alive()
            assert not any("after explicit unload intent" in log for log in pause.logs)
        finally:
            holder.swap_gate.set()
            swap.join(2.0)
            pause.stop()


class TestReplayAndMetrics:
    def test_reload_snapshot_preserves_pair_and_machine_policies(self):
        spec = {"model": "org/Target", "drafter": "org/Drafter", "mode": "dspark",
                "lookup_drafts": False, "confidence_threshold": 0.3,
                "max_draft_tokens": "auto", "small_m": False, "sdpa_split": True,
                "cpu_split": 0.25, "kv_bits": 8, "warmup": False,
                "memory_guard": True, "enable_thinking": False,
                "reasoning_effort": "medium"}
        holder = FakeHolder(FakeEngine(), reload_spec=spec)
        pause = _pause(holder)
        pause.on_power_source("battery")
        _step(pause)
        pause.on_power_source("ac")
        _step(pause)
        assert holder.swaps[-1] == spec

    def test_repo_id_pair_reloads_from_complete_cache_without_online_hub(self, monkeypatch):
        calibrate = importlib.import_module("mlx_dspark.calibrate")
        import mlx_dspark.download as download
        from mlx_dspark import load

        snapshot_calls = []

        def cached_snapshot(repo, **kwargs):
            snapshot_calls.append((repo, kwargs))
            if kwargs.get("local_files_only") is True:
                return "/cached/snapshot"
            raise AssertionError("battery reload attempted an online Hub resolution")

        monkeypatch.setattr(load, "local_dir", lambda _repo: None)
        monkeypatch.setattr(load, "snapshot_download", cached_snapshot)
        monkeypatch.setattr(download, "ensure_local", lambda _repo: None)

        class Target:
            model = object()

        class Drafter:
            config = types.SimpleNamespace(block_size=8)

            def bind(self, _model):
                pass

        def fake_target(repo, **kwargs):
            load._resolve(repo)
            return Target(), object()

        def fake_dflash(repo, **kwargs):
            load._resolve(repo)
            return Drafter(), object()

        monkeypatch.setattr(S, "load_target", fake_target)
        monkeypatch.setattr(S, "load_dflash", fake_dflash)
        monkeypatch.setattr(S, "_generation_defaults", lambda _repo: {})
        monkeypatch.setattr(S, "_context_window", lambda _repo: 1024)
        monkeypatch.setattr(S, "_target_config", lambda _repo: None)
        monkeypatch.setattr(calibrate, "apply_small_m", lambda *a, **k: [])
        monkeypatch.setattr(calibrate, "apply_sdpa_split", lambda *a, **k: None)
        monkeypatch.setattr(calibrate, "apply_wide_gemm", lambda *a, **k: None)
        monkeypatch.setattr(calibrate, "apply_cpu_split", lambda *a, **k: None)

        startup = {"model": "org/Target", "drafter": "org/Drafter", "mode": "dflash",
                   "max_draft_tokens": 1, "prefix_cache": False, "warmup": False,
                   "memory_guard": False}
        holder = EngineHolder(None, startup)
        pause = _pause(holder, initial="battery")
        pause.on_power_source("ac")
        _step(pause)
        assert pause.state == "ready" and holder.current is not None
        pause.on_power_source("battery")
        _step(pause)
        pause.on_power_source("ac")
        _step(pause)
        assert pause.state == "ready" and holder.current is not None
        assert snapshot_calls and all(kwargs == {"local_files_only": True}
                                      for _, kwargs in snapshot_calls)

    def test_battery_metrics_record_attempts_outcomes_and_duration(self):
        holder = FakeHolder(FakeEngine())
        pause = _pause(holder)
        pause.on_power_source("battery")
        _step(pause)
        metrics = pause.metrics()
        assert metrics["unload_attempts_total"] == 1
        assert metrics["unload_successes_total"] == 1
        assert metrics["unload_failures_total"] == 0
        assert metrics["last_unload_seconds"] is not None

    def test_manual_ac_unload_keeps_power_gauge_open_but_no_generation_lease(self):
        holder = FakeHolder(FakeEngine())
        _pause(holder)
        holder.unload()
        snap = holder.lifecycle_snapshot()
        assert snap["state"] == "ready"
        assert snap["admission_open"] is True
        assert snap["model_loaded"] is False
        assert snap["reason"] is None
        assert holder.acquire_generation() is None
        assert holder.acquire_read() is None

    def test_battery_rejection_metric_ignores_ordinary_ac_unavailability(self):
        holder = FakeHolder(FakeEngine())
        pause = _pause(holder)
        holder.unload()
        assert holder.acquire_generation() is None
        assert pause.metrics()["rejected_inference_total"] == 0
        pause.on_power_source("battery")
        assert holder.acquire_generation() is None
        assert pause.metrics()["rejected_inference_total"] == 1

    def test_power_facade_has_no_lifecycle_ownership_methods(self):
        assert not hasattr(BatteryPause, "admit")
        assert not hasattr(BatteryPause, "release")
        assert not hasattr(BatteryPause, "begin_admin_load")
        assert not hasattr(BatteryPause, "finish_admin_load")
        assert not hasattr(BatteryPause, "admin_unload")
        assert not hasattr(EngineHolder, "suspend")
