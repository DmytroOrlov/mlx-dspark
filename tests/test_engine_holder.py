"""EngineHolder delegation + swap bookkeeping — model-free (a fake engine stands in for the
real one, since the swap's actual model load needs weights and is exercised on-device)."""

import threading
import time
import types
import weakref

import pytest

from mlx_dspark import server as S
from mlx_dspark.prefix_cache import PrefixCache
from mlx_dspark.server import EngineHolder


class FakeEngine:
    def __init__(self, model_id="m1"):
        self.model_id = model_id
        self.mode = "dspark"
        self.closed = False

    def close(self):
        self.closed = True

    def metrics(self):
        return {"model": self.model_id}


def _replay_engine(**overrides):
    engine = FakeEngine("running")
    engine.target_repo = "org/Target"
    engine.drafter_repo = "org/Drafter"
    engine.lookup_drafts = False
    engine.confidence_threshold = 0.3
    engine.target = types.SimpleNamespace(kv_bits=8)
    engine.cap_controller = None
    engine.cap_pinned = True
    engine.max_draft_tokens = 7
    engine.template_defaults = {"enable_thinking": False, "reasoning_effort": "medium"}
    engine.small_m = False
    engine.sdpa_split = True
    engine.cpu_split = None
    engine.cpu_split_suspended = 0.25
    engine._cpu_split_requested = 0.25
    engine.warmup_enabled = False
    engine.memory_guard = object()
    for key, value in overrides.items():
        setattr(engine, key, value)
    return engine


def holder(engine=None):
    # load_kwargs is only read by swap(); delegation/status tests never load, so {} is fine.
    return EngineHolder(engine or FakeEngine(), load_kwargs={})


class TestDelegation:
    def test_attributes_delegate_to_current_engine(self):
        h = holder(FakeEngine("qwen"))
        assert h.model_id == "qwen"          # via __getattr__
        assert h.mode == "dspark"

    def test_methods_delegate(self):
        h = holder(FakeEngine("g"))
        assert h.metrics() == {"model": "g"}

    def test_holder_own_attributes_win_over_delegation(self):
        h = holder()
        assert h.ready is True               # a real property, not delegated
        assert callable(h.swap)


class TestStatus:
    def test_ready_when_engine_present(self):
        h = holder(FakeEngine("x"))
        assert h.ready is True
        s = h.status()
        assert s == {"ready": True, "loading": False, "model": "x", "error": None}

    def test_not_ready_without_engine(self):
        h = holder()
        h._engine = None
        assert h.ready is False
        assert h.status()["model"] is None

    def test_access_without_engine_raises_clearly(self):
        h = holder()
        h._engine = None
        # The dispatcher gates on `ready` first; a stray delegated access should still be a
        # clear message, not an AttributeError on None.
        with pytest.raises(RuntimeError, match="no model is loaded"):
            _ = h.model_id


class TestReloadKwargs:
    def test_reproduces_the_running_engine_configuration(self):
        engine = _replay_engine()
        h = EngineHolder(engine, load_kwargs={"model": "startup"})
        kw = h.reload_kwargs()
        assert kw["model"] == "org/Target"
        assert kw["drafter"] == "org/Drafter"
        assert kw["mode"] == "dspark"
        assert kw["lookup_drafts"] is False
        assert kw["confidence_threshold"] == 0.3
        assert kw["max_draft_tokens"] == 7
        assert kw["kv_bits"] == 8
        assert kw["enable_thinking"] is False
        assert kw["reasoning_effort"] == "medium"
        assert kw["warmup"] is False
        assert kw["memory_guard"] is True
        assert kw["small_m"] is False
        assert kw["sdpa_split"] is True
        assert kw["cpu_split"] == 0

    def test_auto_and_derived_caps_and_bf16_round_trip(self):
        engine = _replay_engine(cap_controller=object(), target=types.SimpleNamespace(kv_bits=0))
        h = EngineHolder(engine, load_kwargs={})
        assert h.reload_kwargs()["max_draft_tokens"] == "auto"
        assert h.reload_kwargs()["kv_bits"] is None
        engine.cap_controller = None
        engine.cap_pinned = False
        assert h.reload_kwargs()["max_draft_tokens"] is None

    def test_resume_maps_none_false_and_zero_over_stale_startup_values(self, monkeypatch):
        captured = {}
        holder = EngineHolder(None, load_kwargs={
            "model": "stale", "drafter": "stale-drafter", "kv_bits": 8,
            "lookup_drafts": True, "small_m": True, "sdpa_split": True,
            "cpu_split": 0.5, "warmup": True, "memory_guard": True,
        })
        holder._suspended = S._SuspendedState({
            "model": "org/Current", "drafter": None, "mode": "lookup",
            "max_draft_tokens": None, "lookup_drafts": False,
            "confidence_threshold": 0.2, "small_m": False, "sdpa_split": False,
            "cpu_split": 0, "kv_bits": None, "warmup": False,
            "memory_guard": False, "enable_thinking": False,
            "reasoning_effort": "low",
        })

        def capture(**kwargs):
            captured.update(kwargs)
            return FakeEngine("current")

        monkeypatch.setattr(S.Engine, "load", staticmethod(capture))
        monkeypatch.setattr(S, "maybe_batch_engine", lambda engine, _max_batch: engine)
        holder._resume_now()
        assert captured["model"] == "org/Current"
        assert "drafter" in captured and captured["drafter"] is None
        assert captured["mode"] == "lookup"
        assert captured["max_draft_tokens"] is None
        assert captured["kv_bits"] is None
        assert captured["lookup_drafts"] is False
        assert captured["confidence_threshold"] == 0.2
        assert captured["small_m"] is False
        assert captured["sdpa_split"] is False
        assert captured["cpu_split"] == 0
        assert captured["warmup"] is False
        assert captured["memory_guard"] is False
        assert captured["enable_thinking"] is False
        assert captured["reasoning_effort"] == "low"

    def test_runtime_cpu_split_shed_is_refreshed_after_quiescent_suspend(self):
        engine = _replay_engine(cpu_split=0.25, cpu_split_suspended=None)

        def suspend():
            # Engine.suspend() tears down the guard before the holder refreshes the
            # one runtime-mutable replay field.
            engine.memory_guard = None
            return object()

        engine.suspend = suspend
        h = EngineHolder(engine, load_kwargs={})
        lease = h.acquire_generation()

        h.set_inhibited(active=True)
        saved = h._suspended
        assert saved.resume_spec["cpu_split"] == 0.25
        assert saved.resume_spec["memory_guard"] is True

        # This is the MemoryGuard transition that can happen while the admitted
        # generation is draining, after the early battery snapshot was taken.
        engine.cpu_split_suspended = 0.25
        engine.cpu_split = None
        lease.release()
        h._lifecycle._step()

        assert h._suspended is saved
        assert saved.resume_spec["cpu_split"] == 0
        assert saved.resume_spec["memory_guard"] is True


class TestSwap:
    def test_successful_swap_closes_old_and_installs_new(self, monkeypatch):
        old = FakeEngine("old")
        h = holder(old)
        new = FakeEngine("new")

        import mlx_dspark.server as server
        monkeypatch.setattr(server.Engine, "load", staticmethod(lambda **kw: new))
        monkeypatch.setattr(server, "maybe_batch_engine", lambda e, b: e)

        status = h.swap(model="new-repo")
        assert old.closed is True            # released before the new one
        assert h.current is new
        assert status == {"ready": True, "loading": False, "model": "new", "error": None}

    def test_failed_swap_leaves_no_engine_but_records_the_error(self, monkeypatch):
        old = FakeEngine("old")
        h = holder(old)
        pause = S.BatteryPause(h, log=lambda _msg: None)

        import mlx_dspark.server as server

        def boom(**kw):
            raise ValueError("unknown model")

        monkeypatch.setattr(server.Engine, "load", staticmethod(boom))

        with pytest.raises(ValueError, match="unknown model"):
            h.swap(model="bogus")
        assert old.closed is True            # old was still released
        assert h.ready is False
        assert h.status()["error"] == "unknown model"
        assert h._loading is False           # flag cleared even on failure
        assert h._lifecycle._intent is False
        pause.on_power_source("battery")
        assert pause.state == "paused"
        pause.on_power_source("ac")
        assert pause.state == "ready"
        assert h.current is None
        assert h._lifecycle._action is None

    def test_failed_swap_keeps_machine_policies_from_startup(self, monkeypatch):
        calls = []

        def fake_load(**kwargs):
            calls.append(dict(kwargs))
            if kwargs.get("model") == "org/Bad":
                raise RuntimeError("load failed")
            return FakeEngine("good")

        monkeypatch.setattr(S.Engine, "load", staticmethod(fake_load))
        monkeypatch.setattr(S, "maybe_batch_engine", lambda engine, _max_batch: engine)
        h = EngineHolder(FakeEngine("startup"), load_kwargs={
            "model": "org/Startup", "small_m": True, "sdpa_split": True,
            "cpu_split": 0.25, "warmup": False})
        with pytest.raises(RuntimeError, match="load failed"):
            h.swap(model="org/Bad", small_m=False, sdpa_split=False, cpu_split=0)
        assert h._load_kwargs["small_m"] is True
        assert h._load_kwargs["sdpa_split"] is True
        assert h._load_kwargs["cpu_split"] == 0.25
        h.swap(model="org/Good")
        assert calls[-1]["small_m"] is True
        assert calls[-1]["sdpa_split"] is True
        assert calls[-1]["cpu_split"] == 0.25

    def test_failed_explicit_swap_with_resident_engine_keeps_resume_intent(self,
                                                                            monkeypatch):
        old = FakeEngine("old")
        h = holder(old)
        pause = S.BatteryPause(h, log=lambda _msg: None)

        def fail_while_resident():
            raise ValueError("rejected before release")

        with pytest.raises(ValueError, match="rejected before release"):
            h._lifecycle.explicit_swap(fail_while_resident)
        assert h.current is old
        assert h._lifecycle._intent is True

        suspended = []

        def suspend():
            suspended.append(h._engine)
            h._engine = None

        def resume():
            h._engine = old

        monkeypatch.setattr(h, "_capture_suspended_state", lambda: None)
        monkeypatch.setattr(h, "_suspend_now", suspend)
        monkeypatch.setattr(h, "_resume_now", resume)
        pause.on_power_source("battery")
        h._lifecycle._step()
        assert suspended == [old]
        assert h.current is None
        pause.on_power_source("ac")
        h._lifecycle._step()
        assert h.current is old
        assert pause.state == "ready"

    def test_swap_passes_model_and_overrides_through(self, monkeypatch):
        h = EngineHolder(FakeEngine("old"), load_kwargs={"mode": "dspark", "prefix_cache": True})
        captured = {}

        import mlx_dspark.server as server

        def capture(**kw):
            captured.update(kw)
            return FakeEngine("new")

        monkeypatch.setattr(server.Engine, "load", staticmethod(capture))
        monkeypatch.setattr(server, "maybe_batch_engine", lambda e, b: e)

        h.swap(model="repo", mode="lookup", max_draft=4)
        assert captured["model"] == "repo"
        assert captured["mode"] == "lookup"          # override applied
        assert captured["max_draft_tokens"] == 4
        assert captured["prefix_cache"] is True      # base kwarg preserved

    def test_swap_lookup_drafts_override_and_per_pair_default(self, monkeypatch):
        """An explicit lookup_drafts rides the swap; absent, the server's stored kwargs
        pass through unchanged — a serve started without the flag stores None, so each
        swapped-in pair re-resolves its own registry default inside Engine.load."""
        h = EngineHolder(FakeEngine("old"), load_kwargs={"lookup_drafts": None})
        captured = {}

        import mlx_dspark.server as server

        def capture(**kw):
            captured.update(kw)
            return FakeEngine("new")

        monkeypatch.setattr(server.Engine, "load", staticmethod(capture))
        monkeypatch.setattr(server, "maybe_batch_engine", lambda e, b: e)

        h.swap(model="repo", lookup_drafts=False)
        assert captured["lookup_drafts"] is False    # request override wins
        captured.clear()
        h.swap(model="repo2")
        assert captured["lookup_drafts"] is None     # unset -> Engine.load resolves per pair

    def test_batch_engine_inner_is_also_closed(self, monkeypatch):
        inner = FakeEngine("old-inner")

        class FakeBatch:
            def __init__(self, e):
                self.engine = e
                self.closed = False

            def close(self):
                self.closed = True

        batch = FakeBatch(inner)
        h = EngineHolder(batch, load_kwargs={})

        import mlx_dspark.server as server
        monkeypatch.setattr(server.Engine, "load", staticmethod(lambda **kw: FakeEngine("new")))
        monkeypatch.setattr(server, "maybe_batch_engine", lambda e, b: e)

        h.swap(model="repo")
        assert batch.closed is True          # scheduler stopped
        assert inner.closed is True          # AND the wrapped models freed


class TestUnload:
    def test_unload_releases_and_reports_no_model(self):
        eng = FakeEngine("m")
        h = holder(eng)
        s = h.unload()
        assert eng.closed is True
        assert h.ready is False
        assert s == {"ready": False, "loading": False, "model": None, "error": None}

    def test_unload_twice_is_a_noop(self):
        h = holder()
        h.unload()
        assert h.unload()["ready"] is False   # no raise on the empty holder

    def test_unload_closes_a_batch_engine_inner(self):
        inner = FakeEngine("inner")

        class FakeBatch:
            def __init__(self, e):
                self.engine = e
                self.closed = False

            def close(self):
                self.closed = True

        batch = FakeBatch(inner)
        h = EngineHolder(batch, load_kwargs={})
        h.unload()
        assert batch.closed is True and inner.closed is True

    def test_load_after_unload_works(self, monkeypatch):
        h = holder()
        h.unload()

        import mlx_dspark.server as server
        monkeypatch.setattr(server.Engine, "load", staticmethod(lambda **kw: FakeEngine("new")))
        monkeypatch.setattr(server, "maybe_batch_engine", lambda e, b: e)

        status = h.swap(model="repo")
        assert status["ready"] is True and status["model"] == "new"


class TestSwapConfidence:
    def test_swap_confidence_override_and_default_passthrough(self, monkeypatch):
        """An explicit confidence_threshold rides the swap (the cap+confidence bundle a
        client applies for pairs whose measured best needs it — Qwen3.8-27B-4bit's is
        cap 7 + 0.3); absent, the server's stored kwargs pass through untouched."""
        h = EngineHolder(FakeEngine("old"), load_kwargs={"confidence_threshold": 0.0})
        captured = {}

        import mlx_dspark.server as server

        def capture(**kw):
            captured.update(kw)
            return FakeEngine("new")

        monkeypatch.setattr(server.Engine, "load", staticmethod(capture))
        monkeypatch.setattr(server, "maybe_batch_engine", lambda e, b: e)

        h.swap(model="repo", confidence_threshold=0.3)
        assert captured["confidence_threshold"] == 0.3
        captured.clear()
        h.swap(model="repo2")
        assert captured["confidence_threshold"] == 0.0   # unset -> server default kept

    def test_swap_small_m_override_and_default_passthrough(self, monkeypatch):
        """An explicit small_m rides the swap (the serve-side kernel A/B issue #14 asked
        for); absent, the server's stored kwargs pass through untouched — a serve started
        without the flag stores None, so Engine.load applies its probe-gated default."""
        h = EngineHolder(FakeEngine("old"), load_kwargs={"small_m": None})
        captured = {}

        import mlx_dspark.server as server

        def capture(**kw):
            captured.update(kw)
            return FakeEngine("new")

        monkeypatch.setattr(server.Engine, "load", staticmethod(capture))
        monkeypatch.setattr(server, "maybe_batch_engine", lambda e, b: e)

        h.swap(model="repo", small_m=False)
        assert captured["small_m"] is False              # request override wins
        captured.clear()
        h.swap(model="repo2")
        assert captured["small_m"] is None               # unset -> probe-gated default

    def test_status_reports_download_progress_while_loading(self, monkeypatch):
        """While a swap is fetching weights, status() carries the download progress so
        /health can show a real bar and the client can offer Cancel."""
        import mlx_dspark.download as download

        h = EngineHolder(FakeEngine(), load_kwargs={})
        h._loading = True
        monkeypatch.setattr(download, "progress", lambda: {
            "repo": "org/model", "bytes_done": 5, "bytes_total": 10})
        s = h.status()
        assert s["download"]["repo"] == "org/model"
        h._loading = False
        assert "download" not in h.status()

    def test_swap_context_window_override_is_sticky(self, monkeypatch):
        """context_window rides the swap as a load override (the KV-RAM lever) and is
        STICKY: a later swap that omits it keeps the last explicit value instead of
        reverting to the model's 262k max (community report — scripts set it once).
        Explicit 0 resets to the model's own maximum."""
        h = EngineHolder(FakeEngine("old"), load_kwargs={"context_window": None})
        captured = {}

        import mlx_dspark.server as server

        def capture(**kw):
            captured.update(kw)
            return FakeEngine("new")

        monkeypatch.setattr(server.Engine, "load", staticmethod(capture))
        monkeypatch.setattr(server, "maybe_batch_engine", lambda e, b: e)

        h.swap(model="repo", context_window=32768)
        assert captured["context_window"] == 32768
        captured.clear()
        h.swap(model="repo2")                            # omitted -> keeps 32768
        assert captured["context_window"] == 32768
        captured.clear()
        h.swap(model="repo3", context_window=0)          # 0 -> back to the model's max
        assert captured["context_window"] is None
        captured.clear()
        h.swap(model="repo4")                            # and the reset sticks too
        assert captured["context_window"] is None


class TestKvBitsOverride:
    def test_swap_kv_bits_override_and_zero_means_full_precision(self, monkeypatch):
        """issue #17: `kv_bits` rides /admin/load. Explicit 4/8 quantize the swapped-in
        target's KV cache; explicit 0 means full precision (Engine.load's None); omitted
        keeps whatever the server was started with."""

        import mlx_dspark.server as server

        h = server.EngineHolder(FakeEngine("old"), load_kwargs={"kv_bits": 8})
        captured = {}

        def capture(**kw):
            captured.update(kw)
            return FakeEngine("new")

        monkeypatch.setattr(server.Engine, "load", staticmethod(capture))
        monkeypatch.setattr(server, "maybe_batch_engine", lambda e, b: e)

        h.swap(model="repo", kv_bits=4)
        assert captured["kv_bits"] == 4              # override applied
        captured.clear()
        h.swap(model="repo", kv_bits=0)
        assert captured["kv_bits"] is None           # explicit full precision
        captured.clear()
        h.swap(model="repo")
        assert captured["kv_bits"] == 8              # omitted -> server's startup setting


class _Guard:
    def __init__(self, prefix):
        self.prefix = prefix
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True
        return self

    def stop(self):
        self.stopped = True


def _handoff_engine(fresh, guard):
    engine = object.__new__(S.Engine)
    engine.prefix = fresh
    engine.memory_guard = guard
    return engine


def test_compatible_prefix_handoff_retargets_memory_guard():
    fresh = PrefixCache(list, compatibility=("model", "trim"))
    fresh_ref = weakref.ref(fresh)
    preserved = PrefixCache(list, compatibility=("model", "trim"))
    engine = _handoff_engine(fresh, _Guard(fresh))

    assert engine.attach_preserved_prefix(preserved) is True
    assert engine.prefix is preserved
    assert engine.memory_guard.prefix is preserved
    fresh = None
    assert fresh_ref() is None


def test_incompatible_prefix_handoff_keeps_fresh_and_discards_preserved_state():
    fresh = PrefixCache(list, compatibility=("new-model", "trim"))
    preserved = PrefixCache(list, compatibility=("old-model", "trim"), min_reuse=1)
    preserved.store([], None, [1], [2])
    engine = _handoff_engine(fresh, _Guard(fresh))

    assert engine.attach_preserved_prefix(preserved) is False
    assert engine.prefix is fresh
    assert engine.memory_guard.prefix is fresh
    assert preserved.info()["slots"] == []


def test_compatible_prefix_handoff_preserves_l2_spill_artifact(tmp_path):
    fresh = PrefixCache(list, l2_dir=str(tmp_path), compatibility=("model", "trim"))
    preserved = PrefixCache(list, l2_dir=str(tmp_path), compatibility=("model", "trim"))
    spill = tmp_path / "target_cache_7.safetensors"
    spill.write_bytes(b"preserved spill artifact")
    engine = _handoff_engine(fresh, _Guard(fresh))

    assert engine.attach_preserved_prefix(preserved) is True
    assert spill.exists()


class _Model:
    pass


def _candidate_engine(compatibility):
    fresh = PrefixCache(list, compatibility=compatibility,
                        compatibility_known=True)
    guard = _Guard(fresh)
    candidate = types.SimpleNamespace(
        model_id="candidate", prefix=fresh, memory_guard=guard,
        target=_Model(), drafter=_Model(), tokenizer=None, cap_controller=None,
        _depth_capper=None, _executor=None)
    candidate.attach_preserved_prefix = types.MethodType(
        S.Engine.attach_preserved_prefix, candidate)
    candidate.start_memory_guard = types.MethodType(S.Engine.start_memory_guard, candidate)
    candidate._close_unpublished = types.MethodType(S.Engine._close_unpublished, candidate)
    return candidate


def _preserved_prefix(tmp_path, compatibility):
    preserved = PrefixCache(list, l2_dir=str(tmp_path), min_reuse=1,
                            compatibility=compatibility, compatibility_known=True)
    preserved.store([], None, [1, 2], [3])
    spill = tmp_path / "target_cache_7.safetensors"
    spill.write_bytes(b"preserved spill artifact")
    preserved.detach_model()
    return preserved, spill


def test_battery_handoff_defers_guard_until_preserved_prefix_is_active(monkeypatch, tmp_path):
    compatibility = ("model", "trim")
    preserved, _spill = _preserved_prefix(tmp_path, compatibility)
    candidate = _candidate_engine(compatibility)
    holder = EngineHolder(None, load_kwargs={"model": "repo"})
    holder._preserved_prefix = preserved
    loaded = {}

    def load(**kwargs):
        loaded.update(kwargs)
        assert candidate.memory_guard.started is False
        return candidate

    monkeypatch.setattr(S.Engine, "load", staticmethod(load))
    monkeypatch.setattr(S, "maybe_batch_engine", lambda engine, _max_batch: engine)

    holder.swap(model="repo", preserved_prefix=preserved)

    assert loaded["defer_memory_guard_start"] is True
    assert candidate.memory_guard.started is True
    assert candidate.memory_guard.prefix is preserved


@pytest.mark.parametrize("wrapped", [False, True])
def test_failed_publication_after_compatible_handoff_retains_preserved_state(
        monkeypatch, tmp_path, wrapped):
    compatibility = ("model", "trim")
    preserved, spill = _preserved_prefix(tmp_path, compatibility)
    candidate = _candidate_engine(compatibility)
    target_ref = weakref.ref(candidate.target)
    drafter_ref = weakref.ref(candidate.drafter)
    guard = candidate.memory_guard
    holder = EngineHolder(None, load_kwargs={"model": "repo"})
    holder._preserved_prefix = preserved

    class Batch:
        def __init__(self, engine):
            self.engine = engine
            self.closed = False

        def close(self):
            self.closed = True

        def start_memory_guard(self):
            raise RuntimeError("publication failed")

    monkeypatch.setattr(S.Engine, "load", staticmethod(lambda **_kwargs: candidate))
    if wrapped:
        monkeypatch.setattr(S, "maybe_batch_engine", lambda engine, _max_batch: Batch(engine))
    else:
        monkeypatch.setattr(S, "maybe_batch_engine",
                            lambda _engine, _max_batch: (_ for _ in ()).throw(
                                RuntimeError("publication failed")))

    with pytest.raises(RuntimeError, match="publication failed"):
        holder.swap(model="repo", preserved_prefix=preserved)

    assert holder.current is None
    assert holder._preserved_prefix is preserved
    assert candidate.prefix is None
    assert preserved.make_cache is None and preserved.make_ctx is None
    assert preserved.info()["cached_tokens"] == 2
    assert spill.exists()
    assert guard.stopped is True
    candidate = None
    import gc
    gc.collect()
    assert target_ref() is None and drafter_ref() is None


def test_failed_publication_after_incompatible_handoff_retains_runtime_resume_spec(
        monkeypatch, tmp_path):
    preserved, _spill = _preserved_prefix(tmp_path, ("old-model", "trim"))
    candidate = _candidate_engine(("new-model", "trim"))
    holder = EngineHolder(None, load_kwargs={"model": "org/Startup"})
    holder._preserved_prefix = preserved
    holder._suspended = S._SuspendedState({"model": "org/Runtime"})
    holder._suspended.preserved_prefix = preserved
    calls = []

    monkeypatch.setattr(S.Engine, "load", staticmethod(lambda **kwargs: (
        calls.append(dict(kwargs)) or candidate)))
    monkeypatch.setattr(S, "maybe_batch_engine", lambda *_args: (_ for _ in ()).throw(
        RuntimeError("publication failed")))

    with pytest.raises(RuntimeError, match="publication failed"):
        holder._resume_now()

    assert holder._suspended is not None
    assert holder._suspended.resume_spec["model"] == "org/Runtime"
    assert holder._suspended.preserved_prefix is None
    assert holder._preserved_prefix is None

    retry = FakeEngine("runtime")
    monkeypatch.setattr(S.Engine, "load", staticmethod(lambda **kwargs: (
        calls.append(dict(kwargs)) or retry)))
    monkeypatch.setattr(S, "maybe_batch_engine", lambda engine, _max_batch: engine)
    holder._resume_now()

    assert calls[-1]["model"] == "org/Runtime"


def test_discard_resumable_state_clears_ownership_after_cleanup_failure():
    class Prefix:
        def __init__(self, name, error=False):
            self.name = name
            self.error = error
            self.reset_calls = 0

        def reset(self):
            self.reset_calls += 1
            if self.error:
                raise RuntimeError(f"{self.name} reset failed")

    first = Prefix("first", error=True)
    second = Prefix("second")
    holder = EngineHolder(None, load_kwargs={})
    holder._preserved_prefix = first
    holder._suspended = S._SuspendedState({})
    holder._suspended.preserved_prefix = second

    with pytest.raises(RuntimeError, match="first reset failed"):
        holder._discard_resumable_state()

    assert first.reset_calls == 1
    assert second.reset_calls == 1
    assert holder._preserved_prefix is None
    assert holder._suspended is None

    alias = Prefix("alias", error=True)
    holder._preserved_prefix = alias
    holder._suspended = S._SuspendedState({})
    holder._suspended.preserved_prefix = alias
    with pytest.raises(RuntimeError, match="alias reset failed"):
        holder._discard_resumable_state()
    assert alias.reset_calls == 1
    assert holder._preserved_prefix is None
    assert holder._suspended is None


# --------------------------------------------------------------------------- lifecycle races


def _wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.001)
    return predicate()


def _arm_lifecycle(holder, action, phase):
    lifecycle = holder._lifecycle
    with lifecycle._lock:
        lifecycle._intent = True
        lifecycle._generation_open = False
        lifecycle._read_open = False
        lifecycle._set_phase(phase)
        lifecycle._action = action
        lifecycle._cond.notify_all()
    return lifecycle


def test_explicit_swap_invalidates_inflight_auto_resume(monkeypatch):
    holder = EngineHolder(None, load_kwargs={})
    lifecycle = _arm_lifecycle(holder, "resume", "reloading")
    entered = threading.Event()
    release = threading.Event()

    def auto_resume():
        entered.set()
        assert release.wait(2.0)

    monkeypatch.setattr(holder, "_resume_now", auto_resume)
    explicit_started = threading.Event()

    def explicit_swap():
        explicit_started.set()
        holder._engine = FakeEngine("explicit")

    auto = threading.Thread(target=lifecycle._step)
    auto.start()
    assert entered.wait(1.0)
    explicit = threading.Thread(target=lambda: lifecycle.explicit_swap(explicit_swap))
    explicit.start()
    assert _wait_until(lambda: lifecycle._revision == 1)
    assert lifecycle.acquire_generation() is None
    assert lifecycle.acquire_read() is None
    release.set()
    auto.join(2.0)
    explicit.join(2.0)
    assert not auto.is_alive() and not explicit.is_alive()
    assert explicit_started.is_set()
    assert holder.current.model_id == "explicit"
    assert lifecycle._action is None
    assert lifecycle.phase == "ready"
    assert lifecycle.snapshot()["admission_open"] is True


def test_explicit_swap_invalidates_inflight_auto_suspend(monkeypatch):
    holder = EngineHolder(FakeEngine("startup"), load_kwargs={})
    lifecycle = _arm_lifecycle(holder, "suspend", "unloading")
    entered = threading.Event()
    release = threading.Event()

    def auto_suspend():
        entered.set()
        assert release.wait(2.0)
        holder._engine = None

    monkeypatch.setattr(holder, "_suspend_now", auto_suspend)
    explicit_started = threading.Event()

    def explicit_swap():
        explicit_started.set()
        holder._engine = FakeEngine("explicit")

    auto = threading.Thread(target=lifecycle._step)
    auto.start()
    assert entered.wait(1.0)
    explicit = threading.Thread(target=lambda: lifecycle.explicit_swap(explicit_swap))
    explicit.start()
    assert _wait_until(lambda: lifecycle._revision == 1)
    release.set()
    auto.join(2.0)
    explicit.join(2.0)
    assert not auto.is_alive() and not explicit.is_alive()
    assert explicit_started.is_set()
    assert holder.current.model_id == "explicit"
    assert lifecycle._action is None
    assert lifecycle.phase == "ready"
    assert lifecycle._resume_stats["attempts"] == 0


def test_explicit_unload_keeps_admission_closed_through_stale_auto_resume(monkeypatch):
    holder = EngineHolder(None, load_kwargs={})
    lifecycle = _arm_lifecycle(holder, "resume", "reloading")
    entered = threading.Event()
    release = threading.Event()

    def auto_resume():
        entered.set()
        assert release.wait(2.0)

    monkeypatch.setattr(holder, "_resume_now", auto_resume)
    unloaded = threading.Event()

    def explicit_unload():
        unloaded.set()
        holder._engine = None

    auto = threading.Thread(target=lifecycle._step)
    auto.start()
    assert entered.wait(1.0)
    explicit = threading.Thread(target=lambda: lifecycle.explicit_unload(explicit_unload))
    explicit.start()
    assert _wait_until(lambda: lifecycle._revision == 1)
    assert lifecycle.acquire_generation() is None
    assert lifecycle.acquire_read() is None
    release.set()
    auto.join(2.0)
    explicit.join(2.0)
    assert not auto.is_alive() and not explicit.is_alive()
    assert unloaded.is_set()
    assert holder.current is None
    assert lifecycle.phase == "ready"
    assert lifecycle.snapshot()["admission_open"] is True
    assert lifecycle._generation_open is False
    assert lifecycle.acquire_generation() is None
    assert lifecycle.acquire_read() is None


def test_concurrent_explicit_swaps_have_one_controller_finalizer(monkeypatch):
    holder = EngineHolder(FakeEngine("old"), load_kwargs={})
    lifecycle = holder._lifecycle
    first_entered = threading.Event()
    release_first = threading.Event()
    order = []
    active = 0
    max_active = 0
    guard = threading.Lock()

    def first():
        nonlocal active, max_active
        with guard:
            active += 1
            max_active = max(max_active, active)
        first_entered.set()
        assert release_first.wait(2.0)
        order.append("first")
        holder._engine = FakeEngine("first")
        with guard:
            active -= 1

    def second():
        nonlocal active, max_active
        with guard:
            active += 1
            max_active = max(max_active, active)
        order.append("second")
        holder._engine = FakeEngine("second")
        with guard:
            active -= 1

    t1 = threading.Thread(target=lambda: lifecycle.explicit_swap(first))
    t2 = threading.Thread(target=lambda: lifecycle.explicit_swap(second))
    t1.start()
    assert first_entered.wait(1.0)
    t2.start()
    time.sleep(0.02)
    assert order == []
    assert lifecycle.acquire_generation() is None
    release_first.set()
    t1.join(2.0)
    t2.join(2.0)
    assert not t1.is_alive() and not t2.is_alive()
    assert order == ["first", "second"]
    assert max_active == 1
    assert holder.current.model_id == "second"
    assert lifecycle.phase == "ready"


def test_cancelled_drain_clears_battery_reason():
    holder = EngineHolder(FakeEngine(), load_kwargs={})
    holder._suspended = S._SuspendedState({})
    pause = S.BatteryPause(holder, log=lambda _msg: None)
    lease = holder.acquire_generation()
    pause.on_power_source("battery")
    assert pause.health()["state"] == "draining"
    pause.on_power_source("ac")
    assert pause.health()["state"] == "ready"
    assert pause.health()["admission_open"] is True
    assert pause.health()["reason"] is None
    lease.release()


def test_cancelled_queued_suspend_clears_battery_reason():
    holder = EngineHolder(FakeEngine(), load_kwargs={})
    holder._suspended = S._SuspendedState({})
    pause = S.BatteryPause(holder, log=lambda _msg: None)
    pause.on_power_source("battery")
    assert pause.health()["state"] == "unloading"
    pause.on_power_source("ac")
    assert pause.health()["state"] == "ready"
    assert pause.health()["admission_open"] is True
    assert pause.health()["reason"] is None
    assert holder._lifecycle._action is None


def test_suspend_waits_for_concrete_read_lease(monkeypatch):
    holder = EngineHolder(FakeEngine("read-pinned"), load_kwargs={})
    read = holder.acquire_read()
    assert read is not None
    lifecycle = _arm_lifecycle(holder, "suspend", "unloading")
    entered = threading.Event()

    def suspend():
        entered.set()
        holder._engine = None

    monkeypatch.setattr(holder, "_suspend_now", suspend)
    worker = threading.Thread(target=lifecycle._step)
    worker.start()
    assert _wait_until(lambda: lifecycle._busy)
    assert not entered.is_set()
    read.release()
    worker.join(2.0)
    assert not worker.is_alive()
    assert entered.is_set()
