"""CLI-level tests for --pause-on-battery: the serve-only option must never leak into
other commands (a regression where cmd_generate's namespace check AttributeError'd
every `mlx-dspark generate`), the macOS-only validation lives in cmd_serve, and the
STARTUP power decision picks between a normal load and a paused model-less start —
all model-free (no weights, no Metal, no real power events)."""

from __future__ import annotations

import sys

import pytest

from mlx_dspark import cli
from mlx_dspark import server as S


class _FakeEngine:
    """Bare engine stand-in for EngineHolder construction (nothing MLX)."""

    mode = "lookup"
    model_id = "Fake"
    target_repo = "org/T"
    drafter_repo = None
    prefix = None
    memory_guard = None
    cap_controller = None
    sampling_defaults = {}
    small_m = False
    sdpa_split = False
    cpu_split = None
    cap_pinned = False
    max_draft_tokens = None
    warmup_enabled = False
    template_defaults = {}
    machine = {}
    load_notes = []
    created = 1
    confidence_threshold = 0.0
    stats = {}


class _StubMonitor:
    """PowerMonitor stand-in: no pmset child, no threads — start() feeds the initial
    reading to the coordinator synchronously exactly like the real one, and stop()
    is recordable."""

    source: str | None = "ac"
    instances: list = []

    def __init__(self, on_change, **kw):
        self.on_change = on_change
        self.stopped = False
        _StubMonitor.instances.append(self)

    def start(self):
        if _StubMonitor.source is not None:
            self.on_change(_StubMonitor.source)
        return self

    def stop(self):
        self.stopped = True


class TestParserIsolation:
    def test_generate_command_does_not_know_the_serve_option(self, monkeypatch):
        """Regression: the serve-only --pause-on-battery flag once leaked into
        cmd_generate's post-parse code, breaking EVERY `mlx-dspark generate`
        invocation. The generate parser must parse, resolve, and reach the weights
        loader (stubbed) with no pause_on_battery attribute — the old bug raised
        AttributeError immediately after parse_args, so any generate call failed."""
        import mlx_dspark.load as load_mod

        def boom(*a, **k):
            raise RuntimeError("no model may load in this test")

        monkeypatch.setattr(load_mod, "load_target", boom)
        with pytest.raises(RuntimeError, match="no model may load"):
            cli.cmd_generate(["--model", "org/Unregistered", "--mode", "lookup"])

    def test_serve_help_parses(self, capsys):
        with pytest.raises(SystemExit) as e:
            cli.cmd_serve(["--help"])
        assert e.value.code == 0
        assert "--pause-on-battery" in capsys.readouterr().out

    def test_generate_help_parses(self, capsys):
        with pytest.raises(SystemExit) as e:
            cli.cmd_generate(["--help"])
        assert e.value.code == 0
        assert "--pause-on-battery" not in capsys.readouterr().out  # serve-only flag

    def test_serve_rejects_the_option_off_macos(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "platform", "linux")
        with pytest.raises(SystemExit) as e:
            cli.cmd_serve(["--pause-on-battery"])
        assert e.value.code == 2
        assert "macOS-only" in capsys.readouterr().err


class TestStartupPowerDecision:
    """cmd_serve probes the initial power source BEFORE any load; the coordinator
    starts paused and model-less when the Mac is already on battery. All stubbed:
    Engine.load counts (and would fail if the battery path called it), run_server
    records and returns cleanly, the monitor never spawns."""

    @staticmethod
    def _run(monkeypatch, source):
        import mlx_dspark.power as P

        _StubMonitor.source = source
        monkeypatch.setattr(P, "PowerMonitor", _StubMonitor)
        monkeypatch.setattr(P, "current_power_source", lambda: source)
        _StubMonitor.instances = []
        calls = {"loads": 0, "holder": None, "pause": None, "monitor": None}

        def fake_load(**kw):
            calls["loads"] += 1
            return _FakeEngine()

        def fake_run_server(holder, *, pause=None, **kw):
            calls["holder"], calls["pause"] = holder, pause
            assert pause is not None
            assert pause._holder._lifecycle.thread() is not None  # lifecycle worker runs
            assert len(_StubMonitor.instances) == 1
            return None                           # clean, immediate shutdown

        monkeypatch.setattr(S.Engine, "load", staticmethod(fake_load))
        monkeypatch.setattr(S, "run_server", fake_run_server)
        monkeypatch.setattr(S, "maybe_batch_engine", lambda e, b: e)
        cli.cmd_serve(["--pause-on-battery", "--model", "org/T"])
        assert calls["holder"] is not None and calls["pause"] is not None
        # lexical ownership: run_server returned -> the finally stopped both threads
        assert calls["pause"]._holder._lifecycle.thread() is None
        assert _StubMonitor.instances[-1].stopped is True
        return calls

    def test_startup_on_ac_loads_normally(self, monkeypatch):
        calls = self._run(monkeypatch, "ac")
        assert calls["loads"] == 1                 # the normal startup load ran
        assert calls["holder"].current is not None # the model-less start NOT chosen
        assert calls["holder"].current.model_id == "Fake"
        assert calls["pause"].state == "ready" and calls["pause"].health()["admission_open"]

    def test_startup_on_battery_never_loads_and_starts_paused(self, monkeypatch):
        """THE improvement: a serve launched while already on battery starts
        model-less and paused — it must NOT load+warm 16-20+ GB just to immediately
        unload it. The first AC event performs the one real load."""
        calls = self._run(monkeypatch, "battery")
        assert calls["loads"] == 0                 # Engine.load never ran
        assert calls["holder"].current is None     # nothing resident
        assert calls["pause"].state == "paused"
        h = calls["pause"].health()
        assert h["model_loaded"] is False and h["admission_open"] is False

    def test_unknown_initial_power_starts_loaded(self, monkeypatch):
        """Probe failed (no pmset): the safe default is the normal load, with the
        pause armed for the first observed transition."""
        calls = self._run(monkeypatch, None)
        assert calls["loads"] == 1
        assert calls["pause"].state == "ready"
