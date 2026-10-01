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


class TestDonorBlockSelection:
    @pytest.mark.parametrize("raw, expected", [
        ("62", (62,)),
        ("56-63", (56, 57, 58, 59, 60, 61, 62, 63)),
        ("12,28,41,55,62", (12, 28, 41, 55, 62)),
        ("12,20-23,62", (12, 20, 21, 22, 23, 62)),
        ("62,62,60-62", (60, 61, 62)),
        ("63,0,63", (0, 63)),
    ])
    def test_parser_expands_and_normalizes(self, raw, expected):
        assert cli._parse_donor_blocks(raw) == expected

    @pytest.mark.parametrize("raw", [
        "", " ", ",62", "62,", "62,,63", "-1", "64", "1.0", "+1", "one",
        "1-", "-2", "1--2", "2-1", "1-64", "1-2-3", "1,foo", " 1 ",
    ])
    def test_parser_rejects_invalid_syntax(self, raw):
        with pytest.raises(ValueError):
            cli._parse_donor_blocks(raw)


class TestDonorServeDispatch:
    @staticmethod
    def _stub_server(monkeypatch):
        calls = {"loads": [], "resolver": [], "server": 0}

        def fake_load(**kwargs):
            calls["loads"].append(kwargs)
            return _FakeEngine()

        def fake_run_server(*args, **kwargs):
            calls["server"] += 1

        monkeypatch.setattr(S.Engine, "load", staticmethod(fake_load))
        monkeypatch.setattr(S, "run_server", fake_run_server)
        monkeypatch.setattr(S, "maybe_batch_engine", lambda engine, batch: engine)
        return calls

    @pytest.mark.parametrize("args", [
        ["--donor-model", "org/donor"],
        ["--donor-blocks", "62"],
        ["--donor-model", "org/donor", "--donor-blocks", "2-1"],
        ["--donor-model", "org/donor", "--donor-blocks", "64"],
        ["--donor-model", "org/donor", "--donor-blocks", "62,,63"],
    ])
    def test_invalid_pair_or_blocks_fail_before_resolution_and_load(self, monkeypatch, args):
        import mlx_dspark.load as load_mod

        calls = self._stub_server(monkeypatch)

        def forbidden_resolve(*args, **kwargs):
            pytest.fail("model resolution must not happen for invalid donor options")

        monkeypatch.setattr(load_mod, "_resolve", forbidden_resolve)
        with pytest.raises(SystemExit) as exc:
            cli.cmd_serve(["--mode", "lookup", "--model", "org/qwen", *args])
        assert exc.value.code == 2
        assert calls["loads"] == []

    def test_no_donor_keeps_existing_load_kwargs_without_composition(self, monkeypatch):
        calls = self._stub_server(monkeypatch)
        cli.cmd_serve(["--mode", "lookup", "--model", "org/qwen", "--drafter", "org/draft"])
        kwargs = calls["loads"][0]
        assert kwargs == {
            "mode": "lookup", "model": "org/qwen", "drafter": "org/draft",
            "family": None, "target": None, "drafter_bits": 4, "max_draft_tokens": None,
            "confidence_threshold": 0.0, "enable_thinking": None, "reasoning_effort": None,
            "prefix_cache": True, "prefix_cache_dir": None, "prefix_cache_max_ram_mb": 0,
            "default_max_tokens": 2048, "max_tokens_cap": 32768,
            "default_temperature": None, "default_top_p": None, "default_top_k": None,
            "prefix_cache_slots": 2, "prefix_cache_rungs": 8192, "lookup_drafts": None,
            "lookup_long_draft": 32, "wired_limit": False, "wide_gemm_min": None,
            "cpu_split": None, "small_m": None, "sdpa_split": None, "warmup": True,
            "memory_guard": True, "batch_widths": None, "kv_bits": None,
            "context_window": None,
        }
        assert calls["resolver"] == []
        assert calls["server"] == 1

    def test_donor_composition_reaches_engine_and_preserves_drafter(self, monkeypatch, capsys):
        import mlx_dspark.load as load_mod
        from mlx_dspark.hybrid_target import TargetCompositionRequest

        calls = self._stub_server(monkeypatch)
        target_sha, donor_sha = "1" * 40, "2" * 40
        roots = {
            "org/qwen": f"/cache/models--org--qwen/snapshots/{target_sha}",
            "org/donor": f"/cache/models--org--donor/snapshots/{donor_sha}",
        }

        def fake_resolve(repo):
            calls["resolver"].append(repo)
            return roots[repo]

        monkeypatch.setattr(load_mod, "_resolve", fake_resolve)
        cli.cmd_serve(["--mode", "dflash", "--model", "org/qwen", "--drafter", "org/draft",
                       "--donor-model", "org/donor", "--donor-blocks", "56-63"])
        kwargs = calls["loads"][0]
        request = kwargs["target_composition"]
        assert isinstance(request, TargetCompositionRequest)
        assert request.donor_indices == (56, 57, 58, 59, 60, 61, 62, 63)
        assert request.donor_repo == "org/donor" and request.donor_revision == donor_sha
        assert request.qwen_repo == "org/qwen" and request.qwen_revision == target_sha
        assert kwargs["mode"] == "dflash" and kwargs["drafter"] == "org/draft"
        assert calls["resolver"] == ["org/qwen", "org/donor"]
        output = capsys.readouterr().out
        assert "org/donor" in output and donor_sha in output and "56-63" in output
        assert "Qwen" in output and "embedding" in output and "final norm" in output
        assert "LM head" in output and "unselected decoder blocks" in output

    def test_non_immutable_local_donor_fails_before_engine_load(self, monkeypatch):
        import mlx_dspark.load as load_mod

        calls = self._stub_server(monkeypatch)
        monkeypatch.setattr(load_mod, "_resolve", lambda repo: "/models/ordinary-local-checkpoint")
        with pytest.raises(SystemExit) as exc:
            cli.cmd_serve(["--mode", "lookup", "--model", "org/qwen", "--donor-model",
                           "/models/donor", "--donor-blocks", "62"])
        assert exc.value.code == 2
        assert calls["loads"] == []


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
