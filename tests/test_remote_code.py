"""Issue #26: a checkpoint that asks the loader to import its own Python is refused unless
the process opted in (``--trust-remote-code`` / ``MLX_DSPARK_TRUST_REMOTE_CODE=1``)."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from mlx_dspark import load as L


def _write(tmp_path, name, obj):
    (tmp_path / name).write_text(json.dumps(obj))


def _stub_target_loaders(monkeypatch, calls):
    def lm_load(*args, **kwargs):
        calls.append(("mlx_lm", args, kwargs))
        return object(), object()

    def vlm_load(*args, **kwargs):
        calls.append(("mlx_vlm", args, kwargs))
        return object(), object()

    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace(load=lm_load))
    monkeypatch.setitem(sys.modules, "mlx_vlm", SimpleNamespace(load=vlm_load))


def test_clean_checkpoint_has_no_markers(tmp_path):
    _write(tmp_path, "config.json", {"model_type": "qwen3", "quantization": {"bits": 4}})
    _write(tmp_path, "tokenizer_config.json", {"chat_template": "{{ messages }}"})
    assert L.remote_code_markers(str(tmp_path)) == []
    L.refuse_remote_code(str(tmp_path), "clean")                  # no raise


@pytest.mark.parametrize("name, obj, marker", [
    ("config.json", {"model_type": "qwen3", "model_file": "modeling.py"}, "config.json:model_file"),
    ("config.json", {"model_type": "x", "auto_map": {"AutoModel": "m.M"}}, "config.json:auto_map"),
    ("config.json", {"text_config": {"auto_map": {"AutoModel": "m.M"}}},
     "config.json:text_config.auto_map"),
    ("tokenizer_config.json", {"auto_map": {"AutoTokenizer": ["t.T", None]}},
     "tokenizer_config.json:auto_map"),
    ("processor_config.json", {"auto_map": {"AutoProcessor": "p.P"}},
     "processor_config.json:auto_map"),
])
def test_markers_are_found_and_refused(tmp_path, monkeypatch, name, obj, marker):
    _write(tmp_path, name, obj)
    assert L.remote_code_markers(str(tmp_path)) == [marker]
    monkeypatch.setattr(L, "TRUST_REMOTE_CODE", False)
    with pytest.raises(ValueError, match="import its own Python"):
        L.refuse_remote_code(str(tmp_path), "evil/repo")
    monkeypatch.setattr(L, "TRUST_REMOTE_CODE", True)
    L.refuse_remote_code(str(tmp_path), "evil/repo")              # opted in


def test_unreadable_config_is_not_a_marker(tmp_path):
    (tmp_path / "config.json").write_text("{not json")
    assert L.remote_code_markers(str(tmp_path)) == []


@pytest.mark.parametrize("file_name, config_name, config", [
    ("config.json", "config.json", {"model_type": "k2_custom_text",
                                      "model_file": "sentinel.py"}),
    ("tokenizer_config.json", "config.json", {"model_type": "unknown_family"}),
])
def test_load_target_refuses_remote_code_before_loader_or_sentinel(
        tmp_path, monkeypatch, file_name, config_name, config):
    marker = tmp_path / "executed.txt"
    sentinel = tmp_path / "sentinel.py"
    sentinel.write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n")
    _write(tmp_path, config_name, config)
    if file_name == "tokenizer_config.json":
        _write(tmp_path, file_name, {"auto_map": {"AutoTokenizer": "sentinel.Tokenizer"}})

    calls = []
    _stub_target_loaders(monkeypatch, calls)
    monkeypatch.setattr(L, "_resolve", lambda _repo: str(tmp_path))
    monkeypatch.setattr(L, "TRUST_REMOTE_CODE", False)

    with pytest.raises(ValueError, match="import its own Python"):
        L.load_target(str(tmp_path))

    assert calls == []
    assert not marker.exists()


def _prepare_mocked_lm_load(tmp_path, monkeypatch, loader):
    _write(tmp_path, "config.json", {"model_type": "qwen3"})
    monkeypatch.setattr(L, "_resolve", lambda _repo: str(tmp_path))
    monkeypatch.setattr(L, "checkpoint_identity", lambda *_args: None)
    monkeypatch.setattr(L, "Target", lambda model, tokenizer, **_kwargs: SimpleNamespace())
    monkeypatch.setattr(L, "TRUST_REMOTE_CODE", True)
    monkeypatch.setitem(sys.modules, "mlx_lm", SimpleNamespace(load=loader))


def test_load_target_legacy_mlx_lm_signature_keeps_tokenizer_trust_without_direct_kw(
        tmp_path, monkeypatch):
    calls = []

    def lm_load(path, *, tokenizer_config):
        calls.append((path, tokenizer_config))
        return object(), object()

    _prepare_mocked_lm_load(tmp_path, monkeypatch, lm_load)
    L.load_target(str(tmp_path))

    assert len(calls) == 1
    assert calls[0][1] == {"trust_remote_code": True}


def test_load_target_new_mlx_lm_signature_receives_direct_trust_kw(tmp_path, monkeypatch):
    calls = []

    def lm_load(path, *, tokenizer_config, trust_remote_code=False):
        calls.append((path, tokenizer_config, trust_remote_code))
        return object(), object()

    _prepare_mocked_lm_load(tmp_path, monkeypatch, lm_load)
    L.load_target(str(tmp_path))

    assert calls == [(str(tmp_path), {"trust_remote_code": True}, True)]


@pytest.mark.parametrize("error", [TypeError, ValueError])
def test_load_target_signature_inspection_failure_uses_legacy_call_and_refusal_stays_first(
        tmp_path, monkeypatch, error):
    import inspect

    calls = []

    def lm_load(path, *, tokenizer_config):
        calls.append((path, tokenizer_config))
        return object(), object()

    _prepare_mocked_lm_load(tmp_path, monkeypatch, lm_load)
    original_signature = inspect.signature

    def signature(callable_obj):
        if callable_obj is lm_load:
            raise error("signature unavailable")
        return original_signature(callable_obj)

    monkeypatch.setattr(inspect, "signature", signature)
    L.load_target(str(tmp_path))
    assert calls == [(str(tmp_path), {"trust_remote_code": True})]

    _write(tmp_path, "config.json", {"model_type": "qwen3", "model_file": "sentinel.py"})
    monkeypatch.setattr(L, "TRUST_REMOTE_CODE", False)
    calls.clear()
    with pytest.raises(ValueError, match="import its own Python"):
        L.load_target(str(tmp_path))
    assert calls == []


def test_checkpoint_identity_changes_when_weights_are_replaced(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"weights-a")

    first = L.checkpoint_identity(str(tmp_path))
    assert first is not None

    weights.write_bytes(b"weights-b-replaced")
    second = L.checkpoint_identity(str(tmp_path))
    assert second is not None and second != first

    unverifiable = tmp_path / "not-a-checkpoint"
    unverifiable.mkdir()
    (unverifiable / "config.json").write_text("{}")
    assert L.checkpoint_identity(str(unverifiable)) is None
