"""Model-free contracts for the gated Series-B runner."""
from __future__ import annotations

import builtins
import copy
import importlib.util
import json
import shutil
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


REPO = Path(__file__).resolve().parents[1]
FEATURE = REPO / "specs/002-qwen-bonsai-hybrid-target"
SERIES_B = FEATURE / "evidence/series-b"
GATE = FEATURE / "evidence/gate-b.json"
_SPEC = importlib.util.spec_from_file_location(
    "series_b_model_free_test", FEATURE / "evidence/probes/series_b.py")
assert _SPEC is not None and _SPEC.loader is not None
series_b = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(series_b)


def _write_gate(tmp: Path, mutate=None) -> Path:
    data = json.loads(GATE.read_text())
    if mutate:
        mutate(data)
    path = tmp / "gate-b.json"
    path.write_text(json.dumps(data))
    return path


def _attempt_resolution(path: Path) -> None:
    imported: list[str] = []
    original = builtins.__import__

    def watch(name, *args, **kwargs):
        if name.startswith("huggingface_hub"):
            imported.append(name)
        return original(name, *args, **kwargs)

    with patch("builtins.__import__", watch):
        series_b.resolve_checkpoint_provenance(path)
    assert not imported


@pytest.mark.parametrize("case", ["missing", "closed", "malformed", "stale"])
def test_gate_rejects_before_checkpoint_resolver(case: str, tmp_path: Path) -> None:
    if case == "missing":
        path = tmp_path / "missing.json"
    elif case == "closed":
        path = _write_gate(tmp_path, lambda g: g.update(
            decision="CLOSED", status="CLOSED", final_status="CLOSED"))
    elif case == "malformed":
        path = tmp_path / "malformed.json"
        path.write_text("{")
    else:
        def stale(g):
            key = next(iter(g["closure_checks"]))
            g["closure_checks"][key]["passed"] = not g["closure_checks"][key]["passed"]
        path = _write_gate(tmp_path, stale)
    with pytest.raises(RuntimeError, match="Series B denied"):
        _attempt_resolution(path)


@pytest.mark.parametrize("action,args", [
    (series_b.resolve_checkpoint_provenance, ()),
    (series_b.fetch_verified_bb_weights, ()),
    (series_b.drafter_compatibility_smoke, ()),
    (series_b.build_runner_snapshot, ()),
    (series_b.run_condition, ("H0", "B-Q", "no-run", 1)),
])
def test_entry_guard_cannot_be_bypassed(action, args) -> None:
    sentinel = RuntimeError("guard sentinel")
    real_spec = importlib.util.spec_from_file_location
    real_import = builtins.__import__
    loaded_series_a = []
    forbidden_imports = []

    def watch_module(name, path, *a, **kw):
        if Path(path).name == "series_a.py":
            loaded_series_a.append(path)
        return real_spec(name, path, *a, **kw)

    def watch_import(name, *a, **kw):
        if name.startswith(("huggingface_hub", "mlx", "mlx_dspark")):
            forbidden_imports.append(name)
        return real_import(name, *a, **kw)

    with patch.object(series_b, "require_entry", side_effect=sentinel), \
            patch.object(series_b.importlib.util, "spec_from_file_location", watch_module), \
            patch("builtins.__import__", watch_import):
        with pytest.raises(RuntimeError, match="guard sentinel"):
            action(*args)
    assert loaded_series_a == []
    assert forbidden_imports == []


def test_matrix_generation_is_exactly_ten_cells_and_excludes_exception() -> None:
    matrix, controls = series_b.build_frozen_matrix_and_controls(write=False)
    assert matrix["status"] == controls["status"] == "PASS"
    assert matrix["targets"] == ["H0", "H1c", "H2", "H3", "B0"]
    assert matrix["drafters"] == ["B-Q", "B-B"]
    assert len(matrix["cells"]) == 10
    assert {(c["target"], c["drafter"]) for c in matrix["cells"]} == {
        (target, drafter) for target in matrix["targets"] for drafter in matrix["drafters"]
    }
    assert matrix["isolated_block_exception"]["authorized"] is False
    assert matrix["isolated_block_exception"]["excluded_targets"] == ["H1a", "H1b"]


def test_h1a_h1b_exception_is_rejected_from_current_series_a_evidence() -> None:
    decision = series_b.series_a_decision()
    assert decision["decision"]["h1c_vs_h1a"] == "confirmed_win"
    assert decision["decision"]["h1c_vs_h1b"] == "within_noise"
    assert decision["decision"]["isolated_block_exception_authorized"] is False
    assert {cell["target"] for cell in decision["cells"]}.isdisjoint({"H1a", "H1b"})


def test_paired_target_identity_and_immutable_controls_are_equal() -> None:
    matrix = json.loads((SERIES_B / "matrix.json").read_text())
    controls = json.loads((SERIES_B / "control-manifest.json").read_text())
    assert len(matrix["cells"]) == 10
    assert set(controls["paired_controls"]) == set(matrix["targets"])
    for target, pair in controls["paired_controls"].items():
        assert len(pair["cells"]) == 2
        cells = [c for c in matrix["cells"] if c["target"] == target]
        assert len(cells) == 2
        assert cells[0]["target_identity_sha256"] == cells[1]["target_identity_sha256"]
        assert pair["target_identity_sha256"] == cells[0]["target_identity_sha256"]
        assert pair["only_allowed_differences"] == ["exact drafter checkpoint"]
        assert pair["compatibility_adaptation"] == "none"
    assert controls["common_controls"]["plain_kv"] is True
    assert controls["common_controls"]["kv_bits"] is None
    assert controls["common_controls"]["width_policy"] is False
    assert controls["allowed_pair_differences"] == ["exact drafter checkpoint"]


def test_pair_schedule_is_adjacent_and_arm_order_alternates() -> None:
    schedule = json.loads((SERIES_B / "matrix.json").read_text())["schedule"]
    rows = schedule["rows"]
    assert schedule["adjacent_pairs"] is True
    assert len(rows) == 10
    for offset in range(0, 10, 2):
        left, right = rows[offset:offset + 2]
        assert left["target"] == right["target"]
        assert {left["drafter"], right["drafter"]} == {"B-Q", "B-B"}
    orders = [rows[i]["drafter"] for i in range(0, 10, 2)]
    assert all(a != b for a, b in zip(orders, orders[1:]))


def _valid_result() -> dict:
    return {
        "schema": "qwen-bonsai-series-b-run/v1", "series": "B", "status": "PASS",
        "variant": "H0", "drafter_arm": "B-Q",
        "checkpoints": {"dflash": {"repo_id": series_b.BQ_REPO,
                                      "revision": series_b.BQ_REVISION,
                                      "weights": [{"sha256": "a" * 64}]}},
        "integrity": {"target_integrity_passed": True,
                      "all_generated_outputs_nonempty": True,
                      "serial_and_spec_prompt_ids_match": True},
        "measured": {
            "serial_target": {"tokens_per_sec": 30.0, "decode_seconds": 2.0,
                              "target_forwards": 10, "generated_tokens": 20},
            "speculative": {"decode_seconds": 1.0, "decode_tokens_per_sec": 40.0,
                            "speedup_over_serial": 4 / 3, "draft_acceptance": 0.6,
                            "mean_accept_len": 2.0, "target_forwards": 10,
                            "generated_tokens": 20,
                            "generated_tokens_per_target_forward": 2.0,
                            "accept_lengths": [2.0], "accept_histogram": {"2": 1},
                            "draft_width_distribution": {"2": 2, "4": 8},
                            "cap_distribution": {"2": 2, "4": 8},
                            "per_prompt": [{"decode_tokens_per_sec": 40.0,
                                            "generated_token_count": 20,
                                            "decode_seconds": 1.0,
                                            "prompt_ids_sha256": "a" * 64}]},
            "peak_steady_state_memory": {"peak_bytes_after_load_warmup_reset": 1000},
            "component_timing": None,
            "component_timing_unavailable_reason": "not collected by the accepted runner",
            "decode_duration_seconds": 3.0,
            "generated_token_count": 20, "measurement_duration_seconds": 3.0,
        },
    }


def test_result_schema_metrics_and_checkpoint_provenance_are_required() -> None:
    result = _valid_result()
    series_b.validate_result_integrity(result)
    missing_metric = copy.deepcopy(result)
    del missing_metric["measured"]["speculative"]["draft_acceptance"]
    with pytest.raises(ValueError, match="missing required metrics"):
        series_b.validate_result_integrity(missing_metric)
    missing_provenance = copy.deepcopy(result)
    missing_provenance["checkpoints"]["dflash"]["weights"] = []
    with pytest.raises(ValueError, match="immutable drafter checkpoint provenance"):
        series_b.validate_result_integrity(missing_provenance)


def test_integrity_failure_is_rejected() -> None:
    result = _valid_result()
    result["integrity"]["target_integrity_passed"] = False
    with pytest.raises(ValueError, match="failed a required integrity check"):
        series_b.validate_result_integrity(result)


@pytest.mark.parametrize("validation,checkpoints,smoke", [
    ({}, {"status": "PASS"}, {"status": "PASS"}),
    ({"status": "PASS", "authorization_for_t038": True}, {"status": "FAIL"}, {"status": "PASS"}),
    ({"status": "PASS", "authorization_for_t038": True}, {"status": "PASS"}, {"status": "FAIL"}),
])
def test_t038_requires_t032_t033_and_t037_pass(validation, checkpoints, smoke) -> None:
    with pytest.raises(RuntimeError, match="T038 execution denied"):
        series_b.validate_t038_prerequisites(validation, checkpoints, smoke)


def test_series_b_checkpoint_provenance_is_separate_from_series_a() -> None:
    assert (SERIES_B / "checkpoints.json").resolve() != (FEATURE / "evidence/checkpoints.json").resolve()
    b_record = json.loads((SERIES_B / "checkpoints.json").read_text())
    a_record = json.loads((FEATURE / "evidence/checkpoints.json").read_text())
    assert b_record["B-Q"]["repo_id"] == a_record["checkpoints"]["dflash"]["repo_id"]
    assert b_record["B-Q"]["resolved_revision"] == a_record["checkpoints"]["dflash"]["revision"]


def test_t037_does_not_accept_caller_supplied_test_result() -> None:
    assert not inspect.signature(series_b.build_runner_validation).parameters
    with pytest.raises(TypeError):
        series_b.build_runner_validation(test_count=19, test_result="19 passed")


def test_actual_pytest_nonzero_exit_denies_t038_authorization(monkeypatch) -> None:
    monkeypatch.setattr(series_b, "require_entry", lambda: {"passed": True, "checks": {}})
    monkeypatch.setattr(series_b.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=1, stdout="1 failed", stderr=""))
    monkeypatch.setattr(series_b, "atomic_json", lambda *a, **k: None)
    validation = series_b.build_runner_validation()
    assert validation["status"] == "DENY"
    assert validation["authorization_for_t038"] is False
    assert validation["model_free_test_evidence"]["exit_code"] == 1
    assert validation["checks"]["model_free_tests_pass"]["passed"] is False


def _authorization_tree(tmp_path: Path) -> tuple[Path, dict]:
    root = tmp_path / "repo"
    relative_paths = set(series_b.AUTHORIZATION_ARTIFACTS)
    relative_paths.update({
        "specs/002-qwen-bonsai-hybrid-target/evidence/series-a/discovery-index.json",
        "specs/002-qwen-bonsai-hybrid-target/evidence/probes/series_a.py",
    })
    discovery = json.loads((FEATURE / "evidence/series-a/discovery-index.json").read_text())
    relative_paths.update(discovery["runtime_revision"]["source_sha256"])
    for rel in relative_paths:
        source = REPO / rel
        destination = root / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    return root, series_b.current_authorization_fingerprints(root)


@pytest.mark.parametrize("artifact", series_b.AUTHORIZATION_ARTIFACTS)
def test_changed_authorization_artifact_denies_t038_before_runner_import(
        artifact: str, tmp_path: Path, monkeypatch) -> None:
    root, fingerprint_set = _authorization_tree(tmp_path)
    series_b.atomic_json(root / "specs/002-qwen-bonsai-hybrid-target/evidence/series-b/runner-validation.json", {
        "status": "PASS", "authorization_for_t038": True,
        "authorization_fingerprint_set": fingerprint_set,
    })
    changed = root / artifact
    changed.write_bytes(changed.read_bytes() + b"\nchanged\n")
    monkeypatch.setattr(series_b, "REPO", root)
    monkeypatch.setattr(series_b, "EVIDENCE", root / "specs/002-qwen-bonsai-hybrid-target/evidence")
    monkeypatch.setattr(series_b, "SERIES_B", root / "specs/002-qwen-bonsai-hybrid-target/evidence/series-b")
    monkeypatch.setattr(series_b, "require_entry", lambda *a, **k: {"passed": True})
    loaded = []
    real_spec = importlib.util.spec_from_file_location

    def watch(name, path, *args, **kwargs):
        if Path(path).name == "series_a.py":
            loaded.append(path)
        return real_spec(name, path, *args, **kwargs)

    monkeypatch.setattr(series_b.importlib.util, "spec_from_file_location", watch)
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        series_b.run_condition("H0", "B-Q", "negative", 1)
    assert loaded == []


def _raw_accepted_h0() -> dict:
    path = FEATURE / "evidence/series-a/runs/discovery-01-h0-95426c/result.json"
    return json.loads(path.read_text())


def _frozen_matrix_and_controls() -> tuple[dict, dict]:
    return (json.loads((SERIES_B / "matrix.json").read_text()),
            json.loads((SERIES_B / "control-manifest.json").read_text()))


def test_requested_drafter_provenance_is_bound_both_directions() -> None:
    matrix, controls = _frozen_matrix_and_controls()
    staged = _raw_accepted_h0()
    bb_cell = next(c for c in matrix["cells"] if c["cell_id"] == "H0-B-B")
    with pytest.raises(ValueError, match="DFlash provenance"):
        series_b.validate_requested_cell(copy.deepcopy(staged), "H0", "B-B", bb_cell, matrix, controls)

    bq_cell = next(c for c in matrix["cells"] if c["cell_id"] == "H0-B-Q")
    staged_bb = copy.deepcopy(staged)
    bb = json.loads((SERIES_B / "checkpoints.json").read_text())["B-B"]
    staged_bb["checkpoints"]["dflash"] = {
        "repo_id": bb["repo_id"], "revision": bb["resolved_revision"],
        "config_sha256": bb["config"]["sha256"],
        "weights": bb["verified_downloaded_weight_files"],
    }
    with pytest.raises(ValueError, match="DFlash provenance"):
        series_b.validate_requested_cell(staged_bb, "H0", "B-Q", bq_cell, matrix, controls)


def test_requested_target_and_ownership_provenance_are_bound() -> None:
    matrix, controls = _frozen_matrix_and_controls()
    h0 = next(c for c in matrix["cells"] if c["cell_id"] == "H0-B-Q")
    staged = _raw_accepted_h0()
    staged["variant"] = "H3"
    with pytest.raises(ValueError, match="variant"):
        series_b.validate_requested_cell(staged, "H0", "B-Q", h0, matrix, controls)
    staged = _raw_accepted_h0()
    staged["target_composition"]["embedding_owner"] = "bonsai"
    with pytest.raises(ValueError, match="ownership mismatch"):
        series_b.validate_requested_cell(staged, "H0", "B-Q", h0, matrix, controls)


def test_required_series_a_metric_contract_rejects_memory_width_and_nonfinite() -> None:
    result = _valid_result()
    del result["measured"]["peak_steady_state_memory"]
    with pytest.raises(ValueError, match="peak_steady_state_memory_bytes"):
        series_b.validate_result_integrity(result)
    result = _valid_result()
    del result["measured"]["speculative"]["cap_distribution"]
    with pytest.raises(ValueError, match="cap_distribution"):
        series_b.validate_result_integrity(result)
    result = _valid_result()
    del result["measured"]["speculative"]["draft_width_distribution"]
    with pytest.raises(ValueError, match="draft_width_distribution"):
        series_b.validate_result_integrity(result)
    result = _valid_result()
    result["measured"]["speculative"]["decode_tokens_per_sec"] = float("nan")
    with pytest.raises(ValueError, match="not finite"):
        series_b.validate_result_integrity(result)


@pytest.mark.parametrize("order,target,drafter", [(2, "H0", "B-Q"), (1, "H0", "B-B")])
def test_discovery_requests_must_match_frozen_schedule(order: int, target: str,
                                                       drafter: str) -> None:
    matrix = json.loads((SERIES_B / "matrix.json").read_text())
    with pytest.raises(RuntimeError, match="frozen schedule row"):
        series_b.validate_discovery_schedule(matrix, order, target, drafter, "discovery")


def test_matrix_decision_rejects_changed_referenced_series_a_hash() -> None:
    decision = json.loads((SERIES_B / "matrix-decision.json").read_text())
    matrix = json.loads((SERIES_B / "matrix.json").read_text())
    decision["series_a"]["adjudication"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="matrix-decision"):
        series_b.validate_matrix_decision(decision, matrix)
