#!/usr/bin/env python3
"""Model-load-free Series-A Gate-B validator and Series-B entry guard.

This module deliberately imports only the Python standard library.  It never
imports Prism, MLX, or a model loader; Series-B callers must invoke
``require_series_b_entry`` before importing/loading any checkpoint.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
FEATURE = REPO / "specs/002-qwen-bonsai-hybrid-target"
EVIDENCE = FEATURE / "evidence"
GATE = EVIDENCE / "gate-b.json"
SPEC = FEATURE / "spec.md"
CONSTITUTION = REPO / ".specify/memory/constitution.md"
MEASURED_HEAD = "df8ff57d47b9b0a95c2d432cb0c030a2d4e64472"
EXPECTED_REVISIONS = {
    "qwen": ("mlx-community/Qwen3.8-27B-4bit", "10c35caafbb80f7dc6a7a432cdd11af10a6d4818"),
    "bonsai": ("prism-ml/Ternary-Bonsai-2-27B-mlx-2bit", "fcba37d2117a7077eac6b613b2668d14d9779edd"),
    "dflash": ("incoai/Qwen3.8-27B-DFlash2", "015e795645c74b1a0eeef3b570031fb62e769bc5"),
}
TARGETS = ["H0", "H1a", "H1b", "H1c", "H2", "H3", "B0"]
DISCOVERY_ORDER = ["H0", "H1a", "H2", "H1b", "H3", "H1c", "B0", "H0"]
TAPS = [5, 19, 33, 47, 61]
TAP_COUNTS = {"H0": 0, "H1a": 0, "H1b": 0, "H1c": 0, "H2": 1, "H3": 1, "B0": 5}
COMPARISONS = {
    "H1a-vs-H0", "H1b-vs-H0", "H1c-vs-H0", "H2-vs-H0", "H3-vs-H0", "B0-vs-H0",
    "H1c-vs-H1a", "H1c-vs-H1b",
}
METHOD_SHA256 = "688901863e6660ecef749c310d9f69ff8cfc7cf37db0137716ae1058bc1ff6d4"
GOVERNING_SPEC_SHA256 = "8f844a4cfee6762afcd614a053bd2b20163469e61c01284172fbfb6b8bfaac59"
GOVERNING_CONSTITUTION_SHA256 = "217147f1de1e77c65f9131de250a52ba36979e157b02e0f835b1fa27c227c6c2"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return obj


def rel(path: Path) -> str:
    return path.resolve().relative_to(REPO).as_posix()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def current_runtime_hashes() -> dict[str, str]:
    idx = read_json(EVIDENCE / "series-a/discovery-index.json")
    expected = idx["runtime_revision"]["source_sha256"]
    return {name: sha256_file(REPO / name) for name in expected}


def benchmark_method_hash(source: bytes) -> str:
    """Hash the AST for run_series_condition, matching the recorded method hash."""
    tree = ast.parse(source)
    fn = next((n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
               and n.name == "run_series_condition"), None)
    if fn is None:
        raise ValueError("run_series_condition not found")
    return sha256_bytes(ast.get_source_segment(source.decode(), fn).encode())


def ref(path: Path) -> dict[str, str]:
    return {"path": rel(path), "sha256": sha256_file(path)}


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _close(a: Any, b: Any, *, tolerance: float = 1e-9) -> bool:
    return is_number(a) and is_number(b) and math.isclose(float(a), float(b), rel_tol=0, abs_tol=tolerance)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically replace a JSON document in its existing directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def minimal_closed_gate() -> dict[str, Any]:
    """A valid denial record that needs no evidence reads to construct."""
    return {
        "schema": "qwen-bonsai-series-a-gate-b/v1", "version": 1,
        "decision": "CLOSED", "status": "CLOSED", "final_status": "CLOSED",
        "audit": {"validator": "evidence/gate_b.py", "validator_sha256": "",
                  "initial_status": "CLOSED", "model_load_performed": False,
                  "series_b_measurement_performed": False, "validated": False,
                  "validation_passed": False},
        "governing_inputs": {}, "runtime": {}, "checkpoint_provenance": {},
        "required_targets": [], "discovery_order": [], "dflash_taps": [],
        "direct_bonsai_tap_counts": {}, "references": {}, "closure_checks": {},
    }


def evidence_refs() -> dict[str, Any]:
    idx = read_json(EVIDENCE / "integrity/index.json")
    discovery = read_json(EVIDENCE / "series-a/discovery-index.json")
    schedule = read_json(EVIDENCE / "series-a/confirmation-schedule.json")
    return {
        "checkpoints": ref(EVIDENCE / "checkpoints.json"),
        "integrity_index": ref(EVIDENCE / "integrity/index.json"),
        "official_bonsai_loader": ref(EVIDENCE / "integrity/official-bonsai-load.json"),
        "integrity_targets": {t: ref(REPO / idx["targets"][t]["path"]) for t in TARGETS},
        "runtime_semantics": ref(EVIDENCE / "integrity/runtime-semantics.json"),
        "donor_memory": ref(EVIDENCE / "integrity/donor-memory-release.json"),
        "integrity_repeatability": ref(REPO / idx["repeatability"]["path"]),
        "fresh_h0": ref(EVIDENCE / "integrity/h0-fresh.json"),
        "discovery_index": ref(EVIDENCE / "series-a/discovery-index.json"),
        "discovery_runs": [ref(REPO / r["path"]) for r in discovery["targets"]],
        "discovery_order": ref(REPO / discovery["discovery_order"]["path"]),
        "discovery_runner_snapshot": ref(REPO / discovery["discovery_runner_snapshot"]["path"]),
        "prompt_corpus": ref(REPO / discovery["prompt_corpus"]["path"]),
        "repeat_selection": ref(EVIDENCE / "series-a/repeat-selection.json"),
        "confirmation_schedule": ref(EVIDENCE / "series-a/confirmation-schedule.json"),
        "confirmation_runs": [ref(REPO / run["path"])
                               for pair in schedule["matched_pairs"] for run in pair["runs"]],
        "repeat_decisions": ref(EVIDENCE / "series-a/repeat-decisions.json"),
        "adjudication": ref(EVIDENCE / "adjudication/series-a.json"),
    }


def initial_gate() -> dict[str, Any]:
    checkpoints = read_json(EVIDENCE / "checkpoints.json")
    discovery = read_json(EVIDENCE / "series-a/discovery-index.json")
    runtime = read_json(EVIDENCE / "integrity/runtime-semantics.json")
    spec_bytes, const_bytes = SPEC.read_bytes(), CONSTITUTION.read_bytes()
    current_hashes = current_runtime_hashes()
    return {
        "schema": "qwen-bonsai-series-a-gate-b/v1",
        "version": 1,
        "decision": "CLOSED",
        "audit": {"validator": "evidence/gate_b.py", "validator_sha256": sha256_file(Path(__file__).resolve()),
                  "initial_status": "CLOSED", "model_load_performed": False,
                  "series_b_measurement_performed": False},
        "governing_inputs": {
            "spec": {"path": rel(SPEC), "sha256": sha256_bytes(spec_bytes)},
            "constitution": {"path": rel(CONSTITUTION), "version": "2.0.0",
                             "sha256": sha256_bytes(const_bytes)},
        },
        "runtime": {
            "measured_git_head": discovery["runtime_revision"]["git_head"],
            "current_git_head": git_head(),
            "measured_source_sha256": discovery["runtime_revision"]["source_sha256"],
            "current_source_sha256": current_hashes,
            "benchmark_method_sha256": METHOD_SHA256,
        },
        "checkpoint_provenance": {
            name: {"repo_id": checkpoints["checkpoints"][name]["repo_id"],
                   "revision": checkpoints["checkpoints"][name]["revision"]}
            for name in EXPECTED_REVISIONS
        },
        "required_targets": TARGETS,
        "discovery_order": DISCOVERY_ORDER,
        "dflash_taps": runtime["loaded_dflash"]["loaded_target_layer_ids"],
        "direct_bonsai_tap_counts": runtime["loaded_dflash"]["direct_bonsai_tap_counts"],
        "references": evidence_refs(),
        "closure_checks": {},
        "status": "CLOSED",
        "final_status": "CLOSED",
    }


def _resolve_ref(r: dict[str, str], root: Path) -> Path:
    p = Path(r["path"])
    if p.is_absolute():
        # Accept only paths under this checkout; legacy evidence stores absolute local paths.
        p = p.resolve().relative_to(REPO)
    full = (root / p).resolve()
    full.relative_to(root.resolve())
    return full


def metric_contract(run: dict[str, Any]) -> tuple[bool, str]:
    """Check the required recorded measurements without imposing new thresholds."""
    m = run.get("measured", {})
    serial, spec, memory = m.get("serial_target", {}), m.get("speculative", {}), m.get("peak_steady_state_memory", {})
    positive = [m.get("decode_duration_seconds"), m.get("measurement_duration_seconds"),
                serial.get("decode_seconds"), serial.get("tokens_per_sec"),
                spec.get("decode_seconds"), spec.get("decode_tokens_per_sec"),
                spec.get("speedup_over_serial"), spec.get("generated_tokens_per_target_forward"),
                spec.get("mean_accept_len"), memory.get("peak_bytes_after_load_warmup_reset")]
    if not all(is_number(x) and x > 0 for x in positive):
        return False, "required duration, rate, speedup, mean, or peak-memory metric missing/non-positive/non-finite"
    if not is_number(spec.get("draft_acceptance")) or not 0 <= spec["draft_acceptance"] <= 1:
        return False, "draft acceptance missing or outside [0,1]"
    for key, owner in (("target_forwards", serial), ("generated_tokens", serial),
                       ("target_forwards", spec), ("generated_tokens", spec),
                       ("generated_token_count", m)):
        value = owner.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            return False, f"{key} missing or not a positive integer"
    if spec["generated_tokens"] != m["generated_token_count"]:
        return False, "speculative generated token count disagrees with aggregate generated_token_count"
    expected_speedup = spec["decode_tokens_per_sec"] / serial["tokens_per_sec"]
    if not _close(spec["speedup_over_serial"], expected_speedup, tolerance=1e-6):
        return False, "speculative speedup does not agree with speculative/serial rates"
    if not _close(spec["generated_tokens_per_target_forward"],
                  spec["generated_tokens"] / spec["target_forwards"], tolerance=1e-6):
        return False, "generated tokens per target forward does not agree with counts"
    accept_lengths = spec.get("accept_lengths")
    if (not isinstance(accept_lengths, list) or not accept_lengths or
            any(not is_number(x) or x < 0 or x > 8 for x in accept_lengths) or
            not is_number(spec.get("mean_accept_len")) or not 0 <= spec["mean_accept_len"] <= 8):
        return False, "draft accept lengths/mean missing or outside recorded acceptance range"
    for key in ("accept_histogram", "draft_width_distribution", "cap_distribution"):
        distribution = spec.get(key)
        if not isinstance(distribution, dict) or not distribution:
            return False, f"required {key} missing or empty"
        if any(not is_number(v) or v < 0 for v in distribution.values()):
            return False, f"{key} has non-numeric, negative, or non-finite counts"
    per_prompt = spec.get("per_prompt")
    if not isinstance(per_prompt, list) or not per_prompt:
        return False, "speculative per-prompt measurements missing"
    for p in per_prompt:
        if (not is_number(p.get("decode_tokens_per_sec")) or p["decode_tokens_per_sec"] <= 0
                or not isinstance(p.get("generated_token_count"), int) or p["generated_token_count"] <= 0
                or not is_number(p.get("decode_seconds")) or p["decode_seconds"] <= 0
                or not isinstance(p.get("prompt_ids_sha256"), str)):
            return False, "speculative per-prompt required metric missing/invalid"
    return True, "required Series-A metrics are present, finite, positive/sane, and internally consistent"


def _prompt_contract(run: dict[str, Any], canonical_prompts: list[dict[str, Any]], hashes: list[str]) -> bool:
    controls = run.get("controls", {})
    prompts = controls.get("prompt_ids")
    if prompts != canonical_prompts:
        return False
    recalculated = []
    for p in prompts:
        ids = p.get("input_ids")
        if not isinstance(ids, list) or not ids or any(not isinstance(x, int) or isinstance(x, bool) for x in ids):
            return False
        digest = sha256_bytes(json.dumps(ids, separators=(",", ":")).encode())
        if p.get("input_ids_sha256") != digest:
            return False
        recalculated.append(digest)
    if recalculated != hashes:
        return False
    for metric in (run.get("measured", {}).get("serial_target", {}),
                   run.get("measured", {}).get("speculative", {})):
        per_prompt = metric.get("per_prompt", [])
        if [p.get("prompt_ids_sha256") for p in per_prompt] != hashes:
            return False
    return True


def _matched_run_contract(run: dict[str, Any], variant: str, fixed: dict[str, Any],
                          prompt_ids: list[dict[str, Any]], prompt_hashes: list[str],
                          prompt_corpus: dict[str, Any], measurement_protocol: str,
                          checkpoints: dict[str, tuple[str, str]], measured_head: str,
                          measured_sources: dict[str, str], probe_hash: str,
                          hardware: dict[str, Any], environment: dict[str, Any]) -> tuple[bool, str]:
    controls = run.get("controls", {})
    if any(controls.get(k) != v for k, v in fixed.items()):
        return False, "fixed generation/sampling/KV/drafter/speculative/controller controls differ"
    if controls.get("controller") != "ordinary production CapController, max_draft_tokens=auto":
        return False, "CapController mode/configuration differs"
    if (controls.get("measurement_protocol") != measurement_protocol or
            controls.get("prompt_corpus") != prompt_corpus):
        return False, "measurement protocol or prompt corpus identity differs"
    final = controls.get("controller_final", {})
    if (not isinstance(final, dict) or not isinstance(final.get("cap"), int)
            or not isinstance(final.get("knee_width"), int) or not is_number(final.get("p"))
            or not isinstance(final.get("rounds"), int) or final["rounds"] <= 0
            or not isinstance(final.get("predicted_rates"), dict)
            or not all(is_number(v) and v > 0 for v in final["predicted_rates"].values())):
        return False, "recorded CapController final configuration is incomplete/invalid"
    if not _prompt_contract(run, prompt_ids, prompt_hashes):
        return False, "prompt IDs, hashes, or per-prompt token provenance differ"
    if not owner_manifest_ok(run.get("target_composition", {}), variant):
        return False, "target ownership differs from expected 64-block composition"
    cps = {k: (v.get("repo_id"), v.get("revision")) for k, v in run.get("checkpoints", {}).items()}
    if cps != checkpoints:
        return False, "target/drafter checkpoint identity or revision differs"
    runtime = run.get("runtime", {})
    if runtime.get("git", {}).get("head") != measured_head:
        return False, "measured git HEAD provenance differs"
    sources = runtime.get("runtime_source_sha256", {})
    expected_sources = dict(measured_sources)
    expected_sources["specs/002-qwen-bonsai-hybrid-target/evidence/probes/series_a.py"] = probe_hash
    if sources != expected_sources:
        return False, "complete experiment-relevant runtime source hash set differs"
    if runtime.get("mlx_device") != hardware:
        return False, "hardware/device identity differs from discovery-index canonical hardware"
    if runtime.get("machine_architecture") != environment["machine_architecture"]:
        return False, "machine architecture differs"
    for key in ("os", "python", "packages"):
        if runtime.get(key) != environment[key]:
            return False, f"runtime environment {key} differs"
    metrics_ok, detail = metric_contract(run)
    return metrics_ok, detail


def validate(feature_root: Path = FEATURE, *, gate_path: Path | None = None,
             gate_data: dict[str, Any] | None = None,
             injected: dict[str, Any] | None = None) -> dict[str, Any]:
    """Independently validate evidence and gate. ``injected`` exists for fail-closed tests."""
    root = feature_root.resolve()
    evidence = root / "evidence"
    repo = root.parents[1]
    gate_path = gate_path or evidence / "gate-b.json"
    checks: dict[str, dict[str, Any]] = {}
    injected = injected or {}

    def record(name: str, ok: bool, detail: str) -> None:
        checks[name] = {"passed": bool(ok), "detail": detail}

    def load(relpath: str) -> dict[str, Any]:
        if relpath in injected.get("documents", {}):
            return copy.deepcopy(injected["documents"][relpath])
        return read_json(repo / relpath)

    def load_raw(reference: dict[str, Any]) -> dict[str, Any]:
        path = _resolve_ref(reference, repo)
        key = relpath_from_root(path, repo)
        if key in injected.get("documents", {}):
            return copy.deepcopy(injected["documents"][key])
        return read_json(path)

    try:
        gate = copy.deepcopy(gate_data) if gate_data is not None else read_json(gate_path)
        record("gate_present_and_well_formed", True, "Gate B is a JSON object.")
    except Exception as e:
        gate = {}
        record("gate_present_and_well_formed", False, str(e))
    if gate.get("schema") != "qwen-bonsai-series-a-gate-b/v1" or gate.get("version") != 1:
        record("gate_schema", False, "Unexpected or missing gate schema/version.")
    else:
        record("gate_schema", True, "Recognized v1 schema.")

    try:
        cp = load("specs/002-qwen-bonsai-hybrid-target/evidence/checkpoints.json")
        got = {k: (v.get("repo_id"), v.get("revision")) for k, v in cp["checkpoints"].items()}
        ok = cp.get("accepted_revisions_are_immutable") is True and got == EXPECTED_REVISIONS
        record("immutable_checkpoint_provenance", ok, json.dumps(got, sort_keys=True))
    except Exception as e:
        record("immutable_checkpoint_provenance", False, str(e))

    try:
        proof = load("specs/002-qwen-bonsai-hybrid-target/evidence/integrity/official-bonsai-load.json")
        ok = (proof.get("passed") is True and proof.get("repo_id") == EXPECTED_REVISIONS["bonsai"][0]
              and proof.get("revision") == EXPECTED_REVISIONS["bonsai"][1]
              and proof.get("output_finite") is True and proof.get("parameter_tree_finite") is True
              and proof.get("feature_001_nathansutton_repack_used_as_evidence") is False)
        record("official_bonsai_loader_proof", ok, "Exact official checkpoint and finite load evidence.")
    except Exception as e:
        record("official_bonsai_loader_proof", False, str(e))

    try:
        donor = load("specs/002-qwen-bonsai-hybrid-target/evidence/integrity/donor-memory-release.json")
        obs = donor["observations"]
        ok = (donor.get("passed") is True and donor.get("donor_repo") == EXPECTED_REVISIONS["bonsai"][0]
              and donor.get("donor_revision") == EXPECTED_REVISIONS["bonsai"][1]
              and obs["full_donor_active_reclaim_fraction"] >= donor["thresholds"]["minimum_active_memory_reclaim_fraction"]
              and obs["before_donor_load"]["active_bytes"] < obs["donor_peak"]["active_bytes"]
              and obs["after_pruning_release_cache_cleanup"]["active_bytes"] < obs["donor_peak"]["active_bytes"]
              and obs["immediately_before_qwen_load"]["active_bytes"] == obs["after_pruning_release_cache_cleanup"]["active_bytes"]
              and obs["immediately_before_qwen_load"]["cache_bytes"] == 0
              and donor.get("selective_loading_fallback_used") is False)
        record("donor_memory_release", ok, "Recorded measured MLX observations satisfy donor-release gate.")
    except Exception as e:
        record("donor_memory_release", False, str(e))

    try:
        index = load("specs/002-qwen-bonsai-hybrid-target/evidence/integrity/index.json")
        ok = (index.get("status") == "PASS" and index.get("passed") is True
              and index.get("required_targets") == TARGETS and set(index.get("targets", {})) == set(TARGETS))
        for target in TARGETS:
            ref = index["targets"][target]
            p = _resolve_ref(ref, repo)
            d = load(relpath_from_root(p, repo))
            ok = ok and ref.get("passed") is True and d.get("passed") is True and d.get("variant") == target
            own = d.get("ownership", {})
            expected_nonblocks = "bonsai" if target == "B0" else "qwen"
            ok = ok and owner_manifest_ok(own, target)
            ok = ok and all(own.get(k) == expected_nonblocks for k in ("embedding_owner", "final_norm_owner", "lm_head_owner"))
            ok = ok and sha256_file(p) == ref["sha256"]
        record("seven_target_integrity", ok, "All seven target records and index hashes validated.")
    except Exception as e:
        record("seven_target_integrity", False, str(e))

    try:
        sem = load("specs/002-qwen-bonsai-hybrid-target/evidence/integrity/runtime-semantics.json")
        rep = sem["representation"]
        loaded = sem["loaded_dflash"]
        boundary_ok = (rep.get("bridge_required") is False and
                       all(x.get("outputs_finite") is True for x in rep["actual_boundary_smokes"].values()) and
                       all(edge.get("direct_residual_stream") is True for x in rep["actual_boundary_smokes"].values() for edge in x["edges"]))
        tap_ok = (loaded.get("loaded_target_layer_ids") == TAPS and loaded.get("target_layer_ids") == TAPS
                  and loaded.get("revision") == EXPECTED_REVISIONS["dflash"][1]
                  and loaded.get("direct_bonsai_tap_counts") == TAP_COUNTS
                  and loaded.get("semantics") == "zero-based block output, captured immediately after layer(...) returns")
        record("representation_boundaries", sem.get("passed") is True and boundary_ok,
               "Canonical residual stream at every observed boundary; no bridge required.")
        record("dflash_tap_semantics", sem.get("passed") is True and tap_ok,
               "Loaded taps and direct Bonsai-owned counts match the accepted tap contract.")
    except Exception as e:
        record("representation_boundaries", False, str(e))
        record("dflash_tap_semantics", False, str(e))

    try:
        rpt = load("specs/002-qwen-bonsai-hybrid-target/evidence/integrity/repeatability.json")
        h0_record = load("specs/002-qwen-bonsai-hybrid-target/evidence/integrity/h0.json")
        ok = (rpt.get("passed") is True
              and rpt.get("h0_same_revision_ordinary_path_generated_ids_match") is True
              and rpt.get("h0_same_generation_settings") is True
              and rpt.get("h0_same_prompt_ids") is True
              and rpt.get("h2_fresh_process_manifest_repeatable") is True
              and h0_record.get("target_mode") == "ordinary Qwen H0")
        record("h0_unchanged_and_composition_repeatability", ok,
               "H0 ordinary-path output identity and fresh-process H2 repeatability are recorded PASS.")
    except Exception as e:
        record("h0_unchanged_and_composition_repeatability", False, str(e))

    try:
        discovery = load("specs/002-qwen-bonsai-hybrid-target/evidence/series-a/discovery-index.json")
        runs = discovery["targets"]
        ok = (discovery.get("status") == "PASS" and discovery.get("passed") is True
              and discovery.get("order") == DISCOVERY_ORDER
              and [r["variant"] for r in runs] == DISCOVERY_ORDER
              and [r["order_index"] for r in runs] == list(range(1, 9)))
        for r in runs:
            p = _resolve_ref(r, repo)
            d = read_json(p)
            ok = ok and r.get("status") == "PASS" and sha256_file(p) == r["sha256"]
            ok = ok and d.get("status") == "PASS" and d.get("variant") == r["variant"]
            ok = ok and d.get("runtime", {}).get("git", {}).get("head") == MEASURED_HEAD
        h0 = [r for r in runs if r["variant"] == "H0"]
        ok = ok and len(h0) == 2 and h0[0]["order_index"] == 1 and h0[1]["order_index"] == 8
        record("discovery_matrix_and_hashes", ok, "Eight exact ordered discovery results, including both H0 brackets, hash-checked.")
        h0_runs = [read_json(_resolve_ref(r, repo)) for r in h0]
        rates = [x.get("measured", {}).get("speculative", {}).get("decode_tokens_per_sec") for x in h0_runs]
        fresh = load("specs/002-qwen-bonsai-hybrid-target/evidence/adjudication/series-a.json")["derived_metrics"]["fresh_h0_baseline"]
        h0_ok = all(isinstance(v, (int, float)) for v in rates) and len(rates) == 2
        h0_ok = h0_ok and abs(sum(rates) / 2 - fresh.get("decode_tokens_per_sec", -1)) < 1e-9
        record("fresh_h0_discovery_brackets", h0_ok, f"H0 discovery rates: {rates}; mean checked against adjudication.")
        controls = discovery.get("fixed_controls", {})
        expected_controls = {"drafter_repo": EXPECTED_REVISIONS["dflash"][0],
                             "drafter_revision": EXPECTED_REVISIONS["dflash"][1], "plain_kv": True,
                             "kv_bits": None, "controller": "ordinary production CapController, max_draft_tokens=auto",
                             "mode": "dflash"}
        record("matched_controls", all(controls.get(k) == v for k, v in expected_controls.items()),
               "DFlash, plain KV, and production CapController controls checked.")
        prompt_hashes = discovery.get("fixed_prompt_token_hashes", [])
        runs_match = True
        for entry, run in zip(runs, [read_json(_resolve_ref(x, repo)) for x in runs]):
            prompt_run = [x.get("prompt_ids_sha256") for x in run.get("measured", {}).get("speculative", {}).get("per_prompt", [])]
            run_controls = run.get("controls", {})
            sources = run.get("runtime", {}).get("runtime_source_sha256", {})
            owners = run.get("target_composition", {})
            runs_match = runs_match and run_controls and all(run_controls.get(k) == v for k, v in controls.items())
            runs_match = runs_match and prompt_run == prompt_hashes
            runs_match = runs_match and owner_manifest_ok(owners, entry["variant"])
            runs_match = runs_match and all(sources.get(k) == v for k, v in discovery["runtime_revision"]["source_sha256"].items())
        record("discovery_controls_prompts_ownership", runs_match,
               "All discovery records match fixed controls/prompts and exact 64-block ownership.")
        canonical_run = load_raw(runs[0])
        canonical_prompts = canonical_run.get("controls", {}).get("prompt_ids", [])
        canonical_prompt_corpus = canonical_run.get("controls", {}).get("prompt_corpus", {})
        measurement_protocol = canonical_run.get("controls", {}).get("measurement_protocol", "")
        canonical_hashes = discovery.get("fixed_prompt_token_hashes", [])
        hardware = discovery["runtime_revision"]["hardware"]
        canonical_runtime = canonical_run["runtime"]
        environment = {k: canonical_runtime[k] for k in ("machine_architecture", "os", "python", "packages")}
        checkpoints = EXPECTED_REVISIONS
        measured_head = discovery["runtime_revision"]["git_head"]
        measured_sources = discovery["runtime_revision"]["source_sha256"]
        # The snapshot is source, not a JSON record; its hash is checked by the
        # immutable-reference inventory. Discovery run provenance records that hash.
        probe_hash = discovery["discovery_runner_snapshot"]["sha256"]
        full_discovery = True
        metric_details = []
        for entry in runs:
            run = load_raw(entry)
            matched, detail = _matched_run_contract(run, entry["variant"], controls,
                canonical_prompts, canonical_hashes, canonical_prompt_corpus,
                measurement_protocol, checkpoints, measured_head,
                measured_sources, probe_hash, hardware, environment)
            full_discovery = full_discovery and matched
            metric_details.append(detail)
        record("discovery_complete_matched_conditions_and_metrics", full_discovery,
               "All eight discovery records match complete prompt/control/checkpoint/runtime/hardware/environment provenance and required metrics; "
               + "; ".join(sorted(set(metric_details))))
    except Exception as e:
        record("discovery_matrix_and_hashes", False, str(e))
        record("fresh_h0_discovery_brackets", False, str(e))
        record("matched_controls", False, str(e))
        record("discovery_controls_prompts_ownership", False, str(e))
        record("discovery_complete_matched_conditions_and_metrics", False, str(e))

    try:
        selection = load("specs/002-qwen-bonsai-hybrid-target/evidence/series-a/repeat-selection.json")
        ids = {x["comparison_id"] for x in selection["comparisons"] if x.get("selected") is True}
        ok = (selection.get("status") == "PASS" and selection.get("alternating_order") is True
              and ids == COMPARISONS and set(selection.get("selected_comparisons", [])) == COMPARISONS
              and selection.get("required_matched_pairs_per_comparison") == 3
              and selection.get("planned_run_count") == 48 and len(selection.get("matched_pair_schedule", [])) == 24)
        record("repeat_selection", ok, "All eight required comparisons selected for three pairs each.")
    except Exception as e:
        record("repeat_selection", False, str(e))

    try:
        schedule = load("specs/002-qwen-bonsai-hybrid-target/evidence/series-a/confirmation-schedule.json")
        pairs = schedule["matched_pairs"]
        flat = [run for pair in pairs for run in pair["runs"]]
        order = [run["order_index"] for run in flat]
        ids = [run["run_id"] for run in flat]
        pair_ids = [p["pair_id"] for p in pairs]
        by_comparison: dict[str, list[dict[str, Any]]] = {}
        for pair in pairs:
            by_comparison.setdefault(pair["comparison_id"], []).append(pair)
        alternating = all(
            [p.get("alternation") for p in group] == ["A-B", "B-A", "A-B"]
            for group in by_comparison.values()
        )
        run_ok = (schedule.get("status") == "PASS" and schedule.get("all_runs_passed") is True
                  and schedule.get("run_count") == 48 and len(pairs) == 24
                  and set(p["comparison_id"] for p in pairs) == COMPARISONS
                  and all(sum(p["comparison_id"] == c for p in pairs) == 3 for c in COMPARISONS)
                  and order == list(range(1, 49)) and len(set(ids)) == 48
                  and len(set(pair_ids)) == 24 and alternating)
        hashes_ok = True
        confirmation_probe_hashes: set[str] = set()
        confirmation_raw: list[tuple[dict[str, Any], dict[str, Any]]] = []
        selection_doc = load("specs/002-qwen-bonsai-hybrid-target/evidence/series-a/repeat-selection.json")
        comp_meta = {x["comparison_id"]: x for x in selection_doc["comparisons"]}
        expected_controls = load("specs/002-qwen-bonsai-hybrid-target/evidence/series-a/discovery-index.json")["fixed_controls"]
        expected_prompts = load("specs/002-qwen-bonsai-hybrid-target/evidence/series-a/discovery-index.json")["fixed_prompt_token_hashes"]
        for r in flat:
            p = _resolve_ref(r, repo)
            result = load_raw(r)
            confirmation_raw.append((r, result))
            hashes_ok = hashes_ok and r.get("status") == "PASS" and result.get("status") == "PASS"
            hashes_ok = hashes_ok and sha256_file(p) == r.get("sha256")
            hashes_ok = hashes_ok and result.get("run_id") == r.get("run_id")
            hashes_ok = hashes_ok and result.get("variant") == r.get("target")
            run_controls = result.get("controls", {})
            hashes_ok = hashes_ok and all(run_controls.get(k) == v for k, v in expected_controls.items())
            prompt_hashes = [x.get("prompt_ids_sha256") for x in result.get("measured", {}).get("speculative", {}).get("per_prompt", [])]
            hashes_ok = hashes_ok and prompt_hashes == expected_prompts
            hashes_ok = hashes_ok and owner_manifest_ok(result.get("target_composition", {}), r.get("target"))
            expected_cps = {k: {"repo_id": repo_id, "revision": revision}
                            for k, (repo_id, revision) in EXPECTED_REVISIONS.items()}
            actual_cps = {k: {"repo_id": v.get("repo_id"), "revision": v.get("revision")}
                          for k, v in result.get("checkpoints", {}).items()}
            hashes_ok = hashes_ok and actual_cps == expected_cps
            run_sources = result.get("runtime", {}).get("runtime_source_sha256", {})
            measured_sources = schedule.get("discovery_runtime_source_sha256", {})
            hashes_ok = hashes_ok and all(run_sources.get(k) == v for k, v in measured_sources.items())
            # The confirmation runner added only selection/confirmation machinery.
            # Its benchmark method is separately pinned byte-for-byte below.
            probe_path = "specs/002-qwen-bonsai-hybrid-target/evidence/probes/series_a.py"
            hashes_ok = hashes_ok and set(run_sources) == set(measured_sources) | {probe_path}
            if probe_path in run_sources:
                confirmation_probe_hashes.add(run_sources[probe_path])
        pair_order_ok = True
        for pair in pairs:
            meta = comp_meta.get(pair["comparison_id"], {})
            candidate, control = meta.get("candidate"), meta.get("control")
            expected_targets = ([candidate, control] if pair.get("alternation") == "A-B" else [control, candidate])
            actual_targets = [x.get("target") for x in pair.get("runs", [])]
            pair_order_ok = pair_order_ok and actual_targets == expected_targets
            pair_order_ok = pair_order_ok and pair.get("first_target") == (expected_targets[0] if expected_targets else None)
            pair_order_ok = pair_order_ok and pair.get("second_target") == (expected_targets[1] if len(expected_targets) > 1 else None)
        run_ok = run_ok and pair_order_ok
        hashes_ok = hashes_ok and len(confirmation_probe_hashes) == 1
        record("confirmation_24_pairs_48_runs", run_ok, f"pairs={len(pairs)}, runs={len(flat)}, order=1..48.")
        record("confirmation_alternation", run_ok and alternating, "Matched pair order alternates A-B/B-A.")
        record("confirmation_hashes_and_status", hashes_ok, "All 48 raw result files hash-match and report PASS.")
        record("benchmark_method_identity", schedule.get("benchmark_method_sha256") == METHOD_SHA256,
               "Recorded run_series_condition method hash matches accepted byte-identical method fingerprint; confirmation-only runner additions are isolated to series_a.py.")
        discovery_doc = load("specs/002-qwen-bonsai-hybrid-target/evidence/series-a/discovery-index.json")
        first_discovery = load_raw(discovery_doc["targets"][0])
        canonical_prompts = first_discovery["controls"]["prompt_ids"]
        canonical_prompt_corpus = first_discovery["controls"]["prompt_corpus"]
        measurement_protocol = first_discovery["controls"]["measurement_protocol"]
        environment = {k: first_discovery["runtime"][k]
                       for k in ("machine_architecture", "os", "python", "packages")}
        probe_hash = next(iter(confirmation_probe_hashes)) if len(confirmation_probe_hashes) == 1 else ""
        complete = bool(probe_hash) and schedule.get("discovery_runtime_source_sha256") == discovery_doc["runtime_revision"]["source_sha256"]
        metric_details = []
        for r, result in confirmation_raw:
            matched, detail = _matched_run_contract(result, r["target"], expected_controls,
                canonical_prompts, expected_prompts, canonical_prompt_corpus,
                measurement_protocol, EXPECTED_REVISIONS,
                discovery_doc["runtime_revision"]["git_head"],
                discovery_doc["runtime_revision"]["source_sha256"], probe_hash,
                discovery_doc["runtime_revision"]["hardware"], environment)
            complete = complete and matched
            metric_details.append(detail)
        record("confirmation_complete_matched_conditions_and_metrics", complete,
               "All 48 confirmation records match complete prompt/control/checkpoint/runtime/hardware/environment provenance and required metrics; "
               + "; ".join(sorted(set(metric_details))))
    except Exception as e:
        record("confirmation_24_pairs_48_runs", False, str(e))
        record("confirmation_alternation", False, str(e))
        record("confirmation_hashes_and_status", False, str(e))
        record("benchmark_method_identity", False, str(e))
        record("confirmation_complete_matched_conditions_and_metrics", False, str(e))

    try:
        decisions = load("specs/002-qwen-bonsai-hybrid-target/evidence/series-a/repeat-decisions.json")
        decision_map = {x["comparison_id"]: x for x in decisions["comparisons"]}
        expected_decisions = {"H1a-vs-H0": "confirmed_regression", "H1b-vs-H0": "confirmed_regression",
                              "H1c-vs-H0": "confirmed_regression", "H2-vs-H0": "confirmed_regression",
                              "H3-vs-H0": "confirmed_regression", "B0-vs-H0": "confirmed_regression",
                              "H1c-vs-H1a": "confirmed_win", "H1c-vs-H1b": "within_noise"}
        ok = (decisions.get("status") == "PASS" and decisions.get("all_selected_comparisons_resolved") is True
              and set(decision_map) == COMPARISONS
              and all(decision_map[k].get("adjudicated_result") == v and
                      decision_map[k].get("matched_pair_count") == 3 for k, v in expected_decisions.items()))
        record("repeat_decisions_resolved", ok, "Eight comparisons have the accepted resolved direction/noise labels.")
        arithmetic_ok = True
        arithmetic_errors: list[str] = []
        schedule_doc = load("specs/002-qwen-bonsai-hybrid-target/evidence/series-a/confirmation-schedule.json")
        discovery_doc = load("specs/002-qwen-bonsai-hybrid-target/evidence/series-a/discovery-index.json")
        h0_entries = [x for x in discovery_doc.get("targets", []) if x.get("variant") == "H0"]
        h0_rates = [load_raw(x).get("measured", {}).get("speculative", {}).get("decode_tokens_per_sec")
                    for x in h0_entries]
        if len(h0_rates) != 2 or not all(is_number(x) for x in h0_rates):
            arithmetic_ok = False
            arithmetic_errors.append("fresh H0 bracket rates are ambiguous/missing")
            h0_noise = math.nan
        else:
            h0_noise = abs(h0_rates[0] - h0_rates[1])
        grouped_pairs: dict[str, list[dict[str, Any]]] = {}
        for pair in schedule_doc.get("matched_pairs", []):
            grouped_pairs.setdefault(pair.get("comparison_id"), []).append(pair)
        for cid in COMPARISONS:
            recorded = decision_map.get(cid, {})
            pairs = sorted(grouped_pairs.get(cid, []), key=lambda x: x.get("pair_id", ""))
            candidate, control = recorded.get("candidate"), recorded.get("control")
            expected_axis = {
                "H1a-vs-H0": ("H1a", "H0"), "H1b-vs-H0": ("H1b", "H0"),
                "H1c-vs-H0": ("H1c", "H0"), "H2-vs-H0": ("H2", "H0"),
                "H3-vs-H0": ("H3", "H0"), "B0-vs-H0": ("B0", "H0"),
                "H1c-vs-H1a": ("H1c", "H1a"), "H1c-vs-H1b": ("H1c", "H1b"),
            }[cid]
            if (candidate, control) != expected_axis or any(
                    pair.get("comparison_id") != cid for pair in pairs):
                arithmetic_ok = False
                arithmetic_errors.append(f"{cid}: comparison candidate/control mapping differs from selected contract")
                continue
            deltas: list[float] = []
            pair_deltas: list[float] = []
            if len(pairs) != 3 or candidate not in TARGETS or control not in TARGETS:
                arithmetic_ok = False
                arithmetic_errors.append(f"{cid}: schedule does not identify exactly three candidate/control pairs")
                continue
            for pair in pairs:
                refs = pair.get("runs", [])
                candidate_refs = [x for x in refs if x.get("target") == candidate]
                control_refs = [x for x in refs if x.get("target") == control]
                if len(refs) != 2 or len(candidate_refs) != 1 or len(control_refs) != 1:
                    arithmetic_ok = False
                    arithmetic_errors.append(f"{cid}/{pair.get('pair_id')}: pair target mapping is ambiguous")
                    continue
                candidate_run, control_run = load_raw(candidate_refs[0]), load_raw(control_refs[0])
                try:
                    delta = (candidate_run["measured"]["speculative"]["decode_tokens_per_sec"] -
                             control_run["measured"]["speculative"]["decode_tokens_per_sec"])
                    if not is_number(delta):
                        raise ValueError("non-finite decode delta")
                    deltas.append(delta)
                    pair_deltas.append(delta)
                except Exception as e:
                    arithmetic_ok = False
                    arithmetic_errors.append(f"{cid}/{pair.get('pair_id')}: {e}")
            if len(deltas) != 3:
                continue
            median = statistics.median(deltas)
            spread = max(deltas) - min(deltas)
            envelope = max(h0_noise, spread)
            positive, negative = all(x > 0 for x in deltas), all(x < 0 for x in deltas)
            if positive and median > envelope:
                outcome = "confirmed_win"
            elif negative and abs(median) > envelope:
                outcome = "confirmed_regression"
            elif abs(median) <= envelope or not (positive or negative):
                outcome = "within_noise"
            else:
                outcome = "inconclusive"
            if (len(recorded.get("paired_decode_deltas", [])) != 3 or
                    any(not _close(a, b) for a, b in zip(recorded.get("paired_decode_deltas", []), deltas)) or
                    not _close(recorded.get("median_paired_decode_delta"), median) or
                    not _close(recorded.get("paired_range"), spread) or
                    not _close(recorded.get("observed_h0_bracket_noise_abs_range"), h0_noise) or
                    not _close(recorded.get("effective_observed_noise_envelope"), envelope) or
                    recorded.get("adjudicated_result") != outcome):
                arithmetic_ok = False
                arithmetic_errors.append(f"{cid}: recorded deltas/statistics/noise/outcome disagree with raw runs")
            stored_pairs = recorded.get("pairs", [])
            if (len(stored_pairs) != 3 or any(
                    stored_pairs[i].get("pair_id") != pairs[i].get("pair_id") or
                    not _close(stored_pairs[i].get("candidate_minus_control", {}).get("decode_tokens_per_sec"), deltas[i])
                    for i in range(min(3, len(stored_pairs))))):
                arithmetic_ok = False
                arithmetic_errors.append(f"{cid}: per-pair decision records disagree with schedule/raw runs")
        record("repeat_decision_arithmetic_from_raw_runs", arithmetic_ok,
               "Recomputed 24 matched pairs, all eight paired-delta lists/medians/ranges, fresh-H0 noise envelopes, and outcomes from raw files."
               if arithmetic_ok else "; ".join(arithmetic_errors))
        adjudication = load("specs/002-qwen-bonsai-hybrid-target/evidence/adjudication/series-a.json")
        interp = adjudication.get("interpretation", {})
        facts = adjudication.get("measured_facts", {})
        aok = (adjudication.get("status") == "PASS"
               and interp.get("any_tested_hybrid_improves_decode_over_fresh_h0") is False
               and interp.get("confirmed_hybrid_decode_wins") == []
               and interp.get("gate_b", "").startswith("not opened")
               and len(facts.get("confirmation_decisions", [])) == 8)
        record("series_a_adjudication_complete", aok,
               "Adjudication is complete; no-win outcome is accepted and does not gate B.")
    except Exception as e:
        record("repeat_decisions_resolved", False, str(e))
        record("repeat_decision_arithmetic_from_raw_runs", False, str(e))
        record("series_a_adjudication_complete", False, str(e))

    try:
        spec_path = repo / "specs/002-qwen-bonsai-hybrid-target/spec.md"
        const_path = repo / ".specify/memory/constitution.md"
        spec_hash = injected.get("spec_hash", sha256_file(spec_path))
        const_hash = injected.get("constitution_hash", sha256_file(const_path))
        governing = gate.get("governing_inputs", {})
        spec_ok = (governing.get("spec", {}).get("sha256") == spec_hash
                   and spec_hash == GOVERNING_SPEC_SHA256 and sha256_file(spec_path) == spec_hash)
        const_ok = (governing.get("constitution", {}).get("sha256") == const_hash
                    and governing.get("constitution", {}).get("version") == "2.0.0"
                    and const_hash == GOVERNING_CONSTITUTION_SHA256 and sha256_file(const_path) == const_hash
                    and "**Version**: 2.0.0" in const_path.read_text())
        record("governing_spec_hash", spec_ok, f"current={spec_hash}")
        record("governing_constitution_hash_version", const_ok, f"current={const_hash}; version=2.0.0")
    except Exception as e:
        record("governing_spec_hash", False, str(e))
        record("governing_constitution_hash_version", False, str(e))

    try:
        discovery = load("specs/002-qwen-bonsai-hybrid-target/evidence/series-a/discovery-index.json")
        measured = discovery["runtime_revision"]["source_sha256"]
        now = injected.get("runtime_hashes", current_runtime_hashes())
        present = {name: sha256_file(repo / name) for name in measured}
        hashes_ok = set(now) == set(measured) and set(present) == set(measured)
        hashes_ok = hashes_ok and all(now[k] == measured[k] and present[k] == measured[k] for k in measured)
        gate_runtime = gate.get("runtime", {})
        hashes_ok = hashes_ok and gate_runtime.get("measured_source_sha256") == measured
        hashes_ok = hashes_ok and gate_runtime.get("current_source_sha256") == now
        record("complete_experiment_runtime_hash_set", hashes_ok,
               json.dumps({k: {"measured": measured[k], "current": now.get(k)} for k in measured}, sort_keys=True))
        measured_head_ok = (gate_runtime.get("measured_git_head") == discovery["runtime_revision"]["git_head"]
                            == MEASURED_HEAD)
        record("measured_git_head_preserved", measured_head_ok,
               f"measured={discovery['runtime_revision']['git_head']}; current history is provenance only.")
        source = (repo / "specs/002-qwen-bonsai-hybrid-target/evidence/series-a/runner-used-for-discovery.py").read_bytes()
        method_ok = benchmark_method_hash(source) == METHOD_SHA256
        record("benchmark_implementation_fingerprint", method_ok, f"AST SHA-256={benchmark_method_hash(source)}")
    except Exception as e:
        record("complete_experiment_runtime_hash_set", False, str(e))
        record("measured_git_head_preserved", False, str(e))
        record("benchmark_implementation_fingerprint", False, str(e))

    # Verify every immutable reference stored in the gate, including all raw run records.
    try:
        expected_refs = evidence_refs_for_root(repo, evidence)
        stored = gate.get("references", {})
        refs_ok = stored == expected_refs
        record("referenced_artifact_hashes", refs_ok, "Gate reference inventory equals the live fixed evidence inventory.")
    except Exception as e:
        record("referenced_artifact_hashes", False, str(e))

    try:
        validator_hash = sha256_file(Path(__file__).resolve())
        record("validator_source_integrity", gate.get("audit", {}).get("validator_sha256") == validator_hash,
               f"current gate validator SHA-256={validator_hash}")
    except Exception as e:
        record("validator_source_integrity", False, str(e))

    try:
        checkpoints = load("specs/002-qwen-bonsai-hybrid-target/evidence/checkpoints.json")
        discovery = load("specs/002-qwen-bonsai-hybrid-target/evidence/series-a/discovery-index.json")
        semantics = load("specs/002-qwen-bonsai-hybrid-target/evidence/integrity/runtime-semantics.json")
        expected_cp = {name: {"repo_id": checkpoints["checkpoints"][name]["repo_id"],
                              "revision": checkpoints["checkpoints"][name]["revision"]}
                       for name in EXPECTED_REVISIONS}
        expected_governing = {
            "spec": {"path": "specs/002-qwen-bonsai-hybrid-target/spec.md",
                     "sha256": sha256_file(repo / "specs/002-qwen-bonsai-hybrid-target/spec.md")},
            "constitution": {"path": ".specify/memory/constitution.md", "version": "2.0.0",
                             "sha256": sha256_file(repo / ".specify/memory/constitution.md")},
        }
        sem_taps = semantics["loaded_dflash"]
        expected_runtime = gate.get("runtime", {})
        current_hashes = injected.get("runtime_hashes", current_runtime_hashes())
        authority_ok = (
            gate.get("checkpoint_provenance") == expected_cp == {
                name: {"repo_id": repo_id, "revision": revision}
                for name, (repo_id, revision) in EXPECTED_REVISIONS.items()} and
            gate.get("required_targets") == TARGETS and
            gate.get("discovery_order") == DISCOVERY_ORDER and
            gate.get("dflash_taps") == TAPS == sem_taps.get("loaded_target_layer_ids") and
            gate.get("direct_bonsai_tap_counts") == TAP_COUNTS == sem_taps.get("direct_bonsai_tap_counts") and
            gate.get("governing_inputs") == expected_governing and
            expected_runtime.get("measured_git_head") == discovery["runtime_revision"]["git_head"] == MEASURED_HEAD and
            expected_runtime.get("measured_source_sha256") == discovery["runtime_revision"]["source_sha256"] and
            expected_runtime.get("current_source_sha256") == current_hashes and
            set(current_hashes) == set(discovery["runtime_revision"]["source_sha256"]) and
            all(current_hashes.get(k) == v for k, v in discovery["runtime_revision"]["source_sha256"].items()) and
            expected_runtime.get("benchmark_method_sha256") == METHOD_SHA256 and
            gate.get("references") == evidence_refs_for_root(repo, evidence) and
            gate.get("audit", {}).get("validator") == "evidence/gate_b.py" and
            gate.get("audit", {}).get("validator_sha256") == sha256_file(Path(__file__).resolve())
        )
        record("gate_authoritative_fields", authority_ok,
               "Duplicated checkpoint/target/order/tap/governing/runtime/method/reference/validator fields match live authority.")
    except Exception as e:
        record("gate_authoritative_fields", False, str(e))

    statuses = (gate.get("decision"), gate.get("status"), gate.get("final_status"))
    status_ok = statuses[0] in {"OPEN", "CLOSED"} and statuses[0] == statuses[1] == statuses[2]
    record("gate_decision_status_consistency", status_ok,
           "decision, status, and final_status agree on OPEN or CLOSED.")

    # An OPEN gate carries a snapshot of every live check that authorized it.
    # Compare names and full results, not just a stored all-PASS summary.
    if statuses == ("OPEN", "OPEN", "OPEN"):
        stored_checks = gate.get("closure_checks")
        current_checks = substantive_checks(checks)
        snapshot_ok = (isinstance(stored_checks, dict) and stored_checks == current_checks
                       and all(isinstance(v, dict) and v.get("passed") is True for v in stored_checks.values())
                       and all(v.get("passed") is True for v in current_checks.values()))
        record("stored_closure_checks_consistency", snapshot_ok,
               "OPEN closure snapshot has exactly the current check set/results and all checks pass.")
    else:
        record("stored_closure_checks_consistency", True,
               "Gate is CLOSED; no stored closure snapshot can authorize entry.")

    return {"passed": bool(checks) and all(x["passed"] for x in checks.values()), "checks": checks}


def substantive_checks(report_or_checks: dict[str, Any]) -> dict[str, Any]:
    """Return authorization checks, excluding the live snapshot self-check."""
    checks = report_or_checks.get("checks", report_or_checks)
    return {k: v for k, v in checks.items() if k != "stored_closure_checks_consistency"}


def closure_snapshot(report: dict[str, Any]) -> dict[str, Any]:
    """Persist only the non-recursive substantive authorization results."""
    return substantive_checks(report)


def relpath_from_root(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def expected_owner(variant: str, index: int) -> str:
    starts = {"H0": 64, "H1a": 63, "H1b": None, "H1c": 62, "H2": 60, "H3": 56, "B0": 0}
    if variant == "H1b":
        return "bonsai" if index == 62 else "qwen"
    return "bonsai" if index >= starts[variant] else "qwen"


def owner_manifest_ok(manifest: dict[str, Any], variant: str) -> bool:
    blocks = manifest.get("block_owners", [])
    if len(blocks) != 64:
        return False
    expected = [{"index": i, "owner": expected_owner(variant, i)} for i in range(64)]
    actual = [{"index": x.get("index"), "owner": x.get("owner")} for x in blocks]
    return actual == expected


def evidence_refs_for_root(repo: Path, evidence: Path) -> dict[str, Any]:
    """Equivalent of evidence_refs with paths stable relative to the selected checkout."""
    def mk(p: Path) -> dict[str, str]:
        return {"path": p.resolve().relative_to(repo.resolve()).as_posix(), "sha256": sha256_file(p)}
    idx = read_json(evidence / "integrity/index.json")
    discovery = read_json(evidence / "series-a/discovery-index.json")
    schedule = read_json(evidence / "series-a/confirmation-schedule.json")
    return {
        "checkpoints": mk(evidence / "checkpoints.json"),
        "integrity_index": mk(evidence / "integrity/index.json"),
        "official_bonsai_loader": mk(evidence / "integrity/official-bonsai-load.json"),
        "integrity_targets": {t: mk(repo / idx["targets"][t]["path"]) for t in TARGETS},
        "runtime_semantics": mk(evidence / "integrity/runtime-semantics.json"),
        "donor_memory": mk(evidence / "integrity/donor-memory-release.json"),
        "integrity_repeatability": mk(repo / idx["repeatability"]["path"]),
        "fresh_h0": mk(evidence / "integrity/h0-fresh.json"),
        "discovery_index": mk(evidence / "series-a/discovery-index.json"),
        "discovery_runs": [mk(repo / r["path"]) for r in discovery["targets"]],
        "discovery_order": mk(repo / discovery["discovery_order"]["path"]),
        "discovery_runner_snapshot": mk(repo / discovery["discovery_runner_snapshot"]["path"]),
        "prompt_corpus": mk(repo / discovery["prompt_corpus"]["path"]),
        "repeat_selection": mk(evidence / "series-a/repeat-selection.json"),
        "confirmation_schedule": mk(evidence / "series-a/confirmation-schedule.json"),
        "confirmation_runs": [mk(repo / r["path"]) for p in schedule["matched_pairs"] for r in p["runs"]],
        "repeat_decisions": mk(evidence / "series-a/repeat-decisions.json"),
        "adjudication": mk(evidence / "adjudication/series-a.json"),
    }


def require_series_b_entry(gate_path: Path = GATE, *, injected: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fail-closed entry guard. Call before any Series-B model/checkpoint access."""
    if not gate_path.exists():
        raise RuntimeError(f"Series B denied: Gate B missing at {gate_path}")
    try:
        gate = read_json(gate_path)
    except Exception as e:
        raise RuntimeError(f"Series B denied: malformed Gate B: {e}") from e
    if gate.get("status") != "OPEN" or gate.get("final_status") != "OPEN" or gate.get("decision") != "OPEN":
        raise RuntimeError("Series B denied: Gate B status is not OPEN")
    report = validate(gate_path=gate_path, injected=injected)
    if not report["passed"]:
        failed = [k for k, v in report["checks"].items() if not v["passed"]]
        raise RuntimeError("Series B denied: closure validation failed: " + ", ".join(failed))
    return report


def write_gate(gate_path: Path = GATE) -> dict[str, Any]:
    """Write CLOSED first, validate live evidence, then open only on full PASS."""
    # Establish denial on disk before any evidence/hash/git read can throw.
    _atomic_json(gate_path, minimal_closed_gate())
    try:
        gate = initial_gate()
        _atomic_json(gate_path, gate)
        report = validate(gate_path=gate_path)
        gate["closure_checks"] = closure_snapshot(report)
        gate["audit"].update({"validated": True, "validation_passed": report["passed"]})
        gate["runtime"]["current_git_head"] = git_head()
        gate["runtime"]["current_source_sha256"] = current_runtime_hashes()
        if report["passed"]:
            gate["decision"] = gate["status"] = gate["final_status"] = "OPEN"
            _atomic_json(gate_path, gate)
            final_report = validate(gate_path=gate_path)
            if final_report["passed"]:
                return final_report
            report = final_report
        gate["decision"] = gate["status"] = gate["final_status"] = "CLOSED"
        gate["closure_checks"] = closure_snapshot(report)
        gate["audit"]["validation_passed"] = False
        _atomic_json(gate_path, gate)
        return report
    except Exception as e:
        # Even failures while constructing/populating the full record leave an
        # atomic, schema-valid CLOSED document on disk.
        try:
            existing = read_json(gate_path) if gate_path.exists() else minimal_closed_gate()
        except Exception:
            existing = minimal_closed_gate()
        existing["decision"] = existing["status"] = existing["final_status"] = "CLOSED"
        existing.setdefault("audit", {}).update({"validated": True, "validation_passed": False})
        existing.setdefault("closure_checks", {})["validation_exception"] = {
            "passed": False, "detail": f"{type(e).__name__}: {e}"}
        _atomic_json(gate_path, existing)
        return {"passed": False, "checks": {"validation_exception": {
            "passed": False, "detail": f"{type(e).__name__}: {e}"}}}


def self_test() -> list[dict[str, Any]]:
    """Exercise T030 fail-closed conditions without imports or model loading."""
    global initial_gate
    gate = read_json(GATE)
    base = validate(gate_data=gate)
    cases: list[tuple[str, bool]] = []
    with tempfile.TemporaryDirectory(prefix="gate-b-open-snapshot-") as td:
        generated_path = Path(td) / "gate-b.json"
        generated_report = write_gate(generated_path)
        generated = read_json(generated_path)
        live = validate(gate_path=generated_path)
        persisted = generated.get("closure_checks")
        live_substantive = substantive_checks(live)
        cases.append(("generated_open_snapshot_excludes_live_self_check",
                      generated.get("status") == "OPEN" and isinstance(persisted, dict)
                      and "stored_closure_checks_consistency" not in persisted))
        cases.append(("generated_open_live_self_check_passes",
                      live["checks"].get("stored_closure_checks_consistency", {}).get("passed") is True))
        cases.append(("generated_open_snapshot_matches_live_substantive_checks",
                      persisted == live_substantive and generated_report["passed"]))
        cases.append(("generated_open_snapshot_has_no_closed_gate_detail",
                      all("Gate is CLOSED" not in str(item.get("detail", ""))
                          for item in persisted.values() if isinstance(item, dict))))
        corrupted = copy.deepcopy(generated)
        first_substantive = next(iter(corrupted["closure_checks"]))
        corrupted["closure_checks"][first_substantive]["detail"] = "corrupted persisted authorization"
        corrupted_path = Path(td) / "corrupted-gate-b.json"
        corrupted_path.write_text(json.dumps(corrupted))
        try:
            require_series_b_entry(corrupted_path)
            cases.append(("corrupted_substantive_snapshot_denied", False))
        except RuntimeError:
            cases.append(("corrupted_substantive_snapshot_denied", True))
    with tempfile.TemporaryDirectory(prefix="gate-b-guard-") as td:
        missing = Path(td) / "missing.json"
        try:
            require_series_b_entry(missing)
            cases.append(("missing_gate_denied", False))
        except RuntimeError:
            cases.append(("missing_gate_denied", True))
        malformed = Path(td) / "malformed.json"
        malformed.write_text("{")
        try:
            require_series_b_entry(malformed)
            cases.append(("malformed_gate_denied", False))
        except RuntimeError:
            cases.append(("malformed_gate_denied", True))

    def denied(name: str, changed: dict[str, Any], injected: dict[str, Any] | None = None) -> None:
        with tempfile.TemporaryDirectory(prefix="gate-b-case-") as td:
            path = Path(td) / "gate-b.json"
            path.write_text(json.dumps(changed))
            try:
                require_series_b_entry(path, injected=injected)
                cases.append((name, False))
            except RuntimeError:
                cases.append((name, True))

    closed = copy.deepcopy(gate); closed["decision"] = closed["final_status"] = closed["status"] = "CLOSED"
    with tempfile.TemporaryDirectory(prefix="gate-b-closed-") as td:
        closed_path = Path(td) / "gate-b.json"
        closed_path.write_text(json.dumps(closed))
        try:
            require_series_b_entry(closed_path)
            cases.append(("closed_gate_denied", False))
        except RuntimeError:
            cases.append(("closed_gate_denied", True))
    malformed_data = {"schema": "wrong"}
    denied("malformed_gate_record_denied", malformed_data)
    stale = copy.deepcopy(gate); stale["references"]["checkpoints"]["sha256"] = "0" * 64
    denied("stale_referenced_hash_denied", stale)
    changed = copy.deepcopy(gate); changed["checkpoint_provenance"]["qwen"]["revision"] = "0" * 40
    denied("changed_gate_checkpoint_provenance_denied", changed)
    changed = copy.deepcopy(gate); changed["required_targets"] = ["H0"]
    denied("changed_gate_required_targets_denied", changed)
    changed = copy.deepcopy(gate); changed["discovery_order"] = list(reversed(DISCOVERY_ORDER))
    denied("changed_gate_discovery_order_denied", changed)
    changed = copy.deepcopy(gate); changed["dflash_taps"] = [1]
    denied("changed_gate_taps_denied", changed)
    changed = copy.deepcopy(gate); changed["direct_bonsai_tap_counts"]["H2"] = 4
    denied("changed_gate_tap_counts_denied", changed)
    changed = copy.deepcopy(gate); changed["runtime"]["benchmark_method_sha256"] = "0" * 64
    denied("changed_gate_benchmark_method_denied", changed)
    changed = copy.deepcopy(gate); changed["runtime"]["measured_git_head"] = "0" * 40
    denied("changed_gate_measured_head_denied", changed)
    changed = copy.deepcopy(gate); changed["governing_inputs"]["constitution"]["version"] = "9.9.9"
    denied("changed_gate_governing_input_denied", changed)
    changed = copy.deepcopy(gate); changed["audit"]["validator"] = "other.py"
    denied("changed_gate_validator_identity_denied", changed)
    changed = copy.deepcopy(gate); changed["audit"]["validator_sha256"] = "0" * 64
    denied("changed_gate_validator_hash_denied", changed)
    changed = copy.deepcopy(gate); changed["decision"] = "CLOSED"
    denied("inconsistent_gate_decision_denied", changed)
    denied("changed_spec_denied", gate, {"spec_hash": "0" * 64})
    denied("changed_constitution_denied", gate, {"constitution_hash": "0" * 64})
    rh = current_runtime_hashes(); rh[next(iter(rh))] = "0" * 64
    denied("changed_runtime_source_denied", gate, {"runtime_hashes": rh})
    cpath = "specs/002-qwen-bonsai-hybrid-target/evidence/series-a/confirmation-schedule.json"
    schedule = read_json(REPO / cpath); schedule["run_count"] = 47
    denied("incomplete_confirmation_denied", gate, {"documents": {cpath: schedule}})
    discovery_doc = read_json(EVIDENCE / "series-a/discovery-index.json")
    run_ref = discovery_doc["targets"][0]
    run_path = _resolve_ref(run_ref, REPO)
    run_key = relpath_from_root(run_path, REPO)
    altered_run = read_json(run_path)
    altered_run["runtime"]["mlx_device"]["device_name"] = "Changed hardware"
    denied("changed_hardware_identity_denied", gate, {"documents": {run_key: altered_run}})
    altered_run = read_json(run_path)
    del altered_run["measured"]["speculative"]["decode_tokens_per_sec"]
    denied("missing_required_metric_denied", gate, {"documents": {run_key: altered_run}})
    dpath = "specs/002-qwen-bonsai-hybrid-target/evidence/series-a/repeat-decisions.json"
    decisions = read_json(REPO / dpath)
    decisions["comparisons"][0]["paired_decode_deltas"][0] += 0.25
    denied("corrupted_repeat_decision_arithmetic_denied", gate, {"documents": {dpath: decisions}})
    changed = copy.deepcopy(gate)
    changed["closure_checks"][next(iter(changed["closure_checks"]))]["detail"] = "stale stored PASS"
    denied("stale_closure_checks_denied", changed)
    # Exercise the on-disk failure boundary without touching the repository gate.
    with tempfile.TemporaryDirectory(prefix="gate-b-write-failure-") as td:
        isolated_gate = Path(td) / "gate-b.json"
        original_initial = initial_gate
        def explode() -> dict[str, Any]:
            raise RuntimeError("injected initialization failure")
        initial_gate = explode  # type: ignore[assignment]
        try:
            report = write_gate(isolated_gate)
            denied_state = read_json(isolated_gate)
            cases.append(("validation_exception_leaves_closed_gate",
                          not report["passed"] and denied_state.get("status") == "CLOSED"
                          and denied_state.get("decision") == "CLOSED"
                          and denied_state.get("final_status") == "CLOSED"))
        finally:
            initial_gate = original_initial
    changed = copy.deepcopy(gate); changed["schema"] = "wrong"
    denied("open_bad_schema_denied", changed)
    try:
        passed = require_series_b_entry()
        cases.append(("valid_open_gate_passes_without_model", passed["passed"] and "mlx.core" not in sys.modules))
    except Exception:
        cases.append(("valid_open_gate_passes_without_model", False))
    if not base["passed"]:
        cases.append(("gate_closure_valid", False))
    else:
        cases.append(("gate_closure_valid", True))
    return [{"case": name, "passed": ok} for name, ok in cases]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true", help="write CLOSED, validate, and open iff all checks pass")
    parser.add_argument("--entry-check", action="store_true", help="model-load-free Series-B entry validation")
    parser.add_argument("--self-test", action="store_true", help="run T030 fail-closed validator cases")
    args = parser.parse_args()
    if args.validate:
        report = write_gate()
    elif args.entry_check:
        try:
            report = require_series_b_entry()
        except RuntimeError as e:
            print(json.dumps({"passed": False, "error": str(e)}, indent=2))
            return 1
    elif args.self_test:
        results = self_test()
        print(json.dumps(results, indent=2))
        return 0 if all(x["passed"] for x in results) else 1
    else:
        report = validate()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
