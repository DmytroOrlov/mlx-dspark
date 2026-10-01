#!/usr/bin/env python3
"""Gate-first, Series-B-local orchestration and evidence helpers.

Every externally meaningful action starts by validating the live Gate B. Keep
this module free of model/runtime imports until after that guard succeeds.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
import shutil
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[4]
FEATURE = REPO / "specs/002-qwen-bonsai-hybrid-target"
EVIDENCE = FEATURE / "evidence"
SERIES_B = EVIDENCE / "series-b"
GATE_MODULE = EVIDENCE / "gate_b.py"
AUTHORIZED_TARGETS = ("H0", "H1c", "H2", "H3", "B0")
AUTHORIZED_DRAFTERS = ("B-Q", "B-B")
SERIES_A_METHOD_SHA256 = "688901863e6660ecef749c310d9f69ff8cfc7cf37db0137716ae1058bc1ff6d4"
BB_REPO = "naklitechie/Qwen3.8-27B-DFlash2-ternary-bonsai2"
BQ_REPO = "incoai/Qwen3.8-27B-DFlash2"
BQ_REVISION = "015e795645c74b1a0eeef3b570031fb62e769bc5"
BB_REVISION_LEAD = "3fc0d6ef43e933a20f1e9ee53fac6f56402c0f83"
BB_WEIGHTS_SHA256_LEAD = "eb141d0fb8621cf7a7c81047b83810fa19b8093543bd05529e9478d9555cb8f1"
AUTHORIZATION_ARTIFACTS = (
    "specs/002-qwen-bonsai-hybrid-target/evidence/probes/series_b.py",
    "tests/test_series_b.py",
    "specs/002-qwen-bonsai-hybrid-target/evidence/series-b/matrix-decision.json",
    "specs/002-qwen-bonsai-hybrid-target/evidence/series-b/checkpoints.json",
    "specs/002-qwen-bonsai-hybrid-target/evidence/series-b/drafter-smoke.json",
    "specs/002-qwen-bonsai-hybrid-target/evidence/series-b/runner.json",
    "specs/002-qwen-bonsai-hybrid-target/evidence/series-b/matrix.json",
    "specs/002-qwen-bonsai-hybrid-target/evidence/series-b/control-manifest.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def benchmark_method_hash(source: bytes) -> str:
    tree = ast.parse(source)
    fn = next((node for node in tree.body if isinstance(node, ast.FunctionDef)
               and node.name == "run_series_condition"), None)
    if fn is None:
        raise ValueError("run_series_condition benchmark method is missing")
    segment = ast.get_source_segment(source.decode(), fn)
    if segment is None:
        raise ValueError("could not extract run_series_condition source")
    return hashlib.sha256(segment.encode()).hexdigest()


def finite_parameter_inventory(model: Any) -> tuple[list[dict[str, Any]], list[str]]:
    """Count and check loaded tensors without importing MLX before gate entry."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    flat = [(str(name), value) for name, value in tree_flatten(model.parameters())
            if hasattr(value, "dtype")]
    checks = [mx.all(mx.isfinite(value)) for _, value in flat]
    mx.eval(checks)
    bad = [name for (name, _), check in zip(flat, checks) if not bool(check.item())]
    return ([{"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
             for name, value in flat], bad)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def require_entry(gate_path: Path | None = None) -> dict[str, Any]:
    """Invoke the authoritative model-load-free Gate-B entry guard."""
    spec = importlib.util.spec_from_file_location("series_b_gate_b", GATE_MODULE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Gate-B validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.require_series_b_entry(gate_path or module.GATE)


def series_a_decision() -> dict[str, Any]:
    adjudication_path = EVIDENCE / "adjudication/series-a.json"
    decisions_path = EVIDENCE / "series-a/repeat-decisions.json"
    adjudication = json.loads(adjudication_path.read_text())
    decisions = json.loads(decisions_path.read_text())
    if adjudication.get("status") != "PASS" or decisions.get("status") != "PASS":
        raise RuntimeError("Series-A adjudication and repeat decisions must both PASS")
    outcomes = {row.get("comparison_id"): row.get("adjudicated_result")
                for row in decisions.get("comparisons", [])}
    h1c_vs_h1a = outcomes.get("H1c-vs-H1a")
    h1c_vs_h1b = outcomes.get("H1c-vs-H1b")
    if h1c_vs_h1a != "confirmed_win" or h1c_vs_h1b != "within_noise":
        raise RuntimeError("Series-A evidence does not support the specified exception decision")
    interpretation = adjudication.get("interpretation", {}).get("h1c_interaction", {})
    if (interpretation.get("h1c_vs_h1a") != h1c_vs_h1a
            or interpretation.get("h1c_vs_h1b") != h1c_vs_h1b):
        raise RuntimeError("Series-A adjudication/repeat decision mismatch")
    exception_authorized = (
        h1c_vs_h1a == "confirmed_win" and h1c_vs_h1b == "confirmed_regression"
    )
    cells = [{"target": target, "drafter": drafter}
             for target in AUTHORIZED_TARGETS for drafter in AUTHORIZED_DRAFTERS]
    if exception_authorized:
        raise RuntimeError("isolated-block exception requires a separately authorized matrix")
    return {
        "series_a": {"adjudication": {"path": str(adjudication_path.relative_to(REPO)),
                                         "sha256": sha256_file(adjudication_path)},
                     "repeat_decisions": {"path": str(decisions_path.relative_to(REPO)),
                                          "sha256": sha256_file(decisions_path)}},
        "decision": {
            "h1c_vs_h1a": h1c_vs_h1a,
            "h1c_vs_h1b": h1c_vs_h1b,
            "isolated_block_exception_authorized": False,
            "reason": ("H1c>H1a is confirmed, while H1c-vs-H1b is within_noise; "
                       "therefore neither H1a nor H1b is proven the unambiguous best "
                       "Series-A hybrid."),
        },
        "cells": cells,
    }


def validate_matrix_decision(decision: dict[str, Any], matrix: dict[str, Any]) -> None:
    """Recompute T031's frozen choice and bind it to the current matrix."""
    expected = series_a_decision()
    expected_pairs = {(row["target"], row["drafter"]) for row in expected["cells"]}
    matrix_pairs = {(row.get("target"), row.get("drafter")) for row in matrix.get("cells", [])}
    actual_pairs = {(row.get("target"), row.get("drafter")) for row in decision.get("cells", [])}
    if (decision.get("status") != "PASS" or decision.get("cell_count") != 10
            or decision.get("decision") != expected["decision"]
            or decision.get("series_a") != expected["series_a"]
            or actual_pairs != expected_pairs or matrix_pairs != expected_pairs
            or len(decision.get("cells", [])) != 10
            or decision.get("cells") != expected["cells"]
            or decision.get("model_load_performed") is not False
            or decision.get("performance_run_performed") is not False):
        raise RuntimeError("T038 execution denied: matrix-decision is stale or disagrees with recomputed Series-A decision/matrix")


def write_matrix_decision() -> dict[str, Any]:
    gate = require_entry()
    derived = series_a_decision()
    result = {
        "schema": "qwen-bonsai-series-b-matrix-decision/v1",
        "status": "PASS",
        "gate_b": {"decision": "OPEN", "validated": gate.get("passed") is True},
        **derived,
        "cell_count": len(derived["cells"]),
        "model_load_performed": False,
        "performance_run_performed": False,
    }
    if result["cell_count"] != 10:
        raise RuntimeError("Series-B core matrix must contain exactly ten cells")
    atomic_json(SERIES_B / "matrix-decision.json", result)
    return result


def resolve_checkpoint_provenance(gate_path: Path | None = None) -> dict[str, Any]:
    """Resolve the two pinned Series-B drafters after the live Gate-B guard."""
    gate = require_entry(gate_path)
    # Keep hub code and all network/filesystem checkpoint access below the guard.
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    info = api.model_info(BB_REPO, files_metadata=True)
    revision = info.sha
    if not isinstance(revision, str) or len(revision) != 40 or any(
            c not in "0123456789abcdef" for c in revision.lower()):
        raise RuntimeError("checkpoint resolver did not return an immutable 40-hex revision")
    if not info.siblings:
        raise RuntimeError("resolved B-B revision has no file inventory")
    files = {item.rfilename: item for item in info.siblings}
    indexes = sorted(name for name in files if name.endswith(".safetensors.index.json"))
    indexes += sorted(name for name in files if name.endswith(".bin.index.json"))
    snapshot_patterns = ["config.json", *indexes]
    index_documents: dict[str, dict[str, Any]] = {}
    provisional = snapshot_download(repo_id=BB_REPO, revision=revision,
                                   allow_patterns=snapshot_patterns)
    for index_name in indexes:
        index = json.loads((Path(provisional) / index_name).read_text())
        names = sorted(set(index.get("weight_map", {}).values()))
        if not names or any(name not in files for name in names):
            raise RuntimeError(f"invalid or incomplete weight index: {index_name}")
        index_documents[index_name] = index
        snapshot_patterns.extend(names)
    if indexes:
        selected_weights = sorted({name for doc in index_documents.values()
                                   for name in doc.get("weight_map", {}).values()})
    else:
        selected_weights = sorted(name for name in files
                                  if name.endswith((".safetensors", ".bin", ".npz")))
    if not selected_weights:
        raise RuntimeError("resolved B-B checkpoint has no indexed/recognized weight files")
    # LFS SHA-256 and size are returned by the immutable-revision Hub file
    # inventory. Fetch only config/index metadata here; avoid downloading multi-GB
    # weights solely to repeat the content hash already supplied by LFS.
    snapshot_patterns = sorted(set(snapshot_patterns))
    snapshot = Path(snapshot_download(repo_id=BB_REPO, revision=revision,
                                      allow_patterns=snapshot_patterns)).resolve()
    config_path = snapshot / "config.json"
    if not config_path.is_file():
        raise RuntimeError("resolved B-B snapshot lacks config.json")
    config = json.loads(config_path.read_text())
    weight_inventory = []
    for name in selected_weights:
        metadata = files[name]
        expected_size = getattr(metadata, "size", None)
        lfs = getattr(metadata, "lfs", None)
        lfs_sha = getattr(lfs, "sha256", None)
        if not isinstance(expected_size, int) or expected_size <= 0:
            raise RuntimeError(f"selected B-B weight has no authoritative size: {name}")
        if not isinstance(lfs_sha, str) or len(lfs_sha) != 64:
            raise RuntimeError(f"selected B-B weight has no strong LFS SHA-256: {name}")
        weight_inventory.append({
            "path": name,
            "size_bytes": expected_size,
            "sha256": lfs_sha,
            "hub_lfs_sha256": lfs_sha,
            "fingerprint_method": "Hub LFS SHA-256 for this selected file at the immutable commit; file size from the same revision-bound file metadata",
        })
    config_sha = sha256_file(config_path)
    inventory_hash = hashlib.sha256(json.dumps(
        [{"path": x["path"], "size_bytes": x["size_bytes"], "sha256": x["sha256"]}
         for x in weight_inventory], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    revision_match = revision == BB_REVISION_LEAD
    weight_match = (BB_WEIGHTS_SHA256_LEAD in {x["sha256"] for x in weight_inventory}
                    or inventory_hash == BB_WEIGHTS_SHA256_LEAD)
    bq_files = api.list_repo_files(BQ_REPO, revision=BQ_REVISION)
    if "config.json" not in bq_files:
        raise RuntimeError("accepted B-Q revision lacks config.json")
    bq_snapshot = Path(snapshot_download(repo_id=BQ_REPO, revision=BQ_REVISION,
                                         allow_patterns=["config.json"])).resolve()
    bq_config_path = bq_snapshot / "config.json"
    if not bq_config_path.is_file():
        raise RuntimeError("accepted B-Q config could not be retrieved")
    bq_info = api.model_info(BQ_REPO, revision=BQ_REVISION, files_metadata=True)
    bq_revision = bq_info.sha
    if bq_revision != BQ_REVISION:
        raise RuntimeError("B-Q resolver did not preserve the accepted immutable revision")
    accepted_checkpoints = json.loads((EVIDENCE / "checkpoints.json").read_text())
    accepted_bq = accepted_checkpoints["checkpoints"]["dflash"]
    if (accepted_bq.get("repo_id") != BQ_REPO
            or accepted_bq.get("revision") != BQ_REVISION
            or accepted_bq.get("config", {}).get("sha256") != sha256_file(bq_config_path)):
        raise RuntimeError("B-Q provenance differs from immutable accepted Series-A identity")
    bq_weights = [{
        "path": item["name"], "size_bytes": item["size_bytes"],
        "sha256": item["sha256"],
        "fingerprint_method": item["fingerprint_method"],
        "source": "immutable Series-A checkpoints.json accepted provenance",
    } for item in accepted_bq.get("weights", [])]
    if not bq_weights:
        raise RuntimeError("accepted B-Q provenance has no selected weight inventory")
    result = {
        "schema": "qwen-bonsai-series-b-checkpoints/v1",
        "status": "PASS",
        "gate_b": {"validated": gate.get("passed") is True, "decision": "OPEN"},
        "B-B": {
            "repo_id": BB_REPO,
            "resolved_revision": revision,
            "revision_source": "Hub repository default revision resolved once to immutable commit SHA; all subsequent file access pinned to that SHA",
            "snapshot_path": str(snapshot),
            "config": {"path": "config.json", "sha256": config_sha,
                       "selected_config": config},
            "indexes": [{"path": name, "sha256": sha256_file(snapshot / name),
                         "selected_weight_files": sorted(set(doc.get("weight_map", {}).values()))}
                        for name, doc in sorted(index_documents.items())],
            "selected_weight_files": weight_inventory,
            "weight_inventory_sha256": inventory_hash,
            "provenance_leads": {
                "revision": {"lead": BB_REVISION_LEAD,
                             "comparison": "MATCHED" if revision_match else "DID NOT MATCH"},
                "weights_sha256": {"lead": BB_WEIGHTS_SHA256_LEAD,
                                   "comparison": "MATCHED" if weight_match else "DID NOT MATCH",
                                   "comparison_method": "lead compared with each selected file SHA-256 and deterministic inventory SHA-256"},
            },
        },
        "B-Q": {
            "repo_id": BQ_REPO,
            "resolved_revision": bq_revision,
            "snapshot_path": accepted_bq["resolved_path"],
            "config": {"path": "config.json", "sha256": sha256_file(bq_config_path),
                       "selected_config": json.loads(bq_config_path.read_text())},
            "selected_weight_files": bq_weights,
            "weight_inventory_source": "immutable Series-A checkpoints.json, cross-checked against exact B-Q revision and config SHA-256",
            "revision_source": "accepted immutable Series-A checkpoint revision",
        },
        "checkpoint_resolution_performed": True,
        "performance_run_performed": False,
    }
    atomic_json(SERIES_B / "checkpoints.json", result)
    return result


def fetch_verified_bb_weights(gate_path: Path | None = None) -> dict[str, Any]:
    """Download the selected B-B weight file at its immutable revision and verify bytes."""
    require_entry(gate_path)
    provenance_path = SERIES_B / "checkpoints.json"
    provenance = json.loads(provenance_path.read_text())
    bb = provenance.get("B-B", {})
    if provenance.get("status") != "PASS" or bb.get("repo_id") != BB_REPO:
        raise RuntimeError("T032 provenance is not PASS for the exact B-B repo")
    revision = bb.get("resolved_revision")
    weights = bb.get("selected_weight_files")
    if not isinstance(revision, str) or len(revision) != 40 or not weights:
        raise RuntimeError("T032 provenance lacks an immutable revision or selected weights")
    if os.environ.get("HF_HUB_DISABLE_XET") is None:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import hf_hub_download, snapshot_download

    files = []
    for item in weights:
        path = Path(hf_hub_download(repo_id=BB_REPO, filename=item["path"], revision=revision))
        actual_size = path.stat().st_size
        actual_sha = sha256_file(path)
        if actual_size != item["size_bytes"] or actual_sha != item["sha256"]:
            raise RuntimeError(f"downloaded immutable B-B weight integrity mismatch: {item['path']}")
        files.append({"path": item["path"], "local_path": str(path.resolve()),
                      "size_bytes": actual_size, "sha256": actual_sha,
                      "fingerprint_method": "SHA-256 over complete downloaded file bytes"})
    snapshot = Path(snapshot_download(repo_id=BB_REPO, revision=revision,
                                      allow_patterns=["config.json", *[x["path"] for x in weights]])).resolve()
    for item in weights:
        snapshot_file = snapshot / item["path"]
        if not snapshot_file.is_file() or sha256_file(snapshot_file) != item["sha256"]:
            raise RuntimeError(f"snapshot link is incomplete or changed: {item['path']}")
    bb["verified_downloaded_weight_files"] = files
    bb["snapshot_path"] = str(snapshot)
    provenance["B-B"] = bb
    provenance["weights_downloaded_and_verified"] = True
    atomic_json(provenance_path, provenance)
    return {"status": "PASS", "files": files}


def drafter_compatibility_smoke(gate_path: Path | None = None) -> dict[str, Any]:
    """Minimal real B-B + ordinary Qwen H0 load/forward/cache smoke; never benchmarks."""
    gate = require_entry(gate_path)
    checkpoints_path = SERIES_B / "checkpoints.json"
    provenance = json.loads(checkpoints_path.read_text())
    bb = provenance.get("B-B", {})
    verified_files = bb.get("verified_downloaded_weight_files", [])
    if (provenance.get("status") != "PASS" or provenance.get("weights_downloaded_and_verified") is not True
            or not verified_files):
        raise RuntimeError("T033 denied: exact T032 B-B weight bytes have not been verified locally")
    if any(not Path(row["local_path"]).is_file() for row in verified_files):
        raise RuntimeError("T033 denied: a verified B-B weight file is no longer available")
    accepted = json.loads((EVIDENCE / "checkpoints.json").read_text())
    qwen = accepted["checkpoints"]["qwen"]
    if qwen.get("revision") != "10c35caafbb80f7dc6a7a432cdd11af10a6d4818":
        raise RuntimeError("T033 denied: immutable Series-A Qwen target identity changed")
    sys.path.insert(0, str(REPO / "src"))
    import mlx.core as mx
    from mlx_dspark.load import load_dflash, load_target
    from mlx_dspark.generate import encode_messages

    target = None
    try:
        # Use the existing target and drafter loaders directly. Engine.load("auto")
        # performs cost-curve calibration; the T033 smoke must not measure those curves.
        target, tokenizer = load_target(qwen["resolved_path"], require_tap=True, kv_bits=None)
        drafter, _ = load_dflash(bb["snapshot_path"], quantize=True, bits=4)
        drafter.bind(target.model)
        taps_expected = list(bb["config"]["selected_config"]["dflash_config"]["target_layer_ids"])
        taps_loaded = list(drafter.config.target_layer_ids)
        if taps_loaded != taps_expected:
            raise RuntimeError(f"loaded B-B taps differ from exact checkpoint config: {taps_loaded}")
        target_inventory, target_nonfinite = finite_parameter_inventory(target.model)
        drafter_inventory, drafter_nonfinite = finite_parameter_inventory(drafter)
        if target_nonfinite or drafter_nonfinite:
            raise RuntimeError("loaded target or B-B drafter contains non-finite parameters")
        corpus_path = EVIDENCE / "series-a/prompts.json"
        corpus_index = json.loads((EVIDENCE / "series-a/discovery-index.json").read_text())
        if sha256_file(corpus_path) != corpus_index["prompt_corpus"]["sha256"]:
            raise RuntimeError("accepted Series-A prompt corpus hash changed")
        corpus = json.loads(corpus_path.read_text())
        encoded_prompts = []
        for item in corpus["prompts"]:
            ids = [int(x) for x in encode_messages(
                tokenizer, [{"role": "user", "content": item["text"]}],
                enable_thinking=False)]
            encoded_prompts.append({"id": item["id"], "input_ids": ids,
                                    "input_ids_sha256": hashlib.sha256(json.dumps(
                                        ids, separators=(",", ":")).encode()).hexdigest()})
        prompt_ids = encode_messages(tokenizer,
                                     [{"role": "user", "content": "Say ready."}],
                                     enable_thinking=False)
        if not prompt_ids:
            raise RuntimeError("compatibility prompt encoded to no tokens")
        prompt = mx.array([[int(prompt_ids[-1])]], dtype=mx.int32)
        cache = target.make_cache()
        cache_before = [{"index": i, "class": type(row).__name__,
                         "offset": getattr(row, "offset", None)}
                        for i, row in enumerate(cache)]
        logits, tap = target.prefill(prompt, cache, tap=taps_loaded, want_logits=True)
        mx.eval(logits, tap)
        if not bool(mx.all(mx.isfinite(logits)).item()) or not bool(mx.all(mx.isfinite(tap)).item()):
            raise RuntimeError("target prefill logits or B-B tap output is non-finite")
        tap_shape = list(tap.shape)
        expected_width = len(taps_loaded) * int(bb["config"]["selected_config"]["hidden_size"])
        if tap_shape[-1] != expected_width:
            raise RuntimeError(f"B-B tap width {tap_shape[-1]} != expected {expected_width}")
        verify_tokens = [int(prompt_ids[-1]), int(prompt_ids[-2] if len(prompt_ids) > 1 else prompt_ids[-1]),
                         int(prompt_ids[-3] if len(prompt_ids) > 2 else prompt_ids[-1])]
        verify_ids = mx.array([verify_tokens], dtype=mx.int32)
        verify_logits, verify_tap = target.verify(verify_ids, cache, taps_loaded)
        mx.eval(verify_logits, verify_tap)
        if not bool(mx.all(mx.isfinite(verify_logits)).item()) or not bool(mx.all(mx.isfinite(verify_tap)).item()):
            raise RuntimeError("target verify logits or B-B tap output is non-finite")
        cache_after_verify = [{"index": i, "class": type(row).__name__,
                               "offset": getattr(row, "offset", None)}
                              for i, row in enumerate(cache)]
        target.rollback(cache, n_rejected=1, accepted=[verify_tokens[1]])
        cache_after_rollback = [{"index": i, "class": type(row).__name__,
                                 "offset": getattr(row, "offset", None)}
                                for i, row in enumerate(cache)]
        before_offsets = [row["offset"] for row in cache_before if row["offset"] is not None]
        verified_offsets = [row["offset"] for row in cache_after_verify if row["offset"] is not None]
        rolled_offsets = [row["offset"] for row in cache_after_rollback if row["offset"] is not None]
        cache_advanced = bool(before_offsets and verified_offsets and max(verified_offsets) > max(before_offsets))
        rollback_valid = bool(verified_offsets and rolled_offsets
                              and max(rolled_offsets) == max(verified_offsets) - 1
                              and max(rolled_offsets) > max(before_offsets))
        if not cache_advanced or not rollback_valid:
            raise RuntimeError("plain-KV cache advancement or rejected-suffix rollback failed")
        drafter_cache = drafter.make_cache()
        masked = mx.array([[int(drafter.config.mask_token_id)] * int(verify_ids.shape[1])],
                          dtype=mx.int32)
        draft_logits = drafter(masked, verify_tap, drafter_cache)
        mx.eval(draft_logits)
        if not bool(mx.all(mx.isfinite(draft_logits)).item()):
            raise RuntimeError("B-B drafter forward logits are non-finite")
        if any("quant" in row["class"].lower() for row in cache_before):
            raise RuntimeError("B-B smoke instantiated a non-plain KV cache")
        controller_source = REPO / "src/mlx_dspark/calibrate.py"
        series_a_runner = (EVIDENCE / "series-a/runner-used-for-discovery.py").read_bytes()
        benchmark_method = benchmark_method_hash(series_a_runner)
        if benchmark_method != SERIES_A_METHOD_SHA256:
            raise RuntimeError("accepted speculative benchmark method fingerprint changed")
        return {
            "schema": "qwen-bonsai-series-b-drafter-smoke/v1", "status": "PASS",
            "gate_b": {"validated": gate.get("passed") is True, "decision": "OPEN"},
            "checkpoint": {"repo_id": bb["repo_id"], "revision": bb["resolved_revision"],
                           "snapshot_path": bb["snapshot_path"],
                           "weight_files": verified_files},
            "loaded_dflash": {"target_layer_ids": taps_loaded,
                               "tap_semantics": "zero-based block output; the runtime returns each selected block output in the fused tap hidden-state tensor",
                               "tap_shape": tap_shape, "expected_hidden_width": expected_width,
                               "tap_width_compatible": True},
            "loaded_parameters": {"target_parameter_tensor_count": len(target_inventory),
                                  "drafter_parameter_tensor_count": len(drafter_inventory),
                                  "all_finite": True},
            "forward": {"target_prefill_finite": True, "target_verify_finite": True,
                        "drafter_forward_finite": True,
                        "drafter_logits_shape": list(draft_logits.shape)},
            "cache": {"plain_kv": True, "advanced": cache_advanced,
                      "rejected_suffix_rollback": rollback_valid, "rejected_suffix_tokens": 1,
                      "before": cache_before, "after_verify": cache_after_verify,
                      "after_rollback": cache_after_rollback},
            "controls": {"ordinary_production_cap_controller_implementation": "mlx_dspark.calibrate.CapController",
                          "cap_controller_source_sha256": sha256_file(controller_source),
                          "cap_controller_instantiated_in_smoke": False,
                          "reason_not_instantiated": "Avoid load-time cost-curve calibration; the smoke uses existing loader/target paths directly and does not exercise or retune generation policy.",
                          "width_policy_present": False,
                          "speculative_policy": "not exercised; accepted Series-A Engine benchmark method fingerprint is unchanged and no speculative settings are modified",
                          "benchmark_method_sha256": benchmark_method,
                          "kv_bits": None},
            "prompt_corpus": {"path": str(corpus_path.relative_to(REPO)),
                              "sha256": sha256_file(corpus_path),
                              "tokenized_prompts": encoded_prompts},
            "compatibility_adapter": "none",
            "performance_measurement_performed": False,
            "execution_history": {
                "final_smoke": "direct target/drafter loaders only; no rate or timing metric collected",
                "discarded_attempt": ("An earlier smoke attempt used Engine.load(max_draft_tokens=None) and reached a later assertion. That production path can invoke static_cap cost calibration; no calibration result, timing, or throughput metric was retained. The final PASS record is from the direct-loader smoke above."),
                "series_b_matrix_performance_measurement_performed": False,
            },
        }
    finally:
        if target is not None:
            del target


def validate_result_integrity(result: dict[str, Any], *, expected_target: str | None = None,
                              expected_drafter: str | None = None,
                              expected_cell: dict[str, Any] | None = None,
                              matrix: dict[str, Any] | None = None,
                              control_manifest: dict[str, Any] | None = None) -> None:
    """Reject malformed, non-finite, non-provenanced, or failed-integrity run records."""
    if result.get("schema") != "qwen-bonsai-series-b-run/v1" or result.get("series") != "B":
        raise ValueError("result schema/series is not a Series-B run")
    if result.get("status") != "PASS":
        raise ValueError("result status is not PASS")
    target = expected_target if expected_target is not None else result.get("variant")
    drafter = expected_drafter if expected_drafter is not None else result.get("drafter_arm")
    if (target not in AUTHORIZED_TARGETS or drafter not in AUTHORIZED_DRAFTERS
            or result.get("variant") != target or result.get("drafter_arm") != drafter):
        raise ValueError("result target/drafter does not match the requested frozen matrix cell")
    integrity = result.get("integrity", {})
    if (integrity.get("target_integrity_passed") is not True
            or integrity.get("all_generated_outputs_nonempty") is not True
            or integrity.get("serial_and_spec_prompt_ids_match") is not True):
        raise ValueError("result failed a required integrity check")
    checkpoints = result.get("checkpoints", {})
    dflash = checkpoints.get("dflash", {})
    if (not dflash.get("repo_id") or not dflash.get("revision")
            or len(dflash.get("revision", "")) != 40 or not dflash.get("weights")):
        raise ValueError("result lacks immutable drafter checkpoint provenance")
    if expected_cell is not None:
        expected_cp = expected_cell.get("drafter_checkpoint", {})
        if (dflash.get("repo_id") != expected_cp.get("repo_id")
                or dflash.get("revision") != expected_cp.get("revision")
                or dflash.get("config_sha256") != expected_cp.get("config_sha256")
                or _weight_fingerprints(dflash.get("weights", [])) !=
                _weight_fingerprints(expected_cp.get("weight_files", []))):
            raise ValueError("result DFlash checkpoint/config/weight provenance differs from requested arm")
        if matrix is None or control_manifest is None:
            raise ValueError("requested-cell validation requires frozen matrix and control manifest")
        _validate_target_provenance(result, target, matrix["target_identities"][target])
        _validate_frozen_controls(result, control_manifest)
    measured = result.get("measured", {})
    serial = measured.get("serial_target", {})
    speculative = measured.get("speculative", {})
    required_values = {
        "decode_duration_seconds": measured.get("decode_duration_seconds"),
        "serial_target_decode_seconds": serial.get("decode_seconds"),
        "serial_target_tokens_per_sec": serial.get("tokens_per_sec"),
        "serial_target_forwards": serial.get("target_forwards"),
        "serial_generated_tokens": serial.get("generated_tokens"),
        "speculative_decode_seconds": speculative.get("decode_seconds"),
        "decode_tokens_per_sec": speculative.get("decode_tokens_per_sec"),
        "speedup_over_serial": speculative.get("speedup_over_serial"),
        "draft_acceptance": speculative.get("draft_acceptance"),
        "mean_accept_len": speculative.get("mean_accept_len"),
        "target_forwards": speculative.get("target_forwards"),
        "speculative_generated_tokens": speculative.get("generated_tokens"),
        "generated_tokens_per_target_forward": speculative.get("generated_tokens_per_target_forward"),
        "generated_tokens": measured.get("generated_token_count"),
        "measurement_duration_seconds": measured.get("measurement_duration_seconds"),
        "peak_steady_state_memory_bytes": measured.get("peak_steady_state_memory", {}).get(
            "peak_bytes_after_load_warmup_reset"),
    }
    missing = [key for key, value in required_values.items() if value is None]
    if missing:
        raise ValueError(f"result missing required metrics: {missing}")
    for key, value in required_values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"result metric {key} is not finite numeric data")
    if (required_values["serial_target_tokens_per_sec"] <= 0
            or required_values["decode_tokens_per_sec"] <= 0
            or required_values["generated_tokens"] <= 0
            or required_values["decode_duration_seconds"] <= 0
            or required_values["serial_target_decode_seconds"] <= 0
            or required_values["speculative_decode_seconds"] <= 0
            or required_values["peak_steady_state_memory_bytes"] <= 0
            or required_values["measurement_duration_seconds"] <= 0):
        raise ValueError("throughput, token-count, and duration metrics must be positive")
    for key in ("serial_target_forwards", "serial_generated_tokens", "target_forwards",
                "speculative_generated_tokens", "generated_tokens"):
        value = required_values[key]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"result metric {key} must be a positive integer")
    if not isinstance(measured.get("component_timing"), (dict, type(None))):
        raise ValueError("result component timing must use the Series-A dict-or-null representation")
    if measured.get("component_timing") is None and not isinstance(
            measured.get("component_timing_unavailable_reason"), str):
        raise ValueError("result must state why Series-A component timing is unavailable")
    speculative = measured.get("speculative", {})
    for key in ("accept_histogram", "draft_width_distribution", "cap_distribution"):
        distribution = speculative.get(key)
        if not isinstance(distribution, dict) or not distribution:
            raise ValueError(f"result missing required metrics: {key}")
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or value < 0 for value in distribution.values()):
            raise ValueError(f"result metric {key} has invalid distribution values")
    accept_lengths = speculative.get("accept_lengths")
    if (not isinstance(accept_lengths, list) or not accept_lengths
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) or value < 0 or value > 8
                   for value in accept_lengths)
            or not 0 <= required_values["mean_accept_len"] <= 8
            or not 0 <= required_values["draft_acceptance"] <= 1):
        raise ValueError("result acceptance/mean-accept-length metrics are outside the Series-A range")
    if (speculative.get("generated_tokens") != measured.get("generated_token_count")
            or not math.isclose(required_values["speedup_over_serial"],
                                required_values["decode_tokens_per_sec"] /
                                required_values["serial_target_tokens_per_sec"], rel_tol=1e-6)
            or not math.isclose(required_values["generated_tokens_per_target_forward"],
                                speculative.get("generated_tokens") /
                                speculative.get("target_forwards"), rel_tol=1e-6)):
        raise ValueError("result metric arithmetic disagrees with Series-A counts/rates")
    per_prompt = speculative.get("per_prompt")
    if not isinstance(per_prompt, list) or not per_prompt:
        raise ValueError("result missing required metrics: speculative per_prompt")
    for row in per_prompt:
        if (not isinstance(row.get("prompt_ids_sha256"), str)
                or not row.get("prompt_ids_sha256")
                or not isinstance(row.get("generated_token_count"), int)
                or isinstance(row.get("generated_token_count"), bool)
                or row["generated_token_count"] <= 0
                or not isinstance(row.get("decode_seconds"), (int, float))
                or not math.isfinite(row["decode_seconds"]) or row["decode_seconds"] <= 0
                or not isinstance(row.get("decode_tokens_per_sec"), (int, float))
                or not math.isfinite(row["decode_tokens_per_sec"])
                or row["decode_tokens_per_sec"] <= 0):
            raise ValueError("result speculative per_prompt metric is missing or invalid")


def _weight_fingerprints(rows: list[dict[str, Any]]) -> list[tuple[str, int, str]]:
    return sorted((str(row.get("name") or row.get("path") or "").rsplit("/", 1)[-1],
                   row.get("size_bytes"), row.get("sha256", "")) for row in rows)


def _validate_target_provenance(result: dict[str, Any], target: str,
                                identity: dict[str, Any]) -> None:
    checkpoint_name = "bonsai" if target == "B0" else "qwen"
    actual = result.get("checkpoints", {}).get(checkpoint_name, {})
    for field in ("repo_id", "revision", "config_sha256"):
        if actual.get(field) != identity.get(field):
            raise ValueError(f"result target checkpoint {checkpoint_name}.{field} differs from frozen target identity")
    if _weight_fingerprints(actual.get("weights", [])) != _weight_fingerprints(identity.get("weight_files", [])):
        raise ValueError("result target checkpoint weights differ from frozen target identity")
    expected_donor = identity.get("donor_checkpoint")
    if target != "B0":
        donor = result.get("checkpoints", {}).get("bonsai", {})
        if (not expected_donor or donor.get("repo_id") != expected_donor.get("repo_id")
                or donor.get("revision") != expected_donor.get("revision")
                or donor.get("config_sha256") != expected_donor.get("config_sha256")
                or _weight_fingerprints(donor.get("weights", [])) !=
                _weight_fingerprints(expected_donor.get("weight_files", []))):
            raise ValueError("result donor checkpoint provenance differs from frozen target identity")
    composition = result.get("target_composition", {})
    ownership = identity.get("ownership", {})
    for field in ("block_owners", "embedding_owner", "final_norm_owner", "lm_head_owner"):
        if composition.get(field) != ownership.get(field):
            raise ValueError(f"result target identity/ownership mismatch for {field}")
    expected_donors = (list(range(64)) if target == "B0" else
                       expected_donor.get("donor_block_indices", []))
    if composition.get("donor_block_indices") != expected_donors:
        raise ValueError("result target composition donor ownership differs from frozen target identity")


def _validate_frozen_controls(result: dict[str, Any], manifest: dict[str, Any]) -> None:
    actual = result.get("controls", {})
    expected = manifest.get("common_controls", {})
    fixed = expected.get("fixed_engine_settings", {})
    direct = {"plain_kv": expected.get("plain_kv"), "kv_bits": expected.get("kv_bits"),
              "generation": expected.get("generation"), "measurement_protocol": expected.get(
                  "measurement_protocol"), "warmup": expected.get("measurement_protocol", {}).get("warmup"),
              "memory_guard": expected.get("measurement_protocol", {}).get("memory_guard")}
    if (actual.get("plain_kv") != direct["plain_kv"] or actual.get("kv_bits") != direct["kv_bits"]
            or actual.get("generation") != direct["generation"]
            or actual.get("width_policy", False) is not expected.get("width_policy", False)
            or actual.get("warmup") != direct["warmup"]
            or actual.get("memory_guard") != direct["memory_guard"]
            or actual.get("measurement_protocol") != (
                "one fresh process and Engine per run; Engine load warmup; serial greedy then DFlash speculative on same target; no prefix/KV reuse between prompts")):
        raise ValueError("result prompt/control/runtime provenance differs from frozen control manifest")
    expected_prompt_hashes = [row.get("input_ids_sha256") for row in
                              expected.get("prompt_corpus", {}).get("tokenized_prompts", [])]
    actual_prompt_hashes = [row.get("input_ids_sha256") for row in actual.get("prompt_ids", [])]
    if (actual_prompt_hashes != expected_prompt_hashes
            or actual.get("prompt_corpus", {}).get("sha256") !=
            expected.get("prompt_corpus", {}).get("sha256")):
        raise ValueError("result prompt/control/runtime provenance differs from frozen control manifest")
    for field in ("mode", "drafter_bits", "lookup_drafts", "prefix_cache", "small_m",
                  "sdpa_split", "wide_gemm_min", "cpu_split"):
        if actual.get(field) != fixed.get(field):
            raise ValueError(f"result fixed engine control {field} differs from frozen control manifest")
    if actual.get("controller") != expected.get("controller"):
        raise ValueError("result controller differs from frozen control manifest")
    runtime = result.get("runtime", {})
    runtime_hashes = runtime.get("runtime_source_sha256", {})
    expected_runtime = expected.get("runtime_source_sha256", {})
    if ({name: runtime_hashes.get(name) for name in expected_runtime} != expected_runtime
            or runtime.get("mlx_device") != expected.get("hardware")):
        raise ValueError("result runtime provenance differs from frozen control manifest")
    method_hash = expected.get("measurement_method", {}).get("run_series_condition_ast_sha256")
    if method_hash != SERIES_A_METHOD_SHA256:
        raise ValueError("frozen control manifest benchmark-method fingerprint is not the accepted Series-A method")


def validate_discovery_schedule(matrix: dict[str, Any], order_index: int,
                                target: str, drafter: str,
                                comparison_group: str) -> None:
    if comparison_group != "discovery":
        return
    schedule = matrix.get("schedule", {}).get("rows", [])
    scheduled = next((row for row in schedule if row.get("order_index") == order_index), None)
    if (scheduled is None or scheduled.get("target") != target
            or scheduled.get("drafter") != drafter):
        raise RuntimeError("T038 execution denied: requested discovery order_index/target/drafter does not match the frozen schedule row")


def validate_t038_prerequisites(validation: dict[str, Any], checkpoints: dict[str, Any],
                                smoke: dict[str, Any]) -> None:
    if validation.get("status") != "PASS" or validation.get("authorization_for_t038") is not True:
        raise RuntimeError("T038 execution denied: runner-validation is not PASS/authorized")
    if checkpoints.get("status") != "PASS" or smoke.get("status") != "PASS":
        raise RuntimeError("T038 execution denied: T032/T033 prerequisites are not PASS")


def current_authorization_fingerprints(root: Path = REPO) -> dict[str, Any]:
    artifacts = {name: sha256_file(root / name) for name in AUTHORIZATION_ARTIFACTS}
    expected_runtime = json.loads((root / "specs/002-qwen-bonsai-hybrid-target/evidence/series-a/discovery-index.json").read_text())[
        "runtime_revision"]["source_sha256"]
    runtime = {name: sha256_file(root / name) for name in expected_runtime}
    method_file = root / "specs/002-qwen-bonsai-hybrid-target/evidence/probes/series_a.py"
    method_hash = benchmark_method_hash(method_file.read_bytes())
    return {"artifacts": artifacts, "series_a_benchmark_method_sha256": method_hash,
            "experiment_runtime_source_sha256": runtime}


def validate_t038_authorization(validation: dict[str, Any], *, root: Path = REPO) -> None:
    if validation.get("status") != "PASS" or validation.get("authorization_for_t038") is not True:
        raise RuntimeError("T038 execution denied: runner-validation is not PASS/authorized")
    authorized = validation.get("authorization_fingerprint_set")
    if not isinstance(authorized, dict):
        raise RuntimeError("T038 execution denied: T037 authorization fingerprint set is missing")
    current = current_authorization_fingerprints(root)
    if current != authorized:
        mismatched = [name for name in AUTHORIZATION_ARTIFACTS
                      if current["artifacts"].get(name) != authorized.get("artifacts", {}).get(name)]
        if current["series_a_benchmark_method_sha256"] != authorized.get("series_a_benchmark_method_sha256"):
            mismatched.append("Series-A run_series_condition method")
        if current["experiment_runtime_source_sha256"] != authorized.get("experiment_runtime_source_sha256"):
            mismatched.append("seven experiment runtime files")
        raise RuntimeError("T038 execution denied: authorized artifact fingerprint mismatch: " +
                           (", ".join(mismatched) or "fingerprint set differs"))
    expected_runtime = json.loads((root / "specs/002-qwen-bonsai-hybrid-target/evidence/series-a/discovery-index.json").read_text())[
        "runtime_revision"]["source_sha256"]
    if current["experiment_runtime_source_sha256"] != expected_runtime:
        raise RuntimeError("T038 execution denied: seven experiment runtime hashes no longer match Series A")


def build_runner_snapshot(gate_path: Path | None = None) -> dict[str, Any]:
    gate = require_entry(gate_path)
    series_a_runner = EVIDENCE / "series-a/runner-used-for-discovery.py"
    current_runner = EVIDENCE / "probes/series_a.py"
    expected_runtime = json.loads((EVIDENCE / "series-a/discovery-index.json").read_text())[
        "runtime_revision"]["source_sha256"]
    current_runtime = {name: sha256_file(REPO / name) for name in expected_runtime}
    runtime_matches = current_runtime == expected_runtime
    method_hash = benchmark_method_hash(series_a_runner.read_bytes())
    current_method_hash = benchmark_method_hash(current_runner.read_bytes())
    if not runtime_matches or method_hash != SERIES_A_METHOD_SHA256 or current_method_hash != method_hash:
        raise RuntimeError("accepted runtime or Series-A benchmark function fingerprint mismatch")
    result = {
        "schema": "qwen-bonsai-series-b-runner/v1", "status": "PASS",
        "gate_b": {"validated": gate.get("passed") is True, "decision": "OPEN"},
        "series_b_runner": {"path": str(Path(__file__).resolve().relative_to(REPO)),
                            "sha256": sha256_file(Path(__file__).resolve())},
        "reused_series_a_runner": {
            "snapshot_path": str(series_a_runner.relative_to(REPO)),
            "snapshot_sha256": sha256_file(series_a_runner),
            "current_path": str(current_runner.relative_to(REPO)),
            "current_sha256": sha256_file(current_runner),
            "run_series_condition_ast_sha256": method_hash,
            "accepted_method_sha256": SERIES_A_METHOD_SHA256,
            "identity": "The actual run_series_condition function is imported and called unchanged for each Series-B condition; the wrapper supplies a frozen target/drafter checkpoint record and redirects the accepted function's output into Series-B-local staging before moving the result into series-b/runs.",
        },
        "experiment_runtime_source_sha256": current_runtime,
        "experiment_runtime_hashes_match_series_a": runtime_matches,
        "target_execution_prompt_generation_measurement_controller_metrics":
            "Delegated to the byte-identical accepted run_series_condition function and existing Engine APIs; the wrapper changes only target/drafter selection and Series-B-local output routing.",
        "performance_run_performed": False,
    }
    atomic_json(SERIES_B / "runner.json", result)
    return result


def checkpoint_record_for_arm(target: str, drafter: str) -> dict[str, Any]:
    require_entry()
    provenance = json.loads((SERIES_B / "checkpoints.json").read_text())
    key = "B-Q" if drafter == "B-Q" else "B-B"
    row = provenance[key]
    if key == "B-B":
        weights = row.get("verified_downloaded_weight_files", [])
        path = row.get("snapshot_path")
    else:
        weights = row.get("selected_weight_files", [])
        path = row.get("snapshot_path")
    return {"repo_id": row["repo_id"], "revision": row["resolved_revision"],
            "resolved_path": path,
            "config": {"sha256": row["config"]["sha256"],
                       **row["config"]["selected_config"]},
            "weights": weights}


def run_condition(target_name: str, drafter_name: str, run_id: str,
                  order_index: int, comparison_group: str = "discovery",
                  gate_path: Path | None = None) -> dict[str, Any]:
    """T038-facing wrapper; callable only after T037 authorizes execution."""
    require_entry(gate_path)
    validation_path = SERIES_B / "runner-validation.json"
    validation = json.loads(validation_path.read_text()) if validation_path.is_file() else {}
    try:
        validate_t038_authorization(validation, root=REPO)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        if isinstance(exc, RuntimeError) and str(exc).startswith("T038 execution denied:"):
            raise
        raise RuntimeError(f"T038 execution denied: cannot verify live authorization fingerprints: {exc}") from exc
    for name in ("matrix-decision.json", "matrix.json", "control-manifest.json", "checkpoints.json", "drafter-smoke.json"):
        if not (SERIES_B / name).is_file():
            raise RuntimeError(f"T038 execution denied: missing prerequisite {name}")
    smoke = json.loads((SERIES_B / "drafter-smoke.json").read_text())
    checkpoints = json.loads((SERIES_B / "checkpoints.json").read_text())
    validate_t038_prerequisites(validation, checkpoints, smoke)
    decision = json.loads((SERIES_B / "matrix-decision.json").read_text())
    matrix = json.loads((SERIES_B / "matrix.json").read_text())
    validate_matrix_decision(decision, matrix)
    controls = json.loads((SERIES_B / "control-manifest.json").read_text())
    cell = next((x for x in matrix.get("cells", [])
                 if x.get("target") == target_name and x.get("drafter") == drafter_name), None)
    if cell is None:
        raise RuntimeError("T038 execution denied: cell is not in the frozen 10-cell matrix")
    validate_discovery_schedule(matrix, order_index, target_name, drafter_name,
                                comparison_group)
    if run_id in {"", ".", ".."} or Path(run_id).name != run_id:
        raise ValueError("run_id must be a simple path component")
    accepted = json.loads((EVIDENCE / "checkpoints.json").read_text())
    checkpoint_rows = accepted["checkpoints"]
    series_b_drafter = checkpoint_record_for_arm(target_name, drafter_name)
    if drafter_name == "B-B":
        bb = checkpoints.get("B-B", {})
        snapshot = Path(bb.get("snapshot_path", ""))
        verified = bb.get("verified_downloaded_weight_files", [])
        if not snapshot.is_dir() or not verified or any(not Path(row.get("local_path", "")).is_file()
                                                        for row in verified):
            raise RuntimeError("T038 execution denied: pinned B-B local snapshot/verified weights are unavailable; rerun only exact pinned materialization and byte verification")
    core_checkpoints = {name: checkpoint_rows[name] for name in ("qwen", "bonsai")}
    core_checkpoints["dflash"] = series_b_drafter
    integrity_source = EVIDENCE / "integrity"
    integrity_dest = SERIES_B / "integrity"
    integrity_dest.mkdir(parents=True, exist_ok=True)
    integrity_ref = {}
    for variant in AUTHORIZED_TARGETS:
        source = integrity_source / f"{variant.lower()}.json"
        if not source.is_file():
            raise RuntimeError(f"T038 execution denied: missing accepted integrity record for {variant}")
        destination = integrity_dest / source.name
        shutil.copyfile(source, destination)
        integrity_ref[variant] = {"source_path": str(source.relative_to(REPO)),
                                  "sha256": sha256_file(source),
                                  "series_b_copy_sha256": sha256_file(destination)}

    # Import only after Gate B and every T038 prerequisite passed.
    spec = importlib.util.spec_from_file_location(
        "series_b_reused_series_a_runner", EVIDENCE / "probes/series_a.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import the accepted Series-A benchmark function")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_evidence = module.EVIDENCE
    original_prerequisites = module._validate_series_a_prerequisites
    original_corpus = module.ensure_series_a_corpus
    try:
        module.EVIDENCE = SERIES_B
        module._validate_series_a_prerequisites = lambda: {"checkpoints": core_checkpoints}
        module.ensure_series_a_corpus = lambda: json.loads(
            (EVIDENCE / "series-a/discovery-index.json").read_text())["prompt_corpus"]
        summary = module.run_series_condition(target_name, run_id, order_index, comparison_group)
    finally:
        module.EVIDENCE = original_evidence
        module._validate_series_a_prerequisites = original_prerequisites
        module.ensure_series_a_corpus = original_corpus
    staged = Path(summary["path"])
    staged_result = json.loads(staged.read_text())
    validate_requested_cell(staged_result, target_name, drafter_name, cell, matrix, controls)
    final_dir = SERIES_B / "runs" / run_id
    final_dir.mkdir(parents=True, exist_ok=False)
    final_path = final_dir / "result.json"
    staged_result["schema"] = "qwen-bonsai-series-b-run/v1"
    staged_result["series"] = "B"
    staged_result["drafter_arm"] = drafter_name
    staged_result["paired_target_identity_ref"] = matrix["target_identities"][target_name]
    staged_result["integrity_source_refs"] = integrity_ref
    validate_result_integrity(staged_result, expected_target=target_name,
                              expected_drafter=drafter_name, expected_cell=cell,
                              matrix=matrix, control_manifest=controls)
    final_path.write_text(json.dumps(staged_result, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    staged.unlink()
    staged.parent.rmdir()
    return {"status": "PASS", "run_id": run_id, "target": target_name,
            "drafter": drafter_name, "path": str(final_path),
            "sha256": sha256_file(final_path)}


def validate_requested_cell(result: dict[str, Any], target: str, drafter: str,
                            cell: dict[str, Any], matrix: dict[str, Any],
                            controls: dict[str, Any]) -> None:
    """Validate raw Series-A output against the requested cell before adding labels."""
    if result.get("variant") != target:
        raise ValueError("staged result variant does not match requested target")
    # The staged Series-A record has no synthetic Series-B arm label. Bind its
    # actual DFlash provenance directly to the matrix cell first.
    checkpoints = result.get("checkpoints", {})
    dflash = checkpoints.get("dflash", {})
    expected = cell.get("drafter_checkpoint", {})
    if (dflash.get("repo_id") != expected.get("repo_id")
            or dflash.get("revision") != expected.get("revision")
            or dflash.get("config_sha256") != expected.get("config_sha256")
            or _weight_fingerprints(dflash.get("weights", [])) !=
            _weight_fingerprints(expected.get("weight_files", []))):
        raise ValueError("staged result DFlash provenance does not match requested drafter arm")
    _validate_target_provenance(result, target, matrix["target_identities"][target])
    _validate_frozen_controls(result, controls)
    # Only now may the wrapper attach its Series-B identity labels.
    result["drafter_arm"] = drafter


def build_frozen_matrix_and_controls(*, write: bool = True,
                                     gate_path: Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    gate = require_entry(gate_path)
    runner = json.loads((SERIES_B / "runner.json").read_text())
    provenance = json.loads((SERIES_B / "checkpoints.json").read_text())
    smoke = json.loads((SERIES_B / "drafter-smoke.json").read_text())
    if any(row.get("status") != "PASS" for row in (runner, provenance, smoke)):
        raise RuntimeError("T035 requires PASS runner, checkpoint provenance, and smoke artifacts")
    accepted = json.loads((EVIDENCE / "checkpoints.json").read_text())["checkpoints"]
    discovery = json.loads((EVIDENCE / "series-a/discovery-index.json").read_text())
    prompts = json.loads((EVIDENCE / "series-a/prompts.json").read_text())
    tokenized = smoke["prompt_corpus"]["tokenized_prompts"]
    if [x["id"] for x in tokenized] != [x["id"] for x in prompts["prompts"]]:
        raise RuntimeError("T035 tokenized prompt IDs do not match the immutable prompt corpus")

    def target_identity(name: str) -> dict[str, Any]:
        record_path = EVIDENCE / "integrity" / f"{name.lower()}.json"
        record = json.loads(record_path.read_text())
        if not record.get("passed"):
            raise RuntimeError(f"accepted Series-A target integrity is not PASS for {name}")
        full_bonsai = name == "B0"
        model = accepted["bonsai" if full_bonsai else "qwen"]
        manifest = record["ownership"].get("composition_manifest") or {}
        donor_indices = list(manifest.get("donor_indices", [])) if not full_bonsai else list(range(64))
        owners = record["ownership"]["block_owners"]
        if len(owners) != 64 or [x["index"] for x in owners] != list(range(64)):
            raise RuntimeError(f"T035 target {name} lacks a complete 64-block owner map")
        weights = model["weights"]
        return {
            "target": name,
            "repo_id": model["repo_id"], "revision": model["revision"],
            "resolved_path": model["resolved_path"],
            "config_sha256": model["config"]["sha256"],
            "weight_files": weights,
            "donor_checkpoint": None if full_bonsai else {
                "repo_id": accepted["bonsai"]["repo_id"],
                "revision": accepted["bonsai"]["revision"],
                "config_sha256": accepted["bonsai"]["config"]["sha256"],
                "weight_files": accepted["bonsai"]["weights"],
                "donor_block_indices": donor_indices,
            },
            "ownership": {
                "block_owners": owners,
                "embedding_owner": record["ownership"]["embedding_owner"],
                "final_norm_owner": record["ownership"]["final_norm_owner"],
                "lm_head_owner": record["ownership"]["lm_head_owner"],
            },
            "integrity_source": {"path": str(record_path.relative_to(REPO)),
                                 "sha256": sha256_file(record_path)},
        }

    target_identities = {name: target_identity(name) for name in AUTHORIZED_TARGETS}
    for name, identity in target_identities.items():
        identity["identity_sha256"] = hashlib.sha256(json.dumps(
            {k: v for k, v in identity.items() if k != "identity_sha256"},
            sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    cells = []
    schedule = []
    for pair_index, name in enumerate(AUTHORIZED_TARGETS):
        arms = AUTHORIZED_DRAFTERS if pair_index % 2 == 0 else tuple(reversed(AUTHORIZED_DRAFTERS))
        for arm in arms:
            cell = {"cell_id": f"{name}-{arm}", "target": name, "drafter": arm,
                    "target_identity_sha256": target_identities[name]["identity_sha256"],
                    "drafter_checkpoint": {
                        "repo_id": provenance[arm]["repo_id"],
                        "revision": provenance[arm]["resolved_revision"],
                        "config_sha256": provenance[arm]["config"]["sha256"],
                        "weight_files": provenance[arm]["selected_weight_files"],
                    }}
            cells.append(cell)
            schedule.append({"order_index": len(schedule) + 1, "target": name,
                             "drafter": arm, "cell_id": cell["cell_id"]})
    if len(cells) != 10 or {(x["target"], x["drafter"]) for x in cells} != {
            (t, d) for t in AUTHORIZED_TARGETS for d in AUTHORIZED_DRAFTERS}:
        raise RuntimeError("T035 generated matrix is not exactly the authorized ten cells")
    matrix = {
        "schema": "qwen-bonsai-series-b-matrix/v1", "status": "PASS",
        "gate_b": {"validated": gate.get("passed") is True, "decision": "OPEN"},
        "targets": list(AUTHORIZED_TARGETS), "drafters": list(AUTHORIZED_DRAFTERS),
        "target_identities": target_identities, "cells": cells,
        "cell_count": 10, "isolated_block_exception": {
            "authorized": False, "excluded_targets": ["H1a", "H1b"],
            "reason": "H1c>H1a confirmed; H1c-vs-H1b within_noise.",
        },
        "schedule": {"type": "adjacent matched arms with alternating order by target",
                     "rows": schedule, "adjacent_pairs": True,
                     "arm_order_alternates": True},
        "performance_run_performed": False,
    }
    fixed = discovery["fixed_controls"]
    runtime_hashes = runner["experiment_runtime_source_sha256"]
    common = {
        "target_checkpoint_and_weights": "identical within each target pair; see matrix.target_identities",
        "target_ownership_map": "identical complete 64-block owner map within each pair",
        "prompt_corpus": {"path": str((EVIDENCE / "series-a/prompts.json").relative_to(REPO)),
                          "sha256": sha256_file(EVIDENCE / "series-a/prompts.json"),
                          "tokenized_prompts": tokenized},
        "generation": prompts["generation"],
        "sampling": {k: prompts["generation"][k] for k in
                     ("temperature", "top_p", "top_k", "seed", "stop",
                      "presence_penalty", "frequency_penalty")},
        "plain_kv": True, "kv_bits": None,
        "controller": "ordinary production CapController, max_draft_tokens=auto",
        "width_policy": False,
        "speculative_policy": "accepted production DFlash Engine path; no tuning",
        "runtime_source_sha256": runtime_hashes,
        "hardware": discovery["runtime_revision"]["hardware"],
        "measurement_method": {"run_series_condition_ast_sha256": SERIES_A_METHOD_SHA256,
                               "source": "byte-identical accepted Series-A function"},
        "measurement_protocol": {
            "warmup": fixed["warmup"], "fresh_process_and_engine_per_run": True,
            "serial_greedy_then_speculative": True,
            "no_prefix_or_kv_reuse_between_prompts": True,
            "memory_guard": fixed["memory_guard"],
        },
        "fixed_engine_settings": {k: fixed[k] for k in
                                  ("mode", "drafter_bits", "lookup_drafts", "prefix_cache",
                                   "small_m", "sdpa_split", "wide_gemm_min", "cpu_split")},
    }
    pairs = {}
    for target in AUTHORIZED_TARGETS:
        pair = [x for x in cells if x["target"] == target]
        if len(pair) != 2 or pair[0]["target_identity_sha256"] != pair[1]["target_identity_sha256"]:
            raise RuntimeError(f"T035 target identity mismatch for paired target {target}")
        pairs[target] = {"cells": [x["cell_id"] for x in pair],
                         "target_identity_sha256": target_identities[target]["identity_sha256"],
                         "common_controls_sha256": hashlib.sha256(json.dumps(
                             common, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                         "only_allowed_differences": ["exact drafter checkpoint"],
                         "compatibility_adaptation": "none"}
    controls = {
        "schema": "qwen-bonsai-series-b-control-manifest/v1", "status": "PASS",
        "gate_b": matrix["gate_b"], "common_controls": common,
        "paired_controls": pairs,
        "allowed_pair_differences": ["exact drafter checkpoint"],
        "compatibility_adaptation": "none",
        "fresh_controls": {
            "H0-B-Q": "fresh Series-B production/reference baseline",
            "H0-B-B": "unchanged-Qwen-target drafter-control cell",
            "series_a_throughput_substitution_allowed": False,
        },
        "matrix_cell_count": len(cells), "performance_run_performed": False,
    }
    if write:
        atomic_json(SERIES_B / "matrix.json", matrix)
        atomic_json(SERIES_B / "control-manifest.json", controls)
    return matrix, controls


def build_runner_validation() -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}

    def check(name: str, passed: bool, detail: str) -> None:
        checks[name] = {"passed": bool(passed), "detail": detail}

    try:
        gate = require_entry()
        check("live_gate_b_open_pass", gate.get("passed") is True,
              "existing Gate-B Series-B entry guard returned PASS/OPEN")
    except Exception as exc:
        gate = {}
        check("live_gate_b_open_pass", False, f"entry guard failed: {type(exc).__name__}: {exc}")

    expected_hashes = json.loads((EVIDENCE / "series-a/discovery-index.json").read_text())[
        "runtime_revision"]["source_sha256"]
    current_hashes = {name: sha256_file(REPO / name) for name in expected_hashes}
    check("seven_experiment_runtime_hashes", current_hashes == expected_hashes,
          json.dumps({name: {"expected": expected_hashes[name], "current": current_hashes[name]}
                      for name in expected_hashes}, sort_keys=True))

    def read(name: str) -> dict[str, Any]:
        return json.loads((SERIES_B / name).read_text())

    artifact_errors = []
    try:
        decision, provenance, smoke = (read("matrix-decision.json"), read("checkpoints.json"),
                                       read("drafter-smoke.json"))
        runner, matrix, controls = (read("runner.json"), read("matrix.json"),
                                    read("control-manifest.json"))
    except Exception as exc:
        artifact_errors.append(f"missing/malformed prerequisite: {type(exc).__name__}: {exc}")
        decision = provenance = smoke = runner = matrix = controls = {}
    check("series_b_prerequisites_present", not artifact_errors,
          "; ".join(artifact_errors) if artifact_errors else "all T031-T035 evidence artifacts parse")

    bb = provenance.get("B-B", {})
    bq = provenance.get("B-Q", {})
    verified = bb.get("verified_downloaded_weight_files", [])
    checkpoint_ok = (
        provenance.get("status") == "PASS" and bb.get("repo_id") == BB_REPO
        and isinstance(bb.get("resolved_revision"), str)
        and len(bb.get("resolved_revision", "")) == 40
        and bool(bb.get("config", {}).get("sha256"))
        and len(bb.get("selected_weight_files", [])) > 0
        and len(verified) == len(bb.get("selected_weight_files", []))
        and all(v.get("sha256") == w.get("sha256") and v.get("size_bytes") == w.get("size_bytes")
                and Path(v.get("local_path", "")).is_file()
                for v, w in zip(verified, bb.get("selected_weight_files", [])))
        and provenance.get("B-Q", {}).get("repo_id") == BQ_REPO
        and provenance.get("B-Q", {}).get("resolved_revision") == BQ_REVISION
        and bb.get("provenance_leads", {}).get("weights_sha256", {}).get("comparison") == "MATCHED"
    )
    check("t032_exact_checkpoint_provenance", checkpoint_ok,
          "B-B exact immutable revision/config/selected weights and byte hashes verified; B-Q matches accepted identity; revision and weight leads separately recorded")
    tap_ids = smoke.get("loaded_dflash", {}).get("target_layer_ids", [])
    smoke_ok = (
        smoke.get("status") == "PASS" and tap_ids == [5, 19, 33, 47, 61]
        and smoke.get("loaded_dflash", {}).get("tap_width_compatible") is True
        and smoke.get("loaded_parameters", {}).get("all_finite") is True
        and smoke.get("forward", {}).get("drafter_forward_finite") is True
        and smoke.get("cache", {}).get("advanced") is True
        and smoke.get("cache", {}).get("rejected_suffix_rollback") is True
        and smoke.get("cache", {}).get("plain_kv") is True
        and smoke.get("controls", {}).get("ordinary_production_cap_controller_implementation") ==
            "mlx_dspark.calibrate.CapController"
        and smoke.get("controls", {}).get("width_policy_present") is False
        and smoke.get("compatibility_adapter") == "none"
    )
    check("t033_compatibility_smoke", smoke_ok,
          f"status={smoke.get('status')}; loaded taps={tap_ids}; no compatibility adapter")
    cells = matrix.get("cells", [])
    expected_cells = {(target, drafter) for target in AUTHORIZED_TARGETS
                      for drafter in AUTHORIZED_DRAFTERS}
    actual_cells = {(row.get("target"), row.get("drafter")) for row in cells}
    matrix_ok = (matrix.get("status") == "PASS" and len(cells) == 10
                 and actual_cells == expected_cells
                 and matrix.get("isolated_block_exception", {}).get("authorized") is False
                 and not ({"H1a", "H1b"} & {row.get("target") for row in cells}))
    check("exact_frozen_ten_cell_matrix", matrix_ok,
          f"cells={len(cells)}; targets={sorted({x[0] for x in actual_cells})}; no H1a/H1b")
    control_ok = (controls.get("status") == "PASS"
                  and controls.get("matrix_cell_count") == 10
                  and controls.get("allowed_pair_differences") == ["exact drafter checkpoint"]
                  and controls.get("compatibility_adaptation") == "none"
                  and set(controls.get("paired_controls", {})) == set(AUTHORIZED_TARGETS)
                  and controls.get("fresh_controls", {}).get("H0-B-Q") ==
                      "fresh Series-B production/reference baseline"
                  and controls.get("fresh_controls", {}).get("H0-B-B") ==
                      "unchanged-Qwen-target drafter-control cell")
    if control_ok:
        for target in AUTHORIZED_TARGETS:
            rows = [x for x in cells if x.get("target") == target]
            pair = controls["paired_controls"][target]
            if (len(rows) != 2 or rows[0].get("target_identity_sha256") !=
                    rows[1].get("target_identity_sha256")
                    or pair.get("target_identity_sha256") != rows[0].get("target_identity_sha256")
                    or pair.get("only_allowed_differences") != ["exact drafter checkpoint"]):
                control_ok = False
                break
        schedule = matrix.get("schedule", {}).get("rows", [])
        if len(schedule) != 10 or not matrix.get("schedule", {}).get("adjacent_pairs"):
            control_ok = False
        else:
            first_arms = []
            for offset in range(0, 10, 2):
                a, b = schedule[offset:offset + 2]
                if a.get("target") != b.get("target") or {a.get("drafter"), b.get("drafter")} != set(AUTHORIZED_DRAFTERS):
                    control_ok = False
                    break
                first_arms.append(a.get("drafter"))
            if any(a == b for a, b in zip(first_arms, first_arms[1:])):
                control_ok = False
    check("matrix_control_manifest_consistent", control_ok,
          "paired target identity, immutable common controls, exact controls, baseline labels, and adjacent alternating schedule checked")

    try:
        validate_matrix_decision(decision, matrix)
        decision_ok = True
        decision_detail = "PASS; current Series-A adjudication/repeat hashes and exact ten matrix cells agree"
    except Exception as exc:
        decision_ok = False
        decision_detail = f"{type(exc).__name__}: {exc}"
    check("matrix_decision_recomputed_and_bound", decision_ok, decision_detail)

    runner_script = Path(__file__).resolve()
    method_hash = benchmark_method_hash((EVIDENCE / "series-a/runner-used-for-discovery.py").read_bytes())
    current_method_hash = benchmark_method_hash((EVIDENCE / "probes/series_a.py").read_bytes())
    runner_ok = (runner.get("status") == "PASS"
                 and runner.get("series_b_runner", {}).get("sha256") == sha256_file(runner_script)
                 and runner.get("reused_series_a_runner", {}).get("run_series_condition_ast_sha256") == SERIES_A_METHOD_SHA256
                 and method_hash == current_method_hash == SERIES_A_METHOD_SHA256
                 and runner.get("experiment_runtime_hashes_match_series_a") is True)
    check("runner_fingerprint_and_benchmark_identity", runner_ok,
          f"Series-B source sha256={sha256_file(runner_script)}; method={method_hash}; accepted={SERIES_A_METHOD_SHA256}")
    test_file = REPO / "tests/test_series_b.py"
    test_sha_before = sha256_file(test_file) if test_file.is_file() else None
    command = [sys.executable, "-m", "pytest", "tests/test_series_b.py", "-q"]
    try:
        completed = subprocess.run(command, cwd=REPO, text=True, capture_output=True, check=False)
        output = (completed.stdout or "") + "\n" + (completed.stderr or "")
        summary = re.search(r"(?m)^([0-9]+) passed(?:,|\s|$)", output)
        actual_count = int(summary.group(1)) if summary else None
        actual_result = (output.strip().splitlines()[-1].strip() if output.strip()
                         else "<empty pytest output>")
        test_sha_after = sha256_file(test_file) if test_file.is_file() else None
        tests_ok = (completed.returncode == 0 and actual_count is not None
                    and actual_count > 0 and test_sha_before == test_sha_after)
        test_evidence = {
            "command": command, "exit_code": completed.returncode,
            "pass_count": actual_count, "summary": actual_result,
            "stdout": (completed.stdout or "")[-3000:],
            "stderr": (completed.stderr or "")[-3000:],
            "test_file_sha256": test_sha_after,
        }
        test_detail = json.dumps(test_evidence, sort_keys=True)
    except Exception as exc:
        actual_count, test_sha_after, tests_ok = None, None, False
        test_evidence = {"command": command, "exit_code": None, "pass_count": None,
                         "summary": f"{type(exc).__name__}: {exc}",
                         "test_file_sha256": test_sha_after}
        test_detail = json.dumps(test_evidence, sort_keys=True)
    check("model_free_tests_pass", tests_ok, test_detail)
    check("gate_first_invariant", checks.get("live_gate_b_open_pass", {}).get("passed", False)
          and checks.get("model_free_tests_pass", {}).get("passed", False),
          "each resolver/download/smoke/runner path guards before external imports or runtime loading; model-free guard-bypass tests PASS")
    reference_checks = gate.get("checks", {})
    refs_ok = (reference_checks.get("referenced_artifact_hashes", {}).get("passed") is True
               and reference_checks.get("complete_experiment_runtime_hash_set", {}).get("passed") is True)
    check("series_a_evidence_untouched", refs_ok,
          "live Gate-B immutable referenced-artifact hashes and accepted runtime source hashes remain PASS")
    series_b_runs = SERIES_B / "runs"
    no_matrix_run = not series_b_runs.exists() or not any(series_b_runs.iterdir())
    check("no_series_b_performance_discovery", no_matrix_run,
          "no Series-B run records exist; no performance discovery matrix has run")

    passed = bool(checks) and all(v["passed"] for v in checks.values())
    try:
        fingerprints = current_authorization_fingerprints()
        fingerprint_error = None
    except Exception as exc:
        fingerprints = {}
        fingerprint_error = f"{type(exc).__name__}: {exc}"
        passed = False
        check("authorization_fingerprint_set", False, fingerprint_error)
    else:
        check("authorization_fingerprint_set", True,
              "exact hashes persisted for the eight pre-T038 artifacts, accepted Series-A benchmark method, and all seven experiment runtime files")
    result = {
        "schema": "qwen-bonsai-series-b-runner-validation/v1",
        "status": "PASS" if passed else "DENY",
        "authorization_for_t038": passed,
        "checks": checks,
        "model_free_test_evidence": test_evidence,
        "authorization_fingerprint_set": fingerprints,
        "loaded_b_b_tap_ids": tap_ids,
        "matrix_cell_count": len(cells),
        "matrix_targets": sorted({x[0] for x in actual_cells}),
        "matrix_drafters": sorted({x[1] for x in actual_cells}),
        "no_series_b_matrix_performance_measurement": no_matrix_run,
        "preliminary_smoke_execution_note": smoke.get("execution_history", {}).get("discarded_attempt"),
        "series_a_checkpoint_artifact_modified": False,
        "series_a_raw_artifacts_modified": not refs_ok,
        "experiment_runtime_hashes_match": current_hashes == expected_hashes,
    }
    atomic_json(SERIES_B / "runner-validation.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("matrix-decision", "entry-check", "resolve-checkpoints",
                                             "fetch-bb-weights", "drafter-smoke", "build-runner",
                                             "build-matrix", "build-validation", "run-condition"))
    parser.add_argument("--target", choices=AUTHORIZED_TARGETS)
    parser.add_argument("--drafter", choices=AUTHORIZED_DRAFTERS)
    parser.add_argument("--run-id")
    parser.add_argument("--order-index", type=int)
    parser.add_argument("--comparison-group", default="discovery")
    args = parser.parse_args(argv)
    if args.action == "matrix-decision":
        result = write_matrix_decision()
    elif args.action == "resolve-checkpoints":
        result = resolve_checkpoint_provenance()
    elif args.action == "fetch-bb-weights":
        result = fetch_verified_bb_weights()
    elif args.action == "drafter-smoke":
        try:
            result = drafter_compatibility_smoke()
        except Exception as exc:
            failure = {"schema": "qwen-bonsai-series-b-drafter-smoke/v1",
                       "status": "FAIL", "error_type": type(exc).__name__,
                       "error": str(exc), "performance_measurement_performed": False}
            atomic_json(SERIES_B / "drafter-smoke.json", failure)
            raise
        atomic_json(SERIES_B / "drafter-smoke.json", result)
    elif args.action == "build-runner":
        result = build_runner_snapshot()
    elif args.action == "build-matrix":
        matrix, controls = build_frozen_matrix_and_controls()
        result = {"status": "PASS", "cell_count": matrix["cell_count"],
                  "targets": matrix["targets"], "drafters": matrix["drafters"],
                  "controls_status": controls["status"]}
    elif args.action == "build-validation":
        result = build_runner_validation()
    elif args.action == "run-condition":
        if not args.target or not args.drafter or not args.run_id or args.order_index is None:
            parser.error("run-condition requires --target, --drafter, --run-id, and --order-index")
        result = run_condition(args.target, args.drafter, args.run_id,
                               args.order_index, args.comparison_group)
    else:
        gate = require_entry()
        result = {"status": "PASS", "gate_b_passed": gate.get("passed") is True}
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.action == "build-validation":
        return 0 if result.get("status") == "PASS" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
