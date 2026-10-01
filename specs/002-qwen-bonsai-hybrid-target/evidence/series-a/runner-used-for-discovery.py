#!/usr/bin/env python3
"""Feature-002 checkpoint preflight and exact official Bonsai loader proof.

Preflight fingerprints pinned local snapshots without importing MLX. The proof mode
loads only the official full Bonsai target through the repository loader; it is not
a benchmark and never falls back to the registry or historical repack.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
FEATURE = ROOT / "specs/002-qwen-bonsai-hybrid-target"
EVIDENCE = FEATURE / "evidence"
NUM_LAYERS = 64
REPOS = {
    "qwen": ("mlx-community/Qwen3.8-27B-4bit", "10c35caafbb80f7dc6a7a432cdd11af10a6d4818"),
    "bonsai": ("prism-ml/Ternary-Bonsai-2-27B-mlx-2bit", "fcba37d2117a7077eac6b613b2668d14d9779edd"),
    "dflash": ("incoai/Qwen3.8-27B-DFlash2", "015e795645c74b1a0eeef3b570031fb62e769bc5"),
}
VARIANTS = {
    "H0": (), "H1a": (63,), "H1b": (62,), "H1c": (62, 63),
    "H2": (60, 61, 62, 63), "H3": (56, 57, 58, 59, 60, 61, 62, 63),
    "B0": None,
}
SERIES_A_ORDER = ("H0", "H1a", "H2", "H1b", "H3", "H1c", "B0", "H0")
SERIES_A_PROMPTS = (
    {"id": "p01", "text": "Explain how a hash table handles collisions. Compare separate chaining with open addressing, and give one practical tradeoff for each."},
    {"id": "p02", "text": "Write a Python function that returns the length of the longest increasing subsequence. Include a short explanation of its time complexity."},
    {"id": "p03", "text": "A team is debugging a slow web service. Give a concise, ordered plan for measuring latency, finding the bottleneck, and verifying a fix."},
)
SERIES_A_GENERATION = {
    "max_new_tokens": 128, "temperature": 0.0, "top_p": 1.0, "top_k": 0,
    "seed": 0, "stop": [], "presence_penalty": 0.0, "frequency_penalty": 0.0,
}
DEFAULT_PATHS = {
    "qwen": Path("/Users/do/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots/10c35caafbb80f7dc6a7a432cdd11af10a6d4818"),
    "bonsai": Path("/private/tmp/prism-bonsai2-fcba37d2117a7077eac6b613b2668d14d9779edd"),
    "dflash": Path("/Users/do/.cache/huggingface/hub/models--incoai--Qwen3.8-27B-DFlash2/snapshots/015e795645c74b1a0eeef3b570031fb62e769bc5"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state() -> dict:
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                          check=True, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                           check=True, capture_output=True, text=True).stdout.splitlines()
    return {"head": head, "dirty": bool(dirty), "dirty_paths": dirty}


def checkpoint_path(name: str) -> Path:
    override = os.environ.get(f"DSPARK_002_{name.upper()}_PATH")
    return Path(override).expanduser().resolve() if override else DEFAULT_PATHS[name].resolve()


def inspect_checkpoint(name: str) -> dict:
    repo, revision = REPOS[name]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"{name}: accepted checkpoints require a 40-character immutable revision")
    path = checkpoint_path(name)
    config_path = path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"{name}: missing pinned snapshot config {config_path}")
    config = json.loads(config_path.read_text())
    weights = sorted((*path.glob("*.safetensors"), *path.glob("*.safetensors.index.json")))
    if name != "dflash" and not weights:
        raise FileNotFoundError(f"{name}: no model weight files in {path}")
    files = []
    for weight in weights:
        files.append({"name": weight.name, "path": str(weight.resolve()),
                      "size_bytes": weight.stat().st_size,
                      "fingerprint_method": "SHA-256 over complete file bytes",
                      "sha256": sha256(weight)})
    tokenizer_files = [path / name for name in
                       ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
                       if (path / name).is_file()]
    return {
        "repo_id": repo, "revision": revision, "resolved_path": str(path),
        "config": {"path": str(config_path.resolve()), "sha256": sha256(config_path),
                   "model_type": config.get("model_type"),
                   "architectures": config.get("architectures"),
                   "text_config": config.get("text_config", {})},
        "weights": files,
        "tokenizer_identity": {
            "files": [{"name": p.name, "sha256": sha256(p)} for p in tokenizer_files],
            "shared_with": "qwen" if name == "dflash" else None,
            "vocab_size": (config.get("text_config") or config).get("vocab_size"),
        },
    }


def preflight() -> dict:
    checkpoints = {name: inspect_checkpoint(name) for name in REPOS}
    runtime = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("mlx", "mlx-lm", "safetensors", "huggingface-hub")},
    }
    result = {
        "schema": "qwen-bonsai-hybrid-checkpoints/v1",
        "created_unix": int(time.time()),
        "accepted_revisions_are_immutable": True,
        "checkpoints": checkpoints,
        "runtime": runtime,
        "repository": git_state(),
    }
    (EVIDENCE / "checkpoints.json").parent.mkdir(parents=True, exist_ok=True)
    (EVIDENCE / "checkpoints.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def resolve_inputs() -> dict:
    """T001 model-free resolution: resolve pins/absolute snapshots without model weights."""
    entries = {}
    for name, (repo, revision) in REPOS.items():
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError(f"{name}: floating or malformed revision is forbidden: {revision!r}")
        path = checkpoint_path(name)
        config = path / "config.json"
        if not path.is_dir() or not config.is_file():
            raise FileNotFoundError(f"{name}: pinned absolute snapshot/config is unavailable: {path}")
        entries[name] = {
            "repo_id": repo, "revision": revision,
            "revision_kind": "immutable Hugging Face commit SHA",
            "absolute_snapshot": str(path), "config_sha256": sha256(config),
        }
    result = {"schema": "qwen-bonsai-input-resolution/v1", "passed": True,
              "weight_loading_performed": False, "inference_performed": False,
              "floating_refs_accepted": False, "checkpoints": entries}
    out = EVIDENCE / "input-resolution.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def validate_donor_structure() -> dict:
    donor_path = checkpoint_path("bonsai")
    config_path = donor_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"official immutable Bonsai config is unavailable at {config_path}")
    config = json.loads(config_path.read_text())
    text = config.get("text_config", {})
    layers = text.get("layer_types", [])
    qwen_config = json.loads((checkpoint_path("qwen") / "config.json").read_text())
    qwen_text = qwen_config.get("text_config", qwen_config)
    qwen_layers = qwen_text.get("layer_types", [])
    modules = config.get("modules", [])
    module_paths = [row.get("path") for row in modules]
    block_ids = sorted({int(match.group(1)) for module_path in module_paths
                        if (match := re.search(r"(?:^|\.)layers\.(\d+)\.", module_path))})
    packed_records = [row for row in modules if row.get("path")]
    facts = {
        "repo_id": REPOS["bonsai"][0], "revision": REPOS["bonsai"][1],
        "schema_version": config.get("schema_version"), "model_type": config.get("model_type"),
        "num_hidden_layers": text.get("num_hidden_layers"),
        "hidden_size": text.get("hidden_size"),
        "intermediate_size": text.get("intermediate_size"),
        "vocab_size": text.get("vocab_size"),
        "full_attention_interval": text.get("full_attention_interval"),
        "layer_types": layers,
        "layer_type_counts": {kind: layers.count(kind) for kind in sorted(set(layers))},
        "qwen_dimensions_match": all(text.get(key, config.get(key)) ==
                                      qwen_text.get(key, qwen_config.get(key))
                                      for key in ("num_hidden_layers", "hidden_size",
                                                  "intermediate_size", "vocab_size")),
        "qwen_layer_family_sequence_matches": layers == qwen_layers,
        "layer_indices_in_module_inventory": block_ids,
        "module_record_count": len(modules),
        "quantization": config.get("quantization"),
        "packed_projection_record_count": len(packed_records),
        "required_module_paths_present": all(any(p and p.startswith(prefix) for p in module_paths)
            for prefix in ("model.layers.0.", "model.layers.63.", "model.embed_tokens")),
        "config_sha256": sha256(config_path),
    }
    expected = {"schema_version": 2, "model_type": "prism_hadamard_qwen35",
                "num_hidden_layers": 64, "hidden_size": 5120,
                "intermediate_size": 17408, "vocab_size": 248320}
    mismatches = {key: {"expected": value, "actual": facts.get(key)}
                  for key, value in expected.items() if facts.get(key) != value}
    if facts["layer_indices_in_module_inventory"] != list(range(64)):
        mismatches["layer_indices_in_module_inventory"] = {
            "expected": list(range(64)), "actual": facts["layer_indices_in_module_inventory"]}
    if not facts["required_module_paths_present"]:
        mismatches["required_module_paths_present"] = {
            "required_prefixes": ["model.layers.0.", "model.layers.63.", "model.embed_tokens"],
            "actual_paths": module_paths}
    if not facts["qwen_dimensions_match"]:
        mismatches["qwen_dimensions_match"] = {"qwen": qwen_text, "bonsai": text}
    if not facts["qwen_layer_family_sequence_matches"]:
        mismatches["qwen_layer_family_sequence_matches"] = {
            "qwen": qwen_layers, "bonsai": layers}
    quant = facts["quantization"] or {}
    if (quant.get("bits"), quant.get("group_size"), quant.get("mode")) != (2, 128, "affine"):
        mismatches["quantization"] = {"expected": {"bits": 2, "group_size": 128, "mode": "affine"},
                                       "actual": quant}
    result = {"schema": "qwen-bonsai-donor-structure/v1", "facts": facts,
              "planning_revision": "3f926b415992eaa2ae9dd7b573706494d6bbf787",
              "revision_changed_since_research": REPOS["bonsai"][1] !=
                  "3f926b415992eaa2ae9dd7b573706494d6bbf787",
              "mismatches": mismatches, "passed": not mismatches}
    out = EVIDENCE / "integrity/official-bonsai-structure.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if mismatches:
        raise RuntimeError(f"official Bonsai structure contradicts plan assumptions: {mismatches}")
    return result


def record_official_pack_inventory() -> dict:
    """Index the pinned safetensors headers against every Prism-declared module."""
    from safetensors import safe_open

    path = checkpoint_path("bonsai")
    config = json.loads((path / "config.json").read_text())
    mma_source = ROOT / "src/mlx_dspark/mlx_qmm_mma.py"
    mma_text = mma_source.read_text()
    mma_ast = ast.parse(mma_text)
    mma_constants = {}
    for node in ast.walk(mma_ast):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {
                        "N_MIN", "_SUPPORTED_BITS", "_SUPPORTED_GS"}:
                    mma_constants[target.id] = ast.literal_eval(node.value)
    if set(mma_constants) != {"N_MIN", "_SUPPORTED_BITS", "_SUPPORTED_GS"}:
        raise RuntimeError("Could not read the existing MMA shape eligibility constants")
    tensors = {}
    with safe_open(str(path / "model.safetensors"), framework="numpy") as model_file:
        for key in model_file.keys():
            tensor = model_file.get_slice(key)
            tensors[key] = {"shape": list(tensor.get_shape()), "dtype": tensor.get_dtype()}
    rows = []
    missing = []
    for record in config.get("modules", []):
        module_path = record["path"]
        runtime_path = "language_model." + module_path
        fields = {}
        for field in ("weight", "scales", "biases", "signs"):
            key = f"{runtime_path}.{field}"
            if key in tensors:
                fields[field] = tensors[key]
        if "weight" not in fields:
            missing.append(module_path)
        shape = fields.get("weight", {}).get("shape", [])
        embedding = bool(record.get("embedding", False))
        bits = int((config.get("quantization") or {}).get("bits", 0))
        group_size = int((config.get("quantization") or {}).get("group_size", 0))
        packed_shape_eligible = (len(shape) == 2 and bits > 0 and
                                 bits in mma_constants["_SUPPORTED_BITS"] and
                                 group_size in mma_constants["_SUPPORTED_GS"] and
                                 shape[0] >= mma_constants["N_MIN"] and
                                 (shape[1] * 32 // bits) % 512 == 0)
        rows.append({
            "pack_path": module_path, "runtime_module_path": runtime_path,
            "block": record.get("block"), "embedding": embedding,
            "bits": bits, "group_size": group_size,
            "mode": (config.get("quantization") or {}).get("mode"),
            "packed_tensors": fields,
            "mma_shape_eligible": packed_shape_eligible and not embedding,
        })
    result = {
        "schema": "official-bonsai-packed-inventory/v1",
        "repo_id": REPOS["bonsai"][0], "revision": REPOS["bonsai"][1],
        "path": str(path),
        "weights_sha256": sha256(path / "model.safetensors"),
        "weight_size_bytes": (path / "model.safetensors").stat().st_size,
        "safetensors_tensor_count": len(tensors),
        "declared_module_count": len(rows), "modules": rows,
        "missing_module_weight_tensors": missing,
        "mma_shape_eligible_non_embedding_count": sum(r["mma_shape_eligible"] for r in rows),
        "mma_shape_eligibility_source": {
            "path": str(mma_source), "sha256": sha256(mma_source),
            "constants": mma_constants,
            "predicate": "bits supported, group size supported, N >= N_MIN, and K divisible by 512",
            "dispatch_install": "load_target calls mlx_qmm_mma.install(target.model) after Prism load",
        },
        "all_declared_modules_have_weight_tensors": not missing,
    }
    out = EVIDENCE / "integrity/official-bonsai-inventory.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if missing:
        raise RuntimeError(f"Prism pack declares modules with missing weight tensors: {missing[:16]}")
    return result


def official_loader_proof() -> dict:
    # This mode is intentionally isolated from preflight: it is the exact B0 full-donor proof.
    sys.path.insert(0, str(ROOT / "src"))
    import mlx.core as mx
    from mlx_dspark.load import load_target
    path = checkpoint_path("bonsai")
    target, tokenizer = load_target(str(path))
    ids = mx.array([[int(tokenizer.eos_token_id or 0)]], dtype=mx.int32)
    logits = target.model(ids)
    mx.eval(logits)
    if not bool(mx.all(mx.isfinite(logits)).item()):
        raise RuntimeError("official Bonsai B0 output contains NaN/Inf")
    params = target.model.parameters()
    from mlx.utils import tree_flatten
    flat_params = [(path, value) for path, value in tree_flatten(params)
                   if hasattr(value, "dtype")]
    checks = [mx.all(mx.isfinite(value)) for _, value in flat_params]
    mx.eval(checks)
    nonfinite = [path for (path, _), check in zip(flat_params, checks)
                 if not bool(check.item())]
    if nonfinite:
        raise RuntimeError(f"official Bonsai parameters contain NaN/Inf at {nonfinite[:16]}")
    from mlx_dspark import prism_pack
    packed = [(name, module) for name, module in target.model.named_modules()
              if prism_pack.is_packed(module)]
    if not packed:
        raise RuntimeError("official Bonsai loaded without Prism Packed projection modules")
    inventory_path = EVIDENCE / "integrity/official-bonsai-inventory.json"
    inventory = json.loads(inventory_path.read_text())
    if (inventory.get("revision") != REPOS["bonsai"][1] or
            inventory.get("declared_module_count") != len(packed) or
            not inventory.get("all_declared_modules_have_weight_tensors")):
        raise RuntimeError("official Prism config/weight inventory does not match loaded Packed modules")
    record = {
        "schema": "official-bonsai-load/v1", "passed": True,
        "repo_id": REPOS["bonsai"][0], "revision": REPOS["bonsai"][1],
        "resolved_path": str(path), "model_type": "prism_hadamard_qwen35",
        "output_shape": list(logits.shape), "output_dtype": str(logits.dtype),
        "output_finite": True, "parameter_tree_finite": True,
        "packed_projection_count": len(packed),
        "packed_projection_inventory": [
            {"name": name, "bits": int(module.bits), "group_size": int(module.group_size),
             "mode": str(module.mode), "embedding": bool(module.embedding),
             "weight_shape": list(module.weight.shape),
             "scale_shape": list(module.scales.shape),
             "bias_shape": list(module.biases.shape)} for name, module in packed],
        "packed_inventory_evidence": {
            "path": str(inventory_path), "sha256": sha256(inventory_path),
            "revision": inventory["revision"],
            "mma_shape_eligible_non_embedding_count":
                inventory["mma_shape_eligible_non_embedding_count"],
        },
        "mma_routing": {
            "install_invoked_by_load_target": True,
            "existing_installer": "mlx_qmm_mma.install(target.model)",
            "shape_eligibility_inventory": str(inventory_path),
            "live_verified_win_table_captured": False,
        },
        "feature_001_nathansutton_repack_used_as_evidence": False,
        "loader": "src/mlx_dspark/load.py::load_target -> src/mlx_dspark/prism_pack.py::load",
        "runtime": {"mlx": importlib.metadata.version("mlx"),
                    "mlx-lm": importlib.metadata.version("mlx-lm")},
    }
    out = EVIDENCE / "integrity/official-bonsai-load.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def donor_memory_gate() -> dict:
    """T014: measure pruning of the maximum eight-block donor, without loading Qwen."""
    from mlx_dspark.hybrid_target import (
        TargetCompositionRequest, UnsafeDonorResidencyError,
        load_donor_blocks, require_safe_before_qwen,
    )

    donor_repo, donor_revision = REPOS["bonsai"]
    qwen_repo, qwen_revision = REPOS["qwen"]
    request = TargetCompositionRequest.create(
        donor_path=str(checkpoint_path("bonsai")), donor_repo=donor_repo,
        donor_revision=donor_revision, qwen_repo=qwen_repo,
        qwen_revision=qwen_revision, donor_indices=range(56, 64))
    try:
        donor = load_donor_blocks(request)
        pre_qwen = require_safe_before_qwen(donor)
        memory = dict(donor.memory)
        memory["immediately_before_qwen_load"] = pre_qwen
        passed = True
        failure = None
        selected_bytes = donor.selected_parameter_bytes
    except UnsafeDonorResidencyError as error:
        memory = dict(error.memory)
        memory["immediately_before_qwen_load"] = None
        passed = False
        failure = str(error)
        selected_bytes = memory.get("selected_donor_parameter_bytes")
    record = {
        "schema": "qwen-bonsai-donor-memory-release/v1",
        "passed": passed, "qwen_load_attempted": False,
        "donor_repo": donor_repo, "donor_revision": donor_revision,
        "donor_snapshot": str(checkpoint_path("bonsai")),
        "retained_donor_indices": list(range(56, 64)),
        "selected_donor_parameter_bytes": selected_bytes,
        "observations": memory,
        "thresholds": {
            "minimum_active_memory_reclaim_fraction": 0.50,
            "maximum_post_prune_active_fraction_of_donor_peak": 0.50,
            "maximum_post_prune_cache_bytes":
                memory.get("before_donor_load", {}).get("cache_bytes", 0) + 256 * 1024 * 1024,
            "basis": "Eight retained suffix blocks are a minority of the 64-block donor. Require at least half of peak active MLX residency to disappear, record retained parameter bytes, and clear allocator cache before considering Qwen load.",
        },
        "failure": failure,
        "selective_loading_fallback_used": False,
        "provenance": json.loads((EVIDENCE / "checkpoints.json").read_text())["checkpoints"]["bonsai"],
    }
    out = EVIDENCE / "integrity/donor-memory-release.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def _cache_summary(cache) -> list[dict]:
    rows = []
    for index, item in enumerate(cache):
        offset = getattr(item, "offset", None)
        rows.append({"index": index, "class": type(item).__name__,
                     "offset": int(offset) if isinstance(offset, (int, float)) else None})
    return rows


def _finite_parameter_inventory(model) -> tuple[list[dict], list[str], dict[str, int]]:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    flat = [(str(name), value) for name, value in tree_flatten(model.parameters())
            if hasattr(value, "dtype")]
    checks = [mx.all(mx.isfinite(value)) for _, value in flat]
    mx.eval(checks)
    bad = [name for (name, _), check in zip(flat, checks) if not bool(check.item())]
    dtype_counts: dict[str, int] = {}
    inventory = []
    for name, value in flat:
        dtype = str(value.dtype)
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        inventory.append({"name": name, "shape": list(value.shape), "dtype": dtype})
    return inventory, bad, dtype_counts


def integrity_smoke(variant: str, *, run_id: str | None = None) -> dict:
    """Load one exact target through Engine and run finite/cache/tap integrity checks."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown integrity target {variant!r}")
    sys.path.insert(0, str(ROOT / "src"))
    import mlx.core as mx
    from mlx_dspark.hybrid_target import TargetCompositionRequest, layer_family, packed_module_paths
    from mlx_dspark.server import Engine

    checkpoints = json.loads((EVIDENCE / "checkpoints.json").read_text())["checkpoints"]
    qwen = checkpoints["qwen"]
    donor = checkpoints["bonsai"]
    dflash = checkpoints["dflash"]
    indices = VARIANTS[variant]
    is_b0 = variant == "B0"
    model_path = donor["resolved_path"] if is_b0 else qwen["resolved_path"]
    target_composition = None
    if indices:
        target_composition = TargetCompositionRequest.create(
            donor_path=donor["resolved_path"], donor_repo=donor["repo_id"],
            donor_revision=donor["revision"], qwen_repo=qwen["repo_id"],
            qwen_revision=qwen["revision"], donor_indices=indices)
    dflash_config = json.loads((Path(dflash["resolved_path"]) / "config.json").read_text())
    taps = list(dflash_config.get("dflash_config", {}).get("target_layer_ids", []))
    if not taps:
        raise RuntimeError("pinned original DFlash config has no target_layer_ids")
    engine = None
    try:
        engine = Engine.load(
            mode="dflash", model=model_path, drafter=dflash["resolved_path"],
            target_composition=target_composition, drafter_bits=4,
            max_draft_tokens=None, enable_thinking=False, prefix_cache=False,
            kv_bits=None, context_window=None, warmup=False, memory_guard=False,
            lookup_drafts=False, small_m=False, sdpa_split=False,
            wide_gemm_min=0, cpu_split=0)
        loaded_taps = list(engine.drafter.config.target_layer_ids)
        if loaded_taps != taps:
            raise RuntimeError(
                f"loaded DFlash tap IDs {loaded_taps} differ from pinned config {taps}")
        target = engine.target
        model = target.model
        if indices is None:
            block_owners = ["bonsai"] * NUM_LAYERS
            manifest = None
            role_owners = {"embedding": "bonsai", "final_norm": "bonsai", "lm_head": "bonsai"}
        elif indices:
            manifest = target.composition_manifest
            block_owners = [item["owner"] for item in manifest.as_dict()["block_owners"]]
            role_owners = {"embedding": manifest.embedding_owner,
                           "final_norm": manifest.final_norm_owner,
                           "lm_head": manifest.lm_head_owner}
        else:
            manifest = None
            block_owners = ["qwen"] * NUM_LAYERS
            role_owners = {"embedding": "qwen", "final_norm": "qwen", "lm_head": "qwen"}
        families = [layer_family(layer) for layer in model.layers]
        if len(block_owners) != NUM_LAYERS or len(families) != NUM_LAYERS:
            raise RuntimeError("loaded target does not contain exactly 64 decoder blocks")
        if any(owner != ("bonsai" if i in indices else "qwen")
               for i, owner in enumerate(block_owners)) if indices else False:
            raise RuntimeError("loaded target block owner map differs from requested composition")
        if not is_b0 and not indices and any(owner != "qwen" for owner in block_owners):
            raise RuntimeError("H0 ordinary Qwen path contains a non-Qwen block owner")

        parameter_inventory, nonfinite_parameters, dtype_counts = _finite_parameter_inventory(model)
        actual_packed = list(packed_module_paths(model))
        if indices and not actual_packed:
            raise RuntimeError("composed target has no live Prism Packed projections")
        input_ids = [int(token) for token in engine.tokenizer.encode(
            "Say ready.", add_special_tokens=False)]
        if not input_ids:
            raise RuntimeError("integrity prompt produced no token IDs")
        prompt = mx.array([[input_ids[-1]]], dtype=mx.int32)
        cache = target.make_cache()
        cache_before = _cache_summary(cache)
        prefill_logits, prefill_tap = target.prefill(
            prompt, cache, tap=taps, want_logits=True)
        mx.eval(prefill_logits, prefill_tap)
        if not bool(mx.all(mx.isfinite(prefill_logits)).item()) or not bool(
                mx.all(mx.isfinite(prefill_tap)).item()):
            raise RuntimeError("target prefill logits or tap output is nonfinite")
        cache_after_prefill = _cache_summary(cache)

        verify_ids_list = [input_ids[-1], input_ids[-2] if len(input_ids) > 1 else input_ids[-1],
                           input_ids[-3] if len(input_ids) > 2 else input_ids[-1]]
        verify_ids = mx.array([verify_ids_list], dtype=mx.int32)
        verify_logits, verify_tap = target.verify(verify_ids, cache, taps)
        mx.eval(verify_logits, verify_tap)
        if not bool(mx.all(mx.isfinite(verify_logits)).item()) or not bool(
                mx.all(mx.isfinite(verify_tap)).item()):
            raise RuntimeError("target verify logits or tap output is nonfinite")
        cache_after_verify = _cache_summary(cache)
        target.rollback(cache, n_rejected=1, accepted=[verify_ids_list[1]])
        cache_after_rollback = _cache_summary(cache)
        kv_offsets = lambda rows: [row["offset"] for row in rows if row["offset"] is not None]
        before_offsets = kv_offsets(cache_after_prefill)
        verified_offsets = kv_offsets(cache_after_verify)
        rolled_offsets = kv_offsets(cache_after_rollback)
        cache_advanced = bool(verified_offsets and before_offsets and
                              max(verified_offsets) > max(before_offsets))
        rollback_valid = bool(rolled_offsets and verified_offsets and
                              max(rolled_offsets) == max(verified_offsets) - 1 and
                              max(rolled_offsets) > max(before_offsets))
        if not cache_advanced or not rollback_valid:
            raise RuntimeError("cache advancement or one-token rejected-suffix rollback failed")

        transition_edges = [i for i in range(NUM_LAYERS - 1)
                            if block_owners[i] != block_owners[i + 1]]
        boundary_layers = sorted({layer for edge in transition_edges for layer in (edge, edge + 1)})
        boundaries = {"layers": boundary_layers, "edges": [], "outputs_finite": True}
        if boundary_layers:
            boundary_cache = target.make_cache()
            boundary_logits, boundary_fused = target.run(prompt, boundary_cache, boundary_layers)
            mx.eval(boundary_logits, boundary_fused)
            boundaries["outputs_finite"] = bool(mx.all(mx.isfinite(boundary_logits)).item()) and bool(
                mx.all(mx.isfinite(boundary_fused)).item())
            boundaries["fused_shape"] = list(boundary_fused.shape)
            boundaries["edges"] = [{
                "from_block": edge, "from_owner": block_owners[edge],
                "to_block": edge + 1, "to_owner": block_owners[edge + 1],
                "direct_residual_stream": True,
            } for edge in transition_edges]
            if not boundaries["outputs_finite"]:
                raise RuntimeError("Qwen/Bonsai representation boundary produced nonfinite output")

        directly_replaced_taps = [tap for tap in taps if block_owners[tap] == "bonsai"]
        if int(prefill_tap.shape[-1]) != len(taps) * 5120:
            raise RuntimeError("loaded DFlash tap output width does not match tap count × hidden width")
        result = engine.generate(input_ids, max_tokens=2, temperature=0.0,
                                 top_p=1.0, top_k=0, stop=None, seed=0)
        if not result.token_ids or result.num_tokens != len(result.token_ids):
            raise RuntimeError("short speculative semantic smoke did not generate token IDs")

        record = {
            "schema": "qwen-bonsai-target-integrity/v1", "passed": True,
            "variant": variant, "run_id": run_id or "primary",
            "target_mode": "separate full Bonsai endpoint" if is_b0 else
                "ordinary Qwen H0" if not indices else "opt-in Qwen/Bonsai composition",
            "checkpoints": {"qwen": qwen, "bonsai": donor, "dflash": dflash},
            "ownership": {
                "block_owners": [{"index": i, "owner": owner, "layer_family": families[i]}
                                 for i, owner in enumerate(block_owners)],
                "embedding_owner": role_owners["embedding"],
                "final_norm_owner": role_owners["final_norm"],
                "lm_head_owner": role_owners["lm_head"],
                "composition_manifest": manifest.as_dict() if manifest is not None else None,
                "composition_manifest_canonical_json": (
                    manifest.canonical_json() if manifest is not None else None),
            },
            "parameters": {"inventory": parameter_inventory, "dtype_counts": dtype_counts,
                           "nonfinite_names": nonfinite_parameters,
                           "all_finite": not nonfinite_parameters},
            "packed_donor_modules": {"count": len(actual_packed), "paths": actual_packed,
                                     "present": bool(actual_packed) if indices or is_b0 else None},
            "dflash_taps": {
                "repo_id": dflash["repo_id"], "revision": dflash["revision"],
                "target_layer_ids": taps, "loaded_target_layer_ids": loaded_taps,
                "semantics": "zero-based block output",
                "output_hidden_width": int(prefill_tap.shape[-1]),
                "directly_replaced_tap_ids": directly_replaced_taps,
                "directly_replaced_tap_count": len(directly_replaced_taps),
                "binding_mode": "original DFlash bound to loaded target",
            },
            "representations": {
                "projection_transforms_local": True,
                "canonical_residual_boundaries": True,
                "bridge_required": False,
                "boundary_smoke": boundaries,
            },
            "cache_rollback": {
                "plain_kv_requested": True, "cache_before": cache_before,
                "cache_after_prefill": cache_after_prefill,
                "cache_after_verify": cache_after_verify,
                "cache_after_rollback": cache_after_rollback,
                "cache_advanced": cache_advanced,
                "accepted_prefix_rejected_suffix_exercised": True,
                "rejected_suffix_tokens": 1, "rollback_valid": rollback_valid,
            },
            "generation_smoke": {"prompt_ids": input_ids, "generated_token_ids": result.token_ids,
                                  "settings": {"max_tokens": 2, "temperature": 0.0,
                                               "top_p": 1.0, "top_k": 0, "seed": 0},
                                  "num_tokens": result.num_tokens,
                                  "target_forwards": result.target_forwards,
                                  "accept_lengths": result.accept_lengths,
                                  "finish_reason": result.finish_reason},
            "donor_memory_evidence": getattr(target, "donor_memory_evidence", None),
            "runtime_revision": git_state(),
        }
        out_name = f"{variant.lower()}{'-' + run_id if run_id else ''}.json"
        out = EVIDENCE / "integrity" / out_name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        return record
    finally:
        if engine is not None:
            engine.close()


def repeatability_check() -> dict:
    """Re-run H0 and H2 in fresh Python processes; compare behavior/manifest only."""
    h0 = EVIDENCE / "integrity/h0-loaded.json"
    h2 = EVIDENCE / "integrity/h2-loaded.json"
    if not h0.is_file() or not h2.is_file():
        raise FileNotFoundError("primary H0 and H2 integrity records are required for T018")
    child_records = {}
    for variant in ("H0", "H2"):
        subprocess.run([sys.executable, str(Path(__file__).resolve()),
                        "--integrity", "--variant", variant, "--run-id", "fresh"],
                       cwd=ROOT, check=True)
        child_path = EVIDENCE / "integrity" / f"{variant.lower()}-fresh.json"
        child_records[variant] = json.loads(child_path.read_text())
    base_h0 = json.loads(h0.read_text())
    base_h2 = json.loads(h2.read_text())
    repeat_h0 = child_records["H0"]
    repeat_h2 = child_records["H2"]
    h0_same = base_h0["generation_smoke"]["generated_token_ids"] == \
        repeat_h0["generation_smoke"]["generated_token_ids"]
    h2_same = base_h2["ownership"]["composition_manifest_canonical_json"] == \
        repeat_h2["ownership"]["composition_manifest_canonical_json"]
    result = {
        "schema": "qwen-bonsai-fresh-process-repeatability/v1",
        "passed": h0_same and h2_same and
            base_h0["generation_smoke"]["prompt_ids"] ==
            repeat_h0["generation_smoke"]["prompt_ids"] and
            base_h0["generation_smoke"]["settings"] ==
            repeat_h0["generation_smoke"]["settings"],
        "h0_same_revision_ordinary_path_generated_ids_match": h0_same,
        "h0_same_prompt_ids": base_h0["generation_smoke"]["prompt_ids"] ==
            repeat_h0["generation_smoke"]["prompt_ids"],
        "h0_same_generation_settings": base_h0["generation_smoke"]["settings"] ==
            repeat_h0["generation_smoke"]["settings"],
        "h0_base_ids": base_h0["generation_smoke"]["generated_token_ids"],
        "h0_fresh_ids": repeat_h0["generation_smoke"]["generated_token_ids"],
        "h2_fresh_process_manifest_repeatable": h2_same,
        "h2_base_manifest": base_h2["ownership"]["composition_manifest_canonical_json"],
        "h2_fresh_manifest": repeat_h2["ownership"]["composition_manifest_canonical_json"],
        "hybrid_logits_compared_to_qwen": False,
        "rollback_probe_scope": "single accepted-prefix/rejected-suffix integrity exercise",
    }
    out = EVIDENCE / "integrity/repeatability.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def record_runtime_semantics() -> dict:
    """T016: validate loaded tap ids/counts and bind source semantics to this runtime."""
    expected_counts = {"H0": 0, "H1a": 0, "H1b": 0, "H1c": 0,
                       "H2": 1, "H3": 1, "B0": 5}
    records = {}
    for variant in VARIANTS:
        path = EVIDENCE / "integrity" / f"{variant.lower()}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing required integrity record for {variant}: {path}")
        record = json.loads(path.read_text())
        if not record.get("passed"):
            raise RuntimeError(f"integrity record failed for {variant}")
        records[variant] = record
    tap_sets = {tuple(row["dflash_taps"]["target_layer_ids"]) for row in records.values()}
    if len(tap_sets) != 1:
        raise RuntimeError(f"loaded DFlash tap IDs changed across variants: {tap_sets}")
    taps = list(next(iter(tap_sets)))
    loaded_probe_records = {}
    for variant in ("H0", "H2"):
        path = EVIDENCE / "integrity" / f"{variant.lower()}-loaded.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing loaded-config verification for {variant}: {path}")
        loaded_probe_records[variant] = json.loads(path.read_text())
        if not loaded_probe_records[variant].get("passed"):
            raise RuntimeError(f"loaded-config verification failed for {variant}")
    loaded_tap_sets = {
        tuple(row["dflash_taps"].get("loaded_target_layer_ids", []))
        for row in [*records.values(), *loaded_probe_records.values()]
        if row["dflash_taps"].get("loaded_target_layer_ids") is not None
    }
    if not loaded_tap_sets:
        raise RuntimeError("no integrity record captured the loaded DFlash config tap IDs")
    if loaded_tap_sets != {tuple(taps)}:
        raise RuntimeError(f"loaded DFlash tap IDs differ from expected pinned config: {loaded_tap_sets}")
    actual_counts = {variant: row["dflash_taps"]["directly_replaced_tap_count"]
                     for variant, row in records.items()}
    actual_tap_ids = {variant: row["dflash_taps"]["directly_replaced_tap_ids"]
                      for variant, row in records.items()}
    if actual_counts != expected_counts:
        raise RuntimeError(f"loaded direct tap counts differ from current ownership: {actual_counts}")
    required_sources = {
        "target": ROOT / "src/mlx_dspark/target.py",
        "prism_pack": ROOT / "src/mlx_dspark/prism_pack.py",
        "qwen3_5": ROOT / ".venv/lib/python3.12/site-packages/mlx_lm/models/qwen3_5.py",
        "dflash_config": Path(records["H0"]["checkpoints"]["dflash"]["resolved_path"]) / "config.json",
    }
    source_facts = {name: {"path": str(path), "sha256": sha256(path)}
                    for name, path in required_sources.items()}
    result = {
        "schema": "qwen-bonsai-runtime-semantics/v1", "passed": True,
        "runtime_revision": git_state(),
        "checkpoints": {name: row["checkpoints"] for name, row in records.items()},
        "loaded_dflash": {
            "repo_id": records["H0"]["checkpoints"]["dflash"]["repo_id"],
            "revision": records["H0"]["checkpoints"]["dflash"]["revision"],
            "target_layer_ids": taps, "loaded_target_layer_ids": list(next(iter(loaded_tap_sets))),
            "loaded_config_verification_records": {
                variant: {"path": str(EVIDENCE / "integrity" / f"{variant.lower()}-loaded.json"),
                          "loaded_target_layer_ids": row["dflash_taps"]["loaded_target_layer_ids"],
                          "passed": row["passed"]}
                for variant, row in loaded_probe_records.items()
            },
            "semantics": "zero-based block output, captured immediately after layer(...) returns",
            "direct_bonsai_tap_counts": actual_counts,
            "direct_bonsai_tap_ids": actual_tap_ids,
        },
        "representation": {
            "prism_projection_transform_scope": "Packed.__call__ rotates each projection input locally; no rotation persists across a block boundary",
            "block_residual_output": "Qwen3_5 DecoderLayer returns x + attention_or_gdn + mlp residual; target feeds it directly to the next layer",
            "bridge_required": False,
            "actual_boundary_smokes": {
                variant: row["representations"]["boundary_smoke"] for variant, row in records.items()
            },
        },
        "actual_loaded_owner_counts": {
            variant: {"qwen_blocks": sum(x["owner"] == "qwen" for x in row["ownership"]["block_owners"]),
                      "bonsai_blocks": sum(x["owner"] == "bonsai" for x in row["ownership"]["block_owners"]),
                      "packed_donor_modules": row["packed_donor_modules"]["count"]}
            for variant, row in records.items()
        },
        "source_evidence": source_facts,
        "feature_001_repack_used": False,
        "series_a_discovery_started": False,
    }
    out = EVIDENCE / "integrity/runtime-semantics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def finalize_integrity_index() -> dict:
    """T019: mechanically close the integrity prerequisite set after T016/T018 pass."""
    semantic_path = EVIDENCE / "integrity/runtime-semantics.json"
    repeat_path = EVIDENCE / "integrity/repeatability.json"
    semantic = json.loads(semantic_path.read_text())
    repeat = json.loads(repeat_path.read_text())
    rows = {}
    for variant in VARIANTS:
        path = EVIDENCE / "integrity" / f"{variant.lower()}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing integrity record for {variant}")
        record = json.loads(path.read_text())
        if not record.get("passed"):
            raise RuntimeError(f"integrity record failed for {variant}")
        rows[variant] = {"path": str(path), "sha256": sha256(path), "passed": True}
    passed = bool(semantic.get("passed") and repeat.get("passed") and
                  all(row["passed"] for row in rows.values()))
    result = {
        "schema": "qwen-bonsai-integrity-index/v1", "status": "PASS" if passed else "FAIL",
        "passed": passed, "required_targets": list(VARIANTS),
        "targets": rows,
        "runtime_semantics": {"path": str(semantic_path), "sha256": sha256(semantic_path),
                              "passed": semantic.get("passed")},
        "repeatability": {"path": str(repeat_path), "sha256": sha256(repeat_path),
                          "passed": repeat.get("passed")},
        "series_a_discovery_unblocked": passed,
        "series_a_discovery_started": False,
    }
    out = EVIDENCE / "integrity/index.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not passed:
        raise RuntimeError("T019 integrity prerequisite set did not pass")
    return result


def _canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def ensure_series_a_corpus() -> dict:
    """Freeze the feature-local prompt and generation controls before discovery."""
    directory = EVIDENCE / "series-a"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "prompts.json"
    corpus = {"schema": "qwen-bonsai-series-a-prompts/v1",
              "prompts": list(SERIES_A_PROMPTS),
              "generation": dict(SERIES_A_GENERATION),
              "prompt_encoding": "existing encode_messages with enable_thinking=False"}
    encoded = json.dumps(corpus, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != corpus:
            raise RuntimeError(f"frozen Series-A prompt corpus differs from runner controls: {path}")
    else:
        path.write_text(encoded)
    return {"path": str(path), "sha256": sha256(path), "prompt_count": len(corpus["prompts"]),
            "generation": corpus["generation"]}


def _validate_series_a_prerequisites() -> dict:
    index_path = EVIDENCE / "integrity/index.json"
    index = json.loads(index_path.read_text())
    if index.get("status") != "PASS" or not index.get("passed") or \
            not index.get("series_a_discovery_unblocked"):
        raise RuntimeError("T019 integrity index is not PASS/unblocked; refusing Series-A load")
    for variant in (*VARIANTS,):
        entry = index["targets"][variant]
        path = Path(entry["path"])
        if not path.is_file() or sha256(path) != entry["sha256"]:
            raise RuntimeError(f"T019 integrity evidence is missing/stale for {variant}: {path}")
        record = json.loads(path.read_text())
        if not record.get("passed"):
            raise RuntimeError(f"integrity smoke did not pass for {variant}")
    for key in ("runtime_semantics", "repeatability"):
        entry = index[key]
        path = Path(entry["path"])
        if not entry.get("passed") or not path.is_file() or sha256(path) != entry["sha256"]:
            raise RuntimeError(f"T019 {key} evidence is missing/stale: {path}")
    checkpoints = json.loads((EVIDENCE / "checkpoints.json").read_text())["checkpoints"]
    for name in REPOS:
        selected = checkpoints[name]
        if selected["repo_id"] != REPOS[name][0] or selected["revision"] != REPOS[name][1]:
            raise RuntimeError(f"accepted immutable checkpoint mismatch for {name}")
    return {"index": index, "checkpoints": checkpoints}


def _series_runtime_manifest(mx) -> dict:
    runtime_files = [
        ROOT / "src/mlx_dspark/hybrid_target.py", ROOT / "src/mlx_dspark/load.py",
        ROOT / "src/mlx_dspark/server.py", ROOT / "src/mlx_dspark/generate.py",
        ROOT / "src/mlx_dspark/target.py", ROOT / "src/mlx_dspark/prism_pack.py",
        ROOT / "src/mlx_dspark/mlx_qmm_mma.py",
        ROOT / "specs/002-qwen-bonsai-hybrid-target/evidence/probes/series_a.py",
    ]
    dirty_runtime = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--", *[str(p.relative_to(ROOT)) for p in runtime_files]],
        check=True, capture_output=True, text=True).stdout.splitlines()
    device = mx.device_info()
    if "m4 pro" not in str(device).lower():
        raise RuntimeError(f"Series-A target must be the specified M4 Pro; MLX reports {device!r}")
    return {
        "git": {"head": git_state()["head"], "working_tree_dirty": bool(dirty_runtime),
                "dirty_runtime_paths": dirty_runtime},
        "runtime_source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in runtime_files},
        "python": sys.version,
        "packages": {name: importlib.metadata.version(name)
                     for name in ("mlx", "mlx-lm", "safetensors", "huggingface-hub")},
        "os": platform.platform(), "machine_architecture": platform.machine(),
        "mlx_device": device,
    }


def run_series_condition(variant: str, run_id: str, order_index: int,
                         comparison_group: str = "discovery") -> dict:
    """Measure one target in a fresh process using the production DFlash Engine path."""
    from collections import Counter
    sys.path.insert(0, str(ROOT / "src"))
    import mlx.core as mx
    from mlx_dspark.generate import encode_messages, greedy_generate
    from mlx_dspark.hybrid_target import TargetCompositionRequest
    from mlx_dspark.server import Engine

    prerequisites = _validate_series_a_prerequisites()
    corpus_ref = ensure_series_a_corpus()
    checkpoints = prerequisites["checkpoints"]
    qwen, donor, dflash = (checkpoints[k] for k in ("qwen", "bonsai", "dflash"))
    indices = VARIANTS[variant]
    is_b0 = variant == "B0"
    model_path = donor["resolved_path"] if is_b0 else qwen["resolved_path"]
    composition = None
    if indices:
        composition = TargetCompositionRequest.create(
            donor_path=donor["resolved_path"], donor_repo=donor["repo_id"],
            donor_revision=donor["revision"], qwen_repo=qwen["repo_id"],
            qwen_revision=qwen["revision"], donor_indices=indices)
    settings = dict(SERIES_A_GENERATION)
    engine = None
    start_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        engine = Engine.load(
            mode="dflash", model=model_path, drafter=dflash["resolved_path"],
            target_composition=composition, drafter_bits=4,
            max_draft_tokens="auto", enable_thinking=False, prefix_cache=False,
            kv_bits=None, context_window=None, warmup=True, memory_guard=False,
            lookup_drafts=False, small_m=None, sdpa_split=None,
            wide_gemm_min=None, cpu_split=None)

        if engine.mode != "dflash" or engine.drafter is None or not engine.cap_controller:
            raise RuntimeError("Series A requires the original DFlash drafter and production CapController")
        if engine.max_draft_tokens is not None:
            raise RuntimeError("auto CapController did not retain the ordinary derived-cap path")
        target_record = json.loads((EVIDENCE / "integrity" / f"{variant.lower()}.json").read_text())
        if not target_record.get("passed"):
            raise RuntimeError(f"T019 target integrity record failed for {variant}")
        expected_taps = [5, 19, 33, 47, 61]
        loaded_taps = list(engine.drafter.config.target_layer_ids)
        if loaded_taps != expected_taps:
            raise RuntimeError(f"loaded DFlash taps changed after T019: {loaded_taps}")

        prompt_rows = []
        for prompt in SERIES_A_PROMPTS:
            ids = encode_messages(engine.tokenizer, [{"role": "user", "content": prompt["text"]}],
                                  enable_thinking=False)
            if not ids:
                raise RuntimeError(f"empty tokenized prompt {prompt['id']}")
            prompt_rows.append({"id": prompt["id"], "text": prompt["text"],
                                "input_ids": [int(i) for i in ids],
                                "input_ids_sha256": hashlib.sha256(
                                    _canonical_json([int(i) for i in ids]).encode()).hexdigest()})
        runtime = _series_runtime_manifest(mx)
        active_before = int(mx.get_active_memory())
        cache_before = int(mx.get_cache_memory())
        mx.reset_peak_memory()

        serial_rows, speculative_rows = [], []
        round_events = []
        for prompt in prompt_rows:
            ids = prompt["input_ids"]
            serial_result = engine._executor.submit(
                greedy_generate, engine.target, engine.tokenizer, prompt_ids=ids,
                max_new_tokens=settings["max_new_tokens"], temperature=settings["temperature"],
                top_p=settings["top_p"], top_k=settings["top_k"], seed=settings["seed"],
                stop=settings["stop"], presence_penalty=settings["presence_penalty"],
                frequency_penalty=settings["frequency_penalty"]).result()
            serial_rows.append({"prompt_id": prompt["id"], "prompt_ids_sha256": prompt["input_ids_sha256"],
                                "generated_token_ids": serial_result.token_ids,
                                "generated_token_count": serial_result.num_tokens,
                                "target_forwards": serial_result.target_forwards,
                                "seconds": serial_result.seconds,
                                "prefill_seconds": serial_result.prefill_seconds,
                                "decode_seconds": serial_result.decode_seconds,
                                "decode_tokens_per_sec": serial_result.decode_tokens_per_sec,
                                "finish_reason": serial_result.finish_reason})

            engine.rounds.reset()
            speculative_result = engine.generate(
                ids, max_tokens=settings["max_new_tokens"], temperature=settings["temperature"],
                top_p=settings["top_p"], top_k=settings["top_k"], stop=settings["stop"],
                seed=settings["seed"], presence_penalty=settings["presence_penalty"],
                frequency_penalty=settings["frequency_penalty"])
            events = engine.rounds.snapshot()
            round_events.extend(events)
            speculative_rows.append({
                "prompt_id": prompt["id"], "prompt_ids_sha256": prompt["input_ids_sha256"],
                "generated_token_ids": speculative_result.token_ids,
                "generated_token_count": speculative_result.num_tokens,
                "num_rounds": speculative_result.num_rounds,
                "accept_lengths": speculative_result.accept_lengths,
                "mean_accept_len": speculative_result.mean_accept_len,
                "target_forwards": speculative_result.target_forwards,
                "seconds": speculative_result.seconds,
                "prefill_seconds": speculative_result.prefill_seconds,
                "decode_seconds": speculative_result.decode_seconds,
                "decode_tokens_per_sec": speculative_result.decode_tokens_per_sec,
                "finish_reason": speculative_result.finish_reason,
                "cap_controller": engine.cap_controller.info(),
            })
            if not speculative_result.token_ids:
                raise RuntimeError(f"speculative generation produced no tokens for {prompt['id']}")

        active_after = int(mx.get_active_memory())
        cache_after = int(mx.get_cache_memory())
        peak_steady = int(mx.get_peak_memory())
        serial_tokens = sum(row["generated_token_count"] for row in serial_rows)
        serial_decode_s = sum(row["decode_seconds"] for row in serial_rows)
        spec_tokens = sum(row["generated_token_count"] for row in speculative_rows)
        spec_decode_s = sum(row["decode_seconds"] for row in speculative_rows)
        serial_tps = serial_tokens / max(serial_decode_s, 1e-9)
        speculative_tps = spec_tokens / max(spec_decode_s, 1e-9)
        events_by_cap = Counter(str(row.get("cap")) for row in round_events)
        events_by_drafted = Counter(str(row.get("drafted")) for row in round_events)
        accept_hist = Counter(str(row.get("committed")) for row in round_events)
        drafted_total = sum(int(row.get("drafted", 0)) for row in round_events)
        accepted_total = sum(int(row.get("accepted", 0)) for row in round_events)
        accept_len = spec_tokens / max(sum(row["num_rounds"] for row in speculative_rows), 1)
        integrity_path = EVIDENCE / "integrity" / f"{variant.lower()}.json"
        checkpoint_refs = {k: {"repo_id": checkpoints[k]["repo_id"],
                               "revision": checkpoints[k]["revision"],
                               "resolved_path": checkpoints[k]["resolved_path"],
                               "config_sha256": checkpoints[k]["config"]["sha256"],
                               "weights": checkpoints[k]["weights"]}
                           for k in ("qwen", "bonsai", "dflash")}
        owner = target_record["ownership"]
        result = {
            "schema": "qwen-bonsai-series-a-run/v1", "series": "A",
            "status": "PASS", "run_id": run_id, "variant": variant,
            "comparison_group": comparison_group, "order_index": order_index,
            "invocation": {"argv": [sys.executable, str(Path(__file__).resolve()),
                                      "--run-condition", "--variant", variant,
                                      "--run-id", run_id, "--order-index", str(order_index),
                                      "--comparison-group", comparison_group]},
            "started_utc": start_utc,
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "target_composition": {
                "donor_block_indices": list(indices or range(64)) if is_b0 else list(indices),
                "block_owners": owner["block_owners"],
                "embedding_owner": owner["embedding_owner"],
                "final_norm_owner": owner["final_norm_owner"],
                "lm_head_owner": owner["lm_head_owner"],
                "composition_manifest_sha256": hashlib.sha256(
                    (owner.get("composition_manifest_canonical_json") or "").encode()).hexdigest(),
            },
            "checkpoints": checkpoint_refs,
            "integrity_ref": {"path": str(integrity_path), "sha256": sha256(integrity_path)},
            "runtime": runtime,
            "controls": {
                "mode": "dflash", "drafter_repo": dflash["repo_id"],
                "drafter_revision": dflash["revision"], "drafter_bits": 4,
                "controller": "ordinary production CapController, max_draft_tokens=auto",
                "controller_final": engine.cap_controller.info(),
                "plain_kv": True, "kv_bits": None,
                "lookup_drafts": False, "prefix_cache": False,
                "warmup": True, "memory_guard": False,
                "small_m": None, "sdpa_split": None,
                "wide_gemm_min": None, "cpu_split": None,
                "generation": settings,
                "prompt_corpus": corpus_ref,
                "prompt_ids": prompt_rows,
                "measurement_protocol": "one fresh process and Engine per run; Engine load warmup; serial greedy then DFlash speculative on same target; no prefix/KV reuse between prompts",
            },
            "measured": {
                "serial_target": {"generated_tokens": serial_tokens,
                                   "decode_seconds": serial_decode_s,
                                   "tokens_per_sec": serial_tps,
                                   "target_forwards": sum(row["target_forwards"] for row in serial_rows),
                                   "per_prompt": serial_rows},
                "speculative": {"generated_tokens": spec_tokens,
                                "decode_seconds": spec_decode_s,
                                "decode_tokens_per_sec": speculative_tps,
                                "speedup_over_serial": speculative_tps / max(serial_tps, 1e-9),
                                "target_forwards": sum(row["target_forwards"] for row in speculative_rows),
                                "generated_tokens_per_target_forward": spec_tokens / max(
                                    sum(row["target_forwards"] for row in speculative_rows), 1),
                                "mean_accept_len": accept_len,
                                "draft_acceptance": accepted_total / max(drafted_total, 1),
                                "accept_histogram": dict(sorted(accept_hist.items(), key=lambda x: int(x[0]))),
                                "cap_distribution": dict(sorted(events_by_cap.items())),
                                "draft_width_distribution": dict(sorted(events_by_drafted.items())),
                                "accept_lengths": [value for row in speculative_rows
                                                   for value in row["accept_lengths"]],
                                "round_events": round_events,
                                "per_prompt": speculative_rows},
                "peak_steady_state_memory": {"active_before": active_before,
                                              "cache_before": cache_before,
                                              "active_after": active_after,
                                              "cache_after": cache_after,
                                              "peak_bytes_after_load_warmup_reset": peak_steady},
                "generated_token_count": spec_tokens,
                "measurement_duration_seconds": sum(
                    row["seconds"] for row in [*serial_rows, *speculative_rows]),
                "decode_duration_seconds": serial_decode_s + spec_decode_s,
                "wall_duration_seconds": None,
                "component_timing": None,
                "component_timing_unavailable_reason":
                    "the runtime exposes round timing only; no invasive component profiler added",
            },
            "integrity": {"target_integrity_passed": target_record["passed"],
                          "all_generated_outputs_nonempty": bool(spec_tokens),
                          "serial_and_spec_prompt_ids_match": True},
        }
        result["measured"]["wall_duration_seconds"] = result["measured"]["measurement_duration_seconds"]
        outdir = EVIDENCE / "series-a/runs" / run_id
        outdir.mkdir(parents=True, exist_ok=False)
        out = outdir / "result.json"
        out.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
        return {"run_id": run_id, "variant": variant, "path": str(out),
                "sha256": sha256(out), "serial_tps": serial_tps,
                "speculative_tps": speculative_tps, "speedup": result["measured"]["speculative"]["speedup_over_serial"],
                "generated_tokens": spec_tokens, "passed": True}
    finally:
        if engine is not None:
            engine.close()


def discovery() -> dict:
    """Run the fixed eight-condition bracketing order in separate fresh processes."""
    _validate_series_a_prerequisites()
    ensure_series_a_corpus()
    summaries = []
    for order_index, variant in enumerate(SERIES_A_ORDER, start=1):
        run_id = f"discovery-{order_index:02d}-{variant.lower()}-{os.urandom(3).hex()}"
        command = [sys.executable, str(Path(__file__).resolve()), "--run-condition",
                   "--variant", variant, "--run-id", run_id,
                   "--order-index", str(order_index), "--comparison-group", "discovery"]
        subprocess.run(command, cwd=ROOT, check=True)
        path = EVIDENCE / "series-a/runs" / run_id / "result.json"
        row = json.loads(path.read_text())
        summaries.append({"run_id": run_id, "variant": variant, "order_index": order_index,
                          "path": str(path), "sha256": sha256(path),
                          "serial_tps": row["measured"]["serial_target"]["tokens_per_sec"],
                          "decode_tokens_per_sec": row["measured"]["speculative"]["decode_tokens_per_sec"],
                          "speedup": row["measured"]["speculative"]["speedup_over_serial"],
                          "passed": row["status"] == "PASS"})
    result = {"schema": "qwen-bonsai-series-a-discovery-order/v1",
              "order": list(SERIES_A_ORDER), "runs": summaries,
              "prompt_corpus": ensure_series_a_corpus(),
              "series_a_discovery_started": True, "all_runs_passed": all(x["passed"] for x in summaries)}
    out = EVIDENCE / "series-a/discovery-order.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--resolve-inputs", action="store_true")
    group.add_argument("--validate-donor-structure", action="store_true")
    group.add_argument("--record-official-inventory", action="store_true")
    group.add_argument("--official-bonsai-load", action="store_true")
    group.add_argument("--memory-gate", action="store_true")
    group.add_argument("--integrity", action="store_true")
    group.add_argument("--repeatability", action="store_true")
    group.add_argument("--runtime-semantics", action="store_true")
    group.add_argument("--finalize-integrity", action="store_true")
    group.add_argument("--discovery", action="store_true")
    group.add_argument("--run-condition", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--variant", choices=tuple(VARIANTS))
    parser.add_argument("--run-id")
    parser.add_argument("--order-index", type=int)
    parser.add_argument("--comparison-group", default="discovery")
    parser.add_argument("--order", default=",".join(SERIES_A_ORDER))
    args = parser.parse_args()
    if args.preflight:
        result = preflight()
    elif args.resolve_inputs:
        result = resolve_inputs()
    elif args.validate_donor_structure:
        result = validate_donor_structure()
    elif args.record_official_inventory:
        result = record_official_pack_inventory()
    elif args.memory_gate:
        result = donor_memory_gate()
    elif args.integrity:
        if args.variant is None:
            parser.error("--integrity requires --variant")
        result = integrity_smoke(args.variant, run_id=args.run_id)
    elif args.repeatability:
        result = repeatability_check()
    elif args.runtime_semantics:
        result = record_runtime_semantics()
    elif args.finalize_integrity:
        result = finalize_integrity_index()
    elif args.discovery:
        requested_order = tuple(x.strip() for x in args.order.split(",") if x.strip())
        if requested_order != SERIES_A_ORDER:
            parser.error(f"--discovery order must be exactly {','.join(SERIES_A_ORDER)}")
        result = discovery()
    elif args.run_condition:
        if args.variant is None or not args.run_id or args.order_index is None:
            parser.error("--run-condition requires --variant, --run-id, and --order-index")
        result = run_series_condition(args.variant, args.run_id, args.order_index,
                                     args.comparison_group)
    else:
        result = official_loader_proof()
    if args.integrity:
        print(json.dumps({
            "passed": result["passed"], "variant": result["variant"],
            "record": str((EVIDENCE / "integrity" /
                           f"{args.variant.lower()}{'-' + args.run_id if args.run_id else ''}.json").resolve()),
            "directly_replaced_tap_count": result["dflash_taps"]["directly_replaced_tap_count"],
            "packed_module_count": result["packed_donor_modules"]["count"],
            "rollback_valid": result["cache_rollback"]["rollback_valid"],
            "generated_token_ids": result["generation_smoke"]["generated_token_ids"],
        }, indent=2, sort_keys=True))
    elif args.repeatability:
        print(json.dumps({"passed": result["passed"],
                          "h0_generated_ids_match": result["h0_same_revision_ordinary_path_generated_ids_match"],
                          "h0_prompt_and_settings_match": result["h0_same_prompt_ids"] and
                              result["h0_same_generation_settings"],
                          "h2_manifest_repeatable": result["h2_fresh_process_manifest_repeatable"],
                          "record": str((EVIDENCE / "integrity/repeatability.json").resolve())},
                         indent=2, sort_keys=True))
    elif args.runtime_semantics or args.finalize_integrity:
        print(json.dumps({"passed": result["passed"],
                          "status": result.get("status"),
                          "record": str((EVIDENCE / "integrity" /
                                         ("runtime-semantics.json" if args.runtime_semantics
                                          else "index.json")).resolve())},
                         indent=2, sort_keys=True))
    elif args.run_condition:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.discovery:
        print(json.dumps({"all_runs_passed": result["all_runs_passed"],
                          "order": result["order"], "runs": result["runs"],
                          "record": str((EVIDENCE / "series-a/discovery-order.json").resolve())},
                         indent=2, sort_keys=True))
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
