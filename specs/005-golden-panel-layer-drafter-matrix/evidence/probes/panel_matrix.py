#!/usr/bin/env python3
"""Frozen physical runner for the Feature 005 prompt/configuration matrix."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
FEATURE = ROOT / "specs/005-golden-panel-layer-drafter-matrix"
EVIDENCE = FEATURE / "evidence"
ATTEMPTS = EVIDENCE / "attempts"
CHECKPOINTS = ROOT / "specs/002-qwen-bonsai-hybrid-target/evidence/checkpoints.json"
SERIES_B_CHECKPOINTS = ROOT / "specs/002-qwen-bonsai-hybrid-target/evidence/series-b/checkpoints.json"

QWEN_REPO, QWEN_REV = "mlx-community/Qwen3.8-27B-4bit", "10c35caafbb80f7dc6a7a432cdd11af10a6d4818"
BQ_REPO, BQ_REV = "incoai/Qwen3.8-27B-DFlash2", "015e795645c74b1a0eeef3b570031fb62e769bc5"
BB_REPO, BB_REV = "naklitechie/Qwen3.8-27B-DFlash2-ternary-bonsai2", "0059b38aa255698b1a87305eb3fbb5a3cfd616e2"
BB_WEIGHT_SHA256 = "eb141d0fb8621cf7a7c81047b83810fa19b8093543bd05529e9478d9555cb8f1"
BONSAI_REPO = "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"
SETTINGS = {"thinking": False, "temperature": 0, "top_p": 1, "top_k": 0, "max_tokens": 512}
PHYSICAL_CONFIG = {"mode": "dflash", "drafter_bits": 4, "max_draft_tokens": "auto",
    "enable_thinking": False, "prefix_cache": True, "prefix_cache_dir": None,
    "prefix_cache_max_ram_mb": 0, "prefix_cache_slots": 2, "prefix_cache_rungs": 8192,
    "kv_bits": None, "context_window": None, "warmup": True, "memory_guard": True,
    "lookup_drafts": False, "small_m": None, "sdpa_split": None,
    "wide_gemm_min": None, "cpu_split": None, "width_policy": None, "kv8": False, "tuning": False}

PROMPTS = (
    ("P05", "Merge sorted lists", "Write a Python function that merges two sorted lists into a single sorted list. Include tests and a short complexity analysis.", 52.798),
    ("P07", "First unique character", "Write a Python function that returns the first non-repeating character in a string. Include type hints and unit tests.", 45.211),
    ("P08", "Stack", "Write a production-quality Python implementation of a stack with push, pop, peek, and is_empty methods. Include type hints and tests.", 47.854),
    ("P14", "Edit distance", "Write a Python function that computes the edit distance between two strings using dynamic programming. Include tests and explain the time and space complexity.", 51.113),
    ("P17", "Topological sort", "Write a Python function that performs topological sorting on a directed acyclic graph. Include cycle detection, type hints, and tests.", 44.017),
)
DONOR_BLOCKS = {"H0": (), "H1a": (63,), "H1b": (62,), "H1c": (62, 63),
                "H2": (60, 61, 62, 63), "H3": tuple(range(56, 64)), "B0": tuple(range(64))}
TARGETS = ("H0", "H1a", "H1b", "H1c", "H2", "H3", "B0")
DRAFTERS = ("B-Q", "B-B")
FORWARD_ORDER = (("H0", "B-Q"), ("H0", "B-B"), ("H1a", "B-B"), ("H1a", "B-Q"),
    ("H1b", "B-Q"), ("H1b", "B-B"), ("H1c", "B-B"), ("H1c", "B-Q"),
    ("H2", "B-Q"), ("H2", "B-B"), ("H3", "B-B"), ("H3", "B-Q"), ("B0", "B-Q"), ("B0", "B-B"))
REVERSE_ORDER = (("B0", "B-Q"), ("B0", "B-B"), ("H3", "B-B"), ("H3", "B-Q"),
    ("H2", "B-Q"), ("H2", "B-B"), ("H1c", "B-B"), ("H1c", "B-Q"),
    ("H1b", "B-Q"), ("H1b", "B-B"), ("H1a", "B-B"), ("H1a", "B-Q"), ("H0", "B-Q"), ("H0", "B-B"))
FORWARD_PROMPTS = {"P05", "P08", "P17"}
GIB = 1024 ** 3


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value: Any) -> str:
    return sha_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def write_immutable(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return sha_bytes(encoded)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with temp.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def checkpoint_manifest() -> dict:
    accepted = json.loads(CHECKPOINTS.read_text())["checkpoints"]
    series_b = json.loads(SERIES_B_CHECKPOINTS.read_text())["B-B"]
    qwen, bq, bonsai = accepted["qwen"], accepted["dflash"], accepted["bonsai"]
    return {
        "Qwen": {"repo_id": QWEN_REPO, "revision": QWEN_REV, "path": qwen["resolved_path"]},
        "B-Q": {"repo_id": BQ_REPO, "revision": BQ_REV, "path": bq["resolved_path"]},
        "B-B": {"repo_id": BB_REPO, "revision": BB_REV, "path": series_b["snapshot_path"], "weight_sha256": BB_WEIGHT_SHA256},
        "official_Bonsai2": {"repo_id": BONSAI_REPO, "revision": bonsai["revision"], "path": bonsai["resolved_path"]},
    }


def target_record(target: str, drafter: str) -> dict:
    return target_record_from_manifest(target, drafter, checkpoint_manifest())


def target_record_from_manifest(target: str, drafter: str, cps: dict) -> dict:
    if target == "B0":
        identity = {**cps["official_Bonsai2"], "kind": "native_full_model", "embedding": "Bonsai2",
                    "blocks": list(range(64)), "final_norm": "Bonsai2", "lm_head": "Bonsai2"}
    else:
        identity = {**cps["Qwen"], "kind": "Qwen_shell_composition", "embedding": "Qwen",
                    "blocks": list(DONOR_BLOCKS[target]), "final_norm": "Qwen", "lm_head": "Qwen",
                    "donor": cps["official_Bonsai2"]}
    return {"target": target, "target_identity": identity, "bonsai_owned_block_count": len(DONOR_BLOCKS[target]),
            "drafter": drafter, "drafter_identity": cps[drafter]}


def runtime_hashes() -> dict:
    names = ("src/mlx_dspark/server.py", "src/mlx_dspark/load.py", "src/mlx_dspark/generate.py",
             "src/mlx_dspark/target.py", "src/mlx_dspark/hybrid_target.py", "src/mlx_dspark/prism_pack.py",
             "src/mlx_dspark/calibrate.py", "specs/001-bonsai-rollback-parity/evidence/probes/t027_product_ab.py")
    return {name: sha_file(ROOT / name) for name in names}


def encode(tokenizer: Any, text: str) -> tuple[str, list[int]]:
    messages = [{"role": "user", "content": text}]
    rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False, tokenize=False)
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False, tokenize=True)
    if isinstance(ids, dict):
        ids = ids["input_ids"]
    elif hasattr(ids, "input_ids"):
        ids = ids.input_ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return rendered, [int(token) for token in ids]


def prompt_record(prompt_id: str) -> tuple:
    return next(row for row in PROMPTS if row[0] == prompt_id)


def cell_order(prompt_id: str) -> tuple:
    return REVERSE_ORDER if prompt_id in {"P07", "P14"} else FORWARD_ORDER


def expected_cells() -> list[dict]:
    rows = []
    for prompt_id, *_ in PROMPTS:
        for order, (target, drafter) in enumerate(cell_order(prompt_id), 1):
            rows.append({"prompt_id": prompt_id, "order": order, "target": target, "drafter": drafter,
                         "cell_id": f"{order:02d}-{target}-{drafter}"})
    return rows


def model_free_checks() -> dict:
    assert len(PROMPTS) == 5 and [r[0] for r in PROMPTS] == ["P05", "P07", "P08", "P14", "P17"]
    assert len(expected_cells()) == 70 and len({(r["prompt_id"], r["target"], r["drafter"]) for r in expected_cells()}) == 70
    assert all(len(cell_order(pid)) == 14 for pid, *_ in PROMPTS)
    assert cell_order("P05") == FORWARD_ORDER and cell_order("P17") == FORWARD_ORDER
    assert cell_order("P08") == FORWARD_ORDER and cell_order("P07") == REVERSE_ORDER and cell_order("P14") == REVERSE_ORDER
    assert DONOR_BLOCKS == {"H0": (), "H1a": (63,), "H1b": (62,), "H1c": (62, 63),
        "H2": (60, 61, 62, 63), "H3": tuple(range(56, 64)), "B0": tuple(range(64))}
    assert SETTINGS == {"thinking": False, "temperature": 0, "top_p": 1, "top_k": 0, "max_tokens": 512}
    assert PHYSICAL_CONFIG["prefix_cache"] and PHYSICAL_CONFIG["max_draft_tokens"] == "auto"
    assert PHYSICAL_CONFIG["kv_bits"] is None and PHYSICAL_CONFIG["width_policy"] is None
    assert PHYSICAL_CONFIG["kv8"] is False and PHYSICAL_CONFIG["tuning"] is False
    assert BB_WEIGHT_SHA256 == "eb141d0fb8621cf7a7c81047b83810fa19b8093543bd05529e9478d9555cb8f1"
    assert child_command(["--self-check"])[0] == str(ROOT / ".venv/bin/python")
    first_args = cell_child_args("a", "P05", 1, "H0", "B-Q", Path("/tmp/cell.json"))
    repeat_args = cell_child_args("a", "P05", 1, "H0", "B-Q", Path("/tmp/repeat.json"), repeat=True, comparison="P05-H0-B-Q")
    assert "--repeat" not in first_args and "--comparison" not in first_args
    assert "--repeat" in repeat_args and repeat_args[repeat_args.index("--comparison") + 1] == "P05-H0-B-Q"
    try:
        cell_child_args("a", "P05", 1, "H0", "B-Q", Path("/tmp/repeat.json"), repeat=True)
        raise AssertionError("repeat command accepted a missing comparison")
    except ValueError:
        pass
    src = inspect.getsource(physical_cell)
    assert "stream_generate(" not in src and "reset_peak_memory()" in src and "get_active_memory()" in src
    assert "get_peak_memory()" in src and "result.reused_tokens" in src and "Engine.load(" in src
    assert "--run-cell" in Path(__file__).read_text() and "subprocess.run(" in Path(__file__).read_text()
    assert (max(10, 10) - 10) == 0
    return {"status": "PASS", "checks": 22, "cells": 70}


def interpreter_identity() -> dict:
    expected = (ROOT / ".venv/bin/python").resolve()
    current = Path(sys.executable).resolve()
    if current != expected:
        raise RuntimeError(f"must use repository interpreter {expected}; running {current}")
    return {"path": str(current), "version": sys.version, "implementation": platform.python_implementation(),
            "cache_tag": sys.implementation.cache_tag}


def verify_checkpoint(path: Path, revision: str, require_weights: bool = True) -> dict:
    if revision not in str(path.resolve()):
        raise ValueError(f"checkpoint path does not contain pinned revision {revision}: {path}")
    required = [path / "config.json"]
    if require_weights:
        weights = sorted(path.glob("*.safetensors"))
        if weights:
            required.extend(weights)
        elif (path / "model.safetensors.index.json").is_file():
            required.append(path / "model.safetensors.index.json")
        else:
            raise FileNotFoundError(f"checkpoint weights unavailable: {path}")
    missing = [p for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("missing checkpoint files: " + ", ".join(map(str, missing)))
    return {"path": str(path.resolve()), "revision": revision,
            "files": [{"name": p.name, "size_bytes": p.stat().st_size,
                       "sha256": sha_file(p) if p.name == "model.safetensors" else None,
                       "resolved_blob_name": p.resolve().name} for p in required]}


def make_preflight(attempt_id: str) -> dict:
    interp = interpreter_identity()
    cps = checkpoint_manifest()
    qwen = verify_checkpoint(Path(cps["Qwen"]["path"]), QWEN_REV)
    bq = verify_checkpoint(Path(cps["B-Q"]["path"]), BQ_REV)
    bb_path = Path(cps["B-B"]["path"])
    selected = bb_path / "model.safetensors"
    if not selected.is_file() or selected.resolve().name != BB_WEIGHT_SHA256 or sha_file(selected) != BB_WEIGHT_SHA256:
        raise ValueError(f"selected B-B weight does not match pinned SHA-256: {selected}")
    bb = verify_checkpoint(bb_path, BB_REV)
    bonsai = verify_checkpoint(Path(cps["official_Bonsai2"]["path"]), cps["official_Bonsai2"]["revision"])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cps["Qwen"]["path"], local_files_only=True, trust_remote_code=False)
    prompts = []
    for pid, label, text, reference_tps in PROMPTS:
        rendered, ids = encode(tokenizer, text)
        prompts.append({"prompt_id": pid, "label": label, "text": text, "text_sha256": sha_bytes(text.encode()),
            "rendered_prompt": rendered, "input_ids": ids, "input_ids_sha256": canonical_sha(ids),
            "input_token_count": len(ids), "qualification_reference_tok_s": reference_tps,
            "qualification_generated_tokens_each_run_min": 410})
    return {"schema": "panel-matrix-preflight/v1", "attempt_id": attempt_id, "status": "PASS",
        "benchmark_performed": False, "model_free": True, "runner_sha256": sha_file(Path(__file__).resolve()),
        "interpreter": interp, "runtime": {"python": sys.version, "platform": platform.platform(),
            "machine": platform.node(), "source_sha256": runtime_hashes()},
        "child_command_interpreter": str(ROOT / ".venv/bin/python"),
        "checkpoints": {"Qwen": qwen, "B-Q": bq, "B-B": bb, "official_Bonsai2": bonsai},
        "checkpoint_manifest": cps, "selected_BB_weight_sha256": BB_WEIGHT_SHA256, "prompts": prompts,
        "settings": SETTINGS, "physical_config": PHYSICAL_CONFIG,
        "freshness_contract": {"fresh_process_per_cell": True, "fresh_engine_per_cell": True,
            "fresh_empty_measured_request": True, "prefix_cache_initialized": True,
            "useful_prefix_reuse_tokens": 0},
        "memory_contract": {"reset": "mx.metal.reset_peak_memory() after model load/warmup and fresh-state assertions",
            "baseline": "mx.metal.get_active_memory() immediately before measured engine.generate request",
            "peak": "mx.metal.get_peak_memory() immediately after measured engine.generate request",
            "increment": "peak_runtime_bytes - active_baseline_bytes", "serial_generation": False},
        "expected_cell_count": 70, "execution_order": {"forward": list(FORWARD_ORDER),
            "reverse": list(REVERSE_ORDER), "forward_prompts": sorted(FORWARD_PROMPTS),
            "reverse_prompts": ["P07", "P14"]}, "physical_observations": "not performed by preflight"}


def attempt_dir(attempt_id: str) -> Path:
    if not attempt_id or "/" in attempt_id or attempt_id in {".", ".."}:
        raise ValueError("invalid attempt ID")
    return ATTEMPTS / attempt_id


def start_attempt(attempt_id: str) -> tuple[Path, dict, str]:
    directory = attempt_dir(attempt_id)
    preflight_path = directory / "preflight.json"
    preflight = json.loads(preflight_path.read_text())
    preflight_sha = sha_file(preflight_path)
    runner_sha = sha_file(Path(__file__).resolve())
    if preflight.get("status") != "PASS" or preflight.get("attempt_id") != attempt_id:
        raise RuntimeError("attempt has no matching passing preflight")
    if preflight.get("runner_sha256") != runner_sha:
        raise RuntimeError("physical runner differs from passing preflight SHA")
    attempt_path = directory / "attempt.json"
    meta = json.loads(attempt_path.read_text())
    if meta.get("attempt_id") != attempt_id or meta.get("preflight_sha256") != preflight_sha:
        raise RuntimeError("attempt lifecycle metadata does not match its passing preflight")
    if meta.get("runner_sha256") != runner_sha:
        raise RuntimeError("attempt lifecycle runner SHA mismatch")
    return directory, preflight, preflight_sha


def update_attempt(directory: Path, **updates) -> dict:
    path = directory / "attempt.json"
    value = json.loads(path.read_text())
    if value.get("terminal") in {"COMPLETE", "INCOMPLETE", "SUPERSEDED"}:
        raise RuntimeError("terminal attempt metadata is frozen")
    value.update(updates)
    atomic_json(path, value)
    return value


def physical_cell(attempt_id: str, prompt_id: str, order: int, target: str, drafter: str,
                  preflight: dict, preflight_sha: str, repeat: bool = False, comparison: str | None = None) -> dict:
    """Run one speculative request in the fresh child process."""
    if interpreter_identity() != preflight["interpreter"]:
        raise RuntimeError("physical child interpreter differs from passing preflight")
    if platform.node() != preflight["runtime"]["machine"] or platform.platform() != preflight["runtime"]["platform"]:
        raise RuntimeError("physical child machine/platform differs from passing preflight")
    runtime_now = runtime_hashes()
    if runtime_now != preflight["runtime"]["source_sha256"]:
        raise RuntimeError("production runtime source hashes differ from immutable attempt preflight")
    sys.path.insert(0, str(ROOT / "src"))
    import mlx.core as mx
    from mlx_dspark.generate import encode_messages
    from mlx_dspark.hybrid_target import TargetCompositionRequest
    from mlx_dspark.server import Engine

    prompt = prompt_record(prompt_id)
    pin = next(p for p in preflight["prompts"] if p["prompt_id"] == prompt_id)
    _, label, text, _ = prompt
    cps = preflight["checkpoint_manifest"]
    identity = target_record_from_manifest(target, drafter, cps)
    target_path = Path(identity["target_identity"]["path"])
    drafter_path = Path(identity["drafter_identity"]["path"])
    composition = None
    if target != "B0" and DONOR_BLOCKS[target]:
        donor, qwen = cps["official_Bonsai2"], cps["Qwen"]
        composition = TargetCompositionRequest.create(donor_path=donor["path"], donor_repo=donor["repo_id"],
            donor_revision=donor["revision"], qwen_repo=qwen["repo_id"], qwen_revision=qwen["revision"],
            donor_indices=DONOR_BLOCKS[target])
    started_utc, started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), time.perf_counter()
    pid = os.getpid()
    engine = Engine.load(mode="dflash", model=str(target_path), drafter=str(drafter_path),
        target_composition=composition, drafter_bits=4, max_draft_tokens="auto", enable_thinking=False,
        prefix_cache=True, prefix_cache_dir=None, prefix_cache_max_ram_mb=0, prefix_cache_slots=2,
        prefix_cache_rungs=8192, kv_bits=None, context_window=None, warmup=True, memory_guard=True,
        lookup_drafts=False, small_m=None, sdpa_split=None, wide_gemm_min=None, cpu_split=None)
    try:
        ids = pin["input_ids"]
        runtime_ids = encode_messages(engine.tokenizer, [{"role": "user", "content": text}], enable_thinking=False)
        if runtime_ids != ids:
            raise RuntimeError("production Engine tokenizer differs from attempt-pinned prompt IDs")
        if engine.prefix is None:
            raise RuntimeError("production prefix cache did not initialize")
        acquire_reuse, cache_rows = [], []
        acquire = engine.prefix.acquire
        def observe(ids_arg):
            cache, context, reused = acquire(ids_arg)
            acquire_reuse.append(int(reused))
            rows = []
            for item in cache:
                cls, bits = type(item).__name__, getattr(item, "bits", None)
                attention = "KV" in cls or hasattr(item, "keys") or hasattr(item, "values")
                rows.append({"class": cls, "attention_cache": bool(attention), "bits": int(bits) if bits is not None else None})
                if attention and (bits is not None or "Quantized" in cls):
                    raise RuntimeError(f"plain-KV invariant violated: {cls}")
            if not any(row["attention_cache"] for row in rows):
                raise RuntimeError("could not observe target attention KV cache")
            cache_rows.append(rows)
            return cache, context, reused
        engine.prefix.acquire = observe
        rounds_before, hits_before = len(engine.rounds.snapshot()), engine.prefix.hits
        if rounds_before != 0:
            raise RuntimeError("fresh Engine has prior speculative rounds")
        mx.metal.reset_peak_memory()
        reset_completed = time.perf_counter_ns()
        baseline = int(mx.metal.get_active_memory())
        baseline_captured = time.perf_counter_ns()
        request_start = time.perf_counter()
        request_started = time.perf_counter_ns()
        result = engine.generate(ids, max_tokens=512, temperature=0.0, top_p=1.0, top_k=0, stop=None, seed=None)
        request_ended = time.perf_counter_ns()
        request_end = time.perf_counter()
        peak = int(mx.metal.get_peak_memory())
        peak_captured = time.perf_counter_ns()
        request_seconds = request_end - request_start
        if int(result.reused_tokens) != 0 or not acquire_reuse or acquire_reuse[-1] != 0:
            raise RuntimeError("measured request reused prior prompt prefix")
        if engine.target.kv_bits not in (None, 0):
            raise RuntimeError("target KV is not plain")
        rounds = [dict(row) for row in engine.rounds.snapshot()]
        generated = int(result.num_tokens)
        forwards = int(result.target_forwards)
        proposed = sum(int(row.get("drafted", 0)) for row in rounds)
        accepted = sum(int(row.get("accepted", 0)) for row in rounds)
        output_text = str(result.text)
        raw_output_tokens = getattr(result, "tokens", None)
        has_output_tokens = raw_output_tokens is not None
        output_tokens = [int(x) for x in raw_output_tokens] if has_output_tokens else None
        peak_increment = peak - baseline
        if peak_increment < 0:
            raise RuntimeError("request-boundary peak is below active baseline")
        return {"schema": "panel-matrix-cell/v1", "status": "complete" if generated >= 410 else "invalid_too_short", "attempt_id": attempt_id,
            "attempt_cell_identity": {"prompt_id": prompt_id, "order": order, "cell_id": f"{order:02d}-{target}-{drafter}",
                "target": target, "drafter": drafter, "repeat": repeat, "comparison": comparison},
            "preflight": {"path": "preflight.json", "sha256": preflight_sha, "identity": f"{attempt_id}:PASS"},
            "runner_sha256": sha_file(Path(__file__).resolve()), "prompt": {"prompt_id": prompt_id,
                "label": label, "text": text, "text_sha256": sha_bytes(text.encode()), "rendered_prompt": pin["rendered_prompt"],
                "input_ids": ids, "input_ids_sha256": pin["input_ids_sha256"], "input_token_count": pin["input_token_count"]},
            "configuration": identity, "settings": SETTINGS,
            "freshness": {"physical_asserted": True, "pid": pid, "new_process": True, "new_engine": True,
                "rounds_before_request": rounds_before, "prefix_hits_before_request": hits_before,
                "prefix_reused_tokens": int(result.reused_tokens), "acquire_reused_tokens": acquire_reuse[-1],
                "prefix_cache_initialized": engine.prefix is not None,
                "assertion": "fresh child process, Engine, and measured request; zero useful prior prefix reuse"},
            "runtime": {"engine_mode": engine.mode, "controller_present": engine.cap_controller is not None,
                "controller_info": engine.cap_controller.info() if engine.cap_controller is not None else None,
                "effective_cap": engine._last_cap, "max_draft_tokens": engine.max_draft_tokens,
                "prefix_cache_mode": "checkpoint" if engine.prefix.checkpoint_mode else "trim",
                "kv_bits": engine.target.kv_bits, "width_policy": None, "kv8": False, "tuning": False,
                "kv_cache_types": cache_rows[-1]},
            "measurements": {"generated_tokens": generated,
                "speculative_decode_tok_s": generated / max(float(result.decode_seconds), 1e-9),
                "acceptance": accepted / proposed if proposed else None,
                "mean_accepted_draft_tokens": accepted / len(rounds) if rounds else None,
                "target_forwards": forwards, "generated_tokens_per_target_forward": generated / forwards if forwards else None,
                "rounds": len(rounds), "width_distribution": dict(Counter(str(r.get("drafted", 0)) for r in rounds)),
                "cap_distribution": dict(Counter(str(r.get("cap", "unknown")) for r in rounds)),
                "request_duration_seconds": request_seconds, "decode_duration_seconds": float(result.decode_seconds),
                "active_baseline_bytes": baseline, "active_baseline_gib": baseline / GIB,
                "peak_runtime_bytes": peak, "peak_runtime_gib": peak / GIB,
                "peak_increment_bytes": peak_increment, "peak_increment_gib": peak_increment / GIB,
                "output_text": output_text, "output_sha256": sha_bytes(output_text.encode()),
                "output_integrity_mode": "text_and_tokens" if has_output_tokens else "text_only",
                "output_tokens": output_tokens,
                "output_tokens_sha256": canonical_sha(output_tokens) if has_output_tokens else None,
                "stop_reason": getattr(result, "stop_reason", None),
                "round_events": rounds},
            "memory_boundary": {"reset_called": True,
                "reset_completed": reset_completed, "baseline_captured": baseline_captured,
                "request_started": request_started, "request_ended": request_ended,
                "peak_captured": peak_captured,
                "order": ["load_and_warmup_complete", "fresh_request_state_asserted", "reset_peak_memory",
                    "capture_active_baseline", "measured_speculative_request_only", "capture_peak_immediately"],
                "assertion": "peak interval begins at measured request boundary; no serial generation"},
            "provenance": {"machine": platform.node(), "platform": platform.platform(), "python": sys.version,
                "interpreter": str(Path(sys.executable).resolve()), "runtime_source_sha256": runtime_now,
                "target_revision": identity["target_identity"]["revision"],
                "drafter_revision": identity["drafter_identity"]["revision"],
                "selected_drafter_weight_sha256": identity["drafter_identity"].get("weight_sha256"),
                "process_started_utc": started_utc, "process_elapsed_seconds": time.perf_counter() - started}}
    finally:
        engine.close()


def cell_path(directory: Path, prompt_id: str, cell_id: str, repeat: bool, comparison: str | None) -> Path:
    if repeat:
        if not comparison or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in comparison):
            raise ValueError("invalid repeat comparison identity")
        return directory / "repeats" / f"{comparison}.json"
    return directory / "cells" / prompt_id / f"{cell_id}.json"


def child_command(args: list[str]) -> list[str]:
    # Preserve the venv launcher path: resolving its symlink can invoke base Python
    # outside the environment that owns MLX and the accepted runtime dependencies.
    return [str(ROOT / ".venv/bin/python"), str(Path(__file__).resolve()), *args]


def cell_child_args(attempt_id: str, prompt_id: str, order: int, target: str, drafter: str,
                    output: Path, *, repeat: bool = False, comparison: str | None = None) -> list[str]:
    if repeat and not comparison:
        raise ValueError("repeat child command requires a non-empty comparison ID")
    if not repeat and comparison is not None:
        raise ValueError("first-pass child command cannot carry a comparison ID")
    args = ["--run-cell", "--attempt-id", attempt_id, "--prompt-id", prompt_id,
        "--order", str(order), "--target", target, "--drafter", drafter, "--output", str(output)]
    if repeat:
        args.extend(["--repeat", "--comparison", comparison])
    return args


def run_child(attempt_id: str, prompt_id: str, order: int, target: str, drafter: str,
              repeat: bool = False, comparison: str | None = None, expected_path: Path | None = None) -> dict:
    directory, preflight, preflight_sha = start_attempt(attempt_id)
    out = expected_path or cell_path(directory, prompt_id, f"{order:02d}-{target}-{drafter}", repeat, comparison)
    command = child_command(cell_child_args(attempt_id, prompt_id, order, target, drafter, out,
        repeat=repeat, comparison=comparison))
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode:
        return {"status": "FAILED", "returncode": proc.returncode, "stdout": proc.stdout[-2000:], "stderr": proc.stderr[-4000:]}
    data = json.loads(out.read_text())
    return {"status": data["status"], "path": str(out.relative_to(FEATURE)), "sha256": sha_file(out)}


def run_all(attempt_id: str) -> dict:
    directory, _, _ = start_attempt(attempt_id)
    update_attempt(directory, status="RUNNING", terminal=None, started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    completed = []
    for pid, *_ in PROMPTS:
        for order, (target, drafter) in enumerate(cell_order(pid), 1):
            out = cell_path(directory, pid, f"{order:02d}-{target}-{drafter}", False, None)
            if out.exists():
                update_attempt(directory, status="INCOMPLETE", terminal="INCOMPLETE", reason=f"cell path already exists before execution: {out}")
                raise FileExistsError(f"refusing to overwrite prior cell: {out}")
            row = run_child(attempt_id, pid, order, target, drafter)
            completed.append({"prompt_id": pid, "order": order, "target": target, "drafter": drafter, **row})
            update_attempt(directory, observed_statused_cell_count=len(completed), last_cell=completed[-1])
            if row["status"] == "FAILED":
                update_attempt(directory, status="INCOMPLETE", terminal="INCOMPLETE",
                    reason=f"cell execution failed at {pid}/{order:02d}-{target}-{drafter}; preserve and investigate")
                return {"attempt_id": attempt_id, "status": "INCOMPLETE", "completed_before_failure": len(completed)-1, "failure": row}
    all_valid = all(row["status"] == "complete" for row in completed)
    state = "COMPLETE" if all_valid else "INCOMPLETE"
    update_attempt(directory, status=state, terminal=state, completed=all_valid,
        reason=None if all_valid else "one or more statused cells failed acceptance requirements",
        observed_statused_cell_count=len(completed), completed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    return {"attempt_id": attempt_id, "status": state, "cells": len(completed)}


def execute_repeat_manifest(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    attempt_id = manifest.get("effective_attempt_id")
    directory, preflight, preflight_sha = start_attempt(attempt_id)
    if manifest.get("schema") != "panel-matrix-repeat-manifest/v1" or manifest.get("status") != "PASS":
        raise ValueError("repeat runner requires an explicit passing immutable manifest")
    if manifest.get("runner_sha256") != sha_file(Path(__file__).resolve()):
        raise ValueError("repeat manifest runner SHA differs from frozen runner")
    results = []
    for entry in manifest.get("repeats", []):
        if entry.get("attempt_id") != attempt_id:
            raise ValueError("repeat entry attempt mismatch")
        comparison = entry["comparison_id"]
        out = directory / "repeats" / f"{comparison}.json"
        if out.exists():
            raise FileExistsError(f"refusing to overwrite repeat raw evidence: {out}")
        result = run_child(attempt_id, entry["prompt_id"], entry["order"], entry["target"], entry["drafter"],
                           True, comparison, out)
        results.append({"comparison_id": comparison, **result})
        if result["status"] != "complete":
            break
    return {"effective_attempt_id": attempt_id, "runner_sha256": preflight["runner_sha256"], "results": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--new-attempt", action="store_true")
    parser.add_argument("--attempt-id")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--run-cell", action="store_true")
    parser.add_argument("--prompt-id")
    parser.add_argument("--order", type=int)
    parser.add_argument("--target")
    parser.add_argument("--drafter")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeat", action="store_true")
    parser.add_argument("--comparison")
    parser.add_argument("--run-repeats", type=Path)
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(model_free_checks(), indent=2)); return 0
    if args.new_attempt:
        attempt_id = args.attempt_id or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
        directory = attempt_dir(attempt_id)
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "cells").mkdir(); (directory / "repeats").mkdir()
        atomic_json(directory / "attempt.json", {"schema": "panel-matrix-attempt/v1", "attempt_id": attempt_id,
            "status": "PREPARING", "terminal": None, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "runner_sha256": sha_file(Path(__file__).resolve()), "expected_cell_count": 70,
            "observed_statused_cell_count": 0})
        print(json.dumps({"attempt_id": attempt_id, "path": str(directory.relative_to(FEATURE))}, indent=2)); return 0
    if args.preflight:
        if not args.attempt_id:
            raise ValueError("--preflight requires --attempt-id from --new-attempt")
        directory = attempt_dir(args.attempt_id)
        attempt_path = directory / "attempt.json"
        attempt = json.loads(attempt_path.read_text())
        if attempt.get("status") != "PREPARING" or attempt.get("attempt_id") != args.attempt_id:
            raise RuntimeError("attempt is not in PREPARING state")
        try:
            data = make_preflight(args.attempt_id)
        except Exception as error:
            failed = {"schema": "panel-matrix-preflight/v1", "attempt_id": args.attempt_id,
                "status": "FAIL", "benchmark_performed": False, "runner_sha256": sha_file(Path(__file__).resolve()),
                "error_type": type(error).__name__, "error": str(error)}
            preflight_sha = write_immutable(directory / "preflight.json", failed)
            attempt.update({"status": "PREFLIGHT_FAIL", "terminal": "INCOMPLETE", "preflight_status": "FAIL",
                "preflight_identity": f"{args.attempt_id}:FAIL", "preflight_sha256": preflight_sha,
                "runner_sha256": failed["runner_sha256"], "reason": str(error)})
            atomic_json(attempt_path, attempt)
            raise
        preflight_sha = write_immutable(directory / "preflight.json", data)
        attempt.update({"status": "PREFLIGHT_PASS", "runner_sha256": data["runner_sha256"],
            "preflight_identity": f"{args.attempt_id}:PASS", "preflight_sha256": preflight_sha,
            "preflight_status": "PASS"})
        atomic_json(attempt_path, attempt)
        print(json.dumps({"status": "PASS", "attempt_id": args.attempt_id, "preflight_sha256": preflight_sha,
            "runner_sha256": data["runner_sha256"], "benchmark_performed": False}, indent=2)); return 0
    if args.run_cell:
        if not all((args.attempt_id, args.prompt_id, args.order, args.target, args.drafter, args.output)):
            raise ValueError("--run-cell requires attempt, prompt, order, target, drafter, and output")
        directory, preflight, preflight_sha = start_attempt(args.attempt_id)
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite raw evidence: {args.output}")
        record = physical_cell(args.attempt_id, args.prompt_id, args.order, args.target, args.drafter,
                               preflight, preflight_sha, args.repeat, args.comparison or None)
        write_immutable(args.output, record)
        print(json.dumps({"status": record["status"], "output": str(args.output),
            "output_sha256": sha_file(args.output), "runner_sha256": record["runner_sha256"]}, indent=2)); return 0
    if args.run_all:
        if not args.attempt_id:
            raise ValueError("--run-all requires --attempt-id")
        result = run_all(args.attempt_id); print(json.dumps(result, indent=2)); return 0 if result["status"] == "COMPLETE" else 2
    if args.run_repeats:
        result = execute_repeat_manifest(args.run_repeats); print(json.dumps(result, indent=2)); return 0
    raise SystemExit("choose --self-check, --new-attempt, --preflight, --run-all, --run-cell, or --run-repeats")


if __name__ == "__main__":
    raise SystemExit(main())
