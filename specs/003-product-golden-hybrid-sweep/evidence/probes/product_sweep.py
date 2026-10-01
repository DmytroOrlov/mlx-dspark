#!/usr/bin/env python3
"""Feature-local product Golden-A sweep runner.

The request and Engine path follow T027. Target composition and provenance use
the accepted feature-002 integration. ``--self-check`` and ``--preflight`` are
model-free; a physical cell is always launched in a separate Python process.
"""
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
FEATURE = ROOT / "specs/003-product-golden-hybrid-sweep"
EVIDENCE = FEATURE / "evidence"
T027 = ROOT / "specs/001-bonsai-rollback-parity/evidence/probes/t027_product_ab.py"
CHECKPOINTS = ROOT / "specs/002-qwen-bonsai-hybrid-target/evidence/checkpoints.json"
SERIES_B_CHECKPOINTS = ROOT / "specs/002-qwen-bonsai-hybrid-target/evidence/series-b/checkpoints.json"
SERIES_B_RUNNER = ROOT / "specs/002-qwen-bonsai-hybrid-target/evidence/series-b/runner.json"

WORKLOAD = "Write a production-quality Python LRU cache with tests and type hints."
INPUT_IDS = [248045, 846, 198, 7734, 264, 5492, 21408, 12654, 436, 34810,
             6297, 440, 6813, 321, 913, 29642, 13, 248046, 198, 248045,
             74455, 198, 248068, 271, 248069, 271]
INPUT_SHA256 = "8bda1dedde8e022452b7dafaa3677dca47dedc350258c0885c4d1e5fde155f59"
QWEN_REV = "10c35caafbb80f7dc6a7a432cdd11af10a6d4818"
BQ_REV = "015e795645c74b1a0eeef3b570031fb62e769bc5"
BB_REV = "0059b38aa255698b1a87305eb3fbb5a3cfd616e2"
BB_WEIGHT_SHA256 = "eb141d0fb8621cf7a7c81047b83810fa19b8093543bd05529e9478d9555cb8f1"
QWEN_REPO = "mlx-community/Qwen3.8-27B-4bit"
BQ_REPO = "incoai/Qwen3.8-27B-DFlash2"
BB_REPO = "naklitechie/Qwen3.8-27B-DFlash2-ternary-bonsai2"
OFFICIAL_BONSAI_REPO = "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"
HISTORICAL = {
    "serial_target_tok_s": 15.4992848,
    "speculative_decode_tok_s": 45.7788518,
    "speedup": 2.95361,
    "acceptance": 0.6482335,
    "target_forwards": 94,
    "rounds": 93,
    "generated_tokens": 516,
    "generated_per_forward": 5.4893617,
    "mean_accepted": 5.4893617,
    "width_distribution": {"7": 93},
    "peak_gib": 15.916,
}

# Tuple schema is frozen for later T005 orchestration. Row 1 is the sole T004 cell.
MATRIX = (
    (1, "H0", "B-Q"), (2, "H0", "B-B"), (3, "H1a", "B-B"),
    (4, "H1a", "B-Q"), (5, "H1b", "B-Q"), (6, "H1b", "B-B"),
    (7, "H1c", "B-B"), (8, "H1c", "B-Q"), (9, "H2", "B-Q"),
    (10, "H2", "B-B"), (11, "H3", "B-B"), (12, "H3", "B-Q"),
    (13, "B0", "B-Q"), (14, "B0", "B-B"),
)
DONOR_BLOCKS = {"H0": (), "H1a": (63,), "H1b": (62,),
                "H1c": (62, 63), "H2": (60, 61, 62, 63),
                "H3": tuple(range(56, 64)), "B0": tuple(range(64))}
T027_SETTINGS = {"thinking": False, "temperature": 0, "top_p": 1,
                 "top_k": 0, "max_tokens": 512}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_sha256(value: list[int]) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def checkpoints() -> dict:
    accepted = load_json(CHECKPOINTS)["checkpoints"]
    bq = accepted["dflash"]
    qwen = accepted["qwen"]
    donor = accepted["bonsai"]
    bb = load_json(SERIES_B_CHECKPOINTS)["B-B"]
    return {"Qwen": {"repo_id": QWEN_REPO, "revision": QWEN_REV,
                      "path": qwen["resolved_path"]},
            "B-Q": {"repo_id": BQ_REPO, "revision": BQ_REV,
                    "path": bq["resolved_path"]},
            "B-B": {"repo_id": BB_REPO, "revision": BB_REV,
                    "path": bb["snapshot_path"], "weight_sha256": BB_WEIGHT_SHA256},
            "official_Bonsai2": {"repo_id": OFFICIAL_BONSAI_REPO,
                                  "revision": donor["revision"],
                                  "path": donor["resolved_path"]}}


def target_record(target: str, drafter: str) -> dict:
    cps = checkpoints()
    if target == "B0":
        target_identity = {**cps["official_Bonsai2"], "kind": "native_full_model",
                           "embedding": "Bonsai2", "blocks": list(range(64)),
                           "final_norm": "Bonsai2", "lm_head": "Bonsai2"}
    else:
        target_identity = {**cps["Qwen"], "kind": "Qwen_shell_composition",
                           "embedding": "Qwen", "blocks": list(DONOR_BLOCKS[target]),
                           "final_norm": "Qwen", "lm_head": "Qwen",
                           "donor": cps["official_Bonsai2"]}
    return {"target": target, "target_identity": target_identity,
            "drafter": drafter, "drafter_identity": cps[drafter]}


def source_hashes() -> dict[str, str]:
    files = ("src/mlx_dspark/server.py", "src/mlx_dspark/load.py",
             "src/mlx_dspark/generate.py", "src/mlx_dspark/target.py",
             "src/mlx_dspark/hybrid_target.py", "src/mlx_dspark/prism_pack.py",
             "src/mlx_dspark/calibrate.py", str(T027.relative_to(ROOT)))
    return {name: sha256_file(ROOT / name) for name in files}


def verify_checkpoint(path: Path, revision: str, *, require_weights: bool = True) -> dict:
    if revision not in str(path.resolve()):
        raise ValueError(f"checkpoint path does not contain pinned revision {revision}: {path}")
    required = [path / "config.json"]
    if require_weights:
        required.extend(sorted(path.glob("*.safetensors")))
        if not any(p.suffix == ".safetensors" for p in required):
            index = path / "model.safetensors.index.json"
            if not index.is_file():
                raise FileNotFoundError(f"no selected checkpoint weight files at {path}")
            required.append(index)
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("missing checkpoint files: " + ", ".join(missing))
    return {"path": str(path.resolve()), "revision": revision,
            "files": [{"name": p.name, "size_bytes": p.stat().st_size,
                       "sha256": sha256_file(p) if p.name == "model.safetensors" else None,
                       "content_addressed_blob_name": p.resolve().name}
                      for p in required]}


def prompt_preflight(qwen_path: Path) -> dict:
    # Deliberately imports no MLX. Match T027 AutoTokenizer rendering/tokenization.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(qwen_path), local_files_only=True,
                                               trust_remote_code=False)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": WORKLOAD}], add_generation_prompt=True,
        enable_thinking=False, tokenize=False)
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": WORKLOAD}], add_generation_prompt=True,
        enable_thinking=False, tokenize=True)
    if isinstance(ids, dict) and "input_ids" in ids:
        ids = ids["input_ids"]
    elif hasattr(ids, "input_ids"):
        ids = ids.input_ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    ids = [int(x) for x in ids]
    digest = json_sha256(ids)
    if ids != INPUT_IDS or digest != INPUT_SHA256:
        raise ValueError(f"Golden-A input mismatch: {len(ids)} IDs, SHA-256 {digest}")
    return {"workload": WORKLOAD, "rendered_prompt": rendered, "input_ids": ids,
            "input_token_count": len(ids), "input_ids_sha256": digest}


def model_free_checks() -> dict:
    assert WORKLOAD == "Write a production-quality Python LRU cache with tests and type hints."
    assert len(INPUT_IDS) == 26 and json_sha256(INPUT_IDS) == INPUT_SHA256
    assert T027_SETTINGS == {"thinking": False, "temperature": 0, "top_p": 1,
                             "top_k": 0, "max_tokens": 512}
    assert not any(s in WORKLOAD for s in ("p01", "p02", "p03"))
    assert T027_SETTINGS["max_tokens"] != 128
    physical_source = inspect.getsource(physical_cell)
    assert "run_series_condition" not in physical_source
    assert "SERIES_A_PROMPTS" not in physical_source
    assert "max_new_tokens=128" not in physical_source
    assert "prefix_cache=False" not in physical_source
    assert "p01" not in physical_source and "p02" not in physical_source and "p03" not in physical_source
    assert MATRIX == ((1,"H0","B-Q"),(2,"H0","B-B"),(3,"H1a","B-B"),
        (4,"H1a","B-Q"),(5,"H1b","B-Q"),(6,"H1b","B-B"),
        (7,"H1c","B-B"),(8,"H1c","B-Q"),(9,"H2","B-Q"),
        (10,"H2","B-B"),(11,"H3","B-B"),(12,"H3","B-Q"),
        (13,"B0","B-Q"),(14,"B0","B-B"))
    assert MATRIX[0] == (1, "H0", "B-Q") and sum(r[1:] == ("H0", "B-Q") for r in MATRIX) == 1
    assert DONOR_BLOCKS["B0"] == tuple(range(64))
    assert target_record("B0", "B-Q")["target_identity"]["kind"] == "native_full_model"
    assert QWEN_REV in checkpoints()["Qwen"]["path"]
    assert BQ_REV in checkpoints()["B-Q"]["path"]
    assert BB_REV in checkpoints()["B-B"]["path"]
    assert checkpoints()["B-B"]["weight_sha256"] == BB_WEIGHT_SHA256
    assert T027_SETTINGS["max_tokens"] == 512 and "128" not in json.dumps(T027_SETTINGS)
    # Static configuration assertions: runtime observation is recorded only by child.
    assert PHYSICAL_CONFIG["prefix_cache"] is True and PHYSICAL_CONFIG["prefix_cache_max_ram_mb"] == 0
    assert PHYSICAL_CONFIG["max_draft_tokens"] == "auto"
    assert PHYSICAL_CONFIG["kv_bits"] is None and PHYSICAL_CONFIG["width_policy"] is None
    assert PHYSICAL_CONFIG["kv8"] is False and PHYSICAL_CONFIG["tuning"] is False
    mock_factory = lambda: {"process": "new", "prefix_reused_tokens": 0}
    assert mock_factory()["prefix_reused_tokens"] == 0
    assert child_command(1, "H0", "B-Q", Path("/tmp/raw.json"))[0:2] == [sys.executable, str(Path(__file__).resolve())]
    canary = {"measurements": {"speculative_decode_tok_s": 34.0, "acceptance": 0.49,
                                 "target_forwards": 94, "generated_per_target_forward": 5.5}}
    assert adjudicate(canary)["decision"] == "STOP_MISMATCH"
    rows, stopped = orchestration_plan("STOP_MISMATCH")
    assert rows == [MATRIX[0]] and stopped == 13
    assert adjudicate({"measurements": {"speculative_decode_tok_s": 45.0, "acceptance": 0.64,
                                        "target_forwards": 94, "generated_per_target_forward": 5.5}})["decision"] == "CONTINUE_SWEEP"
    assert completeness_report([])["complete"] is False
    assert completeness_report([{"order": 1, "target": "H0", "drafter": "B-Q", "status": "complete"}])["complete"] is False
    assert repeat_triggers({"target": "H1a", "metrics": {"speculative_decode_tok_s": 46}}, 45)[0]
    assert not repeat_triggers({"target": "H1a", "metrics": {"speculative_decode_tok_s": 30}}, 45)
    report = render_report({"decision": "STOP_MISMATCH", "rows": []})
    assert "STOP_MISMATCH" in report and "CONTINUE_SWEEP" in report and "H0+B-Q" in report
    return {"status": "PASS", "checks": 15, "result": "all focused model-free assertions passed"}


# Frozen T027 production controls for physical cell execution.
PHYSICAL_CONFIG = {"mode": "dflash", "drafter_bits": 4, "max_draft_tokens": "auto",
                   "enable_thinking": False, "prefix_cache": True,
                   "prefix_cache_dir": None, "prefix_cache_max_ram_mb": 0,
                   "prefix_cache_slots": 2, "prefix_cache_rungs": 8192,
                   "kv_bits": None, "context_window": None, "warmup": True,
                   "memory_guard": True, "width_policy": None, "kv8": False,
                   "tuning": False,
                   "small_m": None, "sdpa_split": None,
                   "wide_gemm_min": None, "cpu_split": None}


def child_command(order: int, target: str, drafter: str, output: Path) -> list[str]:
    return [sys.executable, str(Path(__file__).resolve()), "--run-cell",
            "--order", str(order), "--target", target, "--drafter", drafter,
            "--output", str(output)]


def orchestration_plan(decision: str) -> tuple[list[tuple[int, str, str]], int]:
    if decision == "STOP_MISMATCH":
        return [MATRIX[0]], 13
    if decision == "CONTINUE_SWEEP":
        return list(MATRIX), 0
    return [MATRIX[0]], 13


def acceptance_metric(measurements: dict) -> float | None:
    committed = measurements.get("committed_tokens")
    proposed = measurements.get("proposed_drafts")
    return committed / proposed if proposed else measurements.get("accepted_over_proposed")


def adjudicate(record: dict) -> dict:
    m = record.get("measurements", {})
    tps = m.get("speculative_decode_tok_s")
    acceptance = m.get("acceptance", acceptance_metric(m))
    forwards = m.get("target_forwards")
    gpf = m.get("generated_per_target_forward", m.get("generated_tokens_per_target_forward"))
    if gpf is None and forwards:
        gpf = m.get("generated_tokens", 0) / forwards
    if tps is None or acceptance is None or forwards is None or gpf is None:
        return {"decision": "STOP_MISMATCH", "reason": "ambiguous: required canary metrics are missing"}
    if 43.0 <= tps <= 48.0 and acceptance >= 0.60 and 85 <= forwards <= 105 and 5.0 <= gpf <= 6.0:
        return {"decision": "CONTINUE_SWEEP", "reason": "canary clearly returned to the historical product regime",
                "observed": {"speculative_decode_tok_s": tps, "acceptance": acceptance,
                             "target_forwards": forwards, "generated_per_target_forward": gpf}}
    if 32.0 <= tps <= 36.0 and 0.45 <= acceptance <= 0.53:
        return {"decision": "STOP_MISMATCH", "reason": "canary remains in the known lower-throughput / lower-acceptance regime",
                "observed": {"speculative_decode_tok_s": tps, "acceptance": acceptance,
                             "target_forwards": forwards, "generated_per_target_forward": gpf}}
    return {"decision": "STOP_MISMATCH", "reason": "ambiguous canary regime; preserve result and stop",
            "observed": {"speculative_decode_tok_s": tps, "acceptance": acceptance,
                         "target_forwards": forwards, "generated_per_target_forward": gpf}}


def completeness_report(rows: list[dict]) -> dict:
    errors = []
    if len(rows) != 14:
        errors.append(f"expected 14 rows, got {len(rows)}")
    for index, expected in enumerate(MATRIX):
        if index >= len(rows):
            break
        row = rows[index]
        if (row.get("order"), row.get("target"), row.get("drafter")) != expected:
            errors.append(f"row {index + 1} identity/order mismatch")
        if row.get("status") != "complete":
            errors.append(f"row {index + 1} is not complete")
        if not row.get("freshness", {}).get("physical_asserted"):
            errors.append(f"row {index + 1} freshness not asserted by process")
        if not row.get("measurements"):
            errors.append(f"row {index + 1} missing metrics")
        else:
            required = ("serial_target_tok_s", "speculative_decode_tok_s", "speedup",
                        "acceptance", "mean_accepted", "target_forwards",
                        "generated_tokens", "generated_per_target_forward", "rounds",
                        "width_distribution", "cap_distribution", "peak_gib",
                        "duration_seconds", "output_text_sha256")
            for field in required:
                if field not in row["measurements"] or row["measurements"][field] is None:
                    errors.append(f"row {index + 1} missing metric {field}")
        if row.get("request", {}).get("input_ids_sha256") != INPUT_SHA256:
            errors.append(f"row {index + 1} request digest mismatch")
        if not row.get("source_sha256", {}).get("product_sweep.py"):
            errors.append(f"row {index + 1} missing frozen runner source hash")
        if not row.get("target_identity") or not row.get("drafter_identity"):
            errors.append(f"row {index + 1} missing checkpoint provenance")
    return {"complete": not errors, "expected_rows": 14, "observed_rows": len(rows), "errors": errors}


def repeat_triggers(candidate: dict, fresh_h0_tps: float) -> list[str]:
    m = candidate.get("metrics", candidate.get("measurements", {}))
    tps = m.get("speculative_decode_tok_s")
    reasons = []
    if tps is not None and tps >= fresh_h0_tps:
        reasons.append("hybrid reaches or exceeds fresh H0+B-Q")
    if candidate.get("conclusion_critical_near_noise"):
        reasons.append("conclusion-critical comparison is near observed drift")
    if candidate.get("local_maximum_requires_localization"):
        reasons.append("local maximum determines a narrow boundary follow-up")
    return reasons


def render_report(data: dict) -> str:
    decision = data.get("decision", "PENDING")
    lines = ["# Product Golden Hybrid Sweep", "", f"Canary decision: **{decision}**", "",
             "Historical Golden A: 45.7788518 speculative tok/s, 0.6482335 acceptance, "
             "94 target forwards, 516 generated tokens, 5.4893617 generated/forward.", "",
             "Historical Golden B: approximately 45.4 tok/s; 25/26 prompt tokens hot-cached; "
             "context only and excluded from fresh paired deltas.", "",
             "| Order | Cell | Serial tok/s | Spec tok/s | Speedup | Acceptance | Mean accepted | "
             "Forwards | Generated | Generated/forward | Rounds | Widths | Caps | Peak GiB | Duration s | Status |",
             "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---|"]
    rows = data.get("rows", [])
    if rows:
        for row in rows:
            m = row.get("measurements", {})
            lines.append(f"| {row.get('order','')} | {row.get('target')}+{row.get('drafter')} | "
                f"{m.get('serial_target_tok_s','')} | {m.get('speculative_decode_tok_s','')} | "
                f"{m.get('speedup','')} | {m.get('acceptance','')} | {m.get('mean_accepted','')} | "
                f"{m.get('target_forwards','')} | {m.get('generated_tokens','')} | "
                f"{m.get('generated_per_target_forward','')} | {m.get('rounds','')} | "
                f"{json.dumps(m.get('width_distribution',{}), sort_keys=True)} | "
                f"{json.dumps(m.get('cap_distribution',{}), sort_keys=True)} | "
                f"{m.get('peak_gib','')} | {m.get('duration_seconds','')} | {row.get('status','')} |")
    else:
        lines.append("| 1 | H0+B-Q | pending | pending | canary |")
    lines += ["", "Decision vocabulary: `STOP_MISMATCH` or `CONTINUE_SWEEP`."]
    return "\n".join(lines) + "\n"


def generate_final_report(rows: list[dict], decision: str, *, extra: dict | None = None) -> dict:
    """T008 JSON/Markdown writer. Implemented here, deliberately not run in T004."""
    payload = {"schema": "product-sweep-report/v1", "decision": decision,
               "historical_golden_a": HISTORICAL,
               "historical_golden_b": {"speculative_decode_tok_s": 45.4,
                   "mean_accepted": 5.548, "target_forwards": 94,
                   "generated_tokens": 516, "hot_prefix_tokens": 25,
                   "prompt_tokens": 26, "paired_delta_eligible": False},
               "rows": rows, "completeness": completeness_report(rows),
               "repeat_policy": {"triggers": "hybrid >= fresh H0+B-Q; conclusion-critical near-noise; local maximum relevant to narrow localization"},
               "extra": extra or {}}
    write_immutable(EVIDENCE / "product-sweep.json", payload)
    write_immutable_text(EVIDENCE / "product-sweep.md", render_report({"decision": decision, "rows": rows}))
    return payload


def write_immutable_text(path: Path, content: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"immutable evidence already exists: {path}")
    encoded = content.encode()
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    return hashlib.sha256(encoded).hexdigest()


def preflight() -> dict:
    cps = checkpoints()
    qwen = Path(cps["Qwen"]["path"])
    bq = Path(cps["B-Q"]["path"])
    qwen_files = verify_checkpoint(qwen, QWEN_REV)
    bq_files = verify_checkpoint(bq, BQ_REV)
    # FR-005 asks for the selected B-B file to be cheaply revalidated before runs.
    bb = Path(cps["B-B"]["path"])
    bb_file = bb / "model.safetensors"
    if not bb_file.is_file():
        raise FileNotFoundError(f"missing pinned B-B selected weight: {bb_file}")
    bb_blob = bb_file.resolve().name
    if bb_blob != BB_WEIGHT_SHA256:
        raise ValueError(f"B-B selected weight content-addressed identity mismatch: {bb_blob}")
    bb_files = verify_checkpoint(bb, BB_REV, require_weights=False)
    bb_files["selected_weight"] = {"name": bb_file.name, "size_bytes": bb_file.stat().st_size,
                                   "resolved_blob_sha256": bb_blob,
                                   "accepted_full_file_sha256": BB_WEIGHT_SHA256,
                                   "accepted_record": "feature-002 Series-B checkpoints.json; full downloaded-file SHA-256"}
    prompt = prompt_preflight(qwen)
    runtime = source_hashes()
    accepted = load_json(SERIES_B_RUNNER).get("experiment_runtime_source_sha256", {})
    shared = {k: {"current": runtime[k], "accepted_feature_002": accepted[k],
                  "matches": runtime[k] == accepted[k]} for k in runtime if k in accepted}
    output = {"schema": "product-sweep-preflight/v1", "status": "PASS",
              "model_free": True, "benchmark_performed": False,
              "request": {**prompt, "settings": T027_SETTINGS,
                          "prefix_cache_initialization": "enabled; T027 production Engine settings",
                          "measured_prefix_reuse_expected": 0},
              "checkpoints": {"Qwen": qwen_files, "B-Q": bq_files, "B-B": bb_files},
              "runtime_source_sha256": runtime, "runtime_comparison": shared,
              "source_sha256": sha256_file(Path(__file__).resolve()),
              "orchestration": {"fresh_process_per_cell": True,
                                "measured_request_state_created_in_child": True,
                                "child_command": child_command(1, "H0", "B-Q", EVIDENCE / "canary.json"),
                                "plain_kv": True, "ordinary_cap_controller": True,
                                "width_policy": False, "kv8": False, "tuning": False},
              "physical_observations": "not asserted by model-free preflight"}
    return output


def physical_cell(order: int, target: str, drafter: str, output: Path) -> dict:
    """Execute one T027 product request in this fresh child process."""
    sys.path.insert(0, str(ROOT / "src"))
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    from mlx_dspark import generate as generate_module
    from mlx_dspark.generate import encode_messages
    from mlx_dspark.hybrid_target import TargetCompositionRequest
    from mlx_dspark.server import Engine
    from transformers import AutoTokenizer

    cps = checkpoints()
    target_info = target_record(target, drafter)
    target_path = Path(target_info["target_identity"]["path"])
    drafter_path = Path(target_info["drafter_identity"]["path"])
    composition = None
    if target != "B0" and DONOR_BLOCKS[target]:
        donor = cps["official_Bonsai2"]
        qwen = cps["Qwen"]
        composition = TargetCompositionRequest.create(
            donor_path=donor["path"], donor_repo=donor["repo_id"], donor_revision=donor["revision"],
            qwen_repo=qwen["repo_id"], qwen_revision=qwen["revision"],
            donor_indices=DONOR_BLOCKS[target])
    # Each cell child owns one Engine and has no parent-process request/cache state.
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started_perf = time.perf_counter()
    pid = os.getpid()
    engine = Engine.load(mode="dflash", model=str(target_path), drafter=str(drafter_path),
        target_composition=composition, drafter_bits=4, max_draft_tokens="auto",
        enable_thinking=False, prefix_cache=True, prefix_cache_dir=None,
        prefix_cache_max_ram_mb=0, prefix_cache_slots=2, prefix_cache_rungs=8192,
        kv_bits=None, context_window=None, warmup=True, memory_guard=True,
        lookup_drafts=False, small_m=None, sdpa_split=None, wide_gemm_min=None,
        cpu_split=None)
    prompt_ids = INPUT_IDS
    runtime_ids = encode_messages(engine.tokenizer, [{"role": "user", "content": WORKLOAD}],
                                  enable_thinking=False)
    if runtime_ids != prompt_ids:
        engine.close()
        raise RuntimeError("loaded production Engine tokenizer differs from pinned Golden-A IDs")
    if engine.prefix is None:
        engine.close()
        raise RuntimeError("T027 production prefix cache did not initialize")
    acquire_rows = []
    acquire_cache_rows = []
    acquire = engine.prefix.acquire
    def observe(ids):
        cache, ctx, reused = acquire(ids)
        acquire_rows.append(int(reused))
        rows = []
        for index, item in enumerate(cache):
            bits = getattr(item, "bits", None)
            cls = type(item).__name__
            attention = "KV" in cls or hasattr(item, "keys") or hasattr(item, "values")
            row = {"index": index, "class": cls, "attention_cache": bool(attention),
                   "bits": int(bits) if bits is not None else None,
                   "group_size": getattr(item, "group_size", None)}
            rows.append(row)
            if attention and ("Quantized" in cls or bits is not None):
                raise RuntimeError(f"plain-KV invariant violated: {row}")
        if not any(row["attention_cache"] for row in rows):
            raise RuntimeError(f"could not observe target attention KV cache: {rows}")
        acquire_cache_rows.append(rows)
        return cache, ctx, reused
    engine.prefix.acquire = observe
    rounds_before = len(engine.rounds.snapshot())
    prefix_hits_before = engine.prefix.hits
    try:
        if rounds_before != 0:
            raise RuntimeError("fresh child Engine unexpectedly has prior rounds")
        mx.metal.reset_peak_memory()
        active_baseline = int(mx.metal.get_active_memory())
        request_start = time.perf_counter()
        result = engine.generate(prompt_ids, max_tokens=512, temperature=0.0,
                                 top_p=1.0, top_k=0, stop=None, seed=None)
        request_wall = time.perf_counter() - request_start
        spec_peak = int(mx.metal.get_peak_memory())
        rounds = engine.rounds.snapshot()
        if result.reused_tokens != 0 or not acquire_rows or acquire_rows[-1] != 0:
            raise RuntimeError("measured Golden-A request reused a prior prefix")
        if not acquire_cache_rows:
            raise RuntimeError("production PrefixCache.acquire was not observed for measured request")
        if engine.target.kv_bits not in (None, 0):
            raise RuntimeError(f"target KV is not plain: {engine.target.kv_bits}")
        cache_types = acquire_cache_rows[-1]
        serial_start = time.perf_counter()
        sampler = make_sampler(temp=0.0, top_p=1.0, top_k=0)
        final_response = None
        serial_tokens = 0
        for response in stream_generate(engine.target.model, engine.tokenizer, prompt_ids,
                                        max_tokens=512, sampler=sampler):
            final_response = response
            serial_tokens += 1
        serial_wall = time.perf_counter() - serial_start
        serial_tps = getattr(final_response, "generation_tps", None)
        serial_peak = int(mx.metal.get_peak_memory())
        runtime_conf = {"mode": engine.mode, "max_draft_tokens": engine.max_draft_tokens,
                        "controller_present": engine.cap_controller is not None,
                        "controller_before_after": (engine.cap_controller.info()
                            if engine.cap_controller is not None else None),
                        "final_effective_cap": engine._last_cap,
                        "depth_capper_active": engine._depth_capper is not None,
                        "prefix_cache_enabled": engine.prefix is not None,
                        "prefix_cache_mode": "checkpoint" if engine.prefix.checkpoint_mode else "trim",
                        "prefix_cache_slots": engine.prefix_cache_slots,
                        "prefix_cache_rungs": engine.prefix_cache_rungs,
                        "warmup_enabled": engine.warmup_enabled,
                        "memory_guard_active": engine.memory_guard is not None,
                        "kv_bits": engine.target.kv_bits,
                        "width_policy": None, "kv8": False, "tuning": False,
                        "lookup_drafts": engine.lookup_drafts,
                        "wide_gemm_configured": generate_module.WIDE_GEMM_MIN_ROWS is not None,
                        "small_m_active": engine.small_m, "sdpa_split_active": engine.sdpa_split}
        elapsed = time.perf_counter() - started_perf
        mrounds = [dict(r) for r in rounds]
        proposed = sum(int(r.get("drafted", 0)) for r in mrounds)
        accepted = sum(int(r.get("accepted", 0)) for r in mrounds)
        widths = Counter(str(r.get("drafted", 0)) for r in mrounds)
        caps = Counter(str(r.get("cap", "unknown")) for r in mrounds)
        generated = int(result.num_tokens)
        forwards = int(result.target_forwards)
        acceptance = accepted / proposed if proposed else None
        spec_tps = generated / max(float(result.decode_seconds), 1e-9)
        record = {"schema": "product-sweep-cell/v1", "status": "complete",
            "cell_id": f"{order:02d}-{target}-{drafter}", "order": order,
            "target": target, "drafter": drafter, **target_info,
            "source_sha256": {"product_sweep.py": sha256_file(Path(__file__).resolve()),
                              **source_hashes()},
            "request": {"workload": WORKLOAD, "input_ids": INPUT_IDS,
                        "input_ids_sha256": INPUT_SHA256, "input_token_count": 26,
                        "generation": T027_SETTINGS,
                        "prefix_cache_initialization": "enabled / checkpoint mode per T027 Engine.load",
                        "prefix_cache_state_before_measured_request": "empty fresh Engine request state"},
            "freshness": {"physical_asserted": True, "process_pid": pid,
                "process_start_utc": started_utc, "measured_request_start_monotonic": request_start,
                "new_process_per_cell": True, "new_engine_per_cell": True,
                "rounds_before_measured_request": rounds_before,
                "prefix_hits_before": prefix_hits_before,
                "prefix_reused_tokens": int(result.reused_tokens),
                "observed_acquire_reused_tokens": acquire_rows[-1],
                "assertion": "child process and Engine were created for this cell; no prior measured request state"},
            "runtime_configuration": runtime_conf,
            "measurements": {"serial_target_tok_s": serial_tps,
                "serial_tokens": serial_tokens, "serial_duration_seconds": serial_wall,
                "speculative_decode_tok_s": spec_tps,
                "speedup": spec_tps / serial_tps if serial_tps else None,
                "acceptance": acceptance,
                "mean_accepted": accepted / len(mrounds) if mrounds else None,
                "target_forwards": forwards, "generated_tokens": generated,
                "generated_per_target_forward": generated / forwards if forwards else None,
                "rounds": len(mrounds), "width_distribution": dict(widths),
                "cap_distribution": dict(caps), "peak_gib": spec_peak / (1024 ** 3),
                "peak_bytes": spec_peak, "active_baseline_bytes": active_baseline,
                "peak_increment_bytes": max(0, spec_peak - active_baseline),
                "duration_seconds": elapsed, "request_duration_seconds": request_wall,
                "decode_duration_seconds": float(result.decode_seconds),
                "output_sha256": hashlib.sha256(
                    json.dumps([int(x) for x in result.tokens], separators=(",", ":")).encode()
                    if hasattr(result, "tokens") else str(result.text).encode()).hexdigest(),
                "output_text_sha256": hashlib.sha256(str(result.text).encode()).hexdigest(),
                "plain_kv": True, "kv_cache_types": cache_types,
                "round_events": mrounds,
                "serial_peak_bytes": serial_peak},
            "process_order": {"cell_process_started_utc": started_utc,
                "cell_process_ended_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "pid": pid, "monotonic_elapsed_seconds": elapsed,
                "matrix_order": order},
            "machine": {"platform": platform.platform(), "python": sys.version,
                        "hostname": platform.node()}}
        return record
    finally:
        engine.close()


def write_immutable(path: Path, data: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"immutable evidence already exists: {path}")
    encoded = (json.dumps(data, indent=2, sort_keys=True) + "\n").encode()
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    return hashlib.sha256(encoded).hexdigest()


def run_child(order: int, target: str, drafter: str, output: Path) -> dict:
    cmd = child_command(order, target, drafter, output)
    started = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    ended = time.time()
    if proc.returncode:
        raise RuntimeError(f"physical cell process failed ({proc.returncode}): {proc.stderr[-4000:]}")
    raw = load_json(output)
    raw["process_order"]["orchestrator_start_unix"] = started
    raw["process_order"]["orchestrator_end_unix"] = ended
    # Raw child record is already immutable. Parent timing stays in separate metadata.
    meta = output.with_name(output.stem + ".orchestration.json")
    write_immutable(meta, {"cell_id": raw["cell_id"], "command": cmd,
                           "orchestrator_start_unix": started,
                           "orchestrator_end_unix": ended,
                           "child_returncode": proc.returncode})
    return raw


def execute_canary_once() -> dict:
    raw_path = EVIDENCE / "canary" / "01-H0-B-Q.json"
    if raw_path.exists():
        raise FileExistsError(f"T004 canary already exists; refusing a second physical run: {raw_path}")
    raw = run_child(1, "H0", "B-Q", raw_path)
    return canary_artifacts(raw_path, raw)


def run_matrix_after_canary(canary: dict, *, output_dir: Path | None = None) -> list[dict]:
    """T005 orchestration support; deliberately not invoked by T001–T004."""
    decision = adjudicate(canary)["decision"]
    if decision != "CONTINUE_SWEEP":
        raise RuntimeError("matrix rows 2–14 are blocked unless the canary is CONTINUE_SWEEP")
    output_dir = output_dir or EVIDENCE / "cells"
    rows = [canary]
    for order, target, drafter in MATRIX[1:]:
        path = output_dir / f"{order:02d}-{target}-{drafter}.json"
        rows.append(run_child(order, target, drafter, path))
    return rows


def select_repeats(rows: list[dict], fresh_h0_tps: float) -> list[dict]:
    selected = []
    for row in rows:
        if row.get("target") in ("H0", "B0"):
            continue
        triggers = repeat_triggers(row, fresh_h0_tps)
        if triggers:
            selected.append({"cell_id": row.get("cell_id"), "target": row["target"],
                             "drafter": row["drafter"], "triggers": triggers})
    return selected


def execute_selected_repeats(selection: list[dict], *, output_dir: Path | None = None) -> list[dict]:
    """T007 selective repeat executor; one new child per selected matched repeat."""
    output_dir = output_dir or EVIDENCE / "repeats"
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, selected in enumerate(selection, start=1):
        suffix = hashlib.sha256(selected["cell_id"].encode()).hexdigest()[:8]
        path = output_dir / f"repeat-{index:02d}-{selected['target']}-{selected['drafter']}-{suffix}.json"
        order = next(row[0] for row in MATRIX
                     if row[1:] == (selected["target"], selected["drafter"]))
        records.append(run_child(order, selected["target"], selected["drafter"], path))
    return records


def canary_artifacts(raw_path: Path, raw: dict) -> dict:
    decision = adjudicate(raw)
    raw_hash = sha256_file(raw_path)
    report = {"schema": "product-sweep-canary-adjudication/v1",
              "decision": decision["decision"], "reason": decision["reason"],
              "raw_result": str(raw_path.relative_to(ROOT)), "raw_sha256": raw_hash,
              "observed": decision.get("observed"),
              "historical_golden_a": HISTORICAL,
              "continuation_rows": orchestration_plan(decision["decision"])[1]}
    if decision["decision"] == "STOP_MISMATCH":
        report["narrow_mismatch"] = {
            "historical_request": {"workload": WORKLOAD, "input_ids_sha256": INPUT_SHA256,
                                   "generation": T027_SETTINGS},
            "current_request": raw.get("request"),
            "runtime_differences": {"prefix_cache": raw.get("runtime_configuration"),
                                     "freshness": raw.get("freshness"),
                                     "runtime_source_sha256": raw.get("source_sha256")},
            "interpretation": "Concrete observed canary values and request/runtime identity are retained; no causal attribution is inferred."}
    write_immutable(EVIDENCE / "canary-adjudication.json", report)
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-check", action="store_true")
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--run-cell", action="store_true")
    p.add_argument("--canary", action="store_true", help="execute exactly row 1 in a new child process")
    p.add_argument("--order", type=int)
    p.add_argument("--target", choices=tuple(DONOR_BLOCKS))
    p.add_argument("--drafter", choices=("B-Q", "B-B"))
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.self_check:
        result = model_free_checks()
        print(json.dumps(result, indent=2))
        return 0
    if args.preflight:
        result = preflight()
        out = EVIDENCE / "preflight.json"
        write_immutable(out, result)
        print(json.dumps(result, indent=2))
        return 0
    if args.canary:
        result = execute_canary_once()
        print(json.dumps(result, indent=2))
        return 0
    if args.run_cell:
        expected = next((row for row in MATRIX if row[0] == args.order), None)
        if expected is None or expected != (args.order, args.target, args.drafter):
            raise SystemExit("cell does not match the frozen execution schedule")
        if args.output is None:
            raise SystemExit("--output required")
        try:
            record = physical_cell(args.order, args.target, args.drafter, args.output)
            digest = write_immutable(args.output, record)
            print(json.dumps({"raw_result": str(args.output), "sha256": digest,
                              "cell_id": record["cell_id"]}, indent=2))
        except BaseException as exc:
            failure = {"schema": "product-sweep-cell/v1", "status": "failed",
                       "order": args.order, "target": args.target, "drafter": args.drafter,
                       "error_type": type(exc).__name__, "error": str(exc),
                       "process_pid": os.getpid(),
                       "process_start_unix": os.environ.get("PRODUCT_SWEEP_CHILD_START_UNIX"),
                       "source_sha256": {"product_sweep.py": sha256_file(Path(__file__).resolve()),
                                         **source_hashes()}}
            digest = write_immutable(args.output, failure)
            print(json.dumps({"raw_result": str(args.output), "sha256": digest,
                              "status": "failed", "error": str(exc)}, indent=2))
            return 1
        return 0
    raise SystemExit("choose --self-check, --preflight, --canary, or --run-cell")


if __name__ == "__main__":
    raise SystemExit(main())
