#!/usr/bin/env python3
"""Current-runtime, per-prompt Golden discovery using feature-003 T027 semantics."""
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
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
FEATURE = ROOT / "specs/004-golden-prompt-panel"
EVIDENCE = FEATURE / "evidence"
THIRD = ROOT / "specs/003-product-golden-hybrid-sweep"
CHECKPOINTS = ROOT / "specs/002-qwen-bonsai-hybrid-target/evidence/checkpoints.json"
QWEN_REPO, QWEN_REV = "mlx-community/Qwen3.8-27B-4bit", "10c35caafbb80f7dc6a7a432cdd11af10a6d4818"
BQ_REPO, BQ_REV = "incoai/Qwen3.8-27B-DFlash2", "015e795645c74b1a0eeef3b570031fb62e769bc5"
SETTINGS = {"thinking": False, "temperature": 0, "top_p": 1, "top_k": 0, "max_tokens": 512}
THRESHOLD_TOK_S, THRESHOLD_TOKENS = 40.0, 410

PROMPTS = (
    ("P01", "Python LRU cache", "Write a production-quality Python LRU cache with tests and type hints."),
    ("P02", "Longest increasing subsequence", "Write a Python function that returns the length of the longest increasing subsequence. Include a short explanation of its time complexity."),
    ("P03", "Palindrome", "Write a Python function that checks whether a string is a palindrome. Include a short explanation of its time and space complexity."),
    ("P04", "Binary search", "Write a Python function that performs binary search on a sorted list. Include type hints and explain its time complexity."),
    ("P05", "Merge sorted lists", "Write a Python function that merges two sorted lists into a single sorted list. Include tests and a short complexity analysis."),
    ("P06", "Maximum subarray", "Write a Python function that finds the maximum subarray sum using Kadane's algorithm. Explain why the algorithm runs in O(n) time."),
    ("P07", "First unique character", "Write a Python function that returns the first non-repeating character in a string. Include type hints and unit tests."),
    ("P08", "Stack", "Write a production-quality Python implementation of a stack with push, pop, peek, and is_empty methods. Include type hints and tests."),
    ("P09", "Anagrams", "Write a Python function that determines whether two strings are anagrams. Include tests and explain the time complexity."),
    ("P10", "Kth largest", "Write a Python function that finds the kth largest element in a list. Use an efficient algorithm and explain its expected time complexity."),
    ("P11", "Queue using two stacks", "Write a production-quality Python implementation of a queue using two stacks. Include type hints, docstrings, and tests."),
    ("P12", "Linked-list cycle", "Write a Python function that detects whether a linked list contains a cycle. Use O(1) extra space and explain the algorithm."),
    ("P13", "Tree traversal", "Write a Python function that returns the level-order traversal of a binary tree. Include type hints and a short complexity analysis."),
    ("P14", "Edit distance", "Write a Python function that computes the edit distance between two strings using dynamic programming. Include tests and explain the time and space complexity."),
    ("P15", "Group anagrams", "Write a Python function that groups a list of strings into groups of anagrams. Include type hints, tests, and complexity analysis."),
    ("P16", "Thread-safe singleton", "Write a production-quality Python thread-safe singleton class. Include type hints, documentation, and tests."),
    ("P17", "Topological sort", "Write a Python function that performs topological sorting on a directed acyclic graph. Include cycle detection, type hints, and tests."),
)

PHYSICAL_CONFIG = {"mode": "dflash", "drafter_bits": 4, "max_draft_tokens": "auto",
    "enable_thinking": False, "prefix_cache": True, "prefix_cache_dir": None,
    "prefix_cache_max_ram_mb": 0, "prefix_cache_slots": 2, "prefix_cache_rungs": 8192,
    "kv_bits": None, "context_window": None, "warmup": True, "memory_guard": True,
    "lookup_drafts": False, "width_policy": None, "kv8": False, "tuning": False,
    "small_m": None, "sdpa_split": None, "wide_gemm_min": None, "cpu_split": None}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prompt_sha(ids: list[int]) -> str:
    return sha256_bytes(json.dumps(ids, separators=(",", ":")).encode())


def immutable(path: Path, data: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable evidence: {path}")
    content = (json.dumps(data, indent=2, sort_keys=True) + "\n").encode()
    with path.open("xb") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    return sha256_bytes(content)


def checkpoint_paths() -> tuple[Path, Path]:
    cps = json.loads(CHECKPOINTS.read_text())["checkpoints"]
    return Path(cps["qwen"]["resolved_path"]), Path(cps["dflash"]["resolved_path"])


def encode(tokenizer: Any, text: str) -> tuple[str, list[int]]:
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": text}],
        add_generation_prompt=True, enable_thinking=False, tokenize=False)
    ids = tokenizer.apply_chat_template([{"role": "user", "content": text}],
        add_generation_prompt=True, enable_thinking=False, tokenize=True)
    if isinstance(ids, dict):
        ids = ids["input_ids"]
    elif hasattr(ids, "input_ids"):
        ids = ids.input_ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return rendered, [int(x) for x in ids]


def runtime_hashes() -> dict[str, str]:
    paths = ["src/mlx_dspark/server.py", "src/mlx_dspark/load.py", "src/mlx_dspark/generate.py",
             "src/mlx_dspark/target.py", "src/mlx_dspark/hybrid_target.py",
             "src/mlx_dspark/prism_pack.py", "src/mlx_dspark/calibrate.py",
             "specs/001-bonsai-rollback-parity/evidence/probes/t027_product_ab.py"]
    return {name: file_sha(ROOT / name) for name in paths}


def model_free_checks() -> dict:
    assert len(PROMPTS) == 17 and [p[0] for p in PROMPTS] == [f"P{i:02d}" for i in range(1, 18)]
    assert len({p[2] for p in PROMPTS}) == 17
    assert SETTINGS == {"thinking": False, "temperature": 0, "top_p": 1, "top_k": 0, "max_tokens": 512}
    assert SETTINGS["max_tokens"] != 128 and PHYSICAL_CONFIG["prefix_cache"] is True
    assert PHYSICAL_CONFIG["prefix_cache_max_ram_mb"] == 0 and PHYSICAL_CONFIG["max_draft_tokens"] == "auto"
    assert PHYSICAL_CONFIG["kv_bits"] is None and PHYSICAL_CONFIG["width_policy"] is None
    assert PHYSICAL_CONFIG["kv8"] is False and PHYSICAL_CONFIG["tuning"] is False
    src = inspect.getsource(physical_observation)
    assert "stream_generate(" not in src and "serial_target" not in src
    module = Path(__file__).read_text()
    assert "subprocess.run(" in module and "--run-observation" in module
    assert "Engine.load(" in src and "prefix_cache=True" in src
    assert "int(result.reused_tokens) != 0" in src
    assert "GOLDEN_CANDIDATE" in module and "CERTIFIED GOLDEN" in module
    assert THRESHOLD_TOK_S == 40.0 and THRESHOLD_TOKENS == 410
    assert status_for(409, 100.0) == "TOO_SHORT"
    assert status_for(410, 39.999) == "SLOW"
    assert status_for(410, 40.0) == "GOLDEN_CANDIDATE"
    assert child_command("P01", "screen", 1, Path("/tmp/x"))[0] == sys.executable
    assert repeat_plan([{"prompt_id": "P01", "status": "GOLDEN_CANDIDATE", "prompt_role": "screen",
                         "sequence": 1, "measurements": {"speculative_decode_tok_s": 42.0}}])
    sample = {"observations": [], "prompts": [], "anchor_observations": [],
              "counts": {"panel": 17, "long_enough": 0, "screen_40_plus_long": 0, "certified_new": 0,
                         "certified_new_prompt_ids": [], "screen_40_plus_prompt_ids": [], "p02_certified": False}}
    assert "Prompt" in report_markdown(sample) and "CERTIFIED GOLDEN" in report_markdown(sample)
    return {"status": "PASS", "checks": 18, "result": "focused model-free prompt-panel checks passed"}


def status_for(generated: int, tps: float) -> str:
    if generated < THRESHOLD_TOKENS:
        return "TOO_SHORT"
    return "GOLDEN_CANDIDATE" if tps >= THRESHOLD_TOK_S else "SLOW"


def child_command(prompt_id: str, role: str, sequence: int, path: Path) -> list[str]:
    return [sys.executable, str(Path(__file__).resolve()), "--run-observation",
            "--prompt-id", prompt_id, "--role", role, "--sequence", str(sequence),
            "--output", str(path)]


def preflight() -> dict:
    from transformers import AutoTokenizer
    qwen, bq = checkpoint_paths()
    for path, revision in ((qwen, QWEN_REV), (bq, BQ_REV)):
        if revision not in str(path.resolve()) or not (path / "config.json").is_file():
            raise RuntimeError(f"checkpoint identity/files mismatch: {path} expected {revision}")
        if not list(path.glob("*.safetensors")) and not (path / "model.safetensors.index.json").is_file():
            raise FileNotFoundError(f"checkpoint weights unavailable: {path}")
    tokenizer = AutoTokenizer.from_pretrained(str(qwen), local_files_only=True, trust_remote_code=False)
    rows = []
    for pid, label, text in PROMPTS:
        rendered, ids = encode(tokenizer, text)
        rows.append({"prompt_id": pid, "label": label, "text": text, "rendered_prompt": rendered,
                     "input_ids": ids, "input_sha256": prompt_sha(ids), "input_token_count": len(ids)})
    output = {"schema": "golden-prompt-panel-preflight/v1", "status": "PASS",
        "benchmark_performed": False, "target": {"repo_id": QWEN_REPO, "revision": QWEN_REV, "path": str(qwen)},
        "drafter": {"repo_id": BQ_REPO, "revision": BQ_REV, "path": str(bq)},
        "prompts": rows, "generation": SETTINGS,
        "request_semantics": {"engine": "feature-003 T027 production Engine path",
            "prefix_cache_initialization": "enabled, production defaults", "measured_request_prefix_reuse": 0,
            "fresh_process_and_engine_per_observation": True, "plain_kv": True,
            "ordinary_automatic_cap_controller": True, "width_policy": False, "kv8": False, "tuning": False,
            "serial_control": False},
        "runtime_source_sha256": runtime_hashes(), "runner_sha256": file_sha(Path(__file__).resolve()),
        "execution_schedule": {"screen_prompt_count": 17, "anchor_start": "P01 screen observation",
            "anchor_midpoint": "additional P01 after P09", "anchor_end": "additional P01 after P17"}}
    return output


def physical_observation(prompt_id: str, role: str, sequence: int) -> dict:
    sys.path.insert(0, str(ROOT / "src"))
    import mlx.core as mx
    from mlx_dspark.generate import encode_messages
    from mlx_dspark.server import Engine
    from transformers import AutoTokenizer

    prompt = next((p for p in PROMPTS if p[0] == prompt_id), None)
    if prompt is None:
        raise ValueError(f"unknown prompt id {prompt_id}")
    _, label, text = prompt
    qwen_path, bq_path = checkpoint_paths()
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    process_started = time.perf_counter()
    pid = os.getpid()
    engine = Engine.load(mode="dflash", model=str(qwen_path), drafter=str(bq_path),
        drafter_bits=4, max_draft_tokens="auto", enable_thinking=False, prefix_cache=True,
        prefix_cache_dir=None, prefix_cache_max_ram_mb=0, prefix_cache_slots=2,
        prefix_cache_rungs=8192, kv_bits=None, context_window=None, warmup=True,
        memory_guard=True, lookup_drafts=False, small_m=None, sdpa_split=None,
        wide_gemm_min=None, cpu_split=None)
    try:
        from mlx_dspark.generate import encode_messages as runtime_encode
        prompt_ids = runtime_encode(engine.tokenizer, [{"role": "user", "content": text}], enable_thinking=False)
        rendered, expected_ids = encode(engine.tokenizer, text)
        if prompt_ids != expected_ids:
            raise RuntimeError("production Engine prompt encoding differs from recorded prompt panel encoding")
        if engine.prefix is None:
            raise RuntimeError("T027 production prefix cache did not initialize")
        reused = []
        cache_rows = []
        original_acquire = engine.prefix.acquire
        def observe(ids):
            cache, context, n = original_acquire(ids)
            reused.append(int(n))
            rows = [{"class": type(item).__name__, "bits": getattr(item, "bits", None),
                "attention_cache": ("KV" in type(item).__name__ or hasattr(item, "keys") or hasattr(item, "values"))}
                for item in cache]
            if not any(row["attention_cache"] for row in rows):
                raise RuntimeError("could not observe target attention KV cache")
            if any(row["attention_cache"] and (row["bits"] is not None or "Quantized" in row["class"]) for row in rows):
                raise RuntimeError("plain-KV invariant violated")
            cache_rows.append(rows)
            return cache, context, n
        engine.prefix.acquire = observe
        rounds_before = len(engine.rounds.snapshot())
        prefix_hits_before = engine.prefix.hits
        if rounds_before != 0:
            raise RuntimeError("fresh Engine has prior speculative rounds")
        mx.metal.reset_peak_memory()
        baseline = int(mx.metal.get_active_memory())
        request_start = time.perf_counter()
        result = engine.generate(prompt_ids, max_tokens=512, temperature=0.0,
            top_p=1.0, top_k=0, stop=None, seed=None)
        request_duration = time.perf_counter() - request_start
        peak = int(mx.metal.get_peak_memory())
        if int(result.reused_tokens) != 0 or not reused or reused[-1] != 0:
            raise RuntimeError("measured request reused prior prompt prefix")
        if engine.target.kv_bits not in (None, 0):
            raise RuntimeError(f"target KV is not plain: {engine.target.kv_bits}")
        rounds = [dict(r) for r in engine.rounds.snapshot()]
        generated = int(result.num_tokens)
        decode_seconds = float(result.decode_seconds)
        tps = generated / max(decode_seconds, 1e-9)
        proposed = sum(int(r.get("drafted", 0)) for r in rounds)
        accepted = sum(int(r.get("accepted", 0)) for r in rounds)
        widths = Counter(str(r.get("drafted", 0)) for r in rounds)
        caps = Counter(str(r.get("cap", "unknown")) for r in rounds)
        text_out = str(result.text)
        output_tokens = getattr(result, "tokens", None)
        stop_reason = getattr(result, "stop_reason", None)
        return {"schema": "golden-prompt-observation/v1", "status": status_for(generated, tps),
            "prompt_id": prompt_id, "short_label": label, "prompt_role": role, "sequence": sequence,
            "text": text, "rendered_prompt": rendered, "input_ids": expected_ids,
            "input_sha256": prompt_sha(expected_ids), "input_token_count": len(expected_ids),
            "target": {"repo_id": QWEN_REPO, "revision": QWEN_REV, "path": str(qwen_path)},
            "drafter": {"repo_id": BQ_REPO, "revision": BQ_REV, "path": str(bq_path)},
            "source_sha256": {"prompt_panel.py": file_sha(Path(__file__).resolve()), **runtime_hashes()},
            "request": {"generation": SETTINGS, "plain_kv": True,
                "prefix_cache_initialization": "enabled / T027 production Engine settings",
                "prefix_reuse_tokens": int(result.reused_tokens)},
            "freshness": {"physical_asserted": True, "pid": pid, "new_process": True,
                "new_engine": True, "rounds_before_request": rounds_before,
                "prefix_hits_before_request": prefix_hits_before, "acquire_reused_tokens": reused[-1],
                "assertion": "new child process and Engine; no prior measured request state"},
            "runtime_configuration": {"mode": engine.mode, "controller_present": engine.cap_controller is not None,
                "controller_final": engine.cap_controller.info() if engine.cap_controller is not None else None,
                "final_effective_cap": engine._last_cap, "max_draft_tokens": engine.max_draft_tokens,
                "prefix_cache_enabled": engine.prefix is not None, "prefix_cache_mode":
                    "checkpoint" if engine.prefix.checkpoint_mode else "trim",
                "kv_bits": engine.target.kv_bits, "width_policy": None, "kv8": False, "tuning": False},
            "measurements": {"generated_tokens": generated, "speculative_decode_tok_s": tps,
                "acceptance": accepted / proposed if proposed else None,
                "mean_accepted": accepted / len(rounds) if rounds else None,
                "target_forwards": int(result.target_forwards),
                "generated_per_forward": generated / int(result.target_forwards) if result.target_forwards else None,
                "rounds": len(rounds), "width_distribution": dict(widths), "cap_distribution": dict(caps),
                "request_duration_seconds": request_duration, "decode_duration_seconds": decode_seconds,
                "peak_gib": peak / (1024 ** 3), "peak_bytes": peak, "active_baseline_bytes": baseline,
                "output_text": text_out, "output_text_sha256": sha256_bytes(text_out.encode()),
                "output_tokens_sha256": (sha256_bytes(json.dumps([int(x) for x in output_tokens], separators=(",", ":")).encode())
                    if output_tokens is not None else None),
                "stop_reason": stop_reason, "natural_completion_before_410": generated < 410,
                "round_events": rounds, "kv_cache_types": cache_rows[-1] if cache_rows else []},
            "process_order": {"process_start_utc": started_utc,
                "process_end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "pid": pid, "sequence": sequence, "role": role,
                "elapsed_seconds": time.perf_counter() - process_started},
            "machine": {"hostname": platform.node(), "platform": platform.platform(), "python": sys.version}}
    finally:
        engine.close()


def run_one(prompt_id: str, role: str, sequence: int, path: Path) -> dict:
    command = child_command(prompt_id, role, sequence, path)
    start, start_utc = time.time(), time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    end = time.time()
    if proc.returncode:
        if not path.exists():
            immutable(path, {"schema": "golden-prompt-observation/v1", "status": "FAILED",
                "prompt_id": prompt_id, "prompt_role": role, "sequence": sequence,
                "error": proc.stderr[-6000:], "returncode": proc.returncode})
        raise RuntimeError(f"observation failed ({prompt_id}/{role}); raw failure retained: {path}")
    record = json.loads(path.read_text())
    meta = path.with_name(path.stem + ".orchestration.json")
    immutable(meta, {"prompt_id": prompt_id, "role": role, "sequence": sequence,
        "command": command, "start_utc": start_utc, "start_unix": start, "end_unix": end,
        "returncode": proc.returncode, "stdout": proc.stdout[-2000:]})
    return record


def run_one_retaining_failure(prompt_id: str, role: str, sequence: int, path: Path) -> dict:
    try:
        return run_one(prompt_id, role, sequence, path)
    except Exception as exc:
        if path.exists():
            return json.loads(path.read_text())
        failure = {"schema": "golden-prompt-observation/v1", "status": "FAILED",
            "prompt_id": prompt_id, "prompt_role": role, "sequence": sequence,
            "error_type": type(exc).__name__, "error": str(exc)}
        immutable(path, failure)
        return failure


def screen_dir(attempt: int) -> Path:
    if attempt < 1:
        raise ValueError("attempt must be positive")
    return EVIDENCE / ("screen" if attempt == 1 else f"screen-attempt-{attempt:02d}")


def latest_screen_dir() -> Path:
    dirs = [p for p in EVIDENCE.glob("screen*") if p.is_dir() and any(p.glob("*-P[0-9][0-9]-screen.json"))]
    if not dirs:
        return EVIDENCE / "screen"
    return max(dirs, key=lambda p: 1 if p.name == "screen" else int(p.name.rsplit("-", 1)[-1]))


def run_screen(attempt: int = 1) -> list[dict]:
    outdir = screen_dir(attempt)
    if any(outdir.glob("*.json")):
        raise FileExistsError("screen evidence already exists; refusing to rerun/overwrite")
    records = []
    seq = 1
    for index, (pid, _, _) in enumerate(PROMPTS, start=1):
        path = outdir / f"{seq:02d}-{pid}-screen.json"
        records.append(run_one_retaining_failure(pid, "screen", seq, path)); seq += 1
        if index == 9:
            p = outdir / f"{seq:02d}-P01-anchor-mid.json"
            records.append(run_one_retaining_failure("P01", "anchor-mid", seq, p)); seq += 1
    p = outdir / f"{seq:02d}-P01-anchor-end.json"
    records.append(run_one_retaining_failure("P01", "anchor-end", seq, p))
    return records


def repeat_plan(records: list[dict]) -> list[dict]:
    rows = [r for r in records if r.get("prompt_role") == "screen"]
    selected = [r for r in rows if r.get("status") == "GOLDEN_CANDIDATE"]
    return [{"prompt_id": r["prompt_id"], "source_sequence": r["sequence"],
             "first_status": r["status"], "first_tps": r.get("measurements", {}).get("speculative_decode_tok_s"),
             "selection_reason": "screen GOLDEN_CANDIDATE"} for r in selected]


def fallback_repeat_plan(screen: list[dict], repeats: list[dict]) -> list[dict]:
    certified = set()
    for r in repeats:
        m = r.get("measurements", {})
        first = next((s.get("measurements", {}) for s in screen if s.get("prompt_id") == r.get("prompt_id")), {})
        if (m.get("generated_tokens", 0) >= THRESHOLD_TOKENS and m.get("speculative_decode_tok_s", 0) >= THRESHOLD_TOK_S
                and first.get("generated_tokens", 0) >= THRESHOLD_TOKENS
                and first.get("speculative_decode_tok_s", 0) >= THRESHOLD_TOK_S):
            certified.add(r["prompt_id"])
    if len(certified - {"P01"}) >= 2:
        return []
    already_repeated = {r.get("prompt_id") for r in repeats}
    eligible = [r for r in screen if r.get("prompt_role") == "screen" and r.get("prompt_id") not in already_repeated
        and r.get("prompt_id") != "P01" and r.get("measurements", {}).get("generated_tokens", 0) >= THRESHOLD_TOKENS
        and r.get("measurements", {}).get("speculative_decode_tok_s", 0) < THRESHOLD_TOK_S]
    eligible.sort(key=lambda r: (abs(float(r["measurements"].get("speculative_decode_tok_s", 0)) - THRESHOLD_TOK_S),
                                 -float(r["measurements"].get("speculative_decode_tok_s", 0))))
    return [{"prompt_id": r["prompt_id"], "source_sequence": r["sequence"], "first_status": r["status"],
        "first_tps": r.get("measurements", {}).get("speculative_decode_tok_s"),
        "selection_reason": "fallback: nearest-to-40 long-form non-golden"} for r in eligible[:3]]


def run_repeats() -> list[dict]:
    screen = [json.loads(p.read_text()) for p in sorted(latest_screen_dir().glob("*-P[0-9][0-9]-screen.json"))]
    candidates = repeat_plan(screen)
    records, outdir = [], EVIDENCE / "repeats"
    for seq, item in enumerate(candidates, 1):
        path = outdir / f"{seq:02d}-{item['prompt_id']}-repeat.json"
        records.append(run_one_retaining_failure(item["prompt_id"], "repeat", seq, path))
    fallbacks = fallback_repeat_plan(screen, records)
    for seq, item in enumerate(fallbacks, len(records) + 1):
        path = outdir / f"{seq:02d}-{item['prompt_id']}-repeat.json"
        records.append(run_one_retaining_failure(item["prompt_id"], "repeat", seq, path))
    certified = sum(1 for r in records[:len(candidates)] if r.get("measurements", {}).get("generated_tokens", 0) >= THRESHOLD_TOKENS
        and r.get("measurements", {}).get("speculative_decode_tok_s", 0) >= THRESHOLD_TOK_S and
        any(s.get("prompt_id") == r.get("prompt_id") and s.get("measurements", {}).get("generated_tokens", 0) >= THRESHOLD_TOKENS
            and s.get("measurements", {}).get("speculative_decode_tok_s", 0) >= THRESHOLD_TOK_S for s in screen))
    immutable(EVIDENCE / "repeat-selection.json", {"candidate_repeats": candidates,
        "new_certified_before_fallback": certified, "fallback_repeats": fallbacks,
        "fallback_used": bool(fallbacks)})
    return records


def report_data() -> dict:
    observations = []
    screen_dirs = sorted([p for p in EVIDENCE.glob("screen*") if p.is_dir()],
        key=lambda p: 1 if p.name == "screen" else int(p.name.rsplit("-", 1)[-1]))
    for folder in (*screen_dirs, EVIDENCE / "repeats"):
        if folder.exists():
            for path in sorted(folder.glob("*.json")):
                if path.name.endswith(".orchestration.json"):
                    continue
                try:
                    row = json.loads(path.read_text())
                except json.JSONDecodeError:
                    continue
                row["raw_result_path"] = str(path.relative_to(ROOT))
                row["raw_result_sha256"] = file_sha(path)
                observations.append(row)
    screens = {r["prompt_id"]: r for r in observations if r.get("prompt_role") == "screen"}
    repeats = {r["prompt_id"]: r for r in observations if r.get("prompt_role") == "repeat"}
    rows = []
    for pid, label, text in PROMPTS:
        first, second = screens.get(pid), repeats.get(pid)
        m1 = (first or {}).get("measurements", {})
        m2 = (second or {}).get("measurements", {})
        if first is None or first.get("status") == "FAILED" or (second and second.get("status") == "FAILED"):
            final = "FAILED"
        elif m1.get("generated_tokens", 0) < THRESHOLD_TOKENS:
            final = "TOO SHORT"
        elif second and m2.get("generated_tokens", 0) >= THRESHOLD_TOKENS and m2.get("speculative_decode_tok_s", 0) >= THRESHOLD_TOK_S and m1.get("speculative_decode_tok_s", 0) >= THRESHOLD_TOK_S:
            final = "CERTIFIED GOLDEN"
        elif m1.get("speculative_decode_tok_s", 0) >= THRESHOLD_TOK_S:
            final = "GOLDEN CANDIDATE awaiting repeat"
        else:
            final = "SLOW but long enough"
        rows.append({"prompt_id": pid, "short_label": label, "text": text, "generated_tokens": m1.get("generated_tokens"),
            "spec_tps_run1": m1.get("speculative_decode_tok_s"), "spec_tps_run2": m2.get("speculative_decode_tok_s"),
            "minimum_certified_tps": min(m1["speculative_decode_tok_s"], m2["speculative_decode_tok_s"]) if final == "CERTIFIED GOLDEN" else None,
            "acceptance": m1.get("acceptance"), "target_forwards": m1.get("target_forwards"),
            "generated_per_forward": m1.get("generated_per_forward"), "widths": m1.get("width_distribution"),
            "peak_gib": m1.get("peak_gib"), "status": final,
            "input_ids": first.get("input_ids") if first else None,
            "input_sha256": first.get("input_sha256") if first else None,
            "input_token_count": first.get("input_token_count") if first else None,
            "screen_raw_path": first.get("raw_result_path") if first else None,
            "repeat_raw_path": second.get("raw_result_path") if second else None})
    return {"schema": "golden-prompt-panel-report/v1", "thresholds": {"minimum_generated_tokens": THRESHOLD_TOKENS,
        "minimum_speculative_tok_s": THRESHOLD_TOK_S}, "observations": observations, "prompts": rows,
        "counts": {"panel": len(PROMPTS), "long_enough": sum((r["generated_tokens"] or 0) >= THRESHOLD_TOKENS for r in rows),
            "screen_40_plus_long": sum((r["generated_tokens"] or 0) >= THRESHOLD_TOKENS and (r["spec_tps_run1"] or 0) >= THRESHOLD_TOK_S for r in rows),
            "certified_new": sum(r["status"] == "CERTIFIED GOLDEN" and r["prompt_id"] != "P01" for r in rows),
            "certified_prompt_ids": [r["prompt_id"] for r in rows if r["status"] == "CERTIFIED GOLDEN"],
            "long_enough_prompt_ids": [r["prompt_id"] for r in rows if (r["generated_tokens"] or 0) >= THRESHOLD_TOKENS],
            "screen_40_plus_prompt_ids": [r["prompt_id"] for r in rows if (r["generated_tokens"] or 0) >= THRESHOLD_TOKENS and (r["spec_tps_run1"] or 0) >= THRESHOLD_TOK_S],
            "certified_new_prompt_ids": [r["prompt_id"] for r in rows if r["status"] == "CERTIFIED GOLDEN" and r["prompt_id"] != "P01"],
            "p02_certified": next((r["status"] == "CERTIFIED GOLDEN" for r in rows if r["prompt_id"] == "P02"), False)},
        "anchor_observations": [r for r in observations if r.get("prompt_role", "").startswith("anchor") or
             (r.get("prompt_role") == "screen" and r.get("prompt_id") == "P01")],
        "qualification_note": "No throughput average across prompts is used."}


def report_markdown(data: dict) -> str:
    groups = ["CERTIFIED GOLDEN", "GOLDEN CANDIDATE awaiting repeat", "SLOW but long enough", "TOO SHORT", "FAILED"]
    lines = ["# Golden Prompt Panel Results", "", "Each prompt is shown individually. No aggregate throughput is used as a headline.",
        "", f"Screen: {data['counts']['panel']} prompts; {data['counts']['long_enough']} naturally generated at least 410 tokens; "
        f"{data['counts']['screen_40_plus_long']} of those reached 40 tok/s in the screen; "
        f"{data['counts']['certified_new']} new prompts certified.", "",
        "Historical LIS/P02 qualifies only if its current screen and fresh repeat both independently meet the 410-token and 40-tok/s thresholds.", "",
        f"The historical LIS/P02 prompt {'does' if data['counts']['p02_certified'] else 'does not'} certify under this current >=410-token protocol.",
        f"Certified new prompts: {', '.join(data['counts']['certified_new_prompt_ids']) or 'none'}.",
        f"Screen >=40 tok/s and >=410 tokens: {', '.join(data['counts']['screen_40_plus_prompt_ids']) or 'none'}.",
        "If several prompts appear in these lists, fast speculative decoding spans several coding tasks; if the list is concentrated, that is prompt sensitivity.", ""]
    for group in groups:
        lines.extend([f"## {group}", "", "| Prompt | Short label | Generated tokens | Spec tok/s run 1 | Spec tok/s run 2 | Minimum certified tok/s | Acceptance | Target forwards | Generated/forward | Widths | Peak GiB | Status |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|"])
        for r in data["prompts"]:
            if r["status"] != group:
                continue
            fmt = lambda x, n=3: "—" if x is None else f"{x:.{n}f}"
            prompt_cell = r["text"].replace("|", "\\|")
            lines.append(f"| {r['prompt_id']}: {prompt_cell} | {r['short_label']} | {r['generated_tokens'] if r['generated_tokens'] is not None else '—'} | {fmt(r['spec_tps_run1'])} | {fmt(r['spec_tps_run2'])} | {fmt(r['minimum_certified_tps'])} | {fmt(r['acceptance'])} | {r['target_forwards'] if r['target_forwards'] is not None else '—'} | {fmt(r['generated_per_forward'])} | {r['widths'] or '—'} | {fmt(r['peak_gib'])} | {r['status']} |")
        lines.append("")
    lines += ["## P01 drift anchors", "", "Raw P01 observations are retained and not used to normalize candidate rates.", "",
        "| Role | Sequence | Generated | Spec tok/s | Input SHA-256 | Raw evidence |", "|---|---:|---:|---:|---|---|"]
    for r in data["anchor_observations"]:
        m = r.get("measurements", {})
        lines.append(f"| {r.get('prompt_role')} | {r.get('sequence')} | {m.get('generated_tokens')} | {m.get('speculative_decode_tok_s', 0):.3f} | {r.get('input_sha256')} | `{r.get('raw_result_path', 'screen record')}` |")
    return "\n".join(lines) + "\n"


def validate_and_report() -> dict:
    data = report_data()
    missing = [r[0] for r in PROMPTS if not any(o.get("prompt_id") == r[0] and o.get("prompt_role") == "screen" for o in data["observations"])]
    data["validation"] = {"screen_rows_expected": 17, "screen_rows_found": 17-len(missing), "missing_prompt_ids": missing,
        "pass": not missing and all(o.get("input_ids") and o.get("input_sha256") and o.get("freshness", {}).get("physical_asserted")
            for o in data["observations"] if o.get("status") != "FAILED")}
    if missing:
        raise RuntimeError(f"incomplete screen; missing {missing}")
    immutable(EVIDENCE / "prompt-panel.json", data)
    md = EVIDENCE / "prompt-panel.md"
    if md.exists():
        raise FileExistsError(f"refusing to overwrite report: {md}")
    md.write_text(report_markdown(data))
    return data


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-check", action="store_true")
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--screen", action="store_true")
    p.add_argument("--attempt", type=int, default=1, help="immutable screen attempt number")
    p.add_argument("--repeat", action="store_true")
    p.add_argument("--report", action="store_true")
    p.add_argument("--run-observation", action="store_true")
    p.add_argument("--prompt-id", choices=[r[0] for r in PROMPTS])
    p.add_argument("--role", choices=("screen", "anchor-mid", "anchor-end", "repeat"))
    p.add_argument("--sequence", type=int)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if a.self_check:
        print(json.dumps(model_free_checks(), indent=2)); return 0
    if a.preflight:
        data = preflight()
        n = 1
        out = EVIDENCE / "preflight.json"
        while out.exists():
            n += 1
            out = EVIDENCE / f"preflight-{n:02d}.json"
        immutable(out, data)
        print(json.dumps({"status": data["status"], "prompts": len(data["prompts"]), "runner_sha256": data["runner_sha256"], "artifact": str(out)}, indent=2)); return 0
    if a.screen:
        rows = run_screen(a.attempt); print(json.dumps({"observations": len(rows), "attempt": a.attempt, "screen_dir": str(screen_dir(a.attempt))}, indent=2)); return 0
    if a.repeat:
        rows = run_repeats(); print(json.dumps({"repeats": len(rows)}, indent=2)); return 0
    if a.report:
        data = validate_and_report(); print(json.dumps(data["counts"], indent=2)); return 0
    if a.run_observation:
        if not all((a.prompt_id, a.role, a.sequence is not None, a.output)):
            raise SystemExit("--prompt-id, --role, --sequence, and --output required")
        try:
            record = physical_observation(a.prompt_id, a.role, a.sequence)
            digest = immutable(a.output, record)
            print(json.dumps({"status": record["status"], "sha256": digest, "raw_result": str(a.output)}))
            return 0
        except BaseException as exc:
            if not a.output.exists():
                immutable(a.output, {"schema": "golden-prompt-observation/v1", "status": "FAILED",
                    "prompt_id": a.prompt_id, "prompt_role": a.role, "sequence": a.sequence,
                    "error_type": type(exc).__name__, "error": str(exc), "pid": os.getpid()})
            print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr); return 1
    raise SystemExit("choose --self-check, --preflight, --screen, --repeat, --report, or --run-observation")


if __name__ == "__main__":
    raise SystemExit(main())
