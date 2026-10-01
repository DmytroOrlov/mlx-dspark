#!/usr/bin/env python3
"""Exact-input, offline-dry-runnable Bonsai/Chad golden performance runner.

Dry-run only reads config/tokenizer files and never loads model weights. A normal
run loads one runtime, warms it with a disposable request, resets its request
cache, then measures exactly one fresh 512-token golden request.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

TARGET = Path("/Users/do/.cache/huggingface/hub/models--nathansutton--Qwen3.8-27B-Ternary-Bonsai-2-DFlash2-MLX/snapshots/234cc925d77cd94683e47b1493bd964917aaf43b")
SIDECAR = Path("/Users/do/.cache/chad/dflash/015e795645c74b1a0eeef3b570031fb62e769bc5-q4g64")
CHAD_ROOT = Path("/Users/do/git/chad")
PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROMPT = "Write a production-quality Python LRU cache with tests and type hints."
MESSAGES = [{"role": "user", "content": PROMPT}]
PROMPT_IDS = [248045, 846, 198, 7734, 264, 5492, 21408, 12654, 436, 34810,
              6297, 440, 6813, 321, 913, 29642, 13, 248046, 198, 248045,
              74455, 198, 248068, 271, 248069, 271]
EXPECTED = {
    "target_config": "20a7ccac3e519b5b5d7c4eaa6451352dd4188da44b4385ffa6cf87a4f2b8a94a",
    "target_weights_blob": "68541bf9c72747df90764338fa966105b34d94e4e8751dd5d3403d732eedddcf",
    "target_hadamard_blob": "65b947b711b0ba2f654ee6839075f80dfa321685",
    "tokenizer": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    "tokenizer_config": "95c557768e6b88a7128befc7bfd3c7de50e5d51af9b8b33a9f4dee0e04f99679",
    "sidecar_config": "6fd15051e629eba87121298bce299f92f1ef99c14e01e144825bbf0aaeaca86f",
    "sidecar_weights": "876c368b5abfdd5de52ab14fcd5d3cccb07f0059f3903c984e4f17dbcee8a552",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_fingerprint(path: Path) -> str:
    """Use HF's cryptographic blob name for model weights; hash small metadata."""
    resolved = path.resolve()
    if resolved.parent.name == "blobs":
        return resolved.name
    return sha256(path)


def revision(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def tracked_dirty_paths(root: Path) -> list[str]:
    try:
        value = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=no"],
            text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line[3:] for line in value.splitlines() if len(line) >= 4]


def runtime_source_fingerprints(runtime: str) -> dict:
    if runtime == "chad":
        root = CHAD_ROOT
        paths = [
            "src/chad/engine.py",
            "src/chad/mlx_dflash.py",
            "src/chad/mlx_fastpath.py",
            "src/chad/config.py",
        ]
    else:
        root = PROJECT_ROOT
        paths = [
            "src/mlx_dspark/generate.py",
            "src/mlx_dspark/target.py",
            "src/mlx_dspark/dflash_model.py",
            "src/mlx_dspark/dflash_width_policy.py",
            "src/mlx_dspark/load.py",
            "src/mlx_dspark/calibrate.py",
        ]
    return {
        rel: sha256(root / rel)
        for rel in paths
        if (root / rel).is_file()
    }


def sysctl_value(name: str) -> str | None:
    try:
        return subprocess.check_output(
            ["sysctl", "-n", name], text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def machine_record() -> dict:
    ram = sysctl_value("hw.memsize")
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hardware_model": sysctl_value("hw.model"),
        "cpu_brand": sysctl_value("machdep.cpu.brand_string"),
        "ram_bytes": int(ram) if ram and ram.isdigit() else None,
    }


def mlx_peak_memory_bytes() -> int | None:
    try:
        import mlx.core as mx
        getter = getattr(mx.metal, "get_peak_memory", None)
        return int(getter()) if getter is not None else None
    except Exception:
        return None


def versions() -> dict:
    out = {"python": platform.python_version()}
    for name in ("mlx", "mlx-lm"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def paths_and_fingerprints(*, dry_run: bool) -> dict:
    required = [TARGET / "config.json", TARGET / "model.safetensors", TARGET / "hadamard.json",
                TARGET / "tokenizer.json", TARGET / "tokenizer_config.json",
                SIDECAR / "config.json", SIDECAR / "model.safetensors"]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Required local inputs missing: " + ", ".join(missing))
    observed = {
        "target_config": sha256(TARGET / "config.json"),
        "target_weights_blob": file_fingerprint(TARGET / "model.safetensors"),
        "target_hadamard_blob": file_fingerprint(TARGET / "hadamard.json"),
        "tokenizer": sha256(TARGET / "tokenizer.json"),
        "tokenizer_config": sha256(TARGET / "tokenizer_config.json"),
        "sidecar_config": sha256(SIDECAR / "config.json"),
        "sidecar_weights": (
            EXPECTED["sidecar_weights"]
            if dry_run
            else sha256(SIDECAR / "model.safetensors")
        ),
    }
    if observed != EXPECTED:
        raise ValueError(f"Local fingerprint mismatch: {observed!r}")
    sidecar_stat = (SIDECAR / "model.safetensors").stat()
    if sidecar_stat.st_size != 1146451614:
        raise ValueError(f"Physical sidecar weight size changed: {sidecar_stat.st_size}")
    observed["sidecar_weight_size"] = sidecar_stat.st_size
    observed["sidecar_weight_mtime_ns"] = sidecar_stat.st_mtime_ns
    return observed


def template_and_tokenize() -> tuple[str, list[int]]:
    # Transformers tokenizer loading is CPU-only and restricted to this local
    # snapshot. Importing mlx_lm.utils initializes MLX/Metal even though its
    # tokenizer loader does not read weights, defeating model-free dry-run.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(TARGET), local_files_only=True, trust_remote_code=False)
    rendered = tokenizer.apply_chat_template(
        MESSAGES, add_generation_prompt=True, enable_thinking=False,
        tokenize=False)
    encoded = tokenizer.apply_chat_template(
        MESSAGES, add_generation_prompt=True, enable_thinking=False,
        tokenize=True)
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    ids = [int(x) for x in encoded]
    if ids != PROMPT_IDS:
        raise ValueError(f"Golden prompt IDs mismatch: {ids!r}")
    return str(rendered), ids


def common_record(runtime: str, arm: str, output: str, dry_run: bool) -> dict:
    fingerprints = paths_and_fingerprints(dry_run=dry_run)
    rendered, ids = template_and_tokenize()
    target_config = json.loads((TARGET / "config.json").read_text())
    context_config = target_config.get("text_config", target_config)
    context_window = context_config.get("max_position_embeddings")
    chad = runtime == "chad"
    fixed_diag = runtime == "mlx-dspark" and arm == "fixed7"
    effective_env = {k: v for k, v in os.environ.items()
                     if k.startswith(("MLX_DSPARK_", "CHAD_"))}
    if chad:
        effective_env.update({"CHAD_MODEL": str(TARGET), "CHAD_DFLASH_PATH": str(SIDECAR),
                              "CHAD_KV_BITS": "0", "CHAD_NO_PREFIX_CACHE": "1",
                              "CHAD_DFLASH_DRAFT": "7" if arm == "fixed7" else None,
                              "CHAD_DFLASH_ADAPTIVE": "0" if arm == "fixed7" else "1"})
    else:
        effective_env["MLX_DSPARK_DFLASH_FORCE_WIDTH"] = "7" if fixed_diag else None
    return {
        "schema": "bonsai-golden-perf-v1", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": runtime, "arm": arm,
        "runtime_source_revision": revision(CHAD_ROOT if chad else PROJECT_ROOT),
        "runtime_source": {
            "revision": revision(CHAD_ROOT if chad else PROJECT_ROOT),
            "dirty_tracked_paths": tracked_dirty_paths(CHAD_ROOT if chad else PROJECT_ROOT),
            "source_file_sha256": runtime_source_fingerprints(runtime),
        },
        "runner_sha256": sha256(Path(__file__).resolve()),
        "target": {"physical_path": str(TARGET), "snapshot_commit": TARGET.name,
                   "source_model_build": "Prism ML Ternary Bonsai 2 27B; Chad-oriented nathansutton text-only repack",
                   "prism_source_revision": "3f926b415992eaa2ae9dd7b573706494d6bbf787",
                   "fingerprints": {k: v for k, v in fingerprints.items() if k.startswith("target_")}},
        "context_window_tokens": context_window,
        "sidecar": {
            "path": str(SIDECAR),
            "fingerprints": {k: v for k, v in fingerprints.items() if k.startswith("sidecar_")},
            "weight_fingerprint_verification": (
                "expected_sha256_plus_current_size_mtime"
                if dry_run else "actual_sha256_current_file"
            ),
        },
        "tokenizer": {"path": str(TARGET), "fingerprints": {k: v for k, v in fingerprints.items() if k.startswith("tokenizer")}},
        "prompt": PROMPT, "messages": MESSAGES, "rendered_prompt": rendered,
        "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        "prompt_ids": ids, "prompt_token_count": len(ids), "prompt_ids_equal": True,
        "thinking_disabled": {"mechanism": "apply_chat_template(enable_thinking=False)"},
        "sampling": {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "max_tokens": 512},
        "target_kv": {"class": "KVCache", "bits": 0, "group_size": None},
        "width_control": {"mode": arm, "ceiling": 7,
                          "diagnostic_control": fixed_diag,
                          "label": "diagnostic fixed-width execution-cost control; not production configuration or correctness evidence" if fixed_diag else None},
        "prefix_cache": "disabled; fresh request cache",
        "warmup": {"performed": not dry_run, "recorded_method": "one disposable 8-token generation through the selected arm, then a fresh request cache; excluded from measurement"},
        "environment": effective_env,
        "versions": versions(), "machine": machine_record(),
        "dry_run": dry_run, "output_path": str(Path(output).expanduser().resolve()),
        "measurements": None if dry_run else {},
    }


def run_mlx(record: dict, arm: str) -> dict:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from mlx_dspark.generate import dflash_generate
    from mlx_dspark.load import load_dflash, load_target
    from mlx_dspark.calibrate import apply_sdpa_split, apply_small_m

    force_key = "MLX_DSPARK_DFLASH_FORCE_WIDTH"
    before = os.environ.get(force_key)
    if arm == "fixed7":
        os.environ[force_key] = "7"
    else:
        os.environ.pop(force_key, None)
    try:
        target, tokenizer = load_target(str(TARGET), require_tap=True, kv_bits=None)
        drafter, _ = load_dflash(str(SIDECAR), quantize=False)
        drafter.bind(target.model)
        # Match the production CLI setup before dflash_generate: WidthPolicy on
        # this Prism path is gated on the active small-M MMA dispatch.
        apply_small_m(target, drafter, target_repo=str(TARGET), drafter_repo=str(SIDECAR))
        apply_sdpa_split(target, target_repo=str(TARGET))

        def generate(token_count: int, trace: list[dict]):
            return dflash_generate(
                target, tokenizer, drafter, prompt_ids=record["prompt_ids"],
                apply_chat_template=False, max_new_tokens=token_count,
                max_draft_tokens=7, temperature=0.0, top_p=1.0, top_k=0,
                on_round=lambda **row: trace.append(dict(row)))

        warm_trace: list[dict] = []
        warm_result = generate(8, warm_trace)
        trace: list[dict] = []
        result = generate(512, trace)
        peak_memory = mlx_peak_memory_bytes()
        record["measurements"] = {
            "warmup": {"generated_tokens": warm_result.num_tokens, "output_excluded": True,
                       "trace": warm_trace},
            "width_trajectory": [r.get("drafted") for r in trace],
            "width_distribution": None,
            "rounds": trace, "accepted_counts": [r.get("accepted") for r in trace],
            "target_forwards": result.target_forwards,
            "generated_token_ids": result.token_ids,
            "generated_token_count": result.num_tokens, "output_text": result.text,
            "output_text_sha256": hashlib.sha256(result.text.encode()).hexdigest(),
            "decode_tokens_per_sec": result.num_tokens / max(result.decode_seconds, 1e-9),
            "prefill_tokens_per_sec": (len(record["prompt_ids"]) / result.prefill_seconds
                                        if result.prefill_seconds else None),
            "ttft_seconds": None, "round_latency_seconds": None,
            "per_round_margin": None,
            "peak_memory_bytes": peak_memory,
            "unavailable_reasons": {
                "ttft": "low-level generator does not expose TTFT",
                "round_latency": "on_round does not expose duration",
                "per_round_margin": "this generator callback does not expose margin",
                **({"peak_memory": "MLX metal peak-memory API unavailable"} if peak_memory is None else {}),
            },
        }
    finally:
        if before is None:
            os.environ.pop(force_key, None)
        else:
            os.environ[force_key] = before
    return record


def run_chad(record: dict, arm: str) -> dict:
    sys.path.insert(0, str(CHAD_ROOT / "src"))
    from chad.engine import Engine
    os.environ["CHAD_MODEL"] = str(TARGET)
    os.environ["CHAD_DFLASH_PATH"] = str(SIDECAR)
    os.environ["CHAD_KV_BITS"] = "0"
    os.environ["CHAD_NO_PREFIX_CACHE"] = "1"
    if arm == "fixed7":
        os.environ["CHAD_DFLASH_DRAFT"] = "7"
    else:
        os.environ.pop("CHAD_DFLASH_DRAFT", None)
    os.environ["CHAD_DFLASH_ADAPTIVE"] = "0" if arm == "fixed7" else "1"
    engine = Engine(model_id=str(TARGET), kv_bits=0, cache_dir=None)
    engine.load()
    tok = engine.tok
    rendered = tok.apply_chat_template(
        MESSAGES, add_generation_prompt=True, enable_thinking=False, tokenize=False)
    ids = tok.apply_chat_template(
        MESSAGES, add_generation_prompt=True, enable_thinking=False, tokenize=True)
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    ids = [int(x) for x in ids]
    if ids != PROMPT_IDS or ids != record["prompt_ids"]:
        raise ValueError(f"Chad tokenizer prompt IDs mismatch: {ids!r}")
    engine.temp, engine.top_p, engine.top_k = 0.0, 1.0, 0
    _, warm_stats = engine.generate(ids, max_tokens=8)
    engine.reset()
    text, stats = engine.generate(ids, max_tokens=512)
    out_ids = list(stats.gen_ids)
    record["rendered_prompt"] = str(rendered)
    record["rendered_prompt_sha256"] = hashlib.sha256(str(rendered).encode()).hexdigest()
    combined_hist = {}
    for phase_hist in (stats.draft_hist, stats.draft_hist_acting):
        for width, row in phase_hist.items():
            merged = combined_hist.setdefault(int(width), [0] * len(row))
            if len(merged) != len(row):
                raise ValueError("incompatible Chad draft histogram widths")
            for accepted, count in enumerate(row):
                merged[accepted] += int(count)

    peak_memory = mlx_peak_memory_bytes()

    record["measurements"] = {
        "warmup": {"generated_tokens": warm_stats.generated_tokens,
                   "output_excluded": True, "cache_reset_after": True},
        "width_trajectory": None, "rounds": None,
        "draft_hist_reasoning": stats.draft_hist,
        "draft_hist_acting": stats.draft_hist_acting,
        "draft_hist_combined": combined_hist,
        "width_distribution": {
            str(width): sum(row)
            for width, row in combined_hist.items()
        },
        "accepted_counts": None,
        "acceptance_length_distribution": {
            str(width): {
                str(accepted): count
                for accepted, count in enumerate(row)
                if count
            }
            for width, row in combined_hist.items()
        },
        "draft_proposed": stats.draft_proposed,
        "draft_accepted": stats.draft_accepted,
        "acceptance_rate": stats.accept_rate,
        "target_forwards": stats.forwards,
        "generated_token_ids": out_ids, "generated_token_count": len(out_ids),
        "output_text": text,
        "output_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "decode_tokens_per_sec": len(out_ids) / max(stats.gen_s, 1e-9),
        "prefill_tokens_per_sec": (stats.prompt_tokens / stats.prefill_s if stats.prefill_s else None),
        "ttft_seconds": stats.prefill_s or None,
        "round_latency_seconds": None, "per_round_margin": None,
        "peak_memory_bytes": peak_memory,
        "context_window_tokens": engine.effective_ctx,
        "unavailable_reasons": {
            "width_trajectory": "Engine GenStats exposes histogram, not ordered trajectory",
            "accepted_counts": "Engine GenStats exposes acceptance histograms, not ordered accepted-count sequence",
            "round_latency": "Engine GenStats does not expose per-round wall times",
            "per_round_margin": "Engine GenStats does not expose policy margin samples",
            **({"peak_memory": "MLX metal peak-memory API unavailable"} if peak_memory is None else {}),
        },
    }
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runtime", required=True, choices=("chad", "mlx-dspark"))
    ap.add_argument("--arm", required=True, choices=("fixed7", "adaptive"))
    ap.add_argument("--output", required=True, help="JSON result/config output path")
    ap.add_argument("--dry-run", action="store_true", help="tokenizer-only; never load weights")
    args = ap.parse_args()
    record = common_record(args.runtime, args.arm, args.output, args.dry_run)
    if args.dry_run:
        record["resolved_controls"] = {
            "chad": {"fixed7": "dflash_num_draft=7, adaptive=False; CHAD_DFLASH_DRAFT=7, CHAD_DFLASH_ADAPTIVE=0",
                     "adaptive": "dflash_num_draft ceiling=7, adaptive=True; CHAD_DFLASH_ADAPTIVE=1"},
            "mlx-dspark": {"fixed7": "MLX_DSPARK_DFLASH_FORCE_WIDTH=7; diagnostic fixed-width execution-cost control only",
                           "adaptive": "ordinary Prism WidthPolicy, ceiling=7; force-width variable absent"},
        }[args.runtime]
        record["weight_load_performed"] = False
    else:
        record = run_chad(record, args.arm) if args.runtime == "chad" else run_mlx(record, args.arm)
        record["weight_load_performed"] = True
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
