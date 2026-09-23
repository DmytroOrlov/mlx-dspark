#!/usr/bin/env python3
"""Process-local plain-KV product A/B runner; --preflight never imports MLX."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BONSAI = Path("/Users/do/.cache/huggingface/hub/models--nathansutton--Qwen3.8-27B-Ternary-Bonsai-2-DFlash2-MLX/snapshots/234cc925d77cd94683e47b1493bd964917aaf43b")
QWEN = Path("/Users/do/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots/10c35caafbb80f7dc6a7a432cdd11af10a6d4818")
QWEN_DRAFTER = Path("/Users/do/.cache/huggingface/hub/models--incoai--Qwen3.8-27B-DFlash2/snapshots/015e795645c74b1a0eeef3b570031fb62e769bc5")
BONSAI_DRAFTER = Path("/Users/do/.cache/chad/dflash/015e795645c74b1a0eeef3b570031fb62e769bc5-q4g64")
BONSAI_SPECIFIC_DRAFTER = Path("/Users/do/models/Qwen3.8-27B-DFlash2-ternary-bonsai2")
LRU = "Write a production-quality Python LRU cache with tests and type hints."
CODING_TASK = (
    "Review the target resolver in src/mlx_dspark/load.py for Qwen3.6 27B "
    "routing collisions with Ternary Bonsai and dense Qwen3. If a collision or "
    "missing guard exists, make the smallest correction. Preserve quantization-agnostic "
    "Qwen3.6 resolution and Bonsai's variant-specific mapping. Report the changed "
    "behavior and run the focused resolver tests."
)
TEST_COMMAND = "uv run pytest tests/test_resolve.py -q"
PROMPTS = {"lru": LRU, "resolver": CODING_TASK}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize_ids(value) -> list[int]:
    if hasattr(value, "input_ids"):
        value = value.input_ids
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(x) for x in value]


def safetensors_header(path: Path) -> dict:
    """Read only safetensors length/header bytes, never tensor payloads."""
    if not path.is_file():
        return {"path": str(path), "present": False}
    with path.open("rb") as f:
        raw_len = f.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"Truncated safetensors header: {path}")
        header_len = struct.unpack("<Q", raw_len)[0]
        header = json.loads(f.read(header_len))
    metadata = header.pop("__metadata__", {})
    return {"path": str(path), "present": True, "tensor_count": len(header),
            "metadata": metadata,
            "dtype_counts": dict(Counter(v.get("dtype", "?") for v in header.values())),
            "tensor_names": sorted(header)}


def physical_config(candidate: str):
    return (QWEN, QWEN_DRAFTER) if candidate == "qwen" else (BONSAI, BONSAI_DRAFTER)


def chosen_drafter(candidate: str, profile: str):
    if candidate == "bonsai" and profile == "bonsai-specific":
        return BONSAI_SPECIFIC_DRAFTER
    if candidate == "bonsai" and profile == "production":
        return QWEN_DRAFTER
    return physical_config(candidate)[1]


def weight_inventory(paths: list[Path]) -> list[dict]:
    rows = []
    for p in paths:
        blob = p.resolve().name
        digest = blob if len(blob) == 64 and all(c in "0123456789abcdef" for c in blob) else None
        source = "HF content-addressed blob name" if digest else "not independently rehashed in model-free preflight"
        if p.resolve() == (BONSAI_DRAFTER / "model.safetensors").resolve():
            digest = "876c368b5abfdd5de52ab14fcd5d3cccb07f0059f3903c984e4f17dbcee8a552"
            source = "existing verified controls.md fingerprint"
        rows.append({"name": p.name, "physical_blob_identity": blob,
                     "fingerprint_sha256": digest, "fingerprint_source": source,
                     "bytes": p.stat().st_size})
    return rows


def preflight(args) -> dict:
    target, _default_drafter = physical_config(args.candidate)
    drafter = chosen_drafter(args.candidate, args.drafter_profile)
    candidate_target_configs = {}
    for name, candidate_target in (("qwen", QWEN), ("bonsai", BONSAI)):
        cfg_path = candidate_target / "config.json"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Missing {name} target config: {cfg_path}")
        candidate_target_configs[name] = json.loads(cfg_path.read_text())
    target_weights = sorted(target.glob("*.safetensors"))
    drafter_weights = sorted(drafter.glob("*.safetensors"))
    missing = [str(p) for p in (target / "config.json", target / "tokenizer_config.json",
                                drafter / "config.json") if not p.is_file()]
    if not target_weights and not (target / "model.safetensors.index.json").is_file():
        missing.append(str(target / "model.safetensors.index.json"))
    if not drafter_weights and not (drafter / "model.safetensors.index.json").is_file():
        missing.append(str(drafter / "model.safetensors.index.json"))
    if missing:
        raise FileNotFoundError("Required local physical files missing: " + ", ".join(missing))
    if args.kv != "plain":
        raise ValueError("T027 primary runner only supports plain target KV")
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists (pass --overwrite to replace): {output}")
    with tempfile.NamedTemporaryFile(prefix=".t027-writecheck-", dir=output.parent,
                                     delete=True) as probe:
        probe.write(b"t027 output-path preflight\n")
        probe.flush()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(target), local_files_only=True, trust_remote_code=False)
    prompt = PROMPTS[args.workload]
    rendered = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       add_generation_prompt=True, enable_thinking=False,
                                       tokenize=False)
    ids = normalize_ids(tok.apply_chat_template([{"role": "user", "content": prompt}],
                                                add_generation_prompt=True,
                                                enable_thinking=False, tokenize=True))
    if args.workload == "lru":
        expected = [248045, 846, 198, 7734, 264, 5492, 21408, 12654, 436, 34810,
                    6297, 440, 6813, 321, 913, 29642, 13, 248046, 198, 248045,
                    74455, 198, 248068, 271, 248069, 271]
        if ids != expected:
            raise ValueError(f"T026 LRU prompt IDs changed: {ids!r}")
    elif len(ids) != 90:
        raise ValueError(f"Resolver workload tokenization changed: expected 90 IDs, got {len(ids)}")
    dcfg = json.loads((drafter / "config.json").read_text())
    prod_path = QWEN_DRAFTER / "model.safetensors"
    side_path = BONSAI_DRAFTER / "model.safetensors"
    prod_header = safetensors_header(prod_path)
    side_header = safetensors_header(side_path)
    side_source = side_header.get("metadata", {}).get("source", "")
    same_source = Path(side_source).resolve() == QWEN_DRAFTER.resolve()
    relation = ("same source checkpoint but materially different runtime representation"
                if same_source and prod_header.get("dtype_counts") != side_header.get("dtype_counts")
                else "unresolved")
    repo_head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                               check=True, capture_output=True, text=True).stdout.strip()
    result = {
        "schema": "t027-product-ab/v1", "preflight": True,
        "weight_load_performed": False, "inference_performed": False,
        "benchmark_performed": False, "plain_kv_passed": True,
        "candidate": args.candidate, "drafter_profile": args.drafter_profile,
        "target": str(target.resolve()), "target_config_sha256": sha(target / "config.json"),
        "both_candidate_target_configs": {name: {"path": str((QWEN if name == "qwen" else BONSAI).resolve()),
          "config_sha256": sha((QWEN if name == "qwen" else BONSAI) / "config.json"),
          "model_type": cfg.get("model_type"), "architectures": cfg.get("architectures"),
          "max_position_embeddings": cfg.get("max_position_embeddings")}
          for name, cfg in candidate_target_configs.items()},
        "target_format": {k: json.loads((target / "config.json").read_text()).get(k)
                           for k in ("model_type", "architectures", "dtype", "quantization", "quantization_config")},
        "target_weight_representation": ("mlx-community 4-bit packed target weights" if args.candidate == "qwen"
                                         else "Prism Hadamard Bonsai2 ternary target pack (2-bit representation)"),
        "target_weights_inventory": weight_inventory(target_weights) or [{"index": (target / "model.safetensors.index.json").name}],
        "drafter": str(drafter.resolve()), "drafter_config_sha256": sha(drafter / "config.json"),
        "drafter_weights_inventory": weight_inventory(drafter_weights) or [{"index": (drafter / "model.safetensors.index.json").name}],
        "drafter_config": {k: dcfg.get(k) for k in ("architectures", "model_type", "dflash_config", "block_size", "num_hidden_layers", "hidden_size")},
        "drafter_relationship": {"classification": relation, "same_source_checkpoint": same_source,
          "production_config_equals_sidecar_config": json.loads((QWEN_DRAFTER / "config.json").read_text()) == json.loads((BONSAI_DRAFTER / "config.json").read_text()),
          "production_safetensors_header": prod_header, "chad_sidecar_safetensors_header": side_header,
          "interpretation": "The Chad sidecar embeds q4g64 quantization metadata and packed U32 weights; load_dflash on the production raw HF checkpoint applies q4g64 but its default predicate leaves candidate_selector modules unquantized. This is a credible representation confound."},
        "target_kv": {"requested": "plain", "bits": 0, "group_size": None}, "prefix_state": args.prefix,
        "context_window_configured": json.loads((target / "config.json").read_text()).get("max_position_embeddings"),
        "workload": args.workload, "prompt": prompt, "rendered_prompt": rendered,
        "prompt_ids": ids, "prompt_token_count": len(ids),
        "prompt_ids_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
        "sampling": {"temperature": 0, "top_p": 1, "top_k": 0, "thinking": "disabled", "max_tokens": args.max_tokens},
        "runtime_semantics": "mlx-dspark production Engine.load/Engine.generate path; no HTTP or multi-request batch scheduler overhead",
        "repository_fixture": {
            "git_head": repo_head,
            "resolver_source": str(ROOT / "src/mlx_dspark/load.py"),
            "resolver_source_sha256": sha(ROOT / "src/mlx_dspark/load.py"),
            "resolver_test": str(ROOT / "tests/test_resolve.py"),
            "resolver_test_sha256": sha(ROOT / "tests/test_resolve.py"),
            "test_command": TEST_COMMAND,
            "interpretation": "local test/task provenance only; this decode runner does not edit the repository, execute tools, run pytest, or grade task success",
        },
        "test_command": TEST_COMMAND if args.workload == "resolver" else None,
        "output_path": str(output.resolve()), "output_path_writable": True,
        "local_only": True,
    }
    return result


def describe_caches(cache) -> list[dict]:
    rows = []
    for index, item in enumerate(cache):
        bits = getattr(item, "bits", None)
        group_size = getattr(item, "group_size", None)
        cls = type(item).__name__
        is_attention = "KV" in cls or hasattr(item, "keys") or hasattr(item, "values")
        rows.append({"index": index, "class": cls, "attention_cache": bool(is_attention),
                     "bits": int(bits) if bits is not None else None,
                     "group_size": int(group_size) if group_size is not None else None})
    quantized = [r for r in rows if r["attention_cache"] and
                 ("Quantized" in r["class"] or r["bits"] is not None)]
    if quantized:
        raise RuntimeError(f"Plain-KV invariant violated by measured target cache: {quantized}")
    if not any(r["attention_cache"] for r in rows):
        raise RuntimeError(f"Could not observe an attention KV cache in actual request: {rows}")
    return rows


def run_physical(args, record: dict) -> dict:
    # Imports below initialize MLX and load weights; unreachable from --preflight.
    sys.path.insert(0, str(ROOT / "src"))
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    from mlx_dspark.server import Engine
    from mlx_dspark import generate as generate_module
    from mlx_dspark.generate import encode_messages

    target_path, _ = physical_config(args.candidate)
    drafter_path = chosen_drafter(args.candidate, args.drafter_profile)
    engine = Engine.load(mode="dflash", model=str(target_path), drafter=str(drafter_path),
        drafter_bits=4, max_draft_tokens=None, enable_thinking=False,
        prefix_cache=True, prefix_cache_dir=None, prefix_cache_max_ram_mb=0,
        prefix_cache_slots=2, prefix_cache_rungs=8192, kv_bits=None,
        context_window=None, warmup=True, memory_guard=True)
    prompt_ids = record["prompt_ids"]
    runtime_prompt_ids = encode_messages(
        engine.tokenizer, [{"role": "user", "content": record["prompt"]}],
        enable_thinking=False)
    if runtime_prompt_ids != prompt_ids:
        engine.close()
        raise RuntimeError("Loaded production tokenizer prompt IDs differ from model-free preflight")
    record["loaded_runtime_prompt_ids_match_preflight"] = True
    record["loaded_runtime_prompt_token_count"] = len(runtime_prompt_ids)
    acquired_cache_observations: list[dict] = []
    if engine.prefix is None:
        engine.close()
        raise RuntimeError("Production prefix cache failed to initialize")
    original_acquire = engine.prefix.acquire

    def observe_acquire(ids):
        cache, ctx, reused = original_acquire(ids)
        acquired_cache_observations.append({"cache_classes": describe_caches(cache),
                                            "reused_tokens": int(reused)})
        return cache, ctx, reused

    engine.prefix.acquire = observe_acquire
    seed_result = None
    prefix_hits_before = engine.prefix.hits
    try:
        if args.prefix == "reuse":
            # Seed through the real production Engine/PrefixCache with the exact same full
            # prompt. This is a repeated-request cache-reuse control, not a fake shortened
            # prompt and not a claim to model a multi-turn agent transcript.
            seed_result = engine.generate(prompt_ids, max_tokens=32, temperature=0.0,
                                          top_p=1.0, top_k=0, stop=None, seed=None)
        engine.rounds.reset()
        peak_reset = hasattr(mx.metal, "reset_peak_memory")
        if not peak_reset:
            raise RuntimeError("Installed MLX lacks a supported peak-memory reset; fail closed")
        baseline_active = int(mx.metal.get_active_memory())
        mx.metal.reset_peak_memory()
        request_start = time.perf_counter()
        result = engine.generate(prompt_ids, max_tokens=args.max_tokens, temperature=0.0,
                                 top_p=1.0, top_k=0, stop=None, seed=None)
        request_wall = time.perf_counter() - request_start
        speculative_peak = int(mx.metal.get_peak_memory())
        events = engine.rounds.snapshot()
        if args.prefix == "reuse" and result.reused_tokens <= 0:
            raise RuntimeError("Requested production prefix reuse but measured request reused zero tokens")
        if args.prefix == "empty" and result.reused_tokens != 0:
            raise RuntimeError(f"Fresh-prefix run unexpectedly reused {result.reused_tokens} tokens")
        if not acquired_cache_observations:
            raise RuntimeError("No actual target cache acquired by Engine.generate was observed")
        measured_cache = acquired_cache_observations[-1]
        actual_caches = measured_cache["cache_classes"]
        actual_kv_bits = engine.target.kv_bits
        if actual_kv_bits not in (None, 0):
            raise RuntimeError(f"Production Engine target requested quantized KV bits={actual_kv_bits}")
        # Reset independently for the serial control. Its peak is never attributed to spec decode.
        serial_baseline_active = int(mx.metal.get_active_memory())
        mx.metal.reset_peak_memory()
        serial = run_serial(stream_generate, make_sampler, engine.target.model,
                            engine.tokenizer, args, prompt_ids)
        serial_peak = int(mx.metal.get_peak_memory())
        prefix_info = {"enabled": engine.prefix is not None,
                       "hits_before": prefix_hits_before,
                       "hits_after": engine.prefix.hits,
                       "reused_tokens": result.reused_tokens,
                       "measured_acquired_cache_reused_tokens": measured_cache["reused_tokens"],
                       "state": args.prefix,
                       "mechanism": "production Engine.prefix.acquire/checkpoint/store path"}
        runtime_configuration = {
            "mode": engine.mode,
            "derived_adaptive_max_draft_tokens": engine.max_draft_tokens,
            "last_effective_request_cap": engine._last_cap,
            "depth_capper_active": engine._depth_capper is not None,
            "small_m_active": engine.small_m,
            "sdpa_split_active": engine.sdpa_split,
            "wide_gemm_configured": generate_module.WIDE_GEMM_MIN_ROWS is not None,
            "wide_gemm_min_rows": generate_module.WIDE_GEMM_MIN_ROWS,
            "wide_gemm_shapes": sorted(generate_module.WIDE_GEMM_SHAPES) if generate_module.WIDE_GEMM_SHAPES else None,
            "cpu_co_prefill": engine.cpu_split,
            "cpu_split_process_setting": generate_module.CPU_SPLIT,
            "cpu_split_fp32": getattr(__import__("mlx_dspark.wide_gemm", fromlist=["CPU_SPLIT_FP32"]), "CPU_SPLIT_FP32"),
            "prefix_cache_enabled": engine.prefix is not None,
            "prefix_cache_mode": (None if engine.prefix is None else
                                   "checkpoint" if engine.prefix.checkpoint_mode else "trim"),
            "prefix_cache_slots": engine.prefix_cache_slots,
            "prefix_cache_rungs": engine.prefix_cache_rungs,
            "memory_guard_active": engine.memory_guard is not None,
            "warmup_enabled": engine.warmup_enabled,
            "context_window": engine.context_window,
            "serving_path_difference": "Engine.load/Engine.generate production initialization and request path; bypasses HTTP/socket and BatchEngine queue/batching, which cannot batch a single isolated request",
        }
        record["runtime_configuration"] = runtime_configuration
    finally:
        engine.close()

    rounds = events
    proposed = sum(int(r.get("drafted", 0)) for r in rounds)
    accepted = sum(int(r.get("accepted", 0)) for r in rounds)
    committed = sum(int(r.get("committed", 0)) for r in rounds)
    widths = Counter(str(r.get("drafted", 0)) for r in rounds)
    sources = Counter(str(r.get("source", "unknown")) for r in rounds)
    serial_tps = serial["decode_tok_s"]
    record["measurements"] = {
        "prompt_tokens": len(prompt_ids), "generated_tokens": result.num_tokens,
        "request_boundary_wall_seconds": request_wall,
        "decode_wall_seconds": result.decode_seconds,
        "speculative_decode_tok_s": result.num_tokens / max(result.decode_seconds, 1e-9),
        "serial_target_tok_s": serial_tps,
        "speculative_speedup_over_serial": (result.num_tokens / max(result.decode_seconds, 1e-9) / serial_tps) if serial_tps else None,
        "target_forwards": result.target_forwards, "round_count": len(rounds), "rounds": rounds,
        "width_distribution": dict(widths), "source_distribution": dict(sources), "proposed_drafts": proposed,
        "accepted_drafts": accepted, "committed_tokens": committed,
        "accepted_over_proposed": accepted / proposed if proposed else None,
        "committed_tokens_per_target_forward": committed / result.target_forwards if result.target_forwards else None,
        "target_forwards_per_generated_token": result.target_forwards / result.num_tokens if result.num_tokens else None,
        "generated_tokens_per_target_forward": result.num_tokens / result.target_forwards if result.target_forwards else None,
        "prefill_seconds": result.prefill_seconds or None,
        "prefill_tok_s": len(prompt_ids) / result.prefill_seconds if result.prefill_seconds else None,
        "ttft_seconds": result.ttft_seconds or None,
        "speculative_peak_memory": {"peak_bytes": speculative_peak,
            "active_baseline_bytes": baseline_active,
            "peak_increment_over_baseline_bytes": max(0, speculative_peak - baseline_active),
            "scope": "since reset immediately before measured speculative Engine.generate; target/drafter weights and initialized Engine remain resident"},
        "serial_control": serial,
        "serial_peak_memory": {"peak_bytes": serial_peak, "active_baseline_bytes": serial_baseline_active,
            "scope": "separate reset before warm-target serial stream_generate; Engine and any prefix state remain resident"},
        "actual_target_kv_caches": actual_caches,
        "plain_target_kv_assertion": {"passed": True, "target.kv_bits": actual_kv_bits,
            "mechanism": "inspected exact cache returned by production PrefixCache.acquire for measured request; rejects QuantizedKVCache or attention cache with bits"},
        "prefix_cache": prefix_info,
        "target_verify_seconds_by_width": None,
        "drafter_proposal_seconds_by_width": None,
        "accumulated_verify_seconds": None, "accumulated_proposal_seconds": None,
        "residual_loop_overhead_seconds": None,
        "component_timing": {"verify_seconds": None, "proposal_seconds": None,
          "reason": "DFlash proposal/target verification return lazy MLX arrays; accurate isolated device timings require synchronization/materialization that would perturb each round. Host-call durations are not GPU execution time."},
        "unavailable_metrics": {"component_times": "not measured in primary pass: per-call synchronization would alter execution; no separate microtiming run implemented", "residual": "cannot isolate residual without component durations"},
        "primary_timing_policy": "request-boundary wall clock; no per-round synchronization",
        "startup_calibration": "actual Engine.load production defaults; any cache miss calibration and 12-token kernel warmup occur before request timing",
        "runtime_path": "mlx_dspark.server.Engine.load + Engine.generate; single request, no HTTP/batch-queue overhead",
        "runtime_configuration": runtime_configuration,
    }
    return record


def run_serial(stream_generate, make_sampler, target_model, tokenizer, args, prompt_ids):
    # Warm-target serial control using mlx-lm 0.31.3's explicit sampler callable.
    start = time.perf_counter()
    tokens = []
    sampler = make_sampler(temp=0.0, top_p=1.0, top_k=0)
    final_response = None
    for response in stream_generate(target_model, tokenizer, prompt_ids,
                                    max_tokens=args.max_tokens, sampler=sampler):
        tokens.append(int(response.token))
        final_response = response
    seconds = time.perf_counter() - start
    return {"label": "warm-target serial control; same loaded target after server warmup",
            "wall_seconds_including_prefill": seconds, "generated_tokens": len(tokens),
            "decode_tok_s": getattr(final_response, "generation_tps", None),
            "prompt_tok_s": getattr(final_response, "prompt_tps", None),
            "sampler": "make_sampler(temp=0.0, top_p=1.0, top_k=0)"}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", required=True, choices=("qwen", "bonsai"))
    p.add_argument("--target")
    p.add_argument("--drafter")
    p.add_argument("--prefix", choices=("empty", "reuse"), default="empty")
    p.add_argument("--workload", choices=tuple(PROMPTS), required=True)
    p.add_argument("--kv", choices=("plain",), default="plain")
    p.add_argument("--adaptive", action="store_true", required=True)
    p.add_argument("--drafter-profile", choices=("product", "production", "bonsai-specific"), default="product",
                   help="select the locked Bonsai-specific local drafter checkpoint")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--output", required=True)
    p.add_argument("--preflight", action="store_true", help="local files/tokenizer only; never imports MLX")
    p.add_argument("--overwrite", action="store_true")
    return p


def main():
    args = parser().parse_args()
    target, _ = physical_config(args.candidate)
    drafter = chosen_drafter(args.candidate, args.drafter_profile)
    if args.candidate == "qwen" and args.drafter_profile != "product":
        raise SystemExit("Qwen candidate is locked to its production generic drafter")
    if args.target and Path(args.target).resolve() != target.resolve():
        raise SystemExit("CLI target conflicts with locked physical candidate configuration")
    if args.drafter and Path(args.drafter).resolve() != drafter.resolve():
        raise SystemExit("CLI drafter conflicts with locked physical candidate configuration")
    rec = preflight(args)
    if not args.preflight:
        rec["preflight"] = False
        rec["weight_load_performed"] = True
        rec["inference_performed"] = True
        rec = run_physical(args, rec)
    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
