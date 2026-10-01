import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

root = Path("/Users/do/git/mlx-dspark")
sys.path.insert(0, str(root / "src"))
qwen = Path("/private/tmp/mlx-dspark-feat006-local-view/models/mlx-community/Qwen3.8-27B-4bit").resolve()
donor = Path("/private/tmp/mlx-dspark-feat006-local-view/models/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit").resolve()
dflash = Path("/Users/do/.cache/huggingface/hub/models--incoai--Qwen3.8-27B-DFlash2/snapshots/015e795645c74b1a0eeef3b570031fb62e769bc5").resolve()
expected = {
    qwen: "10c35caafbb80f7dc6a7a432cdd11af10a6d4818",
    donor: "fcba37d2117a7077eac6b613b2668d14d9779edd",
    dflash: "015e795645c74b1a0eeef3b570031fb62e769bc5",
}
for path, revision in expected.items():
    if revision not in str(path) or not path.is_dir() or not (path / "config.json").is_file():
        raise SystemExit(f"Required immutable local snapshot is unavailable or mismatched: {path}")
if os.environ.get("HF_HUB_OFFLINE") != "1":
    raise SystemExit("HF_HUB_OFFLINE=1 is required to prevent downloads")

from mlx_dspark.generate import encode_messages, greedy_generate
from mlx_dspark.hybrid_target import TargetCompositionRequest
from mlx_dspark.server import Engine

USER_TEXT = "Сумма чисел от 1 до 100"
MAX_TOKENS = 512

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def sha_ids(ids):
    return hashlib.sha256(canonical([int(x) for x in ids]).encode("utf-8")).hexdigest()

def repeated_ngram(ids, min_n=3, max_n=12, min_repeats=3):
    found = []
    for n in range(min_n, min(max_n, len(ids) // min_repeats) + 1):
        counts = Counter(tuple(ids[i:i+n]) for i in range(len(ids) - n + 1))
        for gram, count in counts.items():
            if count >= min_repeats:
                found.append({"n": n, "count": count, "token_ids": list(gram)})
    found.sort(key=lambda row: (-row["count"], -row["n"], row["token_ids"]))
    return {"detected": bool(found), "criterion": "same contiguous token n-gram (3..12 tokens) occurs at least 3 times", "examples": found[:5]}

def output_sha(ids):
    return hashlib.sha256(canonical([int(x) for x in ids]).encode("utf-8")).hexdigest()

composition = TargetCompositionRequest.create(
    donor_path=str(donor), donor_repo="prism-ml/Ternary-Bonsai-2-27B-mlx-2bit",
    donor_revision="fcba37d2117a7077eac6b613b2668d14d9779edd",
    qwen_repo="mlx-community/Qwen3.8-27B-4bit",
    qwen_revision="10c35caafbb80f7dc6a7a432cdd11af10a6d4818",
    donor_indices=range(56, 64))
engine = None
try:
    engine = Engine.load(
        mode="dflash", model=str(qwen), drafter=str(dflash), target_composition=composition,
        drafter_bits=4, max_draft_tokens="auto", reasoning_effort="xhigh",
        prefix_cache=False, kv_bits=None, context_window=None, warmup=True,
        memory_guard=False, lookup_drafts=False, small_m=None, sdpa_split=None,
        wide_gemm_min=None, cpu_split=None)
    vocab = engine.reasoning_effort_vocab
    if not engine.supports_reasoning_effort or not vocab or "xhigh" not in vocab:
        raise RuntimeError("FAIL BEFORE REQUEST GENERATION: loaded chat template cannot faithfully represent reasoning_effort=xhigh")
    template_kwargs = dict(engine.template_defaults)
    template_kwargs["reasoning_effort"] = engine.map_reasoning_effort("xhigh")
    if template_kwargs["reasoning_effort"] != "xhigh":
        raise RuntimeError("FAIL BEFORE REQUEST GENERATION: production effort mapping changed xhigh")
    messages = [{"role": "user", "content": USER_TEXT}]
    rendered_ids = [int(x) for x in encode_messages(engine.tokenizer, messages, **template_kwargs)]
    if not rendered_ids:
        raise RuntimeError("Rendered prompt produced no input token IDs")
    rendered_prompt = engine.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **template_kwargs)
    if not isinstance(rendered_prompt, str):
        rendered_prompt = None

    controls = {"reasoning_effort_requested": "xhigh", "reasoning_effort_used": template_kwargs["reasoning_effort"],
                "supports_reasoning_effort": engine.supports_reasoning_effort,
                "accepted_efforts": sorted(vocab),
                "enable_thinking": template_kwargs.get("enable_thinking", "template default"),
                "temperature": 0.0, "top_p": 1.0, "top_k": 0,
                "max_new_tokens": MAX_TOKENS, "stop": [], "seed": 0,
                "presence_penalty": 0.0, "frequency_penalty": 0.0,
                "kv": "plain", "prefix_cache": False,
                "one_user_message": USER_TEXT}
    prompt_hash = sha_ids(rendered_ids)

    # This is the accepted Series-A serial measurement seam verbatim in shape: direct
    # greedy_generate on Engine's single MLX executor, with speculation absent.
    serial = engine._executor.submit(
        greedy_generate, engine.target, engine.tokenizer, prompt_ids=rendered_ids,
        max_new_tokens=MAX_TOKENS, temperature=0.0, top_p=1.0, top_k=0,
        seed=0, stop=[], presence_penalty=0.0, frequency_penalty=0.0).result()
    # engine.generate dispatches through the ordinary production DFlash/CapController
    # path; prefix cache is disabled and this call creates fresh target/drafter caches.
    engine.rounds.reset()
    speculative = engine.generate(
        rendered_ids, max_tokens=MAX_TOKENS, temperature=0.0, top_p=1.0, top_k=0,
        stop=[], seed=0, presence_penalty=0.0, frequency_penalty=0.0)
    events = engine.rounds.snapshot()

    a = [int(x) for x in serial.token_ids]
    b = [int(x) for x in speculative.token_ids]
    common = 0
    while common < min(len(a), len(b)) and a[common] == b[common]:
        common += 1
    first = common if common < max(len(a), len(b)) else None
    start = max(0, (first or 0) - 12)
    end_a, end_b = min(len(a), (first or 0) + 13), min(len(b), (first or 0) + 13)
    if first is None:
        window = {"start_index": None, "serial_text": "", "dflash_text": ""}
    else:
        window = {"start_index": start,
                  "serial_text": engine.tokenizer.decode(a[start:end_a]),
                  "dflash_text": engine.tokenizer.decode(b[start:end_b])}
    serial_text = serial.text
    dflash_text = speculative.text
    (Path(os.environ["H3_DIAG_WORK"]) / "serial.txt").write_text(serial_text, encoding="utf-8")
    (Path(os.environ["H3_DIAG_WORK"]) / "dflash.txt").write_text(dflash_text, encoding="utf-8")
    manifest = engine.target.composition_manifest.as_dict()
    expected_blocks = list(range(56, 64))
    donor_owned = [row["index"] for row in manifest["block_owners"] if row["owner"] == "bonsai"]
    qwen_nonblocks = {key: manifest[key] for key in ("embedding_owner", "final_norm_owner", "lm_head_owner")}
    if donor_owned != expected_blocks or any(owner != "qwen" for owner in qwen_nonblocks.values()):
        raise RuntimeError(f"Loaded composition manifest does not prove requested H3 ownership: {manifest}")
    cap_counts = Counter(str(row.get("cap")) for row in events)
    width_counts = Counter(str(row.get("drafted")) for row in events)
    accepted = sum(int(row.get("accepted", 0)) for row in events)
    drafted = sum(int(row.get("drafted", 0)) for row in events)
    result = {
        "schema": "h3-serial-vs-dflash-oneoff/v1", "status": "PASS",
        "prompt": {"user_message": USER_TEXT, "rendered_prompt": rendered_prompt,
                   "input_token_ids": rendered_ids, "input_token_ids_sha256": prompt_hash,
                   "input_token_count": len(rendered_ids), "controls": controls},
        "composition_manifest": manifest,
        "composition_assertions": {"donor_owned_blocks": donor_owned,
                                    "expected_donor_owned_blocks": expected_blocks,
                                    "qwen_nonblock_owners": qwen_nonblocks},
        "serial": {"generated_token_ids": a, "generated_token_count": len(a),
                   "output_sha256": output_sha(a), "decoded_text": serial_text,
                   "finish_reason": serial.finish_reason, "seconds": serial.seconds,
                   "decode_seconds": serial.decode_seconds,
                   "decode_tokens_per_sec": serial.decode_tokens_per_sec,
                   "target_forwards": serial.target_forwards,
                   "repeated_ngram": repeated_ngram(a)},
        "dflash": {"generated_token_ids": b, "generated_token_count": len(b),
                   "output_sha256": output_sha(b), "decoded_text": dflash_text,
                   "finish_reason": speculative.finish_reason, "seconds": speculative.seconds,
                   "decode_seconds": speculative.decode_seconds,
                   "decode_tokens_per_sec": speculative.decode_tokens_per_sec,
                   "acceptance": accepted / max(drafted, 1),
                   "mean_accepted_length": speculative.mean_accept_len,
                   "target_forwards": speculative.target_forwards,
                   "generated_tokens_per_target_forward": len(b) / max(speculative.target_forwards, 1),
                   "rounds": speculative.num_rounds,
                   "width_distribution": dict(sorted(width_counts.items())),
                   "cap_distribution": dict(sorted(cap_counts.items())),
                   "round_events": events, "cap_controller": engine.cap_controller.info(),
                   "repeated_ngram": repeated_ngram(b)},
        "comparison": {"serial_output_sha256": output_sha(a),
                       "dflash_output_sha256": output_sha(b),
                       "exact_token_equality": a == b,
                       "common_prefix_generated_tokens": common,
                       "first_differing_generated_token_index": first,
                       "serial_token_at_first_difference": a[first] if first is not None and first < len(a) else None,
                       "dflash_token_at_first_difference": b[first] if first is not None and first < len(b) else None,
                       "decoded_window_around_first_difference": window},
    }
    out = Path(os.environ["H3_DIAG_WORK"]) / "result.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("SERIAL_SHA=" + result["serial"]["output_sha256"])
    print("DFLASH_SHA=" + result["dflash"]["output_sha256"])
    print("SAME_OUTPUT=" + str(result["comparison"]["exact_token_equality"]).lower())
    print("COMMON_PREFIX_TOKENS=" + str(common))
    print("FIRST_DIFFERENCE=" + (str(first) if first is not None else "null"))
    print("SERIAL_FIRST_500=" + serial_text[:500].replace("\n", "\\n"))
    print("DFLASH_FIRST_500=" + dflash_text[:500].replace("\n", "\\n"))
finally:
    if engine is not None:
        engine.close()
