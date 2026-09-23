"""Process-local Bonsai boundary recorder. No repository source monkeypatch persists.

Run with the matching runtime's Python and --runtime mlx-dspark or chad. Hooks
return the original values and snapshot only the first three logical rounds.
The diagnostic one-behind mlx-dspark prefill is opt-in via --align-prefill.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

TARGET_ID = "nathansutton/Qwen3.8-27B-Ternary-Bonsai-2-DFlash2-MLX"
TARGET = Path("/Users/do/.cache/huggingface/hub/models--nathansutton--Qwen3.8-27B-Ternary-Bonsai-2-DFlash2-MLX/snapshots/234cc925d77cd94683e47b1493bd964917aaf43b")
SIDECAR = Path("/Users/do/.cache/chad/dflash/015e795645c74b1a0eeef3b570031fb62e769bc5-q4g64")
MESSAGE = [{"role": "user", "content": "Write a production-quality Python LRU cache with tests and type hints."}]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def array_record(value):
    import mlx.core as mx
    import numpy as np

    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [array_record(item) for item in value]
    if not isinstance(value, mx.array):
        return value if isinstance(value, (int, float, str, bool)) else repr(value)
    # Copy/materialize now: MLX is lazy and a later cache rewrite must not
    # change the prior boundary. Float32 preserves bf16/fp16 values exactly.
    mx.eval(value)
    copy = np.asarray(value.astype(mx.float32)).copy()
    return {
        "shape": list(value.shape), "dtype": str(value.dtype),
        "float32_sha256": hashlib.sha256(copy.tobytes()).hexdigest(),
        "first_values": copy.reshape(-1)[:8].tolist(),
    }


def cache_record(cache):
    if cache is None:
        return None
    out = []
    for index, c in enumerate(cache):
        row = {"layer": index, "class": type(c).__module__ + "." + type(c).__name__}
        for name in ("offset", "lengths", "left_padding", "step"):
            if hasattr(c, name):
                row[name] = array_record(getattr(c, name))
        if hasattr(c, "keys"):
            row["keys"] = array_record(c.keys)
            row["values"] = array_record(c.values)
        elif hasattr(c, "cache"):
            row["slots"] = array_record(c.cache)
        else:
            try:
                row["slots"] = array_record(list(c))
            except TypeError:
                row["opaque"] = True
        out.append(row)
    return out


def logits_record(logits, fused):
    import mlx.core as mx
    import numpy as np

    mx.eval(logits)
    rows = np.asarray(logits[0].astype(mx.float32)).copy()
    top = np.argpartition(rows, -2, axis=-1)[:, -2:]
    top = np.take_along_axis(top, np.argsort(-np.take_along_axis(rows, top, axis=-1), axis=-1), axis=-1)
    vals = np.take_along_axis(rows, top, axis=-1)
    return {
        "logits": array_record(logits),
        "top2_ids": top.tolist(), "top2_values": vals.tolist(),
        "top2_margins": (vals[:, 0] - vals[:, 1]).tolist(),
        "fused": array_record(fused),
    }


def mlx_gdn_record(stash):
    if stash is None:
        return None
    delta, conv = stash
    names = ("q", "k", "v", "a", "b", "A_log", "dt_bias", "pre_state", "mask", "use_kernel")
    return [{**{"gdn_order": i}, **{name: array_record(value) for name, value in zip(names, args)},
             "conv_input": array_record(conv[i]) if i < len(conv) else None}
            for i, args in enumerate(delta)]


def chad_gdn_record(coll):
    if coll is None:
        return None
    names = ("q", "k", "v", "a", "b", "A_log", "dt_bias", "pre_state", "use_kernel")
    return [{**{"gdn_order": i}, **{name: array_record(value) for name, value in zip(names, args)},
             "mask": None, "conv_input": array_record(coll["conv"][i]) if i < len(coll["conv"]) else None}
            for i, args in enumerate(coll["args"])]


def common_record(runtime, ids, align_prefill):
    import importlib.metadata as md
    import mlx.core as mx

    return {
        "implementation": runtime,
        "target_id": TARGET_ID, "target_path": str(TARGET),
        "target_fingerprint": sha256(TARGET / "model.safetensors"),
        "target_config_fingerprint": sha256(TARGET / "config.json"),
        "drafter_id": str(SIDECAR),
        "drafter_fingerprint": sha256(SIDECAR / "model.safetensors"),
        "tokenizer_fingerprint": sha256(TARGET / "tokenizer.json"),
        "tokenizer_config_fingerprint": sha256(TARGET / "tokenizer_config.json"),
        "prompt_ids": list(ids), "messages": MESSAGE,
        "prefix_state_id": hashlib.sha256(json.dumps(list(ids[:-1]), separators=(",", ":")).encode()).hexdigest(),
        "chat_template": {"add_generation_prompt": True, "enable_thinking": False},
        "sampling": {"temperature": 0, "top_p": 0, "top_k": 0, "mode": "greedy"},
        "max_tokens": 512,
        "tap_ids": [5, 19, 33, 47, 61],
        "python": sys.version, "mlx": md.version("mlx"), "mlx_lm": md.version("mlx-lm"),
        "os": platform.platform(), "device": str(mx.default_device()),
        "model_type": "prism_hadamard_qwen35",
        "align_prefill_diagnostic": align_prefill,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "rounds": [],
        "gate_results": [{"gate": gate, "result": "inconclusive",
                          "missing_control": "independent same-runtime references and five-run numeric controls pending"}
                         for gate in ("A", "B1", "B2", "C", "D", "E")],
    }


def run_mlx(args):
    os.environ["MLX_DSPARK_DFLASH_CHAD_PREFILL"] = "1" if args.align_prefill else "0"
    sys.path.insert(0, "/Users/do/git/mlx-dspark/src")
    import mlx.core as mx
    from mlx_dspark.load import load_dflash, load_target
    from mlx_dspark.generate import dflash_generate
    from mlx_dspark import mlx_qmm_mma

    target, tokenizer = load_target(str(TARGET), require_tap=True, kv_bits=args.mlx_kv_bits)
    drafter, cfg = load_dflash(str(SIDECAR))
    ids = tokenizer.apply_chat_template(MESSAGE, tokenize=True, add_generation_prompt=True,
                                         enable_thinking=False)
    doc = common_record("mlx-dspark", ids, args.align_prefill)
    doc["target_loader"] = "mlx_dspark.prism_pack.load"
    doc["kernel_flags"] = {key: val for key, val in os.environ.items() if key.startswith("MLX_DSPARK_")}
    doc["prism_mma_active"] = mlx_qmm_mma.active()
    doc["kv_bits"] = args.mlx_kv_bits
    doc["sidecar_block_size"] = cfg.block_size
    cache = target.make_cache()
    rounds = doc["rounds"]
    generated = 0

    orig_select = drafter.select_block
    def select(block, pending_ctx, dcache, *a, **kw):
        nonlocal generated
        observed = len(rounds) < 3
        if observed:
            row = {
                "observation_id": f"mlx-r{len(rounds)+1}",
                "generated_before": generated,
                "anchor_or_pending_id": int(kw["anchor_id"]),
                "target_pre_proposal": cache_record(cache),
                "drafter_pre_proposal": cache_record(dcache),
                "pending_context": array_record(pending_ctx),
                "proposal_context_logical_position": generated,
            }
        result = orig_select(block, pending_ctx, dcache, *a, **kw)
        if observed:
            mx.eval(result[0])
            row["draft_ids"] = result[0].tolist()
            row["draft_width"] = len(row["draft_ids"])
            rounds.append(row)
        return result
    drafter.select_block = select

    orig_verify = target.verify
    def verify(verify_ids, round_cache, tap):
        observed = len(rounds) <= 3 and rounds and "verify_ids" not in rounds[-1]
        row = rounds[-1] if observed else None
        if observed:
            mx.eval(verify_ids)
            row["verify_ids"] = verify_ids[0].tolist()
            row["target_sequence_width"] = len(row["verify_ids"])
            row["pre_verify"] = cache_record(round_cache)
        result = orig_verify(verify_ids, round_cache, tap)
        if observed:
            row["verify"] = logits_record(*result)
            row["post_verify"] = cache_record(round_cache)
            row["gdn_capture"] = mlx_gdn_record(target._stash)
        return result
    target.verify = verify

    orig_rollback = target.rollback
    def rollback(round_cache, n_rejected, accepted):
        row = rounds[-1] if rounds and "post_reconcile" not in rounds[-1] else None
        result = orig_rollback(round_cache, n_rejected, accepted)
        if row is not None:
            row["n_rejected"] = n_rejected
            row["accepted_prefix_ids"] = row["verify_ids"][:len(accepted)+1]
            row["post_reconcile"] = cache_record(round_cache)
        return result
    target.rollback = rollback

    def on_round(**event):
        nonlocal generated
        if rounds and "accepted_count" not in rounds[-1]:
            rounds[-1]["accepted_count"] = event["accepted"]
            rounds[-1]["committed_count"] = event["committed"]
        generated += event["committed"]

    result = dflash_generate(
        target, tokenizer, drafter, prompt_ids=list(ids), cache=cache,
        ctx_caches=None, max_new_tokens=args.max_tokens,
        max_draft_tokens=7, lookup_drafts=False, cap_controller=None,
        temperature=0.0, top_p=0.0, top_k=0, on_round=on_round,
    )
    doc["output_ids"] = result.token_ids
    doc["num_rounds"] = result.num_rounds
    doc["finish_reason"] = result.finish_reason
    return doc


def run_chad(args):
    os.environ["CHAD_DFLASH_PATH"] = str(SIDECAR)
    if args.chad_no_fastpath:
        os.environ["CHAD_NO_FASTPATH"] = "1"
    sys.path.insert(0, "/Users/do/git/chad/src")
    import mlx.core as mx
    from chad.engine import Engine
    from chad import engine as engine_module, mlx_dflash, mlx_fastpath
    from mlx_lm.models import qwen3_5

    eng = Engine(model_id=str(TARGET), cache_dir=None, dflash_adaptive=True,
                 temp=0.0, top_p=0.0, top_k=0, kv_bits=args.chad_kv_bits)
    eng.load()
    if eng._dflash is None:
        raise RuntimeError("Chad did not load the exact physical DFlash sidecar")
    ids = list(eng.tok.apply_chat_template(MESSAGE, tokenize=True,
                                            add_generation_prompt=True, enable_thinking=False))
    doc = common_record("chad", ids, False)
    doc["target_loader"] = "chad.prism_pack.load via Engine.load"
    doc["chad_fastpath"] = eng.fastpath
    doc["kv_bits"] = eng.kv_bits
    doc["CHAD_NO_FASTPATH"] = os.environ.get("CHAD_NO_FASTPATH")
    doc["gdn_call_file"] = qwen3_5.GatedDeltaNet.__call__.__code__.co_filename
    doc["gdn_call_line"] = qwen3_5.GatedDeltaNet.__call__.__code__.co_firstlineno
    doc["kernel_flags"] = {key: val for key, val in os.environ.items() if key.startswith("CHAD_")}
    rounds = doc["rounds"]
    generated = 0

    orig_propose = engine_module._DFlashDrafter.propose
    def propose(self, k, anchor, rng):
        nonlocal generated
        observed = len(rounds) < 3
        if observed:
            row = {
                "observation_id": f"chad-r{len(rounds)+1}",
                "generated_before": generated,
                "anchor_or_pending_id": int(anchor),
                "target_pre_proposal": cache_record(self.eng._cache),
                "drafter_pre_proposal": cache_record(self.cache),
                "pending_context": array_record(self.pending),
                "proposal_context_logical_position": generated,
            }
        result = orig_propose(self, k, anchor, rng)
        if observed:
            mx.eval(result[0])
            row["draft_ids"] = [int(x) for x in result[0]]
            row["draft_width"] = k
            rounds.append(row)
        return result
    engine_module._DFlashDrafter.propose = propose

    model_class = type(eng.model.language_model.model)
    orig_model_call = model_class.__call__
    def model_call(self, input_ids, *a, **kw):
        observed = rounds and len(rounds) <= 3 and "verify_ids" not in rounds[-1]
        row = rounds[-1] if observed else None
        if observed:
            mx.eval(input_ids)
            row["verify_ids"] = input_ids[0].tolist()
            row["target_sequence_width"] = len(row["verify_ids"])
            row["pre_verify"] = cache_record(kw.get("cache"))
        hid = orig_model_call(self, input_ids, *a, **kw)
        if observed:
            fused = mx.concatenate([mlx_dflash.TAP[i] for i in eng._dflash.config.target_layer_ids], axis=-1)
            logits = eng.model.language_model.lm_head(hid)
            row["verify"] = logits_record(logits, fused)
            row["post_verify"] = cache_record(kw.get("cache"))
            row["gdn_capture"] = chad_gdn_record(mlx_fastpath.GDN_COLLECTOR)
        return hid
    model_class.__call__ = model_call

    orig_reconcile = engine_module._DFlashDrafter.reconcile
    def reconcile(self, k, accepted, anchor, draft, hid, fused):
        nonlocal generated
        result = orig_reconcile(self, k, accepted, anchor, draft, hid, fused)
        if rounds and "accepted_count" not in rounds[-1]:
            row = rounds[-1]
            row["accepted_count"] = accepted
            row["committed_count"] = accepted + 1
            row["n_rejected"] = k - accepted
            row["accepted_prefix_ids"] = row["verify_ids"][:accepted+1]
            row["post_reconcile"] = cache_record(self.eng._cache)
            row["accepted_fused"] = array_record(fused[:, :accepted+1])
        generated += accepted + 1
        return result
    engine_module._DFlashDrafter.reconcile = reconcile

    try:
        _, stats = eng.generate(ids, args.max_tokens, None, [])
    finally:
        engine_module._DFlashDrafter.propose = orig_propose
        engine_module._DFlashDrafter.reconcile = orig_reconcile
        model_class.__call__ = orig_model_call
    doc["output_ids"] = list(stats.gen_ids or [])
    doc["num_rounds"] = stats.forwards
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", choices=("mlx-dspark", "chad"), required=True)
    ap.add_argument("--output-dir", default="/private/tmp/bonsai-rollback-observations")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--align-prefill", action="store_true")
    ap.add_argument("--chad-kv-bits", type=int, default=None)
    ap.add_argument("--chad-no-fastpath", action="store_true")
    ap.add_argument("--mlx-kv-bits", type=int, default=None)
    args = ap.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    doc = run_mlx(args) if args.runtime == "mlx-dspark" else run_chad(args)
    path = out / (args.runtime + ("-aligned" if args.align_prefill else "") + ".json")
    path.write_text(json.dumps(doc, indent=2, sort_keys=True))
    print(json.dumps({"path": str(path), "prompt_ids": doc["prompt_ids"],
                      "rounds": [{k: r.get(k) for k in ("observation_id", "anchor_or_pending_id", "draft_ids", "verify_ids", "accepted_count")}
                                 for r in doc["rounds"]],
                      "output_ids": doc["output_ids"][:16]}, indent=2))


if __name__ == "__main__":
    main()
