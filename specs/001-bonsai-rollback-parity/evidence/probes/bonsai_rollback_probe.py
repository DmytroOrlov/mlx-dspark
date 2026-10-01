"""Process-local Bonsai boundary recorder. No repository source monkeypatch persists.

Run with the matching runtime's Python and --runtime mlx-dspark or chad. Hooks
return the original values and snapshot only the first three logical rounds.
The diagnostic one-behind mlx-dspark prefill is opt-in via --align-prefill.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from bonsai_probe_support import (
    EvidenceStore, branch_identity, cache_semantic_signature, clone_cache, compare_r1_context_reference,
    evidence_content_view,
    control_plan, freeze_record,
    measure_control_matrix, measure_r1_context_controls, mutable_cache_object_ids, stable_id,
    required_gate_branch_provenance, link_shared_b2_control_set,
    load_array_record,
    required_gate_acquisition_record,
    load_mx_array_record,
    transient_pair_controls, compare_controlled_vectors,
    assert_first_round_e_reconstructable,
    e_source_proposal_adjudication,
)

TARGET_ID = "nathansutton/Qwen3.8-27B-Ternary-Bonsai-2-DFlash2-MLX"
TARGET = Path("/Users/do/.cache/huggingface/hub/models--nathansutton--Qwen3.8-27B-Ternary-Bonsai-2-DFlash2-MLX/snapshots/234cc925d77cd94683e47b1493bd964917aaf43b")
SIDECAR = Path("/Users/do/.cache/chad/dflash/015e795645c74b1a0eeef3b570031fb62e769bc5-q4g64")
MESSAGE = [{"role": "user", "content": "Write a production-quality Python LRU cache with tests and type hints."}]
_STORE: EvidenceStore | None = None
_LAYER_MAP: list[dict] = []


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def array_record(value):
    if _STORE is None:
        raise RuntimeError("EvidenceStore must be set before any model capture")
    return _STORE.capture(value)


def mlx_numpy_value(value):
    """Convert an evaluated MLX value to NumPy without exposing bfloat16 via PEP 3118."""
    import mlx.core as mx
    import numpy as np

    mx.eval(value)
    converted = value.astype(mx.float32) if isinstance(value, mx.array) and value.dtype == mx.bfloat16 else value
    if converted is not value:
        mx.eval(converted)
    return np.asarray(converted).copy()


def model_layer_map(model):
    layers = getattr(model, "layers", None)
    if layers is None:
        layers = model.language_model.model.layers
    rows = []
    for index, layer in enumerate(layers):
        linear = bool(getattr(layer, "is_linear", False))
        rows.append({"layer": index, "kind": "gdn" if linear else "attention",
                     "cache_slots": ["convolution_window", "recurrent_state"] if linear else ["keys", "values"],
                     "tap": index in (5, 19, 33, 47, 61)})
    return rows


def cache_record(cache, *, domain="target"):
    if cache is None:
        return None
    out = []
    for index, c in enumerate(cache):
        row = {"layer": index, "class": type(c).__module__ + "." + type(c).__name__,
               "semantic": (_LAYER_MAP[index] if index < len(_LAYER_MAP) else None)
                           if domain == "target" else {"layer": index, "kind": "drafter_context_kv"},
               "trimmable": bool(c.is_trimmable()) if hasattr(c, "is_trimmable") else hasattr(c, "trim")}
        for name in ("offset", "lengths", "left_padding", "step", "bits", "group_size", "keep", "max_size", "_idx"):
            if hasattr(c, name):
                row[name] = array_record(getattr(c, name))
        if hasattr(c, "keys"):
            keys, values = c.keys, c.values
            if keys is not None:
                if hasattr(c, "_temporal_order"):
                    keys, values = c._temporal_order(keys), c._temporal_order(values)
                    row["temporal_order_applied"] = True
                else:
                    # The state property trims preallocated backing tails. It can
                    # be unavailable for an empty cache, so inspect it only live.
                    try:
                        keys, values = c.state
                    except (AttributeError, TypeError, ValueError):
                        pass
            row["keys"] = array_record(keys)
            row["values"] = array_record(values)
            def live_len(value):
                if value is None:
                    return 0
                if isinstance(value, (tuple, list)):
                    return live_len(value[0])
                return int(value.shape[-2])
            length = live_len(keys)
            offset = getattr(c, "offset", None)
            row["live_length"] = length
            if isinstance(offset, int):
                keep = min(int(getattr(c, "keep", 0)), length)
                positions = ([*range(keep), *range(offset - (length - keep), offset)]
                             if keep and offset > length else list(range(offset - length, offset)))
                row["absolute_positions"] = positions
                row["absolute_range"] = ([positions[0], positions[-1] + 1]
                                         if positions and positions == list(range(positions[0], positions[-1] + 1))
                                         else [offset, offset] if not positions else None)
            else:
                row["absolute_positions"] = None
                row["absolute_range"] = None
        elif hasattr(c, "cache"):
            row["slots"] = array_record(c.cache)
            row["convolution_window"] = array_record(c.cache[0]) if len(c.cache) > 0 else None
            row["recurrent_state"] = array_record(c.cache[1]) if len(c.cache) > 1 else None
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
    return [{**{"gdn_order": i, "model_layer": next((m["layer"] for m in _LAYER_MAP if m["kind"] == "gdn" and sum(x["kind"] == "gdn" for x in _LAYER_MAP[:m["layer"]]) == i), None)},
             **{name: array_record(value) for name, value in zip(names, args)},
             "conv_input": array_record(conv[i]) if i < len(conv) else None,
             "conv_pre_window": array_record(conv[i][:, :conv[i].shape[1] - args[0].shape[1]]) if i < len(conv) else None}
            for i, args in enumerate(delta)]


def chad_gdn_record(coll):
    if coll is None:
        return None
    names = ("q", "k", "v", "a", "b", "A_log", "dt_bias", "pre_state", "use_kernel")
    return [{**{"gdn_order": i, "model_layer": next((m["layer"] for m in _LAYER_MAP if m["kind"] == "gdn" and sum(x["kind"] == "gdn" for x in _LAYER_MAP[:m["layer"]]) == i), None)},
             **{name: array_record(value) for name, value in zip(names, args)},
             "mask": None, "conv_input": array_record(coll["conv"][i]) if i < len(coll["conv"]) else None,
             "conv_pre_window": array_record(coll["conv"][i][:, :coll["conv"][i].shape[1] - args[0].shape[1]]) if i < len(coll["conv"]) else None}
            for i, args in enumerate(coll["args"])]


def semantic_positions(cache_rows, generated_before, prompt_count, pending_rows=None):
    attention = next((row for row in cache_rows or [] if row.get("absolute_range") is not None), None)
    return {
        "generated_before": generated_before,
        "prompt_length": prompt_count,
        "next_target_absolute_position": attention["absolute_range"][1] if attention else None,
        "target_live_range": attention["absolute_range"] if attention else None,
        "pending_context_rows": pending_rows,
        "proposal_context_logical_position": generated_before,
    }


def drafter_context_positions(cache_rows, pending_rows):
    first = next((row for row in cache_rows or [] if row.get("absolute_range") is not None), None)
    if first is None:
        return {"cache_live_rows": None, "cache_absolute_range": None,
                "pending_rows": pending_rows, "context_live_rows": None,
                "context_absolute_end": None}
    return {"cache_live_rows": first["live_length"],
            "cache_absolute_range": first["absolute_range"],
            "pending_rows": pending_rows,
            "context_live_rows": first["live_length"] + pending_rows,
            "context_absolute_end": first["absolute_range"][1] + pending_rows}


def boundary(row, label, cache, *, input_ids=None, extra=None, branch="speculative",
             branch_source=None):
    state = cache_record(cache)
    identity = branch_identity(row["run_id"], row["round_index"], branch, cache, state,
                               source=branch_source)
    record = {
        "observation_id": f'{row["observation_id"]}:{branch}:{label}',
        "round_index": row["round_index"], "boundary": label,
        "branch": identity, "input_ids": list(input_ids) if input_ids is not None else None,
        "position": semantic_positions(state, row["generated_before"], row["prompt_length"]),
        "cache": state, "tap_ids": [5, 19, 33, 47, 61],
        "layer_map": _LAYER_MAP,
        "extra": extra or {},
    }
    return freeze_record(record)


def freeze_s8_preverify(row, live_cache, frozen_roots):
    """Copy the live production boundary before S=8 verify can mutate it."""
    verify = list(row["verify_ids"])
    assert len(verify) == 8 and verify == [row["anchor_or_pending_id"], *row["draft_ids"]]
    assert row["round_index"] not in frozen_roots
    live = row["pre_verify"]
    frozen_cache = clone_cache(live_cache)
    frozen = boundary(row, "frozen_pre_verify", frozen_cache, input_ids=verify,
                      branch="s8_frozen", branch_source="exact production pre-verify clone")
    live_mutable_ids = mutable_cache_object_ids(live_cache)
    frozen_mutable_ids = mutable_cache_object_ids(frozen_cache)
    assert set(live_mutable_ids).isdisjoint(frozen_mutable_ids)
    assert live["branch"]["cache_state_id"] == frozen["branch"]["cache_state_id"]
    assert cache_semantic_signature(live["cache"]) == cache_semantic_signature(frozen["cache"])
    assert live["position"] == frozen["position"]
    assert live["input_ids"] == frozen["input_ids"] == verify
    row["s8_frozen_pre_verify"] = frozen
    row["s8_frozen_independence"] = {
        "live_mutable_object_ids": live_mutable_ids,
        "frozen_mutable_object_ids": frozen_mutable_ids,
        "disjoint": True, "exact_state_id_equal": True,
        "exact_structure_positions_ids_equal": True,
    }
    frozen_roots[row["round_index"]] = frozen_cache


def persist_round(row, stage):
    """Retain a completed observation before later model work can fail."""
    if _STORE is None:
        raise RuntimeError("No active project-local evidence store")
    directory = _STORE.run_dir / "rounds"
    directory.mkdir(exist_ok=True)
    path = directory / f'r{row["round_index"]}-{stage}.json'
    with path.open("x", encoding="utf-8") as handle:
        json.dump(row, handle, sort_keys=True, indent=2)
        handle.write("\n")


def freeze_proposal(row):
    expected = [row["anchor_or_pending_id"], *row["draft_ids"]]
    actual = row["verify_ids"]
    if actual != expected:
        raise AssertionError(f"verify_ids differ from anchor + draft_ids: {actual} != {expected}")
    if len(actual) != 8 or len(row["draft_ids"]) != 7:
        row["width8_control"] = "inconclusive: this round did not propose seven drafts"
    record = {
        "observation_id": f'{row["observation_id"]}:proposal-input',
        "target_input_state_id": stable_id(row["target_pre_proposal"]),
        "drafter_input_state_id": stable_id(row["drafter_pre_proposal"]),
        "generated_before": row["generated_before"],
        "anchor_or_pending_id": row["anchor_or_pending_id"],
        "drafter_block_ids": list(row["drafter_block_ids"]),
        "drafter_block_dtype": row["drafter_block_dtype"],
        "draft_ids": list(row["draft_ids"]), "verify_ids": list(actual),
        "proposal_context_logical_position": row["proposal_context_logical_position"],
        "proposal_context_rows": row["proposal_context_rows"],
        "proposal_context_absolute_end": row["proposal_context_absolute_end"],
        "committed_prefix_ids": list(row["committed_prefix_ids"]),
        "target_positions": row["target_positions"],
        "drafter_positions": row["drafter_positions"],
        "input_state_classification": "pending matched-runtime Gate A comparison",
    }
    row["proposal_input_record"] = freeze_record(record)
    row["proposal_input_sha256"] = stable_id(record)


def source_provenance():
    from mlx_lm.models import cache, gated_delta, qwen3_5

    files = {name: Path(inspect.getfile(module)) for name, module in
             (("qwen3_5.py", qwen3_5), ("gated_delta.py", gated_delta), ("cache.py", cache))}
    call = qwen3_5.GatedDeltaNet.__call__
    update = qwen3_5.gated_delta_update
    return {
        "mlx_lm_source_hashes": {name: sha256(path) for name, path in files.items()},
        "gdn_callable": {
            "module": call.__module__, "file": inspect.getfile(call),
            "line": call.__code__.co_firstlineno, "signature": str(inspect.signature(call)),
            "update_module": update.__module__, "update_file": inspect.getfile(update),
            "update_signature": str(inspect.signature(update)),
            "identity_result": update is gated_delta.gated_delta_update,
        },
    }


def pending_gates(runtime):
    detail = {
        "A": ("exact semantic/logical input and proposal IDs", "matched-precision second-runtime continuous trace", "stop_at_A"),
        "B1": ("exact verify IDs and equivalent pre-verify state", "independent S=8 branch identity/equivalence", "stop_before_B2"),
        "B2": ("same-runtime S=8 verify/capture", "five-run component/tap numeric controls", "stop_before_C"),
        "C": ("post-reconcile exact state and controlled numeric bounds", "S=1/S=2/S=8 five-run controls", "stop_before_D"),
        "D": ("subsequent committed target evaluation", "Gate C pass and S=1 committed reference", "stop_before_E"),
        "E": ("drafter context and next proposals", "Gate D pass and equivalent context state", "stop_before_T016"),
    }
    return [{"gate": gate, "runtime": runtime, "result": "inconclusive",
             "comparison_type": "exact" if gate in ("A", "B1") else "controlled_numeric_and_exact",
             "rule": rule, "observation_ids": [], "earliest_difference": None,
             "numeric_metric": "max_abs" if gate in ("B2", "C", "D", "E") else None,
             "numeric_bound": None,
             "bound_rationale": "No bound until exactly five matched controls calibrate and validate"
                                if gate in ("B2", "C", "D", "E") else None,
             "missing_control": missing, "classification": "not_evaluated",
             "routing": route, "decision": "stop"}
            for gate, (rule, missing, route) in detail.items()]


def historical_trace_measurement(doc):
    """Measure the supplied historical sequence without setting policy inputs."""
    rounds = doc["rounds"]
    def outgoing_margin(row):
        accepted = row.get("accepted_count")
        values = (row.get("verify") or {}).get("top2_margins") or []
        return float(values[accepted]) if accepted is not None and accepted < len(values) else None
    measured = {
        "r1": {"draft_width": rounds[0].get("draft_width"),
               "accepted": rounds[0].get("accepted_count")} if len(rounds) > 0 else None,
        "r2": {"draft_width": rounds[1].get("draft_width"),
               "accepted": rounds[1].get("accepted_count"),
               "incoming_margin": outgoing_margin(rounds[0])} if len(rounds) > 1 else None,
        "pre_r3": {"margin": outgoing_margin(rounds[1]),
                   "chosen_width": rounds[2].get("draft_width")} if len(rounds) > 2 else None,
    }
    expected = {"r1": {"draft_width": 7, "accepted": 0},
                "r2": {"draft_width": 7, "accepted": 1, "incoming_margin": 0.25},
                "pre_r3": {"margin": 0.25 if doc["implementation"] == "chad" else 0.125}}
    reproduces = (measured["r1"] is not None and measured["r2"] is not None and
                  measured["pre_r3"] is not None and
                  all(measured[stage][name] == value for stage, fields in expected.items()
                      for name, value in fields.items()))
    return {"historical_observation": expected, "measured": measured,
            "reproduces_observed_sequence": bool(reproduces),
            "note": "Observed from exact prompt; no accept count, margin, or width was forced"}


def chad_executed_layer_paths(model, cache, width):
    """Record the installed DecoderLayer branch conditions for this exact call."""
    paths = []
    for index, (layer, layer_cache) in enumerate(zip(model.layers, cache)):
        fast_mlp = getattr(layer, "_mlp_fast", None) is not None
        fast_gdn = getattr(layer, "_gdn_fast", None) is not None
        if width == 1 and fast_mlp:
            if layer.is_linear:
                compiled = (fast_gdn and layer_cache[0] is not None and
                            layer_cache[1] is not None and layer_cache.lengths is None)
                path = "compiled_s1_gdn" if compiled else "stock_decoder_layer"
            else:
                path = "compiled_s1_attention"
        else:
            path = "stock_decoder_layer"
        paths.append({"layer": index, "width": width, "path": path,
                      "fused_mlp_installed": hasattr(layer.mlp, "_fused_w"),
                      "fused_gdn_installed": bool(layer.is_linear and
                                                  hasattr(layer.linear_attn, "_fused_w")),
                      "attention_class": (type(layer.self_attn).__module__ + "." +
                                          type(layer.self_attn).__name__) if not layer.is_linear else None})
    return paths


def common_record(runtime, ids, align_prefill):
    import importlib.metadata as md
    import mlx.core as mx

    target_config = json.loads((TARGET / "config.json").read_text())
    sidecar_config = json.loads((SIDECAR / "config.json").read_text())

    return {
        "implementation": runtime, "reference_kind": "speculative",
        "target_id": TARGET_ID, "target_path": str(TARGET),
        "target_fingerprint": sha256(TARGET / "model.safetensors"),
        "target_config_fingerprint": sha256(TARGET / "config.json"),
        "drafter_id": str(SIDECAR),
        "drafter_fingerprint": sha256(SIDECAR / "model.safetensors"),
        "drafter_config_fingerprint": sha256(SIDECAR / "config.json"),
        "tokenizer_fingerprint": sha256(TARGET / "tokenizer.json"),
        "tokenizer_config_fingerprint": sha256(TARGET / "tokenizer_config.json"),
        "prompt_ids": list(ids), "messages": MESSAGE,
        "prefix_state_id": hashlib.sha256(json.dumps(list(ids[:-1]), separators=(",", ":")).encode()).hexdigest(),
        "prefix_token_id": hashlib.sha256(json.dumps(list(ids[:-1]), separators=(",", ":")).encode()).hexdigest(),
        "chat_template": {"add_generation_prompt": True, "enable_thinking": False},
        "sampling": {"temperature": 0, "top_p": 0, "top_k": 0,
                     "mode": "greedy", "seed": None},
        "max_tokens": None,
        "historical_benchmark_max_tokens": 512,
        "tap_ids": [5, 19, 33, 47, 61],
        "python": sys.version, "mlx": md.version("mlx"), "mlx_lm": md.version("mlx-lm"),
        "os": platform.platform(), "device": str(mx.default_device()),
        "model_type": target_config.get("model_type"),
        "target_model_type": target_config.get("model_type"),
        "target_model_config": target_config,
        "sidecar_model_config": sidecar_config,
        "runtime": {"python": sys.version, "mlx": md.version("mlx"),
                    "mlx_lm": md.version("mlx-lm"), "os": platform.platform(),
                    "device": str(mx.default_device())},
        "align_prefill_diagnostic": align_prefill,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "rounds": [],
        "gate_results": pending_gates(runtime),
        "control_plan": control_plan(runtime),
        **source_provenance(),
    }


def mlx_reference_branches(target, tokenizer, row, tap, frozen_s8_cache,
                           measure_controls=False, qmm_calls=None):
    """Keep the exact frozen S=8 root separate from fresh committed S=1/S=2."""
    import mlx.core as mx
    from mlx_dspark.target import Target

    ids = list(row["verify_ids"])
    references = {}
    reference_target = Target(target.model, tokenizer, kv_bits=target.kv_bits,
                              kv_group_size=target.kv_group_size)
    frozen = row["s8_frozen_pre_verify"]
    assert len(ids) == 8 and frozen["input_ids"] == ids
    assert frozen["branch"]["cache_state_id"] == row["pre_verify"]["branch"]["cache_state_id"]
    supplied_root_state = cache_record(frozen_s8_cache)
    assert stable_id(evidence_content_view(supplied_root_state)) == frozen["branch"]["cache_state_id"]
    assert set(mutable_cache_object_ids(frozen_s8_cache)).isdisjoint(
        row["pre_verify"]["branch"]["cache_object_ids"])
    # The S=8 cache was cloned inside the production verify hook, before that
    # forward touched any mutable state. Only this exact root may feed B1/B2.
    replay_cache = reference_target.make_cache()
    # T006: a separate ordinary committed-token reconstruction. Its numeric
    # state may legitimately differ from a speculative post-reconcile cache.
    reference_target.prefill(mx.array([row["initial_prefill_ids"]]), replay_cache,
                             tap, want_logits=False)
    for token in row["previous_committed_ids"]:
        reference_target.run(mx.array([[token]]), replay_cache, tap)
    replay = boundary(row, "fresh_committed_replay", replay_cache,
                      input_ids=row["committed_prefix_ids"],
                      branch="fresh_replay", branch_source="empty cache + committed-token replay")
    references["fresh_replay"] = replay
    references["committed_replay_structure_equal_to_speculative"] = (
        cache_semantic_signature(replay["cache"]) ==
        cache_semantic_signature(row["pre_verify"]["cache"]))
    references["committed_replay_numeric_state_id_equal_to_speculative"] = (
        replay["branch"]["cache_state_id"] ==
        row["pre_verify"]["branch"]["cache_state_id"])

    def execute(kind, pieces, *, root, root_name, clone_root):
        before_dispatch = dict(qmm_calls or {})
        branch_cache = clone_cache(root) if clone_root else root
        same_width = root_name == "exact_frozen_speculative_preverify"
        expected = frozen if same_width else replay
        if same_width:
            assert pieces == [ids]
        pre = boundary(row, "pre_verify", branch_cache,
                       input_ids=[x for p in pieces for x in p],
                       branch=kind, branch_source=root_name)
        assert set(pre["branch"]["cache_object_ids"]).isdisjoint(
            row["pre_verify"]["branch"]["cache_object_ids"])
        assert pre["branch"]["cache_state_id"] == expected["branch"]["cache_state_id"]
        assert cache_semantic_signature(pre["cache"]) == cache_semantic_signature(expected["cache"])
        assert pre["position"] == expected["position"]
        outputs = []
        for piece in pieces:
            x = mx.array([piece])
            logits, fused = (reference_target.verify(x, branch_cache, tap) if same_width
                             else reference_target.run(x, branch_cache, tap))
            outputs.append(logits_record(logits, fused))
        result = {"implementation": "mlx-dspark", "reference_kind": kind,
                  "reference_root": root_name,
                  "root_observation_id": expected["observation_id"],
                  "target_sequence_widths": [len(piece) for piece in pieces],
                  "pre_verify": pre, "outputs": outputs,
                  "post_execute": boundary(row, "post_execute", branch_cache,
                                           input_ids=[x for p in pieces for x in p],
                                           branch=kind)}
        if qmm_calls is not None:
            result["executed_dispatch"] = {key: count - before_dispatch.get(key, 0)
                                           for key, count in qmm_calls.items()
                                           if count > before_dispatch.get(key, 0)}
        if same_width:
            result["gdn_capture"] = mlx_gdn_record(reference_target._stash)
        return result

    if measure_controls:
        def make_pair(role, repetition, side):
            actual = "s2_keep2" if role == "s2_vs_s1_keep2" and side == "left" else (
                     "s1_keep2" if role == "s2_vs_s1_keep2" else role)
            pieces = {
                "s8_verify": [ids], "s1_keep1": [[ids[0]]],
                "s1_keep2": [[ids[0]], [ids[1]]], "s2_keep2": [[ids[0], ids[1]]],
            }[actual]
            same_width = actual == "s8_verify"
            return execute(f"{role}_control_r{repetition}_{side}", pieces,
                           root=frozen_s8_cache if same_width else replay_cache,
                           root_name="exact_frozen_speculative_preverify" if same_width
                                     else "fresh_committed_replay", clone_root=True)
        root_ids = {role: (frozen if role == "s8_verify" else replay)["branch"]["cache_state_id"]
                    for role in ("s8_verify", "s1_keep1", "s1_keep2", "s2_keep2",
                                 "s2_vs_s1_keep2")}
        references["five_run_controls"] = measure_control_matrix(
            "mlx-dspark", _STORE.run_dir, root_ids, make_pair, _STORE)
    references["s8_verify"] = execute("s8_verify_reference", [ids],
                                      root=frozen_s8_cache,
                                      root_name="exact_frozen_speculative_preverify",
                                      clone_root=False)
    references["s1_keep1"] = execute("s1_keep1_reference", [[ids[0]]],
                                     root=replay_cache, root_name="fresh_committed_replay",
                                     clone_root=True)
    references["s1_keep2"] = execute("s1_keep2_reference", [[ids[0]], [ids[1]]],
                                     root=replay_cache, root_name="fresh_committed_replay",
                                     clone_root=True)
    references["s2_keep2"] = execute("s2_keep2_reference", [[ids[0], ids[1]]],
                                     root=replay_cache, root_name="fresh_committed_replay",
                                     clone_root=True)
    return references


def apply_mlx_same_s8_oracle(target, cache, capture, keep):
    """Install accepted S=8 state without invoking Target.rollback()."""
    import mlx.core as mx
    from mlx_lm.models.gated_delta import gated_delta_update

    delta, conv = capture
    linear_index = 0
    rejected = 8 - keep
    for item in cache:
        if target._is_trimmable(item):
            if rejected:
                item.trim(rejected)
            continue
        q, k, v, a, b, A_log, dt_bias, pre_state, mask, use_kernel = delta[linear_index]
        conv_input = conv[linear_index]
        conv_keep = conv_input.shape[1] - 8
        item[0] = mx.contiguous(conv_input[:, keep:keep + conv_keep, :])
        use_mask = mask[:, :keep] if mask is not None else None
        _, recurrent = gated_delta_update(q[:, :keep], k[:, :keep], v[:, :keep],
                                          a[:, :keep], b[:, :keep], A_log,
                                          dt_bias, pre_state, use_mask,
                                          use_kernel=use_kernel)
        item[1] = recurrent
        linear_index += 1
    if linear_index != len(delta):
        raise AssertionError(f"same-S8 oracle used {linear_index} GDN captures for {len(delta)} layers")


def apply_chad_same_s8_oracle(cache, collector, keep):
    """Install accepted S=8 state from Chad's pass-through collector, independently."""
    import mlx.core as mx
    from mlx_lm.models.gated_delta import gated_delta_update

    arrays = [item for item in cache if hasattr(item, "cache")]
    if len(arrays) != len(collector["args"]) or len(arrays) != len(collector["conv"]):
        raise AssertionError("Chad GDN collector/cache layer counts differ")
    rejected = 8 - keep
    for item, conv_input, args in zip(arrays, collector["conv"], collector["args"]):
        q, k, v, a, b, A_log, dt_bias, pre_state, use_kernel = args
        conv_keep = conv_input.shape[1] - 8
        item.cache[0] = mx.contiguous(conv_input[:, keep:keep + conv_keep, :])
        _, recurrent = gated_delta_update(q[:, :keep], k[:, :keep], v[:, :keep],
                                          a[:, :keep], b[:, :keep], A_log,
                                          dt_bias, pre_state, None,
                                          use_kernel=use_kernel)
        item.cache[1] = recurrent
    for item in cache:
        if not hasattr(item, "cache") and hasattr(item, "trim") and rejected:
            item.trim(rejected)


def mlx_r1_context_reference(target, tokenizer, drafter, row, tap, select_block,
                             full_prompt_ids, align_prefill=False,
                             measure_controls=False):
    """Rebuild the exact R1 prefill and drafter proposal on independent caches."""
    import mlx.core as mx
    from mlx_dspark.generate import PREFILL_CHUNK, _prefill_tapped
    from mlx_dspark.target import Target

    assert row["round_index"] == 1 and row["generated_before"] == 0
    prefix = list(row["initial_prefill_ids"])
    assert len(prefix) <= PREFILL_CHUNK
    assert row["proposal_context_rows"] == len(prefix)
    cfg = drafter.config
    all_sliding = bool(cfg.layer_types) and all(
        kind == "sliding_attention" for kind in cfg.layer_types)
    if all_sliding and cfg.sliding_window:
        assert len(prefix) <= cfg.sliding_window - 1
    reference_target = Target(target.model, tokenizer, kv_bits=target.kv_bits,
                              kv_group_size=target.kv_group_size)
    tap_ids = list(tap)

    def rebuild(repetition=None, side=None, *, capture_cache=False, propose=False):
        target_cache = reference_target.make_cache()
        drafter_cache = drafter.make_cache()
        logits, fused = _prefill_tapped(reference_target, prefix, target_cache,
                                       tap_ids, drafter=None)
        mx.eval(logits, fused)
        independent_anchor = (int(full_prompt_ids[-1]) if align_prefill else
                              int(mx.argmax(logits[0, -1]).item()))
        if fused.shape[-1] % len(tap_ids):
            raise AssertionError("R1 fused context width is not divisible by tap count")
        tap_width = fused.shape[-1] // len(tap_ids)
        tap_rows = {f"tap{layer}": array_record(fused[:, :, index * tap_width:(index + 1) * tap_width])
                    for index, layer in enumerate(tap_ids)}
        record = {
            "observation_id": f'{row["observation_id"]}:fresh-r1-prefill'
                              + (f':control-r{repetition}-{side}' if repetition else ''),
            "reference_root": "independent_fresh_r1_prefill",
            "input_ids": prefix, "anchor_id": independent_anchor,
            "anchor_source": "prompt_final_id" if align_prefill else "prefill_greedy_argmax",
            "drafter_block_ids": ([independent_anchor] +
                                   [int(cfg.mask_token_id)] * (cfg.block_size - 1)),
            "drafter_block_dtype": str(mx.array([0]).dtype),
            "tap_order": tap_ids,
            "positions": {"prefill_start": 0, "prefill_end": len(prefix),
                          "context_start": 0, "context_end": int(fused.shape[1])},
            "pending_context": array_record(fused), "tap_rows": tap_rows,
        }
        if capture_cache:
            target_state = cache_record(target_cache)
            drafter_state = cache_record(drafter_cache, domain="drafter")
            record["target_cache"] = target_state
            record["drafter_cache"] = drafter_state
            record["target_positions"] = semantic_positions(
                target_state, 0, row["prompt_length"], int(fused.shape[1]))
            record["drafter_positions"] = drafter_context_positions(
                drafter_state, int(fused.shape[1]))
            record["distinct_target_cache_objects"] = set(
                id(item) for item in target_cache).isdisjoint(
                    row["pre_proposal"]["branch"]["cache_object_ids"])
            assert record["distinct_target_cache_objects"]
        if propose:
            if independent_anchor != row["anchor_or_pending_id"]:
                record["draft_ids"] = None
                return freeze_record(record)
            block = mx.array([[independent_anchor] +
                              [int(cfg.mask_token_id)] * (cfg.block_size - 1)])
            draft = select_block(block, fused, drafter_cache,
                                 cap=row["draft_width"],
                                 anchor_id=independent_anchor)[0]
            mx.eval(draft)
            record["draft_ids"] = [int(value) for value in draft.tolist()]
            record["drafter_cache_after_proposal"] = cache_record(
                drafter_cache, domain="drafter")
        return freeze_record(record)

    controls = (measure_r1_context_controls(
        "mlx-dspark", _STORE.run_dir, stable_id(prefix),
        lambda repetition, side: rebuild(repetition, side), _STORE)
        if measure_controls else None)
    reference = rebuild(capture_cache=True, propose=True)
    return {"reference": reference, "five_run_context_controls": controls,
            "comparison": compare_r1_context_reference(row, reference, _STORE.run_dir,
                                                       controls)}


def run_mlx(args):
    global _LAYER_MAP
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
    doc["max_tokens"] = args.max_tokens
    doc["target_loader"] = "mlx_dspark.prism_pack.load"
    doc["kernel_flags"] = {key: val for key, val in os.environ.items() if key.startswith("MLX_DSPARK_")}
    doc["prism_mma_active"] = mlx_qmm_mma.active()
    doc["kv_bits"] = args.mlx_kv_bits
    doc["requested_kv_bits"] = args.mlx_kv_bits
    doc["sidecar_block_size"] = cfg.block_size
    doc["run_id"] = args.run_id
    doc["target_class"] = f"{type(target).__module__}.{type(target).__name__}"
    doc["target_model_class"] = f"{type(target.model).__module__}.{type(target.model).__name__}"
    _LAYER_MAP = model_layer_map(target.model)
    doc["layer_map"] = _LAYER_MAP
    doc["tap_order"] = list(cfg.target_layer_ids)
    doc["dispatch"] = {"prism_mma_active": mlx_qmm_mma.active(),
                       "gdn_capture_method": "Target._capture_linear scoped pass-through hooks"}
    qmm_wins = mlx_qmm_mma.wins()
    qmm_tile_max = mlx_qmm_mma.tile_max()
    doc["dispatch"]["qmm_wins"] = {str(key): value for key, value in qmm_wins.items()}
    doc["dispatch"]["qmm_tile_max"] = {str(key): value for key, value in qmm_tile_max.items()}
    qmm_phase = {"name": "other"}
    qmm_calls = {}
    orig_qmm = mlx_qmm_mma.qmm
    def traced_qmm(x, wq, sc, bi, group_size, bits):
        key = (x.shape[-1], wq.shape[0], bits, group_size)
        m = 1
        for dimension in x.shape[:-1]:
            m *= int(dimension)
        minimum = qmm_wins.get(key) if x.dtype == mx.bfloat16 else None
        path = ("mma" if minimum is not None and minimum <= m <= mlx_qmm_mma.M_MAX else
                "tiled" if minimum is not None and mlx_qmm_mma.M_MAX < m <= qmm_tile_max.get(key, 0)
                else "stock_quantized_matmul")
        record = (qmm_phase["name"], path, key, m)
        qmm_calls[str(record)] = qmm_calls.get(str(record), 0) + 1
        return orig_qmm(x, wq, sc, bi, group_size, bits)
    cache = target.make_cache()
    rounds = doc["rounds"]
    frozen_s8_roots = {}
    generated = 0
    initial_prefill_ids = list(ids[:-1] if args.align_prefill else ids)
    previous_committed_ids = []

    orig_select = drafter.select_block
    def select(block, pending_ctx, dcache, *a, **kw):
        nonlocal generated
        observed = len(rounds) < 3
        if observed:
            mx.eval(block)
            target_state = cache_record(cache)
            drafter_state = cache_record(dcache, domain="drafter")
            pending = array_record(pending_ctx)
            pending_rows = int(pending_ctx.shape[1]) if pending_ctx is not None else 0
            context_positions = drafter_context_positions(drafter_state, pending_rows)
            row = {
                "run_id": args.run_id, "round_index": len(rounds) + 1,
                "prompt_length": len(ids),
                "observation_id": f"{args.run_id}:mlx:r{len(rounds)+1}",
                "generated_before": generated,
                "initial_prefill_ids": list(initial_prefill_ids),
                "previous_committed_ids": list(previous_committed_ids),
                "committed_prefix_ids": [*initial_prefill_ids, *previous_committed_ids],
                "anchor_or_pending_id": int(kw["anchor_id"]),
                "drafter_block_ids": [int(value) for value in block[0].tolist()],
                "drafter_block_dtype": str(block.dtype),
                "target_pre_proposal": target_state,
                "drafter_pre_proposal": drafter_state,
                "pending_context": pending,
                "proposal_context_rows": context_positions["context_live_rows"],
                "proposal_context_absolute_end": context_positions["context_absolute_end"],
                "proposal_context_logical_position": generated,
                "target_positions": semantic_positions(target_state, generated, len(ids), pending_rows),
                "drafter_positions": context_positions,
            }
            row["pre_proposal"] = boundary(row, "pre_proposal", cache, branch="speculative",
                                           extra={"drafter_cache": drafter_state,
                                                  "pending_context": pending})
        result = orig_select(block, pending_ctx, dcache, *a, **kw)
        if observed:
            mx.eval(result[0])
            row["draft_ids"] = result[0].tolist()
            row["draft_width"] = len(row["draft_ids"])
            rounds.append(row)
            persist_round(row, "proposal")
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
            freeze_proposal(row)
            row["pre_verify"] = boundary(row, "pre_verify", round_cache,
                                         input_ids=row["verify_ids"])
            if row["target_sequence_width"] == 8:
                freeze_s8_preverify(row, round_cache, frozen_s8_roots)
                row["references"] = {"status": "deferred_until_after_continuous_generation"}
            else:
                row["references"] = {"status": "not_run", "reason": "S=8 requires seven drafts"}
        if observed:
            qmm_phase["name"] = f'r{row["round_index"]}:speculative_verify'
        try:
            result = orig_verify(verify_ids, round_cache, tap)
        finally:
            qmm_phase["name"] = "other"
        if observed:
            row["verify"] = logits_record(*result)
            row["post_verify"] = boundary(row, "post_verify_pre_reconcile", round_cache,
                                          input_ids=row["verify_ids"])
            row["gdn_capture"] = mlx_gdn_record(target._stash)
            row["executed_dispatch"] = {key: value for key, value in qmm_calls.items()
                                        if key.startswith(f'(\'r{row["round_index"]}:')}
            persist_round(row, "pre-reconcile")
        return result
    target.verify = verify

    orig_rollback = target.rollback
    def rollback(round_cache, n_rejected, accepted):
        row = rounds[-1] if rounds and "post_reconcile" not in rounds[-1] else None
        result = orig_rollback(round_cache, n_rejected, accepted)
        if row is not None:
            row["n_rejected"] = n_rejected
            row["accepted_prefix_ids"] = row["verify_ids"][:len(accepted)+1]
            row["post_reconcile"] = boundary(row, "post_reconcile", round_cache,
                                             input_ids=row["accepted_prefix_ids"])
            persist_round(row, "post-reconcile")
        return result
    target.rollback = rollback

    def acquire_required_gates(row, frozen_root, b2_reference):
        """Execute isolated forced transitions while all runtime objects are live."""
        from mlx_dspark.target import Target
        import numpy as np

        e_source_contract = assert_first_round_e_reconstructable(row)
        verify_ids = list(row["verify_ids"])
        if len(verify_ids) != 8 or row["draft_width"] != 7:
            raise AssertionError("required branches need an exact S=8/seven-draft source row")
        frozen = row["s8_frozen_pre_verify"]
        root_id = frozen["branch"]["cache_state_id"]
        root_observation_id = frozen["observation_id"]
        logical_position = row["generated_before"]
        s8_controls = [item for item in b2_reference["five_run_controls"]
                       if item.get("key", {}).get("role") == "s8_verify"]
        if not s8_controls or any(len(item.get("samples", [])) != 5 for item in s8_controls):
            raise AssertionError("B2 requires its exact five same-root S=8 controls")
        if any(item["key"].get("logical_state_id") != root_id or
               item["key"].get("branch_width") != 8 for item in s8_controls):
            raise AssertionError("B2 controls do not belong to the selected frozen S8 root")
        control_set_id = stable_id(s8_controls)
        branches = []

        def transition(accepted_count, oracle=False):
            keep = accepted_count + 1
            branch_cache = clone_cache(frozen_root)
            branch_target = Target(target.model, tokenizer, kv_bits=target.kv_bits,
                                   kv_group_size=target.kv_group_size)
            tokens = mx.array([verify_ids], dtype=mx.uint32)
            if oracle:
                logits, fused = branch_target.verify(tokens, branch_cache, list(cfg.target_layer_ids))
                capture = branch_target._stash
                apply_mlx_same_s8_oracle(branch_target, branch_cache, capture, keep)
                kind = "independent_same_s8_prefix_oracle"
            else:
                logits, fused = orig_verify(tokens, branch_cache, list(cfg.target_layer_ids))
                capture = target._stash
                # This is the exact generate.py convention: rejected draft count,
                # then the accepted draft ID list (anchor is already in the cache).
                orig_rollback(branch_cache, 7 - accepted_count,
                              verify_ids[1:1 + accepted_count])
                kind = "production_target_rollback"
            mx.eval(logits, fused)
            return {"kind": kind, "logits": logits, "fused": fused[:, :keep],
                    "cache": branch_cache,
                    "verify_record": logits_record(logits, fused),
                    "accepted_fused_record": array_record(fused[:, :keep]),
                    "gdn_capture": mlx_gdn_record(capture),
                    "post": boundary(row, "isolated_post_reconcile", branch_cache,
                                     input_ids=verify_ids[:keep], branch=kind)}

        for accepted_count in (0, 1):
            keep = accepted_count + 1
            prod = transition(accepted_count, oracle=False)
            oracle = transition(accepted_count, oracle=True)
            if not set(prod["post"]["branch"]["cache_object_ids"]).isdisjoint(
                    oracle["post"]["branch"]["cache_object_ids"]):
                raise AssertionError("C production/oracle cache identity overlap")
            c_state_equal = (prod["post"]["branch"]["cache_state_id"] ==
                             oracle["post"]["branch"]["cache_state_id"])
            c_structure_equal = (cache_semantic_signature(prod["post"]["cache"]) ==
                                 cache_semantic_signature(oracle["post"]["cache"]))
            c_fused_equal = prod["accepted_fused_record"]["sha256"] == oracle["accepted_fused_record"]["sha256"]
            verify_np = load_array_record(prod["verify_record"]["logits"], _STORE.run_dir)
            next_token = int(verify_np[0, keep - 1].argmax())
            d_rows = []
            for source, start in (("production", prod), ("same_s8_oracle", oracle)):
                dcache = clone_cache(start["cache"])
                before = boundary(row, "D_before", dcache, input_ids=[next_token],
                                  branch=f"D_{source}", branch_source=start["kind"])
                dlogits, dfused = target.run(mx.array([[next_token]], dtype=mx.uint32),
                                             dcache, list(cfg.target_layer_ids))
                mx.eval(dlogits, dfused)
                d_rows.append({"source": source, "input_token": next_token,
                               "logical_position": logical_position + keep,
                               "before": before,
                               "output_token": int(mx.argmax(dlogits[0, -1]).item()),
                               "after": boundary(row, "D_after", dcache,
                                                  input_ids=[next_token], branch=f"D_{source}"),
                               "outputs": logits_record(dlogits, dfused)})
            if not set(d_rows[0]["before"]["branch"]["cache_object_ids"]).isdisjoint(
                    d_rows[1]["before"]["branch"]["cache_object_ids"]):
                raise AssertionError("D production/oracle starts are not independent")
            d_root_id = oracle["post"]["branch"]["cache_state_id"]
            def make_d_pair(repetition, control_root_id):
                pair = []
                for _side in ("left", "right"):
                    control_cache = clone_cache(oracle["cache"])
                    control_logits, control_fused = target.run(
                        mx.array([[next_token]], dtype=mx.uint32), control_cache,
                        list(cfg.target_layer_ids))
                    mx.eval(control_logits, control_fused)
                    after = boundary(row, "D_control_after", control_cache,
                                     input_ids=[next_token], branch=f"D_control_{repetition}")
                    pair.append({"outputs": logits_record(control_logits, control_fused),
                                 "post_execute": {"cache": after["cache"]}})
                return pair
            d_controls = transient_pair_controls(root_id=d_root_id,
                component_prefix="", make_pair=make_d_pair,
                run_dir=_STORE.run_dir, store=_STORE)
            d_prod, d_oracle = d_rows
            d_prod_control_view = {"outputs": d_prod["outputs"], "post_execute": {"cache": d_prod["after"]["cache"]}}
            d_oracle_control_view = {"outputs": d_oracle["outputs"], "post_execute": {"cache": d_oracle["after"]["cache"]}}
            d_comparison = compare_controlled_vectors(d_prod_control_view,
                d_oracle_control_view, d_controls, _STORE.run_dir,
                expected_root_id=d_root_id)
            d_distances = {item["component"]: item.get("distance")
                           for item in d_comparison["numeric_comparisons"]}
            d_top2_exact = d_prod["outputs"]["top2_ids"] == d_oracle["outputs"]["top2_ids"]
            d_margin_distances = [abs(float(a) - float(b)) for a, b in zip(
                d_prod["outputs"]["top2_margins"], d_oracle["outputs"]["top2_margins"])]
            d_margin_within = (d_controls["margin_repeatable"] and all(
                value <= d_controls["margin_bound"] for value in d_margin_distances))
            d_status = ("inconclusive" if not d_controls["top2_ids_repeatable"] or
                        not d_controls["margin_repeatable"] else d_comparison["status"])
            if d_status != "inconclusive" and (not d_top2_exact or not d_margin_within):
                d_status = "fail"
            d_structure_exact = (cache_semantic_signature(d_prod["after"]["cache"]) ==
                                 cache_semantic_signature(d_oracle["after"]["cache"]) and
                                 d_prod["after"]["position"] == d_oracle["after"]["position"])
            if not d_structure_exact and d_status == "pass":
                d_status = "fail"

            # E reaches the equivalent next-proposal point on two fresh caches:
            # runtime select_block on one, explicit project_ctx/append_ctx on the other.
            prompt_context = load_mx_array_record(row["pending_context"], _STORE.run_dir, mx)
            anchor = next_token
            accepted_absolute_start = row["pre_verify"]["position"]["next_target_absolute_position"]
            mask = int(cfg.mask_token_id)
            block = mx.array([[anchor] + [mask] * (cfg.block_size - 1)], dtype=mx.uint32)
            prod_dcache = drafter.make_cache()
            first = int(row["anchor_or_pending_id"])
            first_block = mx.array([[first] + [mask] * (cfg.block_size - 1)], dtype=mx.uint32)
            first_prod_ids = orig_select(first_block, prompt_context, prod_dcache,
                                         cap=7, anchor_id=first)[0]
            mx.eval(first_prod_ids)
            first_prod_ids = [int(item) for item in first_prod_ids.tolist()]
            source_first_ids = verify_ids[1:]
            source_proposal = e_source_proposal_adjudication(source_first_ids, first_prod_ids)
            source_proposal_exact = source_proposal["source_proposal_exact"]
            prod_ids = orig_select(block, prod["fused"], prod_dcache,
                                   cap=7, anchor_id=anchor)[0]
            mx.eval(prod_ids)
            ref_dcache = drafter.make_cache()
            ref_state_projected_prompt = drafter.project_ctx(prompt_context)
            ref_state_projected_accepted = drafter.project_ctx(oracle["fused"])
            mx.eval(ref_state_projected_prompt, ref_state_projected_accepted)
            drafter.append_ctx(ref_state_projected_prompt, ref_dcache)
            drafter.append_ctx(ref_state_projected_accepted, ref_dcache)

            ref_prop_cache = drafter.make_cache()
            ref_first_ids = orig_select(first_block, prompt_context, ref_prop_cache,
                                        cap=7, anchor_id=first)[0]
            mx.eval(ref_first_ids)
            ref_first_ids = [int(item) for item in ref_first_ids.tolist()]
            independent_source_proposal_exact = ref_first_ids == source_first_ids
            ref_ids = orig_select(block, oracle["fused"], ref_prop_cache,
                                  cap=7, anchor_id=anchor)[0]
            mx.eval(ref_ids)
            prod_ids = [int(item) for item in prod_ids.tolist()]
            ref_ids = [int(item) for item in ref_ids.tolist()]
            prod_projected_prompt = drafter.project_ctx(prompt_context)
            prod_projected_accepted = drafter.project_ctx(prod["fused"])
            ref_projected_prompt = drafter.project_ctx(prompt_context)
            ref_projected_accepted = drafter.project_ctx(oracle["fused"])
            mx.eval(prod_projected_prompt, prod_projected_accepted,
                    ref_projected_prompt, ref_projected_accepted)
            projected_rows_exact = (
                np.array_equal(mlx_numpy_value(prod_projected_prompt),
                               mlx_numpy_value(ref_projected_prompt)) and
                np.array_equal(mlx_numpy_value(prod_projected_accepted),
                               mlx_numpy_value(ref_projected_accepted)))
            e_projection_distances = {
                "max_abs_prompt": float(np.max(np.abs(mlx_numpy_value(prod_projected_prompt).astype(np.float64) -
                                                      mlx_numpy_value(ref_projected_prompt).astype(np.float64)))),
                "max_abs_accepted": float(np.max(np.abs(mlx_numpy_value(prod_projected_accepted).astype(np.float64) -
                                                        mlx_numpy_value(ref_projected_accepted).astype(np.float64))))}
            def make_e_pair(repetition, control_root_id):
                pair = []
                for _side in ("left", "right"):
                    control_cache = drafter.make_cache()
                    projected_prompt = drafter.project_ctx(prompt_context)
                    projected_accepted = drafter.project_ctx(oracle["fused"])
                    mx.eval(projected_prompt, projected_accepted)
                    drafter.append_ctx(projected_prompt, control_cache)
                    drafter.append_ctx(projected_accepted, control_cache)

                    control_prop_cache = drafter.make_cache()
                    control_first_ids = orig_select(first_block, prompt_context, control_prop_cache,
                                                    cap=7, anchor_id=first)[0]
                    mx.eval(control_first_ids)
                    control_first_ids = [int(item) for item in control_first_ids.tolist()]
                    if control_first_ids != source_first_ids:
                        raise AssertionError("mlx E control failed to reproduce the source first proposal")
                    control_ids = orig_select(block, oracle["fused"], control_prop_cache,
                                              cap=7, anchor_id=anchor)[0]
                    mx.eval(control_ids)
                    pair.append({"proposal_ids": [int(item) for item in control_ids.tolist()],
                        "outputs": {"projected_prompt": array_record(projected_prompt),
                                    "projected_accepted": array_record(projected_accepted)},
                        "post_execute": {"cache": cache_record(control_cache, domain="drafter")}})
                return pair
            e_root_id = stable_id({"prompt": row["pending_context"]["sha256"],
                "accepted": oracle["accepted_fused_record"]["sha256"],
                "logical_position": logical_position, "keep": keep, "anchor": anchor})
            e_controls = transient_pair_controls(root_id=e_root_id, component_prefix="",
                make_pair=make_e_pair, run_dir=_STORE.run_dir, store=_STORE,
                require_top2=False)
            e_reference_root_id = stable_id({
                "prompt_context_sha256": hashlib.sha256(mlx_numpy_value(ref_projected_prompt).tobytes()).hexdigest(),
                "accepted_context_sha256": hashlib.sha256(mlx_numpy_value(ref_projected_accepted).tobytes()).hexdigest(),
                "logical_position": logical_position, "accepted_count": accepted_count,
                "next_anchor": anchor})
            prod_e_state = cache_record(prod_dcache, domain="drafter")
            ref_e_state = cache_record(ref_dcache, domain="drafter")
            e_structure_equal = cache_semantic_signature(prod_e_state) == cache_semantic_signature(ref_e_state)
            e_disjoint = set(mutable_cache_object_ids(prod_dcache)).isdisjoint(
                mutable_cache_object_ids(ref_dcache))
            if not e_disjoint:
                raise AssertionError("E drafter caches are not independent")
            e_prod_view = {"outputs": {"projected_prompt": array_record(prod_projected_prompt),
                                        "projected_accepted": array_record(prod_projected_accepted)},
                           "post_execute": {"cache": prod_e_state}}
            e_ref_view = {"outputs": {"projected_prompt": array_record(ref_projected_prompt),
                                       "projected_accepted": array_record(ref_projected_accepted)},
                          "post_execute": {"cache": ref_e_state}}
            e_numeric_comparison = compare_controlled_vectors(e_prod_view, e_ref_view,
                e_controls, _STORE.run_dir, expected_root_id=e_root_id)
            e_numeric_status = ("inconclusive" if not source_proposal_exact or
                                not independent_source_proposal_exact or
                                not e_controls["proposal_ids_repeatable"] else
                                "fail" if prod_ids != ref_ids or not e_structure_equal else
                                e_numeric_comparison["status"])
            provenance = required_gate_branch_provenance(
                runtime="mlx-dspark", frozen_root_state_id=root_id,
                frozen_root_observation_id=root_observation_id,
                branch_id=f"{args.run_id}:isolated:accepted{accepted_count}",
                logical_position=logical_position, verify_ids=verify_ids,
                accepted_count=accepted_count, b2_control_set_id=control_set_id)
            branches.append({"branch_kind": "isolated_forced",
                "acceptance_source": "forced_state_transition_test_input",
                "accepted_count": accepted_count, "keep": keep,
                "provenance": provenance,
                "c": {"production": {"post": prod["post"], "verify": prod["verify_record"],
                                      "gdn_capture": prod["gdn_capture"]},
                      "independent_same_s8_oracle": {"post": oracle["post"],
                                                      "verify": oracle["verify_record"],
                                                      "accepted_fused": oracle["accepted_fused_record"],
                                                      "gdn_capture": oracle["gdn_capture"]},
                      "production_accepted_fused": prod["accepted_fused_record"],
                      "cache_objects_disjoint": True, "exact_cache_state_equal": c_state_equal,
                      "exact_structure_equal": c_structure_equal,
                      "exact_accepted_fused_equal": c_fused_equal,
                      "accepted_prefix_ids": verify_ids[:keep],
                      "accepted_prefix_exact": prod["post"]["input_ids"] == verify_ids[:keep]},
                "d": {"next_committed_token": next_token, "evaluations": d_rows,
                      "same_input_token": True, "starting_caches_disjoint": True,
                      "five_same_root_width1_controls": d_controls,
                      "control_contract": {"root_state_id": oracle["post"]["branch"]["cache_state_id"],
                                           "width": 1, "repetitions": 5,
                                           "calibration_repetitions": [1, 2, 3],
                                           "validation_repetitions": [4, 5],
                                           "predeclared_components": d_controls["component_set"],
                                           "metrics": "component max_abs; exact structure and discrete IDs"},
                      "control_results": d_controls,
                      "production_oracle_numeric_distance": d_distances,
                      "production_oracle_numeric_comparison": d_comparison,
                      "top2_ids_exact": d_top2_exact,
                      "resulting_structure_exact": d_structure_exact,
                      "margin_distances": d_margin_distances,
                      "margin_within_control_bound": d_margin_within,
                      "numeric_status": d_status,
                      "fresh_s1_role": "serial_semantic_diagnostic_only",
                      "s1_reference_pointer": {
                          "keep": keep,
                          "observation_id": row["references"]["s1_keep1" if keep == 1 else "s1_keep2"]["pre_verify"]["observation_id"],
                          "cache_state_id": row["references"]["s1_keep1" if keep == 1 else "s1_keep2"]["pre_verify"]["branch"]["cache_state_id"]}},
                "e": {"initial_prompt_context": array_record(prompt_context),
                      "source_reconstruction": e_source_contract,
                      "reconstructed_first_proposal_ids": first_prod_ids,
                      "independent_reconstructed_first_proposal_ids": ref_first_ids,
                      "source_first_proposal_ids": source_first_ids,
                      "source_proposal": source_proposal,
                      "source_proposal_exact": source_proposal_exact,
                      "independent_source_proposal_exact": independent_source_proposal_exact,
                      "accepted_context_rows": array_record(oracle["fused"]),
                      "logical_position": logical_position,
                      "accepted_logical_positions": list(range(logical_position, logical_position + keep)),
                      "accepted_absolute_positions": list(range(accepted_absolute_start,
                                                                 accepted_absolute_start + keep)),
                      "projected_prompt_rows": array_record(ref_projected_prompt),
                      "projected_accepted_rows": array_record(ref_projected_accepted),
                      "production_projected_prompt_rows": array_record(prod_projected_prompt),
                      "production_projected_accepted_rows": array_record(prod_projected_accepted),
                      "projection_numeric_distance": e_projection_distances,
                      "projection_control_bound": e_controls["components"],
                      "drafter_cache_numeric_comparison": e_numeric_comparison,
                      "production_drafter_cache": prod_e_state,
                      "independent_drafter_cache": ref_e_state,
                      "drafter_caches_disjoint": e_disjoint,
                      "next_anchor": anchor, "production_next_proposal_ids": prod_ids,
                      "independent_next_proposal_ids": ref_ids,
                      "proposal_ids_equal": prod_ids == ref_ids if source_proposal_exact else None,
                      "numeric_status": e_numeric_status,
                      "drafter_structure_equal": e_structure_equal,
                      "projected_context_rows_exact": projected_rows_exact,
                      "next_block_ids": [anchor] + [mask] * (cfg.block_size - 1),
                      "independent_projection_append": True,
                      "equivalent_logical_proposal_point": (
                          source_proposal_exact and independent_source_proposal_exact),
                      "five_independent_same_runtime_controls": e_controls,
                      "control_contract": {"reference_root_state_id": e_root_id,
                                           "repetitions": 5,
                                           "calibration_repetitions": [1, 2, 3],
                                           "validation_repetitions": [4, 5],
                                           "predeclared_components": e_controls["component_set"],
                                           "metrics": "component max_abs; proposal IDs exact"},
                      "controls_exact_stable": e_controls["status"] == "STABLE"}})
        branches = link_shared_b2_control_set(branches)
        return required_gate_acquisition_record(
                row["observation_id"],
                {"root_state_id": root_id, "root_observation_id": root_observation_id,
                       "logical_position": logical_position, "verify_ids": verify_ids,
                       "control_set_id": control_set_id, "control_count": 5,
                       "control_component_count": len(s8_controls),
                       "reference": b2_reference["s8_verify"], "controls": s8_controls},
                branches)

    def on_round(**event):
        nonlocal generated
        if rounds and "accepted_count" not in rounds[-1]:
            assert stable_id(rounds[-1]["proposal_input_record"]) == rounds[-1]["proposal_input_sha256"]
            rounds[-1]["accepted_count"] = event["accepted"]
            rounds[-1]["committed_count"] = event["committed"]
            rounds[-1]["accepted_prefix_ids"] = rounds[-1]["verify_ids"][:event["accepted"] + 1]
            rounds[-1]["acceptance"] = {"accepted_count": event["accepted"],
                                        "committed_count": event["committed"],
                                        "proposal_input_sha256": rounds[-1]["proposal_input_sha256"]}
            previous_committed_ids.extend(rounds[-1]["accepted_prefix_ids"])
        generated += event["committed"]

    mlx_qmm_mma.qmm = traced_qmm
    try:
        result = dflash_generate(
            target, tokenizer, drafter, prompt_ids=list(ids), cache=cache,
            ctx_caches=None, max_new_tokens=args.max_tokens,
            max_draft_tokens=7, lookup_drafts=False, cap_controller=None,
            temperature=0.0, top_p=0.0, top_k=0, on_round=on_round,
        )
        doc["output_ids"] = result.token_ids
        doc["num_rounds"] = result.num_rounds
        doc["finish_reason"] = result.finish_reason
        doc["historical_trace_measurement"] = historical_trace_measurement(doc)
        (_STORE.run_dir / "continuous.json").write_text(
            json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        # All reference forwards happen after the continuous adaptive trace.
        # The R1 context uses an independent fresh prefill; S=8 instead uses
        # the exact cache clone frozen inside the production verify wrapper.
        doc["r1_context_reference"] = mlx_r1_context_reference(
            target, tokenizer, drafter, rounds[0], list(cfg.target_layer_ids),
            orig_select, ids, args.align_prefill, args.measure_controls)
        s8_rows = [row for row in rounds if row.get("target_sequence_width") == 8
                   and row.get("round_index") in frozen_s8_roots]
        required_source = rounds[0] if args.measure_required_gates and rounds else None
        if args.measure_required_gates and (required_source is None or
                required_source.get("target_sequence_width") != 8 or
                required_source.get("round_index") not in frozen_s8_roots):
            raise RuntimeError("required-gates E currently requires the first logical speculative round to be a valid S=8/seven-draft round; later S=8 roots are not reconstructable")
        for row in s8_rows:
            qmm_phase["name"] = f'r{row["round_index"]}:references'
            try:
                frozen_root = frozen_s8_roots[row["round_index"]]
                reference_root = clone_cache(frozen_root) if args.measure_required_gates else frozen_root
                row["references"] = mlx_reference_branches(
                    target, tokenizer, row, list(cfg.target_layer_ids),
                    reference_root,
                    args.measure_controls and (not args.measure_required_gates or row is required_source),
                    qmm_calls)
                if not args.measure_required_gates:
                    frozen_s8_roots.pop(row["round_index"])
            finally:
                qmm_phase["name"] = "other"
        if args.measure_required_gates:
            doc["required_gate_acquisition"] = acquire_required_gates(
                required_source, frozen_s8_roots[required_source["round_index"]],
                required_source["references"])
    finally:
        drafter.select_block = orig_select
        target.verify = orig_verify
        target.rollback = orig_rollback
        mlx_qmm_mma.qmm = orig_qmm
    doc["dispatch"]["qmm_calls"] = qmm_calls
    doc["output_ids"] = result.token_ids
    doc["num_rounds"] = result.num_rounds
    doc["finish_reason"] = result.finish_reason
    return doc


def run_chad(args):
    global _LAYER_MAP
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
    doc["max_tokens"] = args.max_tokens
    doc["target_loader"] = "chad.prism_pack.load via Engine.load"
    doc["chad_fastpath"] = eng.fastpath
    doc["kv_bits"] = eng.kv_bits
    doc["requested_kv_bits"] = args.chad_kv_bits
    doc["CHAD_NO_FASTPATH"] = os.environ.get("CHAD_NO_FASTPATH")
    doc["gdn_call_file"] = qwen3_5.GatedDeltaNet.__call__.__code__.co_filename
    doc["gdn_call_line"] = qwen3_5.GatedDeltaNet.__call__.__code__.co_firstlineno
    doc["kernel_flags"] = {key: val for key, val in os.environ.items() if key.startswith("CHAD_")}
    doc["run_id"] = args.run_id
    doc["target_class"] = f"{type(eng.model).__module__}.{type(eng.model).__name__}"
    doc["target_model_class"] = f"{type(eng.model.language_model.model).__module__}.{type(eng.model.language_model.model).__name__}"
    _LAYER_MAP = model_layer_map(eng.model.language_model.model)
    doc["layer_map"] = _LAYER_MAP
    doc["tap_order"] = list(eng._dflash.config.target_layer_ids)
    installed_call = qwen3_5.GatedDeltaNet.__call__
    doc["chad_fastpath_record"] = {
        "engine_fastpath": bool(eng.fastpath),
        "CHAD_NO_FASTPATH": os.environ.get("CHAD_NO_FASTPATH"),
        "installed_class": f"{qwen3_5.GatedDeltaNet.__module__}.{qwen3_5.GatedDeltaNet.__name__}",
        "installed_call_file": inspect.getfile(installed_call),
        "installed_call_line": installed_call.__code__.co_firstlineno,
        "call_path_is_fastpath_module": "chad/mlx_fastpath.py" in inspect.getfile(installed_call),
        "projection_replacement_flags": {
            "fused_mlp_layers": sum(hasattr(layer.mlp, "_fused_w") for layer in
                                    eng.model.language_model.model.layers),
            "fused_gdn_layers": sum(bool(layer.is_linear and
                                         hasattr(layer.linear_attn, "_fused_w")) for layer in
                                     eng.model.language_model.model.layers),
            "fused_attention_layers": sum(bool(not layer.is_linear and
                                               "_PrismAttention" in type(layer.self_attn).__name__)
                                          for layer in eng.model.language_model.model.layers),
        },
    }
    rounds = doc["rounds"]
    frozen_s8_roots = {}
    generated = 0
    initial_prefill_ids = list(ids[:-1])
    previous_committed_ids = []

    orig_propose = engine_module._DFlashDrafter.propose
    def propose(self, k, anchor, rng):
        nonlocal generated
        observed = len(rounds) < 3
        if observed:
            target_state = cache_record(self.eng._cache)
            drafter_state = cache_record(self.cache, domain="drafter")
            pending = array_record(self.pending)
            pending_rows = int(self.pending.shape[1]) if self.pending is not None else 0
            context_positions = drafter_context_positions(drafter_state, pending_rows)
            row = {
                "run_id": args.run_id, "round_index": len(rounds) + 1,
                "prompt_length": len(ids),
                "observation_id": f"{args.run_id}:chad:r{len(rounds)+1}",
                "generated_before": generated,
                "initial_prefill_ids": list(initial_prefill_ids),
                "previous_committed_ids": list(previous_committed_ids),
                "committed_prefix_ids": [*initial_prefill_ids, *previous_committed_ids],
                "anchor_or_pending_id": int(anchor),
                "drafter_block_ids": ([int(anchor)] +
                                       [int(self.m.config.mask_token_id)] *
                                       (self.m.config.block_size - 1)),
                "drafter_block_dtype": str(mx.uint32),
                "target_pre_proposal": target_state,
                "drafter_pre_proposal": drafter_state,
                "pending_context": pending,
                "proposal_context_rows": context_positions["context_live_rows"],
                "proposal_context_absolute_end": context_positions["context_absolute_end"],
                "proposal_context_logical_position": generated,
                "target_positions": semantic_positions(target_state, generated, len(ids), pending_rows),
                "drafter_positions": context_positions,
            }
            row["pre_proposal"] = boundary(row, "pre_proposal", self.eng._cache,
                                           extra={"drafter_cache": drafter_state,
                                                  "pending_context": pending})
        result = orig_propose(self, k, anchor, rng)
        if observed:
            mx.eval(result[0])
            row["draft_ids"] = [int(x) for x in result[0]]
            row["draft_width"] = k
            rounds.append(row)
            persist_round(row, "proposal")
        return result
    engine_module._DFlashDrafter.propose = propose

    model_class = type(eng.model.language_model.model)
    orig_model_call = model_class.__call__

    def fresh_chad_cache():
        from mlx_lm.models import cache as cache_utils

        new_cache = cache_utils.make_prompt_cache(eng.model)
        if eng.kv_bits:
            new_cache = [cache_utils.QuantizedKVCache(group_size=64, bits=eng.kv_bits)
                         if type(c) is cache_utils.KVCache else c for c in new_cache]
        return new_cache

    def chad_references(model_self, call_args, call_kw, row, frozen_s8_cache):
        verify = list(row["verify_ids"])
        references = {}
        tap_ids = list(eng._dflash.config.target_layer_ids)
        frozen = row["s8_frozen_pre_verify"]
        assert len(verify) == 8 and frozen["input_ids"] == verify
        supplied_root_state = cache_record(frozen_s8_cache)
        assert stable_id(evidence_content_view(supplied_root_state)) == frozen["branch"]["cache_state_id"]
        assert set(mutable_cache_object_ids(frozen_s8_cache)).isdisjoint(
            row["pre_verify"]["branch"]["cache_object_ids"])
        replay_cache = fresh_chad_cache()
        prior_tap = mlx_dflash.TAP
        prior_collector = mlx_fastpath.GDN_COLLECTOR
        try:
            mlx_dflash.TAP = {}
            mlx_fastpath.GDN_COLLECTOR = None
            orig_model_call(model_self, mx.array([row["initial_prefill_ids"]], dtype=mx.uint32),
                            *call_args, **{**call_kw, "cache": replay_cache})
            for token in row["previous_committed_ids"]:
                orig_model_call(model_self, mx.array([[token]], dtype=mx.uint32),
                                *call_args, **{**call_kw, "cache": replay_cache})
        finally:
            mlx_dflash.TAP = prior_tap
            mlx_fastpath.GDN_COLLECTOR = prior_collector
        replay = boundary(row, "fresh_committed_replay", replay_cache,
                          input_ids=row["committed_prefix_ids"], branch="fresh_replay",
                          branch_source="empty cache + committed-token replay")
        references["fresh_replay"] = replay
        references["committed_replay_structure_equal_to_speculative"] = (
            cache_semantic_signature(replay["cache"]) ==
            cache_semantic_signature(row["pre_verify"]["cache"]))
        references["committed_replay_numeric_state_id_equal_to_speculative"] = (
            replay["branch"]["cache_state_id"] ==
            row["pre_verify"]["branch"]["cache_state_id"])

        def execute(kind, pieces, *, root, root_name, clone_root):
            branch_cache = clone_cache(root) if clone_root else root
            same_width = root_name == "exact_frozen_speculative_preverify"
            expected = frozen if same_width else replay
            if same_width:
                assert pieces == [verify]
            pre = boundary(row, "pre_verify", branch_cache,
                           input_ids=[x for p in pieces for x in p],
                           branch=kind, branch_source=root_name)
            assert set(pre["branch"]["cache_object_ids"]).isdisjoint(
                row["pre_verify"]["branch"]["cache_object_ids"])
            assert pre["branch"]["cache_state_id"] == expected["branch"]["cache_state_id"]
            assert cache_semantic_signature(pre["cache"]) == cache_semantic_signature(expected["cache"])
            assert pre["position"] == expected["position"]
            outputs = []
            captures = []
            dispatch = []
            for piece in pieces:
                prior_tap = mlx_dflash.TAP
                prior_collector = mlx_fastpath.GDN_COLLECTOR
                own_tap = {}
                own_collector = {"conv": [], "args": []}
                try:
                    mlx_dflash.TAP = own_tap
                    mlx_fastpath.GDN_COLLECTOR = own_collector
                    dispatch.append(chad_executed_layer_paths(
                        model_self, branch_cache, len(piece)))
                    hidden = orig_model_call(model_self, mx.array([piece], dtype=mx.uint32),
                                             *call_args, **{**call_kw, "cache": branch_cache})
                    fused = mx.concatenate([own_tap[i] for i in tap_ids], axis=-1)
                    logits = eng.model.language_model.lm_head(hidden)
                    outputs.append(logits_record(logits, fused))
                    captures.append(chad_gdn_record(own_collector))
                finally:
                    mlx_dflash.TAP = prior_tap
                    mlx_fastpath.GDN_COLLECTOR = prior_collector
            return {"implementation": "chad", "reference_kind": kind,
                    "reference_root": root_name,
                    "root_observation_id": expected["observation_id"],
                    "target_sequence_widths": [len(piece) for piece in pieces],
                    "pre_verify": pre, "outputs": outputs,
                    "gdn_capture": captures,
                    "executed_dispatch": dispatch,
                    "post_execute": boundary(row, "post_execute", branch_cache,
                                             input_ids=[x for p in pieces for x in p], branch=kind)}

        if args.measure_controls:
            def make_pair(role, repetition, side):
                actual = "s2_keep2" if role == "s2_vs_s1_keep2" and side == "left" else (
                         "s1_keep2" if role == "s2_vs_s1_keep2" else role)
                pieces = {
                    "s8_verify": [verify], "s1_keep1": [[verify[0]]],
                    "s1_keep2": [[verify[0]], [verify[1]]],
                    "s2_keep2": [[verify[0], verify[1]]],
                }[actual]
                same_width = actual == "s8_verify"
                return execute(f"{role}_control_r{repetition}_{side}", pieces,
                               root=frozen_s8_cache if same_width else replay_cache,
                               root_name="exact_frozen_speculative_preverify" if same_width
                                         else "fresh_committed_replay", clone_root=True)
            root_ids = {role: (frozen if role == "s8_verify" else replay)["branch"]["cache_state_id"]
                        for role in ("s8_verify", "s1_keep1", "s1_keep2", "s2_keep2",
                                     "s2_vs_s1_keep2")}
            references["five_run_controls"] = measure_control_matrix(
                "chad", _STORE.run_dir, root_ids, make_pair, _STORE)
        references["s8_verify"] = execute("s8_verify_reference", [verify],
                                          root=frozen_s8_cache,
                                          root_name="exact_frozen_speculative_preverify",
                                          clone_root=False)
        references["s1_keep1"] = execute("s1_keep1_reference", [[verify[0]]],
                                         root=replay_cache, root_name="fresh_committed_replay",
                                         clone_root=True)
        references["s1_keep2"] = execute("s1_keep2_reference", [[verify[0]], [verify[1]]],
                                         root=replay_cache, root_name="fresh_committed_replay",
                                         clone_root=True)
        references["s2_keep2"] = execute("s2_keep2_reference", [[verify[0], verify[1]]],
                                         root=replay_cache, root_name="fresh_committed_replay",
                                         clone_root=True)
        return references

    def chad_r1_context_reference(row):
        """Use Chad's own prefill primitive with an independent cache and tap sink."""
        assert row["round_index"] == 1 and row["generated_before"] == 0
        prefix = list(row["initial_prefill_ids"])
        assert row["pending_context"]["shape"][1] == len(prefix)
        assert all(item.get("live_length", 0) == 0
                   for item in row["drafter_pre_proposal"])
        tap_ids = list(eng._dflash.config.target_layer_ids)
        cfg = eng._dflash.config
        independent_anchor = int(ids[-1])

        def rebuild(repetition=None, side=None, *, capture_cache=False, propose=False):
            target_cache = fresh_chad_cache()
            drafter_cache = eng._dflash.make_cache()
            saved_target_cache, saved_cached_ids = eng._cache, eng._cached_ids
            saved_tap, saved_collector = mlx_dflash.TAP, mlx_fastpath.GDN_COLLECTOR
            own_tap = {}
            pending = None

            def on_chunk():
                nonlocal pending
                fused = mx.concatenate([own_tap[layer] for layer in tap_ids], axis=-1)
                if pending is not None:
                    eng._dflash.append_ctx(eng._dflash.project_ctx(pending), drafter_cache)
                pending = fused
                own_tap.clear()

            try:
                eng._cache = target_cache
                eng._cached_ids = []
                mlx_dflash.TAP = own_tap
                mlx_fastpath.GDN_COLLECTOR = None
                fed = eng._prefill(prefix, chunk=len(prefix), on_chunk=on_chunk)
                if fed != len(prefix) or pending is None:
                    raise AssertionError("Independent Chad R1 prefill did not finish exactly")
                mx.eval(pending)
            finally:
                eng._cache, eng._cached_ids = saved_target_cache, saved_cached_ids
                mlx_dflash.TAP, mlx_fastpath.GDN_COLLECTOR = saved_tap, saved_collector
            if pending.shape[-1] % len(tap_ids):
                raise AssertionError("R1 fused context width is not divisible by tap count")
            tap_width = pending.shape[-1] // len(tap_ids)
            tap_rows = {f"tap{layer}": array_record(
                pending[:, :, index * tap_width:(index + 1) * tap_width])
                        for index, layer in enumerate(tap_ids)}
            record = {
                "observation_id": f'{row["observation_id"]}:fresh-r1-prefill'
                                  + (f':control-r{repetition}-{side}' if repetition else ''),
                "reference_root": "independent_fresh_r1_prefill",
                "input_ids": prefix, "anchor_id": independent_anchor,
                "anchor_source": "prompt_final_id",
                "drafter_block_ids": ([independent_anchor] +
                                       [int(cfg.mask_token_id)] * (cfg.block_size - 1)),
                "drafter_block_dtype": str(mx.uint32),
                "tap_order": tap_ids,
                "positions": {"prefill_start": 0, "prefill_end": len(prefix),
                              "context_start": 0, "context_end": int(pending.shape[1])},
                "pending_context": array_record(pending), "tap_rows": tap_rows,
            }
            if capture_cache:
                target_state = cache_record(target_cache)
                drafter_state = cache_record(drafter_cache, domain="drafter")
                record["target_cache"] = target_state
                record["drafter_cache"] = drafter_state
                record["target_positions"] = semantic_positions(
                    target_state, 0, row["prompt_length"], int(pending.shape[1]))
                record["drafter_positions"] = drafter_context_positions(
                    drafter_state, int(pending.shape[1]))
                record["distinct_target_cache_objects"] = set(
                    id(item) for item in target_cache).isdisjoint(
                        row["pre_proposal"]["branch"]["cache_object_ids"])
                assert record["distinct_target_cache_objects"]
            if propose:
                if independent_anchor != row["anchor_or_pending_id"]:
                    record["draft_ids"] = None
                    return freeze_record(record)
                block = mx.array([[independent_anchor] +
                                  [int(cfg.mask_token_id)] * (cfg.block_size - 1)],
                                 dtype=mx.uint32)
                draft = eng._dflash.select_block(block, pending, drafter_cache,
                                                 cap=row["draft_width"],
                                                 anchor_id=independent_anchor)[0]
                mx.eval(draft)
                record["draft_ids"] = [int(value) for value in draft.tolist()]
                record["drafter_cache_after_proposal"] = cache_record(
                    drafter_cache, domain="drafter")
            return freeze_record(record)

        controls = (measure_r1_context_controls(
            "chad", _STORE.run_dir, stable_id(prefix),
            lambda repetition, side: rebuild(repetition, side), _STORE)
            if args.measure_controls else None)
        reference = rebuild(capture_cache=True, propose=True)
        return {"reference": reference, "five_run_context_controls": controls,
                "comparison": compare_r1_context_reference(row, reference, _STORE.run_dir,
                                                           controls)}

    def model_call(self, input_ids, *a, **kw):
        observed = rounds and len(rounds) <= 3 and "verify_ids" not in rounds[-1]
        row = rounds[-1] if observed else None
        if observed:
            mx.eval(input_ids)
            row["verify_ids"] = input_ids[0].tolist()
            row["target_sequence_width"] = len(row["verify_ids"])
            freeze_proposal(row)
            row["pre_verify"] = boundary(row, "pre_verify", kw["cache"],
                                         input_ids=row["verify_ids"])
            row["executed_dispatch"] = chad_executed_layer_paths(
                eng.model.language_model.model, kw["cache"], row["target_sequence_width"])
            if row["target_sequence_width"] == 8:
                freeze_s8_preverify(row, kw["cache"], frozen_s8_roots)
                row["references"] = {"status": "deferred_until_after_continuous_generation"}
            else:
                row["references"] = {"status": "not_run", "reason": "S=8 requires seven drafts"}
        hid = orig_model_call(self, input_ids, *a, **kw)
        if observed:
            fused = mx.concatenate([mlx_dflash.TAP[i] for i in eng._dflash.config.target_layer_ids], axis=-1)
            logits = eng.model.language_model.lm_head(hid)
            row["verify"] = logits_record(logits, fused)
            row["post_verify"] = boundary(row, "post_verify_pre_reconcile", kw["cache"],
                                          input_ids=row["verify_ids"])
            row["gdn_capture"] = chad_gdn_record(mlx_fastpath.GDN_COLLECTOR)
            persist_round(row, "pre-reconcile")
        return hid
    model_class.__call__ = model_call

    orig_reconcile = engine_module._DFlashDrafter.reconcile
    def reconcile(self, k, accepted, anchor, draft, hid, fused):
        nonlocal generated
        result = orig_reconcile(self, k, accepted, anchor, draft, hid, fused)
        if rounds and "accepted_count" not in rounds[-1]:
            row = rounds[-1]
            assert stable_id(row["proposal_input_record"]) == row["proposal_input_sha256"]
            row["accepted_count"] = accepted
            row["committed_count"] = accepted + 1
            row["n_rejected"] = k - accepted
            row["accepted_prefix_ids"] = row["verify_ids"][:accepted+1]
            row["post_reconcile"] = boundary(row, "post_reconcile", self.eng._cache,
                                             input_ids=row["accepted_prefix_ids"])
            row["accepted_fused"] = array_record(fused[:, :accepted+1])
            row["acceptance"] = {"accepted_count": accepted, "committed_count": accepted + 1,
                                 "proposal_input_sha256": row["proposal_input_sha256"]}
            previous_committed_ids.extend(row["accepted_prefix_ids"])
            persist_round(row, "post-reconcile")
        generated += accepted + 1
        return result
    engine_module._DFlashDrafter.reconcile = reconcile

    def acquire_required_gates(row, frozen_root, b2_reference):
        """Run isolated S=8 branches while Chad's target and drafter are live."""
        import numpy as np

        e_source_contract = assert_first_round_e_reconstructable(row)
        verify_ids = list(row["verify_ids"])
        frozen = row["s8_frozen_pre_verify"]
        root_id = frozen["branch"]["cache_state_id"]
        root_observation_id = frozen["observation_id"]
        logical_position = row["generated_before"]
        controls = [item for item in b2_reference["five_run_controls"]
                    if item.get("key", {}).get("role") == "s8_verify"]
        if len(verify_ids) != 8 or row["draft_width"] != 7 or not controls or any(
                len(item.get("samples", [])) != 5 for item in controls):
            raise AssertionError("required-gates acquisition lacks S=8/seven-draft B2 controls")
        if any(item["key"].get("logical_state_id") != root_id or
               item["key"].get("branch_width") != 8 for item in controls):
            raise AssertionError("Chad B2 controls do not belong to the selected frozen S8 root")
        control_set_id = stable_id(controls)
        tap_ids = list(eng._dflash.config.target_layer_ids)
        initial_context = load_mx_array_record(row["pending_context"], _STORE.run_dir, mx)
        initial_anchor = int(row["anchor_or_pending_id"])

        def target_forward(cache, tokens):
            own_tap = {}
            own_collector = {"conv": [], "args": []}
            prior_tap, prior_collector = mlx_dflash.TAP, mlx_fastpath.GDN_COLLECTOR
            try:
                mlx_dflash.TAP = own_tap
                mlx_fastpath.GDN_COLLECTOR = own_collector
                hidden = orig_model_call(eng.model.language_model.model,
                                         mx.array([tokens], dtype=mx.uint32), cache=cache)
                fused = mx.concatenate([own_tap[layer] for layer in tap_ids], axis=-1)
                logits = eng.model.language_model.lm_head(hidden)
                mx.eval(logits, fused)
                return hidden, logits, fused, own_collector
            finally:
                mlx_dflash.TAP, mlx_fastpath.GDN_COLLECTOR = prior_tap, prior_collector

        def source_matched_chad_rollback(cache, collector, keep):
            """Mirror _generate_spec's inline hybrid rollback block verbatim in semantics."""
            arrays = [item for item in cache if hasattr(item, "cache")]
            from mlx_lm.models.gated_delta import gated_delta_update
            if len(collector["conv"]) != len(arrays) or len(collector["args"]) != len(arrays):
                raise AssertionError("Chad target rollback requires the matching GDN collector")
            for item, conv_input, args in zip(arrays, collector["conv"], collector["args"]):
                q, k, v, a, b, A_log, dt_bias, pre_state, use_kernel = args
                conv_keep = conv_input.shape[1] - 8
                item.cache[0] = mx.contiguous(conv_input[:, keep:keep + conv_keep, :])
                _, state = gated_delta_update(q[:, :keep], k[:, :keep], v[:, :keep],
                                              a[:, :keep], b[:, :keep], A_log,
                                              dt_bias, pre_state, None,
                                              use_kernel=use_kernel)
                item.cache[1] = state
            rejected = 8 - keep
            for item in cache:
                if not hasattr(item, "cache") and hasattr(item, "trim") and rejected:
                    item.trim(rejected)

        branches = []
        for accepted in (0, 1):
            keep = accepted + 1
            prod_cache = clone_cache(frozen_root)
            hid, logits, fused, collector = target_forward(prod_cache, verify_ids)
            source_matched_chad_rollback(prod_cache, collector, keep)
            prod_drafter = engine_module._DFlashDrafter(
                eng, eng.model.language_model.model.embed_tokens,
                lambda h: eng.model.language_model.lm_head(h))
            prod_drafter.start_turn()
            prod_drafter.pending = initial_context
            orig_propose(prod_drafter, 7, initial_anchor, None)
            orig_reconcile(prod_drafter, 7, accepted, initial_anchor,
                           verify_ids[1:8], hid, fused)
            prod_post = boundary(row, "isolated_post_reconcile", prod_cache,
                                 input_ids=verify_ids[:keep], branch="production_chad_reconcile")

            oracle_cache = clone_cache(frozen_root)
            oracle_hid, oracle_logits, oracle_fused, oracle_target_collector = target_forward(
                oracle_cache, verify_ids)
            apply_chad_same_s8_oracle(oracle_cache, oracle_target_collector, keep)
            oracle_post = boundary(row, "isolated_post_reconcile_oracle", oracle_cache,
                                   input_ids=verify_ids[:keep], branch="independent_same_s8_oracle")
            if not set(prod_post["branch"]["cache_object_ids"]).isdisjoint(
                    oracle_post["branch"]["cache_object_ids"]):
                raise AssertionError("Chad C production/oracle caches overlap")
            c_state_equal = (prod_post["branch"]["cache_state_id"] ==
                             oracle_post["branch"]["cache_state_id"])
            c_structure_equal = (cache_semantic_signature(prod_post["cache"]) ==
                                 cache_semantic_signature(oracle_post["cache"]))
            prod_accepted_fused = array_record(fused[:, :keep])
            oracle_accepted_fused = array_record(oracle_fused[:, :keep])
            c_fused_equal = prod_accepted_fused["sha256"] == oracle_accepted_fused["sha256"]

            logits_np = np.asarray(logits[0].astype(mx.float32))
            next_token = int(np.argmax(logits_np[keep - 1]))
            d_rows = []
            for source, base in (("production", prod_cache), ("same_s8_oracle", oracle_cache)):
                dcache = clone_cache(base)
                before = boundary(row, "D_before", dcache, input_ids=[next_token],
                                  branch=f"D_{source}")
                _, dlogits, dfused, _ = target_forward(dcache, [next_token])
                d_np = np.asarray(dlogits[0].astype(mx.float32))
                top = np.argsort(d_np[-1])[-2:][::-1]
                d_rows.append({"source": source, "input_token": next_token,
                               "logical_position": logical_position + keep,
                               "output_token": int(top[0]), "top2_ids": top.tolist(),
                               "top2_values": d_np[-1, top].tolist(),
                               "top2_margin": float(d_np[-1, top[0]] - d_np[-1, top[1]]),
                               "before": before,
                               "after": boundary(row, "D_after", dcache,
                                                  input_ids=[next_token], branch=f"D_{source}"),
                               "outputs": logits_record(dlogits, dfused)})

            if not set(d_rows[0]["before"]["branch"]["cache_object_ids"]).isdisjoint(
                    d_rows[1]["before"]["branch"]["cache_object_ids"]):
                raise AssertionError("Chad D production/oracle caches are not independent")

            d_root_id = oracle_post["branch"]["cache_state_id"]
            def make_d_pair(repetition, control_root_id):
                pair = []
                for _side in ("left", "right"):
                    control_cache = clone_cache(oracle_cache)
                    _, control_logits, control_fused, _ = target_forward(control_cache, [next_token])
                    after = boundary(row, "D_control_after", control_cache,
                                     input_ids=[next_token], branch=f"D_control_{repetition}")
                    pair.append({"outputs": logits_record(control_logits, control_fused),
                                 "post_execute": {"cache": after["cache"]}})
                return pair
            d_controls = transient_pair_controls(root_id=d_root_id, component_prefix="",
                make_pair=make_d_pair, run_dir=_STORE.run_dir, store=_STORE)
            d_prod, d_oracle = d_rows
            d_comparison = compare_controlled_vectors(
                {"outputs": d_prod["outputs"], "post_execute": {"cache": d_prod["after"]["cache"]}},
                {"outputs": d_oracle["outputs"], "post_execute": {"cache": d_oracle["after"]["cache"]}},
                d_controls, _STORE.run_dir, expected_root_id=d_root_id)
            d_distances = {item["component"]: item.get("distance")
                           for item in d_comparison["numeric_comparisons"]}
            d_top2_exact = d_prod["top2_ids"] == d_oracle["top2_ids"]
            d_margin_distances = [
                abs(float(d_prod["top2_margin"]) - float(d_oracle["top2_margin"]))
            ]
            d_margin_within = (d_controls["margin_repeatable"] and all(
                value <= d_controls["margin_bound"] for value in d_margin_distances))
            d_status = ("inconclusive" if not d_controls["top2_ids_repeatable"] or
                        not d_controls["margin_repeatable"] else d_comparison["status"])
            if d_status != "inconclusive" and (not d_top2_exact or not d_margin_within):
                d_status = "fail"
            d_structure_exact = (cache_semantic_signature(d_prod["after"]["cache"]) ==
                                 cache_semantic_signature(d_oracle["after"]["cache"]) and
                                 d_prod["after"]["position"] == d_oracle["after"]["position"])
            if not d_structure_exact and d_status == "pass":
                d_status = "fail"

            # E production uses a brand-new Chad drafter and its real propose /
            # reconcile staging. The independent branch rebuilds projected rows
            # on a second cache using primitive projection + append calls only.
            anchor = next_token
            accepted_absolute_start = row["pre_verify"]["position"]["next_target_absolute_position"]
            prod_e = engine_module._DFlashDrafter(
                eng, eng.model.language_model.model.embed_tokens,
                lambda h: eng.model.language_model.lm_head(h))
            prod_e.start_turn()
            prod_e.pending = initial_context
            source_first_ids, _ = orig_propose(prod_e, 7, initial_anchor, None)
            source_first_ids = [int(v) for v in source_first_ids]
            source_proposal = e_source_proposal_adjudication(verify_ids[1:], source_first_ids)
            source_proposal_exact = source_proposal["source_proposal_exact"]
            orig_reconcile(prod_e, 7, accepted, initial_anchor,
                           verify_ids[1:8], hid, fused)
            prod_next, _ = orig_propose(prod_e, 7, anchor, None)
            prod_next = [int(v) for v in prod_next]
            ref_e = engine_module._DFlashDrafter(
                eng, eng.model.language_model.model.embed_tokens,
                lambda h: eng.model.language_model.lm_head(h))
            ref_e.start_turn()
            ref_state_projected_prompt = ref_e.m.project_ctx(initial_context)
            ref_state_projected_accepted = ref_e.m.project_ctx(oracle_fused[:, :keep])
            mx.eval(ref_state_projected_prompt, ref_state_projected_accepted)
            ref_e.m.append_ctx(ref_state_projected_prompt, ref_e.cache)
            ref_e.m.append_ctx(ref_state_projected_accepted, ref_e.cache)

            ref_prop = engine_module._DFlashDrafter(
                eng, eng.model.language_model.model.embed_tokens,
                lambda h: eng.model.language_model.lm_head(h))
            ref_prop.start_turn()
            ref_first_block = mx.array(
                [[initial_anchor] + [ref_prop._mask] * (eng._dflash.config.block_size - 1)],
                dtype=mx.uint32)
            ref_next_block = mx.array(
                [[anchor] + [ref_prop._mask] * (eng._dflash.config.block_size - 1)],
                dtype=mx.uint32)
            ref_first_ids = ref_prop.m.select_block(
                ref_first_block, initial_context, ref_prop.cache,
                cap=7, anchor_id=initial_anchor)[0]
            mx.eval(ref_first_ids)
            ref_first_ids = [int(v) for v in ref_first_ids]
            independent_source_proposal_exact = ref_first_ids == verify_ids[1:]
            ref_next = ref_prop.m.select_block(
                ref_next_block, oracle_fused[:, :keep], ref_prop.cache,
                cap=7, anchor_id=anchor)[0]
            mx.eval(ref_next)
            ref_next = [int(v) for v in ref_next]
            prod_projected_prompt = prod_e.m.project_ctx(initial_context)
            prod_projected_accepted = prod_e.m.project_ctx(fused[:, :keep])
            ref_projected_prompt = ref_e.m.project_ctx(initial_context)
            ref_projected_accepted = ref_e.m.project_ctx(oracle_fused[:, :keep])
            mx.eval(prod_projected_prompt, prod_projected_accepted,
                    ref_projected_prompt, ref_projected_accepted)
            projected_rows_exact = (
                np.array_equal(mlx_numpy_value(prod_projected_prompt), mlx_numpy_value(ref_projected_prompt)) and
                np.array_equal(mlx_numpy_value(prod_projected_accepted), mlx_numpy_value(ref_projected_accepted)))
            e_projection_distances = {
                "max_abs_prompt": float(np.max(np.abs(mlx_numpy_value(prod_projected_prompt).astype(np.float64) -
                                                      mlx_numpy_value(ref_projected_prompt).astype(np.float64)))),
                "max_abs_accepted": float(np.max(np.abs(mlx_numpy_value(prod_projected_accepted).astype(np.float64) -
                                                        mlx_numpy_value(ref_projected_accepted).astype(np.float64))))}
            prod_e_state = cache_record(prod_e.cache, domain="drafter")
            ref_e_state = cache_record(ref_e.cache, domain="drafter")
            e_structure_equal = cache_semantic_signature(prod_e_state) == cache_semantic_signature(ref_e_state)
            e_root_id = stable_id({"prompt": row["pending_context"]["sha256"],
                "accepted": oracle_accepted_fused["sha256"], "logical_position": logical_position,
                "keep": keep, "anchor": anchor})
            def make_e_pair(repetition, control_root_id):
                pair = []
                for _side in ("left", "right"):
                    control_e = engine_module._DFlashDrafter(
                        eng, eng.model.language_model.model.embed_tokens,
                        lambda h: eng.model.language_model.lm_head(h))
                    control_e.start_turn()
                    projected_prompt = control_e.m.project_ctx(initial_context)
                    projected_accepted = control_e.m.project_ctx(oracle_fused[:, :keep])
                    mx.eval(projected_prompt, projected_accepted)
                    control_e.m.append_ctx(projected_prompt, control_e.cache)
                    control_e.m.append_ctx(projected_accepted, control_e.cache)

                    control_prop = engine_module._DFlashDrafter(
                        eng, eng.model.language_model.model.embed_tokens,
                        lambda h: eng.model.language_model.lm_head(h))
                    control_prop.start_turn()
                    control_first_block = mx.array(
                        [[initial_anchor] + [control_prop._mask] *
                         (eng._dflash.config.block_size - 1)], dtype=mx.uint32)
                    control_next_block = mx.array(
                        [[anchor] + [control_prop._mask] *
                         (eng._dflash.config.block_size - 1)], dtype=mx.uint32)
                    control_first = control_prop.m.select_block(
                        control_first_block, initial_context, control_prop.cache,
                        cap=7, anchor_id=initial_anchor)[0]
                    mx.eval(control_first)
                    control_first = [int(v) for v in control_first]
                    if control_first != verify_ids[1:]:
                        raise AssertionError("Chad E control failed to reproduce the source first proposal")
                    control_next = control_prop.m.select_block(
                        control_next_block, oracle_fused[:, :keep], control_prop.cache,
                        cap=7, anchor_id=anchor)[0]
                    mx.eval(control_next)
                    control_next = [int(v) for v in control_next]
                    pair.append({"proposal_ids": control_next,
                        "outputs": {"projected_prompt": array_record(projected_prompt),
                                    "projected_accepted": array_record(projected_accepted)},
                        "post_execute": {"cache": cache_record(control_e.cache, domain="drafter")}})
                return pair
            e_controls = transient_pair_controls(root_id=e_root_id, component_prefix="",
                make_pair=make_e_pair, run_dir=_STORE.run_dir, store=_STORE,
                require_top2=False)
            e_reference_root_id = stable_id({
                "prompt_context_sha256": hashlib.sha256(mlx_numpy_value(ref_projected_prompt).tobytes()).hexdigest(),
                "accepted_context_sha256": hashlib.sha256(mlx_numpy_value(ref_projected_accepted).tobytes()).hexdigest(),
                "logical_position": logical_position, "accepted_count": accepted,
                "next_anchor": anchor})
            if not set(mutable_cache_object_ids(prod_e.cache)).isdisjoint(
                    mutable_cache_object_ids(ref_e.cache)):
                raise AssertionError("Chad E production/reference drafter caches overlap")
            e_prod_view = {"outputs": {"projected_prompt": array_record(prod_projected_prompt),
                                        "projected_accepted": array_record(prod_projected_accepted)},
                           "post_execute": {"cache": prod_e_state}}
            e_ref_view = {"outputs": {"projected_prompt": array_record(ref_projected_prompt),
                                       "projected_accepted": array_record(ref_projected_accepted)},
                          "post_execute": {"cache": ref_e_state}}
            e_numeric_comparison = compare_controlled_vectors(e_prod_view, e_ref_view,
                e_controls, _STORE.run_dir, expected_root_id=e_root_id)
            e_numeric_status = ("inconclusive" if not source_proposal_exact or
                                not independent_source_proposal_exact or
                                not e_controls["proposal_ids_repeatable"] else
                                "fail" if prod_next != ref_next or not e_structure_equal else
                                e_numeric_comparison["status"])
            provenance = required_gate_branch_provenance(
                runtime="chad", frozen_root_state_id=root_id,
                frozen_root_observation_id=root_observation_id,
                branch_id=f"{args.run_id}:isolated:accepted{accepted}",
                logical_position=logical_position, verify_ids=verify_ids,
                accepted_count=accepted, b2_control_set_id=control_set_id)
            branches.append({"branch_kind": "isolated_forced",
                "acceptance_source": "forced_state_transition_test_input",
                "accepted_count": accepted, "keep": keep,
                "provenance": provenance,
                "c": {"production": prod_post, "independent_same_s8_oracle": oracle_post,
                      "source_matched_inline_target_rollback": True,
                      "production_accepted_fused": prod_accepted_fused,
                      "oracle_accepted_fused": oracle_accepted_fused,
                      "exact_accepted_fused_equal": c_fused_equal,
                      "accepted_prefix_ids": verify_ids[:keep],
                      "accepted_prefix_exact": prod_post["input_ids"] == verify_ids[:keep],
                      "exact_cache_state_equal": c_state_equal,
                      "exact_structure_equal": c_structure_equal,
                      "cache_objects_disjoint": True},
                "d": {"next_committed_token": next_token, "evaluations": d_rows,
                      "same_input_token": True, "five_same_root_width1_controls": d_controls,
                      "control_contract": {"root_state_id": oracle_post["branch"]["cache_state_id"],
                                           "width": 1, "repetitions": 5,
                                           "calibration_repetitions": [1, 2, 3],
                                           "validation_repetitions": [4, 5],
                                           "predeclared_components": d_controls["component_set"],
                                           "metrics": "component max_abs; exact structure and discrete IDs"},
                      "control_results": d_controls,
                      "production_oracle_numeric_distance": d_distances,
                      "production_oracle_numeric_comparison": d_comparison,
                      "top2_ids_exact": d_top2_exact,
                      "margin_distances": d_margin_distances,
                      "margin_within_control_bound": d_margin_within,
                      "resulting_structure_exact": d_structure_exact,
                      "numeric_status": d_status,
                      "fresh_s1_role": "serial_semantic_diagnostic_only",
                      "s1_reference_pointer": {
                          "keep": keep,
                          "observation_id": row["references"]["s1_keep1" if keep == 1 else "s1_keep2"]["pre_verify"]["observation_id"],
                          "cache_state_id": row["references"]["s1_keep1" if keep == 1 else "s1_keep2"]["pre_verify"]["branch"]["cache_state_id"]}},
                "e": {"initial_prompt_context": array_record(initial_context),
                      "source_reconstruction": e_source_contract,
                      "reconstructed_first_proposal_ids": source_first_ids,
                      "independent_reconstructed_first_proposal_ids": ref_first_ids,
                      "source_first_proposal_ids": verify_ids[1:],
                      "source_proposal": source_proposal,
                      "source_proposal_exact": source_proposal_exact,
                      "independent_source_proposal_exact": independent_source_proposal_exact,
                      "accepted_context_rows": array_record(oracle_fused[:, :keep]),
                      "logical_position": logical_position,
                      "accepted_logical_positions": list(range(logical_position, logical_position + keep)),
                      "accepted_absolute_positions": list(range(accepted_absolute_start,
                                                                 accepted_absolute_start + keep)),
                      "projected_prompt_rows": array_record(ref_projected_prompt),
                      "projected_accepted_rows": array_record(ref_projected_accepted),
                      "production_projected_prompt_rows": array_record(prod_projected_prompt),
                      "production_projected_accepted_rows": array_record(prod_projected_accepted),
                      "production_drafter_cache": prod_e_state,
                      "independent_drafter_cache": ref_e_state,
                      "drafter_caches_disjoint": True, "next_anchor": anchor,
                      "production_next_proposal_ids": prod_next,
                      "independent_next_proposal_ids": ref_next,
                      "proposal_ids_equal": prod_next == ref_next if source_proposal_exact else None,
                      "numeric_status": e_numeric_status,
                      "drafter_structure_equal": e_structure_equal,
                      "projected_context_rows_exact": projected_rows_exact,
                      "next_block_ids": [anchor] + [prod_e._mask] *
                                        (eng._dflash.config.block_size - 1),
                      "equivalent_logical_proposal_point": (
                          source_proposal_exact and independent_source_proposal_exact),
                      "independent_projection_append": True,
                      "five_independent_same_runtime_controls": e_controls,
                      "projection_numeric_distance": e_projection_distances,
                      "control_contract": {"reference_root_state_id": e_root_id,
                                           "repetitions": 5,
                                           "calibration_repetitions": [1, 2, 3],
                                           "validation_repetitions": [4, 5],
                                           "predeclared_components": e_controls["component_set"],
                                           "metrics": "component max_abs; proposal IDs exact"},
                      "projection_control_bound": e_controls["components"],
                      "drafter_cache_numeric_comparison": e_numeric_comparison,
                      "controls_exact_stable": e_controls["status"] == "STABLE"}})
        branches = link_shared_b2_control_set(branches)
        return required_gate_acquisition_record(
                row["observation_id"],
                {"root_state_id": root_id, "root_observation_id": root_observation_id,
                       "logical_position": logical_position, "verify_ids": verify_ids,
                       "control_set_id": control_set_id, "control_count": 5,
                       "control_component_count": len(controls),
                       "reference": b2_reference["s8_verify"], "controls": controls},
                branches)

    rng_seeds = []
    orig_urandom = os.urandom
    def traced_urandom(count):
        value = orig_urandom(count)
        if count == 4:
            rng_seeds.append(int.from_bytes(value, "little"))
        return value
    os.urandom = traced_urandom
    try:
        _, stats = eng.generate(ids, args.max_tokens, None, [])
        doc["sampling"]["os_urandom_4byte_seeds"] = rng_seeds
        doc["output_ids"] = list(stats.gen_ids or [])
        doc["num_rounds"] = stats.forwards
        doc["historical_trace_measurement"] = historical_trace_measurement(doc)
        (_STORE.run_dir / "continuous.json").write_text(
            json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        doc["r1_context_reference"] = chad_r1_context_reference(rounds[0])
        s8_rows = [row for row in rounds if row.get("target_sequence_width") == 8
                   and row.get("round_index") in frozen_s8_roots]
        required_source = rounds[0] if args.measure_required_gates and rounds else None
        if args.measure_required_gates and (required_source is None or
                required_source.get("target_sequence_width") != 8 or
                required_source.get("round_index") not in frozen_s8_roots):
            raise RuntimeError("required-gates E currently requires the first logical speculative round to be a valid S=8/seven-draft round; later S=8 roots are not reconstructable")
        for row in s8_rows:
            root = frozen_s8_roots[row["round_index"]]
            reference_root = clone_cache(root) if args.measure_required_gates else root
            prior_measure_controls = args.measure_controls
            if args.measure_required_gates and row is not required_source:
                args.measure_controls = False
            try:
                row["references"] = chad_references(
                    eng.model.language_model.model, (), {"cache": None}, row,
                    reference_root)
                if not args.measure_required_gates:
                    frozen_s8_roots.pop(row["round_index"])
            finally:
                args.measure_controls = prior_measure_controls
        if args.measure_required_gates:
            doc["required_gate_acquisition"] = acquire_required_gates(
                required_source, frozen_s8_roots[required_source["round_index"]],
                required_source["references"])
    finally:
        os.urandom = orig_urandom
        engine_module._DFlashDrafter.propose = orig_propose
        engine_module._DFlashDrafter.reconcile = orig_reconcile
        model_class.__call__ = orig_model_call
    doc["output_ids"] = list(stats.gen_ids or [])
    doc["num_rounds"] = stats.forwards
    return doc


def finalize_gate_records(doc):
    rounds = doc["rounds"]
    if rounds:
        doc["prefix_state_id"] = rounds[0]["pre_verify"]["branch"]["cache_state_id"]
        doc["observed_target_cache_classes"] = [
            row["class"] for row in rounds[0]["pre_verify"]["cache"]]
        doc["target_kv_observed"] = [
            {"layer": row["layer"], "class": row["class"],
             "bits": row.get("bits"), "group_size": row.get("group_size"),
             "live_length": row.get("live_length"), "absolute_range": row.get("absolute_range")}
            for row in rounds[0]["pre_verify"]["cache"]
            if (row.get("semantic") or {}).get("kind") == "attention"]
    by_gate = {row["gate"]: row for row in doc["gate_results"]}
    by_gate["A"]["observation_ids"] = [row["proposal_input_record"]["observation_id"]
                                          for row in rounds if "proposal_input_record" in row]
    by_gate["A"]["classification"] = "cross_runtime_comparison_pending"
    if doc.get("r1_context_reference"):
        by_gate["A"]["same_runtime_r1_context"] = doc["r1_context_reference"]["comparison"]
        by_gate["A"]["observation_ids"].append(
            doc["r1_context_reference"]["reference"]["observation_id"])
    s8_rows = [row for row in rounds if row.get("target_sequence_width") == 8]
    by_gate["B1"]["observation_ids"] = [row["pre_verify"]["observation_id"]
                                           for row in s8_rows]
    by_gate["B2"]["observation_ids"] = [row["post_verify"]["observation_id"]
                                           for row in s8_rows]
    for row in s8_rows:
        branch_set = row["references"]
        reference = branch_set["s8_verify"]
        speculative = row["pre_verify"]["branch"]
        frozen = row["s8_frozen_pre_verify"]
        independent = (row["s8_frozen_independence"]["disjoint"] and
                       set(reference["pre_verify"]["branch"]["cache_object_ids"]).isdisjoint(
                           speculative["cache_object_ids"]))
        same_state = (reference["pre_verify"]["branch"]["cache_state_id"] ==
                      frozen["branch"]["cache_state_id"] == speculative["cache_state_id"])
        same_structure = (cache_semantic_signature(reference["pre_verify"]["cache"]) ==
                          cache_semantic_signature(row["pre_verify"]["cache"]))
        same_ids_positions = (reference["pre_verify"]["input_ids"] ==
                              frozen["input_ids"] == row["verify_ids"] and
                              reference["pre_verify"]["position"] ==
                              frozen["position"] == row["pre_verify"]["position"])
        status = "pass" if all((independent, same_state, same_structure,
                                same_ids_positions)) else "fail"
        row["b1_local_exact"] = {
            "status": status,
            "same_verify_ids_and_positions": same_ids_positions,
            "same_preverify_state_id": same_state,
            "same_preverify_semantic_structure": same_structure,
            "distinct_mutable_cache_objects": independent,
            "reference_observation_id": reference["pre_verify"]["observation_id"],
            "frozen_snapshot_observation_id": frozen["observation_id"],
            "committed_replay_observation_id": branch_set["fresh_replay"]["observation_id"],
            "committed_replay_numeric_state_id_equal_to_speculative": branch_set[
                "committed_replay_numeric_state_id_equal_to_speculative"],
            "gate_result_pending_A": True,
        }
    if s8_rows and all(row["b1_local_exact"]["status"] == "pass" for row in s8_rows):
        by_gate["B1"]["classification"] = "local_exact_precondition_observed; Gate A pending"
    elif any(row["b1_local_exact"]["status"] == "fail" for row in s8_rows):
        by_gate["B1"]["classification"] = "local_exact_precondition_mismatch; Gate A pending"
        by_gate["B1"]["missing_control"] = "repair frozen S8 ID/state/position or cache independence"
    else:
        by_gate["B1"]["classification"] = "exact frozen S8 preverify control missing or mismatched"
        by_gate["B1"]["missing_control"] = "exact frozen S8 branch and matching verify IDs/state"
    for gate in ("C", "D", "E"):
        by_gate[gate]["observation_ids"] = [row["post_reconcile"]["observation_id"]
                                              for row in rounds if "post_reconcile" in row]
    return doc


def write_run_manifest(run_dir, run_id):
    records = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file():
            records.append({"path": str(path.relative_to(run_dir)), "size": path.stat().st_size,
                            "sha256": sha256(path)})
    manifest_dir = Path(__file__).resolve().parent.parent / "manifests"
    manifest_dir.mkdir(exist_ok=True)
    target = manifest_dir / f"{run_id}.json"
    with target.open("x", encoding="utf-8") as handle:
        json.dump({"run_id": run_id, "run_dir": str(run_dir), "files": records},
                  handle, indent=2)
        handle.write("\n")


def validate_probe_args(ap, args):
    """Validate acquisition-mode constraints before output setup/model loading."""
    if args.measure_required_gates and not args.measure_controls:
        ap.error("--measure-required-gates requires --measure-controls")
    if args.measure_required_gates and args.align_prefill:
        ap.error("--measure-required-gates requires production semantics; --align-prefill is diagnostic only")
    if args.measure_required_gates and args.chad_no_fastpath:
        ap.error("--measure-required-gates requires Chad's normal installed fastpath")
    if args.measure_required_gates and args.runtime == "mlx-dspark" and args.mlx_kv_bits not in (None, 0):
        ap.error("--measure-required-gates requires plain target KV for mlx-dspark")
    if args.measure_required_gates and args.runtime == "chad" and args.chad_kv_bits not in (None, 0):
        ap.error("--measure-required-gates requires plain target KV for Chad")


def main():
    global _STORE
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", choices=("mlx-dspark", "chad"), required=True)
    runs_root = Path(__file__).resolve().parent.parent / "runs"
    ap.add_argument("--output-dir", default=None, help="unused project-local run directory; defaults to runs/<unique-run-id>")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--align-prefill", action="store_true")
    ap.add_argument("--chad-kv-bits", type=int, default=None)
    ap.add_argument("--chad-no-fastpath", action="store_true")
    ap.add_argument("--mlx-kv-bits", type=int, default=None)
    ap.add_argument("--measure-controls", action="store_true",
                    help="run exactly five fresh same-width and width-delta pairs; off by default")
    ap.add_argument("--measure-required-gates", action="store_true",
                    help="after continuous generation, acquire isolated forced keep=1/keep=2 gate branches")
    ap.add_argument("--min-free-gib", type=float, default=6.0,
                    help="refuse to load models when evidence storage is below this free-space floor")
    args = ap.parse_args()
    validate_probe_args(ap, args)
    out = Path(args.output_dir) if args.output_dir else runs_root / uuid.uuid4().hex
    if not out.is_absolute():
        out = Path.cwd() / out
    if not out.resolve().is_relative_to(runs_root.resolve()):
        ap.error(f"output directory must be inside {runs_root}")
    if out.resolve().parent != runs_root.resolve():
        ap.error(f"each execution needs a direct unique child of {runs_root}")
    free = shutil.disk_usage(runs_root).free
    if free < args.min_free_gib * 1024 ** 3:
        ap.error(f"only {free / 1024 ** 3:.2f} GiB free; need {args.min_free_gib:.2f} GiB before loading a model")
    out.mkdir(parents=True, exist_ok=False)
    args.run_id = out.name
    _STORE = EvidenceStore(out)
    path = out / "run.json"
    try:
        doc = run_mlx(args) if args.runtime == "mlx-dspark" else run_chad(args)
        finalize_gate_records(doc)
        path.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8", newline="\n")
    except Exception as exc:
        import traceback
        (out / "interrupted.json").write_text(json.dumps({
            "run_id": args.run_id, "status": "interrupted", "error_type": type(exc).__name__,
            "error": str(exc), "traceback": traceback.format_exc(),
        }, indent=2) + "\n", encoding="utf-8")
        raise
    finally:
        _STORE = None
        write_run_manifest(out, args.run_id)
    print(json.dumps({"path": str(path), "prompt_ids": doc["prompt_ids"],
                      "rounds": [{k: r.get(k) for k in ("observation_id", "anchor_or_pending_id", "draft_ids", "verify_ids", "accepted_count")}
                                 for r in doc["rounds"]],
                      "output_ids": doc["output_ids"][:16]}, indent=2))


if __name__ == "__main__":
    main()
