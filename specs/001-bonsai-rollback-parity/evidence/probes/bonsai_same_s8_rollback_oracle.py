from __future__ import annotations

from pathlib import Path
import hashlib
import io
import json
import math
import sys
import zlib

import mlx.core as mx
import numpy as np
from mlx_lm.models.gated_delta import gated_delta_update


def load_np(run_dir: Path, rec: dict) -> np.ndarray:
    raw = zlib.decompress((run_dir / rec["blob"]).read_bytes())
    return np.load(io.BytesIO(raw), allow_pickle=False)


def load_mx(run_dir: Path, rec: dict):
    arr = load_np(run_dir, rec)
    x = mx.array(arr)

    dtype = rec.get("dtype", "")
    if dtype.endswith("bfloat16"):
        x = x.astype(mx.bfloat16)
    elif dtype.endswith("float16"):
        x = x.astype(mx.float16)
    elif dtype.endswith("float32"):
        x = x.astype(mx.float32)
    elif dtype.endswith("int32"):
        x = x.astype(mx.int32)
    elif dtype.endswith("uint32"):
        x = x.astype(mx.uint32)

    return x


def as_np(x) -> np.ndarray:
    mx.eval(x)
    return np.asarray(x)


def digest(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.tobytes()).hexdigest()


def comparison(expected: np.ndarray, actual: np.ndarray) -> dict:
    same_shape = expected.shape == actual.shape
    if not same_shape:
        return {
            "shape_equal": False,
            "expected_shape": list(expected.shape),
            "actual_shape": list(actual.shape),
            "exact": False,
            "max_abs": math.inf,
        }

    exact = np.array_equal(expected, actual)
    max_abs = (
        float(np.max(np.abs(
            expected.astype(np.float64) - actual.astype(np.float64)
        )))
        if expected.size else 0.0
    )

    return {
        "shape_equal": True,
        "expected_shape": list(expected.shape),
        "actual_shape": list(actual.shape),
        "exact": bool(exact),
        "max_abs": max_abs,
        "expected_sha256_raw": digest(expected),
        "actual_sha256_raw": digest(actual),
    }


def replay_gdn(run_dir: Path, cap: dict, keep: int) -> np.ndarray:
    q = load_mx(run_dir, cap["q"])[:, :keep]
    k = load_mx(run_dir, cap["k"])[:, :keep]
    v = load_mx(run_dir, cap["v"])[:, :keep]
    a = load_mx(run_dir, cap["a"])[:, :keep]
    b = load_mx(run_dir, cap["b"])[:, :keep]

    A_log = load_mx(run_dir, cap["A_log"])
    dt_bias = load_mx(run_dir, cap["dt_bias"])
    pre_state = load_mx(run_dir, cap["pre_state"])

    mask_rec = cap.get("mask")
    mask = None
    if isinstance(mask_rec, dict) and "blob" in mask_rec:
        mask = load_mx(run_dir, mask_rec)
        mask = mask[:, :keep]

    _, new_state = gated_delta_update(
        q, k, v, a, b,
        A_log, dt_bias,
        pre_state,
        mask,
        use_kernel=bool(cap.get("use_kernel", True)),
    )

    return as_np(new_state)


def normalize_gdn_captures(value: object) -> list[dict]:
    captures: list[dict] = []

    def visit(node: object) -> None:
        if isinstance(node, dict):
            if (
                isinstance(node.get("model_layer"), int)
                and isinstance(node.get("gdn_order"), int)
                and all(
                    key in node
                    for key in (
                        "q", "k", "v", "a", "b",
                        "A_log", "dt_bias", "pre_state", "conv_input"
                    )
                )
            ):
                captures.append(node)
                return

            for child in node.values():
                if isinstance(child, (dict, list, tuple)):
                    visit(child)

        elif isinstance(node, (list, tuple)):
            for child in node:
                visit(child)

    visit(value)

    captures.sort(
        key=lambda row: (
            row.get("gdn_order", 10**9),
            row.get("model_layer", 10**9),
        )
    )

    identities = [
        (row["gdn_order"], row["model_layer"])
        for row in captures
    ]

    if len(identities) != len(set(identities)):
        raise AssertionError(
            f"duplicate normalized GDN captures: {identities}"
        )

    if len(captures) != 48:
        raise AssertionError(
            f"expected 48 normalized GDN captures, got {len(captures)}; "
            f"identities={identities}"
        )

    return captures


def capture_schema(value: object) -> dict:
    def describe(node: object, depth: int = 0) -> object:
        if depth >= 4:
            if isinstance(node, dict):
                return {"type": "dict", "keys": sorted(node)[:30]}
            if isinstance(node, (list, tuple)):
                return {"type": type(node).__name__, "length": len(node)}
            return {"type": type(node).__name__}

        if isinstance(node, dict):
            if "model_layer" in node or "gdn_order" in node:
                return {
                    "type": "capture",
                    "model_layer": node.get("model_layer"),
                    "gdn_order": node.get("gdn_order"),
                    "keys": sorted(node),
                }
            return {
                "type": "dict",
                "keys": sorted(node),
                "children": {
                    str(k): describe(v, depth + 1)
                    for k, v in list(node.items())[:8]
                    if isinstance(v, (dict, list, tuple))
                },
            }

        if isinstance(node, (list, tuple)):
            return {
                "type": type(node).__name__,
                "length": len(node),
                "items": [
                    describe(v, depth + 1)
                    for v in list(node)[:3]
                ],
            }

        return {"type": type(node).__name__}

    return describe(value)


def oracle_round(run_dir: Path, row: dict) -> dict:
    keep = row["accepted_count"] + 1
    verify_width = len(row["verify_ids"])

    refs = row["references"]
    s8 = refs["s8_verify"]
    captures_raw = s8["gdn_capture"]
    captures = normalize_gdn_captures(captures_raw)

    prod_cache = {
        item["layer"]: item
        for item in row["post_reconcile"]["cache"]
    }

    s8_pre_cache = {
        item["layer"]: item
        for item in s8["pre_verify"]["cache"]
    }

    s8_post_cache = {
        item["layer"]: item
        for item in s8["post_execute"]["cache"]
    }

    gdn_results = []

    for cap in captures:
        layer = cap["model_layer"]
        prod = prod_cache[layer]

        conv_input = load_np(run_dir, cap["conv_input"])
        n_conv_keep = conv_input.shape[1] - verify_width

        expected_conv = np.ascontiguousarray(
            conv_input[:, keep:keep + n_conv_keep, :]
        )
        actual_conv = load_np(run_dir, prod["convolution_window"])

        replay_results = [
            replay_gdn(run_dir, cap, keep)
            for _ in range(5)
        ]

        replay_hashes = [digest(x) for x in replay_results]
        replay_exact_repeatability = len(set(replay_hashes)) == 1
        expected_state = replay_results[0]
        actual_state = load_np(run_dir, prod["recurrent_state"])

        gdn_results.append({
            "layer": layer,
            "gdn_order": cap["gdn_order"],
            "keep": keep,
            "verify_width": verify_width,
            "conv_window": comparison(expected_conv, actual_conv),
            "recurrent_state": comparison(expected_state, actual_state),
            "oracle_replay_five_run": {
                "hashes": replay_hashes,
                "exact_repeatability": replay_exact_repeatability,
            },
        })

    attention_results = []

    for layer, post in sorted(s8_post_cache.items()):
        if post.get("semantic", {}).get("kind") != "attention":
            continue

        pre = s8_pre_cache[layer]
        prod = prod_cache[layer]

        pre_len = int(pre["live_length"])
        expected_len = pre_len + keep

        fields = {}

        for field in ("keys", "values"):
            full = load_np(run_dir, post[field])
            expected = np.ascontiguousarray(
                full[..., :expected_len, :]
            )
            actual = load_np(run_dir, prod[field])
            fields[field] = comparison(expected, actual)

        expected_positions = list(range(expected_len))

        metadata = {
            "live_length": prod.get("live_length") == expected_len,
            "offset": prod.get("offset") == expected_len,
            "absolute_range": prod.get("absolute_range") == [0, expected_len],
            "absolute_positions": prod.get("absolute_positions") == expected_positions,
        }

        attention_results.append({
            "layer": layer,
            "keep": keep,
            "pre_live_length": pre_len,
            "expected_live_length": expected_len,
            "metadata": metadata,
            **fields,
        })

    accepted_prefix_exact = (
        row["accepted_prefix_ids"]
        == row["verify_ids"][:keep]
    )

    gdn_exact = all(
        item["conv_window"]["exact"]
        and item["recurrent_state"]["exact"]
        and item["oracle_replay_five_run"]["exact_repeatability"]
        for item in gdn_results
    )

    attention_exact = all(
        item["keys"]["exact"]
        and item["values"]["exact"]
        and all(item["metadata"].values())
        for item in attention_results
    )

    fused_exact = (
        row["verify"]["fused"]["sha256"]
        == s8["outputs"][0]["fused"]["sha256"]
    )

    return {
        "round_index": row["round_index"],
        "generated_before": row["generated_before"],
        "accepted_count": row["accepted_count"],
        "keep": keep,
        "verify_width": verify_width,
        "verify_ids": row["verify_ids"],
        "accepted_prefix_ids": row["accepted_prefix_ids"],
        "accepted_prefix_exact": accepted_prefix_exact,
        "same_width_s8_fused_full_hash_equal": fused_exact,
        "gdn": {
            "count": len(gdn_results),
            "all_exact": gdn_exact,
            "first_nonexact": [
                item for item in gdn_results
                if not (
                    item["conv_window"]["exact"]
                    and item["recurrent_state"]["exact"]
                    and item["oracle_replay_five_run"]["exact_repeatability"]
                )
            ][:10],
            "layers": gdn_results,
        },
        "attention": {
            "count": len(attention_results),
            "all_exact": attention_exact,
            "first_nonexact": [
                item for item in attention_results
                if not (
                    item["keys"]["exact"]
                    and item["values"]["exact"]
                    and all(item["metadata"].values())
                )
            ][:10],
            "layers": attention_results,
        },
        "same_s8_rollback_oracle": (
            "PASS"
            if accepted_prefix_exact and fused_exact and gdn_exact and attention_exact
            else "FAIL"
        ),
    }


def run_one(run_dir: Path) -> dict:
    doc = json.loads((run_dir / "run.json").read_text())

    selected = [
        row for row in doc["rounds"]
        if row["accepted_count"] in (0, 1)
        and row.get("references", {}).get("s8_verify")
    ]

    results = [oracle_round(run_dir, row) for row in selected]

    return {
        "run_id": doc["run_id"],
        "implementation": doc.get("implementation"),
        "align_prefill_diagnostic": doc.get("align_prefill_diagnostic"),
        "kv_bits": doc.get("kv_bits"),
        "output_ids": doc.get("output_ids"),
        "rounds": results,
        "accepted_counts_covered": sorted(
            {row["accepted_count"] for row in selected}
        ),
        "all_selected_rounds_pass": (
            bool(results)
            and all(
                row["same_s8_rollback_oracle"] == "PASS"
                for row in results
            )
        ),
    }


def main():
    output = Path(sys.argv[1])
    runs = [Path(x) for x in sys.argv[2:]]

    run_results = []

    for run in runs:
        try:
            result = run_one(run)
            result["oracle_execution_status"] = "PASS"
            run_results.append(result)
        except Exception as exc:
            diagnostic = {
                "run_dir": str(run),
                "oracle_execution_status": "ERROR",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

            try:
                doc = json.loads((run / "run.json").read_text())
                selected = [
                    row for row in doc["rounds"]
                    if row["accepted_count"] in (0, 1)
                    and row.get("references", {}).get("s8_verify")
                ]
                diagnostic["run_id"] = doc.get("run_id")
                diagnostic["implementation"] = doc.get("implementation")
                diagnostic["selected_rounds"] = [
                    {
                        "round_index": row.get("round_index"),
                        "accepted_count": row.get("accepted_count"),
                        "gdn_capture_schema": capture_schema(
                            row["references"]["s8_verify"].get("gdn_capture")
                        ),
                    }
                    for row in selected
                ]
            except Exception as diag_exc:
                diagnostic["diagnostic_error"] = (
                    f"{type(diag_exc).__name__}: {diag_exc}"
                )

            run_results.append(diagnostic)

    report = {
        "oracle": (
            "Independent post-reconcile oracle derived from the frozen "
            "same-runtime S=8 B2 reference capture. It never calls production "
            "rollback/reconcile and never uses fresh S1/S2 projection numerics "
            "as the numeric oracle."
        ),
        "runs": run_results,
    }

    output.write_text(json.dumps(report, indent=2) + "\n")

    compact = {
        "runs": [
            {
                "run_id": run.get("run_id"),
                "implementation": run.get("implementation"),
                "oracle_execution_status": run.get("oracle_execution_status"),
                "accepted_counts_covered": run.get("accepted_counts_covered"),
                "all_selected_rounds_pass": run.get("all_selected_rounds_pass"),
                "error": run.get("error"),
                "rounds": [
                    {
                        "round": row["round_index"],
                        "accepted": row["accepted_count"],
                        "keep": row["keep"],
                        "gdn_all_exact": row["gdn"]["all_exact"],
                        "attention_all_exact": row["attention"]["all_exact"],
                        "fused_s8_exact": row["same_width_s8_fused_full_hash_equal"],
                        "result": row["same_s8_rollback_oracle"],
                    }
                    for row in run.get("rounds", [])
                ],
            }
            for run in report["runs"]
        ]
    }

    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
