from __future__ import annotations

from pathlib import Path
import io
import json
import math
import sys
import zlib

import numpy as np


run_dir = Path(sys.argv[1])
run_path = run_dir / "run.json"
doc = json.loads(run_path.read_text())


def tensor_records(value: object, path: str = "") -> dict[str, dict]:
    out: dict[str, dict] = {}

    def visit(x: object, p: str) -> None:
        if isinstance(x, dict):
            if {"blob", "sha256", "dtype"} <= set(x):
                out[p] = x
                return
            for key, item in x.items():
                visit(item, f"{p}.{key}" if p else str(key))
        elif isinstance(x, list):
            for index, item in enumerate(x):
                visit(item, f"{p}.{index}" if p else str(index))

    visit(value, path)
    return out


def load_record(record: dict) -> np.ndarray:
    raw = zlib.decompress((run_dir / record["blob"]).read_bytes())
    return np.load(io.BytesIO(raw), allow_pickle=False)


def record_shape(record: dict) -> tuple:
    return tuple(record.get("shape") or ())


def exact_record_compatible(left: dict, right: dict) -> bool:
    return (
        left.get("dtype") == right.get("dtype")
        and record_shape(left) == record_shape(right)
    )


def max_abs_records(left: dict, right: dict) -> float:
    if not exact_record_compatible(left, right):
        return math.inf

    if left.get("sha256") == right.get("sha256"):
        return 0.0

    a = load_record(left).astype(np.float64)
    b = load_record(right).astype(np.float64)

    if a.shape != b.shape:
        return math.inf

    return float(np.max(np.abs(a - b))) if a.size else 0.0


def derived_parent(component: str) -> tuple[str, tuple | None]:
    if ".fused.tap" in component and ".row" in component:
        parent, rest = component.split(".tap", 1)
        tap_text, row_text = rest.split(".row", 1)
        tap = int(tap_text)
        row = int(row_text)
        return parent, ("fused_tap_row", tap, row)

    if ".logits.row" in component:
        parent, row_text = component.rsplit(".row", 1)
        return parent, ("logits_row", int(row_text))

    return component, None


def max_abs_component(
    left_records: dict[str, dict],
    right_records: dict[str, dict],
    component: str,
) -> float:
    parent, derived = derived_parent(component)

    left = left_records.get(parent)
    right = right_records.get(parent)

    if left is None or right is None:
        return math.inf

    if not exact_record_compatible(left, right):
        return math.inf

    if left.get("sha256") == right.get("sha256"):
        return 0.0

    a = load_record(left).astype(np.float64)
    b = load_record(right).astype(np.float64)

    if a.shape != b.shape:
        return math.inf

    if derived is None:
        return float(np.max(np.abs(a - b))) if a.size else 0.0

    if derived[0] == "logits_row":
        row = derived[1]
        a = a[:, row, :]
        b = b[:, row, :]
        return float(np.max(np.abs(a - b))) if a.size else 0.0

    if derived[0] == "fused_tap_row":
        tap = derived[1]
        row = derived[2]
        taps = (5, 19, 33, 47, 61)

        if tap not in taps or a.ndim != 3 or a.shape[-1] % len(taps):
            return math.inf

        width = a.shape[-1] // len(taps)
        index = taps.index(tap)
        lo = index * width
        hi = (index + 1) * width

        a = a[:, row, lo:hi]
        b = b[:, row, lo:hi]
        return float(np.max(np.abs(a - b))) if a.size else 0.0

    return math.inf


def stable(control: dict | None) -> bool:
    if control is None or control.get("status") != "STABLE":
        return False

    samples = control.get("samples") or []
    if len(samples) != 5:
        return False

    bound = control.get("bound")
    if bound is None:
        return False

    return (
        bound == max(samples[:3])
        and all(value <= bound for value in samples[3:])
    )


def control_map(round_doc: dict, role: str) -> dict[str, dict]:
    return {
        row["key"]["component"]: row
        for row in round_doc["references"]["five_run_controls"]
        if row["key"]["role"] == role
    }


def summarize_controls(round_doc: dict) -> dict:
    roles: dict[str, dict] = {}

    for row in round_doc["references"]["five_run_controls"]:
        role = row["key"]["role"]
        bucket = roles.setdefault(
            role,
            {"count": 0, "stable": 0, "unstable": 0, "other": 0},
        )
        bucket["count"] += 1

        if row.get("status") == "STABLE":
            bucket["stable"] += 1
        elif row.get("status") == "UNSTABLE":
            bucket["unstable"] += 1
        else:
            bucket["other"] += 1

    return roles


def compare_components(
    left_records: dict[str, dict],
    right_records: dict[str, dict],
    components: list[str],
    bound_for,
) -> dict:
    rows = []

    for component in components:
        distance = max_abs_component(left_records, right_records, component)
        bound, control_ok, rationale = bound_for(component)

        if not control_ok or bound is None:
            status = "INCONCLUSIVE"
        elif distance <= bound:
            status = "PASS"
        else:
            status = "FAIL"

        rows.append(
            {
                "component": component,
                "distance": distance,
                "bound": bound,
                "status": status,
                "control_ok": control_ok,
                "rationale": rationale,
            }
        )

    statuses = {row["status"] for row in rows}

    overall = (
        "FAIL"
        if "FAIL" in statuses
        else "INCONCLUSIVE"
        if "INCONCLUSIVE" in statuses
        else "PASS"
    )

    interesting = [row for row in rows if row["status"] != "PASS"]

    return {
        "status": overall,
        "components": len(rows),
        "pass_count": sum(row["status"] == "PASS" for row in rows),
        "fail_count": sum(row["status"] == "FAIL" for row in rows),
        "inconclusive_count": sum(
            row["status"] == "INCONCLUSIVE" for row in rows
        ),
        "first_nonpass": interesting[:25],
        "max_distance": max(
            (row["distance"] for row in rows if math.isfinite(row["distance"])),
            default=0.0,
        ),
    }


def b2_for_round(round_doc: dict) -> dict:
    refs = round_doc["references"]

    speculative = tensor_records(
        {
            "post_execute": round_doc["post_verify"]["cache"],
            "outputs": [round_doc["verify"]],
            "gdn_capture": round_doc["gdn_capture"],
        }
    )

    reference = tensor_records(
        {
            "post_execute": refs["s8_verify"]["post_execute"]["cache"],
            "outputs": refs["s8_verify"]["outputs"],
            "gdn_capture": refs["s8_verify"]["gdn_capture"],
        }
    )

    controls = control_map(round_doc, "s8_verify")
    components = sorted(controls)

    def bound_for(component: str):
        control = controls.get(component)
        return (
            None if control is None else control.get("bound"),
            stable(control),
            "same-runtime same-width S8 five-run bound",
        )

    result = compare_components(
        speculative,
        reference,
        components,
        bound_for,
    )

    base_left = set(speculative)
    base_right = set(reference)

    result["base_tensor_keys_equal"] = base_left == base_right
    result["base_tensor_count"] = len(base_left)
    result["base_sha_mismatches"] = sorted(
        component
        for component in base_left & base_right
        if speculative[component].get("sha256")
        != reference[component].get("sha256")
    )

    return result


def c_keep2_for_round(round_doc: dict) -> dict:
    refs = round_doc["references"]

    production = tensor_records(
        {"post_execute": round_doc["post_reconcile"]["cache"]}
    )
    s2 = tensor_records(
        {"post_execute": refs["s2_keep2"]["post_execute"]["cache"]}
    )
    s1 = tensor_records(
        {"post_execute": refs["s1_keep2"]["post_execute"]["cache"]}
    )

    r2_controls = control_map(round_doc, "s2_keep2")
    r1_controls = control_map(round_doc, "s1_keep2")
    w21_controls = control_map(round_doc, "s2_vs_s1_keep2")

    components = sorted(
        component
        for component in w21_controls
        if component.startswith("post_execute.")
    )

    def a_bound(component: str):
        control = r2_controls.get(component)
        return (
            None if control is None else control.get("bound"),
            stable(control),
            "A <= repeated S2 bound",
        )

    def b_bound(component: str):
        control = w21_controls.get(component)
        return (
            None if control is None else control.get("bound"),
            stable(control),
            "B <= controlled S2-versus-S1+S1 width envelope",
        )

    def c_bound(component: str):
        r1 = r1_controls.get(component)
        r2 = r2_controls.get(component)
        w21 = w21_controls.get(component)

        ok = stable(r1) and stable(r2) and stable(w21)

        if not ok:
            return None, False, "missing or unstable R1/R2/W21 control"

        bound = r1["bound"] + r2["bound"] + w21["bound"]
        return bound, True, "C <= W21 + R2 + R1"

    a = compare_components(
        production,
        s2,
        components,
        a_bound,
    )

    b = compare_components(
        s2,
        s1,
        components,
        b_bound,
    )

    c = compare_components(
        production,
        s1,
        components,
        c_bound,
    )

    production_fused = round_doc["verify"]["fused"]
    reference_fused = refs["s8_verify"]["outputs"][0]["fused"]

    fused_exact = (
        production_fused.get("sha256")
        == reference_fused.get("sha256")
    )

    statuses = {a["status"], b["status"], c["status"]}

    overall = (
        "FAIL"
        if "FAIL" in statuses or not fused_exact
        else "INCONCLUSIVE"
        if "INCONCLUSIVE" in statuses
        else "PASS"
    )

    return {
        "status": overall,
        "A_post_reconcile_vs_S2": a,
        "B_S2_vs_S1_plus_S1": b,
        "C_post_reconcile_vs_S1_plus_S1": c,
        "accepted_fused_same_width_S8_full_hash_equal": fused_exact,
        "accepted_fused_rows_originating_from_S8": round_doc["accepted_count"] + 1,
        "note": (
            "Full fused S8 hash equality implies exact equality for every "
            "accepted fused row selected from that captured S8 verify. "
            "Semantic row membership/ordering remains an exact structural gate."
        ),
    }


r1_controls = doc.get("r1_context_reference", {}).get(
    "five_run_context_controls", []
)

report = {
    "run_id": doc["run_id"],
    "runtime": doc["runtime"],
    "implementation": doc["implementation"],
    "align_prefill_diagnostic": doc["align_prefill_diagnostic"],
    "kv_bits": doc["kv_bits"],
    "max_tokens": doc["max_tokens"],
    "output_ids": doc["output_ids"],
    "acceptance_sequence": [
        row["accepted_count"] for row in doc["rounds"]
    ],
    "draft_widths": [
        row["draft_width"] for row in doc["rounds"]
    ],
    "historical_trace_measurement": doc.get("historical_trace_measurement"),
    "r1_same_runtime_reference": doc["r1_context_reference"]["comparison"],
    "r1_context_five_run_controls": {
        "count": len(r1_controls),
        "stable": sum(stable(row) for row in r1_controls),
        "all_stable": bool(r1_controls) and all(stable(row) for row in r1_controls),
    },
    "rounds": [],
}

for round_doc in doc["rounds"]:
    row = {
        "round_index": round_doc["round_index"],
        "draft_width": round_doc["draft_width"],
        "accepted_count": round_doc["accepted_count"],
        "accepted_prefix_ids": round_doc["accepted_prefix_ids"],
        "verify_ids": round_doc["verify_ids"],
        "b1_local_exact": round_doc["b1_local_exact"],
        "control_summary": summarize_controls(round_doc),
        "b2": b2_for_round(round_doc),
    }

    if round_doc["accepted_count"] == 1:
        row["c_keep2"] = c_keep2_for_round(round_doc)

    report["rounds"].append(row)


accepted_counts = sorted(
    {round_doc["accepted_count"] for round_doc in doc["rounds"]}
)

b2_round_statuses = [
    round_doc["b2"]["status"] for round_doc in report["rounds"]
]

keep2_rows = [
    round_doc["c_keep2"]
    for round_doc in report["rounds"]
    if "c_keep2" in round_doc
]

report["coverage"] = {
    "accepted_counts_seen": accepted_counts,
    "has_accepted_0": 0 in accepted_counts,
    "has_accepted_1": 1 in accepted_counts,
    "required_accept0_and_accept1_present": (
        0 in accepted_counts and 1 in accepted_counts
    ),
}

report["offline_gate_adjudication"] = {
    "production_semantics_confirmed": not doc["align_prefill_diagnostic"],
    "mlx_r1_same_runtime_reference": (
        doc["r1_context_reference"]["comparison"]["status"]
    ),
    "b1_all_observed_rounds": (
        "PASS"
        if all(
            round_doc["b1_local_exact"]["status"] == "pass"
            for round_doc in doc["rounds"]
        )
        else "FAIL_OR_INCONCLUSIVE"
    ),
    "b2_all_observed_rounds": (
        "PASS"
        if b2_round_statuses
        and all(status == "PASS" for status in b2_round_statuses)
        else "FAIL"
        if "FAIL" in b2_round_statuses
        else "INCONCLUSIVE"
    ),
    "b2_task_coverage": (
        "COMPLETE"
        if report["coverage"]["required_accept0_and_accept1_present"]
        else "INCOMPLETE_MISSING_ACCEPTED_0"
    ),
    "c_keep2_observed_rounds": (
        "PASS"
        if keep2_rows
        and all(row["status"] == "PASS" for row in keep2_rows)
        else "FAIL"
        if any(row["status"] == "FAIL" for row in keep2_rows)
        else "INCONCLUSIVE"
    ),
    "c_keep1_coverage": (
        "PRESENT"
        if report["coverage"]["has_accepted_0"]
        else "MISSING_ACCEPTED_0"
    ),
    "d_status": "NOT_ADJUDICATED",
    "e_status": "NOT_ADJUDICATED",
    "t016_status": "OPEN",
}

out = run_dir / "offline-adjudication.json"
out.write_text(json.dumps(report, indent=2) + "\n")

print(
    json.dumps(
        {
            "run_id": report["run_id"],
            "acceptance_sequence": report["acceptance_sequence"],
            "coverage": report["coverage"],
            "offline_gate_adjudication": report["offline_gate_adjudication"],
            "rounds": [
                {
                    "round": row["round_index"],
                    "accepted": row["accepted_count"],
                    "B1": row["b1_local_exact"]["status"],
                    "B2": row["b2"]["status"],
                    "B2_sha_mismatches": len(
                        row["b2"]["base_sha_mismatches"]
                    ),
                    "C_keep2": (
                        None
                        if "c_keep2" not in row
                        else row["c_keep2"]["status"]
                    ),
                }
                for row in report["rounds"]
            ],
            "output": str(out),
        },
        indent=2,
    )
)
