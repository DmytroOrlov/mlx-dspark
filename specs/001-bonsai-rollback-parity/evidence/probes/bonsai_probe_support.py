"""Project-local, process-only helpers for Bonsai boundary evidence.

This module does not import or alter a model until a caller executes a run.
Each array is materialized into the caller's unique run directory at capture
time; a JSON record alone is deliberately not treated as a full tensor copy.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import re
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


def stable_id(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def required_gate_branch_provenance(*, runtime: str, frozen_root_state_id: str,
                                    frozen_root_observation_id: str,
                                    branch_id: str, logical_position: int,
                                    verify_ids: list[int], accepted_count: int,
                                    b2_control_set_id: str,
                                    natural: bool = False) -> dict:
    """Build provenance for an isolated required-gate branch.

    A B2 set is reusable across keep=1/keep=2 only when its exact logical root
    matches. Acceptance forcing is recorded as a test input, never scheduler
    behavior.
    """
    if accepted_count not in (0, 1) or len(verify_ids) != 8:
        raise ValueError("required branches need S=8 verify IDs and accepted_count 0 or 1")
    keep = accepted_count + 1
    if not frozen_root_state_id or not frozen_root_observation_id or not b2_control_set_id:
        raise ValueError("root state, root observation, and B2 control-set identities are required")
    verify_ids_sha256 = hashlib.sha256(json.dumps(verify_ids, separators=(",", ":")).encode()).hexdigest()
    return {
        "schema": "bonsai-required-gate-branch/v1",
        "runtime": runtime,
        "branch_id": branch_id,
        "branch_kind": "natural" if natural else "isolated_forced",
        "acceptance_source": "scheduler" if natural else "forced_state_transition_test_input",
        "frozen_root_state_id": frozen_root_state_id,
        "frozen_root_observation_id": frozen_root_observation_id,
        "logical_position": logical_position,
        "verify_ids_sha256": verify_ids_sha256,
        "verify_width": 8,
        "draft_width": 7,
        "verify_ids": list(verify_ids),
        "accepted_count": accepted_count,
        "keep": keep,
        "b2_control_set_id": b2_control_set_id,
        "b2_control_root_state_id": frozen_root_state_id,
        "b2_control_root_observation_id": frozen_root_observation_id,
        "b2_control_logical_position": logical_position,
        "b2_control_verify_ids_sha256": verify_ids_sha256,
        "b2_control_shared_across_keep_branches": False,
    }


def link_shared_b2_control_set(branches: list[dict]) -> list[dict]:
    """Link keep=1/keep=2 branches only when their frozen B2 root is identical."""
    if len(branches) != 2 or {row.get("keep") for row in branches} != {1, 2}:
        raise ValueError("shared B2 linkage requires one keep=1 and one keep=2 branch")
    identity_fields = ("frozen_root_state_id", "frozen_root_observation_id",
                       "logical_position", "verify_ids_sha256", "b2_control_set_id")
    control_fields = ("b2_control_root_state_id", "b2_control_root_observation_id",
                      "b2_control_logical_position", "b2_control_verify_ids_sha256",
                      "b2_control_set_id")
    def provenance(row):
        return row.get("provenance", row)
    if any(provenance(row).get(field) is None or provenance(row).get(field) == "" for row in branches for field in
           ("frozen_root_state_id", "frozen_root_observation_id", "logical_position",
            "verify_ids_sha256", "b2_control_set_id")):
        raise ValueError("shared B2 linkage requires content-derived root and control identities")
    for row in branches:
        record = provenance(row)
        verify_ids = record.get("verify_ids")
        if (record.get("runtime") != provenance(branches[0]).get("runtime") or
                not isinstance(verify_ids, list) or len(verify_ids) != 8 or
                hashlib.sha256(json.dumps(verify_ids, separators=(",", ":")).encode()).hexdigest() !=
                record.get("verify_ids_sha256") or
                record.get("keep") != record.get("accepted_count", -1) + 1):
            raise ValueError("shared B2 branches need same-runtime, internally consistent S=8 provenance")
    left = tuple(provenance(branches[0]).get(field) for field in identity_fields)
    right = tuple(provenance(branches[1]).get(field) for field in identity_fields)
    left_control = tuple(provenance(branches[0]).get(field) for field in control_fields)
    right_control = tuple(provenance(branches[1]).get(field) for field in control_fields)
    if left != right or left != left_control or right != right_control:
        raise ValueError("shared B2 linkage requires identical state, observation, position, verify IDs, and control set")
    linked = copy.deepcopy(branches)
    shared_branch_ids = sorted(provenance(item)["branch_id"] for item in linked)
    for row in linked:
        provenance(row)["b2_control_shared_across_keep_branches"] = True
        row["b2_control_shared_across_keep_branches"] = True
        row["shared_b2_branch_ids"] = shared_branch_ids
    return linked


def required_gate_acquisition_record(source_continuous_round: str, b2: dict,
                                     branches: list[dict]) -> dict:
    """Build the isolated acquisition envelope without mixing it into natural rounds."""
    if not source_continuous_round or len(branches) != 2:
        raise ValueError("required-gate acquisition needs a continuous source and two branches")
    if {row.get("accepted_count") for row in branches} != {0, 1}:
        raise ValueError("required-gate branches must cover accepted_count 0 and 1")
    if any(row.get("branch_kind") != "isolated_forced" or
           row.get("acceptance_source") != "forced_state_transition_test_input"
           for row in branches):
        raise ValueError("forced branches must be labeled outside natural scheduler behavior")
    if any(row.get("provenance", row).get("branch_kind") != "isolated_forced"
           for row in branches):
        raise ValueError("branch provenance must identify isolated forced input")
    return {"mode": "isolated_forced", "source_continuous_round": source_continuous_round,
            "b2": copy.deepcopy(b2), "branches": copy.deepcopy(branches)}


def assert_first_round_e_reconstructable(row: dict) -> dict:
    """Reject later speculative roots whose drafter history is not reconstructable."""
    if row.get("round_index") != 1 or row.get("generated_before") != 0:
        raise ValueError("required-gates E requires the first logical speculative round (generated_before=0)")
    pending = row.get("pending_context")
    if not isinstance(pending, dict) or not pending.get("shape") or len(pending["shape"]) < 3:
        raise ValueError("required-gates E requires durable first-round prompt pending/context rows")
    pre = row.get("drafter_pre_proposal")
    if not isinstance(pre, list) or not pre:
        raise ValueError("required-gates E requires a captured fresh drafter cache before the first proposal")
    for layer in pre:
        live = layer.get("live_length", 0)
        if live not in (None, 0):
            raise ValueError("required-gates E refuses a nonempty pre-proposal drafter cache")
        for name in ("keys", "values"):
            record = layer.get(name)
            if isinstance(record, dict) and record.get("shape"):
                shape = record["shape"]
                if len(shape) >= 2 and shape[-2] != 0:
                    raise ValueError("required-gates E refuses nonempty pre-proposal drafter KV")
    return {"first_logical_round": True, "generated_before": 0,
            "fresh_empty_drafter_cache": True,
            "prompt_context_rows": int(pending["shape"][1]),
            "pending_context_sha256": pending.get("sha256")}


def e_source_proposal_adjudication(source_ids: list[int], reconstructed_ids: list[int]) -> dict:
    exact = list(source_ids) == list(reconstructed_ids)
    return {"source_proposal_exact": exact,
            "next_proposal_comparable": exact,
            "numeric_status_if_mismatch": "inconclusive" if not exact else None,
            "source_ids": list(source_ids),
            "reconstructed_ids": list(reconstructed_ids)}


def evidence_content_view(value: object) -> object:
    """Canonicalize captured evidence for content identity, independent of storage."""
    if isinstance(value, dict):
        if (
            {"shape", "dtype", "sha256"} <= set(value)
            and ("encoding" in value or "blob" in value or "transient_key" in value)
        ):
            return {
                "shape": value["shape"],
                "dtype": value["dtype"],
                "stored_dtype": value.get("stored_dtype"),
                "sha256": value["sha256"],
            }
        return {key: evidence_content_view(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [evidence_content_view(item) for item in value]
    return value


class EvidenceStore:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.array_dir = run_dir / "arrays"
        self.array_dir.mkdir(exist_ok=False)
        self.transient_arrays: dict[str, object] = {}
        self.transient_active = False

    def begin_transient(self) -> None:
        if self.transient_active or self.transient_arrays:
            raise RuntimeError("A control pair is already using transient arrays")
        self.transient_active = True

    def end_transient(self) -> None:
        self.transient_arrays.clear()
        self.transient_active = False

    def capture(self, value: object) -> object:
        import mlx.core as mx
        import numpy as np

        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (list, tuple)):
            return [self.capture(item) for item in value]
        if isinstance(value, dict):
            return {str(key): self.capture(item) for key, item in value.items()}
        if not isinstance(value, mx.array):
            return repr(value)

        mx.eval(value)
        original_dtype = str(value.dtype)
        # NumPy cannot represent MLX bfloat16. Its conversion to float32 is
        # value-exact; integers and quantized packed words retain native dtype.
        converted = value.astype(mx.float32) if value.dtype == mx.bfloat16 else value
        data = np.asarray(converted).copy()
        digest = hashlib.sha256(data.tobytes(order="C")).hexdigest()
        shape_tag = "x".join(str(part) for part in data.shape)
        if self.transient_active:
            transient_key = f"{digest}-{data.dtype}-{shape_tag}"
            self.transient_arrays[transient_key] = data
            return {
                "shape": list(value.shape), "dtype": original_dtype,
                "stored_dtype": str(data.dtype), "encoding": "transient-control-pair",
                "sha256": digest, "transient_key": transient_key,
                "first_values": data.reshape(-1)[:8].tolist(),
            }
        name = f"{digest}-{str(data.dtype)}-{shape_tag}.npy.zlib"
        path = self.array_dir / name
        if not path.exists():
            buffer = io.BytesIO()
            np.save(buffer, data, allow_pickle=False)
            path.write_bytes(zlib.compress(buffer.getvalue(), level=1))
        return {
            "shape": list(value.shape), "dtype": original_dtype,
            "stored_dtype": str(data.dtype), "encoding": "npy+zlib-level1",
            "sha256": digest,
            "blob": str(path.relative_to(self.run_dir)),
            "first_values": data.reshape(-1)[:8].tolist(),
        }


def clone_cache(cache):
    """Independently allocate every mutable cache and materialize its MLX arrays."""
    import mlx.core as mx
    import numpy as np

    def clone(value):
        if isinstance(value, mx.array):
            dtype = value.dtype
            mx.eval(value)
            as_numpy = np.asarray(value.astype(mx.float32) if dtype == mx.bfloat16 else value)
            return mx.array(as_numpy.copy()).astype(dtype)
        if isinstance(value, list):
            return [clone(item) for item in value]
        if isinstance(value, tuple):
            return tuple(clone(item) for item in value)
        if isinstance(value, dict):
            return {key: clone(item) for key, item in value.items()}
        return copy.deepcopy(value)

    result = []
    for source in cache:
        target = copy.copy(source)
        target.__dict__ = {key: clone(value) for key, value in source.__dict__.items()}
        assert target is not source
        result.append(target)
    return result


def mutable_cache_object_ids(cache) -> list[int]:
    """Identify cache containers and MLX arrays, including arrays nested in slots."""
    import mlx.core as mx

    found: set[int] = set()
    seen: set[int] = set()

    def visit(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, mx.array):
            found.add(id(value))
        elif isinstance(value, (list, tuple)):
            if isinstance(value, list):
                found.add(id(value))
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            found.add(id(value))
            for item in value.values():
                visit(item)
        elif hasattr(value, "__dict__"):
            found.add(id(value))
            visit(vars(value))

    visit(cache)
    return sorted(found)


def branch_identity(run_id: str, round_index: int, kind: str, cache, state: object,
                    source: str | None = None) -> dict:
    return {
        "branch_id": f"{run_id}:r{round_index}:{kind}",
        "cache_object_ids": [id(item) for item in cache],
        "cache_state_id": stable_id(evidence_content_view(state)),
        "source": source or ("independent fresh replay" if kind != "speculative" else "production cache"),
    }


def freeze_record(value: dict) -> dict:
    """Round-trip through JSON so later dictionary/cache mutation cannot change it."""
    return json.loads(json.dumps(value, sort_keys=True))


@dataclass(frozen=True)
class ControlKey:
    runtime: str
    branch_width: int
    role: str
    component: str
    layer: int
    row: int | None
    dtype: str
    metric: str
    logical_state_id: str
    execution_shape: tuple[int, ...]
    reference_root: str


@dataclass
class FiveRunControl:
    key: ControlKey
    samples: list[float] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    sample_hashes: list[dict] = field(default_factory=list)

    def add(self, distance: float, evidence_id: str,
            left_sha256: str | None = None, right_sha256: str | None = None) -> None:
        if len(self.samples) >= 5:
            raise ValueError("Exactly five control repetitions are permitted")
        if not math.isfinite(distance) or distance < 0:
            raise ValueError("Control distance must be finite and nonnegative")
        self.samples.append(float(distance))
        self.evidence_ids.append(evidence_id)
        self.sample_hashes.append({"left_sha256": left_sha256,
                                   "right_sha256": right_sha256})

    def result(self) -> dict:
        if len(self.samples) != 5:
            return {"status": "INCOMPLETE", "samples": list(self.samples),
                    "sample_hashes": list(self.sample_hashes),
                    "missing": 5 - len(self.samples), "key": vars(self.key)}
        bound = max(self.samples[:3])
        stable = all(value <= bound for value in self.samples[3:])
        return {"status": "STABLE" if stable else "UNSTABLE", "bound": bound,
                "calibration": self.samples[:3], "validation": self.samples[3:],
                "samples": list(self.samples), "evidence_ids": list(self.evidence_ids),
                "sample_hashes": list(self.sample_hashes),
                "key": vars(self.key),
                "rationale": "max of repetitions 1-3; repetitions 4-5 must not exceed it"}


def execute_five_controls(
    key: ControlKey,
    make_independent_pair: Callable[[int], tuple[object, object, str]],
    distance: Callable[[object, object], float],
) -> dict:
    """Measure five predeclared equivalent pairs before any failing sample is read.

    The caller must return fresh independently allocated branches for each
    repetition, and must predeclare the component/dtype/metric in ``key``.
    This function is preparation machinery; no controls run on import.
    """
    series = FiveRunControl(key)
    for repetition in range(1, 6):
        left, right, evidence_id = make_independent_pair(repetition)
        series.add(distance(left, right), evidence_id)
    return series.result()


def control_plan(runtime: str) -> list[dict]:
    return [
        {"role": "R1 fresh prefill context", "width": "prompt-prefix",
         "root": "independent_fresh_r1_prefill", "repetitions": 5},
        {"role": "S8-versus-S8 verify/tap/fused", "width": 8,
         "root": "exact_frozen_speculative_preverify", "repetitions": 5},
        {"role": "S1-versus-S1 keep=1 committed", "width": 1,
         "root": "fresh_committed_replay", "repetitions": 5},
        {"role": "S1+S1-versus-S1+S1 keep=2 committed", "width": 1,
         "root": "fresh_committed_replay", "repetitions": 5},
        {"role": "S2-versus-S2 committed prefix", "width": 2,
         "root": "fresh_committed_replay", "repetitions": 5},
        {"role": "S2-versus-S1+S1 width delta", "width": 2,
         "root": "fresh_committed_replay", "repetitions": 5},
    ]


def numeric_vectors(record: dict, run_dir: Path, *, include_outputs: bool = True,
                    store: EvidenceStore | None = None) -> dict[str, tuple[object, str, str]]:
    """Read a reference's captured component arrays for predeclared metrics."""
    values = {}

    def visit(value, path):
        if (isinstance(value, dict) and "dtype" in value and
                ("blob" in value or "transient_key" in value)):
            data = load_array_record(value, run_dir, store)
            values[path] = (data, value["dtype"], value["sha256"])
            if path == "outputs.0.fused" and data.ndim == 3 and data.shape[-1] % 5 == 0:
                width = data.shape[-1] // 5
                for row in range(data.shape[1]):
                    for tap_index, tap_id in enumerate((5, 19, 33, 47, 61)):
                        part = data[:, row, tap_index * width:(tap_index + 1) * width]
                        values[f"{path}.tap{tap_id}.row{row}"] = (
                            part, value["dtype"], hashlib.sha256(part.tobytes()).hexdigest())
            if path == "outputs.0.logits" and data.ndim == 3:
                for row in range(data.shape[1]):
                    part = data[:, row, :]
                    values[f"{path}.row{row}"] = (
                        part, value["dtype"], hashlib.sha256(part.tobytes()).hexdigest())
        elif isinstance(value, dict):
            for key, item in value.items():
                visit(item, f"{path}.{key}" if path else str(key))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{path}.{index}" if path else str(index))

    fields = {"post_execute": record["post_execute"]["cache"]}
    if include_outputs:
        fields.update({"outputs": record["outputs"],
                       "gdn_capture": record.get("gdn_capture")})
    visit(fields, "")
    return values


def transient_pair_controls(*, root_id: str, component_prefix: str,
                            make_pair: Callable[[int, str], dict],
                            run_dir: Path, store: EvidenceStore,
                            require_top2: bool = True) -> dict:
    """Collect five independent left/right cache-and-output pairs transiently."""
    import numpy as np

    samples: dict[str, list[float]] = {}
    specs: dict[str, dict] = {}
    ids: list[str] = []
    discrete: list[dict] = []
    pair_hashes: list[dict] = []
    structure_signatures: list[str] = []
    for repetition in range(1, 6):
        store.begin_transient()
        try:
            left, right = make_pair(repetition, root_id)
            left_vec = numeric_vectors(left, run_dir, store=store)
            right_vec = numeric_vectors(right, run_dir, store=store)
            allowed_cache_slots = (".keys", ".values", ".convolution_window", ".recurrent_state")
            left_vec = {key: value for key, value in left_vec.items()
                        if key.startswith("outputs.") or any(key.endswith(slot) for slot in allowed_cache_slots)}
            right_vec = {key: value for key, value in right_vec.items()
                         if key.startswith("outputs.") or any(key.endswith(slot) for slot in allowed_cache_slots)}
            if left_vec.keys() != right_vec.keys():
                raise ValueError("control pair component sets differ")
            left_structure = cache_semantic_signature(left["post_execute"]["cache"])
            right_structure = cache_semantic_signature(right["post_execute"]["cache"])
            structure_signatures.extend((stable_id(left_structure), stable_id(right_structure)))
            ids.append(stable_id({"root_id": root_id, "repetition": repetition,
                                  "left": {k: v[2] for k, v in left_vec.items()},
                                  "right": {k: v[2] for k, v in right_vec.items()}}))
            pair_hashes.append({"repetition": repetition,
                "components": {key: {"left_sha256": left_vec[key][2],
                                     "right_sha256": right_vec[key][2]}
                               for key in left_vec}})
            for key, (a, dtype, _) in left_vec.items():
                b, right_dtype, _ = right_vec[key]
                if dtype != right_dtype or np.shape(a) != np.shape(b):
                    raise ValueError(f"control component metadata differs: {key}")
                name = f"{component_prefix}:{key}" if component_prefix else key
                value = float(np.max(np.abs(np.asarray(a, dtype=np.float64) -
                                            np.asarray(b, dtype=np.float64)))) if np.size(a) else 0.0
                samples.setdefault(name, []).append(value)
                specs[name] = {"component": key, "dtype": dtype,
                               "shape": list(np.shape(a)), "metric": "max_abs",
                               "root_id": root_id}
            discrete.append({"repetition": repetition,
                "left_top2_ids": left.get("outputs", {}).get("top2_ids"),
                "right_top2_ids": right.get("outputs", {}).get("top2_ids"),
                "left_top2_margins": left.get("outputs", {}).get("top2_margins"),
                "right_top2_margins": right.get("outputs", {}).get("top2_margins"),
                "left_proposal_ids": left.get("proposal_ids"),
                "right_proposal_ids": right.get("proposal_ids")})
        finally:
            store.end_transient()
    results = {}
    for name, distances in samples.items():
        bound = max(distances[:3])
        results[name] = {**specs[name], "samples": distances,
                         "calibration": distances[:3], "validation": distances[3:],
                         "bound": bound,
                         "stable": all(value <= bound for value in distances[3:]),
                         "control_ids": ids}
    top2_stable = (all(item["left_top2_ids"] == item["right_top2_ids"] for item in discrete) and
                   len({json.dumps(item["left_top2_ids"], sort_keys=True) for item in discrete}) == 1)
    margin_distances = [max((abs(float(a) - float(b)) for a, b in zip(
        (item["left_top2_margins"] or []), (item["right_top2_margins"] or []))), default=0.0)
        for item in discrete]
    margin_bound = max(margin_distances[:3]) if margin_distances else None
    margin_stable = bool(margin_distances) and all(x <= margin_bound for x in margin_distances[3:])
    proposal_ids_stable = (all(item["left_proposal_ids"] == item["right_proposal_ids"]
                               for item in discrete) and
                           len({json.dumps(item["left_proposal_ids"], sort_keys=True)
                                for item in discrete}) == 1)
    structure_stable = len(set(structure_signatures)) == 1
    discrete_stable = (top2_stable and margin_stable) if require_top2 else True
    return {"root_id": root_id, "repetitions": 5,
            "calibration_repetitions": [1, 2, 3], "validation_repetitions": [4, 5],
            "components": results, "component_set": list(specs),
            "pair_hashes": pair_hashes, "discrete": discrete,
            "structure_stable": structure_stable,
            "structure_signature_ids": structure_signatures,
            "top2_ids_repeatable": top2_stable,
            "proposal_ids_repeatable": proposal_ids_stable,
            "margin_repeatable": margin_stable, "margin_bound": margin_bound,
            "margin_distances": margin_distances,
            "status": "STABLE" if all(x["stable"] for x in results.values()) and
                      discrete_stable and proposal_ids_stable and structure_stable else "UNSTABLE"}


def compare_controlled_vectors(sample: dict, reference: dict, control: dict,
                               run_dir: Path, *, expected_root_id: str) -> dict:
    """Compare durable production/oracle samples against their predeclared components."""
    import numpy as np
    if control.get("root_id") != expected_root_id:
        raise ValueError("numeric control bounds cannot transfer across logical roots")
    left = numeric_vectors(sample, run_dir)
    right = numeric_vectors(reference, run_dir)
    allowed_cache_slots = (".keys", ".values", ".convolution_window", ".recurrent_state")
    left = {key: value for key, value in left.items()
            if key.startswith("outputs.") or any(key.endswith(slot) for slot in allowed_cache_slots)}
    right = {key: value for key, value in right.items()
             if key.startswith("outputs.") or any(key.endswith(slot) for slot in allowed_cache_slots)}
    comparisons = []
    if left.keys() != right.keys():
        return {"status": "inconclusive", "reason": "sample component sets differ",
                "sample_components": sorted(left), "reference_components": sorted(right),
                "numeric_comparisons": []}
    for key in sorted(left):
        a, dtype, _ = left[key]
        b, other_dtype, _ = right[key]
        name = f"{key}"
        entry = control["components"].get(name)
        if entry is None or dtype != other_dtype or list(np.shape(a)) != list(np.shape(b)):
            comparisons.append({"component": key, "status": "inconclusive"})
            continue
        distance = float(np.max(np.abs(np.asarray(a, dtype=np.float64) -
                                       np.asarray(b, dtype=np.float64)))) if np.size(a) else 0.0
        comparisons.append({"component": key, "dtype": dtype, "shape": list(np.shape(a)),
                            "distance": distance, "bound": entry["bound"],
                            "status": "pass" if entry["stable"] and distance <= entry["bound"]
                            else "fail" if entry["stable"] else "inconclusive"})
    status = "inconclusive" if control["status"] != "STABLE" or any(
        item["status"] == "inconclusive" for item in comparisons) else (
        "fail" if any(item["status"] == "fail" for item in comparisons) else "pass")
    return {"status": status, "numeric_comparisons": comparisons}


def measure_control_matrix(runtime: str, run_dir: Path, root_state_ids: dict[str, str],
                           make_reference: Callable[[str, int, str], dict],
                           store: EvidenceStore) -> list[dict]:
    """Run exactly five fresh pairs for each predeclared reference width/role.

    The callback receives a role, repetition 1..5, and side left/right. S=8
    uses the exact frozen speculative root; committed widths use a separate
    fresh semantic replay root. Each call independently clones its own root.
    Measurements are collected before the caller examines the speculative
    branch, so no failing sample can influence calibration or validation.
    """
    import numpy as np

    roles = ("s8_verify", "s1_keep1", "s1_keep2", "s2_keep2", "s2_vs_s1_keep2")
    roots = {"s8_verify": "exact_frozen_speculative_preverify",
             "s1_keep1": "fresh_committed_replay",
             "s1_keep2": "fresh_committed_replay",
             "s2_keep2": "fresh_committed_replay",
             "s2_vs_s1_keep2": "fresh_committed_replay"}
    if set(root_state_ids) != set(roles):
        raise ValueError("Every control role needs its own declared root state ID")
    results = []
    for role in roles:
        series: dict[str, FiveRunControl] = {}
        declared_components = None
        for repetition in range(1, 6):
            store.begin_transient()
            left = make_reference(role, repetition, "left")
            right = make_reference(role, repetition, "right")
            include_outputs = role != "s2_vs_s1_keep2"
            left_values = numeric_vectors(left, run_dir, include_outputs=include_outputs,
                                          store=store)
            right_values = numeric_vectors(right, run_dir, include_outputs=include_outputs,
                                           store=store)
            if left_values.keys() != right_values.keys():
                raise ValueError(f"Control component set differs for {role} repetition {repetition}")
            if declared_components is None:
                declared_components = tuple(sorted(left_values))
            elif tuple(sorted(left_values)) != declared_components:
                raise ValueError(f"Control component set changed after predeclaration: {role}")
            for component in declared_components:
                a, dtype_a, sha_a = left_values[component]
                b, dtype_b, sha_b = right_values[component]
                if dtype_a != dtype_b or a.shape != b.shape:
                    raise ValueError(f"Exact dtype/shape mismatch in {role}:{component}")
                layer_match = re.search(r"(?:post_execute|cache)\.(\d+)", component)
                row_match = re.search(r"\.row(\d+)", component)
                width = 8 if role == "s8_verify" else 2 if role in ("s2_keep2", "s2_vs_s1_keep2") else 1
                key = ControlKey(runtime, width, role, component,
                                 int(layer_match.group(1)) if layer_match else -1,
                                 int(row_match.group(1)) if row_match else None,
                                 dtype_a, "max_abs", root_state_ids[role],
                                 tuple(a.shape), roots[role])
                current = series.setdefault(component, FiveRunControl(key))
                distance = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.0
                current.add(distance, f"{role}:r{repetition}:{component}", sha_a, sha_b)
            del left, right, left_values, right_values
            store.end_transient()
        results.extend(current.result() for current in series.values())
    return results


def measure_r1_context_controls(runtime: str, run_dir: Path, prefix_state_id: str,
                                make_context: Callable[[int, str], dict],
                                store: EvidenceStore) -> list[dict]:
    """Five paired fresh prefills; context and each tap row retain separate bounds."""
    import numpy as np

    series: dict[str, FiveRunControl] = {}
    components = None
    for repetition in range(1, 6):
        store.begin_transient()
        left = make_context(repetition, "left")
        right = make_context(repetition, "right")
        if (left["input_ids"] != right["input_ids"] or
                left["positions"] != right["positions"] or
                left["anchor_id"] != right["anchor_id"] or
                left["drafter_block_ids"] != right["drafter_block_ids"] or
                left["drafter_block_dtype"] != right["drafter_block_dtype"] or
                left["tap_order"] != right["tap_order"]):
            raise ValueError("R1 context controls did not start at identical logical input")
        names = ["pending_context", *sorted(left["tap_rows"])]
        if names != ["pending_context", *sorted(right["tap_rows"])]:
            raise ValueError("R1 context control tap set changed")
        if components is None:
            components = names
        elif components != names:
            raise ValueError("R1 context control component set changed")
        for name in names:
            a_rec = left["pending_context"] if name == "pending_context" else left["tap_rows"][name]
            b_rec = right["pending_context"] if name == "pending_context" else right["tap_rows"][name]
            if a_rec["shape"] != b_rec["shape"] or a_rec["dtype"] != b_rec["dtype"]:
                raise ValueError(f"R1 context dtype/shape changed for {name}")
            a = load_array_record(a_rec, run_dir, store)
            b = load_array_record(b_rec, run_dir, store)
            if name == "pending_context":
                rows = [(name, a, b, None, None, a_rec["sha256"], b_rec["sha256"])]
            else:
                match = re.fullmatch(r"tap(\d+)", name)
                if match is None:
                    raise ValueError(f"Unknown R1 tap component {name}")
                rows = [(f"{name}.row{index}", a[:, index, :], b[:, index, :],
                         int(match.group(1)), index,
                         hashlib.sha256(a[:, index, :].tobytes()).hexdigest(),
                         hashlib.sha256(b[:, index, :].tobytes()).hexdigest())
                        for index in range(a.shape[1])]
            for component, aa, bb, layer, row_index, sha_a, sha_b in rows:
                key = ControlKey(runtime, int(a_rec["shape"][1]),
                                 "r1_prefill_context", component, layer or -1,
                                 row_index, a_rec["dtype"], "max_abs",
                                 prefix_state_id, tuple(aa.shape),
                                 "independent_fresh_r1_prefill")
                current = series.setdefault(component, FiveRunControl(key))
                distance = float(np.max(np.abs(aa.astype(np.float64) -
                                               bb.astype(np.float64)))) if aa.size else 0.0
                current.add(distance, f"r1_prefill_context:r{repetition}:{component}",
                            sha_a, sha_b)
        del left, right, a, b
        store.end_transient()
    return [value.result() for value in series.values()]


def load_array_record(record: dict, run_dir: Path, store: EvidenceStore | None = None):
    import numpy as np

    if "transient_key" in record:
        if store is None or not store.transient_active:
            raise RuntimeError("Transient control array is unavailable")
        return store.transient_arrays[record["transient_key"]]
    encoded = (run_dir / record["blob"]).read_bytes()
    return np.load(io.BytesIO(zlib.decompress(encoded)), allow_pickle=False)


def load_mx_array_record(record: dict, run_dir: Path, mx):
    """Restore a captured array to its recorded MLX dtype for live reference work."""
    import numpy as np

    if "transient_key" in record:
        raise ValueError("live gate reconstruction requires durable, not transient, arrays")
    encoded = (run_dir / record["blob"]).read_bytes()
    data = np.load(io.BytesIO(zlib.decompress(encoded)), allow_pickle=False)
    value = mx.array(data)
    dtype_name = record.get("dtype", "")
    for suffix, dtype in (("bfloat16", mx.bfloat16), ("float16", mx.float16),
                          ("float32", mx.float32), ("int32", mx.int32),
                          ("uint32", mx.uint32)):
        if dtype_name.endswith(suffix):
            value = value.astype(dtype)
            break
    return value


def compare_r1_context_reference(row: dict, reference: dict, run_dir: Path,
                                 controls: list[dict] | None) -> dict:
    """Classify same-runtime R1 inputs before interpreting context or proposals."""
    import numpy as np

    production_shape = row["pending_context"]["shape"]
    tap_count = len(reference["tap_order"])
    tap_width = production_shape[-1] // tap_count if tap_count else 0
    tap_shapes_equal = (tap_count > 0 and production_shape[-1] == tap_count * tap_width and
                        set(reference["tap_rows"]) ==
                        {f"tap{layer}" for layer in reference["tap_order"]} and
                        all(value["shape"] == [production_shape[0], production_shape[1], tap_width]
                            and value["dtype"] == row["pending_context"]["dtype"]
                            for value in reference["tap_rows"].values()))
    exact = {
        "prefill_ids": reference["input_ids"] == row["initial_prefill_ids"],
        "anchor_id": reference["anchor_id"] == row["anchor_or_pending_id"],
        "drafter_block_ids": reference["drafter_block_ids"] == row["drafter_block_ids"],
        "drafter_block_dtype": reference["drafter_block_dtype"] == row["drafter_block_dtype"],
        "tap_shapes_dtype": tap_shapes_equal,
        "target_positions": reference["target_positions"] == row["target_positions"],
        "drafter_positions": reference["drafter_positions"] == row["drafter_positions"],
        "target_cache_structure": cache_semantic_signature(reference["target_cache"]) ==
                                  cache_semantic_signature(row["target_pre_proposal"]),
        "drafter_cache_structure": cache_semantic_signature(reference["drafter_cache"]) ==
                                   cache_semantic_signature(row["drafter_pre_proposal"]),
        "fused_shape_dtype": (reference["pending_context"]["shape"],
                              reference["pending_context"]["dtype"]) ==
                             (row["pending_context"]["shape"],
                              row["pending_context"]["dtype"]),
    }
    result = {"reference_kind": "independent_same_runtime_r1_prefill_and_proposal",
              "observation_ids": [row["observation_id"], reference["observation_id"]],
              "exact_input_checks": exact,
              "context_sha256_equal": (reference["pending_context"]["sha256"] ==
                                       row["pending_context"]["sha256"]),
              "proposal_ids_equal": reference["draft_ids"] == row["draft_ids"],
              "control_status": "not_required_if_context_exact" if
                                reference["pending_context"]["sha256"] ==
                                row["pending_context"]["sha256"] else "missing",
              "numeric_comparisons": []}
    if not all(exact.values()):
        result.update(status="inconclusive", routing="repair_exact_r1_input_control")
        return result
    if result["context_sha256_equal"]:
        result.update(status="pass" if result["proposal_ids_equal"] else "fail",
                      routing="continue" if result["proposal_ids_equal"] else
                              "localize_same_runtime_drafter_proposal")
        return result
    if controls is None:
        result.update(status="inconclusive", routing="measure_five_r1_context_controls")
        return result
    by_component = {item["key"]["component"]: item for item in controls}
    prod = load_array_record(row["pending_context"], run_dir)
    fresh = load_array_record(reference["pending_context"], run_dir)
    names = ["pending_context"]
    width = prod.shape[-1] // len(reference["tap_order"])
    if prod.shape[-1] != width * len(reference["tap_order"]):
        result.update(status="inconclusive", routing="repair_tap_shape")
        return result
    for tap in reference["tap_order"]:
        names.extend(f"tap{tap}.row{index}" for index in range(prod.shape[1]))
    for name in names:
        if name == "pending_context":
            aa, bb = prod, fresh
        else:
            match = re.fullmatch(r"tap(\d+)\.row(\d+)", name)
            tap_index = reference["tap_order"].index(int(match.group(1)))
            row_index = int(match.group(2))
            start = tap_index * width
            aa = prod[:, row_index, start:start + width]
            bb = fresh[:, row_index, start:start + width]
        distance = float(np.max(np.abs(aa.astype(np.float64) -
                                       bb.astype(np.float64)))) if aa.size else 0.0
        control = by_component.get(name)
        status = "inconclusive" if control is None or control["status"] != "STABLE" else (
            "pass" if distance <= control["bound"] else "fail")
        result["numeric_comparisons"].append({"component": name, "distance": distance,
                                               "bound": control.get("bound") if control else None,
                                               "status": status})
    outcomes = [item["status"] for item in result["numeric_comparisons"]]
    result["control_status"] = "stable" if "inconclusive" not in outcomes else "inconclusive"
    result["status"] = ("inconclusive" if "inconclusive" in outcomes else
                        "fail" if "fail" in outcomes or not result["proposal_ids_equal"] else "pass")
    result["routing"] = ("repair_r1_control" if result["status"] == "inconclusive" else
                         "localize_same_runtime_r1" if result["status"] == "fail" else "continue")
    return result


def cache_semantic_signature(rows: list[dict] | None) -> object:
    """Cross-runtime logical structure; implementation wrapper names are omitted."""
    if rows is None:
        return None
    result = []
    for row in rows:
        cls = row.get("class", "")
        family = ("QuantizedKVCache" if "QuantizedKVCache" in cls else
                  "KVCache" if "KVCache" in cls else
                  "ArraysCache" if "ArraysCache" in cls else "unknown")
        def shape(value):
            if isinstance(value, dict) and "shape" in value:
                return {"shape": value["shape"], "dtype": value["dtype"]}
            if isinstance(value, list):
                return [shape(item) for item in value]
            return None
        result.append({
            "layer": row.get("layer"), "kind": (row.get("semantic") or {}).get("kind"),
            "family": family, "trimmable": row.get("trimmable"),
            "bits": row.get("bits"), "group_size": row.get("group_size"),
            "offset": row.get("offset"), "lengths": row.get("lengths"),
            "live_length": row.get("live_length"),
            "absolute_positions": row.get("absolute_positions"),
            "absolute_range": row.get("absolute_range"),
            "keys": shape(row.get("keys")), "values": shape(row.get("values")),
            "convolution_window": shape(row.get("convolution_window")),
            "recurrent_state": shape(row.get("recurrent_state")),
        })
    return result


def compare_gate_a(chad: dict, mlx: dict) -> dict:
    """Classify exact Gate A controls and retain runtime-specific proposals.

    Cross-runtime numeric context and its proposal IDs are diagnostic when
    distinct valid target graphs each reproduce their own R1 reference.
    """
    evidence = []
    result = {"gate": "A", "result": "inconclusive", "classification": "missing_control",
              "comparison_type": "exact_semantic_and_logical", "observation_ids": evidence,
              "earliest_difference": None, "missing_control": None,
              "routing": "stop_at_A", "decision": "stop",
              "proposal_observations": [], "proposal_diagnostics": [],
              "same_runtime_r1_context": {}}
    for index, (left_row, right_row) in enumerate(
            zip(chad.get("rounds") or [], mlx.get("rounds") or []), 1):
        result["proposal_observations"].append({
            "round": index, "chad": left_row.get("draft_ids"),
            "mlx_dspark": right_row.get("draft_ids"),
            "chad_verify_ids": left_row.get("verify_ids"),
            "mlx_dspark_verify_ids": right_row.get("verify_ids")})
    def difference(path, left, right, classification):
        uncontrolled = classification == "control_mismatch"
        result.update(result="inconclusive" if uncontrolled else "fail", classification=classification,
                      earliest_difference={"field": path, "chad": left, "mlx_dspark": right},
                      missing_control=f"repair mismatched {path}" if uncontrolled else None,
                      routing="repair_controls" if uncontrolled else
                              "localize_input_state" if classification == "earliest_input_state_difference" else
                              "localize_proposal_context")
        return result

    identity_fields = ("target_id", "target_fingerprint", "target_config_fingerprint",
                       "drafter_id", "drafter_fingerprint", "drafter_config_fingerprint",
                       "tokenizer_fingerprint", "tokenizer_config_fingerprint",
                       "prompt_ids", "chat_template", "max_tokens", "tap_ids", "tap_order",
                       "target_model_type",
                       "mlx_lm_source_hashes")
    for field_name in identity_fields:
        left, right = chad.get(field_name), mlx.get(field_name)
        if left is None or right is None:
            result["missing_control"] = f"missing {field_name} in at least one run"
            return result
        if left != right:
            return difference(field_name, left, right, "control_mismatch")
    sampling_keys = ("temperature", "top_p", "top_k", "mode")
    for field_name in sampling_keys:
        left = (chad.get("sampling") or {}).get(field_name)
        right = (mlx.get("sampling") or {}).get(field_name)
        if left is None or right is None:
            result["missing_control"] = f"sampling {field_name} missing"
            return result
        if left != right:
            return difference(f"sampling.{field_name}", left, right, "control_mismatch")
    if not chad.get("rounds") or not mlx.get("rounds"):
        result["missing_control"] = "continuous round records missing"
        return result
    for index, (left_row, right_row) in enumerate(zip(chad["rounds"], mlx["rounds"]), 1):
        evidence.extend([left_row.get("observation_id"), right_row.get("observation_id")])
        fields = ("round_index", "initial_prefill_ids", "previous_committed_ids",
                  "committed_prefix_ids", "generated_before", "anchor_or_pending_id",
                  "drafter_block_ids", "proposal_context_logical_position",
                  "proposal_context_rows", "proposal_context_absolute_end",
                  "target_positions", "drafter_positions")
        for field_name in fields:
            left, right = left_row.get(field_name), right_row.get(field_name)
            if left is None or right is None:
                result["missing_control"] = f"R{index} missing {field_name}"
                return result
            if field_name == "target_positions" and any(
                value.get("next_target_absolute_position") is None or value.get("target_live_range") is None
                for value in (left, right)):
                result["missing_control"] = f"R{index} missing exact target positions"
                return result
            if field_name == "drafter_positions" and any(
                value.get("context_absolute_end") is None or value.get("context_live_rows") is None
                for value in (left, right)):
                result["missing_control"] = f"R{index} missing exact drafter context positions"
                return result
            if left != right:
                return difference(f"R{index}.{field_name}", left, right,
                                  "earliest_input_state_difference")
        left_shape = (left_row.get("pending_context") or {}).get("shape")
        right_shape = (right_row.get("pending_context") or {}).get("shape")
        if left_shape is None or right_shape is None:
            result["missing_control"] = f"R{index} missing proposal-context shape"
            return result
        if left_shape != right_shape:
            return difference(f"R{index}.pending_context.shape", left_shape, right_shape,
                              "earliest_input_state_difference")
        for cache_name in ("target_pre_proposal", "drafter_pre_proposal"):
            left = cache_semantic_signature(left_row.get(cache_name))
            right = cache_semantic_signature(right_row.get(cache_name))
            if left is None or right is None:
                result["missing_control"] = f"R{index} missing {cache_name}"
                return result
            if any(item.get("family") == "unknown" for item in [*left, *right]):
                result["missing_control"] = f"R{index} unknown cache family in {cache_name}"
                return result
            if left != right:
                family_changed = any(a.get("family") != b.get("family") or a.get("bits") != b.get("bits")
                                     for a, b in zip(left, right)) or len(left) != len(right)
                classification = "control_mismatch" if family_changed else "earliest_input_state_difference"
                return difference(f"R{index}.{cache_name}", left, right, classification)
    if len(chad["rounds"]) != len(mlx["rounds"]):
        result["missing_control"] = "different continuous capture length"
        return result

    # Verify each runtime's recorded input sequence independently. Distinct
    # proposals imply distinct verify IDs, without invalidating B1 within either
    # runtime. A missing or inconsistent local sequence is still a Gate A error.
    for index, (left_row, right_row) in enumerate(zip(chad["rounds"], mlx["rounds"]), 1):
        for runtime, row in (("chad", left_row), ("mlx_dspark", right_row)):
            drafts, verify = row.get("draft_ids"), row.get("verify_ids")
            if drafts is None or verify is None:
                result["missing_control"] = f"R{index} missing {runtime} draft/verify IDs"
                return result
            if (not isinstance(drafts, list) or
                    verify != [row["anchor_or_pending_id"], *drafts] or
                    len(drafts) != row.get("draft_width") or
                    len(verify) != row.get("target_sequence_width")):
                result.update(result="fail", classification="same_runtime_verify_input_violation",
                              earliest_difference={"field": f"R{index}.{runtime}.verify_ids",
                                                   "draft_ids": drafts, "verify_ids": verify},
                              routing="localize_input_state")
                return result
        if left_row["draft_ids"] != right_row["draft_ids"]:
            result["proposal_diagnostics"].append({
                "round": index, "field": f"R{index}.draft_ids",
                "chad": left_row["draft_ids"], "mlx_dspark": right_row["draft_ids"],
                "chad_verify_ids": left_row["verify_ids"],
                "mlx_dspark_verify_ids": right_row["verify_ids"],
                "classification": "cross_runtime_implementation_dependent_diagnostic"})

    # A local failure is independently actionable; a cross-runtime draft
    # difference needs both references and evidence of distinct valid graphs.
    required_checks = {"prefill_ids", "anchor_id", "drafter_block_ids",
                       "drafter_block_dtype", "tap_shapes_dtype", "target_positions",
                       "drafter_positions", "target_cache_structure",
                       "drafter_cache_structure", "fused_shape_dtype"}
    for runtime, run in (("chad", chad), ("mlx_dspark", mlx)):
        local = run.get("r1_context_reference") or {}
        comparison, reference = local.get("comparison") or {}, local.get("reference") or {}
        if not comparison:
            if result["proposal_diagnostics"]:
                result["missing_control"] = f"missing {runtime} independent R1 context/proposal reference"
                return result
            continue
        result["same_runtime_r1_context"][runtime] = comparison
        evidence.extend(comparison.get("observation_ids") or [])
        exact = comparison.get("exact_input_checks") or {}
        context_exact = (comparison.get("context_sha256_equal") is True and
                         reference.get("pending_context", {}).get("sha256") ==
                         run["rounds"][0].get("pending_context", {}).get("sha256"))
        controls = local.get("five_run_context_controls") or []
        numeric = comparison.get("numeric_comparisons") or []
        by_component = {item.get("key", {}).get("component"): item for item in controls}
        controlled = (comparison.get("control_status") == "stable" and bool(numeric) and
                      set(by_component) == {item.get("component") for item in numeric} and
                      all(item.get("status") == "pass" and
                          item.get("distance", float("inf")) <=
                          by_component[item.get("component")].get("bound", -1)
                          for item in numeric) and
                      all(item.get("status") == "STABLE" and
                          len(item.get("samples") or []) == 5 and
                          item.get("bound") == max(item["samples"][:3]) and
                          all(value <= item["bound"] for value in item["samples"][3:])
                          for item in controls))
        if (not required_checks <= exact.keys() or not all(exact.values()) or
                comparison.get("reference_kind") !=
                "independent_same_runtime_r1_prefill_and_proposal" or
                reference.get("distinct_target_cache_objects") is not True or
                len(set(comparison.get("observation_ids") or [])) != 2 or
                reference.get("draft_ids") != run["rounds"][0].get("draft_ids") or
                comparison.get("proposal_ids_equal") is not True or
                comparison.get("status") != "pass" or
                not (context_exact or controlled)):
            result.update(result="fail" if runtime == "mlx_dspark" else "inconclusive",
                          classification=f"{runtime}_same_runtime_r1_reference_violation",
                          missing_control=None if runtime == "mlx_dspark" else
                                          "localize Chad's independent R1 reference",
                          routing=f"localize_{runtime}_r1_context")
            return result

    if result["proposal_diagnostics"]:
        unreferenced_rounds = [item["round"] for item in result["proposal_diagnostics"]
                               if item["round"] != 1]
        if unreferenced_rounds:
            result["missing_control"] = (
                f"independent same-runtime proposal references missing for rounds {unreferenced_rounds}")
            return result
        fastpath = chad.get("chad_fastpath_record") or {}
        replacements = fastpath.get("projection_replacement_flags") or {}
        distinct_graphs = (chad.get("implementation") == "chad" and
                           mlx.get("implementation") == "mlx-dspark" and
                           str(chad.get("target_loader", "")).startswith("chad.prism_pack.") and
                           str(mlx.get("target_loader", "")).startswith("mlx_dspark.prism_pack.") and
                           fastpath.get("engine_fastpath") is True and
                           fastpath.get("call_path_is_fastpath_module") is True and
                           any(value for value in replacements.values()))
        if not distinct_graphs:
            result["missing_control"] = "distinct valid target graph execution not established"
            return result
        result["target_graph_diagnostic"] = {
            "chad_fastpath_installed": True,
            "chad_projection_replacements": replacements,
            "chad_target_loader": chad["target_loader"],
            "mlx_target_loader": mlx["target_loader"]}
    result.update(result="pass", classification=(
                  "equivalent_exact_gate_a_fields_with_implementation_dependent_proposals"
                  if result["proposal_diagnostics"] else "equivalent_exact_gate_a_fields"),
                  missing_control=None, routing="B1", decision="continue")
    return result
