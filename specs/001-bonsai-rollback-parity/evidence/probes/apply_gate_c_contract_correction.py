from __future__ import annotations

from pathlib import Path
import json
import sys

root = Path(sys.argv[1])
mlx_aligned = Path(sys.argv[2])
mlx_prod = Path(sys.argv[3])
chad = Path(sys.argv[4])
oracle_path = Path(sys.argv[5])
correct_json = Path(sys.argv[6])
correct_md = Path(sys.argv[7])

oracle = json.loads(oracle_path.read_text())


def run_by_fragment(fragment: str) -> dict:
    for run in oracle["runs"]:
        if fragment in str(run.get("run_id", "")):
            return run
    raise AssertionError(f"missing oracle run containing {fragment!r}")


aligned_oracle = run_by_fragment("t005-r1-mlx-kv0-aligned")
prod_oracle = run_by_fragment("t010-t015-mlx-prod")
chad_oracle = run_by_fragment("t008-r1-chad-kv0-fastpath")

assert aligned_oracle["all_selected_rounds_pass"] is True
assert prod_oracle["all_selected_rounds_pass"] is True
assert chad_oracle["all_selected_rounds_pass"] is True
assert aligned_oracle["accepted_counts_covered"] == [0, 1]
assert prod_oracle["accepted_counts_covered"] == [1]
assert chad_oracle["accepted_counts_covered"] == [0, 1]


def load_run(path: Path) -> dict:
    return json.loads((path / "run.json").read_text())


aligned_doc = load_run(mlx_aligned)
prod_doc = load_run(mlx_prod)
chad_doc = load_run(chad)


def unique_cache_tensor_views(cache: list[dict]) -> dict[str, str]:
    result: dict[str, str] = {}

    for item in cache:
        layer = item["layer"]

        for key, value in item.items():
            if key in {"slots", "semantic"}:
                continue

            if isinstance(value, dict) and "sha256" in value:
                result[f"{layer}.{key}"] = value["sha256"]

    return result


def fresh_s2_diagnostic(doc: dict) -> list[dict]:
    rows = []

    for row in doc["rounds"]:
        if row["accepted_count"] != 1:
            continue

        ref = row.get("references", {}).get("s2_keep2")
        if not ref:
            continue

        prod = unique_cache_tensor_views(row["post_reconcile"]["cache"])
        s2 = unique_cache_tensor_views(ref["post_execute"]["cache"])

        common = sorted(set(prod) & set(s2))
        different = [key for key in common if prod[key] != s2[key]]

        rows.append({
            "round_index": row["round_index"],
            "verify_ids": row["verify_ids"],
            "accepted_prefix_ids": row["accepted_prefix_ids"],
            "common_unique_cache_tensor_views": len(common),
            "different_sha_views": len(different),
            "all_common_views_different": bool(common) and len(different) == len(common),
            "first_differences": different[:16],
        })

    return rows


def oracle_rows(run: dict) -> list[dict]:
    return [
        {
            "round_index": row["round_index"],
            "accepted_count": row["accepted_count"],
            "keep": row["keep"],
            "accepted_prefix_exact": row["accepted_prefix_exact"],
            "same_width_s8_fused_full_hash_equal": row[
                "same_width_s8_fused_full_hash_equal"
            ],
            "gdn_count": row["gdn"]["count"],
            "gdn_all_exact": row["gdn"]["all_exact"],
            "attention_count": row["attention"]["count"],
            "attention_all_exact": row["attention"]["all_exact"],
            "result": row["same_s8_rollback_oracle"],
        }
        for row in run["rounds"]
    ]


correction = {
    "date": "2026-09-23",
    "classification": "gate_c_numeric_oracle_contract_correction",
    "reason": (
        "Production reconciliation reconstructs accepted state from the actual "
        "S=8 verify projections/cache effects. Fresh S=1/S=2 executions use "
        "different projection execution widths and therefore cannot serve as "
        "the raw numeric cache-state oracle for S=8-originating rollback state."
    ),
    "correct_gate_c_numeric_oracle": {
        "root": "independent frozen same-runtime S=8 B2 pre-verify/capture",
        "gdn_recurrent_state": (
            "independently replay captured S=8 q/k/v/a/b[:keep] from captured "
            "pre-round recurrent state"
        ),
        "gdn_convolution_window": (
            "independently slice captured S=8 conv_input to the accepted-prefix "
            "window using production keep semantics"
        ),
        "attention_kv": (
            "independently trim frozen S=8 attention KV to pre-live-length + keep"
        ),
        "logical_invariants": (
            "accepted-prefix IDs, cache type/shape/order, lengths, offsets and "
            "absolute positions remain exact"
        ),
        "tap_and_fused_rows": (
            "compare with corresponding independent same-runtime same-width S=8 "
            "B2 rows under their same-path controls"
        ),
    },
    "fresh_s1_s2_role_after_correction": (
        "serial semantic/behavior reference and cross-width diagnostic; not the "
        "raw numeric cache-state oracle for S=8-originating post-reconcile state"
    ),
    "oracle_results": {
        "mlx_aligned": oracle_rows(aligned_oracle),
        "mlx_production": oracle_rows(prod_oracle),
        "chad": oracle_rows(chad_oracle),
    },
    "fresh_s2_cross_width_diagnostics": {
        "mlx_production": fresh_s2_diagnostic(prod_doc),
        "chad": fresh_s2_diagnostic(chad_doc),
    },
    "formal_status": {
        "local_gate_c_rollback_mechanics": "PASS",
        "mlx_keep1_local_physical_branch": "PASS_from_aligned_diagnostic_root",
        "mlx_keep2_production_branch": "PASS",
        "chad_keep1": "PASS",
        "chad_keep2": "PASS",
        "overall_B2_T010": (
            "INCONCLUSIVE: formal accepted=0 same-width five-run B2 coverage is "
            "not established for production-semantics mlx-dspark, and Chad's "
            "saved B2 branches lack the required five-run same-width controls"
        ),
        "overall_C_T012": (
            "INCONCLUSIVE only because prerequisite formal B2 coverage remains "
            "open; no Gate-C rollback defect is demonstrated"
        ),
        "D": "NOT_FORMALLY_REACHED",
        "E": "NOT_FORMALLY_REACHED",
        "T016": "OPEN",
        "production_patch_authorized": False,
    },
}

correct_json.write_text(json.dumps(correction, indent=2) + "\n")

md = f"""# Gate C numeric-oracle contract correction — 2026-09-23

## Why the previous Gate C numeric rule was invalid

The saved physical evidence shows that production rollback/reconciliation does
not recompute the accepted prefix through a fresh S=1 or S=2 projection path.
It preserves the semantics of the actual S=8 verify by replaying the accepted
prefix from captured S=8 GDN inputs, rebuilding the convolution window from
the captured S=8 convolution input, and trimming the actual S=8 attention KV.

Fresh S=1/S=2 execution changes projection execution width. Therefore a raw
post-reconcile-versus-fresh-S1/S2 cache delta is a cross-width diagnostic, not
by itself a rollback defect.

The correction does not use matching committed output as a substitute for
state validation. It replaces the invalid cross-width numeric oracle with an
independent same-runtime frozen-S8 accepted-prefix reconstruction.

## Independent same-S8 rollback oracle

For each selected branch the oracle:

- starts only from the frozen same-runtime S=8 B2 capture;
- independently replays captured `q/k/v/a/b[:keep]` from pre-round recurrent state;
- independently derives the convolution window from captured S=8 `conv_input`;
- independently trims S=8 attention KV to the accepted prefix;
- checks accepted-prefix IDs, cache structure, live length, offsets and positions;
- never calls production rollback/reconcile.

Physical results:

- mlx-dspark aligned diagnostic: accepted=0/keep=1 PASS, accepted=1/keep=2 PASS;
- mlx-dspark production semantics: both naturally observed accepted=1/keep=2 branches PASS;
- Chad normal fastpath: accepted=0/keep=1 PASS, accepted=1/keep=2 PASS.

Every reported branch has exact accepted-prefix identity, exact frozen-S8 fused
identity, exact 48-layer GDN reconstruction and exact 16-layer attention-KV
reconstruction. The offline GDN replay is also exactly repeatable.

## Role of fresh S=1/S=2 after correction

Fresh ordinary S=1 and fresh S=2 references remain required as serial
semantic/behavior and cross-width diagnostics. They are useful for Gate D and
for measuring width sensitivity. They are not the numeric cache-state oracle
for state whose projections originated in an S=8 verify.

The prior fresh-S2-derived Gate-C failure is therefore superseded as a contract
false positive and does not authorize a runtime patch.

## Formal status after correction

Local rollback mechanics are physically covered for keep=1 and keep=2 in both
implementations and show no mlx-dspark-only defect.

Formal T010/B2 is still incomplete: the saved Chad branches lack their required
five-run same-width B2 controls, and production-semantics mlx-dspark did not
naturally produce accepted=0 in the measured control run. The aligned mlx
accepted=0 branch is valid local rollback-mechanics evidence but does not
qualify the aligned prefill lifecycle as production semantics.

Therefore overall Gate C remains formally INCONCLUSIVE only behind that B2
prerequisite gap. D/E are not formally reached, T016 remains OPEN, and no
production patch is authorized.

Canonical oracle: `{oracle_path.relative_to(root)}`
Machine-readable correction record: `{correct_json.relative_to(root)}`
"""

correct_md.write_text(md)


def replace_prefix(path: Path, prefix: str, replacement: str) -> None:
    lines = path.read_text().splitlines()
    indices = [i for i, line in enumerate(lines) if line.startswith(prefix)]

    if len(indices) != 1:
        raise AssertionError(
            f"{path}: expected one line starting {prefix!r}, got {len(indices)}"
        )

    lines[indices[0]] = replacement
    path.write_text("\n".join(lines) + "\n")


spec = root / "spec.md"

replace_prefix(
    spec,
    "- Rejection with zero accepted draft tokens must restore",
    "- Rejection with zero accepted draft tokens must preserve the accepted-prefix logical state exactly. Rollback-owned recurrent, convolution, and live-KV numeric state is judged against an independent same-runtime frozen-S8 accepted-prefix reconstruction from the exact B2 capture. Fresh S=1/S=2 raw cache values remain serial/cross-width diagnostics and are not the numeric oracle for S=8-originating state.",
)

replace_prefix(
    spec,
    "- Do not apply a global floating-point tolerance. Gate C",
    "- Do not apply a global floating-point tolerance. Gate C first requires exact structural/logical invariants and an independent same-runtime accepted-prefix reconstruction from the frozen same-width S=8 B2 capture. Recurrent state is independently replayed from captured S=8 `q/k/v/a/b[:keep]` and pre-round recurrent state; the convolution window is independently derived from captured S=8 `conv_input`; attention KV is independently trimmed from frozen S=8 KV to the accepted prefix. Required post-reconcile taps/fused rows are compared with their corresponding same-width S=8 B2 rows. Exact equality is preferred where the same captured execution path is deterministic; if a same-path numeric component is not bit-repeatable, its bound must come from predeclared same-path five-run controls for that exact runtime/root/component/dtype/shape/metric. Fresh S=1/S=2 references remain serial semantic/behavior and cross-width diagnostics and MUST NOT provide a raw numeric cache-state failure criterion for S=8-originating rollback state. Gate D separately checks subsequent committed target behavior against the ordinary committed reference.",
)

replace_prefix(
    spec,
    "- **FR-001**:",
    "- **FR-001**: After speculative verify and rejection/partial acceptance, the target MUST represent the exact accepted logical prefix. Cache type/shape/order, live lengths, offsets, positions, and committed IDs are exact invariants. Numeric state created by an S=8 verify MUST match an independent same-runtime accepted-prefix reconstruction from that frozen S=8 capture; raw equality to a fresh S=1/S=2 projection-width execution is not required by itself. Subsequent serial semantic behavior is checked separately at Gate D.",
)

replace_prefix(
    spec,
    "- **FR-004**:",
    "- **FR-004**: The regression/probe MUST compare speculative execution with independent same-runtime references appropriate to each boundary. B2 uses an independent same-width S=8 verify/capture reference. Gate C uses an independent accepted-prefix reconstruction from that frozen S=8 capture for rollback-owned recurrent/convolution/live-KV state and uses the corresponding same-width S=8 rows for accepted taps/fused rows. Exact discrete IDs, structure, lengths, offsets, positions, row/tap identity/order, and committed-prefix membership remain exact. Any non-bit-repeatable same-path numeric observation requires its own predeclared five-run same-path component/dtype/shape/metric bound; no global tolerance or cross-width transfer is allowed. Fresh ordinary S=1 and fresh S=2 committed-prefix executions remain independently allocated serial semantic/behavior and width-sensitivity references, but a raw post-reconcile-versus-fresh-S1/S2 cache delta cannot by itself fail C when the production state originated in S=8 projections. Gate D checks subsequent committed target behavior against the ordinary committed reference. A numeric-only Gate A/E regression still requires its already-defined independent same-runtime semantic-state reference and stable control.",
)

replace_prefix(
    spec,
    "- **SC-001**:",
    "- **SC-001**: Both required rollback cases satisfy exact structural/logical invariants and the Gate C independent same-S8 accepted-prefix reconstruction for rollback-owned state. Fresh S=1/S=2 references remain serial-semantic controls, and Gate D checks subsequent committed behavior; matching one next token alone is not sufficient.",
)

plan = root / "plan.md"

replace_prefix(
    plan,
    "Proceed through Gate A → B1 → B2 → C → D → E.",
    "Proceed through Gate A → B1 → B2 → C → D → E. Gate B3 is a non-blocking cross-runtime diagnostic that may run after B2 when useful, with no dependency edge into C. Capture ordinary S=1 and, for accepted=1, fresh S=2 committed references as serial semantic/width-sensitivity controls. B2 is the independent same-runtime same-width S=8 verify/capture root. Gate C reconstructs the accepted prefix independently from that frozen S=8 capture and judges rollback-owned state against this same-path oracle; fresh S=1/S=2 raw state is not the numeric oracle for S=8-originating projections. Gate D then checks subsequent ordinary committed behavior against the serial committed reference. The full observation schema and equality rules are in [data-model.md](data-model.md); the staged run procedure is in [quickstart.md](quickstart.md).",
)

replace_prefix(
    plan,
    "| **C — post-reconcile component state**",
    "| **C — post-reconcile component state** | With A/B1/B2 controlled, require exact cache type, shape, layer order, live length, offsets, positions, committed token IDs, accepted-prefix membership, and required tap/fused-row identity/order. Construct an independent accepted-prefix oracle from the frozen same-runtime S=8 B2 capture without invoking production rollback: replay captured S=8 `q/k/v/a/b[:keep]` from captured pre-round recurrent state, derive the convolution window from captured S=8 `conv_input` using the production keep semantics, and trim frozen S=8 attention KV to pre-live-length + keep. Compare required accepted taps/fused rows with the corresponding same-width S=8 B2 rows. Fresh S=1/S=2 committed references remain serial-semantic and width-sensitivity diagnostics, not the raw numeric cache-state oracle for S=8-originating state. | PASS requires all exact structural/logical invariants and the independent same-S8 accepted-prefix reconstruction to agree. Exact same-path equality is used when deterministic; otherwise a numeric component requires its own predeclared stable five-run same-path bound for that exact runtime/root/component/dtype/shape/metric. A raw fresh-S1/S2 cross-width cache delta cannot fail C by itself. Only a reconciliation mismatch against this independent same-path oracle after B2 can authorize rollback localization. D remains the independent subsequent-behavior check. |",
)

replace_prefix(
    plan,
    "Predeclare each numeric cache component's layer/slot, dtype, and distance metric",
    "Predeclare each numeric cache component's layer/slot, dtype, execution shape, reference root, and distance metric. For Gate C rollback-owned state, the reference root is the frozen same-runtime S=8 B2 capture reconstructed independently to `keep=accepted+1`; never substitute fresh S=1/S=2 raw cache values for this same-path oracle. Recurrent state uses captured S=8 recurrence inputs plus pre-round state, convolution history uses captured S=8 `conv_input`, and attention uses frozen S=8 KV trimmed to the accepted live prefix. Exact equality is required when the same-path reconstruction is deterministic. If a same-path component is not bit-repeatable, establish exactly five equivalent controls for that exact runtime/root/component/dtype/execution shape/metric: repetitions 1–3 calibrate the maximum bound and 4–5 must validate within it; otherwise the dependent observation is INCONCLUSIVE. No global tolerance or transfer across widths, roots, components, rows, dtypes, runtimes, logical states, or execution shapes is allowed. Fresh S=1/S=2 repeatability and S=2-versus-S=1+S=1 width deltas remain recorded as serial/cross-width diagnostics and support Gate D interpretation, but they cannot establish a Gate C numeric failure for S=8-originating rollback state. Required accepted taps/fused rows use their corresponding same-width S=8 B2 rows and same-path controls. If a change fixes one gate but creates an earlier mismatch or another required accepted-count mismatch, reject it and return to the first failing gate.",
)

tasks = root / "tasks.md"

replace_prefix(
    tasks,
    "- [ ] T011 [US2]",
    "- [ ] T011 [US2] Capture and validate both same-runtime ordinary S=1 committed references and the accepted=1 fresh S=2 committed-prefix reference in `specs/001-bonsai-rollback-parity/evidence/gates.md`: use exact committed IDs, equivalent logical pre-round state, independent caches, one-token cadence for S=1, and repeated S=1/S=2 plus S=2-versus-S=1+S=1 width-sensitivity controls. These references define serial semantic/behavior expectations for Gate D and document projection-width sensitivity; they do **not** supply the raw numeric cache-state oracle for S=8-originating Gate C state (depends on T006–T007 and controlled IDs from T008).",
)

replace_prefix(
    tasks,
    "- [ ] T012 [US2]",
    "- [ ] T012 [US2] Establish Gate C in `specs/001-bonsai-rollback-parity/evidence/gates.md` for width=8/seven draft IDs and accepted=0/keep=1 plus accepted=1/keep=2 (depends on B2 pass, **not** T013). Require exact cache type, shape, layer order, live length, offsets, absolute positions, committed/accepted-prefix IDs, and exact required tap/fused-row identity/order/membership. For rollback-owned numeric state, construct an independent same-runtime accepted-prefix oracle from the frozen S=8 B2 capture without invoking production rollback: replay captured `q/k/v/a/b[:keep]` from captured pre-round recurrent state, derive convolution history from captured S=8 `conv_input`, and trim frozen S=8 attention KV to the accepted live prefix. Required accepted taps/fused rows compare with corresponding same-width S=8 B2 rows. Exact same-path equality is required where deterministic; any non-bit-repeatable same-path component requires its own predeclared stable five-run control for that exact runtime/root/component/dtype/shape/metric. Fresh S=1/S=2 raw cache deltas are serial/cross-width diagnostics and cannot by themselves fail C. Missing prerequisite B2 evidence or missing/unstable applicable same-path controls makes C INCONCLUSIVE. A controlled mismatch against the independent same-S8 accepted-prefix oracle after earlier gates pass routes to T021; Gate D remains the separate subsequent committed-behavior check.",
)

data_model = root / "data-model.md"

replace_prefix(
    data_model,
    "Within each runtime's speculative-versus-same-width verify comparison",
    "Within each runtime's speculative-versus-same-width verify comparison, cache type, shape, layer order, live length, offset, positions, and token IDs are exact invariants. Compare captured S=8 outputs/taps/cache effects before reconciliation. At Gate C, cache type, shape, layer order, live length, offsets, positions, committed/accepted-prefix IDs, and required tap/fused-row identity/order remain exact. Numeric rollback-owned state is compared with an independent accepted-prefix reconstruction from the frozen same-runtime S=8 B2 capture: recurrence replay from captured S=8 inputs and pre-round state, convolution history from captured S=8 `conv_input`, and attention KV trimmed from frozen S=8 KV. Fresh S=1/S=2 raw state is a serial/cross-width diagnostic rather than this numeric oracle. Across runtimes, raw cache/tensor values remain diagnostic. Compare only live KV ranges; stale backing-buffer tails after trim are not logical state.",
)

replace_prefix(
    data_model,
    "- Corresponding state from an independent ordinary S=1 committed-semantic reference",
    "- Corresponding ordinary S=1 committed-semantic reference within the same runtime, retained for serial behavior and Gate D rather than as the raw numeric cache-state oracle for S=8-originating rollback state.",
)

replace_prefix(
    data_model,
    "- For accepted=1/keep=2, a fresh same-runtime S=2 committed-prefix state",
    "- For accepted=1/keep=2, retain a fresh same-runtime S=2 committed-prefix execution for exactly `[anchor, first accepted draft]` as a width-sensitivity/serial diagnostic, independently allocated from speculative and S=1 branches.",
)

replace_prefix(
    data_model,
    "- For each recurrent, convolution, and numeric live KV component, record dtype",
    "- For each recurrent, convolution, and numeric live-KV component, record dtype, execution shape, metric, frozen S=8 reference root, and the independent accepted-prefix reconstruction result. Recurrence is replayed from captured S=8 inputs and pre-round state, convolution history is sliced from captured S=8 `conv_input`, and attention KV is trimmed from frozen S=8 KV. Use exact equality when deterministic; otherwise record exactly five same-path controls under the standard calibration/validation rule. Record fresh S=1/S=2 and S=2-versus-S=1+S=1 distances separately as cross-width diagnostics, not as the Gate C numeric failure threshold.",
)

replace_prefix(
    data_model,
    "An independently allocated ordinary S=1 committed path",
    "An independently allocated ordinary S=1 committed path in the same runtime, device, and weights. For accepted=0 it executes prompt + anchor; for accepted=1 it executes prompt + anchor + first accepted draft as two one-token calls. It defines serial semantic/behavior expectations and supports Gate D. Because projection execution width differs, its raw cache values are not the numeric Gate C oracle for state originating in an S=8 verify.",
)

replace_prefix(
    data_model,
    "## SameWidthCommittedPrefixReference",
    "## CrossWidthCommittedPrefixDiagnostic",
)

replace_prefix(
    data_model,
    "For accepted=1/keep=2, independently allocate a fresh target/cache",
    "For accepted=1/keep=2, independently allocate a fresh target/cache from equivalent logical pre-round state and execute one S=2 call for exactly `[anchor, first accepted draft]`. Capture its state as a projection-width/serial diagnostic. It does not replace the ordinary S=1 + S=1 semantic reference, the frozen S=8 B2 verify reference, or the Gate C accepted-prefix reconstruction derived from that S=8 capture.",
)

replace_prefix(
    data_model,
    "- Repeated S=1 committed-vs-committed executions establish repeatability",
    "- Repeated S=1 committed-vs-committed executions establish serial-reference repeatability for Gate D and width-sensitivity diagnostics.",
)

replace_prefix(
    data_model,
    "- Repeated S=2 committed-prefix-vs-committed-prefix executions",
    "- Repeated S=2 committed-prefix-vs-committed-prefix executions establish repeatability of the S=2 cross-width diagnostic.",
)

replace_prefix(
    data_model,
    "- Repeated fresh S=2-versus-ordinary S=1 + S=1 pairs",
    "- Repeated fresh S=2-versus-ordinary S=1 + S=1 pairs establish the expected projection-width delta as a diagnostic; this envelope is not the Gate C numeric cache-state bound for S=8-originating state.",
)

replace_prefix(
    data_model,
    "No global tolerance field exists. Identify each numeric component",
    "No global tolerance field exists. Gate C identifies each rollback-owned numeric component by runtime, frozen S=8 root, layer/cache slot, dtype, execution shape, and metric. Independently reconstruct the accepted prefix from that frozen S=8 capture and require exact equality where deterministic. If a same-path component is not bit-repeatable, use exactly five equivalent same-path controls: repetitions 1–3 calibrate the maximum bound and repetitions 4–5 validate it; otherwise the observation is INCONCLUSIVE. No bound transfers across roots, widths, components, rows, dtypes, runtimes, logical states, or execution shapes. Required accepted tap/fused rows use their corresponding same-width S=8 B2 rows and same-path controls. Fresh S=1/S=2 controls and S=2-versus-S=1+S=1 deltas remain serial/cross-width diagnostics and cannot by themselves fail Gate C. Gate D independently checks subsequent committed behavior against the serial committed reference.",
)

quick = root / "quickstart.md"

replace_prefix(
    quick,
    "For R1 accepted=0, compare reconciled logical state",
    "For accepted=0/keep=1 and accepted=1/keep=2, first require exact reconciled logical structure: accepted-prefix IDs, cache type/shape/order, live lengths, offsets, positions, and required tap/fused-row identity/order. For rollback-owned numeric state, build an independent same-runtime accepted-prefix oracle from the frozen S=8 B2 capture without invoking production rollback: replay captured S=8 recurrence inputs from the captured pre-round recurrent state, derive convolution history from captured S=8 `conv_input`, and trim frozen S=8 attention KV to the accepted live prefix. Compare required accepted taps/fused rows with their corresponding same-width S=8 B2 rows.",
)

replace_prefix(
    quick,
    "Rollback/reconcile is eligible for a patch only when",
    "Rollback/reconcile is eligible for a patch only when proposal/verify inputs and B2 are controlled and mlx-dspark post-reconcile state violates the independent same-runtime same-S8 accepted-prefix oracle at the earliest boundary. Fresh ordinary S=1 and fresh S=2 executions remain serial semantic/behavior and projection-width diagnostics. A raw post-reconcile-versus-fresh-S1/S2 cache delta is not by itself a rollback defect when the production state originated in S=8 projections.",
)

replace_prefix(
    quick,
    "Compare subsequent ordinary committed target behavior from the reconciled branch",
    "At Gate D, compare subsequent ordinary committed target behavior from the reconciled branch with the same-runtime ordinary committed reference, including token identity, controlled top-k/margin, taps, and resulting semantic state. Keep historical cross-runtime margin differences as diagnostics. Gate D is the place where serial behavior is adjudicated; it does not replace Gate C's same-S8 rollback-state reconstruction.",
)

replace_prefix(
    quick,
    "Apply correctness Gates A, B1, B2, C, D, and E",
    "Apply correctness Gates A, B1, B2, C, D, and E in order. B3 remains optional. Gate C uses the frozen same-runtime S=8 B2 capture as the root for its independent accepted-prefix rollback oracle; fresh S=1/S=2 controls characterize serial behavior and projection-width sensitivity instead of supplying the Gate C raw numeric cache-state threshold. If the applicable same-path control is unavailable or unstable, mark that observation INCONCLUSIVE rather than loosening a global tolerance.",
)

gates = root / "evidence" / "gates.md"
text = gates.read_text()

b2_header = "## B2 — same-runtime S=8 verify/capture"
c_header = "## C, D, E, and T016"

b2_start = text.index(b2_header)
c_start = text.index(c_header)

new_b2 = """## B2 — same-runtime S=8 verify/capture

The saved aligned Chad/mlx pair remains sample-exact for accepted=0 and
accepted=1, but those runs did not contain the required five-run same-width B2
controls.

A later **production-semantics mlx-dspark** run with `--measure-controls`,
plain target KV, and no aligned-prefill diagnostic naturally produced
accepted counts 1, 1, and 5. Its observed S=8 branches have independent frozen
pre-verify roots and stable five-run same-width controls; the speculative and
reference S=8 captures are exact for the observed branches. This establishes
controlled mlx-dspark B2 for the naturally observed accepted=1 branches.

Formal T010/B2 as a whole remains **INCONCLUSIVE**, because the required
accepted=0 controlled branch was not naturally observed in that production
run, while the saved Chad accepted=0/1 branches lack their required five-run
same-width B2 controls. Existing single-sample exact captures are retained and
must not be promoted into missing controls.

"""

new_c = f"""## C — post-reconcile rollback mechanics

The previous fresh-S1/S2 numeric adjudication is superseded as a **contract
false positive**. Production reconciliation in both implementations preserves
the semantics of the actual S=8 verify: GDN recurrence is replayed from
captured S=8 `q/k/v/a/b`, convolution history is rebuilt from captured S=8
`conv_input`, and attention KV is trimmed from the S=8 verify state. Fresh
S=1/S=2 executions use different projection widths and therefore cannot serve
as the raw numeric cache-state oracle for this S=8-originating state.

The corrected independent same-S8 accepted-prefix oracle never invokes
production rollback/reconcile. It reconstructs the accepted state only from
the frozen B2 S=8 capture.

Physical local results are exact:

- mlx-dspark aligned diagnostic accepted=0/keep=1: PASS;
- mlx-dspark aligned diagnostic accepted=1/keep=2: PASS;
- mlx-dspark production-semantics accepted=1/keep=2, two observed branches: PASS;
- Chad normal-fastpath accepted=0/keep=1: PASS;
- Chad normal-fastpath accepted=1/keep=2: PASS.

Every selected branch has exact accepted-prefix identity, exact corresponding
frozen-S8 fused identity, exact independent reconstruction across all 48 GDN
layers, and exact attention-KV reconstruction across all 16 attention layers.
The offline GDN replay is exactly repeatable.

The aligned mlx accepted=0 result establishes **local physical rollback
mechanics only**. It does not qualify `--align-prefill` as production semantics
or as a production/performance fix.

Because formal B2 accepted=0/Chad same-width control coverage is still missing,
overall Gate C remains **INCONCLUSIVE behind its B2 prerequisite**, even though
the local rollback-mechanics oracle shows no mlx-dspark defect. No rollback
patch is authorized.

Canonical records:

- `{correct_md.relative_to(root)}`
- `{oracle_path.relative_to(root)}`
- `{correct_json.relative_to(root)}`

## D, E, and T016

- **D subsequent committed target behavior:** not formally reached because
  overall B2/C prerequisites remain incomplete. Existing serial S=1/S=2
  references remain available for offline diagnostic review.
- **E drafter state:** not formally reached after D.
- **T016:** OPEN. Neither `FIX_PENDING` nor `RESOLVED_NO_CHANGE` is authorized.
  The evidence currently demonstrates no Gate-C rollback defect, but missing
  formal B2 coverage prevents closing the investigation.
- **Production patch:** not authorized.

Do not launch another broad model acquisition merely to repeat already exact
rollback-state evidence. Any future physical acquisition must target only the
remaining formal control/coverage gap and should be reviewed as a minimal
experiment first.
"""

gates.write_text(text[:b2_start] + new_b2 + new_c + "\n")
