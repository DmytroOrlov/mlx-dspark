#!/usr/bin/env python3
"""Offline completeness validation and final rendering for frozen panel evidence."""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
RUNNER = Path(__file__).with_name("prompt_panel.py")
SPEC = importlib.util.spec_from_file_location("prompt_panel_frozen", RUNNER)
panel = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(panel)


REQUIRED_MEASUREMENTS = ("generated_tokens", "speculative_decode_tok_s", "acceptance", "target_forwards",
    "generated_per_forward", "rounds", "width_distribution", "cap_distribution", "request_duration_seconds", "peak_gib")


def observation_contract_mismatches(row: dict, pinned: dict, text: str, frozen_sha: str) -> list[str]:
    m = row.get("measurements", {})
    freshness = row.get("freshness", {})
    runtime = row.get("runtime_configuration", {})
    checks = {
        "prompt_text": row.get("text") == text,
        "input_ids": row.get("input_ids") == pinned.get("input_ids"),
        "input_sha256": row.get("input_sha256") == pinned.get("input_sha256"),
        "input_token_count": row.get("input_token_count") == pinned.get("input_token_count"),
        "generation_settings": row.get("request", {}).get("generation") == panel.SETTINGS,
        "physical_assertion": freshness.get("physical_asserted") is True,
        "fresh_process": freshness.get("new_process") is True,
        "fresh_engine": freshness.get("new_engine") is True,
        "no_acquire_prefix_reuse": freshness.get("acquire_reused_tokens") == 0,
        "no_request_prefix_reuse": row.get("request", {}).get("prefix_reuse_tokens") == 0,
        "target_revision": row.get("target", {}).get("revision") == panel.QWEN_REV,
        "drafter_revision": row.get("drafter", {}).get("revision") == panel.BQ_REV,
        "frozen_runner_sha256": row.get("source_sha256", {}).get("prompt_panel.py") == frozen_sha,
        "controller_present": runtime.get("controller_present") is True,
        "plain_kv_request": row.get("request", {}).get("plain_kv") is True,
        "plain_kv_runtime": runtime.get("kv_bits") in (None, 0),
        "required_measurements": all(k in m and m[k] is not None for k in REQUIRED_MEASUREMENTS),
    }
    return [name for name, passed in checks.items() if not passed]


def preflight_number(path: Path) -> int:
    match = re.fullmatch(r"preflight(?:-(\d+))?", path.stem)
    return int(match.group(1)) if match and match.group(1) else 1


def select_preflight(evidence: Path, source_hashes: set[str]) -> tuple[Path, dict]:
    if len(source_hashes) != 1:
        raise SystemExit(f"physical observations do not share one frozen runner SHA: {sorted(source_hashes)}")
    expected_sha = next(iter(source_hashes))
    candidates = []
    for path in evidence.glob("preflight*.json"):
        if not re.fullmatch(r"preflight(?:-\d+)?\.json", path.name):
            continue
        record = json.loads(path.read_text())
        if record.get("status") == "PASS" and record.get("runner_sha256") == expected_sha:
            candidates.append((preflight_number(path), path, record))
    if not candidates:
        raise SystemExit(f"no passing preflight matches physical runner SHA {expected_sha}")
    _, path, record = max(candidates, key=lambda item: item[0])
    return path, record


def main() -> int:
    data = panel.report_data()
    effective_dir = panel.latest_screen_dir()
    effective = [json.loads(p.read_text()) for p in sorted(effective_dir.glob("*-P[0-9][0-9]-screen.json"))]
    repeats = [row for row in data["observations"] if row.get("prompt_role") == "repeat"]
    physical_rows = [row for row in effective + repeats if row.get("status") != "FAILED"]
    observation_shas = {row.get("source_sha256", {}).get("prompt_panel.py") for row in physical_rows}
    preflight_path, preflight = select_preflight(panel.EVIDENCE, observation_shas)
    pinned = {row["prompt_id"]: row for row in preflight["prompts"]}
    expected = {pid: text for pid, _, text in panel.PROMPTS}
    missing = sorted(set(expected) - {row.get("prompt_id") for row in effective})
    duplicate = sorted(pid for pid in expected if sum(row.get("prompt_id") == pid for row in effective) != 1)
    failed, mismatch = [], []
    for row in effective:
        pid = row.get("prompt_id")
        if row.get("status") == "FAILED":
            failed.append(pid)
            continue
        fields = observation_contract_mismatches(row, pinned.get(pid, {}), expected.get(pid), preflight["runner_sha256"])
        if fields:
            mismatch.append({"prompt_id": pid, "fields": fields})
    screen_ids = {row.get("prompt_id") for row in effective if row.get("status") == "GOLDEN_CANDIDATE"}
    repeat_problems = []
    repeat_provenance_mismatches = []
    for pid in sorted(screen_ids):
        row = next((r for r in repeats if r.get("prompt_id") == pid), None)
        if row is None or row.get("status") == "FAILED" or row.get("measurements", {}).get("generated_tokens", 0) < panel.THRESHOLD_TOKENS or row.get("measurements", {}).get("speculative_decode_tok_s", 0) < panel.THRESHOLD_TOK_S:
            repeat_problems.append(pid)
        if row is not None and row.get("status") != "FAILED":
            fields = observation_contract_mismatches(row, pinned.get(pid, {}), expected.get(pid), preflight["runner_sha256"])
            if fields:
                repeat_provenance_mismatches.append({"prompt_id": pid, "fields": fields})
    # Anchors serve drift context only. Keep valid physical observations in the table;
    # failed early attempts remain in the full raw observation ledger below.
    data["anchor_observations"] = [r for r in data["anchor_observations"] if r.get("status") != "FAILED"]
    data["preflight_path"] = str(preflight_path.relative_to(ROOT))
    data["frozen_runner_sha256"] = preflight["runner_sha256"]
    data["attempt_history"] = [{"path": r.get("raw_result_path"), "prompt_id": r.get("prompt_id"),
        "role": r.get("prompt_role"), "status": r.get("status"), "error": r.get("error")}
        for r in data["observations"] if r.get("status") == "FAILED"]
    data["validation"] = {"effective_screen_dir": str(effective_dir.relative_to(ROOT)),
        "selected_preflight_runner_sha": preflight["runner_sha256"],
        "preflight_runner_sha_matches_observations": observation_shas == {preflight["runner_sha256"]},
        "screen_rows_expected": 17, "screen_rows_found": len(effective), "missing_prompt_ids": missing,
        "duplicate_prompt_ids": duplicate, "failed_prompt_ids": failed, "contract_mismatch_prompt_ids": mismatch,
        "repeat_rows": len(repeats), "candidate_prompt_ids": sorted(screen_ids),
        "missing_or_failed_candidate_repeats": repeat_problems,
        "repeat_provenance_mismatches": repeat_provenance_mismatches,
        "pass": not missing and not duplicate and not failed and not mismatch and not repeat_problems
            and not repeat_provenance_mismatches and observation_shas == {preflight["runner_sha256"]}
            and preflight["status"] == "PASS"}
    if not data["validation"]["pass"]:
        raise SystemExit("panel validation failed: " + json.dumps(data["validation"], sort_keys=True))
    json_path = panel.EVIDENCE / "prompt-panel.json"
    json_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    markdown = panel.report_markdown(data)
    marker = "Each prompt is shown individually. No aggregate throughput is used as a headline.\n"
    markdown = markdown.replace(marker, marker + "\nThe first two attempts failed before usable measurements; their failure artifacts remain preserved in the JSON observation ledger. Corrected attempt 3 supplies the results below. Frozen physical runner SHA-256: `" + preflight["runner_sha256"] + "`.\n")
    (panel.EVIDENCE / "prompt-panel.md").write_text(markdown)
    print(json.dumps({"validation": data["validation"], "counts": data["counts"],
        "json": str(json_path), "markdown": str(panel.EVIDENCE / "prompt-panel.md")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
