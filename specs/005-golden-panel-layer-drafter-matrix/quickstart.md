# Quickstart: Golden Panel Layer-Drafter Matrix

This is the planned execution sequence. No benchmark is run during planning.

## Components

- `evidence/probes/panel_matrix.py`: physical manifests, physical model-free checks, preflight, fixed first-pass orchestration, immutable raw cell writing, freshness/provenance/memory capture, and repeat execution from a manifest.
- `evidence/probes/finalize_matrix.py`: attempt discovery/effective-attempt selection, offline derivation, completeness/provenance validation, repeat candidate and manifest writing, explicit decision handling, repeat validation, and final JSON/Markdown tables and views. It performs no model inference.

## Execution sequence

1. Implement the physical runner.
2. Implement the offline finalizer.
3. Complete model-free checks for both components. Verify fixed identities/order, raw schema, derived arithmetic, repeat policy, report structure, and physical runner request-boundary memory instrumentation.
4. Assign a new attempt ID and create `evidence/attempts/<attempt_id>/`. Run cheap physical preflight into that directory at `preflight.json`, pinning current prompt IDs, checkpoints, settings, freshness behavior, output paths, interpreter identity/version, runtime, and runner SHA. Preflight performs no measured generation and is not shared with another attempt.
5. **Freeze `panel_matrix.py` after successful attempt-linked preflight and immediately before first-pass cell 1.** Record `attempt.json` in the same attempt directory with attempt ID, runner SHA, passing preflight identity/SHA, start/status, complete/incomplete/superseded state, expected count 70, observed/statused count, and supersession reason when applicable; finalize its lifecycle status when the attempt ends. Every first-pass raw file records that attempt ID, a reference to this directory's preflight, its SHA, and runner SHA. The finalizer SHA is tracked separately in derived/report artifacts.
6. Execute the 70 first-pass cells in the exact forward/reverse order documented in [plan.md](plan.md), each in a new process, Engine, and request, writing only to `evidence/attempts/<attempt_id>/cells/<prompt>/<ordered-cell>.json`.
7. The finalizer discovers attempt directories and selects only a complete/statused 70-cell attempt with exactly one runner SHA matching the passing preflight inside that same directory. It verifies every cell's preflight reference/SHA against that file, never combines cells across attempts or SHAs, and records the effective attempt ID in `evidence/attempts.json` and every derived/report artifact. Incomplete/superseded directories remain byte-for-byte preserved and visible in the global ledger. Derive each prompt only after its H0+B-Q raw baseline exists; P07/P14 normalize after the late baseline. Never edit raw. If arithmetic, validation, Pareto, or report code needs repair, preserve raw, patch the finalizer, and rerun offline only.
8. Once the effective first pass validates, the finalizer automatically selects (a) any non-H0 cell with first-pass tok/s >= its same-prompt H0+B-Q and (b) any H1b+B-B cell with retention >=97% of that same-prompt H0+B-Q. It also writes `repeat-candidates.json` for potential conclusion-critical near-drift comparisons and surprising local target maxima, with relevant raw identities/metrics but no subjective selection.
9. If additional subjective repeats are desired, write explicit `repeat-decisions.json` entries with prompt, target, drafter, triggering first-pass raw artifact identity/SHA, trigger class, and short reason. An absent/empty file selects none. The finalizer combines objective triggers and explicit decisions into immutable `repeat-manifest.json`; it applies no hidden thresholds or heuristics. Each comparison gets at most one initial repeat.
10. The physically frozen runner consumes only `repeat-manifest.json` and executes only its listed fresh matched repeats under unchanged settings, writing each to `evidence/attempts/<effective_attempt_id>/repeats/<comparison>.json`. Repeat records must identify the effective attempt ID and use its frozen runner SHA. If a runner defect requiring source change is found during repeats, preserve existing physical artifacts unchanged and mark the repeat unresolved unless a new full physical attempt is explicitly authorized; do not mix new-SHA repeats into the matrix.
11. The finalizer validates repeat provenance and creates final JSON/Markdown with the five required tables, both per-prompt layer paths, H1a/H1b/H1c interaction, repeat ledger, and descriptive Pareto view.

## Memory request boundary

Reuse the accepted Feature 003/004 MLX pattern. Finish model/drafter loading and normal warmup, establish fresh measured-request state, call `mx.metal.reset_peak_memory()`, capture `mx.metal.get_active_memory()` immediately before the measured request, execute only the measured speculative request, then immediately call `mx.metal.get_peak_memory()`. Raw evidence records the ordering/reset assertions and `peak_increment_bytes = peak_runtime_bytes - active_baseline_bytes`; all three values are also stored in GiB. Preflight/model-free checks verify this instrumentation. Same-prompt memory savings are finalizer-only calculations.

## Interpreter/runtime policy

Use the repository's `.venv/bin/python` for preflight and all physical child processes. Preflight pins its resolved identity and concrete version; every physical observation records and matches that pin. Do not provision Python 3.11 separately or silently switch interpreters/runtime between cells.

## Repair boundary

An offline-only defect preserves all physical raw evidence and requires only patching/rerunning the finalizer. If `panel_matrix.py` requires any source change after first-pass cell 1, stop the current attempt, preserve its preflight and all raw artifacts unchanged, finalize `attempt.json` as incomplete/superseded with a reason, and exclude it from the effective matrix. Repair the runner, rerun model-free physical checks, and create a new attempt directory with a new preflight, runner SHA, and attempt ID. Restart all 70 first-pass cells from cell 1 under the new frozen SHA; never overwrite or reuse paths from the prior attempt. Incomplete/superseded attempt directories remain visible in the global ledger.
