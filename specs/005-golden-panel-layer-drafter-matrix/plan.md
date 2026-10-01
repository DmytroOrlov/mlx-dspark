# Implementation Plan: Golden Panel Layer-Drafter Matrix

**Branch**: `005-golden-panel-layer-drafter-matrix` | **Date**: 2026-09-26 | **Spec**: [spec.md](spec.md)

## Summary

Use two feature-local components. `panel_matrix.py` owns physical preparation and inference observations; `finalize_matrix.py` owns offline derivation, validation, repeat selection/manifest writing, and reporting. Reuse the accepted Feature 003 composition/cell patterns and Feature 004 prompt, tokenizer, freshness, MLX memory, and report-validation patterns. Capture exactly 70 immutable first-pass observations, then derive prompt-matched H0+B-Q comparisons and selectively run repeats. No production runtime change is planned.

## Technical Context

**Language/Version**: Repository `.venv/bin/python` and current accepted experiment runtime. Record concrete interpreter/version in preflight and every physical observation; do not create a separate Python-version environment.
**Primary Dependencies**: Existing mlx-dspark runtime, MLX and Transformers; standard-library JSON, hashes, subprocess, and report generation.
**Storage**: Feature-local JSON evidence and Markdown under `specs/005-golden-panel-layer-drafter-matrix/evidence/`.
**Checks**: Model-free assertions, physical preflight, offline matrix/provenance validation, and report consistency. No benchmark during planning.
**Target Platform**: Existing Apple Silicon / MLX experiment host and normal repository runtime.
**Project Type**: Internal experiment runner and evidence/report tooling.
**Performance Goals**: Primary metric is speculative decode tok/s. Every physical observation measures active baseline, request-boundary peak runtime, and peak increment memory.
**Constraints**: Exactly five pinned prompts × seven accepted targets × two pinned drafters; a fresh process, Engine, and request per cell; fixed sampling/controller/KV settings; no serial pass, tuning, WidthPolicy, KV8, or production changes. Raw evidence is immutable and contains no derived comparisons. All cells use the same interpreter/runtime identity recorded at preflight.
**Scale/Scope**: 70 mandatory first-pass physical observations; at most one initial fresh matched repeat per triggered comparison.

## Constitution Check

- **I, throughput is the decision metric**: Pass. Speculative decode tok/s is primary; mechanisms and memory explain results. Prompt-level values remain visible, without a headline aggregate score.
- **II, one axis at a time**: Pass. Target composition and drafter are the compared axes; runtime controls remain fixed.
- **III, preserve vanilla path**: Pass. No production source change; only feature-local experiment tooling is added.
- **IV, representation/tap semantics**: Pass by relying on Feature 003's accepted composition/provenance; no correctness archaeology is reopened.
- **V, runtime integrity**: Pass. Preflight and validation cover pinned identities, settings, freshness, metrics, and provenance; failed cells remain visible and block complete-panel claims.
- Matched and hardware-backed evidence: Pass. All cells use fresh processes on one host, frozen physical-runner source, and immutable checkpoint revisions.

No constitution exceptions or complexity justification are required.

## Component Boundaries

### Physical runner: `evidence/probes/panel_matrix.py`

Owns prompt/target/drafter manifests; model-free checks required for physical execution; cheap preflight; one-cell physical execution; the fixed 70-cell orchestration and stipulated order; immutable raw cell writing; physical freshness/provenance; request-boundary memory capture; and actual matched-repeat execution.

This is the only source file physically frozen for measurement comparability. Every first-pass and repeat raw observation records its SHA. The runner consumes an explicit immutable `repeat-manifest.json`; it does not decide triggers or calculate same-prompt memory savings.

### Offline finalizer: `evidence/probes/finalize_matrix.py`

Discovers immutable physical attempt directories and selects only one complete 70-cell first-pass attempt whose cells share one physical runner SHA matching that attempt's passing preflight. Each attempt directory contains its own `attempt.json`, `preflight.json`, first-pass cells, and repeats; no physical output path is shared across attempts. It preserves a global attempt ledger, including incomplete/superseded attempts, then derives same-prompt H0+B-Q values into feature-level `derived-matrix.json`; validates raw provenance and derived arithmetic; builds repeat candidates and accepts explicit offline repeat decisions; combines those decisions with objective automatic triggers into feature-level `repeat-manifest.json`; validates repeats; and renders final JSON/Markdown, the five required tables, per-prompt layer paths, H1a/H1b/H1c interaction, and descriptive Pareto view. Every derived/report artifact records the effective attempt ID.

The finalizer never performs model inference. Its SHA is recorded separately in derived/report artifacts. An offline-only repair preserves all raw evidence and requires only rerunning offline derivation/validation/reporting. It never combines first-pass cells across attempts or runner SHAs.

## Reused Patterns and Decisions

1. Carry forward Feature 003's `DONOR_BLOCKS`, `target_record`, checkpoint verification, child process, source hashing, and cell-provenance approach. Preserve its accepted target family and B0 identity without modifying Feature 003.
2. Carry forward Feature 004's exact prompt definitions, chat-template tokenization/preflight, freshness and output digest fields, and offline validation/report conventions. Preserve accepted certification; do not reselect prompts.
3. Keep raw and derived evidence separate. Each physical attempt writes its own immutable file containing measured/provenance fields only. Derived normalization references that file and SHA; raw is never enriched after measurement.
4. **Memory semantics**: Reuse accepted MLX semantics from Feature 003/004 (`mx.metal.reset_peak_memory()`, `get_active_memory()`, `get_peak_memory()`). For every cell: (a) complete model/drafter loading and normal warmup; (b) establish fresh measured-request state; (c) reset the MLX peak counter at that request boundary; (d) capture active memory immediately before the measured request; (e) execute only the measured speculative request; (f) immediately capture the request-boundary peak. Store boundary/order and reset assertions proving the peak interval. Raw increment is `peak_runtime_bytes - active_baseline_bytes`, in bytes and GiB. Preflight/model-free checks verify this instrumentation. Same-prompt memory deltas/savings are calculated only by the offline finalizer.
5. Use `.venv/bin/python` for preflight and child processes. Record its resolved interpreter identity and concrete version in preflight and each cell; refuse physical execution if runtime identity differs from the preflight pin.

## Execution Stages

1. Implement the physical runner, manifests, raw schema, physical model-free checks, and preflight.
2. Implement the offline finalizer, including derivation, validation, trigger evaluation/manifest writing, repeat validation, and all final reporting.
3. Complete model-free checks for both components, including exact matrix/order assertions, formulas, repeat triggers, report structure, and a static assertion for request-boundary memory instrumentation.
4. Run cheap physical preflight. Pin current tokenizer IDs, checkpoint identities, settings, runtime/interpreter identity, output paths, freshness, and memory instrumentation. Preflight performs no measured generation.
5. **Physical source-freeze point**: assign a new attempt ID, link the passing physical preflight to that attempt, and after preflight passes and records `panel_matrix.py` SHA, freeze the runner immediately before first-pass cell 1. Freeze only the physical runner; record finalizer SHA separately when generating offline artifacts.
6. Execute exactly 70 first-pass cells, each with a new process, Engine, and fresh measured request. Use production prefix-cache initialization with zero useful prefix reuse; `thinking=false`, temperature 0, `top_p=1`, `top_k=0`, `max_tokens=512`, plain KV, ordinary automatic CapController, no WidthPolicy, no KV8, no tuning. Do not run a serial pass per cell.
7. Preserve this exact physical order and adjacent pairs. Forward prompts P05/P08/P17: 1 H0+B-Q; 2 H0+B-B; 3 H1a+B-B; 4 H1a+B-Q; 5 H1b+B-Q; 6 H1b+B-B; 7 H1c+B-B; 8 H1c+B-Q; 9 H2+B-Q; 10 H2+B-B; 11 H3+B-B; 12 H3+B-Q; 13 B0+B-Q; 14 B0+B-B. Reverse prompts P07/P14: 1 B0+B-Q; 2 B0+B-B; 3 H3+B-B; 4 H3+B-Q; 5 H2+B-Q; 6 H2+B-B; 7 H1c+B-B; 8 H1c+B-Q; 9 H1b+B-Q; 10 H1b+B-B; 11 H1a+B-B; 12 H1a+B-Q; 13 H0+B-Q; 14 H0+B-B.
8. The finalizer discovers `evidence/attempts/<attempt_id>/` directories and accepts only one complete, statused 70-cell attempt whose cells and attempt metadata agree on exactly one runner SHA and whose per-attempt preflight passes. It verifies all 70 cells against `attempts/<attempt_id>/preflight.json`, selects no cells from outside that directory, and records the effective attempt ID in the feature-level global ledger and every derived/report artifact. It never merges cells across attempts or SHAs; incomplete/superseded directories remain byte-for-byte preserved and visible in `evidence/attempts.json`. Derive each prompt's comparisons only after its H0+B-Q raw observation exists; for reverse P07/P14 wait for the late baseline. Never rewrite raw. Offline-only defects are repaired in the finalizer and rerun offline without physical reruns.
9. After validating the effective first-pass attempt, evaluate the two objective automatic repeat triggers offline: (a) any non-H0 cell with tok/s >= its same-prompt H0+B-Q; (b) any H1b+B-B cell with throughput retention >=97% of that same-prompt H0+B-Q. Also write `repeat-candidates.json` for potential conclusion-critical near-drift comparisons and surprising local target maxima, with relevant raw identities/metrics but no subjective selection. If desired, record selected subjective candidates in `repeat-decisions.json`, each with prompt, target, drafter, triggering first-pass raw identity/SHA, trigger class, and short explicit reason. An absent/empty decisions file adds no subjective selections. Combine the objective triggers and explicit decisions into immutable `repeat-manifest.json`; at most one initial repeat per comparison. The frozen physical runner executes only manifest-listed fresh matched repeats.
10. Offline validate repeat provenance and generate final JSON/Markdown: exactly the five required tables (Raw Prompt × Config Matrix; Throughput Retention + Memory; Drafter Delta; Mechanism Breakdown; Speed / Memory Tradeoff), per-prompt B-Q/B-B paths, H1a/H1b/H1c interaction, descriptive Pareto view, repeat ledger, and validation summary. Keep prompt cells visible before summaries; no weighted score or headline aggregate.

If `panel_matrix.py` requires any source change after first-pass cell 1: stop the current attempt; preserve every existing preflight and raw cell/repeat artifact byte-for-byte; finalize its `attempt.json` as incomplete/superseded with the reason and never use or merge it into the effective matrix; repair the runner; rerun model-free physical checks and create a new attempt directory with its own passing preflight; obtain a new runner SHA and attempt ID; then restart all 70 first-pass cells from cell 1 under that new frozen SHA. A new attempt must never overwrite prior attempt files. This fail-closed restart keeps the effective matrix source-homogeneous. Do not rerun physical work for an offline-only `finalize_matrix.py` defect. Repeat records must use the effective first-pass runner SHA and be written under that effective attempt's `repeats/`; if repeat execution reveals a runner defect requiring a source change, preserve the evidence and report the repeat unresolved unless a new full physical attempt is explicitly authorized.

## Planned Evidence Schema

### Raw physical cell

One JSON object per observation keyed by prompt, target, drafter, attempt ID, and attempt-local cell identity, stored only at `evidence/attempts/<attempt_id>/cells/<prompt>/<ordered-cell>.json`. Fields: prompt identity/text SHA/input-IDs SHA; exact target composition and donor ownership; drafter and immutable revision/selected weight SHA; settings; generated tokens, speculative tok/s, acceptance, mean accepted, target forwards, generated/forward, rounds, widths, caps, duration; active baseline, request-boundary peak, and increment in bytes/GiB; freshness, machine/runtime and interpreter identity/version; checkpoint revisions; physical-runner SHA; relative/reference identity for that attempt's `preflight.json`, its SHA and passing identity; output SHA; status. Each attempt directory also contains immutable `attempt.json` (attempt ID, runner SHA, passing preflight identity/SHA, start/status, complete/incomplete/superseded state, expected count 70, observed/statused count, and supersession reason where applicable) and its immutable preflight. Each attempt is identified by attempt ID, runner SHA, and matching passing preflight. Memory boundary/order and reset assertions show the peak belongs to the measured-request interval. No normalized deltas, retention, or savings appear in raw.

### Derived matrix and repeat manifest

Each derived record keys to the raw artifact path/SHA from the one effective attempt and same-prompt H0+B-Q identity/value; the feature-level `derived-matrix.json` records the effective attempt ID and contains throughput delta/retention and peak-memory delta/GiB saved/saving percent. H0+B-Q has zero deltas/savings and 100% retention. Finalizer SHA is recorded separately. `repeat-candidates.json`, optional `repeat-decisions.json`, `repeat-manifest.json`, `matrix-validation.json`, `panel-matrix.json`, and `panel-matrix.md` identify the effective attempt ID; each repeat entry also identifies its effective-attempt trigger raw artifact path/SHA. The physical repeat runner writes only to `evidence/attempts/<effective_attempt_id>/repeats/<comparison>.json` and consumes the feature-level manifest. `repeat-candidates.json` records subjective candidate classes and relevant raw identities/metrics. Optional `repeat-decisions.json` explicitly records selected subjective candidates with prompt, target, drafter, triggering raw identity/SHA, trigger class, and short reason. The immutable repeat manifest combines only objective automatic triggers and explicit decisions; the physical runner only consumes it.

### Validation and report

Validation covers attempt discovery/ledger, expected/observed count, identity/order, missing/duplicates/status, raw schema, freshness, exact settings/revisions, one effective 70-cell attempt and one runner SHA matching its passing preflight, raw output SHA, memory completeness/boundary evidence, interpreter pin, same-prompt baseline arithmetic, objective repeat triggers, explicit repeat decisions, repeat-manifest policy, and repeat provenance. Repeat provenance requires the effective first-pass runner SHA. The final panel JSON is the machine-readable source for Markdown tables and descriptive Pareto classification and records the finalizer SHA.

## Project Structure

```text
specs/005-golden-panel-layer-drafter-matrix/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
└── evidence/
    ├── probes/panel_matrix.py
    ├── probes/finalize_matrix.py
    ├── attempts.json
    ├── attempts/
    │   └── <attempt_id>/
    │       ├── attempt.json
    │       ├── preflight.json
    │       ├── cells/<prompt>/<ordered-cell>.json
    │       └── repeats/<comparison>.json
    ├── derived-matrix.json
    ├── repeat-candidates.json
    ├── repeat-decisions.json (optional)
    ├── repeat-manifest.json
    ├── repeats/<comparison>.json
    ├── matrix-validation.json
    ├── panel-matrix.json
    └── panel-matrix.md
```

No `contracts/` directory is needed because this internal experiment tooling exposes no external API/service. `tasks.md` remains for the later task-generation phase. No production source or Feature 003/004 artifact is modified.

## Post-Design Constitution Check

Pass. Fixed controls, accepted target composition, fresh hardware-backed measurements, and prompt-level reporting are preserved. Physical observations are insulated from offline reporting repairs by the explicit component boundary. The physical runner freezes after preflight and before cell 1. No weighted aggregation, automatic causal inference, quality claim, or production source change is introduced.
