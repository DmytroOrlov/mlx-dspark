# Tasks: Product Golden Hybrid Sweep

**Input**: `spec.md`, `plan.md`, `research.md`, `data-model.md`, and `quickstart.md`

**Execution rule**: The H0+B-Q canary is matrix cell 1. A failed or ambiguous canary is a valid terminal experiment outcome; in that case, do not execute rows 2–14 and do not fabricate missing-cell results.

**Source freeze**: After T003 passes, no feature-003 benchmark/orchestration source modification is allowed before completion of T004–T008. A source defect discovered during physical execution stops the experiment; repair requires a new preflight and fresh affected measurements rather than silently continuing.

## Phase 1: Product runner and canary (US1 — P1)

**Goal**: Reproduce the current Golden-A product baseline or record the narrow mismatch that prevents a product comparison.

**Independent test**: Model-free checks confirm pinned request/configuration identities, target/drafter definitions, frozen matrix order, and stop-on-canary behavior; the canary record then supports an explicit `STOP_MISMATCH` or `CONTINUE_SWEEP` decision.

- [X] T001 [US1] Implement the complete feature-local runner in `specs/003-product-golden-hybrid-sweep/evidence/probes/product_sweep.py`, including T027 product request/Engine semantics; feature-002 target/drafter composition injection; raw immutable per-cell result writing; canary adjudication; 14-row orchestration; completeness/provenance validation functions; selective-repeat decision/execution support; and final JSON/Markdown report generation support. (Known adjudication metadata defect: `continuation_rows` is 0 on CONTINUE_SWEEP; source frozen after T003, reported with T004.)
- [X] T002 [US1] Implement all lightweight model-free tests/checks for runner behavior in `specs/003-product-golden-hybrid-sweep/evidence/probes/product_sweep.py` and its focused test/check entrypoint, covering the exact Golden-A request, no feature-002 benchmark-control leakage, native B0, exact matrix order, canary as cell 1, `STOP_MISMATCH` blocking rows 2–14, completeness validator behavior, repeat-trigger behavior, and report/table structure.
- [X] T003 [US1] Execute the model-free checks and cheap preflight using `specs/003-product-golden-hybrid-sweep/evidence/probes/product_sweep.py`, including checkpoint/file identity, relevant reused runtime source identity, prompt digest, and process/cache freshness. After T003 passes, benchmark/orchestration source is frozen for the first-pass physical experiment.
- [X] T004 [US1] Execute only one fresh H0+B-Q canary through `specs/003-product-golden-hybrid-sweep/evidence/probes/product_sweep.py`, write its immutable raw result, and adjudicate `STOP_MISMATCH` or `CONTINUE_SWEEP`; make no source edits.

## Phase 2: Conditional fixed matrix (US2 — P1)

**Goal**: On `CONTINUE_SWEEP` only, collect the remaining 13 matched product cells exactly once each in frozen order.

**Independent test**: Mechanical completeness checks confirm 14 distinct ordered fresh observations with required provenance and metrics; this phase is intentionally skipped after `STOP_MISMATCH`.

- [X] T005 [US2] On `CONTINUE_SWEEP` only, execute rows 2–14 once each through `specs/003-product-golden-hybrid-sweep/evidence/probes/product_sweep.py`, each in a fresh process/cache and without changing request/runtime controls; retain cell failures without substitution and make no source edits.
- [X] T006 [US2] Execute the already-implemented completeness/provenance/metric validator in `specs/003-product-golden-hybrid-sweep/evidence/probes/product_sweep.py` over immutable first-pass observations, checking ownership, revisions, request digest, freshness, serial/speculative throughput, acceptance, forwards, generated/forward, cap/width data, memory, duration, and status; do not add validation code here.

## Phase 3: Adjudication and report (US3 — P2)

**Goal**: Publish either the canary mismatch outcome or the completed product comparison, with historical context clearly distinct from fresh measurements.

**Independent test**: `product-sweep.md` and `product-sweep.json` agree on the canary outcome, provenance, derived metrics, and—when the matrix ran—the required historical context and 14 fresh cells.

- [X] T007 [US3] Evaluate the already-implemented repeat policy in `specs/003-product-golden-hybrid-sweep/evidence/probes/product_sweep.py` and execute only triggered matched repeats: a hybrid at/above fresh H0+B-Q, a conclusion-critical near-noise comparison, or a local maximum relevant to a narrow boundary follow-up. If no trigger fires, record that and run zero repeats; do not modify `product_sweep.py` here.
- [X] T008 [US3] Execute the already-implemented report generation to write `specs/003-product-golden-hybrid-sweep/evidence/product-sweep.json` and `specs/003-product-golden-hybrid-sweep/evidence/product-sweep.md`: produce the `STOP_MISMATCH` report after a failed/ambiguous canary, or the complete product table/report after a successful matrix and any triggered repeats; do not modify runner source here.

## Dependencies and execution order

- T001–T003 complete and check the runner before T004; after T003 passes, the benchmark/orchestration source is frozen through T008.
- T004 is the branch point. `STOP_MISMATCH` proceeds directly to the mismatch report in T008; T005–T007 are skipped. `CONTINUE_SWEEP` proceeds to T005, T006, optional policy-triggered T007, then T008.
- All work is feature-local under `specs/003-product-golden-hybrid-sweep/evidence/`. Do not modify production runtime or feature-002 evidence, run feature-002 T040–T044, or reopen accepted feature-002 integrity findings.

## Parallel opportunities

No execution tasks need to run in parallel: the preflight and canary determine whether later matrix work is authorized by the experiment design. Keep implementation within the single feature-local runner to avoid unnecessary task splitting.

## MVP scope

Complete T001–T004 and produce the T008 `STOP_MISMATCH` report if the canary does not clearly recover the historical product regime. A passing canary makes the 13 remaining cells in T005 mandatory before claiming a completed matrix.

## Post-freeze explanation

- [X] T009 [US3] Produce `evidence/human-summary.md` and `evidence/historical-prompt-sensitivity.json` from immutable feature-003 results and narrowly scoped read-only historical feature-002 raw evidence; explain the historical p02 observations, aggregate-versus-prompt rates, and current Golden-A matrix without running new benchmarks.
