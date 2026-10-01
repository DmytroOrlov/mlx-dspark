---
description: "Dependency-ordered tasks for the Golden Panel Layer-Drafter Matrix"
---

# Tasks: Golden Panel Layer-Drafter Matrix

**Input**: Design documents in `specs/005-golden-panel-layer-drafter-matrix/`.

**Scope**: Feature-local evidence tooling only. Do not modify production runtime or Feature 003/004 evidence. The physical runner and offline finalizer remain separate components. The mandatory first pass is exactly 70 cells; there is no serial pass per cell and no layer-localization run.

## Phase A: Implement Physical Runner

**Purpose**: Build `panel_matrix.py` and its physical contracts before any inference.

- [X] T001 [US1] Define the exact five prompt records, text SHA rules, and qualification context in `specs/005-golden-panel-layer-drafter-matrix/evidence/probes/panel_matrix.py`.
- [X] T002 [US1] Define H0/H1a/H1b/H1c/H2/H3/B0 target ownership and explicit native B0 identity in `specs/005-golden-panel-layer-drafter-matrix/evidence/probes/panel_matrix.py`.
- [X] T003 [US1] Pin B-Q/B-B repository revisions and selected B-B weight SHA-256 in `specs/005-golden-panel-layer-drafter-matrix/evidence/probes/panel_matrix.py`.
- [X] T004 [US1] Encode the exact forward P05/P08/P17 and reverse P07/P14 14-cell orders, with adjacent drafter pairs, in `specs/005-golden-panel-layer-drafter-matrix/evidence/probes/panel_matrix.py`.
- [X] T005 [US1] Implement current-tokenizer preflight that regenerates exact prompt input IDs/digests and pins them before any measured run in `specs/005-golden-panel-layer-drafter-matrix/evidence/probes/panel_matrix.py`.
- [X] T006 [US1] Pin `.venv/bin/python` resolved identity/version and the accepted runtime identity in preflight and reject physical children that differ in `specs/005-golden-panel-layer-drafter-matrix/evidence/probes/panel_matrix.py`.
- [X] T007 [US1] Define the versioned attempt-scoped storage/schema: unique `evidence/attempts/<attempt_id>/` with `attempt.json`, immutable `preflight.json`, `cells/`, and `repeats/`; include attempt ID, attempt-local cell identity, matching passing preflight identity/SHA, measured fields, settings, provenance, status, and output/runner SHA fields in `specs/005-golden-panel-layer-drafter-matrix/evidence/probes/panel_matrix.py`.
- [X] T008 [US1] Implement one-cell execution using a fresh process, Engine, and empty measured request, normal production prefix-cache initialization with zero useful prior reuse, fixed sampling, plain KV, ordinary CapController, and no WidthPolicy/KV8/tuning in `specs/005-golden-panel-layer-drafter-matrix/evidence/probes/panel_matrix.py`.
- [X] T009 [US1] Implement fixed first-pass orchestration for exactly 5 × 7 × 2 cells per attempt, writing one immutable raw artifact per cell only under `specs/005-golden-panel-layer-drafter-matrix/evidence/attempts/<attempt_id>/cells/<prompt>/<ordered-cell>.json`.
- [X] T010 [US1] Record attempt ID, relative/reference identity to the same attempt directory's preflight, preflight SHA/passing identity, physical runner SHA, target/drafter revisions, machine/runtime/interpreter identity, freshness, prompt/tokenizer identities, output SHA, and status in each raw artifact written by `specs/005-golden-panel-layer-drafter-matrix/evidence/probes/panel_matrix.py`.
- [X] T011 [US3] Implement repeat execution that accepts only an explicit immutable `repeat-manifest.json`, verifies the effective attempt ID and runner SHA, and writes immutable repeat raw artifacts only under `specs/005-golden-panel-layer-drafter-matrix/evidence/attempts/<effective_attempt_id>/repeats/<comparison>.json`.

## Phase B: Request-Boundary Memory Instrumentation

**Purpose**: Make per-cell memory a first-class physical measurement tied to the measured request interval.

- [X] T012 [US1] Implement the exact memory sequence in `specs/005-golden-panel-layer-drafter-matrix/evidence/probes/panel_matrix.py`: finish model/drafter loading and normal warmup, establish fresh measured-request state, reset MLX peak, capture active baseline immediately before the request, execute only the measured speculative request, and capture peak immediately afterward.
- [X] T013 [US1] Store active baseline, request-boundary peak, and peak-minus-baseline increment in bytes and GiB, plus reset/order/boundary assertions sufficient to establish request ownership, in raw cells written by `specs/005-golden-panel-layer-drafter-matrix/evidence/probes/panel_matrix.py`.
- [X] T014 [US1] Add model-free assertions for all three memory values, units, nonnegative increment arithmetic, reset call, and request-boundary ordering in `specs/005-golden-panel-layer-drafter-matrix/evidence/probes/panel_matrix.py`.

## Phase C: Implement Offline Finalizer

**Purpose**: Derive and validate evidence without inference or raw-cell mutation.

- [X] T015 [US1] Implement immutable raw discovery/loading and stable raw path/SHA indexing in `specs/005-golden-panel-layer-drafter-matrix/evidence/probes/finalize_matrix.py`.
- [X] T016 [US1] Derive each cell only against the same-prompt H0+B-Q raw baseline, including throughput delta/retention, peak-memory delta, GiB saved, and memory-saving percentage; enforce H0 zero-delta/100%-retention semantics in `specs/005-golden-panel-layer-drafter-matrix/evidence/probes/finalize_matrix.py`.
- [X] T017 [US1] Implement exact arithmetic recomputation and validation for all derived throughput and memory fields in `specs/005-golden-panel-layer-drafter-matrix/evidence/probes/finalize_matrix.py`.
- [X] T018 [US1] Discover attempt directories and update the global ledger; select only one eligible physically complete 70-cell attempt directory (normal COMPLETE lifecycle or the narrow, explicitly documented output-length-only INCOMPLETE adjudication; SUPERSEDED is always ineligible); validate every cell's preflight reference/SHA against that directory's passing preflight, one runner SHA, expected identities/order, missing/duplicate/status cells, raw schema, freshness, fixed settings, target/drafter revisions, runtime/interpreter pin, and output SHA in `specs/005-golden-panel-layer-drafter-matrix/evidence/probes/finalize_matrix.py`.
- [X] T019 [US1] Validate active-baseline/peak/increment units and arithmetic, plus reset and measured-request boundary metadata, without deriving savings in raw evidence in `specs/005-golden-panel-layer-drafter-matrix/evidence/probes/finalize_matrix.py`.
- [X] T020 [US1] Write feature-level `derived-matrix.json` and first-pass `matrix-validation.json` with effective attempt ID, finalizer SHA, and attempt-scoped raw references, leaving every raw artifact immutable, under `specs/005-golden-panel-layer-drafter-matrix/evidence/`.

## Phase D: Repeat Policy and Reporting (Model-Free)

**Purpose**: Complete all offline selection/report logic before physical execution.

- [X] T021 [US3] Implement only the objective automatic repeat triggers in `specs/005-golden-panel-layer-drafter-matrix/evidence/probes/finalize_matrix.py`: any non-H0 cell with tok/s >= same-prompt H0+B-Q, and any H1b+B-B cell with retention >=97% of same-prompt H0+B-Q.
- [X] T022 [US3] Generate model-free feature-level `repeat-candidates.json` entries for potential conclusion-critical near-drift comparisons and surprising local target maxima, including effective attempt ID and relevant attempt-scoped raw identities/metrics without subjective selection; accept optional `repeat-decisions.json` entries containing effective attempt ID, prompt, target, drafter, triggering raw identity/SHA, trigger class, and short explicit reason; combine objective triggers and explicit decisions into immutable feature-level `repeat-manifest.json` in `specs/005-golden-panel-layer-drafter-matrix/evidence/probes/finalize_matrix.py`.
- [X] T023 [US2] Render Raw Prompt × Config Matrix and Throughput Retention + Memory tables with all five prompt cells before summaries and all required robustness/memory statistics in `specs/005-golden-panel-layer-drafter-matrix/evidence/probes/finalize_matrix.py`.
- [X] T024 [US2] Render Drafter Delta and Mechanism Breakdown tables for all target/prompt pairs and all physical mechanism fields, labeling measurements separately from interpretation, in `specs/005-golden-panel-layer-drafter-matrix/evidence/probes/finalize_matrix.py`.
- [X] T025 [US2] Render Speed / Memory Tradeoff and descriptive Pareto classification using throughput robustness and memory only, with no weighted score or arbitrary ranking, in `specs/005-golden-panel-layer-drafter-matrix/evidence/probes/finalize_matrix.py`.
- [X] T026 [US2] Render per-prompt B-Q/B-B layer paths, the H1a/H1b/H1c interaction view, B0 interpretation fields, repeat ledger, and measured/derived/interpretation distinctions in `specs/005-golden-panel-layer-drafter-matrix/evidence/probes/finalize_matrix.py`.
- [X] T027 [US2] Write feature-level machine-readable panel data and Markdown report, including all five required tables and the effective attempt ID in both artifacts, in `specs/005-golden-panel-layer-drafter-matrix/evidence/panel-matrix.json` and `specs/005-golden-panel-layer-drafter-matrix/evidence/panel-matrix.md`.

## Phase E: Model-Free Checks and Physical Preflight

**Purpose**: Establish every contract before freezing physical source; preflight generates no measured benchmark data.

- [X] T028 [US1] Add/run model-free checks for exact 70 identities, target ownership, drafter revisions/B-B weight SHA, and both exact orderings in `specs/005-golden-panel-layer-drafter-matrix/evidence/probes/panel_matrix.py`.
- [X] T029 [US1] Add/run model-free checks for raw schema, freshness/settings declarations, memory boundary instrumentation, and derived formulas in `specs/005-golden-panel-layer-drafter-matrix/evidence/probes/`.
- [X] T030 [US3] Add/run model-free checks for the two exact objective triggers, candidate reporting without subjective selection, absent/empty decisions behavior, explicit decision schema, merged immutable manifest, one-repeat limit, effective-attempt repeat provenance, five-table structure, prompt-cell visibility, layer paths, interaction view, and Pareto rules in `specs/005-golden-panel-layer-drafter-matrix/evidence/probes/finalize_matrix.py`.
- [X] T031 [US1] Create `specs/005-golden-panel-layer-drafter-matrix/evidence/attempts/<attempt_id>/` and run physical preflight linked to that attempt, pinning `.venv/bin/python` identity/version, checkpoint availability/identity, current tokenizer inputs, settings, runtime, output paths, and freshness; verify preflight performs no measured generation and write immutable `preflight.json` inside the attempt directory.
- [X] T032 [US1] Record passing preflight identity and runner SHA in attempt-local `preflight.json`, and record its computed SHA plus passing identity and runner SHA in `attempt.json`, linked by attempt ID, after all Phase A–E checks pass; finalize `attempt.json` lifecycle status when the attempt ends.

## Phase F: SOURCE FREEZE and First Pass

**Purpose**: Freeze the sole physical source immediately before cell 1, then execute one bounded 70-cell phase.

- [X] T033 [US1] SOURCE FREEZE: After T001–T032 pass and immediately before first-pass cell 1, freeze `specs/005-golden-panel-layer-drafter-matrix/evidence/probes/panel_matrix.py`; link the exact `evidence/attempts/<attempt_id>/` directory, its immutable passing `preflight.json`, and recorded runner SHA, which every first-pass raw cell must reference.
- [X] T034 [US1] Execute exactly the 70 first-pass physical cells once in the stipulated per-prompt forward/reverse order, each with a fresh process/Engine/request, writing immutable statused evidence only inside `specs/005-golden-panel-layer-drafter-matrix/evidence/attempts/<attempt_id>/cells/`; do not run serial controls or per-cell serial passes.
- [X] T035 [US1] If `panel_matrix.py` requires any source change after cell 1, stop the current attempt; preserve its preflight and all raw artifacts byte-for-byte, finalize `attempt.json` as incomplete/superseded with reason and mark it ineligible for the effective matrix; repair the runner, rerun model-free physical checks and attempt-linked preflight in a new attempt directory, obtain a new runner SHA and attempt ID, then restart all 70 first-pass cells from cell 1 without overwriting any prior attempt path. (Triggered for b1; c1 was subsequently preflighted and ran its 70-cell first pass. Its short-output eligibility issue is an offline finalizer defect, documented below.)

## Phase G: Offline Derivation and First-Pass Validation

**Purpose**: Derive only after all 70 identities are present/statused; offline repairs never invalidate raw observations.

- [X] T036 [US1] After the complete first pass is present/statused, derive the same-prompt normalized matrix and validate the complete first pass, including late H0+B-Q baselines for P07/P14, using `specs/005-golden-panel-layer-drafter-matrix/evidence/probes/finalize_matrix.py`; c1 selected by offline adjudication with 70 physically valid observations and one short-output annotation.
- [X] T037 [US1] Write feature-level `specs/005-golden-panel-layer-drafter-matrix/evidence/derived-matrix.json` and `specs/005-golden-panel-layer-drafter-matrix/evidence/matrix-validation.json`, recording the effective attempt ID and attempt-scoped raw references; verify raw cell files remain byte-for-byte untouched; the c1 SHA manifest and protected attempt files remain unchanged.
- [X] T038 [US1] For an offline-only defect, repair `specs/005-golden-panel-layer-drafter-matrix/evidence/probes/finalize_matrix.py` and rerun derivation/validation/report checks from unchanged raw evidence, with no physical reruns.

## Phase H: Selective Matched Repeats

**Purpose**: Execute only comparisons selected by the validated offline policy.

- [X] T039 [US3] From the validated effective first pass, merge the two objective triggers with only entries explicitly selected in `repeat-decisions.json`, deduplicate by comparison, include effective attempt ID and triggering attempt-scoped raw artifact path/SHA in each entry, and write immutable feature-level `specs/005-golden-panel-layer-drafter-matrix/evidence/repeat-manifest.json` with at most one entry per comparison.
- [X] T040 [US3] Have the frozen runner consume exactly that manifest and execute at most one initial fresh matched repeat per listed comparison, using the effective first-pass runner SHA and writing immutable repeat raw evidence only under `specs/005-golden-panel-layer-drafter-matrix/evidence/attempts/<effective_attempt_id>/repeats/`; do not repeat all 70 cells. (16/16 immutable manifest repeats were executed successfully from the ordinary host Terminal with frozen runner SHA `eef85356dd60c6be8f7622c304e2f6fae41202e4f2e811310dabe44081c1b072`; the earlier no-Metal failure was specific to agent sandbox/headless execution.)
- [X] T041 [US3] Validate repeat identity, effective attempt ID, trigger path/SHA linkage, effective first-pass runner SHA, attempt-scoped output path, freshness, settings, revisions, output SHA, and memory boundary offline in `specs/005-golden-panel-layer-drafter-matrix/evidence/probes/finalize_matrix.py`; if a runner defect requires source change, preserve evidence and report the repeat unresolved unless a new full attempt is authorized.

## Phase I: Final Super Tables

**Purpose**: Produce the auditable final matrix/report after repeat validation.

- [X] T042 [US1] Generate final feature-level `specs/005-golden-panel-layer-drafter-matrix/evidence/matrix-validation.json` and `specs/005-golden-panel-layer-drafter-matrix/evidence/panel-matrix.json` from one attempt directory's immutable first-pass/repeat evidence, recording effective attempt ID and finalizer SHA.
- [X] T043 [US2] Generate `specs/005-golden-panel-layer-drafter-matrix/evidence/panel-matrix.md` with effective attempt ID, all five prompt cells before summaries, the five required tables, Pareto view, both per-prompt drafter paths, H1 interaction, repeat ledger, and measured/derived/interpretation labels.
- [X] T044 [US2] Check the final report states highest minimum-rate configuration, all-five-prompt 40/45 tok/s sets, per-target drafter deltas, H1b repeat status, memory savings across ownership, H1b+B-B memory/throughput behavior, H3 tradeoff, and B0 memory/acceptance context without causal overclaim in `specs/005-golden-panel-layer-drafter-matrix/evidence/panel-matrix.md`.

### Feature 005 closure

Effective attempt `feat005-20260926-c1`; first pass 70/70 physically valid and 69/70 long-form-qualified; repeats 16/16 valid; finalizer self-check 32 PASS; final validation PASS; physical runner frozen SHA `eef85356dd60c6be8f7622c304e2f6fae41202e4f2e811310dabe44081c1b072` unchanged. No additional inference was required. The attempt lifecycle remains `INCOMPLETE`; its narrowly documented offline adjudication is recorded in the validation and report artifacts.

## Dependencies and Execution Order

- **Phase A → B → C → D → E**: Implement physical measurement and offline derivation/reporting, then complete all model-free checks and physical preflight. Physical execution cannot begin earlier.
- **T033 SOURCE FREEZE** is the physical source-freeze boundary. It follows all Phase A–E checks and occurs immediately before first-pass cell 1. The exact attempt directory, its immutable passing preflight, attempt ID, and runner SHA are linked before freeze.
- **T034** is the single 70-cell first-pass execution phase. T035 is conditional repair handling only when a physical defect occurs.
- **Phase G** follows completion/statusing of all 70 first-pass identities. Raw evidence is immutable; finalizer-only repairs rerun offline.
- **Phase H** follows successful first-pass validation. The runner executes only the immutable selected repeat manifest.
- **Phase I** follows repeat validation (or confirmation that no repeats were triggered).
- Story mapping: **US1** complete per-prompt matrix and valid normalized evidence; **US2** mechanisms and human-facing interpretation/report; **US3** repeat adjudication and bounded robustness.

## Parallel Opportunities

- Within Phase A, prompt definitions, ownership definitions, drafter pins, and order declarations (T001–T004) can be authored independently before integrating them into the single runner.
- Within Phase D, report-table implementations (T023–T026) can be developed independently against the agreed derived schema, then integrated by T027.
- Phase C finalizer work can proceed alongside Phase A/B runner implementation once the raw schema in T007 is stable; no physical execution is parallelized or split into per-cell tasks.
- Phases E–I are sequential at their boundaries; source freeze and all physical execution remain one controlled ordered phase.

### Offline adjudication note: c1 short output

The c1 first pass exposed an offline eligibility classification defect: Feature 004's prompt-qualification threshold of 410 generated tokens was incorrectly reused as a Feature 005 per-cell physical-validity threshold. Feature 005 does not define that threshold as physical invalidation. No physical semantics were affected, no raw artifact was changed, and no runner source was changed; c1 remains source-homogeneous. The repair is finalizer-only. P07/H3/B-Q remains statused `invalid_too_short` at 400 tokens and is reported as a physically valid short-output observation, with observed-throughput and strict long-form robustness reported separately.

## Independent Acceptance Criteria

- **US1**: The effective matrix comes from exactly one eligible physically complete 70-cell attempt directory, under one runner SHA matching its attempt-local immutable preflight. Eligibility requires either a normal COMPLETE lifecycle or a narrow offline-adjudicated INCOMPLETE lifecycle where all 70 cells pass physical/provenance validation and the only non-complete status is `invalid_too_short` with the recognized output-length acceptance reason; SUPERSEDED and all other lifecycle states are ineligible. The effective attempt's lifecycle truth and any adjudication reason remain explicit in validation/report output. All other attempt directories remain ledgered and cannot be overwritten or merged. Every valid raw cell contains pinned attempt/preflight/prompt/target/drafter identity, settings, hashes, and request-boundary baseline/peak/increment memory in bytes and GiB. Derived values record the effective attempt ID, recompute from that prompt's H0+B-Q, and leave raw files unchanged.
- **US2**: All five tables expose prompt-level cells before summaries, include the required throughput/mechanism/memory fields, show per-prompt B-Q/B-B paths and H1 interaction, and distinguish measured values, derived values, hypotheses, and interpretations without weighted scoring or automatic causal claims.
- **US3**: Objective repeats are selected only by the two specified formulas. Subjective near-drift/local-maximum candidates are reported for explicit offline adjudication and never silently selected. No comparison receives more than one initial matched repeat; only manifest-listed repeats execute, and repeats use the effective first-pass runner SHA.

## Implementation Strategy

Complete all model-free implementation and checks before preflight. The pre-freeze scope ends when T032 records the passing physical runner SHA and attempt linkage. T033 freezes the physical runner; T034 performs the single exact first-pass phase. Then derive and validate offline, select and execute only manifest-listed repeats, and generate final tables. Offline-only defects are fixed in the finalizer and rerun offline; any physical-runner source change after cell 1 invalidates that attempt for the effective matrix, requiring a new preflight/SHA/attempt ID and a full 70-cell restart from cell 1. No conditional layer-localization experiment is authorized here.
