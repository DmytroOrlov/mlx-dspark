# Tasks: Qwen-Bonsai Hybrid Target Experiment

**Input**: `specs/002-qwen-bonsai-hybrid-target/{spec,plan,research,data-model,quickstart}.md` and `.specify/memory/constitution.md` v2.0.0.

**Scope**: These tasks implement only the planned composition path, required integrity evidence, Series-A discovery/confirmation/adjudication, and the mechanically closed Gate B. They do not start Series B, change the constitution, introduce experiment axes, or modify feature-001 evidence.

**Execution rule**: Complete tasks in numeric order unless explicitly marked parallel. A task's dependencies are mandatory; later tasks must not be pulled forward around a failed integrity gate. Every task is a checklist item with its file/artifact scope.

## Phase 1: Checkpoint and Provenance Preflight

**Purpose**: Freeze the precise inputs and prove the official donor identity before hybrid work or performance measurement.

- [X] T001 [US1] Implement model-free input preflight in `specs/002-qwen-bonsai-hybrid-target/evidence/probes/series_a.py`; resolve the exact Qwen `mlx-community/Qwen3.8-27B-4bit`, official donor `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`, and original drafter `incoai/Qwen3.8-27B-DFlash2` to immutable revisions and absolute snapshots, rejecting floating refs for accepted runs.
- [X] T002 [US1] Record config and weight provenance for all three checkpoints in `specs/002-qwen-bonsai-hybrid-target/evidence/checkpoints.json`, including `config.json` SHA-256, every loaded weight file's size and fingerprint, fingerprint method, resolved repo/revision/path, tokenizer identity, and runtime/tool versions; depend on T001.
- [X] T003 [US1] Compare the selected exact donor revision against the schema and model/module assumptions in `specs/002-qwen-bonsai-hybrid-target/research.md` and `specs/002-qwen-bonsai-hybrid-target/evidence/checkpoints.json`; verify Prism schema/model family, 64-layer layout, dimensions, attention/GatedDelta family sequence, pack metadata, and required module paths. If revision differs from the planning revision, explicitly revalidate all affected assumptions; any mismatch blocks T004 and all hybrid work.

## Phase 2: Exact Official Bonsai2 Load Proof

**Purpose**: Verify the exact selected official donor through the current repository Prism loader, independently of historical feature-001 repack evidence.

- [X] T004 [US1] Add a pre-hybrid official-donor proof command to `specs/002-qwen-bonsai-hybrid-target/evidence/probes/series_a.py` that passes the exact donor repo and immutable revision to the current `prism_pack`/repository target loader (never the stale registry alias); depend on passing T001–T003.
- [X] T005 [US1] Load the exact official donor and record a minimal B0 finite-output and integrity smoke plus packed projection/module inventory in `specs/002-qwen-bonsai-hybrid-target/evidence/integrity/official-bonsai-load.json`; verify expected model structure, projection `Packed` presence and MMA-compatible module inventory, shapes/dtypes, and finite parameters/output. Explicitly state that feature-001's nathansutton repack is not evidence for this proof; any schema, loader, family, or weight-layout mismatch is a blocking integrity failure, with no substitution allowed; depend on T004.

## Phase 3: Hybrid Composition Implementation

**Purpose**: Build the minimal opt-in composition path only after exact checkpoint and official loader proof pass.

- [X] T006 [US1] Implement the isolated generic composition, donor-first extraction, immutable ownership manifest, module-identity checks, and deterministic canonical serialization in `src/mlx_dspark/hybrid_target.py`; support sorted arbitrary valid donor indices and validate same-index layer families; depend on T005 passing.
- [X] T007 [US1] Add a narrow optional replacement handoff to `src/mlx_dspark/load.py` so requested donor blocks replace Qwen blocks before `Target` initialization and existing MMA routing is installed for retained Prism `Packed` modules; `None` must retain the ordinary loader path exactly; depend on T006.
- [X] T008 [US1] Add the optional internal target-composition handoff to `src/mlx_dspark/server.py` `Engine.load`; use it only when explicitly supplied and leave ordinary Qwen/drafter defaults and production CLI unchanged; depend on T007.
- [X] T009 [US1] Add donor memory-release instrumentation and hard stop logic to `src/mlx_dspark/hybrid_target.py` and `specs/002-qwen-bonsai-hybrid-target/evidence/probes/series_a.py`: record MLX memory before donor load, donor-load peak, after pruning/release/cache cleanup, and immediately before Qwen load; establish unused donor wrapper, blocks, embedding, norm, head, and temporary state are materially reclaimed before Qwen loading; depend on T006–T008.

## Phase 4: Model-Free Composition Tests

**Purpose**: Lock down composition and default-path contracts without loading model checkpoints.

- [X] T010 [US1] Add model-free tests in `tests/test_hybrid_target.py` for sorted donor-index validation, arbitrary valid composition, and duplicate/out-of-range rejection; depend on T006–T008.
- [X] T011 [US1] Add exact H1a `[63]`, H1b `[62]`, H1c `[62,63]`, H2 `[60,61,62,63]`, and H3 `[56,57,58,59,60,61,62,63]` mapping tests plus H0 as no-composition vanilla Qwen; depend on T010.
- [X] T012 [US1] Add tests in `tests/test_hybrid_target.py` proving B0 is a separate full-Bonsai endpoint, same-index layer-family validation rejects mismatch, and ownership is determined by live module identity rather than class name; depend on T010.
- [X] T013 [US1] Add tests in `tests/test_hybrid_target.py` for donor `Packed` projection presence, deterministic canonical manifest serialization, and default loader/`Engine.load` behavior when composition is absent; depend on T010–T012.

## Phase 5: Donor Memory-Release Proof

**Purpose**: Prove unused donor state is reclaimed before the hybrid path loads Qwen.

- [X] T014 [US1] Execute the donor-memory release gate on the selected immutable donor and save measured checkpoints, thresholds/basis, and pass/fail evidence to `specs/002-qwen-bonsai-hybrid-target/evidence/integrity/donor-memory-release.json`; if unsafe donor residency remains, stop before Qwen load and implement only the plan-authorized selective read of required block tensors/modules from the existing Prism pack, preserving existing Prism packing/Hadamard semantics and reusing `Packed`/MMA; then repeat this gate. Do not preemptively implement selective loading; depend on T009 and T010–T013.

## Phase 6: Real-Checkpoint Hybrid Integrity Smoke

**Purpose**: Establish all target integrity/semantic prerequisites against exact loaded checkpoints and current runtime before any performance run.

- [X] T015 [US1] Add real-checkpoint integrity smoke modes and manifest emission to `specs/002-qwen-bonsai-hybrid-target/evidence/probes/series_a.py`, requiring passing T002, T005, T014, and T013; load H0 through the ordinary no-composition Qwen path and construct each hybrid only through the opt-in composition path.
- [X] T016 [US1] Revalidate and record against the actual runtime revision in `specs/002-qwen-bonsai-hybrid-target/evidence/integrity/runtime-semantics.json` that Prism projection transforms are local, block residual boundaries are canonical, no Qwen↔Bonsai bridge is needed, loaded DFlash `target_layer_ids` are actual, and tap capture is zero-based block output; for expected taps `[5,19,33,47,61]`, record direct Bonsai-owned tap counts H0=0, H1a=0, H1b=0, H1c=0, H2=1, H3=1, B0=5. If loaded taps differ, stop and recalculate/document from the actual config; do not reuse stale counts; depend on T015.
- [X] T017 [US1] Run and retain one integrity record per H0, H1a, H1b, H1c, H2, H3, and B0 under `specs/002-qwen-bonsai-hybrid-target/evidence/integrity/`; cover exact target ownership manifest, parameter shape/dtype and finiteness, finite outputs, representative Qwen→Bonsai and Bonsai→Qwen boundaries, `Target` initialization after replacement, donor packed MMA routing, cache advancement, and a narrow accepted-prefix/rejected-suffix rollback exercise; depend on T016.
- [X] T018 [US1] Add a fresh-process repeatability check and same-revision ordinary-H0 behavior comparison to `specs/002-qwen-bonsai-hybrid-target/evidence/probes/series_a.py`, recording repeatable composition and unchanged no-composition H0 behavior without requiring hybrid logits to match Qwen; keep the rollback probe limited to this integrity smoke and do not reopen feature-001 rollback-parity investigation; depend on T017.
- [X] T019 [US1] Mark the integrity prerequisite set complete only when all seven target records and semantic evidence pass, in `specs/002-qwen-bonsai-hybrid-target/evidence/integrity/index.json`; failed or missing records invalidate performance eligibility; depend on T018.

## Phase 7: Series-A Discovery

**Purpose**: Measure the fixed target matrix only after every integrity prerequisite passes.

- [X] T020 [US1] Implement the feature-local Series-A runner in `specs/002-qwen-bonsai-hybrid-target/evidence/probes/series_a.py` using the existing Engine path, plain KV, ordinary production `CapController`, original incoai drafter, identical frozen prompt corpus and generation/sampling settings, fixed runtime/checkpoint revisions, and a fresh process or fully reset engine per run; serialize complete provenance and raw results; depend on T019.
- [X] T021 [US1] Add discovery collection in `specs/002-qwen-bonsai-hybrid-target/evidence/series-a/runs/` in the exact order `H0 → H1a → H2 → H1b → H3 → H1c → B0 → H0`; each run must link its integrity record and capture serial target tokens/sec, speculative decode tokens/sec, speedup, acceptance/accept length, target forwards, generated tokens per target forward, width/cap distribution, peak steady-state memory, generated count, duration, available non-invasive timing breakdowns, and all provenance; reject historical feature-001 throughput as anything beyond context; depend on T020.
- [X] T022 [US1] Validate completeness and matched controls for all seven discovery conditions plus bracketing fresh H0, and write `specs/002-qwen-bonsai-hybrid-target/evidence/series-a/discovery-index.json`; invalid, incomplete, or mismatched runs cannot proceed to repeat selection/adjudication; depend on T021.

## Phase 8: Repeat Selection and Required Confirmations

**Purpose**: Select confirmations from discovery evidence instead of repeating every condition automatically.

- [X] T023 [US1] Generate `specs/002-qwen-bonsai-hybrid-target/evidence/series-a/repeat-selection.json` from immutable discovery records; identify apparent winners, regressions, near-noise comparisons, and whether H1c-vs-H1a/H1b interaction is decision-relevant; record variability basis, comparison group shuffle seed, selected comparisons, and rationale; depend on T022.
- [X] T024 [US1] Run only selected comparisons from T023 in `specs/002-qwen-bonsai-hybrid-target/evidence/series-a/runs/` as at least three alternating matched pairs per selected comparison (and separate pairs for each decision-relevant H1c interaction); retain actual run order, prompt IDs, settings, and all records; do not reuse one observation as two independent pairs; depend on T023.
- [X] T025 [US1] Record repeat/noise outcomes and whether each selected comparison is confirmed, within noise, inconclusive, or invalid in `specs/002-qwen-bonsai-hybrid-target/evidence/series-a/repeat-decisions.json`; no single marginal discovery result may be adjudicated as a confirmed win or regression; depend on T024.

## Phase 9: Series-A Adjudication

**Purpose**: Produce the immutable evidence-based decision artifact and preserve tap-compatibility causal distinctions.

- [X] T026 [US1] Generate `specs/002-qwen-bonsai-hybrid-target/evidence/adjudication/series-a.json` exclusively from immutable run, manifest, integrity, and repeat records; separate measured facts, derived metrics, and interpretation, include absolute results and deltas against fresh H0, and answer every Series-A question in `specs/002-qwen-bonsai-hybrid-target/spec.md`; explicitly allow hybrid win, vanilla H0 win, no meaningful difference, or no production change; depend on T019, T022, and T025.
- [X] T027 [US1] Include the measured causal tap distinction in `specs/002-qwen-bonsai-hybrid-target/evidence/adjudication/series-a.json`: H1a/H1b/H1c substitutions are after block 61 and produce zero directly captured taps from Bonsai-owned blocks; H2/H3 own block 61 and produce one; B0 produces five. Do not infer unchanged drafter compatibility or logits from zero directly replaced taps, and do not invent untested suffix boundaries; depend on T026.

## Phase 10: Gate B Evaluation (Series B Remains Blocked)

**Purpose**: Create and validate a mechanical CLOSED-by-default barrier. This phase does not implement or start Series-B measurements.

- [X] T028 [US2] Create `specs/002-qwen-bonsai-hybrid-target/evidence/gate-b.json` with schema/version, current spec/constitution/runtime hashes, required targets, references, closure checks, audit fields, and `status: CLOSED` by default; missing, malformed, stale, or incomplete evidence must fail closed; depend on T026–T027.
- [X] T029 [US2] Implement a model-load-free Gate B validator in `specs/002-qwen-bonsai-hybrid-target/evidence/gate_b.py` that permits status OPEN only after validating exact checkpoint provenance, official donor proof, donor-memory-release proof, all seven target integrity records, all required discovery results and fresh H0, repeat/noise decisions, required confirmation pairs, completed Series-A adjudication, and current spec/constitution/runtime hashes; depend on T028.
- [X] T030 [US2] Wire the future Series-B entry boundary to validate Gate B before any model load, without implementing Series-B target measurements or matrix tasks in this feature pass; record fail-closed behavior for missing/stale/malformed/CLOSED gate in `specs/002-qwen-bonsai-hybrid-target/evidence/gate_b.py`; depend on T029.

## Dependencies & Execution Order

### Critical dependency chain

`T001 → T002 → T003 → T004 → T005 → T006 → T007 → T008 → T009 → T010–T013 → T014 → T015 → T016 → T017 → T018 → T019 → T020 → T021 → T022 → T023 → T024 → T025 → T026 → T027 → T028 → T029 → T030`

The required DAG is therefore: checkpoint/provenance preflight → exact official Bonsai2 load proof → hybrid composition implementation → model-free tests → donor memory-release proof → real-checkpoint hybrid integrity smoke → Series-A discovery → repeat selection → required confirmation runs → Series-A adjudication → Gate B evaluation. The memory gate is an additional hard dependency before Qwen may load for composition. If it fails, T014's selective-loading fallback and repeated memory gate must pass before T015.

### Gate constraints

- No composition implementation or performance work before T005 passes for the selected immutable official donor.
- No Qwen load following donor pruning until T014 proves material donor-state reclamation; unsafe residency stops the path and triggers only the plan-authorized selective-loading fallback.
- No performance discovery before T019 passes for every target.
- No floating checkpoint refs may appear in accepted manifests or results.
- No Series-B model loading or measurement before Gate B is validated OPEN; this task set includes only the fail-closed validator boundary, not Series-B measurement work.
- No feature-001 evidence mutation, old rollback causal probes, Chad adapters, rollback gates, or dead-end scheduler work.

### Parallel opportunities

There are no safe parallel tasks across the required critical chain. Within Phase 4, T011 and T012 may be implemented in parallel after T010 if separate test sections/files are coordinated; T013 requires their combined coverage. The two fresh-H0 controls in discovery are sequential elements of the prescribed order, not parallel jobs. Repeat comparisons must preserve their matched pairing and actual alternating order.

## Feature-001 Reuse Classification

- Stable benchmark/provenance helper concepts may be generalized into `specs/002-qwen-bonsai-hybrid-target/evidence/probes/series_a.py`.
- Historical feature-001 benchmark numbers are context only and never substitute for fresh Series-A measurements.
- Rollback causal probes, Chad adapters, old rollback gates, and dead-end scheduler work are not imported; feature-001 evidence remains unchanged.

## Implementation Strategy

Complete the provenance and official-loader proof first, then implement the isolated composition route and model-free tests. Prove donor memory reclamation in T014 before any Qwen load in the composed path. Complete all seven target integrity records before discovery. Select repeats from discovery evidence, then adjudicate Series A from immutable records. Finish Series A by validating the mechanical Gate B barrier, which remains CLOSED unless every specified artifact/hash check passes. After Gate B validates OPEN, execute only the appended gated Series-B chain.

## Phase 11: Series-B Gate Entry and Frozen Matrix Decision

**Purpose**: Begin the one-axis drafter experiment only after the existing Gate B is validated OPEN, then freeze the exact authorized targets before resolving or loading any Series-B checkpoint.

- [X] T031 [US3] Add Series-B entry validation and a deterministic matrix/exception decision in `specs/002-qwen-bonsai-hybrid-target/evidence/probes/series_b.py` and `specs/002-qwen-bonsai-hybrid-target/evidence/series-b/matrix-decision.json`; call the existing model-load-free Gate-B validator before any Series-B checkpoint resolution, download, load, composition, or benchmark action, and fail closed for missing, malformed, stale, or CLOSED `evidence/gate-b.json`. Derive the isolated-block exception mechanically from `evidence/adjudication/series-a.json` and `evidence/series-a/repeat-decisions.json`: H1c is confirmed faster than H1a but H1c-vs-H1b is `within_noise`, so neither H1a nor H1b is unambiguously the confirmed best Series-A hybrid; authorize exactly H0, H1c, H2, H3, and B0 with both drafters (10 cells), with no exception; depend on completed T030.

## Phase 12: Exact Series-B Drafter Provenance

**Purpose**: Reverify the Bonsai-specific drafter from the selected immutable snapshot and create new, Series-B-local provenance without changing the hash-pinned Series-A checkpoint record.

- [X] T032 [US3] After T031 passes, resolve and reverify `naklitechie/Qwen3.8-27B-DFlash2-ternary-bonsai2` to one exact immutable revision and snapshot, inspect its actual config and every file loaded as weights, and write repo ID, revision, resolved snapshot, config fingerprint, loaded-file inventory, sizes, and fingerprints for both B-B and the accepted B-Q checkpoint to `specs/002-qwen-bonsai-hybrid-target/evidence/series-b/checkpoints.json`; compare the B-B revision lead `3fc0d6ef43e933a20f1e9ee53fac6f56402c0f83` and weights SHA-256 lead `eb141d0fb8621cf7a7c81047b83810fa19b8093543bd05529e9478d9555cb8f1` as leads only, reject floating refs, mismatches, silent substitutions, or incomplete provenance, and leave `evidence/checkpoints.json` unchanged; depend on T031.

## Phase 13: Minimal B-B Runtime Compatibility Smoke

**Purpose**: Establish only the checkpoint/runtime compatibility needed to authorize paired performance measurements, using the existing speculative runtime and without starting a broad rollback investigation.

- [X] T033 [US3] Add and run a narrow B-B compatibility smoke through the T031 Gate-B entry guard using the exact T032 snapshot; record actual loaded DFlash target-layer/tap configuration, finite required tensors and forward behavior, compatible hidden/tap shapes, cache advancement, rejected-suffix rollback, and unchanged ordinary `CapController`, plain-KV, and speculative policy in `specs/002-qwen-bonsai-hybrid-target/evidence/series-b/drafter-smoke.json`; stop Series B on any failure and record any strictly necessary serialization/loading adaptation without changing speculative policy; depend on T032.

## Phase 14: Series-B Runner, Controls, and Model-Free Validation

**Purpose**: Implement a Series-B-local orchestration layer around accepted Series-A composition and measurement semantics, then validate its safety and paired-control contracts without loading checkpoints.

- [X] T034 [US3] Extend the T031-created `specs/002-qwen-bonsai-hybrid-target/evidence/probes/series_b.py` into the Series-B-local runner that calls the T031 Gate-B guard before all model/checkpoint access, reuses the accepted Series-A composition loader and measurement path, and records a reproducible runner snapshot/fingerprint in `specs/002-qwen-bonsai-hybrid-target/evidence/series-b/runner.json`; prove the core benchmark method is byte-identical where practical, otherwise semantically identical, to accepted `run_series_condition` fingerprint `688901863e6660ecef749c310d9f69ff8cfc7cf37db0137716ae1058bc1ff6d4`, covering target execution, metrics, prompts, generation, controller, and measurement semantics; depend on T033.
- [X] T035 [US3] Encode the authorized target-by-drafter cells and immutable paired controls in `specs/002-qwen-bonsai-hybrid-target/evidence/series-b/matrix.json` and `specs/002-qwen-bonsai-hybrid-target/evidence/series-b/control-manifest.json`; pin target identity/ownership, prompt corpus and token IDs, generation length/settings, plain KV, no KV8, ordinary production `CapController`, no `WidthPolicy`, speculative policy, runtime source bytes, machine/hardware, measurement method, and warmup/reset protocol, allowing only drafter checkpoint and any documented necessary loading adaptation to differ; include fresh H0+B-Q as the production/reference baseline and H0+B-B as the unchanged-Qwen-target drafter control; depend on T034.
- [X] T036 [US3] Add model-free tests in `tests/test_series_b.py` for Gate B rejection before any resolver/downloader/loader/composition/benchmark call, frozen 10-cell matrix generation, mechanical refusal of the Series-A H1a/H1b exception when H1c-vs-H1b is `within_noise`, paired-control equality, alternating adjacent-arm schedule construction, required result fields, and integrity rejection; depend on T034–T035.
- [X] T037 [US3] Validate the Series-B runner, manifest, and model-free test contracts without resolving or loading checkpoints in `specs/002-qwen-bonsai-hybrid-target/evidence/series-b/runner-validation.json`; confirm the gate-first invariant and the smoke/provenance prerequisites are enforced on every execution path before authorizing hardware discovery; depend on T036.

## Phase 15: Complete Fresh Paired Discovery Matrix

**Purpose**: Collect the full authorized Series-B matrix from fresh processes under adjacent, order-balanced B-Q/B-B pairs; do not reuse Series-A throughput.

- [X] T038 [US3] After T037 passes, execute one complete fresh discovery observation for every authorized cell in `specs/002-qwen-bonsai-hybrid-target/evidence/series-b/runs/`, scheduling each target's B-Q and B-B arms adjacently and alternating/balancing arm order across targets; record actual order, use fresh Series-B H0+B-Q and H0+B-B controls, and capture serial target tokens/sec, speculative decode tokens/sec, speculative speedup, draft acceptance/mean accept length, target-forward count, generated tokens per target forward, available draft-width/cap distributions, peak steady-state memory, generated tokens, duration, exact target/drafter provenance, runtime source hashes, prompt IDs/hashes, controls, hardware/environment, and integrity status; require paired serial-target agreement or invalidate causal attribution pending resolution; depend on T037.
- [X] T039 [US3] Validate every fresh discovery result against the frozen matrix and control manifest, required metrics/provenance, integrity criteria, actual adjacent order, fresh-process/reset records, paired serial-target consistency, and result hashes; write `specs/002-qwen-bonsai-hybrid-target/evidence/series-b/discovery-index.json` and stop repeat selection for incomplete, mismatched, or invalid cells; depend on T038.

## Phase 16: Discovery Validation and Repeat Selection

**Purpose**: Choose only decision-relevant matched repeats using observed variability and drift, rather than applying a fixed significance percentage or repeating every cell automatically.

- [ ] T040 [US3] Derive `specs/002-qwen-bonsai-hybrid-target/evidence/series-b/repeat-selection.json` exclusively from the validated fresh Series-B discovery index and immutable run records; select at least three alternating matched pairs for each apparent B-B-vs-B-Q improvement/regression, near-noise drafter difference, apparent crossover transition, apparent hybrid/drafter win over fresh H0, or unresolved FR-020 comparison, explaining the observed paired variability/drift basis and selecting no repeats solely by an invented percentage threshold; depend on T039.

## Phase 17: Required Matched Confirmations

**Purpose**: Run only selected confirmations with every pair fresh and order-balanced; preserve each observation as one independent run.

- [ ] T041 [US3] Execute only the comparisons in T040 as at least three alternating matched pairs per decision-relevant comparison, each as a fresh process or equivalently fully reset execution, in `specs/002-qwen-bonsai-hybrid-target/evidence/series-b/runs/`; record schedules and actual order in `specs/002-qwen-bonsai-hybrid-target/evidence/series-b/confirmation-schedule.json`, preserve all frozen controls and exact provenance, and never reuse an observation as two independent pairs; depend on T040.

## Phase 18: Repeat and Noise Decisions

**Purpose**: Resolve confirmed direction, observed-noise overlap, inconclusive evidence, and invalid comparisons before final interpretation.

- [ ] T042 [US3] Generate `specs/002-qwen-bonsai-hybrid-target/evidence/series-b/repeat-decisions.json` solely from immutable discovery and confirmation records; report paired decode deltas, observed variability/drift, serial-target consistency, and confirmed-win/regression, within-noise, inconclusive, or invalid status for every selected decision; do not call one marginal discovery pair significant; depend on T041.

## Phase 19: Series-B Adjudication and Experiment Conclusion

**Purpose**: Answer every FR-020 question from immutable Series-B evidence and state the evidence-supported experiment conclusion without converting throughput alone into a production recommendation.

- [ ] T043 [US3] Generate `specs/002-qwen-bonsai-hybrid-target/evidence/adjudication/series-b.json` only from immutable Series-B run, checkpoint provenance, integrity, repeat-selection, schedule, and repeat-decision records; separate measured facts, derived metrics, and interpretation, answer whether B-Q remains better for every tested target, whether B-B wins only at B0, whether/where a tested drafter crossover exists, the tested composition interval if any, whether any tested target/drafter beats fresh H0 end-to-end, and whether the best tested pair justifies a separate later boundary-localization experiment; state that no exact untested crossover block is inferred and apply the spec's basic generation sanity checks before calling any apparent H0-beating configuration viable; depend on T042.
- [ ] T044 [US3] Record the final Series-B experiment conclusion and evidence-supported no-production-change disposition in `specs/002-qwen-bonsai-hybrid-target/evidence/adjudication/series-b.json`; distinguish a possible later boundary-localization experiment from a production recommendation, explicitly state that throughput evidence alone does not satisfy the separate coding-agent/task-quality evaluation required for any production recommendation, and do not introduce another target, sweep axis, or runtime change; depend on T043.

### Series-B critical dependency chain

`T030 → T031 (Gate B OPEN + frozen 10-cell matrix/exception decision) → T032 (exact B-B provenance) → T033 (B-B compatibility smoke) → T034 → T035 → T036 → T037 (runner/control validation) → T038 (complete fresh paired discovery) → T039 → T040 → T041 (selected confirmations only) → T042 → T043 → T044`

No hardware performance task precedes OPEN Gate B, exact B-B provenance, B-B compatibility smoke, and runner/control validation. The Series-B checkpoint record lives only under `evidence/series-b/`; the Series-A checkpoint record and all accepted Series-A artifacts remain immutable. The H1a/H1b exception is not authorized because the Series-A adjudication establishes H1c over H1a while H1c-vs-H1b is within observed noise, so neither isolated-block target is an unambiguous confirmed best hybrid.

### Series-B parallel opportunities

There are no parallel hardware tasks: paired arms must remain adjacent, their order must alternate/balance, and all runs must share the pinned runtime, machine, prompts, and reset protocol. Model-free implementation and validation proceed in the explicit T034–T037 dependency chain so the gate-first guard and paired schedule cannot be bypassed.

### Series-B completion summary

- Newly appended tasks: 14 (T031–T044); authorized discovery matrix: 10 target/drafter cells.
- Phase allocation: Gate/matrix decision 1; checkpoint provenance 1; compatibility smoke 1; runner/controls/model-free validation 4; discovery 2; repeat selection 1; confirmations 1; repeat decisions 1; adjudication/conclusion 2.
- Story allocation: Series-B crossover scenario US3, 14 tasks.
- Independent criteria: T036–T037 validate gate-first execution and paired-control contracts without model access; T039 validates all fresh discovery cells; T042 resolves every selected comparison; T043–T044 produce the FR-020 evidence-backed conclusion.
- Parallel opportunities: none for hardware measurements or paired conditions; keep the whole Series-B chain sequential.
- MVP: complete the gated 10-cell fresh discovery and adjudicate only after evidence-derived required confirmations.
- Format: all appended tasks use unchecked checklist items, sequential IDs, the US3 label, concrete file/artifact paths, and explicit dependency references. T001–T030 remain the completed Series-A/Gate-B task set.

## Completion Summary

- Total tasks: 44 (T001–T044).
- By phase: Series-A/prerequisite preflight 3; official donor proof 2; composition 4; model-free tests 4; donor-memory proof 1; real-checkpoint integrity 5; Series-A discovery 3; Series-A repeat selection/confirmation 3; Series-A adjudication 2; Gate B 3; Series-B gate/matrix 1; Series-B checkpoint provenance 1; Series-B compatibility smoke 1; Series-B runner/controls/model-free validation 4; Series-B discovery 2; Series-B repeat selection 1; Series-B confirmations 1; Series-B repeat decisions 1; Series-B adjudication/conclusion 2.
- User-story allocation: US1 Series-A and prerequisite work 27; US2 Gate B 3; US3 Series-B crossover 14.
- Independent criteria: composition contracts are model-free tested in T010–T013; each real target must pass T017–T019; Series-A discovery completeness is checked by T022; selected Series-A comparisons are resolved by T025; Series-A acceptance is T026–T027; mechanical Series-B refusal/open eligibility is T028–T030; Series-B gate/matrix/provenance/smoke/runner readiness is T031–T037; complete fresh Series-B discovery is validated by T038–T039; required repeats are selected and resolved by T040–T042; final Series-B adjudication/conclusion is T043–T044.
- MVP: complete the accepted Series-A/Gate-B chain first; for Series B, establish T031–T037 gate/matrix/provenance/smoke/runner readiness before any performance discovery, then complete the fresh 10-cell matrix and only evidence-required confirmations before T043–T044 adjudication.
- Format: all tasks use checkbox, sequential ID, required story label, concrete artifact/file paths, and explicit dependency references. No task is marked parallel because the gated chain is sequential.
