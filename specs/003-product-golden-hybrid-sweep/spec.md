# Feature Specification: Product Golden Hybrid Sweep

**Feature Branch**: `003-product-golden-hybrid-sweep`

**Created**: 2026-09-26

**Status**: Draft

**Input**: User description: Create a narrow product-anchored benchmark sweep starting from the historical Qwen product configuration near 45 decode tokens/sec and measure seven specified Qwen/Bonsai2 target compositions with both the original Qwen drafter and Bonsai-specific drafter. Establish a fresh H0+B-Q golden canary first, stop before hybrids if it returns to the known 32–36 tokens/sec / ~49% acceptance regime, and otherwise report matched product measurements. Do not run benchmarks during specification.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Establish the Fresh Product Golden (Priority: P1)

As a product performance owner, I need to know whether the current production Qwen configuration still reproduces the historical ~45 decode tokens/sec regime under its original workload, so that hybrid results are interpreted against the correct product baseline.

**Why this priority**: Every composition comparison depends on the fresh Qwen product control. A failed canary means subsequent hybrid cells cannot answer the product question.

**Independent Test**: Execute only the specified fresh H0+B-Q canary with the Golden-A request semantics, inspect its result and compare its throughput/acceptance regime with the explicit stop condition.

**Acceptance Scenarios**:

1. **Given** the immutable Golden-A target and drafter, exact request, and a fresh process/cache, **When** the H0+B-Q canary is measured, **Then** its record includes product throughput and acceptance plus enough request/runtime identity to adjudicate whether it returned to the historical product regime.
2. **Given** the canary is in the known 32–36 tokens/sec and ~49% acceptance regime, **When** the result is adjudicated, **Then** the sweep stops before any hybrid cell and records a narrow mismatch artifact naming concrete differences from Golden A.
3. **Given** the canary clearly returns to the historical ~45 tokens/sec product regime, **When** it is adjudicated, **Then** that same immutable observation is retained as matrix cell H0+B-Q and execution proceeds directly to the remaining 13 fresh matched cells without broad historical investigation.

### User Story 2 - Compare the Fixed Composition and Drafter Matrix (Priority: P1)

As a product performance owner, I need matched measurements for each prescribed target composition with both drafter checkpoints, so that I can see how replacing Qwen blocks with Bonsai2 blocks changes end-to-end speculative throughput under the real product workload.

**Why this priority**: This is the sole product question of the feature and the decision-making evidence.

**Independent Test**: With a passing canary, verify one fresh process and empty/fresh cache for each of the seven targets × two drafters, and verify all required measurements and configuration identity are present.

**Acceptance Scenarios**:

1. **Given** a passing fresh H0+B-Q canary, **When** the mandatory matrix runs, **Then** it contains exactly H0, H1a, H1b, H1c, H2, H3, and B0 paired with B-Q and B-B, each as a fresh product cell.
2. **Given** any matrix cell, **When** its result is recorded, **Then** it identifies exact block ownership, immutable drafter identity/revision, runtime source hashes, execution order, prompt/input digest, fresh process/cache status, and all required throughput, acceptance, forward, generation, cap/width, and memory measurements.
3. **Given** the matrix is complete, **When** results are presented, **Then** the monotonic path H0 → H1a → H1c → H2 → H3 → B0 is visually clear for each drafter and H1b is shown as an interaction diagnostic beside H1a/H1c.

### User Story 3 - Adjudicate Product Outcome and Follow-up (Priority: P2)

As a product performance owner, I need a concise report comparing current fresh results with both historical observations, so that I can determine whether the Qwen product baseline was reproduced, how each drafter responds to Bonsai2 substitution, and whether a small local follow-up is justified.

**Why this priority**: Measurement is useful only if the final report answers the bounded product questions without overclaiming.

**Independent Test**: Review the human-readable table and machine-readable artifact against the required context rows, all 14 current cells (when run), derived deltas, and five interpretation questions.

**Acceptance Scenarios**:

1. **Given** historical and current evidence, **When** the report is produced, **Then** Golden A is reported as 45.779 tokens/sec and Golden B as ~45.4 tokens/sec clearly labeled hot-prefix/context-only, with Golden B excluded from fresh paired deltas.
2. **Given** completed fresh cells, **When** the report is reviewed, **Then** it answers only the five specified product interpretation questions and does not infer an exact untested crossover, make model-quality claims, or recommend a production change from one marginal run.
3. **Given** a hybrid reaches or exceeds fresh H0+B-Q, two conclusion-critical cells are plausibly within observed fresh-H0 drift, or a surprising local maximum determines localization, **When** repeat decisions are made, **Then** only the relevant matched additional observations are performed; no universal significance threshold or automatic tripling is used.

### Edge Cases

- If the canary result is ambiguous between historical and known lower-throughput regimes, do not start hybrid cells; preserve the observed result and the narrow request/runtime mismatch evidence for adjudication.
- If an immutable target/drafter revision or selected weight digest differs from the specified identity, do not count that cell as a matched observation.
- If a cell uses reused prefix/KV/drafter state, or its process/cache freshness cannot be asserted, exclude it from fresh paired deltas and record the failure.
- If a mandatory cell fails to load or complete, retain the failure and its execution order; do not silently substitute another configuration or claim a complete matrix.
- If repeat observations are warranted by the stated policy, compare like-for-like cells under the same controls and identify which original observation each repeat adjudicates.
- If a promising region lies between tested suffix depths, report only that a bounded localization may be warranted; do not claim the exact crossover block.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The experiment MUST answer only how end-to-end speculative decode throughput changes when the real Qwen product target progressively replaces specified blocks with Bonsai2 blocks, and whether the Bonsai-specific drafter reaches or exceeds the fresh Qwen product level.
- **FR-002**: The Golden-A context record MUST identify target `mlx-community/Qwen3.8-27B-4bit` at revision `10c35caafbb80f7dc6a7a432cdd11af10a6d4818`, drafter `incoai/Qwen3.8-27B-DFlash2` at revision `015e795645c74b1a0eeef3b570031fb62e769bc5`, and the historical observation: serial 15.4992848 tok/s, speculative 45.7788518 tok/s, 2.95361× speedup, acceptance 0.6482335, 94 target forwards, 93 rounds, 516 generated tokens, 5.4893617 generated tokens/target-forward, draft width 7 for all 93 rounds, and approximately 15.916 GiB peak resident runtime.
- **FR-003**: The Golden-A request MUST use the workload “Write a production-quality Python LRU cache with tests and type hints.”, rendered prompt count 26, input IDs `[248045, 846, 198, 7734, 264, 5492, 21408, 12654, 436, 34810, 6297, 440, 6813, 321, 913, 29642, 13, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271]`, input-ID SHA-256 `8bda1dedde8e022452b7dafaa3677dca47dedc350258c0885c4d1e5fde155f59`, temperature 0, top_p 1, top_k 0, thinking disabled, and max_tokens 512.
- **FR-004**: The experiment record MUST include Golden B as a secondary context-only observation (~45.4 decode tokens/sec, accept_len 5.548, 94 target forwards, 516 completion tokens) and MUST mark that its hot prefix cached 25 of 26 prompt tokens; it MUST NOT be mixed into fresh-run paired deltas.
- **FR-005**: Before performance execution, only cheap/live integrity checks necessary to ensure reused runtime and checkpoints have not materially changed MUST be performed. The exact selected B-B files MUST be cheaply revalidated against revision `0059b38aa255698b1a87305eb3fbb5a3cfd616e2` and weight SHA-256 `eb141d0fb8621cf7a7c81047b83810fa19b8093543bd05529e9478d9555cb8f1`. Broad correctness re-proving is out of scope.
- **FR-006**: Performance execution MUST begin with exactly one fresh H0+B-Q Golden-A canary, which is also the H0+B-Q cell of the 14-cell first-pass matrix. If it remains in the known 32–36 tok/s and ~49% acceptance regime, execution MUST stop before hybrids and produce a narrow mismatch artifact listing concrete request/runtime differences from Golden A. If it clearly returns to the historical ~45 tok/s product regime, its immutable result MUST be retained as H0+B-Q and execution MUST continue directly to only the remaining 13 cells. An optional final fresh H0+B-Q drift bracket may be run only if substantial runtime duration/order drift is suspected or it is needed to adjudicate a product-competitive result; it is not a mandatory matrix cell.
- **FR-007**: The target family MUST contain exactly these first-pass compositions: H0 (Qwen embedding, blocks 0–63, final norm and LM head; no Bonsai), H1a (Qwen embedding, blocks 0–62, final norm and LM head; Bonsai block 63), H1b (Qwen embedding, blocks 0–61 and 63, final norm and LM head; Bonsai block 62), H1c (Qwen embedding, blocks 0–61, final norm and LM head; Bonsai blocks 62–63), H2 (Qwen embedding, blocks 0–59, final norm and LM head; Bonsai blocks 60–63), H3 (Qwen embedding, blocks 0–55, final norm and LM head; Bonsai blocks 56–63), and B0 (full official Bonsai2 target with its native embedding, all decoder blocks, final norm, and LM head; endpoint control, not a hybrid or “H64”). H1b MUST be labeled as an interaction diagnostic, not a monotonic suffix point.
- **FR-008**: The first-pass matrix MUST contain exactly 14 fresh physical observations total, including the passing canary as H0+B-Q, with every target paired with B-Q (`incoai/Qwen3.8-27B-DFlash2`, revision `015e795645c74b1a0eeef3b570031fb62e769bc5`) and B-B (`naklitechie/Qwen3.8-27B-DFlash2-ternary-bonsai2`, revision `0059b38aa255698b1a87305eb3fbb5a3cfd616e2`, selected weight SHA-256 as above). Following a passing canary, run only the remaining 13 cells. Do not rerun H0+B-Q to satisfy matrix cardinality. The fixed order MUST be: H0+B-Q, H0+B-B, H1a+B-B, H1a+B-Q, H1b+B-Q, H1b+B-B, H1c+B-B, H1c+B-Q, H2+B-Q, H2+B-B, H3+B-B, H3+B-Q, B0+B-Q, B0+B-B.
- **FR-009**: Every cell MUST use identical Golden-A product request semantics, T027 production Engine request path and normal production prefix-cache initialization semantics, plain target KV, ordinary production CapController/automatic cap behavior, no WidthPolicy, no KV8, no experimental scheduler, no cap/width tuning, same warmup and memory-guard semantics, same runtime revision and machine, and a fresh process with empty/fresh request state and no cross-cell prefix/KV/drafter reuse. Preserve normal prefix-cache initialization while asserting no reused prompt state for the fresh measured request.
- **FR-010**: Only target composition and drafter checkpoint MAY differ among measured cells. Production runtime MUST NOT be changed to make a benchmark pass.
- **FR-011**: Each cell MUST record target composition and exact block ownership; drafter identity/revision; serial target tok/s; speculative decode tok/s; speedup over serial; delta versus fresh H0+B-Q; draft acceptance; mean accepted length using the existing accept_len definition; target-forward count; generated tokens per target forward; generated token count; draft-width/cap distribution; peak memory; prompt/input-ID digest; current runtime source hashes; actual execution order; and fresh-process/fresh-cache assertion.
- **FR-012**: One complete fresh observation per mandatory matrix cell is sufficient for the first pass. Additional matched repetitions MUST be limited to cases where a hybrid appears to reach/beat fresh golden H0+B-Q, two conclusion-critical cells plausibly fall within observed fresh-H0 drift/noise, or a surprising local maximum determines whether to localize a boundary. No universal percentage threshold or automatic tripling is permitted.
- **FR-013**: The final human-readable table MUST contain historical Golden A and Golden B context rows plus all 14 current fresh cells when the matrix runs. Columns MUST include Target, Bonsai-owned blocks, Drafter, Serial tok/s, Spec tok/s, Delta vs fresh H0+B-Q, Speedup, Acceptance, Mean accepted, Target forwards, Generated/forward, Peak GiB, and Status/notes.
- **FR-014**: The final presentation MUST make the monotonic sequence H0 → H1a → H1c → H2 → H3 → B0 obvious separately for B-Q and B-B, with H1b displayed beside H1a/H1c, and MUST provide one machine-readable artifact.
- **FR-015**: The final interpretation MUST answer only: whether fresh H0 reproduces the historical ~45 tok/s product regime; throughput change along the tested composition path for B-Q; the same path for B-B; whether any tested hybrid+B-B reaches/exceeds fresh H0+B-Q; and whether an observed region merits a small boundary-localization follow-up. It MUST NOT infer an exact untested crossover, make model-quality claims, or recommend production changes solely from one marginal throughput run.
- **FR-016**: Any later boundary localization MUST be narrowly limited to an observed product-competitive local maximum, sign change, or plausible ~45 tok/s region between tested suffix depths. No arbitrary additional boundaries are permitted in the first pass.
- **FR-017**: Existing feature-002 Series-A and Series-B evidence MUST remain untouched. T040–T044 MUST NOT be executed as part of this feature. Feature-002 performance values MUST NOT be used as the product baseline.
- **FR-018**: The experiment MUST reuse accepted feature-002 evidence and implementation for mixed-target construction, 64-block ownership, canonical residual boundaries, no Qwen/Bonsai bridge, Prism packed donor execution, DFlash taps `[5,19,33,47,61]`, cache advance/rollback integrity, official Bonsai2 target provenance, original and Bonsai-specific DFlash provenance, plain target KV compatibility, and ability to execute H0/H1a/H1b/H1c/H2/H3/B0. It MUST NOT reopen Gate B, rollback proof, Prism representation, tap semantics, loader provenance, scheduler research, or feature-002 Series-B adjudication.
- **FR-020**: The T027 product runner/request path MUST be authoritative for product performance semantics. Feature-002 `run_series_condition` MUST NOT be imported or called unchanged when that would retain its p01/p02/p03 corpus, 128-token generation, prefix-cache-off setting, or other feature-002 request controls. Reuse of small helpers is allowed only where Golden-A request and T027 Engine semantics remain intact.
- **FR-019**: The experiment MUST NOT include feature-001 archaeology beyond the exact T027 product runner and Qwen Golden-A raw result needed to reproduce Golden A. It MUST NOT investigate Chad parity, test KV8, tune cap/max-draft, add WidthPolicy, change scheduler, train/fine-tune drafters, build an invasive profiler, or perform model-quality evaluation.

### Key Entities *(include if data involved)*

- **Target Composition**: A named target configuration with explicit ownership of each of the 64 transformer blocks and Qwen/Bonsai2 family identity.
- **Drafter Checkpoint**: An immutable drafter identity, revision, and selected-file digest associated with a measured cell.
- **Product Cell Observation**: One fresh process/cache measurement pairing a target composition with one drafter and recording request, runtime, execution order, performance, acceptance, generation, cap/width, and memory data.
- **Historical Golden Context**: A historical product observation with its own request/cache context and explicit limits on comparison.
- **Canary Adjudication**: The decision to stop or continue based on whether fresh H0+B-Q returns to the historical product regime.
- **Sweep Report**: The human-readable comparison table and machine-readable evidence artifact with deltas, status, and bounded interpretation.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The feature produces exactly one fresh H0+B-Q canary as matrix cell 1 before any hybrid performance cell; a lower-regime canary prevents all hybrid cells from running.
- **SC-002**: After a passing canary, the first-pass evidence contains exactly 14 mandatory physical observations total: the canary retained as H0+B-Q plus one fresh observation for each of the remaining 13 target/drafter pairs, with all required cell fields present.
- **SC-003**: Every included product observation can be traced to the exact Golden-A prompt/input digest, target and drafter revisions, runtime source hashes, machine, execution order, and fresh-process/fresh-cache assertion.
- **SC-004**: The report visibly distinguishes the 45.779 tok/s Golden A result, the ~45.4 tok/s hot-prefix Golden B context, and fresh current cells; Golden B contributes to zero fresh paired deltas.
- **SC-005**: The report presents both drafter paths in the prescribed composition order and answers all five bounded interpretation questions without asserting an untested crossover or a model-quality conclusion.
- **SC-006**: No feature-002 Series-A/Series-B evidence is modified, and no feature-002 T040–T044 execution is included.

## Assumptions

- The currently accepted feature-002 runtime and evidence remain authoritative for the listed integrity and provenance questions; only the specified cheap/live checks need confirmation before performance.
- “Clearly returns to the historical ~45 tok/s product regime” is adjudicated against the paired fresh H0+B-Q canary’s throughput and acceptance signature, with the stated 32–36 tok/s/~49% regime as the explicit stop condition; no arbitrary statistical threshold is introduced.
- A fresh process and empty/fresh request cache can be asserted for each cell while retaining the production Engine’s normal prefix-cache initialization behavior.
- Delta versus fresh H0+B-Q uses the accepted fresh canary value as its denominator/reference for the product matrix; Golden B and feature-002 observations are context only.
- The feature may end with a canary mismatch artifact and no hybrid table rows if the canary fails; that is a valid outcome and does not authorize a broader investigation.
- No boundary localization is part of the mandatory first pass; it is a separate narrowly scoped follow-up only if the stated observed-region conditions apply.
