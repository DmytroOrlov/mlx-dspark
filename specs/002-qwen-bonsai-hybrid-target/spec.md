# Feature Specification: Qwen-Bonsai Hybrid Target Experiment

**Feature Branch**: `002-qwen-bonsai-hybrid-target`

**Created**: 2026-09-25

**Status**: Draft

**Input**: User description: Create a new feature specification governed by constitution v2.0.0 for the full Qwen3.8/Bonsai2 hybrid-target experimental program: Series A with the original Qwen DFlash2 drafter, Gate B, and Series B drafter crossover.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Establish a reproducible hybrid compatibility curve (Priority: P1)

An experiment owner compares progressively larger suffix substitutions of Bonsai2 transformer blocks into the Qwen target while keeping the original production Qwen DFlash2 drafter and all other benchmark controls fixed. The owner needs fresh, matched evidence that separates serial target acceleration from speculative efficiency and end-to-end throughput.

**Why this priority**: Series A tests the core hypothesis and must produce an adjudicated result before any drafter crossover work begins.

**Independent Test**: Run the required target compositions on the target hardware with the fixed original drafter, capture all required metrics and provenance, and produce the Series-A adjudication artifact against a fresh H0 control.

**Acceptance Scenarios**:

1. **Given** the prescribed experiment controls and a runtime revision, **When** Series A is run, **Then** H0, H1a, H1b, H1c, H2, H3, and B0 each have valid matched results and a fresh H0 is included in that series.
2. **Given** any hybrid block substitution, **When** performance is considered, **Then** the Qwen/Bonsai representation boundary and the DFlash tap semantics have been established from the actual repository behavior, and runtime-integrity gates have passed.
3. **Given** Series-A measurements, **When** the series is adjudicated, **Then** absolute metrics, deltas versus fresh H0, measured facts, derived metrics, and interpretation are separated in one reproducibility-backed artifact.
4. **Given** a possible winner or difference near measurement noise, **When** it is reviewed, **Then** repeated matched runs distinguish it from run-to-run noise before it is treated as an adjudicated result.

### User Story 2 - Decide whether Series B may begin (Priority: P1)

The experiment owner uses an explicit gate to prevent a drafter crossover from starting until Series A is complete, technically valid, and adjudicated.

**Why this priority**: Without an enforced barrier, changing the drafter before the target-composition curve is understood would confound the experiment and invite scope drift.

**Independent Test**: Review the gate record against every required precondition and confirm it remains closed if any Series-A result, semantic fact, integrity check, repeat, or adjudication is missing.

**Acceptance Scenarios**:

1. **Given** any incomplete or invalid Series-A prerequisite, **When** Gate B is reviewed, **Then** it remains closed and no Series-B measurement is authorized.
2. **Given** all Gate-B prerequisites are satisfied, **When** the Series-A adjudication is complete, **Then** the gate record may be opened for only the fixed Series-B drafter axis and target matrix.

### User Story 3 - Measure the drafter crossover on frozen targets (Priority: P2)

After Gate B opens, the experiment owner compares both existing DFlash2 checkpoints against the same frozen set of target compositions. This reveals whether and where the Bonsai-specific drafter becomes more effective as the target becomes more Bonsai-like.

**Why this priority**: Series B builds on the valid target curve from Series A and determines whether a drafter change can improve a tested target configuration.

**Independent Test**: Run both drafter arms for every authorized target/drafter cell under paired conditions and produce a Series-B adjudication that answers the specified crossover questions.

**Acceptance Scenarios**:

1. **Given** an open Gate B, **When** the core Series-B matrix is run, **Then** H0, H1c, H2, H3, and B0 are each run with both drafter checkpoints, for ten cells total.
2. **Given** the confirmed best Series-A hybrid is H1a or H1b, **When** the exception is evaluated, **Then** only that one isolated-block variant is additionally paired with both drafters.
3. **Given** a paired Series-B target, **When** its two drafter runs are compared, **Then** target weights and ownership, runtime revision, prompt corpus, plain KV, controller, sampling, generation, and measurement method are identical; only the drafter checkpoint and necessary loading compatibility differ.
4. **Given** completed Series-B results, **When** they are adjudicated, **Then** conclusions identify only tested winners or tested crossover intervals and explicitly allow a no-production-change outcome.

### Edge Cases

- A Qwen/Bonsai boundary may have matching tensor dimensions while carrying an incompatible persistent representation. In that case the substitution is invalid unless a correct explicit bridge is established; performance runs cannot proceed for that composition.
- A Bonsai donor block, its target checkpoint, or the Bonsai-specific drafter may be unavailable or lack reproducible revision provenance. The affected result is invalid and the gate remains closed until the missing artifact is resolved; no substitute checkpoint may be silently used.
- A target tap at an index adjacent to a substituted block may be unaffected if the runtime captures the preceding block output. Tap effects must follow actual runtime semantics, not assumed architecture conventions.
- A benchmark may terminate early, generate a different token count, fail runtime-integrity checks, or lack required metrics. Such a run is not a valid matched result.
- A single apparent win, loss, or marginal difference may fall within run noise. It requires the repeat procedure below before adjudication.
- A Series-B loading adaptation may be needed for a checkpoint. It may address serialization or loading compatibility only and must not change speculative policy or any other paired control.
- No tested configuration may improve end-to-end throughput over fresh H0. This is a valid program outcome.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The experiment MUST use the current `.specify/memory/constitution.md` version 2.0.0 as its governing constitution. The constitution is not part of this feature's deliverables and MUST NOT be modified by this feature.
- **FR-002**: The experiment MUST preserve the production baseline as context: target `mlx-community/Qwen3.8-27B-4bit`, drafter `incoai/Qwen3.8-27B-DFlash2`, plain KV, ordinary production `CapController`, and approximately 45 tok/s on the target M4 Pro machine. Historical results MUST be labeled as context/regression anchors and MUST NOT substitute for fresh matched measurements.
- **FR-003**: Every hybrid target MUST use Qwen embedding, Qwen final norm, and Qwen LM head. Only transformer block ownership may change. Composition MUST be declared reproducibly by explicit donor block ownership or an equivalent extensible configuration and MUST NOT require a separate hardcoded model class for each named variant.
- **FR-004**: Series A MUST have exactly one primary changing axis: transformer block ownership. It MUST use only `incoai/Qwen3.8-27B-DFlash2`, with the same checkpoint weights, architecture, plain KV, ordinary production `CapController`, speculative algorithm, draft-controller behavior, sampling and generation settings, prompt corpus, benchmark Engine path, target hardware, methodology, and relevant runtime revision across matched comparisons.
- **FR-005**: Series A MUST include exactly the following target compositions:

  | Variant | Embedding | Qwen blocks | Bonsai2 blocks | Final norm | LM head |
  | --- | --- | --- | --- | --- | --- |
  | H0 | Qwen | 0–63 | none | Qwen | Qwen |
  | H1a | Qwen | 0–62 | 63 | Qwen | Qwen |
  | H1b | Qwen | 0–61, 63 | 62 | Qwen | Qwen |
  | H1c | Qwen | 0–61 | 62–63 | Qwen | Qwen |
  | H2 | Qwen | 0–59 | 60–63 | Qwen | Qwen |
  | H3 | Qwen | 0–55 | 56–63 | Qwen | Qwen |
  | B0 | Bonsai2 | 0–63 | 0–63 | Bonsai2 | Bonsai2 |

  B0 MUST use the repository-supported full Bonsai2 target, original Qwen drafter, plain KV, ordinary production controller, and cleaned runtime conditions.
- **FR-006**: Series A MUST NOT add the Bonsai-specific drafter, drafter retraining or fine-tuning, cap/max-draft tuning as an experimental variable, WidthPolicy, scheduler experiments, KV8, new rollback semantics, Chad comparisons, unrelated kernel optimization, projection-level ablations, boundaries deeper than H3, or model-quality optimization.
- **FR-007**: Before accepting any Series-A performance result, the experiment MUST establish from the actual repository implementation whether the existing Prism/Bonsai packed Hadamard path returns canonical residual-stream representations at block input/output and across Qwen/Bonsai boundaries. Compatible dimensions alone MUST NOT be treated as proof. If a rotated basis persists across blocks, a correct explicit bridge MUST be demonstrated or the substitution rejected. The existing verified Prism/Bonsai implementation MUST be reused; ternary math MUST NOT be reimplemented for this experiment.
- **FR-008**: Before accepting any Series-A performance result, the experiment MUST inspect and record the actual DFlash target tap meaning (including whether indices denote block input, block output, hidden-state-array offset, or another concrete semantic), and record the number of drafter-consumed hidden states modified for every H0/H1/H2/H3 composition. The drafter MUST NOT be altered to compensate.
- **FR-009**: Based on the current repository Qwen tap loop, indices are zero-based block outputs. With taps `[5, 19, 33, 47, 61]`, H0 changes zero tapped hidden states; H1a, H1b, and H1c change zero; H2 and H3 change one; B0 changes five. This fact MUST be revalidated and recorded against the runtime revision used for the experiment before performance results are accepted.
- **FR-010**: Every Series-A target MUST record serial target tokens/sec with speculation disabled; speculative `decode_tokens_per_sec`; speculative speedup versus that target's serial result; acceptance/accept_len under existing runtime definitions; target-forward count; generated tokens per target forward; available width/cap distribution; peak steady-state memory; generated token count; benchmark duration; and reproducibility provenance. Existing timing decomposition SHOULD be retained when available without invasive instrumentation. A new profiler MUST NOT be created solely to obtain per-component milliseconds.
- **FR-011**: Each benchmark series MUST include a fresh H0 measured under that series' matched conditions. Comparisons MUST use identical prompt corpus, generation settings, plain-KV path, controller, drafter where fixed, runtime revision, machine, and measurement method. Any result missing a fresh H0 or a required metric is not adjudicable.
- **FR-012**: The benchmark plan MUST define a repeat/noise procedure. At minimum, each discovery condition MUST receive a matched run; any apparent winner, regression, or difference near noise MUST receive at least three matched repetitions per compared condition, with run order controlled or randomized and the decision based on paired results and observed variability. One marginal run MUST NOT be called significant.
- **FR-013**: Series A MUST conclude with one adjudication artifact containing exact target composition and checkpoint/revision provenance for every target, fresh H0 and all required H1a/H1b/H1c/H2/H3/B0 results, absolute metrics, deltas versus fresh H0, and measured facts separated from derived metrics and interpretation. It MUST answer: whether any hybrid improves end-to-end decode over H0; serial-target acceleration; accompanying speculative-efficiency loss; whether H1c is explained by H1a/H1b or shows material interaction; where the largest useful throughput/acceptance transition occurs among tested points; and whether the original drafter remains effective as the target becomes more Bonsai-like.
- **FR-014**: Gate B MUST remain closed until every required Series-A target has a valid matched result; fresh H0 exists; runtime-integrity requirements pass; representation-boundary and DFlash tap semantics are established; Series-A adjudication is complete; and any apparent winner is confirmed sufficiently to distinguish it from measurement noise. Series B MUST NOT begin before Series-A adjudication.
- **FR-015**: Opening Gate B MUST authorize only the Series-B drafter axis defined in this specification. It MUST NOT authorize new target variants, scheduler changes, cap tuning, KV changes, training, or projection ablations.
- **FR-016**: Before Series B starts, the Bonsai-specific drafter MUST be identified as `naklitechie/Qwen3.8-27B-DFlash2-ternary-bonsai2` and its exact checkpoint revision recorded. Existing local project evidence identifies revision `3fc0d6ef43e933a20f1e9ee53fac6f56402c0f83` and weights SHA-256 `eb141d0fb8621cf7a7c81047b83810fa19b8093543bd05529e9478d9555cb8f1`; the actual selected files and revision MUST be reverified and recorded in the run manifest. No guessed namespace or silent drafter substitution is permitted.
- **FR-017**: Series B MUST use frozen target configurations H0, H1c, H2, H3, and B0, each paired with both original Qwen DFlash2 and Bonsai-specific DFlash2 drafters, for ten core target/drafter cells. If and only if the confirmed best Series-A hybrid by `decode_tokens_per_sec` is H1a or H1b rather than H1c/H2/H3, that single winning isolated-block variant MAY be added and MUST be tested with both drafters. No other Series-A variant or new boundary may be added without a new explicit scope decision.
- **FR-018**: Within each Series-B target pair, target weights and all target ownership MUST be identical. Plain KV, controller, speculative settings, prompt corpus, generation settings, runtime revision, and measurement method MUST be identical except for requirements inherently encoded by the drafter checkpoint and strictly necessary loading compatibility. Any loading adaptation MUST NOT silently change speculative policy.
- **FR-019**: Every Series-B cell MUST record the same metrics and provenance required in Series A. Serial target speed MUST remain constant for the paired runs of each target and serve as a control. The adjudication MUST compare decode throughput, acceptance/accept_len, target forwards, generated tokens per target forward, and speculative speedup, and explicitly attribute paired changes to the drafter only when controls support that attribution.
- **FR-020**: Series B MUST conclude with an adjudication answering whether the original drafter is best for all tested hybrids; whether the Bonsai-specific drafter is best only at B0; whether and within which tested compositions a hybrid crossover occurs; whether either drafter makes a hybrid beat fresh H0 end-to-end; whether the best target/drafter pair justifies a new boundary-localization experiment; and whether evidence supports no production change. It MUST NOT claim an exact untested crossover block.
- **FR-021**: Runtime integrity MUST pass before a result is used for performance claims. Gates MUST cover successful loading, manifest-matched block ownership, valid tensor shapes and dtypes, absence of NaN/Inf corruption, consistent cache advancement and rollback under the existing speculative runtime, deterministic reproducibility, unchanged Qwen-only H0 behavior, and repeatable composition for the same configuration. Hybrid logits or top-1 outputs are not required to match vanilla Qwen.
- **FR-022**: A hybrid appearing to beat H0 MUST pass basic generation sanity checks before being described as viable. Any production recommendation MUST additionally require evaluation on the relevant coding-agent/task-quality workload; throughput, acceptance, and runtime integrity alone do not establish model-quality equivalence.
- **FR-023**: The program MUST permit all of these outcomes: H0 with the original drafter remains best; an H1/H2/H3 hybrid with the original drafter wins; a hybrid with the Bonsai-specific drafter wins; the Bonsai-specific drafter overtakes only near full Bonsai; a crossover exists but no tested pair beats H0; no meaningful hybrid benefit exists; evidence supports a separate later boundary-localization experiment; or no production code is retained beyond reusable backend/support changes.
- **FR-024**: Success MUST mean reproducible evidence that answers the experiment questions, not proof that the hybrid hypothesis is correct. No production change MUST be a valid conclusion.
- **FR-025**: Until both Series-A and Series-B adjudications are complete, implementation and experimentation MUST remain within the defined experiment. Historical rollback correctness investigation, Chad parity, historical high-throughput reconstruction, model-quality optimization as a sweep variable, and any unlisted experiment axes remain out of scope.

### Key Entities *(include if data involved)*

- **Target composition**: A reproducible mapping from each transformer block index to its Qwen or Bonsai2 owner, plus the embedding, final norm, and LM-head owners.
- **Drafter checkpoint**: An exact DFlash2 checkpoint identity, revision, and file provenance used by one experimental arm.
- **Benchmark condition**: The target/drafter cell plus the fixed runtime, controller, KV, sampling, prompt, generation, hardware, and measurement controls.
- **Run result**: A measured benchmark record containing required performance and explanatory metrics, provenance, integrity status, and raw artifact reference.
- **Series adjudication**: A reproducible decision artifact separating measured facts, derived metrics, and interpretations for Series A or Series B.
- **Gate B record**: The explicit closed/open decision and evidence that all Series-A prerequisites are satisfied before Series B begins.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All seven required Series-A target configurations have valid, provenance-backed matched results, including a fresh H0 control, with every required metric present.
- **SC-002**: Before Series-A throughput is accepted, both the Qwen/Bonsai representation boundary and actual DFlash tap semantics are documented against the measured runtime revision, and every required target passes runtime-integrity gates.
- **SC-003**: The repeat procedure provides at least three matched repetitions per compared condition for any apparent winner, regression, or near-noise difference; no adjudicated marginal result relies on one run.
- **SC-004**: Gate B remains closed until all specified Series-A prerequisites pass and Series A has an adjudication artifact.
- **SC-005**: After Gate B opens, all ten core Series-B cells have matched results, or an explicitly documented validity issue prevents completion; an authorized H1a/H1b exception, if triggered, adds exactly two cells.
- **SC-006**: The two adjudication artifacts answer every required Series-A and Series-B question and identify tested crossover intervals without assigning an exact untested boundary.
- **SC-007**: Final program conclusions explicitly name one permitted outcome, including the possibility that H0 remains best and no production change is warranted.

## Assumptions

- The feature number is the next sequential number, `002`, because `specs/001-bonsai-rollback-parity` is the only existing feature directory.
- The ordinary production benchmark environment is the target M4 Pro machine and production Engine path described in the project constitution and user-provided baseline; exact hardware/runtime provenance will be captured for each run.
- The current repository-supported full Bonsai2 target is `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`; its exact revision must be captured when measured.
- Repository implementation evidence indicates Prism packed projections apply their Hadamard transform to projection inputs, while embedding lookup applies the inverse transform; projection outputs and the residual stream are canonical. Series-A runtime revision must still verify this at actual Qwen/Bonsai block boundaries.
- The Qwen DFlash tap loop currently captures each requested hidden state after its indexed transformer block. Its tap list in the Bonsai-specific checkpoint is `[5, 19, 33, 47, 61]`; the current effect counts are specified in FR-009 and must be rechecked for the exact runtime and drafter arm.
- The Bonsai-specific drafter's local cached checkpoint revision and file hash are provenance leads from existing project evidence, not substitutes for re-verifying the checkpoint actually used in Series B.
- The prompt corpus, generation length, exact target revision, benchmark duration, and runtime revision will be pinned in the later experiment plan and run manifests; paired arms use the same values.
- The request's final out-of-scope sentence ends after “T”; the explicit exclusions above capture the exclusions fully stated earlier in the request and current constitution. No additional scope is inferred from the incomplete fragment.
