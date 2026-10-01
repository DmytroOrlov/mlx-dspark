# Feature Specification: Golden Panel Layer-Drafter Matrix

**Feature Branch**: `005-golden-panel-layer-drafter-matrix`

**Created**: 2026-09-26

**Status**: Draft

**Input**: User description: Measure how target-layer replacement and drafter choice affect speculative decode on five already-certified long-form golden prompts, retaining prompt-level evidence and reporting robustness across the panel.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Compare every configuration per prompt (Priority: P1)

As an experiment reviewer, I need the full target-by-drafter matrix visible separately for every prompt so I can determine whether a result generalizes instead of being hidden by an aggregate.

**Why this priority**: The primary purpose is to establish cross-prompt robustness and avoid conclusions driven by one prompt.

**Independent Test**: Inspect the first-pass evidence and verify there are exactly 70 observations, covering each of five prompts × seven targets × two drafters, with all five prompt cells visible in the raw and retention tables.

**Acceptance Scenarios**:

1. **Given** the five pinned prompt inputs and seven target variants, **When** the first-pass run is complete, **Then** each prompt/configuration pair has one fresh-process observation and a corresponding provenance record.
2. **Given** completed observations, **When** the raw and retention tables are produced, **Then** every configuration row shows P05, P07, P08, P14, and P17 individually, and every baseline comparison uses that prompt’s own H0+B-Q observation.
3. **Given** any missing or invalid cell, **When** the panel is summarized, **Then** the missing status is visible and no complete-panel robustness claim is made for that configuration.

### User Story 2 - Explain throughput differences (Priority: P2)

As a reviewer, I need drafter deltas and per-cell mechanism measurements so I can see how acceptance, target forwards, controller behavior, and target runtime accompany throughput changes.

**Why this priority**: The experiment compares two drafter choices and target compositions; throughput alone cannot describe the observed behavior.

**Independent Test**: Verify the drafter-delta table covers all seven targets and five prompts, and the mechanism table gives acceptance, generated tokens per target forward, forwards, width/cap behavior, and throughput for all observations.

**Acceptance Scenarios**:

1. **Given** paired B-Q and B-B results for a target and prompt, **When** drafter deltas are reported, **Then** the value is B-B tok/s minus B-Q tok/s for that same target and prompt.
2. **Given** mechanism metrics and throughput, **When** results are interpreted, **Then** measurements are distinguished from hypotheses and no causal claim is made solely from association.
3. **Given** B0 results, **When** its speculative outcomes are discussed, **Then** they are compared with its recorded acceptance and its high serial-target speed context without assuming that serial speed compensates for acceptance changes.

### User Story 3 - Adjudicate robustness and targeted repeats (Priority: P3)

As an experiment owner, I need explicit per-configuration thresholds and a limited repeat policy so conclusions identify broadly competitive settings while keeping the experiment bounded.

**Why this priority**: The panel is intended to distinguish repeatable performance from isolated prompt wins and ordinary run drift.

**Independent Test**: Review the tables and repeat log for minimum/median/maximum rates and retention, counts meeting 40/45/50 tok/s, highest minimum-rate configuration, configurations meeting the threshold on all prompts, and repeat decisions limited to the specified triggers.

**Acceptance Scenarios**:

1. **Given** five valid prompt results for each configuration, **When** robustness is reported, **Then** the highest minimum-throughput configuration and all configurations reaching at least 40 tok/s and at least 45 tok/s on all five prompts are explicitly identified.
2. **Given** a non-H0 result at or above same-prompt H0+B-Q, a conclusion-critical near-drift comparison, a relevant H1b+B-B result within roughly 3%, or a surprising local maximum, **When** adjudication selects repeats, **Then** only the triggered comparisons receive one fresh matched repeat for initial adjudication.
3. **Given** H1b appears distinct from H1a and H1c across multiple prompts, **When** the primary panel is adjudicated, **Then** a bounded block-localization follow-up may be proposed; no exhaustive layer sweep begins as part of this feature.

### Edge Cases

- The current tokenizer may produce input IDs or digests different from historical qualification; exact prompt text remains unchanged, and current-tokenizer IDs/digests are regenerated and pinned before physical runs.
- A run may fail, produce fewer than the qualified minimum of 410 generated tokens, violate freshness/provenance, or use mismatched revisions/settings; it is marked invalid or failed, not silently substituted into the matrix.
- H0+B-Q is itself measured freshly per prompt and is the sole baseline for that prompt’s deltas and retention.
- A configuration may exceed a threshold on some prompts and miss it on others; all cells remain visible and counts/minimums reflect the five individual observations.
- A repeat may differ from its discovery result; both raw results and the matched adjudication remain visible.
- B0 is the full native official Bonsai2 target (embedding, all blocks, final norm, and LM head), and is never represented as H64.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The experiment MUST use exactly these five prompt IDs and exact prompt texts: P05, “Write a Python function that merges two sorted lists into a single sorted list. Include tests and a short complexity analysis.”; P07, “Write a Python function that returns the first non-repeating character in a string. Include type hints and unit tests.”; P08, “Write a production-quality Python implementation of a stack with push, pop, peek, and is_empty methods. Include type hints and tests.”; P14, “Write a Python function that computes the edit distance between two strings using dynamic programming. Include tests and explain the time and space complexity.”; P17, “Write a Python function that performs topological sorting on a directed acyclic graph. Include cycle detection, type hints, and tests.”
- **FR-002**: Before physical execution, the experiment MUST regenerate and pin exact input IDs and their digest for each exact prompt under the current tokenizer. The qualification reference rates MUST be recorded as context: P05 52.798, P14 51.113, P08 47.854, P07 45.211, and P17 44.017 tok/s; each prompt previously generated at least 410 tokens in both qualification runs.
- **FR-003**: The experiment MUST use the accepted target family unchanged: H0 all Qwen; H1a Bonsai block 63 only; H1b Bonsai block 62 only; H1c Bonsai blocks 62–63; H2 Bonsai blocks 60–63; H3 Bonsai blocks 56–63; and B0 the full native official Bonsai2 target, including embedding, all blocks, final norm, and LM head. H1a/H1b/H1c ownership and B0 identity MUST be explicit.
- **FR-004**: The two drafters MUST be B-Q `incoai/Qwen3.8-27B-DFlash2` at revision `015e795645c74b1a0eeef3b570031fb62e769bc5` and B-B `naklitechie/Qwen3.8-27B-DFlash2-ternary-bonsai2` at revision `0059b38aa255698b1a87305eb3fbb5a3cfd616e2`. The selected B-B weight SHA-256 MUST equal `eb141d0fb8621cf7a7c81047b83810fa19b8093543bd05529e9478d9555cb8f1`.
- **FR-005**: The mandatory first pass MUST comprise exactly 70 physical observations: five prompts × seven targets × two drafters. Each observation MUST use a fresh process, fresh Engine, fresh/empty measured request, normal production prefix-cache initialization with zero useful prior prefix reuse, thinking disabled, temperature 0, top_p 1, top_k 0, max_tokens 512, plain KV, ordinary production CapController, no WidthPolicy, no KV8, no tuning, the same machine/runtime, and immutable target/drafter revisions.
- **FR-006**: Runs MUST keep each target’s B-Q/B-B pair adjacent and preserve the defined drafter order within each target pair. The physical order for forward prompt blocks P05, P08, and P17 MUST be: (1) H0+B-Q, (2) H0+B-B, (3) H1a+B-B, (4) H1a+B-Q, (5) H1b+B-Q, (6) H1b+B-B, (7) H1c+B-B, (8) H1c+B-Q, (9) H2+B-Q, (10) H2+B-B, (11) H3+B-B, (12) H3+B-Q, (13) B0+B-Q, (14) B0+B-B. The physical order for reverse prompt blocks P07 and P14 MUST reverse the target-pair order while preserving each pair's already-defined drafter order: (1) B0+B-Q, (2) B0+B-B, (3) H3+B-B, (4) H3+B-Q, (5) H2+B-Q, (6) H2+B-B, (7) H1c+B-B, (8) H1c+B-Q, (9) H1b+B-Q, (10) H1b+B-B, (11) H1a+B-B, (12) H1a+B-Q, (13) H0+B-Q, (14) H0+B-B.
- **FR-007**: The primary metric MUST be speculative decode tok/s. The feature MUST NOT run serial target generation for all 70 cells; serial control is already available from feature 003.
- **FR-008**: Every raw physical observation MUST record the following measured-per-cell fields, and each written raw physical artifact MUST remain immutable: prompt identity (prompt ID, prompt-text SHA, and input-IDs SHA); target, exact target composition, Bonsai-owned block count, and drafter; generated tokens; speculative tok/s; acceptance; mean accepted draft tokens; target forwards; generated tokens per target forward; rounds; width distribution; cap distribution; request duration; active baseline memory before measured generation in bytes and GiB; peak runtime memory in bytes and GiB; peak increment above active baseline in bytes and GiB; freshness/provenance; exact revisions; runner SHA; output SHA; and status. Raw physical artifacts MUST NOT be rewritten or enriched with derived fields after they are written.
- **FR-008a**: After the same prompt's H0+B-Q observation exists, a derived matrix/index artifact, final reports/tables, and validation output as appropriate MUST contain the derived throughput delta and retention, peak-memory delta, GiB saved, and memory saving percentage for each cell. The derived values MUST use that prompt's fresh H0+B-Q observation. For forward prompt blocks, these values MAY be calculated as soon as a cell's same-prompt baseline exists. For reverse prompt blocks, these values MUST be calculated after the H0+B-Q observation is available; they MUST NOT be required in raw cell evidence written before that baseline exists. Raw measured memory remains mandatory in every physical observation.
- **FR-009**: Derived throughput delta MUST be calculated as cell tok/s minus same-prompt H0+B-Q tok/s. Throughput retention MUST be cell tok/s divided by same-prompt H0+B-Q tok/s, expressed as a percentage. Peak-memory delta MUST be cell peak runtime memory minus same-prompt H0+B-Q peak runtime memory; GiB saved MUST be same-prompt H0+B-Q peak runtime memory minus cell peak runtime memory; memory saving percentage MUST be GiB saved divided by same-prompt H0+B-Q peak runtime memory, expressed as a percentage. Cross-prompt baselines MUST NOT be used. For H0+B-Q itself, throughput delta is 0, throughput retention is 100%, peak-memory delta is 0, GiB saved is 0, and memory saving is 0%.
- **FR-010**: Reporting MUST provide five human-facing tables, with all five individual prompt observations visible before summary statistics:
  1. **Raw Prompt × Config Matrix**: All 14 target/drafter rows with P05/P07/P08/P14/P17 tok/s, minimum/median/maximum tok/s, and counts at least 40/45/50 tok/s; also median peak GiB across the five prompts, maximum peak GiB, median GiB saved versus same-prompt H0+B-Q, and median memory saving percentage.
  2. **Throughput Retention + Memory**: For every configuration and prompt, show tok/s, throughput retention versus same-prompt H0+B-Q, peak GiB, and GiB saved versus same-prompt H0+B-Q. Summaries MUST include minimum and median throughput retention, median and maximum peak GiB, median GiB saved, and median memory saving percentage.
  3. **Drafter Delta**: For every target and prompt, show B-B tok/s minus B-Q tok/s, with median drafter delta and counts of B-B and B-Q wins.
  4. **Mechanism Breakdown**: For every prompt × target × drafter, show tok/s, acceptance, mean accepted draft tokens, generated tokens per target forward, target forwards, rounds, width distribution, cap distribution, active baseline GiB, peak GiB, peak increment GiB, and request duration. This table MUST NOT be used to infer causality automatically.
  5. **Speed / Memory Tradeoff**: One row per target/drafter configuration, showing target, Bonsai-owned block count, drafter, minimum tok/s across five prompts, median tok/s, counts at least 40/45/50 tok/s, minimum throughput retention, median throughput retention, median peak GiB, maximum peak GiB, median GiB saved versus H0+B-Q, and median memory saving percentage. Speed and memory MUST NOT be collapsed into a weighted score.
- **FR-010a**: All derived memory deltas and savings MUST be calculated against the same prompt's fresh H0+B-Q observation in the derived matrix/index, reports/tables, or validation output. They MUST NOT require rewriting immutable raw physical observations. No historical memory value, including feature-003 values, may substitute for a feature-005 measurement.
- **FR-011**: The report MUST show two explicit visual paths, B-Q and B-B, each H0 → H1a → H1c → H2 → H3 → B0, for every prompt. H1b MUST appear separately as the block-62 interaction diagnostic, and H1a/H1b/H1c MUST also be displayed together.
- **FR-012**: No more than one fresh matched repeat per comparison is required for first adjudication. Repeats MUST be limited to cases where a non-H0 configuration reaches/exceeds same-prompt H0+B-Q, H1b+B-B is within roughly 3% of same-prompt H0+B-Q and relevant to interpretation, conclusion-critical configurations plausibly fall within ordinary run drift, or a surprising local target maximum appears. Repeat selection and outcomes MUST be reported.
- **FR-013**: The experiment MUST NOT begin a full 64-layer sweep. Only if the five-prompt matrix establishes reproducible H1b behavior distinct from H1a/H1c may the report propose bounded single-block swaps (60, 61, 62, 63) and/or suffixes (61–63, 60–63, 59–63, 58–63, 57–63, 56–63) with both drafters. These are a conditional proposal, not part of the mandatory 70-cell run.
- **FR-014**: The report MUST show the full prompt matrix before any aggregate summary and MUST NOT headline one aggregate tok/s or label an overall drafter winner. It MUST state the configuration with highest minimum tok/s, configurations at or above 40 and 45 tok/s on all five prompts, per-target B-B versus B-Q deltas, whether H1b behavior repeats, and whether B0’s high serial target speed compensates for lower speculative acceptance on any prompts, based on measured observations.
- **FR-014a**: The report MUST include a descriptive speed/memory Pareto view. A configuration may be called Pareto-dominated only when another measured configuration is at least as good on throughput robustness and memory use, and strictly better on at least one of those dimensions. The report MUST NOT assign an arbitrary overall score or ranking.
- **FR-014b**: Based on per-observation memory and throughput evidence, the final report MUST explicitly answer: how much measured memory is saved as Bonsai ownership increases; at what substitutions memory begins to fall materially; which configurations remain at or above 40 tok/s on all five prompts while saving memory; which remain at or above 45 tok/s while saving memory; whether H1b+B-B preserves throughput while providing little or no meaningful memory saving; whether H3 is a useful measured speed/memory tradeoff; and the size of B0's memory saving and the throughput/acceptance loss accompanying it.
- **FR-015**: Evidence MUST distinguish measured facts, derived metrics, hypotheses, and interpretation; claims MUST NOT infer causality automatically or make model-quality claims. The work MUST NOT tune prompts, change production runtime, modify feature-003 or feature-004 raw evidence, or introduce unrelated tuning axes.
- **FR-016**: Every observation and report MUST retain enough reproducibility provenance to identify runner SHA, machine/runtime, immutable checkpoint revisions and selected weight digest, prompt/tokenizer inputs, settings, freshness, output digest, and raw artifact identity.

### Key Entities *(include if feature involves data)*

- **Golden Prompt**: One of five certified, immutable prompt texts, its ID, current-tokenizer input IDs, digests, and qualification context.
- **Target Configuration**: A named Qwen/Bonsai composition with exact donor block ownership; includes the full native Bonsai2 endpoint B0.
- **Drafter Configuration**: The B-Q or B-B checkpoint, immutable revision, and selected weight identity.
- **Physical Observation**: One fresh-process measurement for one prompt, target, and drafter, with metrics, provenance, status, and raw output identity.
- **Matched Repeat**: A fresh rerun selected by the repeat policy to adjudicate a specific first-pass comparison.
- **Panel Report**: The five required tables, including the speed/memory tradeoff table, per-prompt visual paths, repeat decisions, descriptive Pareto view, and bounded interpretation of the evidence.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The first-pass matrix contains 70 valid or explicitly failed/statused cells across exactly five prompts, seven targets, and two drafters, with prompt text and current-tokenizer input identity pinned before execution.
- **SC-002**: All five required tables display every applicable prompt-level observation before summary statistics; no prompt cell is hidden by aggregation.
- **SC-003**: Every derived throughput delta/retention and memory delta/saving value can be recomputed from immutable raw physical observations and the same prompt’s H0+B-Q observation, with no cross-prompt baseline substitution or rewriting of raw cells.
- **SC-004**: The report explicitly identifies the highest minimum-rate configuration and the complete sets of configurations meeting at least 40 and 45 tok/s on all five prompts.
- **SC-005**: The report includes per-prompt B-B versus B-Q deltas at every target and explicitly assesses whether H1b’s distinction repeats across prompts and whether B0 offsets acceptance changes in speculative throughput.
- **SC-006**: Every measurement has an auditable raw artifact and provenance record sufficient to establish the stated prompt, configuration, runtime, freshness, and revisions.
- **SC-007**: Any repeats are tied to a stated trigger and are limited to one fresh matched repeat per comparison for initial adjudication; no exhaustive 64-layer sweep is run by default.
- **SC-008**: Every immutable raw physical observation records auditable active baseline, peak, and peak-increment memory in bytes and GiB. After its same-prompt H0+B-Q observation exists, the derived matrix/index and five tables report the corresponding peak-memory delta and saving percentage, with individual prompt-level evidence visible before summaries.
- **SC-009**: The speed/memory tradeoff table and descriptive Pareto view allow reviewers to identify measured dominance using only throughput robustness and memory use, without a weighted score or arbitrary ranking.

## Assumptions

- Feature 003’s accepted target family, execution path, serial-control evidence, and ordinary production controller remain valid references for this experiment.
- The five supplied qualification rates and minimum generated-token counts are historical selection context, not substitutes for fresh measurements under the current tokenizer/runtime.
- Historical feature-003 memory values (including H0 approximately 15.9 GiB, H3 approximately 15.1 GiB, and B0 approximately 8.9 GiB) are context only and MUST NOT substitute for feature-005 per-observation measurements.
- A failed or incomplete cell remains visible with status and may prevent claims requiring a complete five-prompt panel.
- “Roughly 3%” is an operational repeat trigger interpreted against the same prompt’s fresh H0+B-Q rate; repeat selection remains explicitly documented.
- A conditional localization proposal does not authorize running that follow-up within this feature’s mandatory scope.
