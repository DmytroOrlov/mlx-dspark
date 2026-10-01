# Feature Specification: Bonsai Rollback Parity

**Feature Branch**: `[001-bonsai-rollback-parity]`

**Created**: 2026-09-23

**Status**: Draft

**Input**: User description: Investigate and fix the earliest semantic divergence between Chad and mlx-dspark for Bonsai2 / Qwen3.8 DFlash adaptive speculative decoding.

## Clarifications

### Session 2026-09-23

- Q: What must the deterministic regression demonstrate before and after an applicable fix? → A: Before the fix, reproduce or expose the earliest semantic divergence deterministically; after the fix, satisfy the corrected invariant. A pre-fix pass is not required.
- Q: How should numeric comparisons be judged? → A: Keep discrete token/ID/structural invariants exact where their semantic contract requires equality. Require exact top-k identity and deterministic margin identity only for controlled comparisons shown repeatable on the same runtime at the applicable execution width. Derive numeric state bounds per component and dtype from the applicable same-runtime width controls; cross-runtime B3 logits and margins are diagnostic.
- Q: When must rollback cease to be the patch target? → A: If verify/tap state differs before rollback, investigate verify/tap; if target state is equivalent but drafter state differs, investigate drafter reconcile/context.
- Q: What is allowed for Chad and Git during this investigation? → A: Chad may be inspected and run diagnostically but must remain unmodified; no Git command may change repository state.
- Q: What does fixed7 establish, and when may hardware results be claimed? → A: fixed7 remains a control and does not prove variable-width transitions; M4 Pro results may be claimed only after the maintainer actually runs the commands.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Preserve target state after speculative rejection (Priority: P1)

An mlx-dspark maintainer needs a rejected or partially accepted speculative round to leave every target state component as if the target had evaluated only the anchor and accepted draft prefix. This correctness is required before adaptive scheduling can produce comparable widths and token streams.

**Why this priority**: The historical pre-R3 margin report (Chad 0.25, mlx-dspark 0.125) motivated the investigation. Fresh matched-plain-KV captures instead measure 0.125 in both runtimes; the historical scheduling symptom did not reproduce under that control. Rollback remains a hypothesis requiring same-runtime evidence, not an established cause.

**Independent Test**: On a deterministic Qwen3.8 hybrid-cache fixture, compare speculative verify followed by rejection/partial acceptance and the next evaluation against an independently constructed serial/replay execution.

**Acceptance Scenarios**:

1. **Given** identical target, prompt, and initial state, **When** a verify evaluates anchor plus seven draft tokens and zero draft tokens are accepted, **Then** rollback and the next target evaluation match serial execution of the anchor alone for every stateful component.
2. **Given** identical target, prompt, and initial state, **When** a verify evaluates anchor plus seven draft tokens and one draft token is accepted, **Then** rollback and the next target evaluation match serial execution of anchor plus the accepted token for every stateful component.

### User Story 2 - Localize the earliest semantic difference (Priority: P2)

A maintainer needs controlled, round-by-round evidence to determine whether divergence begins in target verification, rollback/replay, hidden-state tap state, or drafter context, so any fix addresses the earliest proven cause.

**Why this priority**: Downstream widths, acceptance, token stream, and throughput are consequences until an earlier difference is ruled out.

**Independent Test**: Record deterministic target and cache observations after each controlled round and compare equivalent states before moving investigation to a later subsystem.

**Acceptance Scenarios**:

1. **Given** matched target/drafter checkpoints, prompt, initial policy state, and widths, **When** the first two rounds are compared, **Then** the evidence identifies the first differing target state or logit observation before attributing later adaptive differences.
2. **Given** rollback is proven equivalent to serial replay for all relevant state, **When** investigating continues, **Then** rollback is not patched and the next earlier divergence is localized.

### User Story 3 - Retain controlled adaptive parity validation (Priority: P3)

A maintainer needs a reproducible local validation procedure after correctness is established, while keeping performance claims separate from deterministic correctness evidence.

**Why this priority**: The user needs to re-check the intended adaptive behavior on the target hardware, but benchmark results cannot establish correctness.

**Independent Test**: Run documented M4 Pro commands with fixed controls and report only measurements actually collected on that machine.

**Acceptance Scenarios**:

1. **Given** deterministic correctness checks pass, **When** local hardware validation is run, **Then** commands cover controlled fixed-width and adaptive comparisons using the same target and physical drafter weights.
2. **Given** no hardware run was performed, **When** results are reported, **Then** no performance result is claimed.

### Edge Cases

- Rejection with zero accepted draft tokens must preserve the accepted-prefix logical state exactly. Rollback-owned recurrent, convolution, and live-KV numeric state is judged against an independent same-runtime frozen-S8 accepted-prefix reconstruction from the exact B2 capture. Fresh S=1/S=2 raw cache values remain serial/cross-width diagnostics and are not the numeric oracle for S=8-originating state.
- Partial acceptance must preserve only the accepted prefix, including when attention and recurrent cache implementations have different rollback mechanisms.
- Cache lengths/offsets may not fully describe recurrent or convolution state; those states and hidden-state taps must also be compared.
- A verify/tap mismatch observed before rollback MUST stop rollback patching and move investigation to the earlier verify/tap path.
- Gate A MUST separate an uncontrolled comparison/control mismatch, an earlier exact semantic/logical input-state difference, a violation of mlx-dspark's own independent same-runtime context/proposal reference, and a cross-runtime proposal difference downstream of distinct valid target projection/op graphs. Record the latter exactly as a diagnostic when both same-runtime R1 references reproduce their own production contexts and proposals; it does not fail A or authorize a drafter/context patch. Committed-output equality alone cannot pass A.
- If target state is equivalent to serial execution but drafter/proposal state differs, target rollback patching MUST stop and investigation MUST move to drafter reconcile/context.
- Rollback is the primary hypothesis only. A rollback fix is permitted only when observations first match through pre-rollback verify/tap and the earliest divergence is introduced by rollback/reconcile.
- Compare discrete token IDs, cache types/shapes, live lengths/offsets/positions, layer order, and accepted-prefix/replay state exactly where their semantic contract requires equality. Exact top-k identity and deterministic top-2 margin identity apply only to controlled same-runtime comparisons at the applicable execution width. Cross-runtime B3 raw logits and margins are diagnostic.
- Do not apply a global floating-point tolerance. Gate C first requires exact structural/logical invariants and an independent same-runtime accepted-prefix reconstruction from the frozen same-width S=8 B2 capture. Recurrent state is independently replayed from captured S=8 `q/k/v/a/b[:keep]` and pre-round recurrent state; the convolution window is independently derived from captured S=8 `conv_input`; attention KV is independently trimmed from frozen S=8 KV to the accepted prefix. Required post-reconcile taps/fused rows are compared with their corresponding same-width S=8 B2 rows. Exact equality is preferred where the same captured execution path is deterministic; if a same-path numeric component is not bit-repeatable, its bound must come from predeclared same-path five-run controls for that exact runtime/root/component/dtype/shape/metric. Fresh S=1/S=2 references remain serial semantic/behavior and cross-width diagnostics and MUST NOT provide a raw numeric cache-state failure criterion for S=8-originating rollback state. Gate D separately checks subsequent committed target behavior against the ordinary committed reference.
- Chad source and runtime may be inspected or executed for diagnosis, but no Chad file may be modified.
- No Git command that changes repository state may be executed during this investigation or its validation; this includes staging, committing, branch/worktree mutation, checkout/switch, reset/restore, stash, merge/rebase/cherry-pick, clean, and tag creation.
- M4 Pro validation commands may be prepared in advance. Throughput, latency, memory, and parity results may be claimed only after the maintainer actually runs the documented comparison on that hardware.
- fixed7 MUST remain a fixed-width parity control. Passing fixed7 does not establish correctness of variable-width state transitions or rollback/reconcile.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: After speculative verify and rejection/partial acceptance, the target MUST represent the exact accepted logical prefix. Cache type/shape/order, live lengths, offsets, positions, and committed IDs are exact invariants. Numeric state created by an S=8 verify MUST match an independent same-runtime accepted-prefix reconstruction from that frozen S=8 capture; raw equality to a fresh S=1/S=2 projection-width execution is not required by itself. Subsequent serial semantic behavior is checked separately at Gate D.
- **FR-002**: A deterministic regression MUST cover verify width eight with draft width seven, followed by zero-accepted rollback and a subsequent target evaluation.
- **FR-003**: A deterministic regression MUST cover verify width eight with draft width seven, followed by one-accepted rollback and a subsequent target evaluation.
- **FR-004**: The regression/probe MUST compare speculative execution with independent same-runtime references appropriate to each boundary. B2 uses an independent same-width S=8 verify/capture reference. Gate C uses an independent accepted-prefix reconstruction from that frozen S=8 capture for rollback-owned recurrent/convolution/live-KV state and uses the corresponding same-width S=8 rows for accepted taps/fused rows. Exact discrete IDs, structure, lengths, offsets, positions, row/tap identity/order, and committed-prefix membership remain exact. Any non-bit-repeatable same-path numeric observation requires its own predeclared five-run same-path component/dtype/shape/metric bound; no global tolerance or cross-width transfer is allowed. Fresh ordinary S=1 and fresh S=2 committed-prefix executions remain independently allocated serial semantic/behavior and width-sensitivity references, but a raw post-reconcile-versus-fresh-S1/S2 cache delta cannot by itself fail C when the production state originated in S=8 projections. Gate D checks subsequent committed target behavior against the ordinary committed reference. A numeric-only Gate A/E regression still requires its already-defined independent same-runtime semantic-state reference and stable control.
- **FR-005**: Investigation MUST record observations before rollback and after reconciliation, distinguishing earlier input-state, target verify/tap, rollback/reconcile, and drafter proposal-state differences. At Gate A, uncontrolled comparison/control mismatch MUST be repaired and A repeated without a product patch; an earlier exact prefill/tokenization/target-cache/input-state difference MUST stop proposal/target-verify analysis and be localized there. Record cross-runtime proposal IDs exactly, but their difference alone MUST NOT fail A when exact semantic/logical invariants pass and each runtime's independent same-runtime R1 reconstruction reproduces its production numerical context (or passes its predeclared applicable five-run control) and proposal. With distinct valid target graphs, classify that difference as implementation-dependent diagnostic. A Gate A proposal/context failure eligible for mlx-dspark localization requires violation of its own applicable independent same-runtime semantic reference/invariant. Equal committed outputs alone cannot pass A. If verify/tap state already differs before rollback, rollback patching MUST stop and investigation MUST move to verify/tap. If target state is equivalent but drafter/proposal state differs under the applicable same-runtime reference, target rollback patching MUST stop and investigation MUST move to drafter reconcile/context. Rollback MAY be patched only if it is the earliest demonstrated divergence.
- **FR-006**: Any runtime fix MUST be limited to mlx-dspark, minimal, and justified by the earliest demonstrated semantic divergence or deterministic failing regression.
- **FR-007**: The investigation MUST preserve fixed7 as a fixed-width parity control; fixed7 success MUST NOT be used as proof that variable-width state transitions are correct. The investigation MUST NOT change target or drafter quantization, selector weights/math, Prism MMA kernels, general fixed-width DFlash execution, width-policy threshold, acceptance priors, cost seeds, or adaptive scheduling to mask the issue.
- **FR-008**: Chad MUST remain read-only as both repository and product reference. Source inspection and diagnostic execution are allowed, but Chad files MUST NOT be modified. The exact physical Chad-built sidecar and named target MUST be used for any controlled comparison.
- **FR-009**: Local M4 Pro validation commands MAY be prepared and MUST document the controls. Throughput, latency, memory, parity, or other run-result claims MUST be limited to results actually measured by the maintainer on that hardware; commands alone are not evidence of results.
- **FR-010**: Relevant existing tests MUST pass after any fix.
- **FR-011**: No Git command that changes repository state MUST be executed during the investigation or validation. This includes add, commit, checkout, switch, reset, restore, stash, merge, rebase, cherry-pick, clean, tag creation, and branch or worktree mutation. Read-only Git inspection is allowed.

### Key Entities

- **Target state**: All state that can affect subsequent target evaluation, including attention KV, recurrent/GDN state, convolution/window state, cache offsets and absolute positions, accepted-prefix replay state, and hidden-state taps.
- **Speculative round**: A target verification of an anchor and proposed draft tokens followed by acceptance or rejection and state reconciliation.
- **Serial/replay reference**: An independently constructed target execution that evaluates only the anchor and accepted draft prefix.
- **Controlled comparison**: A comparison holding target and drafter checkpoints, tokenizer, prompt, sampling, widths, prefix-cache state, warmup assumptions, and relevant kernels constant.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Both required rollback cases satisfy exact structural/logical invariants and the Gate C independent same-S8 accepted-prefix reconstruction for rollback-owned state. Fresh S=1/S=2 references remain serial-semantic controls, and Gate D checks subsequent committed behavior; matching one next token alone is not sufficient.
- **SC-002**: Exact top-k identities and deterministic top-2 margins are required only for controlled comparisons shown repeatable on the same runtime at the applicable execution width. Raw logits are compared exactly when repeatable at that width; otherwise any bound is documented and justified per observation by applicable same-runtime controls. Cross-runtime Chad-versus-mlx B3 raw logits and margins are diagnostic and need not be exactly equal. Discrete token/ID/structural invariants remain exact where their semantic contract requires equality. No global tolerance is used.
- **SC-003**: The report identifies the earliest proven divergence, distinguishing an uncontrolled comparison, an earlier input-state difference, pre-rollback verify/tap, post-rollback/reconcile, or drafter state, and records why each disproved subsystem was not patched. The investigation-level outcome is explicitly one of **RESOLVED_FIX**, **RESOLVED_NO_CHANGE**, or **UNRESOLVED**. T016 is only the initial localization decision and may record exactly one of FIX_PENDING, RESOLVED_NO_CHANGE, or UNRESOLVED; it MUST NOT record RESOLVED_FIX. FIX_PENDING means the earliest controlled mlx-dspark defect has been localized and exactly one causal regression/patch branch T019–T022 is authorized. RESOLVED_NO_CHANGE requires all reachable required correctness gates needed for the decision to be conclusive and evidence that no mlx-dspark production defect requiring a patch exists. UNRESOLVED means required evidence remains inconclusive and no production patch is authorized. Only T024 may finalize RESOLVED_FIX, after successful applicable regression, production change, and validation work in T019–T024. A failure to establish the regression, failed fix validation, or newly inconclusive required evidence yields final UNRESOLVED, not RESOLVED_FIX. T030 carries the final outcome from T024. Any required gate or required continuous R1→R2→pre-R3 evidence that remains INCONCLUSIVE is UNRESOLVED, not successful no-change. For UNRESOLVED, no production patch is authorized; record the unresolved gate and missing evidence/control and do not claim SC-003/Definition-of-Done completion.
- **SC-004**: Before an applicable fix, a deterministic or repeatable causal regression/probe reproducibly exposes the earliest semantic divergence by asserting the identified causal invariant and reproducing the pre-fix violation under its exact controls; it is retained with provenance, rerun after an applicable fix, and satisfies the corrected invariant post-fix. A diagnostic snapshot without an asserted reproducible invariant is not a captured mismatch for this criterion. If rollback is disproved, the regression/diagnostic records the earlier divergence and rollback remains unpatched.
- **SC-005**: The validation instructions include reproducible M4 Pro commands with comparison controls. The report makes no throughput, latency, memory, parity, or other run-result claim unless the maintainer actually ran and recorded that measurement on the M4 Pro.

## Assumptions

- The target is `nathansutton/Qwen3.8-27B-Ternary-Bonsai-2-DFlash2-MLX` and the drafter is the exact physical Chad-built generic sidecar at `~/.cache/chad/dflash/015e795645c74b1a0eeef3b570031fb62e769bc5-q4g64`.
- Chad source and runtime may be inspected for behavioral diagnosis but remain read-only as the product solution.
- The supplied fixed-width and adaptive observations are investigation controls, not new benchmark acceptance thresholds.
- The implementation scope is limited to the earliest demonstrated semantic divergence; if rollback is equivalent to serial replay, investigation continues earlier in the target verify/tap path or into drafter state as evidence directs.
- Hardware performance validation is expected to be run by the maintainer on an M4 Pro after deterministic correctness work.
