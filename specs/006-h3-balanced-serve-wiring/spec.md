# Feature Specification: Donor Block Serve Wiring

**Feature Branch**: `006-h3-balanced-serve-wiring`

**Created**: 2026-09-26

**Status**: Draft

**Input**: User description: "Update the existing feature to expose hybrid target blocks through paired `--donor-model` and `--donor-blocks` serve options, with H3 (Bonsai2 blocks 56–63) as the human test case and no-donor serve behavior unchanged."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Serve with Donor Blocks (Priority: P1)

An operator starts the ordinary server and names a donor checkpoint and the decoder block indices to take from it. The target remains Qwen, with its embedding, every unlisted decoder block, final norm, and LM head. For the concrete H3 human-test case, the donor is `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` and the donor blocks are `56-63`.

The primary human command is:

```sh
MLX_DSPARK_SLOW_ROUND_LOG_S=1 mlx-dspark serve \
  --api-key 'jA-s#q' \
  --model mlx-community/Qwen3.8-27B-4bit \
  --mode dflash \
  --drafter incoai/Qwen3.8-27B-DFlash2 \
  --donor-model prism-ml/Ternary-Bonsai-2-27B-mlx-2bit \
  --donor-blocks 56-63 \
  --host 0.0.0.0 \
  --port 13885 \
  --context-window 262144 \
  --reasoning-effort xhigh
```

**Why this priority**: This is the requested public serving interface for trying the H3 target through the existing serve workflow.

**Independent Test**: Start serve with the paired options, confirm startup reports the donor and normalized indices plus Qwen-owned components, then complete a normal OpenAI-compatible request when the required model artifacts and hardware are available.

**Acceptance Scenarios**:

1. **Given** a supported Qwen target and donor, **When** the operator supplies `--donor-model prism-ml/Ternary-Bonsai-2-27B-mlx-2bit --donor-blocks 56-63`, **Then** those same-index decoder blocks are taken from the donor and the remaining target components retain Qwen ownership.
2. **Given** the paired options, **When** startup information is displayed, **Then** it reports the donor identifier, resolved donor revision when readily available, normalized donor block indices/ranges, and Qwen ownership of embedding, final norm, and LM head.
3. **Given** the operator selects a DFlash drafter, **When** donor blocks are selected, **Then** drafter selection remains independent of target composition.

### User Story 2 - Preserve Ordinary Serve (Priority: P1)

An operator invokes serve without either donor option and continues to use the existing target-loading behavior unchanged.

**Why this priority**: The hybrid selection is opt-in; existing users must not get a composition request or change behavior by omission.

**Independent Test**: Invoke the existing no-donor serve command and verify that no target composition is requested and the current model-loading path is retained.

**Acceptance Scenarios**:

1. **Given** neither donor option is present, **When** serve resolves its target, **Then** ordinary behavior proceeds without requesting target composition.
2. **Given** an existing H0 serve command, **When** it is run without modification, **Then** it remains valid and follows the no-donor path.

### User Story 3 - Validate Donor Selection Before Loading (Priority: P1)

An operator gets a concise error for incomplete or malformed donor selection before any model is loaded.

**Why this priority**: Early validation prevents expensive checkpoint loading for invalid input.

**Independent Test**: Exercise the options with only one member of the pair, malformed syntax, out-of-range indices, and descending ranges; each must fail before model loading begins.

**Acceptance Scenarios**:

1. **Given** exactly one donor option is supplied, **When** serve validates its arguments, **Then** it reports that `--donor-model` and `--donor-blocks` must be supplied together and exits before model loading.
2. **Given** donor blocks `62` or `56-63`, **When** serve validates the selection, **Then** it accepts the single index and inclusive range respectively.
3. **Given** comma-separated indices/ranges such as `12,28,41,55,62` or `12,20-23,62`, **When** serve validates the selection, **Then** it accepts them if supported by the small block-list syntax and normalizes them to the selected indices.
4. **Given** a non-integer, an index outside 0–63, a descending range, an empty element, or otherwise malformed block syntax, **When** serve validates arguments, **Then** it fails concisely before model loading. Duplicate indices either normalize to one index or fail with a clear duplicate-specific error.

### Edge Cases

- Neither donor option is present: do not request a target composition.
- Exactly one donor option is present: fail with a concise paired-option error before loading.
- Block index 0 or 63: accept as valid boundaries; 64 and negative indices are invalid.
- Range endpoints are equal: treat as a single block or report a clear syntax error; ranges must never silently reverse.
- Duplicate selections: normalize or reject clearly; they must not produce ambiguous ownership.
- Donor model uses a repository/path form handled by existing model and snapshot resolution: resolve it through those existing mechanisms and report the revision when readily available.
- Model resolution fails: preserve the existing model-resolution error behavior.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The ordinary serve command MUST expose `--donor-model` and `--donor-blocks` as the public interface for opt-in donor block selection; it MUST NOT require or present `--target-composition h3-balanced` as the requested interface.
- **FR-002**: The options MUST be a pair. Supplying exactly one MUST fail with a concise error before model loading.
- **FR-003**: `--donor-model` MUST identify the donor checkpoint, repository, or path using the existing model and snapshot resolution behavior.
- **FR-004**: `--donor-blocks` MUST identify same-index decoder blocks replaced by donor blocks. It MUST support at minimum one block (`62`) and one inclusive range (`56-63`). A small comma-separated list of indices and ranges SHOULD also be accepted, without introducing a general composition language.
- **FR-005**: Block selection MUST accept only integer indices in the inclusive range 0–63, require ranges to be ascending, and normalize duplicates or reject them with a clear error. Malformed syntax MUST fail before model loading.
- **FR-006**: The paired options MUST expand into the existing `TargetCompositionRequest` and use the existing hybrid loader; the feature MUST NOT add another hybrid loader or composition payload.
- **FR-007**: A donor composition MUST retain Qwen ownership of the embedding, all unlisted decoder blocks, final norm, and LM head.
- **FR-008**: Drafter selection MUST remain independent of donor target selection.
- **FR-009**: Startup MUST report the donor model, resolved donor revision when readily available, normalized donor block indices/ranges, and Qwen ownership of embedding, final norm, and LM head.
- **FR-010**: When neither donor option is supplied, serve MUST retain its current behavior and MUST NOT request target composition.
- **FR-011**: The concrete H3 human-test case MUST use donor `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` with donor blocks `56-63` through the ordinary serve command.
- **FR-012**: This feature MUST NOT re-prove Features 002, 003, or 005; benchmark H3; evaluate quality; add an evidence framework; or alter speculative policy, CapController, KV/cache behavior, tokenizer behavior, reasoning effort, or generation defaults.

### Key Entities *(include if data involved)*

- **Donor selection**: The paired donor model identifier and decoder block selection supplied for one serve invocation.
- **Target composition request**: The existing request describing Qwen ownership and selected donor block ownership for the target.
- **Normalized donor block selection**: The validated set of selected indices, represented clearly as individual indices and/or ranges for startup reporting.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: An operator can request the concrete H3 composition in the ordinary serve command using `--donor-model ... --donor-blocks 56-63`, without a preset selector.
- **SC-002**: The concrete selection resolves to exactly decoder indices 56 through 63 from the named donor, while Qwen retains embedding, all other decoder blocks, final norm, and LM head.
- **SC-003**: With neither donor option supplied, a model-free configuration check confirms no composition request and unchanged ordinary serve selection.
- **SC-004**: Every one-sided or malformed donor selection in the defined validation cases is rejected before model loading with a concise error.
- **SC-005**: Startup output identifies the donor, available resolved revision, normalized selected indices/ranges, and Qwen-owned embedding, final norm, and LM head; an operator can complete a normal request when hardware and model availability permit.

## Assumptions

- Existing model/snapshot resolution and the existing `TargetCompositionRequest`/hybrid loader are available for reuse.
- The target for the concrete H3 case is `mlx-community/Qwen3.8-27B-4bit`; the drafter remains independently selected as `incoai/Qwen3.8-27B-DFlash2` in the documented human command.
- A donor revision is reported only when existing resolution makes it readily available; revision lookup does not add a new resolution mechanism.
- Hardware-backed startup and request completion depend on access to the target and donor artifacts and suitable hardware.
- This is serve-interface wiring only; composition validity, performance, and model quality are not reevaluated here.
