# Tasks: K2 Custom MLX-LM Routing

**Input**: Design documents from `specs/007-k2-custom-mlx-lm-routing/`

**Prerequisites**: `spec.md`, `plan.md`, `research.md`, `data-model.md`, and `quickstart.md`

**Scope**: `src/mlx_dspark/load.py`, `tests/test_formats.py`, `tests/test_remote_code.py`, and this feature's documentation/task artifacts only. No dependency, generation, decoding, lookup, cache, battery, server lifecycle, prefix-cache, or memory-management changes.

**Tests**: Required by the user. Add local-fixture tests before the corresponding production changes; no Hugging Face downloads, real model weights, or inference in automated tests.

## Format

Every task uses `- [ ] T### [P?] [US#] Description with exact file path(s)`. `[P]` is used only where tasks edit distinct files and have no dependency.

## Phase 1: Setup

**Purpose**: No project setup is required; use the existing Python package and pytest suite.

## Phase 2: Foundational

**Purpose**: No shared production prerequisite is needed. The existing config fixtures, route function, remote-code refusal gate, and monkeypatch style are the test seams.

## Phase 3: User Story 1 - Load a trusted custom text checkpoint (Priority: P1) 🎯 MVP

**Goal**: Establish the K2-shaped custom-text routing regression, protect existing route decisions, then implement the custom-text route.

**Independent Test**: A config with unknown `model_type`, top-level `model_file`, and no modality markers selects mlx-lm. Existing protected route cases retain their expected loader.

### Tests for User Story 1

> Add the regression tests first and confirm the K2-shaped case fails against the current fallback before changing production code.

- [X] T001 [P] [US1] Add a K2-shaped route test for unknown `model_type` plus top-level `model_file` and no modality markers, asserting `_route_target` selects mlx-lm, in `tests/test_formats.py`.
- [X] T002 [US1] Add or extend routing tests in `tests/test_formats.py` to assert `model_file` plus `vision_config` selects mlx-vlm, `model_file` plus `audio_config` selects mlx-vlm, both installed `qwen3_5` and `qwen3_5_moe` text-capable multimodal exceptions retain mlx-lm, built-in and `MODEL_REMAPPING` text routes remain mlx-lm, and unknown configs without `model_file` retain the mlx-vlm fallback.
- [X] T003 [US1] Run the focused routing cases in `tests/test_formats.py` before the production change; confirm the K2-shaped case demonstrates the current mlx-vlm route while the protected existing routes pass.

### Implementation for User Story 1

- [X] T004 [US1] Update only `_route_target` in `src/mlx_dspark/load.py` to preserve the installed Qwen3.5/Qwen3.5-MoE exception first, explicit vision/audio routing second, route otherwise-unmatched top-level `model_file` configs to mlx-lm third, then preserve remapping/built-in detection and the mlx-vlm fallback; do not add a `k2_horizon` special case or error-time retry.

**Checkpoint**: The K2-shaped fixture selects mlx-lm and the routing preservation tests pass without changing explicit multimodal or existing built-in behavior.

## Phase 4: User Story 2 - Refuse custom code without opt-in (Priority: P1)

**Goal**: Prove refusal ordering and both mlx-lm trust API shapes, then conditionally forward direct model trust while preserving tokenizer trust.

**Independent Test**: With trust disabled, a custom checkpoint is refused before any target framework loader call or custom-module side effect. With trust enabled, loader arguments match the callable's supported signature.

### Tests for User Story 2

> Add these tests before changing the mlx-lm invocation. Local temporary configs and mocked loaders must not import a real checkpoint module or download a model.

- [X] T005 [P] [US2] Add `load_target` refusal regressions in `tests/test_remote_code.py` using temporary checkpoints with (a) custom `config.json:model_file` and (b) custom `tokenizer_config.json:auto_map`, each paired with a local import-side-effect sentinel module; with `TRUST_REMOTE_CODE=False`, assert `load_target` refuses before either mlx-lm or mlx-vlm loader is called and neither sentinel executes.
- [X] T006 [US2] Add mocked-loader compatibility cases in `tests/test_remote_code.py` proving tokenizer configuration retains the trust value; the old mlx-lm signature receives no `trust_remote_code` keyword; a signature explicitly exposing `trust_remote_code` receives `True` when enabled; and signature-introspection failure uses the old-compatible arguments while the pre-load refusal remains effective.
- [X] T007 [US2] Run the focused remote-code cases in `tests/test_remote_code.py` before the production compatibility change; confirm trust-disabled refusal and legacy-call behavior, and confirm the new-signature forwarding expectation exposes the missing compatibility behavior.

### Implementation for User Story 2

- [X] T008 [US2] Update only the mlx-lm call path in `src/mlx_dspark/load.py` to inspect `lm_load` for an explicit `trust_remote_code` parameter, pass `TRUST_REMOTE_CODE` only when supported, preserve `tokenizer_config={"trust_remote_code": TRUST_REMOTE_CODE}`, and retain `refuse_remote_code` before any loader import/call; do not add package-version checks or a dependency floor change.

**Checkpoint**: Disabled trust still refuses before custom code or framework loader execution; old and new mlx-lm signatures both receive compatible arguments.

## Phase 5: User Story 3 - Keep multimodal and existing model routing (Priority: P1)

**Goal**: Verify the complete offline route and trust regression set against the integrated production change, independently of live model availability.

**Independent Test**: The relevant routing and remote-code tests pass using local config fixtures and mocked loaders only.

- [X] T009 [US3] Run `uv run pytest -q tests/test_formats.py tests/test_remote_code.py` and record the result; these two existing files are the narrowest relevant combined routing/loading suite, and the run must cover the newly added route, refusal, tokenizer, and old/new signature regressions without network or model weights.

**Checkpoint**: All offline routing and trust regressions pass; live K2 loading is not part of this checkpoint.

## Phase 6: Polish & Cross-Cutting Verification

**Purpose**: Optional real-checkpoint validation followed by a final scope audit.

- [X] T010 On Apple Silicon only, follow the manual validation in `specs/007-k2-custom-mlx-lm-routing/quickstart.md`: run `mlx-dspark serve --model mlx-community/K2-Horizon-3.7B-4bit --trust-remote-code --mode baseline --host 127.0.0.1 --port 13886`, verify mlx-lm custom-model loading with no former mlx-vlm `custom_model.ModelConfig` failure, and complete one real generation; mark this task not run when Apple Silicon/Metal is unavailable, without blocking implementation completion.
- [X] T011 Audit the final diff in `src/mlx_dspark/load.py`, `tests/test_formats.py`, `tests/test_remote_code.py`, and `specs/007-k2-custom-mlx-lm-routing/tasks.md`; confirm no other production/test surfaces, dependencies, or excluded runtime behaviors changed, and report offline regression status separately from real K2 validation status.

## Dependencies & Execution Order

### Phase Dependencies

- Setup and foundational phases require no actions; the repository already has the needed test and loader seams.
- User Story 1 regression tests (T001-T003) must precede its routing production change (T004).
- User Story 2 trust tests (T005-T007) must precede the loader compatibility change (T008).
- The final offline suite (T009) depends on T004 and T008.
- Manual K2 validation (T010) follows offline tests and is optional; it does not block implementation completion.
- The scope audit (T011) is last and reports offline and manual outcomes separately.

### User Story Dependencies

- **User Story 1 (P1)**: T001-T004 implement custom text routing. Route-preservation cases are included because they guard the same routing change.
- **User Story 2 (P1)**: T005-T008 depend on no new route API; execute after the routing test-first sequence to keep the review path linear and isolate trust-call compatibility.
- **User Story 3 (P1)**: T009 verifies the integrated routing preservation and remote-code behavior. It depends on both production changes.

### Parallel Opportunities

- T001 and T005 edit different test files and are independent; they may be prepared in parallel.
- T002 follows T001 because it extends the same `tests/test_formats.py` file. T006 follows T005 because it extends the same `tests/test_remote_code.py` file.
- Production edits T004 and T008 both touch `src/mlx_dspark/load.py` and must remain sequential.
- Final tests, manual validation, and scope audit follow their stated dependencies and are not parallel tasks.

## Parallel Example

```text
T001: K2-shaped routing regression in tests/test_formats.py
T005: Trust-disabled refusal and import-side-effect regression in tests/test_remote_code.py
```

These two test tasks can proceed concurrently because they edit separate files and neither depends on the other. All subsequent tasks follow the dependency order above.

## Implementation Strategy

1. Establish route regressions and show the K2-shaped case fails against the current route.
2. Establish security and API-signature regressions before changing the mlx-lm call.
3. Make the routing change, then add the minimal trust-signature forwarding compatibility in the same production file as a separate review step.
4. Run the focused offline suite. Keep real K2 serve-and-generate validation optional and clearly report whether it ran.
5. Finish with the allowed-surface diff audit. The MVP is User Story 1 (T001-T004); full completion also requires User Story 2 and the offline regression task.
