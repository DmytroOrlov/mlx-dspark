# Feature Specification: K2 Custom MLX-LM Routing

**Feature Branch**: `007-k2-custom-mlx-lm-routing`
**Created**: 2026-10-07
**Status**: Draft
**Input**: User description: Repair loading/routing for text-only mlx-lm checkpoints that declare a custom architecture with `config.json:model_file`, preserve multimodal routing and explicit remote-code trust, and keep changes limited to model loading/routing.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Load a trusted custom text checkpoint (Priority: P1)

As an operator serving a text-only checkpoint with an architecture supplied by the checkpoint, I can explicitly enable remote code and have the checkpoint use the text-model loader that supports its custom architecture.

**Why this priority**: This is the reported failure and the feature's primary user outcome.

**Independent Test**: Use a local fixture with an unknown `model_type` and `model_file`; assert that explicit trust selects the text loader and that the loader receives the trust option when its API supports it.

**Acceptance Scenarios**:

1. **Given** a text-only checkpoint with `model_file` and an unsupported built-in `model_type`, **When** the operator enables remote code and loads it, **Then** the checkpoint is routed through mlx-lm and its custom architecture loads.
2. **Given** a version of mlx-lm whose loader accepts a model trust argument, **When** the operator enables remote code, **Then** the model loader receives `trust_remote_code=True` directly.
3. **Given** the installed mlx-lm API has no model trust argument, **When** the operator enables remote code, **Then** the compatible loader call succeeds without an unsupported keyword and mlx-dspark's explicit trust gate remains in force.

---

### User Story 2 - Refuse custom code without opt-in (Priority: P1)

As an operator, I can load ordinary checkpoints without implicitly allowing checkpoint-supplied Python to execute.

**Why this priority**: The routing repair must retain the existing security boundary while enabling the requested use case.

**Independent Test**: A local fixture with `model_file` and trust disabled is rejected before either loader or custom module import is invoked.

**Acceptance Scenarios**:

1. **Given** a checkpoint declares `model_file` and remote-code trust is disabled, **When** loading is attempted, **Then** loading is refused before model or tokenizer code from the checkpoint runs.
2. **Given** a checkpoint has no remote-code markers, **When** it is loaded with trust disabled, **Then** ordinary loading remains available.

---

### User Story 3 - Keep multimodal and existing model routing (Priority: P1)

As an operator, I can continue loading multimodal checkpoints with the multimodal loader and built-in text or VLM checkpoints through their established paths.

**Why this priority**: Routing a multimodal architecture as a text model can break existing model support; the fix must be narrow.

**Independent Test**: Exercise routing fixtures with and without modality markers, including a multimodal fixture that also declares `model_file`, and compare established built-in routes.

**Acceptance Scenarios**:

1. **Given** a checkpoint has explicit vision or audio configuration, including one that also has `model_file`, and is not covered by an existing text-capable family exception, **When** it is routed, **Then** it remains on mlx-vlm's multimodal path.
2. **Given** a recognized built-in mlx-lm text checkpoint, **When** it is routed, **Then** it remains on mlx-lm.
3. **Given** a recognized multimodal checkpoint such as Gemma 4, **When** it is routed, **Then** it remains on mlx-vlm, subject to existing text-capable multimodal exceptions.

## Assessment Findings

- Repository code in `src/mlx_dspark/load.py` routes a config with an unknown `model_type` to mlx-vlm unless that type resolves to an installed mlx-lm module. It currently does not use `model_file` to recognize custom text architectures. The existing unknown-family test codifies this fallback.
- The repository lockfile and local environment identify mlx-lm 0.31.3 and mlx-vlm 0.7.2. In mlx-lm 0.31.3, `mlx_lm.load` accepts `tokenizer_config` but has no `trust_remote_code` parameter; its `load_model` imports `model_file` unconditionally when called. mlx-dspark's `refuse_remote_code` check is therefore the effective model-code security gate in this environment, while `tokenizer_config` governs tokenizer trust. Current upstream mlx-lm source has added a direct trust parameter to `load`/`load_model`; loader forwarding must account for the API actually installed.
- mlx-vlm 0.7.2 also imports `config.json:model_file` in `get_model_and_args`, then expects the imported module to implement its VLM `ModelConfig` contract. Therefore `model_file` alone is not proof that a checkpoint is text-only. Explicit modality configuration must retain priority.
- The Hugging Face model card for `mlx-community/K2-Horizon-3.7B-4bit` describes it as a decoder-only text model, states that its repository supplies `k2_horizon.py` referenced by `model_file`, and documents use through mlx-lm. This corroborates the reproduction's checkpoint shape.
- The reported routing and fallback are consistent with repository source: an unknown `k2_horizon` does not resolve to an installed mlx-lm module and therefore falls through to mlx-vlm, whose custom model contract can produce the observed missing-`ModelConfig` failure.
- The focused existing routing tests could not execute in this session because importing MLX requires a Metal device unavailable in the sandbox. Conclusions above come from repository tests/source, locked package versions, and installed package source inspection; a live K2 checkpoint load was not performed.

**Smallest safe remediation direction**: Adjust target routing narrowly so an unknown-model `model_file` config without explicit multimodal markers uses mlx-lm, while existing explicit multimodal markers and current text-capable multimodal exceptions keep their current behavior. Retain the pre-load remote-code refusal. Pass `trust_remote_code` to mlx-lm's model loader only when that installed API supports it; preserve tokenizer trust configuration, and avoid silently dropping the security gate on older APIs. Add offline loader-routing, trust, refusal, and multimodal precedence regressions. A network-dependent K2 integration test is optional; a local fixture representing its config shape is the portable baseline.

## Edge Cases

- A checkpoint contains both `model_file` and explicit vision or audio configuration; modality routing takes priority.
- A checkpoint has a custom `model_file` but remote-code trust is disabled; refusal occurs before the custom file is imported.
- The installed mlx-lm API accepts no direct `trust_remote_code` parameter; compatible loading must avoid an invalid keyword while retaining the local trust gate.
- A text-capable multimodal family already handled by the project's explicit exception continues to follow that exception.
- A malformed or missing model file fails clearly through the selected loader and does not trigger a fallback that executes it through another framework.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The loader MUST route a text-only checkpoint that declares `model_file` and has no explicit multimodal markers to mlx-lm, even when its `model_type` is not a built-in mlx-lm family.
- **FR-002**: The loader MUST preserve the established multimodal route for checkpoints with vision or audio configuration, including checkpoints that also declare `model_file`, except for existing documented text-capable multimodal family exceptions.
- **FR-003**: The loader MUST preserve established routing for built-in mlx-lm text models, mlx-vlm multimodal models, remapped model families, and documented text-capable multimodal exceptions.
- **FR-004**: Checkpoint-supplied model or tokenizer Python MUST NOT execute unless the operator explicitly enabled remote-code trust.
- **FR-005**: When explicit remote-code trust is enabled, the loader MUST provide that trust setting directly to the mlx-lm model-loading API whenever the installed API exposes such an option.
- **FR-006**: When the installed mlx-lm API does not expose a model trust option, loading MUST remain compatible without passing an unsupported argument, and the existing pre-load trust check MUST remain effective.
- **FR-007**: The loader MUST preserve the operator's trust setting for tokenizer loading.
- **FR-008**: Regression coverage MUST prove text custom-model routing, explicit-trust forwarding where supported, refusal before custom-code execution when disabled, multimodal precedence with `model_file`, and unchanged built-in routing.
- **FR-009**: The repair MUST remain limited to target model routing and model loading; it MUST NOT alter speculative decoding, lookup, battery pause/resume, HF caching, or server lifecycle behavior.

### Key Entities *(include if data involved)*

- **Checkpoint metadata**: Model type, custom model file declaration, and modality configuration used to choose a loader.
- **Remote-code trust setting**: The operator's explicit process-level permission for checkpoint-supplied Python execution.
- **Loader route**: The selected text-model or multimodal loading path.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All offline routing regressions pass for custom text, built-in text, multimodal, and multimodal-with-`model_file` fixtures.
- **SC-002**: In every tested trust-disabled custom-code case, zero custom module imports occur and loading is refused before a framework loader is called.
- **SC-003**: In every tested trust-enabled custom text case, the model loader receives `trust_remote_code=True` when the installed API supports it, and the model is not sent through mlx-vlm.
- **SC-004**: A local K2-Horizon-shaped config fixture selects mlx-lm without requiring a network download; a real-checkpoint integration check may supplement, but not replace, this test.
- **SC-005**: No files or behavior outside model loading and routing are changed as part of implementation.

## Assumptions

- Remote-code trust remains a process-level opt-in controlled by the existing CLI/environment setting.
- Configurations with explicit vision or audio markers are treated as multimodal unless an existing documented family exception overrides that route.
- Offline fixtures are sufficient for required regression coverage; a live model integration test is optional because it downloads a checkpoint and needs Apple Silicon/Metal.
- Assessment is complete without production-source changes; the findings describe the observed baseline and the smallest safe repair direction.
