# Research: K2 Custom MLX-LM Routing

## Decision 1: Classify custom text by `model_file` after existing multimodal routing

**Decision**: Preserve the `qwen3_5`/`qwen3_5_moe` installed text-module exception first, preserve `vision_config`/`audio_config` routing second, then route unmatched configs with top-level `model_file` to mlx-lm. Keep remapping/module detection and unknown-no-file mlx-vlm fallback afterward.

**Rationale**: This closes the `k2_horizon` hole without a model-specific branch. mlx-vlm 0.7.2 also imports `model_file`, so the field alone cannot override explicit modality markers. The current Qwen3.5 exception must stay ahead of markers to preserve the existing hidden-state tap path.

**Alternatives considered**:

- Route every `model_file` config to mlx-lm first: rejected because mlx-vlm can legitimately use the field and explicit multimodal configs would change routes.
- Add a special `k2_horizon` case: rejected because the general custom text shape is the requested support contract and should not need per-model maintenance.
- Retry through the other framework after a loader error: rejected because it can load checkpoint code under a different interface and masks the initial classification error.

## Decision 2: Keep trust refusal before all target loader calls and use signature capability detection

**Decision**: Retain `refuse_remote_code` after config inspection and before importing/calling a target framework loader. Inspect the `mlx_lm.load` signature for an explicit `trust_remote_code` parameter. Forward the process trust value only when that parameter exists; otherwise call with the currently supported arguments. Continue forwarding the trust value through `tokenizer_config`.

**Rationale**: The locked mlx-lm 0.31.3 `load` signature has no direct trust option; its model loader imports `model_file` once called. Newer upstream mlx-lm exposes direct model trust on `load`/`load_model`. The local refusal gate is necessary on the locked API and remains the earliest model-code barrier on either API. Signature detection avoids a brittle version comparison.

**Alternatives considered**:

- Always pass `trust_remote_code`: rejected because it raises `TypeError` on 0.31.3.
- Upgrade the dependency floor: rejected because the old API is safely supported by the existing gate and upgrade is explicitly out of scope.
- Depend only on `tokenizer_config`: rejected because this config governs tokenizer loading and does not gate model-file import in newer mlx-lm APIs.
- Trust only the newer loader's gate: rejected because the supported locked version has no model trust option.

## Decision 3: Extend existing test files with offline fixtures

**Decision**: Extend `tests/test_formats.py` routing tests and `tests/test_remote_code.py` refusal tests, using temporary JSON configs and monkeypatched framework loader callables. Use a local K2-shaped metadata fixture and a Python side-effect sentinel for import refusal.

**Rationale**: These files already cover routing and remote-code marker behavior. Mocks can prove loader selection, keyword forwarding, and zero downstream calls without Hugging Face access, weights, or model execution.

**Alternatives considered**:

- Add a real K2 checkpoint to required tests: rejected because it makes routine tests network-, storage-, and hardware-dependent.
- Build a separate integration framework: rejected because the existing pytest and monkeypatch seams suffice.

## Decision 4: Keep real K2 validation optional and manual

**Decision**: Include the requested Apple Silicon serve-and-generate command in `quickstart.md` as optional manual validation, not as a CI requirement.

**Rationale**: It verifies the actual checkpoint payload and generation behavior while preserving deterministic, offline routine regression coverage.

**Alternatives considered**:

- Make live K2 loading a required test: rejected because it downloads remote weights and requires a supported Apple Silicon/Metal host.
