# Implementation Plan: K2 Custom MLX-LM Routing

**Branch**: `007-k2-custom-mlx-lm-routing` | **Date**: 2026-10-07 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification at `specs/007-k2-custom-mlx-lm-routing/spec.md`

## Summary

Route custom text checkpoints identified by top-level `config.json:model_file` through mlx-lm when they do not have explicit vision/audio markers, even if their model type is not in the installed mlx-lm registry. Preserve the earlier text-capable multimodal exception, explicit multimodal routing, built-in/remapped routing, and the `refuse_remote_code` security gate. Forward `trust_remote_code` to mlx-lm only when its installed loader signature supports that keyword; always preserve tokenizer trust configuration. Add local fixture and mocked-loader regressions in the existing routing and remote-code tests. No dependency change or K2-specific branch is needed.

## Technical Context

**Language/Version**: Python >=3.10 (repository runtime currently Python 3.12)

**Primary Dependencies**: mlx-lm (locked 0.31.3; declared floor >=0.31.3), mlx-vlm (locked 0.7.2; declared floor >=0.6.12), pytest

**Storage**: Local/Hugging Face checkpoint files; no new persistence

**Testing**: Existing pytest tests with temporary JSON checkpoint fixtures, monkeypatching, and mocked loaders; no model download or tensor execution in new regressions

**Target Platform**: Existing mlx-dspark CLI/server environments; automated regression logic must not load a real checkpoint or require model weights, Hugging Face access, or Metal operations

**Project Type**: Python library and CLI/server

**Performance Goals**: No runtime performance change; retain existing loader behavior for non-custom-text targets

**Constraints**: Do not change dependencies, generation, cache/download resolution, server lifecycle, or other runtime subsystems. The locked mlx-lm loader has no direct model `trust_remote_code` argument; newer compatible signatures expose it.

**Scale/Scope**: One routing predicate and the mlx-lm loader invocation in `src/mlx_dspark/load.py`; focused regressions in `tests/test_formats.py` and `tests/test_remote_code.py`; documentation artifacts for design and validation

## Current Behavior and Routing Contract

`load_target` resolves the checkpoint and reads `config.json`, performs the existing unsupported-quantization check and Nanbeige registration, then calls `refuse_remote_code(path, repo_or_path)` before importing either target loader or invoking checkpoint-supplied model/tokenizer code. After this gate, the Prism special loader remains separate. Other targets call `_route_target` and then exactly one loader. An mlx-vlm load failure is wrapped with the existing error message; there is no retry through the other framework.

Current `_route_target` order in `src/mlx_dspark/load.py` is:

1. If `model_type` is `qwen3_5` or `qwen3_5_moe` and its mlx-lm module is installed, choose mlx-lm. This check intentionally precedes modality markers because these text-capable multimodal families use mlx-lm's text module and require its target tap.
2. If `vision_config` or `audio_config` is present, choose mlx-vlm.
3. Apply mlx-lm's `MODEL_REMAPPING` table, then choose mlx-lm if the resulting module is installed.
4. Otherwise choose mlx-vlm.

A custom `model_file` is not considered today, so an unknown text model such as `k2_horizon` reaches step 4 and is sent to mlx-vlm. mlx-vlm also imports `model_file` and expects its own `ModelConfig` interface, which explains the observed fallback exception.

The repaired order will be:

1. Preserve the installed `qwen3_5`/`qwen3_5_moe` text-capable multimodal exception first.
2. Preserve explicit vision/audio routing to mlx-vlm second.
3. If top-level `model_file` is present and neither prior rule selected a route, choose mlx-lm. This covers custom text models without requiring a built-in model type and avoids a K2-specific rule.
4. Preserve the existing model-type remapping and installed-module lookup for built-in/remapped mlx-lm models.
5. Preserve the final mlx-vlm fallback for unrecognized checkpoints without `model_file`.

This retains the existing exception and all modality-marker precedence while closing only the unknown custom-text gap. The check applies to the config field already consumed by mlx-lm (`config.json:model_file`); nested/custom multimodal interpretation is outside this change.

## Remote-Code Trust Compatibility

`load_target` performs `refuse_remote_code` after reading the metadata needed for routing and before importing/calling `mlx_lm.load` or `mlx_vlm.load`. The refusal scans checkpoint metadata but never imports checkpoint files. Keep this gate before the new custom-text route can reach any loader. With trust disabled, the function must raise before loader spies are called and before a custom module sentinel can execute.

In the locked mlx-lm 0.31.3 API, `mlx_lm.load(path, tokenizer_config=...)` has no direct model trust argument, while `mlx_lm.load_model` imports `model_file` without its own trust check. On newer mlx-lm APIs, `load` and `load_model` expose `trust_remote_code`. The implementation will inspect the installed `lm_load` callable's signature for an explicit `trust_remote_code` parameter. If present, pass `TRUST_REMOTE_CODE` directly. If absent, omit that keyword. If signature inspection cannot be performed, use the old-compatible call shape and rely on the existing pre-load refusal. In both cases, continue passing `tokenizer_config={"trust_remote_code": TRUST_REMOTE_CODE}`. Do not use package-version checks or raise a new dependency floor.

## Production and Test Files

Expected production change:

- `src/mlx_dspark/load.py`: update `_route_target` with the model-file rule at the specified position; add the minimum signature-capability check/conditional keyword forwarding around the mlx-lm load call. Keep the pre-load refusal and current error handling/order intact.

Expected test changes:

- `tests/test_formats.py`: extend existing routing tests for a K2-shaped unknown-type config with `model_file`, a multimodal config with `model_file`, and unchanged known, remapped, unknown-no-file, and text-capable multimodal cases. Reuse `_route_target` and the current config-dictionary style.
- `tests/test_remote_code.py`: extend the current temporary-config and monkeypatch style with `load_target` mocked-loader cases covering trust refusal, no import/loader call, tokenizer trust, and both old/new loader signatures. Use a local `model_file` fixture with an import side-effect sentinel; no Hub access or weight files are needed. Keep the test doubles narrow and do not introduce a second testing framework.

The added tests must not download models or execute model tensors. Existing project test collection imports MLX through its current setup; do not expand that dependency or require Metal for the new test logic. A focused pytest run and the full relevant suite should be performed in the normal supported test environment.

## Constitution Check

**Pre-design gate**:

- **Preserve the Vanilla Qwen Path**: Pass. The route change is based on `model_file` for otherwise-unmatched text-only configs. It retains the current Qwen/Qwen3.5 exception, built-in route and does not change generation or hybrid composition.
- **Runtime Integrity Is the Correctness Gate**: Pass. Loading success, selected loader, and trust refusal are directly asserted through local fixtures and mocks. No performance claim is introduced.
- **One Experiment Axis at a Time**: Pass. Only model loading/routing and the direct trust-argument compatibility branch are in scope.
- Other constitution principles concern the Bonsai hybrid performance experiment and do not apply to this loader repair.

**Post-design gate**: Pass. The design changes one route predicate and one loader call's optional keyword handling. It adds no dependencies, persistent entities, public CLI options, or side effects outside existing checkpoint loading. Manual model validation is optional and does not become a CI gate.

## Design Decisions

- **Prefer `model_file` only after explicit multimodal checks**: mlx-vlm also supports `model_file`; the field is therefore a custom architecture signal, not a modality classifier. Explicit modality markers retain precedence, except for the existing earlier Qwen3.5 text-module exception.
- **Use signature capability detection**: Detect only an explicit `trust_remote_code` parameter on `lm_load`; this supports the old and new API shapes without package-version comparisons or an unrelated dependency upgrade. Signature-inspection failure uses the older-compatible call.
- **Do not add fallback behavior**: Route once from metadata. A load failure from mlx-lm remains an mlx-lm failure and cannot execute the same custom file through mlx-vlm.
- **Keep the test seam local and mocked**: Routing is tested directly; trust/load behavior uses a local config and patched loader modules. The live K2 model belongs only in manual validation.

## Project Structure

### Documentation (this feature)

```text
specs/007-k2-custom-mlx-lm-routing/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/       # Not created: no external API or CLI contract changes
└── tasks.md         # Created later by $speckit-tasks
```

### Source Code (repository root)

```text
src/mlx_dspark/load.py
tests/test_formats.py
tests/test_remote_code.py
```

**Structure Decision**: Use the existing single Python package and pytest suite. No new package, test directory, CLI surface, or contract is required.

## Complexity Tracking

No constitution violations or added architectural complexity require justification.
