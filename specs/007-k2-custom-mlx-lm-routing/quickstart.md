# Quickstart: Validate K2 Custom MLX-LM Routing

## Automated offline regressions

Run the routing and remote-code tests in the repository's supported pytest environment:

```sh
uv run pytest -q tests/test_formats.py tests/test_remote_code.py
```

The required new cases use local metadata fixtures and mocked loaders. They must not download from Hugging Face, read model weights, or perform model inference. Confirm coverage for:

- K2-shaped config (`model_type` unknown, `model_file` present, no modality markers) selects mlx-lm.
- Trust disabled raises before a loader call or custom-module side effect.
- A config with `model_file` and vision/audio markers selects mlx-vlm, subject to the preserved earlier Qwen3.5 text-capable exception.
- Built-in and remapped mlx-lm routing and existing mlx-vlm routing remain unchanged.
- Old mlx-lm load signature receives no unsupported model trust keyword; new signature receives `trust_remote_code=True` when enabled.
- Tokenizer `trust_remote_code` remains the configured value for either signature.

## Optional manual K2-Horizon validation (Apple Silicon)

This check downloads and loads the real checkpoint and is not required in CI. On a supported Apple Silicon host with mlx-dspark installed, start the server:

```sh
mlx-dspark serve \
  --model mlx-community/K2-Horizon-3.7B-4bit \
  --trust-remote-code \
  --mode baseline \
  --host 127.0.0.1 \
  --port 13886
```

Confirm from startup/logging that the model uses the mlx-lm custom-model path, loads successfully, and does not report the former mlx-vlm `custom_model.ModelConfig` failure. Send one chat/completion request to the local server and confirm that at least one generated response completes. Stop the server after the check.

## Scope check

Review the final diff to confirm production changes are limited to `src/mlx_dspark/load.py` and regressions are limited to `tests/test_formats.py` and `tests/test_remote_code.py`. No dependency, cache, server lifecycle, speculative decoding, lookup, battery, prefix-cache, or memory-management change is part of this feature.
