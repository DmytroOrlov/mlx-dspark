# Data Model: K2 Custom MLX-LM Routing

This feature introduces no persisted data or new public configuration. It interprets existing checkpoint metadata and an existing process-level trust setting.

## Checkpoint Metadata

| Field | Source | Meaning in routing | Validation/handling |
| --- | --- | --- | --- |
| `model_type` | Top-level `config.json` | Existing family lookup and text-capable multimodal exception | Missing/unknown values retain existing behavior unless `model_file` classifies a no-modality custom text checkpoint |
| `model_file` | Top-level `config.json` | Declares checkpoint-supplied architecture code | Presence selects mlx-lm only after existing family exception and explicit vision/audio checks |
| `vision_config` | Top-level `config.json` | Explicit vision modality marker | Routes to mlx-vlm after the existing Qwen3.5 exception |
| `audio_config` | Top-level `config.json` | Explicit audio modality marker | Routes to mlx-vlm after the existing Qwen3.5 exception |
| `MODEL_REMAPPING` entry | Installed mlx-lm metadata | Maps aliases to a built-in mlx-lm module | Existing lookup remains after custom-text routing |

## Remote-Code Trust Setting

| Field | Source | Meaning | Validation/handling |
| --- | --- | --- | --- |
| `TRUST_REMOTE_CODE` | Existing CLI/environment configuration | Operator's explicit process-level permission | `refuse_remote_code` runs after metadata read and before target loader call/import; value also remains in tokenizer configuration and is forwarded directly to mlx-lm if its loader signature supports it |

## Route Result

`mlx_lm` or `mlx_vlm`, selected from checkpoint metadata. This is transient loader control data and does not persist beyond the call. A selected loader error propagates through that loader's existing path; no alternate-framework retry is introduced.
