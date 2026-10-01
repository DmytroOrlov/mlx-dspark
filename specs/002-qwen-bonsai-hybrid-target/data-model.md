# Data Model: Qwen-Bonsai Hybrid Target Experiment

This document defines the minimal records needed to compose a target, reject invalid runs, adjudicate Series A, enforce Gate B, and later reuse the same target matrix for Series B. It does not define new speculative behavior.

## TargetComposition

One immutable declarative target identity.

| Field | Type | Validation |
| --- | --- | --- |
| `composition_id` | string | Stable label (`H0`, `H1a`, `H1b`, `H1c`, `H2`, `H3`, `B0`) for required cells; labels do not select ownership by themselves. |
| `qwen_checkpoint` | `CheckpointRef` | Exact repo id/local resolved snapshot, revision, config digest, weight-file fingerprints. |
| `bonsai_checkpoint` | `CheckpointRef` | Exact Bonsai2 Prism repo `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` or its exact resolved snapshot, revision, config digest, pack weight fingerprint. Pass explicitly; do not use the stale `ternary-bonsai-27b` registry row. |
| `donor_block_indices` | sorted unique `list[int]` | Each integer is in `[0, 63]`; composition is generated from this set. `[]` is H0. |
| `block_owner_by_index` | `list[Owner]` length 64 | Expanded deterministic map. `bonsai` exactly at donor indices and `qwen` elsewhere. |
| `embedding_owner`, `final_norm_owner`, `lm_head_owner` | `Owner` | For H0–H3 each is `qwen`. For B0 each is `bonsai`. |
| `model_dimensions` | object | Includes hidden/intermediate/vocabulary sizes, layer count, and full-attention interval from both sources. |
| `layer_families` | list of `attention` / `gated_delta` | Same-index Qwen and Bonsai family must match; order/length must equal 64. |

Required mapping:

| Composition | `donor_block_indices` | Special owners |
| --- | --- | --- |
| H0 | `[]` | Qwen embedding/norm/head |
| H1a | `[63]` | Qwen embedding/norm/head |
| H1b | `[62]` | Qwen embedding/norm/head |
| H1c | `[62, 63]` | Qwen embedding/norm/head |
| H2 | `[60, 61, 62, 63]` | Qwen embedding/norm/head |
| H3 | `[56, 57, 58, 59, 60, 61, 62, 63]` | Qwen embedding/norm/head |
| B0 | full Bonsai loader, not a mixed composition | Bonsai embedding/norm/head and all blocks |

H0–H3 can be expressed by any valid explicit index list in the generic interface, but Series A accepts only the table above. No separate model class or named-model conditional is required.

## CheckpointRef

| Field | Type | Validation |
| --- | --- | --- |
| `repo_id` | string | Exact model namespace from the selected checkpoint. |
| `resolved_path` | string | Absolute local snapshot path actually loaded. |
| `revision` | string | Immutable snapshot commit/revision; mandatory for accepted result. |
| `config_sha256` | hex string | Hash of the selected `config.json`. |
| `weight_files` | list of `{path, bytes, fingerprint}` | Every loaded safetensors file or content-addressed blob is represented. |
| `fingerprint_method` | string | Identifies the source of each fingerprint (SHA-256, HF content-addressed blob id, or verified index). |

The drafter uses the same checkpoint record. The Bonsai-specific Series-B drafter is specifically `naklitechie/Qwen3.8-27B-DFlash2-ternary-bonsai2`; its previously recorded revision/hash are leads only and are reverified when that arm is actually run.

## CompositionManifest

One deterministic JSON record emitted before benchmark acceptance and embedded/referenced by every result.

Required fields:

- schema/version, feature id, run id, runtime Git SHA and dirty state, relevant source hashes, Python/MLX/mlx-lm versions, OS/hardware identity;
- `TargetComposition`, including the explicit owner for every index and special module;
- Qwen, Bonsai, and drafter `CheckpointRef`s;
- `tap_ids`, `tap_semantics="zero-based block output"`, and `affected_tap_ids`/count derived from `block_owner_by_index`;
- per-index live Python module identity check, owner module type, block family, parameter names/shapes/dtypes, and Prism `Packed` path presence for donor projections;
- integrity statuses for load, ownership, module interfaces, finite parameters, deterministic composition, tap behavior, cache transitions, and H0 unchanged behavior;
- controller, plain-KV setting, speculative mode, generation/sampling options, prompt corpus digest, tokenization IDs or IDs digest, warmup/measurement protocol, and raw output path.

The manifest must be sorted/canonical when serialized so same-input compositions can be compared deterministically. Every requested donor index must refer to the retained donor module object. Every complement index must refer to the original Qwen module object. Embedding/norm/head identity is checked against the Qwen source objects for H0–H3.

## BenchmarkCondition

A paired experimental cell with one target composition and one drafter checkpoint.

| Field | Type | Validation |
| --- | --- | --- |
| `series` | enum | `A` or later `B`. |
| `composition_id` | string | Series A is one of all seven required targets. Series B is H0/H1c/H2/H3/B0 plus only the spec-authorized H1a/H1b exception. |
| `drafter` | `CheckpointRef` | Series A always original `incoai` checkpoint. Series B is one of the two named exact checkpoints. |
| `controls` | object | Frozen runtime, ordinary CapController configuration, plain KV, prompts, generation/sampling, hardware, and measurement method. |
| `replicate_id` | string/int | `discovery` or matched confirmation replicate/group identifier. |
| `order_index` | int | Actual order within the declared discovery/repeat schedule. |

For paired Series-B cells, only drafter checkpoint and strictly necessary serialization/loading compatibility may differ; the frozen `TargetComposition` and all other controls must compare equal.

## RunResult

One successful or invalidated measurement attempt; invalid attempts remain recorded with failure status and are excluded from adjudication.

Required measured fields:

- serial target tokens/sec, speculative `decode_tokens_per_sec`, speculative speedup over that target’s serial result;
- acceptance/accept length using existing runtime definitions, target-forward count, generated tokens per target forward, available width/cap distribution;
- peak steady-state memory, generated token count, benchmark duration, prompt count/IDs digest;
- per-run start/end UTC, order, warmup/reset state, runtime/target/drafter provenance, and integrity statuses.

Unavailable component timing must be represented as `null` with a reason. Do not derive missing performance values from the feature-001 historical records.

## RepeatDecision

| Field | Type | Meaning |
| --- | --- | --- |
| `comparison` | string | Candidate vs fresh H0, or H1c vs H1a/H1b. |
| `discovery_run_ids` | list | Inputs to repeat selection. |
| `noise_basis` | object | Observed within-H0/order and paired variability; no invented universal percentage threshold. |
| `selected` | bool | True for apparent win, regression, near-noise comparison, or decision-relevant interaction. |
| `reason` | string | Reproducible rationale from discovery facts. |
| `matched_repeat_run_ids` | list | If selected, at least three run pairs per compared condition, with alternating order. |
| `adjudicated_result` | enum | `confirmed_win`, `confirmed_regression`, `within_noise`, `inconclusive`, or `not_selected`. |

## SeriesAdjudication

One feature-local artifact for Series A or B. It references immutable run records and reports three separate layers:

1. **Measured facts**: raw run values and per-condition observed variability.
2. **Derived metrics**: paired differences and percent changes from fresh H0; serial/speculative ratios; acceptance/forward/memory deltas.
3. **Interpretation**: answers each question in FR-013 (Series A) or FR-020 (Series B), including a permitted no-win/no-production-change result. It must not infer an exact untested crossover block.

Series-A required condition set is exactly H0, H1a, H1b, H1c, H2, H3, B0. It includes at least one fresh H0 discovery control; the proposed protocol brackets discovery with two fresh H0 runs. Confirmation runs do not replace required discovery records.

## GateBRecord

Machine-verifiable `evidence/gate-b.json` record.

| Field | Type | Validation |
| --- | --- | --- |
| `status` | enum | Defaults to `CLOSED`; only `OPEN` permits Series B. |
| `spec_hash`, `constitution_version`, `runtime_revision` | string | Must match the governing feature inputs and experiment runtime. |
| `required_targets` | list | Exactly the seven Series-A targets. |
| `run_ids_by_target` | mapping | Every target maps to valid integrity-passing discovery results; H0 includes fresh evidence. |
| `semantic_evidence` | refs | Prism boundary and exact loaded drafter tap list/semantics/counts for every variant. |
| `repeat_decisions` | refs | Every apparent winner/loss/near-noise decision has at least three matched pairs per compared condition; unresolved decisions keep gate closed. |
| `adjudication_path`, `adjudication_sha256` | string | Present, schema-valid, and answers all Series-A questions. |
| `closure_checks` | mapping bool | Every requirement passes before status can be `OPEN`. |
| `decision_time`, `decision_owner` | string | Audit trail. |

The future Series-B entry point validates status, all closure checks, referenced hashes, and fixed Series-B matrix before any model load or benchmark. Missing, stale, malformed, or closed gate is a hard refusal.
