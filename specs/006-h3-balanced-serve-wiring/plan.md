# Implementation Plan: H3 Balanced Serve Wiring

**Branch**: `006-h3-balanced-serve-wiring` | **Date**: 2026-09-26 | **Spec**: [spec.md](spec.md)

**Input**: Revised donor-block serve specification. The public interface is the paired `--donor-model` and `--donor-blocks` options.

## Summary

Add paired optional donor arguments to `mlx-dspark serve`, parse and normalize the small block-selection syntax before resolving models, resolve donor and Qwen snapshots through the existing model/snapshot path, construct the existing `TargetCompositionRequest`, and pass it through `Engine.load(target_composition=...)`. No donor arguments means the existing ordinary serve load kwargs remain unchanged. DFlash selection remains independent.

For H3, `--donor-blocks 56-63` expands to `(56, 57, 58, 59, 60, 61, 62, 63)` from `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`; Qwen keeps its embedding, unselected blocks, final norm, and LM head. Duplicate indices are normalized by sorting and deduplicating. Malformed input and one-sided option pairs fail before model resolution/loading.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: argparse; existing `mlx_dspark.load` resolution; existing `TargetCompositionRequest` and `Engine.load` composition seam
**Storage**: Existing model and Hugging Face snapshots; no new persistent state
**Testing**: Focused model-free CLI/hybrid tests; optional single Apple-Silicon smoke
**Target Platform**: CLI on supported Python platforms; hardware-backed smoke on Apple Silicon
**Project Type**: Python CLI and serving library
**Performance Goals**: None; serve-wiring only
**Constraints**: Small opt-in diff; preserve ordinary serving, immutable snapshot/path checks, target ownership, and independent drafter selection. Do not alter DFlash, CapController, cache/KV, tokenizer, reasoning, sampling, or generation defaults.
**Scale/Scope**: Two paired serve options, one tiny parser, one existing request/handoff; no preset selector or general composition language

## Constitution Check

Applicable principle: III, Preserve the Vanilla Qwen Path. The composition remains explicitly opt-in and the omitted-option path passes no composition request. The work does not revisit compatibility, quality, or performance evidence and creates no experiment axis.

**Gate before research**: PASS. Scope is limited to CLI wiring and focused model-free checks.

## Project Structure

```text
specs/006-h3-balanced-serve-wiring/
├── plan.md
├── research.md
├── data-model.md
├── contracts/
│   └── serve-cli.md
└── quickstart.md

src/mlx_dspark/cli.py             # paired flags, block parser, request construction, startup report
tests/test_cli_pause.py           # focused model-free CLI behavior tests, if existing fixtures fit
tests/test_hybrid_target.py       # focused exact request/ownership wiring assertions as needed
```

**Structure Decision**: Expected production changes are confined to `src/mlx_dspark/cli.py`: add two argparse flags, a small parser/formatter, early pair/syntax validation, donor/Qwen identity resolution using the established path, request construction, conditional `Engine.load` kwarg, and concise startup output. Focused additions belong in existing CLI/hybrid tests. `server.py`, `hybrid_target.py`, and `load.py` can remain unchanged because the request type, immutable path checks, composition handoff, and ordinary `_resolve` path resolver already exist.

## Smallest Expected Production Diff

One file, `src/mlx_dspark/cli.py`. The parser accepts a single index, ascending inclusive range, and trivial comma-separated combinations. It validates integer tokens, bounds 0–63, empty parts, and range order; normalizes sorted unique indices. It first rejects incomplete option pairs and malformed selections, before resolving either model. With both flags, it uses the existing `_resolve` checkpoint-path resolver and extracts only an immutable revision present in the resolved snapshot path, then passes a `TargetCompositionRequest`; without them, it omits the composition request and preserves the current call path. It prints donor ID, resolved revision when the path provides it, normalized indices/ranges, and Qwen ownership of embedding/final norm/LM head.

No production edits are planned in `server.py`, `hybrid_target.py`, or `load.py`. The existing immutable revision and resolved-path checks remain mandatory; paths without an embedded immutable snapshot SHA fail closed rather than receiving an invented identity. Resolution failures continue through existing error behavior.

## Test Scope

Add only model-free checks for: default no-composition path; exact H3 expansion 56–63; single index 62; paired-option rules; malformed, out-of-range, and descending selection rejected before loading; and preservation of explicitly selected drafter. If comma syntax is implemented, cover comma lists. No model loads are needed for these checks.

Optional final smoke: start the exact H3 command from [quickstart.md](quickstart.md) once and send one tiny OpenAI-compatible request. No throughput benchmark or quality comparison.

## Constitution Check (post-design)

**Gate after design**: PASS. The default path remains structurally unchanged, and the new flags only construct the existing composition request when paired. All unrelated runtime and experimental controls remain fixed.

## Complexity Tracking

No constitution violations.
