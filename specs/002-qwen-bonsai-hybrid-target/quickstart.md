# Quickstart: Qwen-Bonsai Hybrid Target Experiment

This is the execution guide for the later implementation. The commands below describe the planned feature-local runner interface; they are not available until the implementation phase. Planning does not load models, run these commands, or produce benchmark results.

## Prerequisites

- Run on the target Apple Silicon M4 Pro environment with the project’s pinned Python/MLX/mlx-lm dependencies.
- Resolve and pin `mlx-community/Qwen3.8-27B-4bit`, `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`, and `incoai/Qwen3.8-27B-DFlash2`. Record immutable revisions and weight/config fingerprints. The spec’s current assumption names the older Qwen3.6 sibling; correct that id before runtime execution. Do not silently use the stale `ternary-bonsai-27b` registry target, the `nathansutton` Bonsai repack, or another drafter.
- Use plain target KV, ordinary production `CapController` (`max_draft_tokens="auto"` through `Engine.load`), existing production Engine generation path, and one frozen prompt corpus/generation configuration for every matched condition.
- Ensure `specs/002-qwen-bonsai-hybrid-target/evidence/` is writable. Keep all feature-002 artifacts there; feature-001 remains read-only.

## 1. Model-free preflight

After implementation, inspect checkpoint configs, weight inventories/fingerprints, revisions, tokenizer IDs, the loaded original DFlash tap list, current runtime revision, machine facts, expected composition table, and output paths without loading model weights or running inference:

```sh
.venv/bin/python specs/002-qwen-bonsai-hybrid-target/evidence/probes/series_a.py --preflight
```

Expected: H0/H1a/H1b/H1c/H2/H3/B0 are listed; H0–H3 have Qwen embedding/norm/head; donor indices match the spec; B0 is marked full Bonsai; DFlash taps are `[5,19,33,47,61]` with block-output semantics and counts `0,0,0,0,1,1,5`. A changed tap list, missing revision, changed model shape/family, missing weight file, dirty/unexpected runtime, or non-writable output is a preflight failure that produces no benchmark result.

## 2. Composition and integrity smoke

Use a small deterministic prompt/token sequence and fresh caches. Run each target in required Series-A order:

```sh
for variant in H0 H1a H1b H1c H2 H3 B0; do
  .venv/bin/python specs/002-qwen-bonsai-hybrid-target/evidence/probes/series_a.py \
    --integrity --variant "$variant"
done
```

Expected: each run records resolved source checkpoints/revisions, exact 64-index ownership, embedding/norm/head owners, shape/dtype/layer-family checks, finite parameters and outputs, tap output width/count, packed donor projection presence, and deterministic composition status. A short speculative request must exercise existing target cache advance and rollback. H0 must use the unchanged ordinary Qwen loading path. A failed integrity status blocks performance measurements for that target and keeps Gate B closed.

## 3. Series-A discovery

Run the exact seven target conditions with a single matched discovery pass and fresh H0 controls bracketing the sequence:

```sh
.venv/bin/python specs/002-qwen-bonsai-hybrid-target/evidence/probes/series_a.py \
  --discovery --order H0,H1a,H2,H1b,H3,H1c,B0,H0
```

The runner uses `Engine.load`/`Engine.generate`, ordinary `CapController`, and plain KV. It measures serial target decode with speculation disabled, then speculative decode under identical prompt IDs and generation settings. Every condition is a fresh process or equivalently reset engine with unique run output. The record includes `decode_tokens_per_sec`, serial target tokens/sec, speedup, existing acceptance/accept length, target forwards, generated tokens per forward, width/cap distribution, steady-state memory, generated token count/duration, provenance, and integrity. Per-component milliseconds remain unavailable unless the existing runtime exposes them without adding synchronization-heavy instrumentation.

## 4. Confirmation and Series-A adjudication

After discovery, create a repeat selection record for all target-vs-H0 comparisons and the H1c interaction comparisons that remain decision-relevant. Repeat only apparent winners, regressions, and near-noise comparisons. For every selected comparison, collect at least three matched pairs in alternating order and record the selection basis/seed:

```sh
.venv/bin/python specs/002-qwen-bonsai-hybrid-target/evidence/probes/series_a.py \
  --confirm-from specs/002-qwen-bonsai-hybrid-target/evidence/series-a/repeat-selection.json

.venv/bin/python specs/002-qwen-bonsai-hybrid-target/evidence/probes/series_a.py \
  --adjudicate
```

Expected: the adjudication keeps measured facts, derived metrics/deltas, and interpretation separate; compares all required targets with fresh H0; answers the FR-013 questions; and links raw run records and manifests. One marginal run cannot produce a confirmed winner.

## 5. Gate B and future Series B

The Series-A adjudicator writes `specs/002-qwen-bonsai-hybrid-target/evidence/gate-b.json` closed by default. Gate closure is a data validation step; it does not depend on a conversational approval. A later Series-B runner must validate the open gate, feature/spec/constitution/runtime hashes, all closure checks, and the exact fixed matrix before loading any model. If open, it reuses the same composition and measurement harness for H0/H1c/H2/H3/B0 with both named DFlash2 checkpoints, plus only the single H1a/H1b exception allowed by FR-017.
