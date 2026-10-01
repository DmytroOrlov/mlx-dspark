# Research: Golden Panel Layer-Drafter Matrix

## Decision: Split physical execution from offline finalization

**Decision**: Use `evidence/probes/panel_matrix.py` for manifests, physical preflight, fixed first-pass orchestration, immutable raw writes, provenance/memory capture, and repeat execution. Use `evidence/probes/finalize_matrix.py` for raw reading, derived calculations, validation, repeat selection/manifest writing, and reporting. The finalizer performs no model inference.

**Rationale**: Offline arithmetic, validation, Pareto classification, or Markdown fixes must not invalidate otherwise valid raw inference measurements. A finalizer-only repair therefore preserves raw evidence and reruns only offline steps. Repeat trigger selection is also offline, while physical repeat execution stays with the frozen physical runner and consumes an explicit immutable manifest.

**Alternatives considered**: One source-frozen script for physical work and all reporting. Rejected because a harmless reporting defect discovered after expensive observations would unnecessarily couple raw validity to offline code.

## Decision: Reuse Feature 003 composition and physical-cell patterns

**Decision**: Carry forward declarative donor-block definitions, explicit target/drafter identity records, checkpoint verification, fresh child execution, source hashing, and physical provenance into the feature-local runner.

**Rationale**: `specs/003-product-golden-hybrid-sweep/evidence/probes/product_sweep.py` provides the accepted seven-target composition and B0 identity, immutable revisions and selected B-B digest, child-process pattern, and provenance conventions. The supplied contract fixes these facts; Feature 005 does not reopen correctness work.

**Alternatives considered**: Change production runtime or import the old runner as a mutable dependency. Neither is needed; both broaden scope or couple the new runner to an older feature.

## Decision: Reuse Feature 004 prompt and offline-validation patterns

**Decision**: Keep the exact five certified prompt definitions, regenerate current tokenizer IDs/digests at physical preflight, and apply the accepted freshness/output-digest and offline validation/report conventions.

**Rationale**: `specs/004-golden-prompt-panel/evidence/probes/prompt_panel.py` defines the prompt identities and demonstrates chat-template tokenization, fresh request assertions, output hashes, and MLX memory capture. Its accepted panel report supplies the qualification context. Prompt selection is already settled.

**Alternatives considered**: Requalify prompts or assume historical IDs remain current. Neither improves this bounded experiment; preflight pins current IDs while preserving exact certified prompt text.

## Decision: Use the accepted request-boundary MLX memory pattern

**Decision**: Follow Feature 003/004 semantics using `mx.metal.reset_peak_memory()`, `get_active_memory()`, and `get_peak_memory()`: after loading/warmup and fresh request-state setup, reset the peak counter, record active baseline immediately before generation, execute only the measured request, and immediately read peak. Record assertions/boundary metadata in raw evidence.

**Rationale**: Existing feature probes establish the MLX memory API and request measurement boundary. This preserves comparability and ties peak memory to the measured request interval. Peak increment is raw peak minus raw baseline; normalized memory comparisons are computed only offline.

**Alternatives considered**: Infer memory from target ownership, use historical values, or invent a new measurement API. All conflict with measured per-cell memory and accepted runtime semantics.

## Decision: Pin the repository's accepted Python runtime

**Decision**: Use the current repository `.venv/bin/python` for preflight and physical children. Record its resolved identity and concrete interpreter version in preflight and every physical observation; reject execution if it differs from the preflight pin.

**Rationale**: The experiment requires a stable runtime across cells but does not require a new environment or an obsolete hard-coded minor version.

**Alternatives considered**: Provision a distinct Python 3.11 environment or silently use whichever interpreter is on PATH. Neither is needed or sufficiently reproducible.

## Decision: Preserve immutable raw evidence and a physical-only source freeze

**Decision**: Write each raw observation once with physical-runner SHA. Freeze `panel_matrix.py` after passing preflight and immediately before first-pass cell 1. Record the finalizer SHA separately in derived/report artifacts. A physical defect triggers affected fresh observations under a newly preflighted runner SHA; an offline-only defect preserves all raw and reruns only finalization.

**Rationale**: Raw evidence must stay auditable while physical orchestration remains comparable. Separating sources gives offline artifacts independent repairability.

**Alternatives considered**: Freeze all offline report code as a condition of raw validity, or edit raw cells to repair derived values. Both unnecessarily risk valid physical evidence.

## Unresolved research

None. Prompts, targets, drafters, controls, physical order, derived formulas, repeat triggers, and report contract are fixed. No optional research or benchmark is needed to plan implementation.
