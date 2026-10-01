# Implementation Plan: Qwen-Bonsai Hybrid Target Experiment

**Branch**: `002-qwen-bonsai-hybrid-target` | **Date**: 2026-09-25 | **Spec**: [spec.md](spec.md)

**Input**: `specs/002-qwen-bonsai-hybrid-target/spec.md`, governed by the current constitution v2.0.0.

## Summary

Add an opt-in experimental target composition path that replaces explicitly listed Qwen3.8 decoder blocks with the same-index decoder modules from the repository-supported Prism/Bonsai2 checkpoint. H0–H3 keep the Qwen embedding, final norm, and LM head; only block owners vary. B0 stays the separately loaded full-Bonsai endpoint control. Load the donor first through the existing verified Prism loader, retain only requested donor block modules, release the remaining donor state, then load Qwen and compose before constructing `Target`. This avoids holding two complete targets at once while reusing the existing packed-linear, Hadamard, and MMA implementation.

An experimental benchmark runner will pass the composition opt-in through the existing `Engine.load`/`Engine.generate` path and reuse the feature-001 measurement/provenance patterns. The default target-plus-drafter path is unchanged. No benchmark is run during this planning pass.

## Technical Context

**Language/Version**: Python 3.12 in the recorded project runtime; MLX arrays/modules.

**Primary Dependencies**: mlx-dspark target/runtime; installed `mlx-lm` 0.31.3 Qwen3.5 hybrid model classes; MLX 0.32.2; existing `prism_pack`/`mlx_qmm_mma`; `safetensors`/Hugging Face checkpoint loading.

**Storage**: Hugging Face snapshot files; feature-local JSON manifests and raw result records under `specs/002-qwen-bonsai-hybrid-target/evidence/`.

**Testing**: Existing pytest suite plus small deterministic composition/manifest tests; runtime-integrity smoke checks before any performance run. No tests or benchmarks are run by this planning pass.

**Target Platform**: The project’s Apple Silicon M4 Pro MLX runtime, plain target KV, ordinary production DFlash2/CapController path.

**Project Type**: Python library/CLI/server with an experimental model-loading and benchmark path.

**Performance Goals**: Produce adjudicable serial-target and speculative decode evidence for exactly H0, H1a, H1b, H1c, H2, H3, and B0. The goal is evidence; a hybrid win is not presumed.

**Constraints**: Preserve the current constitution and spec; do not change target/drafter/control axes; do not require two complete 27B targets resident simultaneously; no changes to ordinary Qwen loading when the opt-in is absent; no production recommendation without task-quality evaluation.

**Scale/Scope**: One 64-block Qwen3.8/Bonsai2 target family; explicit donor block-index set; two already existing DFlash2 checkpoints across Series A, Gate B, and the later fixed Series-B matrix.

### Repository facts resolved before design

- Both inspected target configurations use the `mlx_lm.models.qwen3_5.Model` / `TextModel` / `Qwen3_5TextModel` stack and `DecoderLayer` blocks. The Qwen checkpoint has 64 blocks, hidden size 5120, intermediate size 17408, vocabulary 248320, and full attention interval 4. The inspected Bonsai2 Prism pack has the same values and `model_type=prism_hadamard_qwen35`, 2-bit affine g128 packing.
- The two `DecoderLayer` objects expose the same callable interface `(x, mask=None, cache=None) -> residual`; same-index layers share the attention-vs-GatedDeltaNet pattern. Their projection implementations intentionally differ: Qwen quantized linear modules versus Prism `Packed` modules.
- Bonsai `Packed.__call__` rotates projection inputs locally before its packed matmul. The embedding applies the inverse rotation at lookup. Block output is the ordinary residual sum; no hidden-state rotation survives a decoder block. Qwen↔Bonsai boundaries therefore need no bridge. Existing `prism_pack.py` and `mlx_qmm_mma.py` remain the authority for packed math and kernel routing.
- `Target._body_hybrid` captures hidden state immediately after `layer(...)` for a zero-based block index. The current inspected DFlash2 checkpoint configs both request `[5, 19, 33, 47, 61]`, so taps are block outputs. H0/H1a/H1b/H1c affect zero consumed taps; H2/H3 affect one (block 61); B0 affects all five. The checkpoint tap list and counts are still asserted and recorded at runtime.
- Feature-001 physical evidence is not a substitute for Series-A measurements or the selected official donor revision. It establishes a working Prism representation/loader path and reusable measurement/provenance techniques. The locally recorded Qwen snapshot was `10c35caafbb80f7dc6a7a432cdd11af10a6d4818`; the older feature’s Bonsai run used a separate `nathansutton` repack. The correct official Bonsai2 repo is `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`; its exact weight fingerprint must be recorded in each run manifest. The spec currently names the older Qwen3.6 sibling in its Bonsai2 assumption and needs that repo-id correction before runtime execution.

The detailed source evidence and the one provenance caveat are in [research.md](research.md).

## Constitution Check

**Pre-design gate: PASS WITH SPEC CORRECTION NOTED.** The design keeps the seven specified targets and Series-A fixed controls, uses the existing Prism implementation, keeps the ordinary Qwen path opt-in-free by default, treats runtime integrity as a prerequisite, and makes end-to-end decode the decision metric. H0–H3 change only transformer-block ownership. B0 is explicitly treated as the full-Bonsai endpoint control described by the spec and the user’s interpretation; its Bonsai embedding/norm/head do not weaken the one-axis claim for H0–H3. Research found that the spec assumption names the wrong Bonsai2 repo id; update only that checkpoint identifier before runtime implementation. No constitution amendment or experiment-axis change is proposed.

**Post-design gate: PASS.** The opt-in loader is isolated, has no effect when absent, retains the original drafter and serving controls, makes ownership auditable, and does not change speculative or rollback semantics. Runtime cache checks are limited to establishing integrity of these composed targets; feature-001 rollback-parity work is not reopened.

## Design Decisions

1. **Composition**: Use an extensible explicit donor-index set, not H1/H2/H3-specific classes. A new `hybrid_target.py` coordinates composition; `load_target` accepts an internal opt-in replacement payload so replacement happens before `Target` discovers GatedDeltaNet modules and installs its capture hooks.
2. **Memory**: Load the donor through the current `prism_pack.load` while Qwen is absent. Keep references only to requested donor `DecoderLayer` modules, drop the donor wrapper/embedding/head/unselected layers, clear reclaimable MLX cache, then load the Qwen target and replace the requested indices. This uses a temporary full donor load, but never keeps two complete 27B targets resident. The feature-001 measured model peaks (about 17.1 GB for Qwen and 9.4 GB for Bonsai in the recorded task) show why simultaneous complete models are an unsafe default. Do not implement a second Prism tensor loader or math path. If smoke checks show the retained donor modules plus Qwen exceed device memory, stop and resolve memory before performance runs; selective safetensors block reads are the fallback implementation decision, not a speculative parallel framework.
3. **Runtime path**: Add an optional composition argument to `Engine.load`, consumed only by the feature-local runner. With no composition argument, `Engine.load` continues its current `load_target` route. Series A explicitly requests the ordinary production `CapController` via `max_draft_tokens="auto"` for every cell; this fixes the controller policy and does not create a cap sweep. Keep the user-facing ordinary `--model Qwen --drafter DFlash` invocation unchanged and do not add a production CLI option.
4. **B0**: Load `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` with the existing full Bonsai target loader, passing its repo id explicitly with the drafter. Do not resolve through the stale `ternary-bonsai-27b` registry row, and do not force Qwen embedding/head ownership onto B0.
5. **Benchmarking**: Generalize the feature-001 single-request Engine harness to the exact feature-002 matrix. Capture required values from `GenResult`, existing round stats, existing MLX peak-memory calls, serial greedy generation, and checkpoint/runtime fingerprints. Do not add component timers requiring synchronization or invasive hooks.
6. **Gate B**: Write a machine-readable `evidence/gate-b.json`. It remains closed unless every required Series-A result and integrity/semantic/adjudication/repeat condition is linked and passes. The later Series-B runner must validate this artifact and refuse to run with absent, stale, or closed evidence.

## Expected Files to Change in Runtime Implementation

- `src/mlx_dspark/hybrid_target.py` (new): composition specification, donor-first loading/extraction, compatibility checks, pre-`Target` block replacement, exact ownership manifest, and deterministic composition assertions.
- `src/mlx_dspark/load.py`: add an internal optional block-replacement input to `load_target`; apply it to the raw Qwen model before constructing `Target`, then install existing MMA routing when Prism donor blocks are present. Default `None` path stays unchanged.
- `src/mlx_dspark/server.py`: add an optional `target_composition` argument to `Engine.load`; route to the isolated composer only when set. Keep target/drafter defaults, calibration, controller, generation and cleanup behavior intact.
- `tests/test_hybrid_target.py` (new): test explicit arbitrary index ownership, all H0–H3 maps, B0 as separate full target, shape/layer-type rejection, module identity checks, deterministic manifest rendering, and default loader behavior. Model-free tests use small fixtures; the measured runtime smoke validates real checkpoints.
- `specs/002-qwen-bonsai-hybrid-target/evidence/probes/series_a.py` (new): feature-local runner based on the T027 Engine/metrics harness, with composition matrix, plain-KV checks, preflight, and per-run manifests. Add a later Gate-B-guarded Series-B entry point only after Gate B is met; no Series-B targets or measurements are introduced in this plan.
- `specs/002-qwen-bonsai-hybrid-target/evidence/` (new outputs): immutable input/runtime manifests, integrity records, discovery/repeat run JSON, Series-A adjudication, and Gate-B record. These are feature-002 artifacts; feature-001 evidence remains read-only.

`src/mlx_dspark/prism_pack.py`, `src/mlx_dspark/target.py`, `src/mlx_dspark/generate.py`, and the DFlash model implementation do not need math, tap, speculative-algorithm, or rollback changes. The composition occurs before `Target` initialization and reuses its existing hybrid forward/verify/rollback logic. `tests/test_bonsai.py` and feature-001 probes are not wholesale copied or rewritten.

## Project Structure

```text
src/mlx_dspark/
├── hybrid_target.py       # opt-in target composition and ownership manifest
├── load.py                # optional replacement hook; vanilla default unchanged
└── server.py              # optional Engine.load composition handoff

tests/
└── test_hybrid_target.py  # small deterministic composition/integrity-contract tests

specs/002-qwen-bonsai-hybrid-target/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
└── evidence/
    ├── probes/series_a.py
    ├── runs/<run-id>/
    ├── adjudication/series-a.json
    └── gate-b.json
```

**Structure Decision**: Keep generic loader integration minimal in existing `load.py` and `server.py`; put all hybrid ownership/load logic and experiment tooling in feature-specific modules/artifacts. No public CLI, model-family subclass, `Target` math branch, or new benchmark package is needed.

## Complexity Tracking

No constitution violations or justified scope additions. A contracts directory is omitted because the opted-in composition call is an internal experimental loader interface, not a new external API or ordinary CLI contract; its exact contract is documented in [quickstart.md](quickstart.md) and [data-model.md](data-model.md).
