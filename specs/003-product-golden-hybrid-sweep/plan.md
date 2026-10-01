# Implementation Plan: Product Golden Hybrid Sweep

**Branch**: `003-product-golden-hybrid-sweep` | **Date**: 2026-09-26 | **Spec**: [spec.md](spec.md)

## Summary

Build the shortest path from accepted feature-002 target composition machinery to a product-anchored sweep. A small feature-local runner combines feature-002 target/drafter composition with the authoritative T027 Golden-A product benchmark path. Cheap preflight checks checkpoint identity, reused runtime identity, prompt digest, and fresh process/cache setup. Run one fresh H0+B-Q canary as matrix cell 1; stop and write a concrete mismatch record if it returns to the 32–36 tok/s / ~49% regime. Otherwise retain it as H0+B-Q and run only the remaining 13 cells, make only policy-selected repeats, and publish a product table and machine-readable results.

## Technical Context

**Language/Version**: Python 3.12 (repository runtime)
**Primary Dependencies**: mlx, mlx-lm, existing `mlx_dspark` Engine, Prism packed loader
**Storage**: Feature-local JSON evidence under `specs/003-product-golden-hybrid-sweep/evidence/`
**Testing**: Cheap model-free preflight and runner argument/config checks; no broad correctness suite or benchmark before the canary
**Target Platform**: Existing Apple Silicon MLX machine and normal runtime environment
**Project Type**: Python inference runtime with feature-local experiment tooling
**Performance Goals**: Determine whether fresh Golden-A H0+B-Q returns to the ~45 tok/s product regime, then measure exactly 14 matched product cells
**Constraints**: Preserve T027 production Engine request semantics, same runtime/machine, plain KV, ordinary CapController, fresh process/cache per cell; no production runtime changes, WidthPolicy, KV8, tuning, or architecture redesign
**Scale/Scope**: Seven compositions × two drafters; the first H0+B-Q observation is both canary and matrix cell 1, for 14 mandatory physical observations total; only selective repeats per spec

## Constitution Check

- **Throughput is the decision metric**: pass. Speculative decode tok/s is primary; all other fields explain it.
- **One experiment axis at a time**: pass. Target composition and drafter are the only cell differences; workload/runtime controls are fixed.
- **Preserve vanilla Qwen**: pass. Feature-local composition/benchmark adapter only; no production runtime change is planned.
- **Reuse representation and tap evidence**: pass. Feature 002 established ownership, canonical boundaries, Prism packed execution, taps `[5,19,33,47,61]`, and runtime integrity. No re-proof is in scope.
- **Runtime integrity before performance claims**: pass. Only cheap current-file/checkpoint/prompt/freshness preflight is planned; accepted feature-002 gates remain the authority.
- **Matched benchmark discipline**: pass. Exact T027 request and production Engine controls; the fresh canary is cell 1 of 14, followed by the 13 remaining fresh matched cells; selective repeats only.
- **Historical and feature-002 performance values**: context only. Feature-002's ~32–36 tok/s results are not the product baseline; Golden A is 45.779 tok/s and Golden B ~45.4 tok/s is hot-prefix context only.

## Minimal Implementation Stages

1. **Reuse audit**: Reuse `src/mlx_dspark/hybrid_target.py` composition requests, donor loading, block replacement, ownership manifest, and existing loader/Engine integration. Reuse feature-002 checkpoint manifests and accepted integrity/runtime-semantics records for hybrid target construction, donor/drafter provenance, and runtime behavior. For performance request/Engine semantics, use the T027 product path in `specs/001-bonsai-rollback-parity/evidence/probes/t027_product_ab.py`, anchored by its exact Qwen Golden-A raw result. Feature-002 `run_series_condition` is not the benchmark to wrap unchanged: its p01/p02/p03 corpus, 128-token generation, prefix-cache-off control, and other Series-A settings differ from T027. Small helpers may be reused only when they do not change T027 request semantics. No benchmark or redesign in this stage.
2. **Product runner**: Add one feature-local runner that composes the T027 product benchmark/request path with feature-002 target composition and drafter injection, selects the seven target identities and two immutable drafters, and writes feature-local records. B0 loads the full official Bonsai2 target with native embedding, all blocks, final norm, and LM head as the accepted endpoint control. Keep model/runtime behavior in existing code; do not change production runtime.
3. **Cheap preflight**: Check pinned checkpoint revisions/files, including B-B revision `0059b38aa255698b1a87305eb3fbb5a3cfd616e2` and selected-weight SHA-256 `eb141d0fb8621cf7a7c81047b83810fa19b8093543bd05529e9478d9555cb8f1`; compare relevant reused runtime source hashes with accepted feature-002 evidence; verify T027 input IDs and SHA-256 `8bda1dedde8e022452b7dafaa3677dca47dedc350258c0885c4d1e5fde155f59`; assert a new process and fresh/empty measured request state per cell while preserving normal T027 production prefix-cache initialization. This is a direct preflight, not a gate framework.
4. **Golden canary / matrix cell 1**: Run exactly one fresh H0+B-Q Golden-A cell first. If it remains around 32–36 tok/s / ~49% acceptance, stop before hybrids and save a narrow artifact listing concrete request/runtime differences from Golden A. If it returns to the historical ~45 tok/s product regime, retain that immutable result as matrix cell H0+B-Q and continue with only the remaining 13 cells. Do not rerun it for cardinality.
5. **Mandatory first-pass sweep**: Exactly 14 fresh physical observations total, in this frozen order (each row a new process/cache): H0+B-Q (the canary/cell 1), H0+B-B, H1a+B-B, H1a+B-Q, H1b+B-Q, H1b+B-B, H1c+B-B, H1c+B-Q, H2+B-Q, H2+B-B, H3+B-B, H3+B-Q, B0+B-Q, B0+B-B. Each uses the Golden-A request, plain KV, ordinary production CapController, no WidthPolicy/KV8/tuning, same runtime/machine. The allowed differences are target composition and drafter; B0 is the native full Bonsai2 endpoint control.
6. **Lightweight repeats**: Repeat only a hybrid reaching/beating fresh H0+B-Q, conclusion-critical near-noise cells, or a local maximum that determines whether narrow boundary localization is warranted. Do not automatically repeat all cells.
7. **Final product table**: Show Historical Golden A = 45.779, Historical Golden B = ~45.4 (hot-prefix context), fresh H0+B-Q (the canary/matrix cell 1), and all 14 current cells total. Include target, Bonsai-owned blocks, drafter, serial tok/s, speculative tok/s, delta vs fresh H0+B-Q, speedup, acceptance, mean accepted, target forwards, generated/forward, peak GiB, and notes/status. Plot/arrange B-Q and B-B paths H0 → H1a → H1c → H2 → H3 → B0; show H1b as an interaction diagnostic.

## Reuse and Context Decisions

**Reused unchanged**: Feature-002 declarative hybrid composition/loader, drafter selection/provenance, exact checkpoint manifests, ownership, canonical boundary/no-bridge, Prism-packed donor, DFlash tap, cache advancement/rollback, and accepted target execution/integrity evidence. T027 product runner/request path is authoritative for product measurement and Engine semantics. Feature-002 `run_series_condition` is NOT reused unchanged when that would retain p01/p02/p03, 128-token generation, prefix-cache-off, or other feature-002 request controls. Only small measurement/provenance helpers are eligible for reuse when they leave T027 semantics intact.

**Target distinction**: H0/H1a/H1b/H1c/H2/H3 retain Qwen embedding, final norm, and LM head while replacing only the listed decoder blocks. B0 is the full official Bonsai2 model with its native embedding, every decoder block, final norm, and LM head; it is the accepted feature-002 endpoint control, not a 64-block Qwen hybrid.

**Context only**: Feature-002 Series-A/Series-B performance, including ~32–36 tok/s observations, cannot serve as the feature-003 product baseline. Golden B (~45.4 tok/s) had 25 of 26 prompt tokens hot-cached and is excluded from paired deltas. Golden A is the historical product anchor at 45.7788518 tok/s.

**Smallest feature-local artifacts**: `evidence/probes/product_sweep.py` (T027-path runner plus preflight/canary/matrix orchestration, composing feature-002 targets); `evidence/product-sweep.json` (machine-readable provenance and cell results); `evidence/product-sweep.md` (final table or narrow canary mismatch report). No contracts or production source changes are planned.

## Project Structure

```text
specs/003-product-golden-hybrid-sweep/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
└── evidence/
    ├── probes/product_sweep.py
    ├── product-sweep.json
    └── product-sweep.md
```

**Structure Decision**: Keep this experiment and its runner/evidence feature-local. Reuse runtime modules by import/composition; no production runtime modification or external interface is required.

## Constitution Check After Design

All gates remain passed. The design holds product request/runtime controls fixed, uses accepted correctness and provenance evidence without reopening it, and makes the fresh product canary the explicit execution decision. No unresolved technical choice requires additional research.
