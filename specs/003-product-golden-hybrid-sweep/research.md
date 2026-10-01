# Research: Product Golden Hybrid Sweep

## Decision: Reuse feature-002 construction, not its benchmark request path

**Decision**: Reuse feature-002 `hybrid_target.py` composition/loading, exact checkpoint identities, drafter injection/provenance, accepted integrity/runtime evidence, and only measurement/provenance helpers that do not alter request semantics. The T027 product runner/request path is authoritative for performance: compose it with feature-002 target construction in a feature-local runner.

**Rationale**: The accepted feature-002 `run_series_condition` is tied to the Series-A p01/p02/p03 corpus, `max_new_tokens=128`, and `prefix_cache=false` (also reflected in the Series-B control manifest). Those settings cannot reproduce T027 Golden A. T027 uses the one LRU workload, exact pinned 26 input IDs/digest, thinking disabled, temperature 0, top_p 1, top_k 0, max_tokens 512, production Engine request path, ordinary production prefix-cache initialization, and a fresh/empty measured request state for each cell.

**Alternatives considered**: Call `run_series_condition` unchanged and add a wrapper; rejected because it would preserve incompatible benchmark controls. Reimplement all target/runtime logic; rejected because feature-002 already provides accepted composition, provenance, and runtime machinery. The feature-003 runner must use the T027 product benchmark path plus feature-002 target composition/drafter injection.

## Decision: Treat B0 as the native full-Bonsai2 endpoint

**Decision**: H0/H1a/H1b/H1c/H2/H3 retain Qwen embedding, final norm, and LM head and replace only specified decoder blocks. B0 loads the full official `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` target with its native embedding, all decoder blocks, final norm, and LM head, matching the accepted feature-002 B0 endpoint control.

**Rationale**: B0 is an endpoint control and must not be modeled as a Qwen-owned non-block shell around 64 donor blocks.

## Decision: Count the canary as matrix cell 1 and freeze the execution order

**Decision**: The fresh H0+B-Q canary is the matrix H0+B-Q observation. On pass, retain it and run only 13 more fresh cells, for 14 mandatory physical observations total. Freeze the row order in `plan.md`; keep target pairs adjacent and alternate drafter order. Each row uses a fresh process/cache. A final fresh H0+B-Q drift bracket is optional only when substantial runtime duration/order drift is suspected or it is needed to adjudicate a product-competitive result.

**Rationale**: This avoids an unnecessary duplicate H0+B-Q run while preserving the fresh product baseline as the denominator. The frozen order is an execution sequence, not a gate or authorization framework.

## Decision: Keep the narrow stop-on-canary and repeat policy

If the canary remains around 32–36 tok/s / ~49% acceptance, stop before any hybrid cell and save a narrow mismatch record. If it returns to the historical ~45 tok/s regime, immediately run the remaining 13 matrix rows. Repeat only a hybrid reaching/beating fresh H0+B-Q, conclusion-critical near-noise cells, or a local maximum that determines whether narrow boundary localization is warranted.

## Accepted evidence boundary

Feature-002 remains authoritative for mixed-target construction, ownership, canonical residual boundaries, no explicit bridge, Prism packed donor execution, DFlash taps `[5,19,33,47,61]`, cache advance/rollback, checkpoint and drafter provenance, and successful execution of H0/H1a/H1b/H1c/H2/H3/B0. Its performance values are context only. Feature-002 T040–T044 are excluded.

Feature-001 inspection is limited to the exact T027 product runner and Qwen Golden-A raw result required to recover product request/Engine semantics and historical measurements; no broader archaeology is needed.
