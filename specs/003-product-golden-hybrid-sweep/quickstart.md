# Quickstart: Product Golden Hybrid Sweep

This is an implementation/run guide for the feature-local runner. No benchmark is run during planning.

## Preconditions

1. Use the repository's normal Python/MLX environment on the same Apple Silicon machine used for product measurements.
2. Ensure the exact Qwen, official Bonsai2, B-Q, and B-B checkpoint revisions are locally available. Verify B-B selected weights against the pinned SHA-256 in [data-model.md](data-model.md).
3. Run the runner's model-free preflight. It must report matching runtime source hashes, checkpoint revisions/files, exact T027 prompt IDs/digest, and process/cache freshness behavior before any inference.

## Execution sequence

1. Run exactly one fresh H0+B-Q canary using the authoritative T027 product benchmark/request path, Golden-A request semantics, and normal production prefix-cache initialization. This observation is also matrix cell 1.
2. If the canary remains around 32–36 tok/s / ~49% acceptance, stop. Save the narrow request/runtime mismatch record and do not run hybrid cells.
3. If it returns to the historical ~45 tok/s product regime, retain its immutable result as H0+B-Q and immediately run only the remaining 13 cells. The first-pass matrix is 14 mandatory physical observations total, each in the frozen order below, each in its own fresh process/cache:

   1. H0+B-Q (canary / matrix cell 1)
   2. H0+B-B
   3. H1a+B-B
   4. H1a+B-Q
   5. H1b+B-Q
   6. H1b+B-B
   7. H1c+B-B
   8. H1c+B-Q
   9. H2+B-Q
   10. H2+B-B
   11. H3+B-B
   12. H3+B-Q
   13. B0+B-Q
   14. B0+B-B

   B0 uses the full official Bonsai2 target with its native embedding, all decoder blocks, final norm, and LM head. It is the accepted feature-002 endpoint control.
4. Add repeats only for a hybrid reaching/beating fresh H0+B-Q, conclusion-critical near-noise cells, or a local maximum that determines whether narrow boundary localization is warranted. A final H0+B-Q drift bracket is optional only if substantial runtime duration/order drift is suspected or it is needed to adjudicate a product-competitive result; it is not a mandatory fifteenth matrix cell.
5. Produce the table and machine-readable artifact described in [data-model.md](data-model.md). Separate Golden A and hot-prefix Golden B context rows from fresh-cell deltas.

## Expected outcome

The run ends either with a concrete H0+B-Q mismatch artifact and no hybrid cells, or with all 14 first-pass cells and a report comparing both drafter paths. A passing canary is matrix H0+B-Q and becomes the denominator for every current-cell delta. T027 is authoritative for product performance semantics; feature-002 `run_series_condition` must not be reused unchanged if it would preserve its feature-002 corpus, 128-token generation, prefix-cache-off setting, or other incompatible controls. Feature-002 target composition/drafter machinery remains reusable.

Do not run feature-002 T040–T044 or reopen its accepted integrity/provenance findings as part of this workflow.
