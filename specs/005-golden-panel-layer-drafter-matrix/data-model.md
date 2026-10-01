# Data Model: Golden Panel Layer-Drafter Matrix

## Golden Prompt

- **Identity**: One of P05, P07, P08, P14, P17.
- **Fields**: ID, short label, exact text, text SHA-256, current-tokenizer input IDs and SHA-256, token count, accepted qualification context.
- **Rules**: Exact text is immutable. Preflight regenerates and pins current IDs/digest; every cell for the prompt must match.

## Target Configuration

- **Identity**: H0, H1a, H1b, H1c, H2, H3, B0.
- **Fields**: Target, exact Qwen/Bonsai ownership, explicit Bonsai block indices/count, checkpoint revisions, embedding/final-norm/head ownership.
- **Rules**: Preserve accepted compositions. B0 is native full Bonsai2 including embedding, blocks 0–63, final norm, and LM head.

## Drafter Configuration

- **Identity**: B-Q or B-B.
- **Fields**: Repository, immutable revision, checkpoint identity/path, selected weight SHA-256 where applicable.
- **Rules**: Preserve supplied revisions and B-B selected weight digest.

## Physical Observation (immutable raw)

- **Identity**: Prompt + target + drafter + attempt ID + attempt-local cell identity.
- **Fields**: Prompt/target/drafter identity; settings; generated tokens, speculative tok/s, acceptance, mean accepted, target forwards, generated/forward, rounds, width/cap distributions, request duration; active baseline, request-boundary peak runtime, and peak increment in bytes/GiB; memory boundary/reset/order assertions; process/Engine/request freshness and prefix reuse; machine/runtime and concrete interpreter identity/version; checkpoint revisions; attempt ID; physical-runner SHA; relative/reference identity for the attempt's preflight and its SHA/passing identity; output SHA; status and raw artifact identity.
- **Attempt rules**: Each attempt owns a unique `evidence/attempts/<attempt_id>/` directory containing `attempt.json`, `preflight.json`, `cells/`, and `repeats/`; no physical artifact path is shared across attempts. `attempt.json` records attempt ID, runner SHA, passing preflight SHA/identity, start/status, complete/incomplete/superseded state, expected cell count 70, observed/statused count, and supersession reason when applicable. Its lifecycle status is finalized when the attempt ends; preflight and raw cell/repeat artifacts are immutable once written and are never overwritten. The passing preflight is stored at `attempts/<attempt_id>/preflight.json`; it is not written to a shared feature-level preflight path. A first-pass cell is stored only at `attempts/<attempt_id>/cells/<prompt>/<ordered-cell>.json` and references the preflight in the same directory. A complete attempt contains the exact 70 expected statused first-pass identities, all under one runner SHA matching its passing preflight. The effective first pass for a final matrix is exactly one such complete attempt. The finalizer never combines cells across attempts or runner SHAs. Incomplete/superseded attempt artifacts remain byte-for-byte preserved and visible in the feature-level `evidence/attempts.json` ledger but are excluded from the effective matrix.
- **Rules**: Exactly 70 first-pass cells per attempt. Each uses a fresh process, Engine, and request with stipulated settings. Every first-pass and repeat raw record includes attempt ID and physical-runner SHA. Raw files contain measured/provenance fields only and are never enriched with derived comparisons. New attempts never overwrite prior attempt paths.
- **Memory boundary**: Loading and warmup finish before establishing fresh measured-request state. Reset the MLX peak counter, capture active memory immediately before the measured request, execute only that request, then immediately capture peak. Recorded metadata must establish that the peak belongs to this interval. Peak increment equals peak bytes minus baseline bytes.
- **Failure state**: Failed/incomplete observations remain visible and cannot support claims requiring a complete panel.

## Derived Cell (offline)

- **Identity**: Raw artifact path and SHA.
- **Fields**: Same-prompt H0+B-Q baseline identity/value; throughput delta and retention; peak-memory delta, GiB saved, and memory saving percentage; finalizer SHA.
- **Rules**: Computed only by the offline finalizer after the matching prompt's H0+B-Q exists. Never written back to raw. P07/P14 are normalized only after their late baseline completes.

## Repeat Manifest and Matched Repeat

- **Objective automatic triggers**: (1) any non-H0 cell whose first-pass tok/s is greater than or equal to its prompt's H0+B-Q tok/s; (2) any H1b+B-B cell whose throughput retention is at least 97% of its prompt's H0+B-Q. These are computed mechanically from the effective immutable first pass.
- **Repeat candidate fields**: `repeat-candidates.json` records potential conclusion-critical near-drift comparisons and surprising local target maxima, with all relevant raw identities and metrics. Candidate generation does not decide subjective meaning or select repeats.
- **Explicit adjudication fields**: Optional `repeat-decisions.json` entries select a subjective candidate and contain prompt, target, drafter, triggering first-pass raw artifact identity/SHA, trigger class, and a short explicit reason. If this file is absent or empty, no subjective trigger is selected.
- **Manifest fields**: Effective attempt ID; prompt, target, drafter; triggering first-pass raw artifact path/identity/SHA; trigger class/reason; and comparison identity. The finalizer combines objective automatic triggers with explicit adjudication decisions into immutable feature-level `repeat-manifest.json`; it invents no hidden thresholds or heuristics. At most one initial repeat per comparison. The frozen physical runner consumes the manifest and writes only to `evidence/attempts/<effective_attempt_id>/repeats/<comparison>.json`, executing only listed fresh matched observations with unchanged physical settings. Each repeat raw record includes the effective first-pass attempt ID and runner SHA. If a repeat reveals a physical defect requiring a runner source change, preserve the evidence and mark the repeat unresolved unless a new full physical attempt is explicitly authorized; do not mix new-SHA repeats into the effective matrix.
- **Repeat fields**: Manifest entry identity, fresh physical metrics/provenance, raw artifact identity, and outcome. Validation is offline.

## Validation and Panel Report

- **Validation fields**: Attempt-directory discovery and global ledger; expected/observed identities and order; missing, duplicate, or failed status; raw schema and required memory fields; freshness/settings/revisions; exactly one effective complete 70-cell attempt and one physical runner SHA matching its passing preflight in the same attempt directory; all raw cell preflight references/SHA values matching that preflight; raw output SHA; interpreter/runtime pin; memory-boundary assertions; same-prompt derivation arithmetic; objective repeat triggers; explicit repeat decisions; repeat-manifest triggers and attempt-scoped repeat provenance requiring the effective first-pass SHA; effective attempt ID in every derived/report artifact; finalizer SHA.
- **Report fields**: Five required tables, per-prompt B-Q/B-B paths, H1a/H1b/H1c interaction, robustness summaries, repeat ledger, and descriptive Pareto view.
- **Rules**: Offline finalizer performs no inference. Prompt cells precede summaries; no weighted score, hidden aggregate, or automatic causal inference.
