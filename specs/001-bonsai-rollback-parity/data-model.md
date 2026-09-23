# Data Model: Rollback Parity Observations

This is an internal diagnostic schema for comparing one logical speculative run, width-specific within-runtime references, and optional cross-runtime observations. It does not add a product API. The same-width S=8 verify reference isolates verify/capture from rollback; the ordinary S=1 committed reference defines semantic state and subsequent committed behavior. A fresh same-runtime S=2 committed-prefix reference controls the accepted=1/keep=2 sequence-width effect.

## ControlledRun

Identifies the comparison conditions shared by all traces.

| Field | Meaning | Equality rule |
|---|---|---|
| `implementation` | Chad/mlx-dspark speculative run, same-width verify-reference run, or committed-semantic-reference run | Exact label, including runtime and reference kind |
| `target_id` / `target_fingerprint` | Target repo/path and local checkpoint fingerprint | Exact |
| `drafter_id` / `drafter_fingerprint` | Exact physical Chad-built sidecar and local fingerprint | Exact |
| `tokenizer_fingerprint` | Tokenizer files/config used to produce IDs | Exact |
| `runtime` | Python, MLX, mlx-lm, OS, device, model config and active kernel flags | Exact within each speculative-versus-serial comparison. Record both profiles for cross-runtime observations; known profile differences do not by themselves invalidate token/top-k observations or make raw numeric equality mandatory. |
| `mlx_lm_source_hashes` | SHA-256 for installed `qwen3_5.py`, `gated_delta.py`, and `cache.py` | Record per runtime; expected hashes are in [research.md](research.md). |
| `gdn_callable` | Actual callable module/file/line/signature and identity result for `qwen3_5.gated_delta_update is gated_delta.gated_delta_update` | Record in each runtime; verified identity is true in both. Capture actual call path/flags as run metadata. |
| `target_model_type` / `target_loader` | Physical model config and selected Prism loader | Expected `prism_hadamard_qwen35` through both Prism loaders. |
| `chad_fastpath` | `Engine.fastpath`, `CHAD_NO_FASTPATH`, installed class/call-path identity | Required in every Chad model-backed diagnostic; record actual state, do not infer from source. |
| `prompt_ids` | Tokenized prompt | Exact |
| `sampling` | Temperature, top-p, top-k, seed and greedy/sampled mode | Exact |
| `prefix_state_id` | Identity of the prefill/cache snapshot from which branches start | Exact among branches within one runtime; cross-runtime runs record corresponding logical prefill conditions without requiring bit-identical cache tensors |
| `target_sequence_width` | Number of target rows: 8 for speculative and same-width verify-reference runs; 1 per call in the ordinary committed path | Exact for the reference being compared |
| `anchor_or_pending_id` | Current target token at the proposal/verify boundary | Exact |
| `draft_ids` | All seven candidate token IDs in a width-8 round | Record exactly; cross-runtime differences may be diagnostic under Gate A's local-reference rule |
| `verify_ids` | Exact target input sequence `[anchor] + draft` | Exact within each runtime and its S=8 reference, captured before verify; cross-runtime equality is not required when valid proposals differ |
| `proposal_context_logical_position` | Logical generated-token/context position used to produce proposals | Exact semantic position; record before verify |
| `accepted_count` | Number of accepted draft tokens | Capture only after anchor, draft IDs, verify IDs and proposal context position are recorded |
| `tap_ids` | Target layers whose residuals feed DFlash | Exact and same order |

For a committed-semantic-reference run, `draft_ids` and `verify_ids` identify the associated speculative round; the ordinary S=1 path consumes only the committed prefix through one-token calls.

Matching draft width `k` and matching accepted count do not establish equal `verify_ids`. Gate A requires exact cross-runtime equality only for semantic/logical invariants: target, sidecar, and tokenizer identities/fingerprints; prompt and committed-prefix IDs; generated-token position; anchor/pending ID; target/drafter live lengths, offsets, and absolute positions; proposal-context position and row count; tap identities/order/shapes; and cache family/precision/semantic structure. Equivalent semantic cache layouts need not use identical wrapper names. Raw cross-runtime numeric cache/recurrent/convolution/hidden/fused/tap/context values are diagnostic under distinct valid projection/op graphs. Record each runtime's proposal IDs exactly and assert its own `verify_ids = [anchor] + draft`. A cross-runtime proposal difference does not fail A when exact invariants pass and independent same-runtime R1 reconstructions in both runtimes reproduce the production context exactly (or satisfy their predeclared applicable stable five-run controls) and proposals exactly. A failed exact semantic/logical invariant stops at that boundary; a violation of mlx-dspark's own applicable independent same-runtime semantic/context/proposal reference may fail A for localization. Equal committed outputs alone cannot establish A. Fixed7 performance and forward counts are separate control metadata.

## ProposalInputRecord

The early pre-verify record for one logical generated-token position. Capture fields in this order:

1. Equivalent target/drafter input-state identity and logical generated-token position.
2. Current anchor/pending token ID.
3. All seven proposed draft token IDs.
4. Exact `verify_ids = [anchor] + draft`.
5. Proposal-context logical position.
6. Accepted count, only after fields 1–5 are immutable.

The record carries separate Chad and mlx-dspark values. Gate A records each required semantic/logical field and classifies the result as equivalent, inconclusive, control mismatch, earliest exact input-state difference, same-runtime reference violation, or exact-equivalent controls with an implementation-dependent proposal diagnostic. Store the diagnostic's two proposal lists and verify lists even on A PASS, along with both local reference observation IDs, context equality/control status, graph identity, and routing. Cross-runtime raw numeric state samples remain diagnostic. A missing applicable local reference/control makes a differing proposal inconclusive, never a pass or an mlx-dspark defect.

## BoundarySnapshot

An immutable observation of one `ControlledRun` at a semantic point. Each snapshot contains a logical round, boundary label, token position, and the state listed below. Materialize/copy MLX arrays at capture time so mutable caches and lazy evaluation cannot change earlier observations.

### Target input and cache state

- Exact token IDs entering the operation, including anchor and proposal positions.
- Per-layer cache class, layer index, array shapes, dtypes, and whether the cache is trimmable.
- Attention KV live keys/values plus live length, `offset`, and derived absolute positions.
- GDN convolution-history array and recurrent state array from the two `ArraysCache` slots.
- Mask, sequence lengths, left-padding metadata, and the order that recurrence caches map to model layers.
- Tap IDs and incoming drafter context/cache positions where relevant.

Within each runtime's speculative-versus-same-width verify comparison, cache type, shape, layer order, live length, offset, positions, and token IDs are exact invariants. Compare captured S=8 outputs/taps/cache effects before reconciliation. At Gate C, cache type, shape, layer order, live length, offsets, positions, committed/accepted-prefix IDs, and required tap/fused-row identity/order remain exact. Numeric rollback-owned state is compared with an independent accepted-prefix reconstruction from the frozen same-runtime S=8 B2 capture: recurrence replay from captured S=8 inputs and pre-round state, convolution history from captured S=8 `conv_input`, and attention KV trimmed from frozen S=8 KV. Fresh S=1/S=2 raw state is a serial/cross-width diagnostic rather than this numeric oracle. Across runtimes, raw cache/tensor values remain diagnostic. Compare only live KV ranges; stale backing-buffer tails after trim are not logical state.

### Verify capture and output

For each GDN layer and verify token row:

- Pre-round recurrent state passed into gated-delta update.
- `q`, `k`, `v`, `a`, `b`, `A_log`, `dt_bias`, `mask`, and `use_kernel` passed to the update.
- Convolution input, pre-round convolution window, and resulting conv cache window.
- Verify logits, top-2 token IDs and values, top-2 margin, and fused tapped hidden rows in tap-ID order.

The function object/module path and device dispatch are metadata for the run. Callable identity is verified true in both environments, but same callable identity does not establish equal inputs, flags, projection/op graphs, or logits. Chad and mlx-dspark use different target projection/op paths for this Prism target. The speculative branch and same-width reference must use the exact same eight verify IDs and equivalent pre-verify target state within one runtime; the S=1 committed path is not a raw pre-reconciliation logits oracle.

## ChadFastpathRecord

Required for the model-backed Chad probe:

- `Engine.fastpath` value for the loaded model.
- `CHAD_NO_FASTPATH` environment state.
- Installed fastpath class identity and relevant install/call-path identity.
- Whether model projections/compiled paths were actually replaced or fused in this run.

Run normal-fastpath and `CHAD_NO_FASTPATH=1` diagnostics in separate fresh processes, loading a new target model for each branch with the environment set before `mlx_fastpath.install()` mutates the model. Never toggle the variable after installation and treat that already loaded model as an independent control. Restore equivalent target weights, input IDs, sampling settings, and cache snapshot for the two branches. Compare target margin/logits and captured projection inputs as an op-graph sensitivity diagnostic. Do not use branch throughput as correctness evidence.

### Reconciliation result

For each `accepted` count:

- `accepted_prefix_ids`, including the anchor separately from accepted draft IDs.
- `n_rejected`, verify width, and computed `keep = accepted + 1`.
- Post-reconcile attention KV live ranges and offsets.
- Post-reconcile recurrent states and convolution windows.
- Corresponding ordinary S=1 committed-semantic reference within the same runtime, retained for serial behavior and Gate D rather than as the raw numeric cache-state oracle for S=8-originating rollback state.
- For accepted=1/keep=2, retain a fresh same-runtime S=2 committed-prefix execution for exactly `[anchor, first accepted draft]` as a width-sensitivity/serial diagnostic, independently allocated from speculative and S=1 branches.
- For each recurrent, convolution, and numeric live-KV component, record dtype, execution shape, metric, frozen S=8 reference root, and the independent accepted-prefix reconstruction result. Recurrence is replayed from captured S=8 inputs and pre-round state, convolution history is sliced from captured S=8 `conv_input`, and attention KV is trimmed from frozen S=8 KV. Use exact equality when deterministic; otherwise record exactly five same-path controls under the standard calibration/validation rule. Record fresh S=1/S=2 and S=2-versus-S=1+S=1 distances separately as cross-width diagnostics, not as the Gate C numeric failure threshold.
- For every required post-reconcile hidden-state tap and accepted fused row in the committed prefix, record exact token/row identity, tap identity, shape, sequence position, layer/tap order, and prefix membership. Compare numeric values against the corresponding row captured by the independent same-runtime same-width S=8 B2 reference, using the exact same `verify_ids` and equivalent pre-verify state. Exactly five equivalent same-width S=8 controls establish a bound per exact tap/row, component, dtype, and predeclared metric under the rule below. Do not derive this bound from fresh S=1/S=2 projection-width differences. Record the ordinary S=1 semantic reference separately; it is not the numeric oracle for an accepted fused row originating in S=8 verify.

The required cases are width 8 / seven draft IDs / accepted=0 / keep=1, and width 8 / seven draft IDs / accepted=1 / keep=2.

### Drafter context state

- Accepted fused target rows selected for the next context.
- Logical context row positions and absolute positions.
- Projected context rows if directly available.
- Per-layer drafter KV contents, cache type, live length, and absolute `offset` after accepted rows are consumed.
- Next proposal token IDs and selector candidate identities if the next draft is part of the comparison.

Compare equivalent logical context after accounting for Chad's pending-row staging and mlx-dspark's accepted-row append timing.

## SameWidthVerifyReference

One speculative S=8 verify/capture transition and a fresh/cloned target execution in the same runtime. The reference starts from equivalent pre-verify target state, receives the exact same `verify_ids = [anchor] + seven draft IDs`, executes the same eight-row sequence width, and captures the same target outputs, taps, and cache effects before reconciliation. It uses independently allocated target/cache state and does not use the speculative rollback/reconcile result or its mutated cache as a reference. This isolates verify/capture/projection behavior from rollback. If this comparison differs, localize that behavior before investigating reconciliation.

## CommittedSemanticReference

An independently allocated ordinary S=1 committed path in the same runtime, device, and weights. For accepted=0 it executes prompt + anchor; for accepted=1 it executes prompt + anchor + first accepted draft as two one-token calls. It defines serial semantic/behavior expectations and supports Gate D. Because projection execution width differs, its raw cache values are not the numeric Gate C oracle for state originating in an S=8 verify.

## CrossWidthCommittedPrefixDiagnostic

For accepted=1/keep=2, independently allocate a fresh target/cache from equivalent logical pre-round state and execute one S=2 call for exactly `[anchor, first accepted draft]`. Capture its state as a projection-width/serial diagnostic. It does not replace the ordinary S=1 + S=1 semantic reference, the frozen S=8 B2 verify reference, or the Gate C accepted-prefix reconstruction derived from that S=8 capture.

## NumericalControlPair

Keep controls width-specific and label each pair's role:

- Repeated S=1 committed-vs-committed executions establish serial-reference repeatability for Gate D and width-sensitivity diagnostics.
- Repeated S=2 committed-prefix-vs-committed-prefix executions establish repeatability of the S=2 cross-width diagnostic.
- Repeated fresh S=2-versus-ordinary S=1 + S=1 pairs establish the expected projection-width delta as a diagnostic; this envelope is not the Gate C numeric cache-state bound for S=8-originating state.
- Repeated same-width S=8 executions with identical verify IDs and equivalent pre-verify state establish repeatability of the B2 verify reference.
- Repeated equivalent same-width S=8 executions with identical verify IDs and equivalent pre-verify state establish repeatability separately for each required Gate C tap/accepted fused row, component, dtype, and predeclared metric.

No global tolerance field exists. Gate C identifies each rollback-owned numeric component by runtime, frozen S=8 root, layer/cache slot, dtype, execution shape, and metric. Independently reconstruct the accepted prefix from that frozen S=8 capture and require exact equality where deterministic. If a same-path component is not bit-repeatable, use exactly five equivalent same-path controls: repetitions 1–3 calibrate the maximum bound and repetitions 4–5 validate it; otherwise the observation is INCONCLUSIVE. No bound transfers across roots, widths, components, rows, dtypes, runtimes, logical states, or execution shapes. Required accepted tap/fused rows use their corresponding same-width S=8 B2 rows and same-path controls. Fresh S=1/S=2 controls and S=2-versus-S=1+S=1 deltas remain serial/cross-width diagnostics and cannot by themselves fail Gate C. Gate D independently checks subsequent committed behavior against the serial committed reference.

For numeric-only Gate A/E defects, record semantic relevance and evidence that the observation is the earliest controlled mlx-dspark defect, not merely a cross-runtime numeric difference. Record exact drafter/context component, logical position, dtype, and predeclared distance metric. Before examining the failing sample, use exactly five measured equivalent same-runtime control repetitions under identical runtime, device, weights, input, logical-state, and applicable execution-shape conditions for that exact component/row, dtype, and metric. Repetitions 1–3 define the calibration bound as the maximum observed control distance; repetitions 4–5 validate it, and the bound is STABLE only when both validation distances are ≤ calibration bound. Otherwise it is UNSTABLE and the dependent gate is INCONCLUSIVE. Record all five observations and status first. Do not widen the bound, add runs, or adapt the rule after seeing the failing sample. A zero calibration bound is valid, but either validation distance above zero makes it UNSTABLE. The pre-fix mismatch must reproducibly exceed a stable bound; after an applicable fix, the same asserted invariant must fall within it. No global tolerance or transfer across components, rows, dtypes, widths, runtimes, logical states, or execution shapes is permitted. If no stable applicable bound can be established, mark evidence INCONCLUSIVE and do not authorize a production patch. Exact IDs, shapes, offsets, positions, and logical-row invariants remain exact. A raw Chad-versus-mlx numeric difference by itself never qualifies.

## CrossRuntimeDiagnostic

Optionally after B2, compare Chad and mlx-dspark token identities, top-k, margins, q/k/v/a/b, and other tensors. Record raw numeric comparisons as diagnostics; cross-runtime raw logits and margins need not be exactly equal, including when the runtimes use different valid projection/op graphs or sequence-width kernel shapes. Discrete token/ID/structural invariants remain exact where their semantic contract requires equality. B3 has no dependency edge into C.

## GoldenBenchmarkCase

The separate established final hardware comparison:

| Field | Required value |
|---|---|
| Prompt | `Write a production-quality Python LRU cache with tests and type hints.` |
| Thinking | Disabled |
| Temperature | 0 |
| `max_tokens` | 512 |
| Drafter | Same exact physical Chad-built sidecar in both runs, verified by fingerprint |

Keep this result separate from the rollback boundary probe. The generic benchmark CLI may be an additional control but cannot replace this case. Record measured output/token identity and run metadata; do not fabricate an expected output.

## GateResult

One row per gate A, B1, B2, optional B3, C, D, or E: earliest observed difference (or equivalent), observation IDs, comparison type, exact/controlled numeric/diagnostic rule, result (`pass`, `fail`, `inconclusive`), next permitted investigation layer, and stop/continue decision. A separately records cross-runtime exact semantic/logical checks, runtime-specific numeric context, exact proposal/verify IDs, independent same-runtime R1 reference checks and applicable control status, plus any implementation-dependent proposal diagnostic. Its failures distinguish unmatched controls, an earlier exact semantic/logical difference, and an applicable same-runtime reference violation; diagnostic proposal differences are not failures. A later correctness gate cannot override an earlier correctness-gate failure. B2 is the same-runtime S=8 verify/capture comparison. B3 is diagnostic and non-blocking, with no edge into C. C records component-level keep=1 and A/B/C keep=2 comparisons and passes only under the exact and controlled bounds above. A matching next token alone cannot pass C.

## PortableFixtureScope

The existing `_tiny_hybrid()` fixture uses stock mlx-lm Qwen3.5 and does not execute the physical Prism/Chad-fastpath target graph. Use it only to validate generic rollback-state invariants. A passing tiny-hybrid test is not a reproduction of the physical Bonsai divergence.
