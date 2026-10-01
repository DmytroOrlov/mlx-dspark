# Research: Qwen-Bonsai Hybrid Target Experiment

**Scope**: Resolve repository and runtime facts needed to plan feature 002. No model benchmark, target load, runtime code edit, or constitution edit was performed for this research.

## 1. Target model structure and load paths

### Decision

Build H0–H3 by replacing same-index `DecoderLayer` objects in the Qwen `qwen3_5` model. The correct Bonsai2 checkpoint is `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`, whose Prism `model_type` routes through this repository’s existing Prism loader. The exact checkpoint is not the similarly named `prism-ml/Ternary-Bonsai-27B-mlx-2bit` currently listed in `REGISTRY`; that registry row’s own comment identifies it as the older Qwen3.6 sibling. Use the Bonsai2 repo explicitly with the original DFlash drafter explicit, and correct the mistaken model id in the spec assumption before runtime execution.

### Evidence

- `src/mlx_dspark/load.py:898-923` routes text-capable `qwen3_5` configs to mlx-lm. `load_target` at `src/mlx_dspark/load.py:1024-1136` resolves the checkpoint, parses `config.json`, loads ordinary Qwen through `mlx_lm.load`, and wraps the result as `Target`.
- The Bonsai `model_type=prism_hadamard_qwen35` is detected at `src/mlx_dspark/load.py:1071`. That route calls `prism_pack.load`, then wraps the model in `Target` and installs the existing Prism MMA route at `src/mlx_dspark/load.py:1118-1136`.
- `src/mlx_dspark/load.py` has a `REGISTRY` row for `prism-ml/Ternary-Bonsai-27B-mlx-2bit` (`ternary-bonsai-27b` near lines 125-145), but that row describes the Qwen3.6 sibling, not Bonsai2. The official Bonsai2 27B checkpoint is `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`. Its official config declares `schema_version=2`, `model_type=prism_hadamard_qwen35`, 64 layers, hidden 5120, intermediate 17408, vocabulary 248320, and the same every-fourth-full-attention layout. Its module inventory includes the same Qwen3.5 `model.layers.N.*` decoder paths and 2-bit affine g128 metadata. The repository’s `load_target` routes this Prism model type through `prism_pack.load`; callers must pass the exact Bonsai2 repo explicitly because the current registry preset is stale for this experiment. Official checkpoint/config evidence is [the Bonsai2 config](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit/blob/3f926b415992eaa2ae9dd7b573706494d6bbf787/config.json) and [the Bonsai2 model card](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit).
- The inspected Qwen local snapshot has 64 layers, hidden size 5120, intermediate size 17408, vocabulary size 248320, and `full_attention_interval=4`. The official Prism Bonsai2 pack has the same dimensions and layer family with 2-bit affine g128 projection records. The project evidence also records the shared 64-layer/5120-hidden family in `specs/001-bonsai-rollback-parity/research.md` and `evidence/controls.md`.
- The current local feature-001 hardware evidence used a `nathansutton/Qwen3.8-27B-Ternary-Bonsai-2-DFlash2-MLX` repack; its `config.json` and weights were loaded through this same Prism model class. It corroborates actual runtime structure but must not replace the exact official Bonsai2 repo in feature 002.
- Installed `mlx-lm` 0.31.3 source at `.venv/lib/python3.12/site-packages/mlx_lm/models/qwen3_5.py:209-240` defines `DecoderLayer.__call__(x, mask=None, cache=None)`. `Qwen3_5TextModel` owns `embed_tokens`, a Python list of 64 `DecoderLayer`s, and `norm` (`:243-279`). `TextModel` owns that inner model and, when embeddings are untied, `lm_head` (`:282-315`). `Model.layers` returns `language_model.model.layers` (`:132-134` and the property later in the file).
- Both model routes use this exact Python `DecoderLayer` class. In Qwen its projections are mlx-lm quantized linear modules. In Prism `prism_pack.load` (`src/mlx_dspark/prism_pack.py:126-225`) constructs `qwen3_5.Model`, reads the pack, creates `_packed_class()` modules from the pack module inventory, assigns them by dotted path, and loads weights strictly. `Packed` exposes `__call__`, `bits`, `group_size`, and `mode` for the current fused paths.

### Alternatives considered

- Separate H1/H2/H3 model classes: rejected. Explicit ownership is the experiment parameter; named classes duplicate policy and block extension.
- Treat packed Bonsai projection weights as ordinary Qwen weights: rejected. `prism_pack.py` documents that ordinary loading silently skips rotations and gives plausible but invalid output.
- Replace only selected projections inside Qwen blocks: rejected by scope. The required axis is full transformer-block ownership.

### Revision caveat

The Qwen snapshot available to feature-001 evidence is revision `10c35caafbb80f7dc6a7a432cdd11af10a6d4818`. The official Bonsai2 revision for the inspected config is `3f926b415992eaa2ae9dd7b573706494d6bbf787`; its selected model-weight fingerprint must still be recorded from the actual feature-002 snapshot. The exact model identity in the spec assumption must be corrected before runtime execution.

## 2. Prism/Hadamard representation boundary

### Decision

Direct Qwen→Bonsai and Bonsai→Qwen block boundaries are valid at the residual stream. No boundary bridge is needed for the current 2-bit Prism loader and path.

### Evidence

- `prism_pack.py` module docstring (`:1-31`) describes the packed basis and says the runtime transform is applied to activations at each projection; the embedding lookup receives the inverse transform.
- `rotate` (`:94-116`) transforms only the passed activation. `Packed.__call__` (`:146-160`) applies `rotate(x, ...)` immediately before the packed linear call and returns the linear result directly. There is no inverse transform on projection output and no persistent model-level basis state.
- `prism_pack.load` (`:173-225`) installs `Packed` only at the pack-declared projection paths. It shares sign vectors for local projection fusion; this does not change block input/output representation.
- The installed `DecoderLayer.__call__` (`qwen3_5.py:228-240`) consumes the ordinary residual `x`, adds attention/GatedDeltaNet output, then adds MLP output and returns the resulting residual. `Target._body_hybrid` (`src/mlx_dspark/target.py:154-175`) feeds each returned `h` directly into the next layer. The Prism feature-001 evidence records this loader/model family as the working production target path (`specs/001-bonsai-rollback-parity/evidence/controls.md:16`, `research.md` “Prism and quantized target path”).
- The pack embedding handling is distinct from block substitution: `Packed` can inverse-rotate an embedding result, but H0–H3 retain Qwen’s embedding. B0 uses the full Bonsai model and its matching embedding, as specified.

### Alternatives considered

- Insert a bridge at each ownership transition: rejected because transforms are projection-local and block outputs are canonical residuals. A bridge would incorrectly rotate an already canonical stream.
- Reimplement Prism/Hadamard operations in a hybrid module: rejected. The existing pack loader, `Packed`, fusion paths, and `mlx_qmm_mma` remain authoritative.

### Boundary acceptance test

The implementation smoke should call one representative Qwen→Bonsai and Bonsai→Qwen pair at the actual boundary positions used in H1c/H2 and prove finite `[batch, sequence, 5120]` residual outputs with expected dtype. It should also inspect that the selected donor projections are `prism_pack.is_packed(...)`. This verifies the configured modules and canonical interface without requiring hybrid logits to match vanilla Qwen.

## 3. DFlash target tap semantics

### Decision

The current DFlash IDs are zero-based decoder-block output taps. They are captured after block N, not before it and not at N+1.

### Evidence

- The actual installed target path is `Target._body_hybrid` in `src/mlx_dspark/target.py:154-175`: it enumerates `mm.layers`, assigns `h = layer(...)`, and only then appends `h` when `i in tapset`. `_body_mlxlm` uses the same post-layer order for dense models (`:114-129`). `verify()` and `prefill()` route through these bodies.
- `dflash_generate` reads `cfg.target_layer_ids` at `src/mlx_dspark/generate.py:658`; speculative verify uses `target_model.verify(..., tap)` and passes the fused captured rows to the drafter (`:1380-1444`). Thus the captured values are consumed as target hidden-state features by the original drafter.
- The actual cached configs inspected for both `incoai/Qwen3.8-27B-DFlash2` and `naklitechie/Qwen3.8-27B-DFlash2-ternary-bonsai2` declare `[5, 19, 33, 47, 61]`. Feature-001 result manifests independently record this list (`specs/001-bonsai-rollback-parity/evidence/t027-results/qwen-lru.json:94-101` and `bonsai-lru.json:74-81`). The runtime revision and loaded checkpoint config remain authoritative during each run.

For each H-family variant, ownership intersects those exact tapped block indices as follows:

| Target | Bonsai-owned blocks | Affected DFlash block-output taps |
| --- | --- | ---: |
| H0 | none | 0 |
| H1a | 63 | 0 |
| H1b | 62 | 0 |
| H1c | 62–63 | 0 |
| H2 | 60–63 | 1 (61) |
| H3 | 56–63 | 1 (61) |
| B0 | full Bonsai target, 0–63 | 5 (5, 19, 33, 47, 61) |

Tap count does not imply that untapped changed blocks cannot alter later tapped activations; it counts the tapped hidden-state slots whose owning block changed, as FR-008/009 require. H0–H3 remain the one-axis Qwen-boundary series. B0 is the distinct full-Bonsai endpoint control.

### Alternatives considered

- Treat a tap ID as block input: rejected by the actual append-after-call code.
- Assume the old list forever: rejected. Record the loaded drafter config, runtime source revision, tap ids, and recomputed ownership intersection in each run manifest; reject an unexpected tap list until the spec/runtime selection is reconciled.

## 4. Minimal hybrid loading design and memory

### Decision

Use donor-first temporary full loading followed by pruning, then load Qwen and replace explicit layer indices before `Target` construction. Do not hold both complete 27B targets simultaneously. Keep the composition interface generic (`donor_block_indices: sorted unique list[int]`).

### Evidence and sequence

1. Resolve and fingerprint the Qwen and exact official Bonsai2 Prism snapshots; pass the Bonsai2 repo id explicitly because the registry alias refers to the older sibling.
2. Call the existing Prism route (`prism_pack.load` via a narrow loader helper) while no Qwen target is resident. Preserve references only to Bonsai `model.layers[i]` requested by the composition. Drop unselected layers, Bonsai embedding/norm/head and the full donor wrapper; evaluate retained module parameters and clear reclaimable MLX cache.
3. Load the Qwen target and tokenizer using the ordinary mlx-lm path.
4. Replace `model.layers[i]` in the still-unwrapped Qwen model with the selected donor module for every explicit index. Validate same-index block types, shapes, dtypes, and GatedDeltaNet/full-attention position before replacement. Apply no replacement at unlisted indices.
5. Construct `Target` after replacement so its hybrid/cache discovery and recurrent capture hooks bind the final module objects. Run existing tap verification and call existing `mlx_qmm_mma.install` on the composed model so retained donor `Packed` projections take the supported Prism route.
6. Load the original DFlash2 drafter with its pinned revision, bind it as the existing code does, and run `Engine.generate`.

`prism_pack.load` currently calls `mx.load` on the pack’s `model.safetensors`, then deletes the returned full dictionary after converting/configuring weights (`src/mlx_dspark/prism_pack.py:165-174`). The feature-001 product A/B measured peak memory around 17.1 GB for Qwen and 9.4 GB for full Bonsai (`specs/001-bonsai-rollback-parity/evidence/t027-product-ab-adjudication-20260924.json:45,101`). These observations make loading the complete donor alongside Qwen unsafe, while sequential full donor loading fits the observed standalone range. The final hybrid retains only suffix blocks; the first all-target smoke records peak memory and fails closed if loading or steady-state memory is unsafe.

### Alternatives considered

- Load both complete targets and select afterward: rejected; observed full-model peaks sum beyond a safe single-machine budget.
- Implement selective safetensors reads now: deferred. The existing Prism loader is proven and already handles packed module inventory, sign vectors, dtype normalization, strict loading and kernel-compatible modules. Sequential loading is the smallest path. Add selective reads only if real smoke evidence shows donor-first load/prune/release plus final composition cannot fit or does not reclaim memory.
- Deserialize Bonsai weights into a Qwen layer or convert Prism weights: rejected; invalid representation/math path.
- Separate hardcoded H1a/H1b/H1c/H2/H3 classes: rejected; use explicit owner list.

## 5. Ownership and provenance safety

### Decision

Emit a deterministic composition manifest before accepting the runtime and assert ownership against live module identities immediately after composition.

### Manifest fields

- schema version, run id, feature id, timestamp, runtime git SHA/dirty state, source-file fingerprints, Python/MLX/mlx-lm versions, machine identity;
- Qwen repo/path, resolved snapshot revision, `config.json` digest and safetensors file/blob fingerprints;
- Bonsai repo/path, resolved snapshot revision, config digest and pack weights fingerprint;
- composition id and sorted donor indices; explicit owner for every block index 0–63; embedding, final norm, and LM-head owner;
- loaded DFlash repo/path, snapshot revision, config and weights fingerprint, `target_layer_ids`, plus its binding mode;
- model dimensions, per-index layer family, shape/dtype records and integrity status.

For H0–H3, manifest declares embedding/norm/head=`qwen`; unlisted blocks=`qwen`; listed blocks=`bonsai`. For B0 the owners are `bonsai` for all roles and all blocks. The same expanded owner array is used by runtime assertions and emitted JSON, rather than independently maintained named-variant metadata.

After model composition, compare each target block object with the retained donor reference for the requested set and the pre-composition Qwen reference for its complement. Check expected Prism `Packed` modules are present within each donor block; selected donor and unselected Qwen block family/parameter shapes match the manifest. Fingerprint the source snapshots so the reference objects are tied to identified files. A module type assertion alone is insufficient because parent `DecoderLayer` classes are shared.

### Alternatives considered

- Record only named variant string: rejected; it cannot prove actual per-index ownership.
- Hash every runtime tensor independently: unnecessary at first. Snapshot/config/weight fingerprints plus exact source-module identity maps and per-module loaded key/type/shape checks are the minimal deterministic proof. Run-level hashes can be added only if those checks expose ambiguity.

## 6. Runtime integrity

### Required gates before performance use

1. Both snapshots resolve to pinned revisions and expected model families; all target, donor, and drafter loading succeeds.
2. Ownership manifest matches live module identity at every index and correct embedding/norm/head objects.
3. Same-index attention/GatedDeltaNet kind, required tensor shapes, and parameter dtypes are valid. Run `mx.isfinite` over composed model parameters and reject any NaN/Inf.
4. Same explicit composition produces the same owner manifest and module inventory. Donor projections use the existing Prism `Packed` path/MMA routing; no converted ordinary-linear fallback.
5. Existing `Target.verify_tap()` proves the composed target wrapper’s replicated forward agrees with its own `model` forward where that standard probe applies; a tapped-vs-plain forward must return finite outputs with expected hidden width/dtype.
6. Tiny controlled speculative generation exercises cache advance, at least one accepted prefix and one rollback/rejected suffix through existing `verify`/`rollback`; assert valid cache lengths, finite logits/taps, and complete generation. This is a hybrid runtime smoke, not a re-run of feature-001 rollback parity investigation. If and only if this composition reveals new defect evidence, record that evidence and plan the smallest directly implicated integrity follow-up.
7. H0 runs with no composition option through the existing loading/generation route. Compare same-revision, same-settings H0 output IDs/behavior against ordinary Qwen H0 for unchanged-path validation. Hybrid logits/top-1 need not equal Qwen.
8. Repeat a same-config composition/load in a fresh process and compare manifests/ownership; record expected deterministic floating output diagnostic, not an equality-to-Qwen oracle.

No feature-001 zero/partial acceptance matrix, cross-runtime Chad parity, historical divergence trace, or rollback numeric oracle is imported into these gates.

## 7. Benchmark harness and feature-001 reuse

### Decision

Use `Engine.load`/`Engine.generate` for the single-request production Engine path, with its existing round stats, plus an isolated warmed serial target pass using the existing mlx-lm stream or `Target.plain` path. A feature-local runner generalizes stable T027 measurements and writes a fresh result directory per run. Do not change `generate.py` to collect new per-call timers.

### Reuse classification

| Feature-001 item | Classification | Feature-002 use |
| --- | --- | --- |
| `evidence/probes/t027_product_ab.py` model-free preflight, prompt/tokenizer consistency, source/checkpoint fingerprints, Engine request capture, plain-KV assertion, round/width stats, and serial/speculative memory measurements | **Reusable after small generalization** | Replace fixed physical Qwen/Bonsai choices with explicit target composition and exact Series-A matrix; keep only feature-002 outputs. |
| `evidence/probes/bonsai_golden_perf.py` environment/version/git-source/hardware capture, MLX peak-memory query, seeded request, and existing `on_round`/metrics patterns | **Reusable after small generalization** | Port small independent helpers to the new feature runner; remove hardcoded target/Chad setup and one-workload assumptions. |
| `evidence/t027-product-ab-adjudication-20260924.json`, `evidence/t027-results/*.json`, T026/T027 hardware throughput/memory numbers | **Historical evidence only** | Explain why mixed residency is unsafe and provide loader/runtime provenance leads. Never substitute for fresh feature-002 H0 or any Series-A result. |
| `tests/test_bonsai.py` small fixture/cache-rollback invariants | **Reusable unchanged** | Existing runtime behavior may be invoked as existing checks; do not copy feature-001 rollback investigation into feature 002. New coverage focuses on actual mixed block ownership and a narrow hybrid smoke. |
| `evidence/probes/bonsai_rollback_probe.py`, `bonsai_same_s8_rollback_oracle.py`, `bonsai_probe_support.py`, T016/T026 causal traces, Chad comparison adapters, and Gate A–D rollback evidence | **Obsolete/dead-end and not to be reused** | They answer a different parity investigation and are explicitly outside this feature. Keep their evidence untouched. |

### Metrics and run record

Capture serial target tokens/s with speculation disabled; speculative `decode_tokens_per_sec`; speedup versus that target’s serial rate; existing acceptance/mean accept length; target forwards; generated tokens per target forward; available width/cap distribution; steady-state peak memory; generated tokens and duration. Record the ordinary controller and all fixed settings, prompt IDs/corpus hash, target/drafter/donor revisions, runtime revision/dirty paths, owner map, output checksum/reference, and integrity status. Preserve existing timing decomposition only when available without synchronization-heavy instrumentation; explicitly mark unavailable per-component times.

Each run is a fresh process or fully reset Engine with no reused target/drafter/prompt cache state, and has a unique directory beneath `specs/002-qwen-bonsai-hybrid-target/evidence/series-a/runs/<run-id>/`. Store exact inputs/manifests next to raw metrics. The `001` tree is read-only.

## 8. Repeat/noise procedure and execution order

### Discovery

Run each required Series-A condition once under the same frozen prompt set, token count, generation/sampling, plain KV, ordinary CapController, runtime commit and machine. Start with a clean smoke/integrity pass for all target configs in this order: H0, H1a, H1b, H1c, H2, H3, B0. Only after every smoke passes, run the matched discovery order:

`H0 → H1a → H2 → H1b → H3 → H1c → B0 → H0`

The second H0 brackets the discovery sweep and estimates drift; both H0 records are fresh controls. The non-monotone middle ordering avoids measuring suffix depth as a simple clock-warmup sequence. Record start time, temperature/power status available from existing telemetry, and actual order. Do not tune cap, width, or another axis.

### Confirmation

Compare every discovery condition to fresh H0 for decode throughput and its explanatory measures. Flag an apparent winner, regression, or comparison whose paired difference is within observed H0/order variability. Also flag H1c-vs-H1a/H1b when the required interaction question cannot be decided from discovery.

Only flagged comparisons are repeated. For each comparison, perform three matched pairs at minimum (same prompt IDs, generation length, settings, and build), with order alternated:

`pair 1: H0 → candidate`; `pair 2: candidate → H0`; `pair 3: H0 → candidate`.

Shuffle the order of candidate comparison groups from a recorded seed. For the H1c interaction, use the same three-pair minimum for H1c versus H1a and H1c versus H1b when either comparison is decision-relevant. If a candidate is both flagged and compared to both endpoints, do not reuse one observation as two independent matched repetitions. Base decisions on paired differences and observed within-condition variability; one run cannot adjudicate a marginal result. Preserve every repeat and the comparison selection rationale.

This procedure avoids tripling every condition while meeting FR-012. If no comparison triggers a repeat threshold, record the variability basis and why none was near noise; an apparent outcome may not be declared a winner until repeats distinguish it from variability.

## 9. Gate B and later Series B reuse

`evidence/gate-b.json` is the explicit record with status `CLOSED` by default. It links required smoke/runtime/representation/tap records; valid discovery/repeat runs for H0, H1a, H1b, H1c, H2, H3, B0 and fresh H0; Series-A adjudication; repeat/noise rationale; and exact input/runtime hashes. Set `OPEN` only if all these references exist, pass their schema and integrity states, the Series-A adjudication answers FR-013, and every apparent win/regression/near-noise conclusion has its required matched confirmation. Any missing/invalid result, stale hash, unresolved semantic check, or unconfirmed conclusion keeps it closed.

The Series-B runner is a later implementation artifact and must fail closed unless this gate validates as `OPEN` and points to the current feature-002 adjudication/runtime/spec hashes. It reuses the identical owner-map serialization, composition loader, prompt/generation/measurement harness, and frozen H0/H1c/H2/H3/B0 targets, pairing each with the exact original `incoai` drafter and the exact `naklitechie` Bonsai-specific drafter. If Series A confirms H1a or H1b as best instead of H1c/H2/H3, add only that one required exception paired with both drafters. This gives the ten core cells (or twelve with the one authorized exception); no additional target composition, training, policy sweep, or Series-B run occurs in this planning pass.

## 10. Assumption audit and blockers

| Assumption / spec statement | Observed repository fact | Experiment impact | Required spec correction? |
| --- | --- | --- | --- |
| Same-index Qwen/Bonsai blocks may be directly substituted if dimensions agree. | Both use the same `DecoderLayer` implementation and layer family; Prism transformations are projection-local and outputs are canonical residuals. | Direct block boundaries are valid; additionally verify actual loaded layer family and projection types. | No. |
| DFlash taps `[5,19,33,47,61]` are block outputs. | `Target._body_hybrid` captures after `layer(...)`; both actual inspected DFlash configs list those IDs. | FR-009 counts are confirmed: 0,0,0,0,1,1,5 for H0,H1a,H1b,H1c,H2,H3,B0. Recompute from loaded checkpoint each run. | No. |
| One primary hybrid path can preserve Qwen embedding/final norm/LM head while B0 is full Bonsai. | The spec’s target table explicitly assigns B0 Bonsai roles; full Bonsai already loads through Prism path. | Treat B0 as endpoint control, outside the H0–H3 ownership-only claim, exactly per user instruction. | No. |
| Both entire target checkpoints may be loaded before selection. | Feature-001 single-target peaks are about 17.1 GB Qwen and 9.4 GB Bonsai. | Simultaneous full residency is rejected. Sequential donor load, prune, then Qwen load is the planned minimum. | No. |
| The spec assumption names the current repository-supported Bonsai2 target as `prism-ml/Ternary-Bonsai-27B-mlx-2bit`. | That `load.py` registry row’s comment identifies a Qwen3.6 sibling. The official Qwen3.8 Bonsai2 checkpoint is `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`, whose Prism schema and layer modules are compatible with the local loader and the Qwen3.8 target. Feature-001 hardware runs used a nathansutton repack. | The spec-assumed id selects the wrong checkpoint; this blocks exact target selection, but the intended Bonsai2 module structure is compatible. | **Yes, before runtime implementation/measurement:** change only the Bonsai2 repo id in the spec assumption to `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` and state that the experiment passes this repo explicitly instead of using the stale `REGISTRY` alias. Do not change the target matrix or controls. |

There is no module-interface or representation blocker. The incorrect donor repo id in the spec’s Assumptions section is a target-identity blocker until corrected; the smallest correction is recorded above. Runtime implementation can proceed after that correction without changing the constitution, target matrix, or experiment scope, subject to exact checkpoint and drafter revision verification before measurement.
