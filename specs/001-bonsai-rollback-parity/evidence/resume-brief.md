# Bonsai rollback parity — resume brief

Prepared offline from the project-local saved evidence on 2026-09-23. No model, generation, benchmark, test, runtime edit, or Git state change was performed for this brief. The saved captures are diagnostics; T004 and T005 remain incomplete.

## 1. Verified saved state

### Identities and input

- Target: `nathansutton/Qwen3.8-27B-Ternary-Bonsai-2-DFlash2-MLX`, snapshot `234cc925d77cd94683e47b1493bd964917aaf43b`; config SHA-256 `20a7ccac3e519b5b5d7c4eaa6451352dd4188da44b4385ffa6cf87a4f2b8a94a`; weights SHA-256 `68541bf9c72747df90764338fa966105b34d94e4e8751dd5d3403d732eedddcf`.
- Sidecar: `/Users/do/.cache/chad/dflash/015e795645c74b1a0eeef3b570031fb62e769bc5-q4g64`; config SHA-256 `6fd15051e629eba87121298bce299f92f1ef99c14e01e144825bbf0aaeaca86f`; weights SHA-256 `876c368b5abfdd5de52ab14fcd5d3cccb07f0059f3903c984e4f17dbcee8a552`. Direct mlx-dspark loader accepted these exact bytes: 47 b4g64 and 2 b8g64 modules; DFlash2 block size 8, width 7, taps `(5,19,33,47,61)`.
- Tokenizer: `tokenizer.json` SHA-256 `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523`; `tokenizer_config.json` SHA-256 `95c557768e6b88a7128befc7bfd3c7de50e5d51af9b8b33a9f4dee0e04f99679`.
- Prompt is the one user message `Write a production-quality Python LRU cache with tests and type hints.`, chat template `add_generation_prompt=True`, thinking disabled, greedy/temperature 0. Exact regenerated 26-token prefix: `[248045,846,198,7734,264,5492,21408,12654,436,34810,6297,440,6813,321,913,29642,13,248046,198,248045,74455,198,248068,271,248069,271]`. No historical token-ID dump survived; each resumed run must regenerate and record this.
- Runtime profiles: mlx-dspark Python 3.12.13; Chad Python 3.11.15; both MLX 0.32.2 / mlx-lm 0.31.3. Installed qwen3_5.py, gated_delta.py, and cache.py hashes match across environments. GDN callable identity was verified true in both inspections; this does not establish equal inputs or call flags.
- Chad normal load was directly observed with `Engine.fastpath=True`, `CHAD_NO_FASTPATH` unset, installed fused attributes, and GDN `__call__` routed via `src/chad/mlx_fastpath.py`. Fastpath is run-specific; the no-fastpath diagnostic did set `CHAD_NO_FASTPATH=1` before load and recorded `Engine.fastpath=False`.

### Policy trace, not a correctness verdict

Saved policy traces have matching seed costs `[1.0,1.63,2.15,2.52,2.48,2.5,2.5,2.53]`. Both record R1 proposal depth 7 / accepted 0 and R2 depth 7 / accepted 1. At the next decision, Chad records margin `0.25` and proposes depth 7; mlx-dspark records `0.125` and proposes depth 1. The subsequent trace records (Chad accepted 1 from proposed 7; mlx accepted 1 from proposed 1). These are saved trace facts, not proof of a state or rollback cause. The historical values 0.25/0.125 were supplied controls, not assumed fresh expected outcomes.

### Five saved physical run files and actual cache configuration

The JSON probe omitted a top-level `kv_bits` field for the default Chad runs. Use the recorded layer-3 cache implementation as the actual saved setting: `QuantizedKVCache` means quantized KV; `KVCache` means plain KV. Exact bit width for default-quantized Chad is **not serialized** and cannot be reconstructed from these JSON records alone. Do not report it as a verified numeric value.

| Saved run | Runtime / fastpath | Actual saved target KV / alignment | R1 / R2 / R3 acceptance; notable R3 |
|---|---|---|---|
| `runs/chad.json` | Chad; true; env unset | QuantizedKVCache (bits omitted); align false | 0 / 1 / 1; R3 draft `[10417,12654,29806,18887,29806,271,550]` |
| `runs/mlx-dspark-aligned.json` | mlx-dspark | Plain KVCache; `align_prefill_diagnostic=true` | 0 / 1 / 1; R3 draft `[10417]` (the run had width 1) |
| `runs/kv0/chad.json` | Chad; true; env unset | Plain KVCache; align false | 0 / 1 / 1; R3 draft `[10417]` |
| `runs/kv8/mlx-dspark-aligned.json` | mlx-dspark | QuantizedKVCache, `kv_bits=8`; align true | 0 / 1 / 1; R3 draft `[10417,12654,29806,18887,29806,271,550]` |
| `runs/no-fastpath-kv0/chad.json` | Chad; false; `CHAD_NO_FASTPATH=1` | Plain KVCache; align false | 0 / 1 / 1; R3 draft `[10417]` |

All five contain a 26-ID prompt and first 16 output IDs. The aligned base files are not a matched KV control: historical Chad is quantized, mlx-dspark is plain KV. The `kv0` Chad and `kv8` mlx runs share the saved R1 draft sequence; their later proposal/R3 evidence is informative but changes KV precision across runtime, and the fastpath/no-fastpath pair also differs in configuration.

For the continuous R1→R2→pre-R3 decision, R1 anchor is 271, accepted 0; R2 anchor is 2, accepted 1; pre-R3 anchor is 27325. At the anchor row immediately relevant to R3, the principal captures include:

| Capture | R2 anchor-row top-2 IDs / values / margin | R3 anchor-row top-2 IDs / values / margin |
|---|---|---|
| Chad default quantized | `[27325,12] / [19.25,19.0] / 0.25` | `[10417,899] / [21.875,14.5625] / 7.3125` |
| mlx-dspark aligned plain | `[27325,12] / [19.125,19.0] / 0.125` | `[10417,899] / [21.875,14.5625] / 7.3125` |
| Chad kv0 | `[27325,12] / [19.125,19.0] / 0.125` | `[10417,899] / [21.875,14.5] / 7.375` |
| mlx-dspark kv8 aligned | `[27325,12] / [19.125,18.875] / 0.25` | `[10417,899] / [21.875,14.5] / 7.375` |
| Chad no-fastpath kv0 | `[27325,12] / [19.125,19.0] / 0.125` | `[10417,899] / [21.875,14.4375] / 7.4375` |

These per-run rows are evidence snapshots. Do not infer causality from numeric cross-runtime differences. Full logits/fused observations have hashes and shapes in the JSON; selected state hashes/samples were retained, not full tensors.

## 2. Offline run comparison

### What the records show

- Every file uses the same target, sidecar weights and tokenizer fingerprints, prompt IDs, temperature 0, and first-three-round capture window. Output arrays provide the first 16 generated IDs. Run options are not uniformly serialized: default Chad `kv_bits` is absent; mlx-dspark plain run's option is absent though its cache class is plain. Cache class is the evidence for plain versus quantized.
- R1 default Chad and aligned mlx-dspark have identical seven draft IDs and verify IDs; both accept zero. At R2 their seven draft IDs and verify IDs also match and both accept one. Their R2 anchor-row top-2 identity is `[27325,12]`, but margin is 0.25 versus 0.125. Their R3 anchor top-2 identity is `[10417,899]`, and the saved margin is 7.3125 in both; proposal widths differ (7 versus 1). This pair is not KV-controlled.
- Chad default versus Chad kv0 changes R1 draft IDs (last positions), R2 draft IDs (one position), and R3 from seven drafts to one; R2 anchor margin shifts 0.25→0.125. This is suggestive that the default-vs-plain Chad configuration is associated with changed proposals/margin in these saved executions. The JSON does not serialize the exact default quantization bit count, and the two runs are not repeated same-width controls.
- Chad kv0 versus mlx-dspark kv8 aligned have matching R1 draft IDs, but differ in R2 proposal IDs and R3 margins/rows slightly. Since one is plain and the other 8-bit KV and runtimes differ, this does not isolate a runtime effect.
- mlx-dspark aligned plain versus mlx-dspark kv8 aligned: the R2 margin shifts 0.125→0.25 and R3 proposal width shifts 1→7. This is a same-runtime precision sensitivity observation, but lacks repeated equivalent controls and does not prove quantization caused the changes.
- Chad fastpath kv0 versus no-fastpath kv0: proposals differ at R1/R2, while R2 anchor margin remains 0.125; R3 margin varies 7.375 versus 7.4375. This is a useful fastpath sensitivity diagnostic, not an equivalently initialized independent control or proof of cause.

Cache records include per-layer classes, logical offsets, steps, hashes/shapes/samples. Target cache uses `ArraysCache` for recurrent/convolution slots and attention `QuantizedKVCache` or `KVCache`; drafter cache records are present at proposal boundaries. Captures do **not** provide immutable full cache tensors, a complete independent same-runtime reference, five-repeat bounds, or a controlled post-reconcile comparison. The current probe also omits derived absolute positions and some required live cache metadata.

### Evidence status

- **VERIFIED:** actual saved cache family/config above; fastpath true/false observations; matched prompt/tokenizer/weight identities; policy trace R1/R2/pre-R3 values; saved proposal IDs, acceptance counts, top-2 records, hashes and selected cache metadata.
- **SUGGESTIVE:** changing Chad from default quantized KV to plain KV is accompanied by changed proposals and a one-token R3 width in that run; mlx-dspark plain→8-bit has a corresponding R3 width change; fastpath state changes proposal IDs. These suggest configuration sensitivity.
- **NOT YET CONTROLLED:** exact default Chad bit width; matched same-runtime repeated comparisons; causal attribution to quantization, fastpath, alignment, rollback, verify, or drafter; identical full initial/cache state across runtime; independent S=8 reference; S=1/S=2 committed references and five-run numeric bounds; pre-reconcile verify/tap parity; Gate A–E outcomes. No saved cross-runtime numeric difference is a causal result.

### What kv0, kv8, and no-fastpath-kv0 prove

- `kv0/chad.json` proves a Chad run with observed plain `KVCache`, normal fastpath, and no prefill-alignment diagnostic; it records the altered proposal sequence and R3 width 1. It does not prove why those changed.
- `kv8/mlx-dspark-aligned.json` proves an mlx-dspark run with explicit `kv_bits=8`, observed `QuantizedKVCache`, and alignment diagnostic enabled; it restores a seven-token R3 proposal in this one run. It does not match Chad kv0's KV setting or prove a cross-runtime relationship.
- `no-fastpath-kv0/chad.json` proves a separate fresh Chad load with fastpath disabled and plain KV; it is a fastpath sensitivity capture. It does not isolate fastpath from all run variation or establish correctness.

## 3. T004 completeness audit (probe vs tasks/data-model)

Field statuses refer to `evidence/probes/bonsai_rollback_probe.py` as saved. “Present” means the field/data is actually recorded; it does not mean a gate has passed.

### ControlledRun

| Required field | Status | Audit |
|---|---|---|
| implementation/reference kind | PARTIAL | Runtime label recorded; no reference kind exists yet. |
| target ID/fingerprint | PRESENT | ID and weights/config fingerprints. |
| drafter ID/fingerprint | PARTIAL | Sidecar path and weights hash; sidecar config hash is not recorded by `common_record`. |
| tokenizer fingerprint | PRESENT | tokenizer and tokenizer-config hashes. |
| runtime profile | PARTIAL | Python, MLX, mlx-lm, OS, device; model config/kernel flags partial. No complete runtime identity including all required settings. |
| mlx-lm source hashes | MISSING | Not emitted. |
| GDN callable identity and call path/flags | PARTIAL | Chad call file/line and captured call args; callable identity/signature and verified identity result absent. mlx GDN capture records some args; dispatch details incomplete. |
| target type/loader | PARTIAL | model type and loader string recorded; exact selected loader/config proof is incomplete. |
| Chad fastpath / env / install state | PARTIAL | Fastpath bool and env state recorded for Chad; installed class/call-path identity and actual replaced/fused branch per run not completely recorded. |
| prompt IDs / sampling | PRESENT | IDs, greedy settings and template metadata. |
| prefix state ID | PARTIAL | Hash of prompt IDs excluding last token; not a state snapshot identity/fingerprint. |
| target sequence width | PRESENT | Captured at verify entry. |
| anchor/pending ID | PRESENT | Captured before proposal. |
| draft IDs / verify IDs | PRESENT | Saved before accepted count. |
| proposal context logical position | PARTIAL | Generated count recorded; context row count and logical/absolute positions are missing. |
| accepted count | PRESENT | Filled by round callback/reconcile after proposal fields. |
| tap IDs/order | PARTIAL | Tap IDs in run; order of actual capture and mapping to model layers is not fully documented. |

### ProposalInputRecord

| Required field | Status | Audit |
|---|---|---|
| target/drafter equivalent input-state identity | PARTIAL | Cache record hashes/samples captured, but no branch identity or equivalence classification. |
| logical generated-token position | PRESENT | `generated_before` / proposal context logical position. |
| anchor/pending token | PRESENT | Recorded before proposal. |
| seven draft IDs | PRESENT | Captured after proposal; draft width is recorded but not asserted to be seven. |
| exact `verify_ids=[anchor]+draft` | PRESENT | Captured at verify hook; not independently validated against the formula. |
| proposal-context logical position | PARTIAL | Scalar generated count only; context rows and absolute positions absent. |
| accepted count after fields 1–5 immutable | PARTIAL | Assigned later, but no immutable record/version or ordering assertion ensures fields remain fixed. |
| separate Chad/mlx values and Gate A classification | MISSING | One run per file; no explicit equivalence/control-mismatch/input-state/proposal classification. |

### BoundarySnapshot

| Required observation | Status | Audit |
|---|---|---|
| immutable boundary, round, label, token position | PARTIAL | Round IDs and named dict boundaries exist; snapshots are not a typed immutable object and some boundary labels/positions are implicit. |
| materialized copied MLX arrays | PARTIAL | `array_record` evaluates/copies recognized MLX arrays into hashes/samples; not full arrays. Cache value extraction is incomplete for several cache forms. |
| input token IDs | PRESENT | `verify_ids` captured; prompt IDs at run scope. |
| per-layer cache class/order/shapes/dtypes/trimmable | PARTIAL | Class/order and some shapes/dtypes recorded; trimmable, complete slots and layer semantic mapping not recorded. |
| attention live K/V, live length, offset, absolute positions | PARTIAL | Cache class/offset and backing tensor shapes/hash can appear; live range/length and derived absolute positions absent. |
| GDN conv/recurrent state and layer mapping | PARTIAL | `ArraysCache` slots and GDN arg/conv samples are recorded, but required named state-to-layer mapping/complete immutable observation is not guaranteed. |
| mask/sequence lengths/left padding/cache recurrence order | PARTIAL | Some generic attributes may be introspected; GDN mask is always recorded `None` in Chad; exact sequence/mapping absent. |
| tap IDs, drafter context/cache positions | PARTIAL | Tap IDs and selected context array/cache record exist; positions and accepted-row mapping absent. |
| pre-round GDN state, q/k/v/a/b/A_log/dt_bias/mask/use_kernel | PARTIAL | Hook records these when present, but no explicit pre-round state guarantee; Chad mask omitted; no complete call identity/dispatch record. |
| conv input/window/result | PARTIAL | Conv input recorded; pre-window/resulting window not captured as named fields beyond generic post cache. |
| verify logits/top2 IDs/values/margins/fused rows | PRESENT | Recorded with shape/hash/selected values; not full immutable tensors. |
| accepted-prefix/reconciliation/post-state | PARTIAL | accepted prefix and cache record saved; fused accepted row only Chad; no independent semantic comparison or required exact invariants. |
| drafter accepted fused rows/context positions/KV/next proposal | PARTIAL | Some cache and context arrays captured around proposal/reconcile, no complete logical/absolute row mapping or projected rows. |
| controlled numeric rules and repeated controls | MISSING | No per-component dtype/metric/five samples/bound/stability/rationale. |

### GateResult

| Required field | Status | Audit |
|---|---|---|
| gate identity and result | PARTIAL | A/B1/B2/C/D/E placeholders exist as `inconclusive`. |
| evidence/observation IDs and exact invariants | MISSING | No gate-specific evidence links or evaluated invariant result. |
| numeric rule/bound/rationale | MISSING | No controlled bounds exist. |
| missing control and classification | PARTIAL | Generic missing-reference/control string only; no gate-specific control or routing classification. |
| causal stop/routing decision | MISSING | Does not encode stop-before-rollback, pass/fail, or allowed next layer. |

**Output collision risk:** `main()` writes `<runtime>[-aligned].json` into the output directory; repeated runs with the same runtime/alignment overwrite prior bytes. A user-provided common `--output-dir` can collide across KV and fastpath variants. The already-documented corrected `mlx-dspark-aligned.json` overwrote an earlier run. Use unique explicit run directories/filenames in future work; do not rewrite existing captures.

## 4. T005 recovery plan — independent same-runtime, same-width S=8 reference

Implement later; this is a checklist, not a change to the interruption marker.

1. Keep a speculative branch and a reference branch in one runtime with same target weights, sidecar/tokenizer, device/kernel configuration, exact logical committed prefix, and equivalent pre-verify target state. Allocate fresh target caches for both. Do not branch from a cache after rollback/reconcile or share mutable cache objects; if cloning is used, deep-copy/materialize every cache slot and verify matching logical state.
2. Establish the pre-verify branch at the exact current prefix: prefill once from the identical prompt/logical committed IDs, then ensure both branches have the same anchor/pending token semantics, attention lengths/offsets/absolute positions, recurrent/convolution state, layer order, taps and shapes. Record state IDs and structural checks before proposal/verification. Do not assume equality from matching draft width or accepted count.
3. Freeze the proposal record before verify: seven draft IDs, anchor, context logical position; define and assert `verify_ids == [anchor] + draft_ids` exactly. Feed this exact eight-ID sequence to both speculative and S=8 reference branches, with target sequence width 8. The reference executes the same target verify/capture operation but never calls production reconciliation on the reference cache.
4. Before any reconciliation, capture both branches' logits/top-2 IDs, values and margin for all eight rows; fused tap rows in tap-ID order; q/k/v/a/b/A_log/dt_bias/mask/use_kernel and incoming recurrent state; convolution input/window; attention live K/V and logical lengths/offsets/positions; recurrent/convolution cache slots, shapes, dtypes, hashes and exact ID/row position mappings. Materialize/copy MLX arrays at capture time. Capture cache and projection/dispatch metadata sufficient to identify the actual executed path.
5. Compare the two same-runtime S=8 observations first (B1/B2). Do not treat S=1 logits as the S=8 verify oracle. For Gate C fused/tap rows after reconcile, compare against the corresponding row from this untouched S=8 reference with identical verify IDs and equivalent pre-verify state.
6. Ensure independent reference state survives suspect reconciliation: allocate it before or independently of the speculative branch; never pass the speculative cache, reconciled cache, mutated target, or captured mutable array view to the reference. Snapshot before mutation and record distinct branch/cache identity. A full fresh model is preferred where model internals can be mutated by capture hooks; otherwise prove model weights are shared read-only and every mutable cache/tap/stash is branch-local.
7. Safe reuse from the existing rollback probe: identity hashing (`sha256`), `array_record` only after upgrading it to preserve required complete immutable evidence, `cache_record` only after adding live-range/position/semantic metadata, `logits_record`, `mlx_gdn_record` / `chad_gdn_record` only after adding missing call fields, and the existing tokenizer/message/identity constants plus model loader setup. Reuse wrappers only if hooks return original runtime values and are restored in `finally`.
8. Keep runtime-specific implementation separate. Chad uses `Engine`/Prism fastpath hooks and GDN collector; record fastpath installation state and do not mutate Chad files. mlx-dspark uses `load_target`, `target.verify`, its capture stash and `target.rollback`; capture whether Prism MMA is active. Preserve each runtime's actual call signature, cache classes, layer mapping and context timing. Do not force identical implementation-specific wrapper classes across runtimes.

## 5. Minimum next model-backed runs

Do not repeat a saved variant just to regenerate its current diagnostic facts. All correctness gates remain open. Model loads become necessary after completing offline T004/T005 code and controls.

| Next run | Runtime / kv_bits / fastpath / alignment | Output destination | Resolves |
|---|---|---|---|
| Controlled continuous Gate A trace from fresh state, same exact prompt, complete A semantic/logical fields; keep R1 k=7/acc=0 → R2 k=7/acc=1 → pre-R3 capture continuous | Chad normal fastpath; explicit plain KV (`kv_bits=0`); align false | New unique `evidence/runs/gate-a-continuous/chad-kv0-fastpath.json` | Removes historical Chad-vs-mlx KV mismatch; captures proposal-state inputs and the pre-R3 decision under one Chad config. |
| Same controlled continuous trace | mlx-dspark; plain KV (`kv_bits=0`); align-prefill diagnostic true | New unique `evidence/runs/gate-a-continuous/mlx-kv0-aligned.json` | Matched KV precision; tests whether the alignment mode yields comparable logical state and fresh proposal/margin facts. |
| Same-runtime S=8 independent verify reference for exact chosen verify IDs and equivalent pre-verify state (can be captured alongside the fresh gate run if implementation supports independent branches without extra process) | Each runtime separately; same kv_bits/fastpath/alignment as its controlled trace | `evidence/runs/b2-reference/chad-...json` and `mlx-...json` (unique names) | B1/B2 and the S=8 fused/tap numeric oracle for C. |
| Fresh semantic/reference/control workload only after A/B1/B2; accepted=0 S=1 and accepted=1 S=1 + S=2 paths, then exactly five controls per required component/tap/row/metric as data-model specifies | Each runtime independently, matched to the applicable runtime config; S=1, S=2 and S=8 widths kept separate | Unique subdirectories under `evidence/runs/controls/{chad,mlx-dspark}/` | T006/T007/T011 and Gate C bounds. Do not start bound samples after seeing a production/failing sample. |

Skip another standalone R1/R2/pre-R3 policy-only diagnostic: both trace files already contain this fact pattern. Skip repeating `kv0/chad`, `kv8/mlx`, and `no-fastpath-kv0/chad` as one-off sensitivity runs; they are saved and do not close a gate. Fastpath-off Chad remains optional B3 and cannot delay C. No benchmark run is required or permitted by this preparation request. The first necessary model load is after T004/T005 (and control implementation): a fresh controlled Gate A capture, not a benchmark.

## 6. Known losses and do-not-repeat notes

- Lost when model processes ended: loaded weights/model objects, lazy MLX graphs, full unreduced logits/taps, full live KV/recurrent/convolution arrays. Saved JSON retains selected values, hashes, shapes and samples only.
- The prior bytes at `/private/tmp/bonsai-rollback-observations/mlx-dspark-aligned.json` were overwritten by the corrected one-behind aligned run; those bytes cannot be recovered. The current project file is the corrected run.
- `/private/tmp/bonsai_reference_probe.py` was never created before interruption. The project `bonsai_reference_probe.py` is only a marker that raises; there is no unfinished source to recover.
- Disproved as starting fixes in baseline: grouped-convolution dtype; nesting adaptive WidthPolicy under old CapController; raw versus converted generic sidecar after exact-sidecar fixed7 parity; one-behind prefill as production/throughput fix; threshold/acceptance-prior/cost-seed tuning for earlier target/logit divergence. Revisit only if direct new evidence identifies the same operation.
- Historical fixed7 rates/forward counts are practical-execution controls, not fresh measurements or variable-width correctness proof. Historical R1/R2 margins are observations, not forced expectations. Existing five physical JSON files are short 16-token diagnostics, not final golden 512-token validation or Gate A–E completion. Supplied user-facing terminal output not saved to disk is not evidence.

## 7. Exact resume point

1. **Offline first:** T004 remains open. Complete the rollback probe schema against the audit above, including per-gate results and unique output destinations; do not mark the task complete until the required fields are actually emitted.
2. **Offline next:** implement T005's independent fresh/cloned same-runtime S=8 verify/capture reference under the exact contract above. Keep this reference independent from reconciled state. T005 remains open until code is complete.
3. **Then:** implement T006 ordinary S=1 and fresh accepted=1 S=2 semantic references, then T007's predeclared five-run width-specific numeric controls; preserve separate runtime branches and output files.
4. **First required model load:** only after those offline pieces, make fresh controlled Gate A continuous runs in both runtimes with explicit plain KV, record all A inputs and R1→R2→pre-R3 evidence, and attach/capture independent S=8 branches for exact B1/B2 verify IDs where feasible.
5. Proceed in dependency order: T008 A/continuous trace; T009 B1; T010 B2; T011 references/controls; T012 C; T014 D; T015 E; T016 initial decision. B3 remains optional. Do not mark T004/T005 complete now, and do not authorize a patch from current cross-runtime or precision-sensitivity evidence.
