# Research: Bonsai Rollback Parity

## Decisions and Findings

### Fixed7 and observed trace

**Decision**: Use fixed7 only as the matched execution control. Begin diagnosis at the target state before R1 and the pre-rollback verify/tap path, not quantization, selector weights/math, Prism MMA, generic full-width execution, or width scheduling.

**Evidence**: The supplied controlled run used the same physical target and drafter: Chad fixed7 was about 33.9 tok/s / 149 forwards and mlx-dspark about 33.6 tok/s / 146 forwards. In the aligned trace, R1 is k=7/accepted=0 on both; R2 is k=7/accepted=1 on both with incoming margin 0.25 and matching policy state. Before R3, the target margin is 0.25 in Chad and 0.125 in mlx-dspark. The near-tie width change occurs after that target-logit observation.

**Alternatives considered**: Changing the `<0.25` gate or scheduler might make widths agree but would hide the earlier target-state/logit difference. Starting with quantization or the verify kernels is unsupported by the fixed7 control and existing evidence. Revisit those execution details only if a captured boundary directly localizes a divergence there.

### Proposal inputs before target verify

**Decision**: Make exact proposal-input capture the first gate before comparing target verify or rollback. At the same logical generated-token position, record the current anchor/pending token ID, all seven proposed draft IDs, exact `verify_ids = [anchor] + draft`, and proposal-context logical position. Record accepted count only after those values are captured.

**Rationale**: Matching draft width `k` and matching accepted count do not establish that the target saw identical verify IDs. If proposal IDs differ while equivalent target/drafter input state is expected, stop target rollback investigation and localize drafter/context generation. Do not compare downstream target margins as though verify inputs matched.

### Cross-runtime environment and source controls

**Decision**: Do not treat the gated-delta recurrence implementation or mlx-lm source-version skew as an initial hypothesis. Establish within-runtime verify/capture correctness with a fresh same-width S=8 reference, and establish post-reconcile semantics with a separate ordinary S=1 committed reference. Compare raw cross-runtime numeric tensors only after proposal IDs and both within-runtime references are understood.

**Independently verified evidence**:

| Environment | Python | MLX | mlx-lm |
|---|---:|---:|---:|
| Chad | 3.11.15 | 0.32.2 | 0.31.3 |
| mlx-dspark | 3.12.13 | 0.32.2 | 0.31.3 |

The installed source files are byte-identical across both environments:

| File | SHA-256 |
|---|---|
| `qwen3_5.py` | `f0daa30bba5cb521c8bdfa7093101a544c6a37bbba09bca582288219cb04ae3a` |
| `gated_delta.py` | `79c8376a51c694b03e54d2f996ced6ea6c8c42868b8571529f97334db165a3e1` |
| `cache.py` | `819ed95dcbf755652363cfdb15a639890447abb534a06dcefd52c7fff5055750` |

In both environments, `qwen3_5.gated_delta_update is gated_delta.gated_delta_update` evaluates true. The callable is `mlx_lm.models.gated_delta.gated_delta_update`, at `gated_delta.py:262`, with the same signature. This rules out source-version skew and distinct recurrence function objects as starting hypotheses. It does not establish equal inputs, projection graphs, flags, device dispatch, or logits.

**Alternatives considered**: Begin by searching for a recurrence-source difference; reject because both installed files and runtime callable identity are independently verified equal. Require bit-identical raw logits across runtimes; reject because the projection/op graphs differ as documented below.

### Hybrid rollback and verify boundaries

**Decision**: Treat rollback as a hypothesis, not a conclusion. Observe state before verify, verify/taps before rollback, reconciled state, and subsequent verify separately.

**Evidence**: mlx-dspark's `Target.verify()` and capture hooks are in `src/mlx_dspark/target.py:452-573`; accepted-prefix GDN replay, conv reconstruction, and KV trim are in `src/mlx_dspark/target.py:599-617`. Chad's GDN input capture is in `~/git/chad/src/chad/mlx_fastpath.py:284-333`; its accepted-prefix replay and attention trim are in `~/git/chad/src/chad/engine.py:2455-2503`. Both source paths keep `accepted + 1` rows (anchor plus accepted drafts), replay recurrence from captured pre-round state, rebuild the conv history, and trim rejected KV. Both return verify logits/taps before reconciliation. These parallel formulas do not establish same input tensors or same numeric output.

**Alternatives considered**: Assume the matching formulas prove the rollback is correct; reject because captured inputs, state, call path, and arrays can differ. Patch `Target.rollback()` preemptively; reject until Gate C identifies it as the first differing boundary.

Chad's GDN fastpath fuses target projections before it forms q/k/v/a/b, whereas mlx-dspark's replicated hybrid loop calls the installed `qwen3_5` layers. Compare actual layer residuals and post-projection q/k/v/a/b, but do not make cross-runtime bit identity the correctness oracle. Gate B1 establishes exact verify IDs and equivalent pre-verify state. Gate B2 compares each runtime's speculative S=8 verify/capture with a fresh/cloned same-runtime target execution that starts from equivalent pre-verify state, receives the exact same eight IDs, runs the same width, and captures outputs/taps/cache effects before reconciliation. This isolates verify/capture/projection behavior from rollback. It must not use the speculative rollback/reconcile result as its reference, and an S=1 committed run is not the raw pre-reconciliation numeric oracle. Establish and understand both within-runtime references before Gate B3: the same-width S=8 verify/capture reference and the ordinary S=1 committed-semantic path. Gate B3 compares cross-runtime token identities, top-k, margins, and numeric diagnostics only after those references are understood. A cross-runtime raw tensor/logit mismatch across valid but different graphs is not automatically a defect.

Use a distinct ordinary S=1 committed-semantic reference for post-reconcile state: prompt + anchor for accepted=0 and prompt + anchor + first accepted draft for accepted=1. This is the primary rollback/reconcile oracle and the reference for subsequent committed behavior, not for raw width-8 verify logits.

### Required captured values

**Decision**: Compare the following per GDN layer and in stable layer order: pre-round recurrent state; q/k/v/a/b, `A_log`, `dt_bias`, mask and `use_kernel` passed to the recurrence update; convolution input and pre-round window; cache type/order and KV live contents, lengths and offsets; target taps; and proposal context.

**Evidence**: mlx-dspark `_capture_linear()` records recurrence arguments and the marked conv input at `target.py:458-490`; its hybrid body captures tapped per-layer residuals at `target.py:167-183`. Chad captures equivalent inputs in `mlx_fastpath.py:284-333` and taps selected target-layer outputs in `mlx_dflash.py:756+`. The two implementations must be compared at the same semantic position, not by variable name alone.

The accepted-prefix branches for a width-8 verify are `keep=1` for accepted=0 and `keep=2` for accepted=1. Compare the same-width S=8 verify/capture before reconciliation against its independent fresh target reference. Then compare reconciled logical target state and subsequent committed behavior against a separately allocated S=1 committed reference after anchor-only and anchor-plus-one-accepted-token execution. Raw S=8-versus-S=1 numeric deltas are recorded but do not alone establish a defect.

### Gated-delta implementation identity

**Decision**: The installed `gated_delta_update` callable identity and source are independently verified equal in both runtimes. Retain runtime capture to document the actual call path and arguments during the model-backed diagnostic, not to re-open recurrence source skew as the initial hypothesis.

**Evidence**: The two installed environments have the versions and byte-identical source hashes listed above. In both, `qwen3_5.gated_delta_update is gated_delta.gated_delta_update` is true; the shared callable is `mlx_lm.models.gated_delta.gated_delta_update` at `gated_delta.py:262), with matching signature. mlx-dspark capture patches the defining module of the installed GDN class, while rollback imports the original `gated_delta` function (`src/mlx_dspark/target.py:464-484,599-617`). During the model-backed probe, retain the exact runtime module/file/line/signature/call path and call arguments to localize inputs and dispatch.

Installed Qwen3.5 hybrid cache construction uses `ArraysCache(size=2)` for GDN layers and `KVCache` for full-attention layers (`qwen3_5.py:268-305`). The two array slots represent convolution history and recurrent state. KV cache trim decrements its live offset; unused tail storage must be interpreted through the cache's live length/offset, not compared as live content (`cache.py:309-312,378-381`).

**Alternatives considered**: Infer equivalence from spelling alone; reject. The source/hash and callable identity evidence is stronger, but still does not prove same call flags, inputs, device dispatch, or projection results.

### Prism and quantized target path

**Decision**: The physical target config is `prism_hadamard_qwen35`, and both runtimes route it through their Prism loader. Record active projection modules, fastpath state, and kernel dispatch for the actual model-backed run.

**Evidence**: `src/mlx_dspark/prism_pack.py` routes the `prism_hadamard_qwen35` model type and implements rotated 2-bit affine g128 packed projections that call `mlx_qmm_mma.qmm`. The target config is verified as that type and both loaders select their Prism path. Chad's Engine attempts `mlx_fastpath.install()` immediately after loading the model. For supported Prism Qwen3.5 targets, this fastpath changes the target projection/op graph before `gated_delta_update`: fused qkv|z, fused b|a, fused q/k/v attention projections, Prism rotation/sign folding changes, and compiled/fused layer paths where applicable. mlx-dspark does not execute this identical Chad fastpath graph.

The diagnostic must explicitly capture Chad's `Engine.fastpath`, `CHAD_NO_FASTPATH`, and relevant installed fastpath class/call-path identity. Do not infer that installation succeeded from supported source code. Run normal-fastpath and `CHAD_NO_FASTPATH=1` diagnostics in separate fresh processes, loading a new target model in each process with the environment set before `mlx_fastpath.install()` can mutate the model. Do not toggle the variable after installation and treat the loaded model as an independent control. Use equivalent weights, input IDs, sampling settings, and cache state. This is a margin-sensitivity diagnostic only; throughput from the branch is not correctness evidence.

**Alternatives considered**: Treat the shared recurrence function as proof of equal q/k/v/a/b or logits; reject because the upstream projection/op graphs differ. Treat source support as proof that Chad installed the fastpath in a given run; reject until runtime state is recorded.

### Drafter-context staging

**Decision**: Keep the drafter context gate separate from target rollback. If target state is serial-equivalent but drafter state differs, investigate the accepted fused rows and context cache append/position path.

**Evidence**: mlx-dspark slices accepted target taps after rollback in `src/mlx_dspark/generate.py:1645-1651`; `src/mlx_dspark/dflash_model.py:178-210,427-465` appends target context rows and advances drafter KV positions. Chad keeps accepted fused taps pending until its next drafting transition (`~/git/chad/src/chad/engine.py:2698-2701`). The staging boundary differs, so compare equivalent logical rows and absolute positions rather than requiring the same intermediate object shape.

The R3 width gate lives downstream in `src/mlx_dspark/dflash_width_policy.py`; scheduler work waits until target logits/margins are reconciled.

### Deterministic test architecture

**Decision**: Reuse the existing seeded tiny Qwen3.5 hybrid fixture only for generic rollback-state invariants. It uses stock mlx-lm Qwen3.5, not the physical Prism/Chad-fastpath target path, so a passing fixture is not reproduction of the physical Bonsai divergence. Keep separate same-width S=8 verify/capture and S=1 committed-semantic references in the regression; tie any causal regression to the first failing boundary established by the physical trace.

**Evidence**: `tests/test_bonsai.py` defines `_tiny_hybrid()` using the installed mlx-lm Qwen3.5 model and already has a fresh-cache reference helper plus partial-accept, zero-accept, KV trim and recurrent-state tests (`tests/test_bonsai.py:260-394`). The existing `_committed_reference()` sends its token list in one `Target.run()` call, so the new regression must add an independent token-by-token committed-path reference rather than assume that helper has generation's one-token execution cadence. Existing tests do not record every requested boundary or require width 8. `tests/conftest.py` permits a CPU fallback when Metal is unavailable. There is no current diagnostic CLI or script that exports round-level state/tap traces.

Repeated S=1 committed-vs-committed runs establish determinism of the committed reference. Repeated same-width S=8 controls establish determinism of the verify reference. Do not derive an S=8-versus-S=1 tolerance solely from S=1 repetition; a numerical difference across those widths is not by itself a verify or rollback defect when post-reconcile semantic state and token behavior are correct.

**Alternatives considered**: Use a throughput benchmark as the regression; reject because it is not deterministic semantic evidence. Use an S=1 committed run as the raw pre-reconciliation S=8 logits oracle; reject because S=8 and S=1 may use different MLX/Metal kernel shapes and produce small within-runtime floating-point differences. Rely only on the current tiny test; reject because it may not reproduce a defect dependent on the full checkpoint. Add a full downloaded 27B fixture to every test run; reject unless the evidence proves it is needed, because it adds heavyweight model/hardware dependence. If the observed divergence cannot be minimized, keep an explicitly labeled M4-only model-backed integration probe with checkpoint/runtime provenance.

### Golden hardware comparison

**Decision**: Preserve the established golden benchmark as a separate required final hardware parity validation: prompt `Write a production-quality Python LRU cache with tests and type hints.`, thinking disabled, temperature 0, max_tokens 512, and the same exact physical Chad-built sidecar.

**Rationale**: The generic benchmark CLI is useful as an additional execution control but cannot replace the known golden comparison. Record actual token/output identity and run metadata; do not invent a golden output or claim a result from a prepared command.

## Research Tasks Resolved for Planning

- Source inspection identifies candidate state boundaries; the correctness path follows Gate A, B1 (exact verify IDs and equivalent pre-state), B2 (same-width S=8 verify/capture reference within each runtime), then C–E (post-reconcile S=1 semantic state and subsequent behavior). B3 is a non-blocking cross-runtime diagnostic that may be run after B2 when useful. Capture the S=1 reference before C; B3 is not a prerequisite for C.
- Runtime versions, identical installed source hashes, and recurrence callable identity are independently verified. Runtime call-path/flags and fastpath installation state remain model-backed capture fields.
- S=1 committed-vs-committed repetition characterizes S=1 determinism; repeated S=8 controls characterize the verify reference. Do not use S=1 repetition alone to set an S=8-versus-S=1 numeric tolerance.
- Existing deterministic hybrid test home and focused/full pytest commands are identified: `uv run pytest tests/test_bonsai.py -q`, `uv run pytest tests/test_bonsai.py tests/test_dflash2.py -q`, and `uv run pytest tests/ -q`.
- CLI supports fixed `--max-draft` and adaptive `--max-draft auto`; it does not expose the needed cache/tap/input trace. The user-specified physical sidecar must be confirmed loadable by the local CLI before using a benchmark command; if it is not, use a direct model-backed harness with that exact physical sidecar rather than substituting a registry drafter.

## Sources Inspected

- `specs/001-bonsai-rollback-parity/spec.md`
- `.specify/memory/constitution.md`
- `src/mlx_dspark/target.py`
- `src/mlx_dspark/generate.py`
- `src/mlx_dspark/dflash_model.py`
- `src/mlx_dspark/dflash_width_policy.py`
- `src/mlx_dspark/prism_pack.py` (control only)
- `tests/test_bonsai.py`, `tests/test_dflash2.py`, `tests/conftest.py`
- `src/mlx_dspark/cli.py`, `pyproject.toml`
- `~/git/chad/src/chad/engine.py`, `mlx_fastpath.py`, `mlx_dflash.py`, `prism_pack.py` (read-only)
- Installed `.venv/lib/python3.12/site-packages/mlx_lm/models/qwen3_5.py`, `gated_delta.py`, `cache.py`
