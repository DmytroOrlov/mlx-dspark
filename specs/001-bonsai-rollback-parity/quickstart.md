# Quickstart: Rollback Parity Validation

This is a future investigation/validation procedure. No probe, test, benchmark, or M4 Pro command was run while preparing this plan. The state-trace harness described below does not exist yet; keep its source under `evidence/probes/` and write each invocation's output directly to a new unique directory under `evidence/runs/`.

## Project-local evidence policy

Model-backed Bonsai runs are expensive. Earlier temporary-harness work was interrupted or compacted before durable capture, causing unsaved probe source, overwritten run bytes, and process-only state to be lost. Investigation artifacts are first-class reproducibility evidence, not disposable scratch files. Every artifact that can be serialized must be written directly to this feature's evidence tree: probe/helper source in `evidence/probes/`, each run in a unique `evidence/runs/<unique-run-id>/` directory, imported raw history in `evidence/raw/`, and manifests/checksums in `evidence/manifests/`. `/tmp` or `/private/tmp` may appear only as historical provenance or for OS/runtime internals outside our control, never as the sole or canonical store for an investigation artifact. Unique run directories prevent overwrite collisions. This changes storage and artifact lifecycle only; it does not change gate semantics, model behavior, or production architecture. Process-only RAM state cannot always be preserved.

## Prerequisites

- Local M4 Pro with both comparison environments available:
  - Chad: Python 3.11.15, MLX 0.32.2, mlx-lm 0.31.3.
  - mlx-dspark: Python 3.12.13, MLX 0.32.2, mlx-lm 0.31.3.
- Confirm the installed `qwen3_5.py`, `gated_delta.py`, and `cache.py` hashes match the evidence in [research.md](research.md). In both runtimes, verify `qwen3_5.gated_delta_update is gated_delta.gated_delta_update`; capture its module/file/line/signature and actual call path/flags.
- Physical target config `prism_hadamard_qwen35`, routed through each runtime's Prism loader, and exact sidecar `~/.cache/chad/dflash/015e795645c74b1a0eeef3b570031fb62e769bc5-q4g64`.
- Confirm the sidecar is accepted by the mlx-dspark loader. If not, use a direct model-backed diagnostic adapter for that exact sidecar; do not silently resolve or substitute a registry drafter.
- Same tokenizer, prompt IDs, sampling, logical prefix/cache state, warmup assumptions, and relevant kernel settings within each paired comparison. The Python minor versions are known to differ. Use a same-width S=8 target execution as each runtime's verify/capture reference, and a separate ordinary S=1 committed execution for post-reconcile semantics and subsequent committed behavior.
- Chad checkout available for source inspection and diagnostic execution only. Never write under `~/git/chad`.

## 1. Establish reproducible state and record the actual Chad fastpath

Record target and sidecar identifiers/fingerprints; Python, MLX and mlx-lm versions; source hashes; device; model config and loader; tokenizer; tap IDs; prompt/token IDs; sampling; cache/prefix state; and active kernel flags.

For each Chad model-backed run, explicitly record `Engine.fastpath`, `CHAD_NO_FASTPATH`, and the installed fastpath class/call-path identity. Source support is not proof that installation succeeded. Record whether the supported Prism path actually applied fused qkv|z, b|a, and q/k/v attention projections, Prism rotation/sign folding, or compiled/fused layer paths.

Create separate equivalent prefill/cache snapshots for the speculative run, its same-width S=8 verify reference, and the ordinary S=1 committed-semantic reference. Never reuse a cache mutated by another branch. The S=8 reference receives the exact same `verify_ids` and captures outputs/taps/cache effects before reconciliation. The S=1 path uses that runtime's ordinary committed one-token target path and is used only for post-reconcile semantics and subsequent behavior.

Run Chad's normal-fastpath and `CHAD_NO_FASTPATH=1` diagnostics in separate fresh processes. Set the environment before loading the target and before `mlx_fastpath.install()` can mutate it; load a new target model for each branch. Do not toggle the variable after installation and treat the already loaded model as an independent control.

## 2. Gate A: capture proposal inputs before target verify

At the same logical generated-token position, record in this order:

1. Target/drafter input-state identity and logical position.
2. Current anchor/pending token ID.
3. All seven proposed draft token IDs.
4. Exact `verify_ids = [anchor] + draft`.
5. Proposal-context logical position.
6. Accepted count, only after the preceding values have been captured.

Matching `k=7` and matching accepted count do not establish identical verify inputs. Compare exact cross-runtime semantic/logical controls first, then record each runtime's proposal and assert its own `verify_ids = [anchor] + drafts`. Different valid target projection/op graphs may produce different numerical contexts and proposals. If proposals differ, require independent same-runtime R1 reconstruction in both runtimes: exact local IDs/positions/cache structure, production-context SHA equality or a predeclared applicable stable five-run control, and exact local proposal reproduction. When these pass, retain the cross-runtime proposal difference as an implementation-dependent diagnostic and continue to B1. An earlier exact semantic/logical difference or a violation of mlx-dspark's own independent reference stops A for localization. Matching committed outputs alone do not establish A.

The proposed harness invocation remains a future interface, not a current command:

```sh
RUN_ID="$(uuidgen | tr '[:upper:]' '[:lower:]')"
uv run python specs/001-bonsai-rollback-parity/evidence/probes/bonsai_rollback_probe.py \
  --target nathansutton/Qwen3.8-27B-Ternary-Bonsai-2-DFlash2-MLX \
  --drafter ~/.cache/chad/dflash/015e795645c74b1a0eeef3b570031fb62e769bc5-q4g64 \
  --verify-width 8 --draft-width 7 --rounds 2 \
  --accept-sequence 0,1 --temperature 0 \
  --s1-controls 5 --s8-controls 5 --output-dir "specs/001-bonsai-rollback-parity/evidence/runs/$RUN_ID"
```

## 3. Gates B1–B2: exact verify inputs and same-width references

**B1 — exact verify IDs and equivalent pre-verify state.** For each runtime, confirm the speculative branch and its reference start from equivalent pre-verify target state and receive the exact same `verify_ids = [anchor] + seven draft IDs`. Resolve a mismatch here before interpreting verify behavior.

**B2 — same-width verify/capture reference within each runtime.** Run a fresh/cloned target execution in each runtime from an equivalent pre-verify state. It must:

- Receive the exact same eight verify IDs as the speculative target verify.
- Execute the same sequence width (8 rows).
- Capture the same target outputs, taps, and cache effects before reconciliation.
- Use independently allocated target/cache state and not use the speculative rollback/reconcile result as its reference.

Run width 8 (one anchor plus seven drafts) with accepted=0 and accepted=1 from equivalent fresh states. Compare token IDs, top-k IDs, cache classes/shapes/live lengths/offsets/positions, recurrence and convolution state, taps, and accepted-prefix invariants exactly where deterministic. If a runtime's speculative verify differs from its same-width reference before reconciliation, stop rollback analysis and localize verify, capture, or projection behavior. Repeated same-width S=8 controls establish determinism of the verify reference. Do not use raw S=1 committed logits as the pre-reconciliation numeric oracle.

Also capture each runtime's independent ordinary S=1 committed-semantic reference from fresh state for both prompt + anchor and prompt + anchor + first accepted draft. Gate C evaluates reconciled state against these references immediately after B2. B3 is a non-blocking cross-runtime diagnostic and may be run after B2 when useful, but it is not a prerequisite for C.

**B3 — cross-runtime verify observations.** Only after proposal IDs and both within-runtime references are understood in each runtime—the same-width S=8 verify/capture reference and the ordinary S=1 committed-semantic path—compare token identities, top-k identities and margins, then q/k/v/a/b and other raw tensors/logits as diagnostics. Both runtimes use byte-identical mlx-lm sources and the same gated-delta callable, but Chad's supported Prism fastpath changes projections/op paths before gated-delta update and mlx-dspark does not execute that identical graph. Do not require bit-identical cross-runtime raw logits; a numeric mismatch across these valid graphs is not automatically a correctness defect.

Run the Chad semantic boundary probe with normal fastpath and with `CHAD_NO_FASTPATH=1` in separate fresh processes, loading a new target model in each with the environment set before installation. Restore equivalent target weights, input IDs, sampling, and cache snapshot. Record fastpath state in both runs and compare the target margin and captured projection inputs to determine whether the near-tie is sensitive to the alternate graph. This branch is diagnostic; throughput is not correctness evidence.

## 4. Gate C and later boundaries

For accepted=0/keep=1 and accepted=1/keep=2, first require exact reconciled logical structure: accepted-prefix IDs, cache type/shape/order, live lengths, offsets, positions, and required tap/fused-row identity/order. For rollback-owned numeric state, build an independent same-runtime accepted-prefix oracle from the frozen S=8 B2 capture without invoking production rollback: replay captured S=8 recurrence inputs from the captured pre-round recurrent state, derive convolution history from captured S=8 `conv_input`, and trim frozen S=8 attention KV to the accepted live prefix. Compare required accepted taps/fused rows with their corresponding same-width S=8 B2 rows.

Rollback/reconcile is eligible for a patch only when proposal/verify inputs and B2 are controlled and mlx-dspark post-reconcile state violates the independent same-runtime same-S8 accepted-prefix oracle at the earliest boundary. Fresh ordinary S=1 and fresh S=2 executions remain serial semantic/behavior and projection-width diagnostics. A raw post-reconcile-versus-fresh-S1/S2 cache delta is not by itself a rollback defect when the production state originated in S=8 projections.

At Gate D, compare subsequent ordinary committed target behavior from the reconciled branch with the same-runtime ordinary committed reference, including token identity, controlled top-k/margin, taps, and resulting semantic state. Keep historical cross-runtime margin differences as diagnostics. Gate D is the place where serial behavior is adjudicated; it does not replace Gate C's same-S8 rollback-state reconstruction.

Apply correctness Gates A, B1, B2, C, D, and E in order. B3 remains optional. Gate C uses the frozen same-runtime S=8 B2 capture as the root for its independent accepted-prefix rollback oracle; fresh S=1/S=2 controls characterize serial behavior and projection-width sensitivity instead of supplying the Gate C raw numeric cache-state threshold. If the applicable same-path control is unavailable or unstable, mark that observation INCONCLUSIVE rather than loosening a global tolerance.

## 5. Portable regression scope

The existing `_tiny_hybrid()` fixture uses stock mlx-lm Qwen3.5, not the physical Prism/Chad-fastpath target path. Use it only to exercise generic rollback-state invariants. A passing tiny-hybrid test is not reproduction of the physical Bonsai divergence. A causal regression for a physical-only defect must follow the earliest failing boundary from the model-backed trace.

Planned implementation-phase test commands (not run during planning):

```sh
uv run pytest tests/test_bonsai.py -q
uv run pytest tests/test_bonsai.py tests/test_dflash2.py -q
uv run pytest tests/ -q
```

## 6. Required final golden hardware comparison

Keep the established golden case separate from the rollback boundary probe and include it in final hardware parity validation:

- Prompt: `Write a production-quality Python LRU cache with tests and type hints.`
- Thinking: disabled.
- Temperature: 0.
- `max_tokens`: 512.
- Drafter: the same exact physical Chad-built sidecar in both runtimes, verified by fingerprint.

Use the established Chad/mlx-dspark comparison procedure and record generated output/token identity and run metadata. Do not invent an expected output or substitute another sidecar. This exact golden comparison is required; the generic benchmark CLI below is only an additional control.

## 7. Optional fixed7 and adaptive controls

After deterministic gates establish correctness, a fixed7 performance/control run may be collected with the exact sidecar, only if the local CLI loads it:

```sh
mlx-dspark benchmark \
  --model nathansutton/Qwen3.8-27B-Ternary-Bonsai-2-DFlash2-MLX \
  --drafter ~/.cache/chad/dflash/015e795645c74b1a0eeef3b570031fb62e769bc5-q4g64 \
  --modes dflash --caps 7 --trials 3 --max-new-tokens 200
```

This is a fixed-width performance/control run and does not replace the golden case or correctness probes. For the adaptive symptom, use the same controlled prompt and record round widths, margin, acceptance, and token stream from both implementations. Use Chad's actual invocation from its inspected CLI; mlx-dspark's command form is:

```sh
mlx-dspark generate \
  --model nathansutton/Qwen3.8-27B-Ternary-Bonsai-2-DFlash2-MLX \
  --drafter ~/.cache/chad/dflash/015e795645c74b1a0eeef3b570031fb62e769bc5-q4g64 \
  --mode dflash --max-draft auto --temperature 0 --max-new-tokens 200 \
  --prompt '<same controlled prompt>'
```

Adaptive CLI output alone is not a substitute for the boundary trace. Report only observations collected on the M4 Pro; commands and supplied fixed7 values are not new parity or performance results.

## Post-correctness operating-point comparison

Do not tune target quantization or target-KV precision while trying to make a correctness gate pass. Once deterministic correctness and the exact golden validation are complete, evaluate product configurations separately.

The primary product comparison is the established Qwen3.8-27B 4-bit + generic DFlash2 production baseline versus two Bonsai points: plain target KV and quantized target KV. The two Bonsai points must use the same exact generic Chad-built physical sidecar so target-KV precision is the intended variable. Add a 4-bit + quantized-KV point only if supported and useful. For Bonsai, keep fixed7 and adaptive measurements distinct.

For every measured point record actual cache class/precision, target/drafter fingerprints, throughput, prefill/TTFT, acceptance/widths/target forwards, peak memory, context window, and prefix-cache conditions. Choose among these as product operating points; do not use the result as correctness evidence for an earlier gate.

