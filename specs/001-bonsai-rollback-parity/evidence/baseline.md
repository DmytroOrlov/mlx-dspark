# Pre-probe provenance and hypothesis ledger (T003)

This is a read-only Git/source preflight recorded on 2026-09-23 before writing any temporary probe. Git state was not changed. The planned probe must treat the three mlx-dspark layers separately.

## Repository states

| Layer | Observed state |
|---|---|
| Vanilla mlx-dspark base | `68b6930cf09bf8deca90a6391c8ffa1196e117f6` |
| Experiment branch HEAD | `d05f06f6e7dd9b2ae1409dfb7b4cd960c44ff07e` on `bonsai2-dflash`, one commit after the base |
| Base to HEAD behavioral paths | `src/mlx_dspark/{calibrate.py,dflash_model.py,dflash_width_policy.py,generate.py,load.py,mlx_qmm_mma.py,prism_pack.py}` and `uv.lock`; inspected with read-only `git diff` |
| HEAD to current worktree/index | No tracked runtime/test edits at preflight. The staged index adds `.agents/skills/`, `.specify/`, and feature specification files; `plan.md`, `spec.md`, and `tasks.md` also have unstaged edits. This user work was not cleaned, staged, or restored. |
| Chad | HEAD `8d7c1d9c7dfa084893916a910c5bdb62a506397c` on `main`; current worktree modifies `benchmarks/spec_decode.py` (thinking disabled) and `src/chad/mlx_dflash.py` (policy trace), and contains untracked `benchmarks/spec_lru.py`. Chad was inspected and run only. |

The base-to-HEAD diff adds the mlx-dspark Prism loader and MMA path, physical sidecar quantized skeleton support, adaptive WidthPolicy, forced-width/width-log diagnostics, and the env-gated one-behind prefill mode. Thus observed experiment-branch behavior is not asserted to be vanilla-base behavior. The current uncommitted changes under the mlx-dspark index are spec/skill infrastructure, not additional runtime instrumentation. Chad's uncommitted trace wrapper and benchmark helper are diagnostics, not clean-HEAD behavior.

## Classification and stop conditions

| Classification | Finding and basis |
|---|---|
| KEEP | The exact physical generic Chad-built DFlash2 sidecar is the comparison drafter. The loader accepted its bytes directly (`controls.md`). Supplied fixed7 ~33.9/33.6 tok/s and 149/146 forwards is a historical practical-execution control, not a measurement made in this preflight or proof of adaptive correctness. The accepted mlx-dspark D0 pending-context concatenation behavior is in `generate.py`; Prism target loading and existing Prism MMA are controls, not current patch targets. |
| DIAGNOSTIC | `MLX_DSPARK_DFLASH_CHAD_PREFILL` in `generate.py` aligns one-behind lifecycle traces only and is not production behavior or a throughput fix. `MLX_DSPARK_DFLASH_POLICY_TRACE` in `dflash_width_policy.py`, `MLX_DSPARK_DFLASH_FORCE_WIDTH` and `MLX_DSPARK_DFLASH_WIDTH_LOG` in `generate.py`, Chad's uncommitted `CHAD_DFLASH_POLICY_TRACE`, and Chad's uncommitted `benchmarks/spec_lru.py`/modified `spec_decode.py` are diagnostic or benchmark helpers. The source also exposes `MLX_DSPARK_PRISM_ROT_FP32` and MMA calibration flags; any use must be recorded per run. |
| DISPROVED as a starting fix | Prior work reported grouped-convolution dtype, nesting the adaptive WidthPolicy under old CapController, raw-versus-converted generic sidecar after exact-sidecar fixed7 parity, one-behind prefill as production/throughput fix, and WidthPolicy threshold/acceptance-prior/cost-seed tuning for an earlier target/logit divergence. The new gate probe may revisit an earlier operation only if directly observed; none of these warrants a speculative patch now. |
| DEFERRED | Speculative `max_tokens` tail overshoot/completion-count issue is separate from the state/rollback investigation. |
| UNKNOWN | First actual differing semantic boundary in the physical R1→R2→pre-R3 sequence; whether fresh runs reproduce historical acceptances and margins; per-run mlx-dspark MMA dispatch; whether width-specific numeric bounds are stable. The maintainer supplied the historical user message/template settings, and `controls.md` records a fresh tokenizer-derived prefix, but there is no preserved historical token-ID dump. |

Chad `src/chad/engine.py:2440-2450` calls `stats.note_round` and `policy.record` only when `k > 0`. The historical width histogram and the uncommitted `spec_lru.py` recorder observe those policy-recorded rounds; absence of a `d0` histogram entry therefore does not establish that zero depth-0 rounds occurred. The supplied R1/R2/pre-R3 margins in `research.md` are historical observations and must not be promoted to fresh measurements.

No production path is authorized for an edit at this stage. A physical fix requires Gate A→B1→B2→C→D→E evidence and a reproducible causal regression as specified by `tasks.md` and the constitution.
