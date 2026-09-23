# Investigation artifact migration

Migration date: 2026-09-23. The canonical root is `specs/001-bonsai-rollback-parity/evidence/`. `manifests/artifacts.json` records the original path, project path, SHA-256, byte size, task/gate/run, status, and purpose for each artifact below. The manifest excludes itself because a file cannot contain its own final SHA-256.

| Artifact written during this run | Project location | Status |
|---|---|---|
| `/private/tmp/bonsai_rollback_probe.py` (exact original bytes) | `raw/bonsai_rollback_probe.original.py` | Partial T004 probe |
| `/private/tmp/bonsai_rollback_probe.py` (project copy with local output default) | `probes/bonsai_rollback_probe.py` | Partial T004 probe |
| `/private/tmp/__pycache__/bonsai_rollback_probe.cpython-314.pyc` | `raw/bonsai_rollback_probe.cpython-314.pyc` | Completed compile artifact |
| `/private/tmp/chad-bonsai-policy-trace.jsonl` | `raw/chad-bonsai-policy-trace.jsonl` | Completed diagnostic trace |
| `/private/tmp/mlx-bonsai-policy-trace.jsonl` | `raw/mlx-bonsai-policy-trace.jsonl` | Completed diagnostic trace |
| `/private/tmp/bonsai-rollback-observations/chad.json` | `runs/chad.json` | Completed 16-token diagnostic run |
| `/private/tmp/bonsai-rollback-observations/mlx-dspark-aligned.json` | `runs/mlx-dspark-aligned.json` | Completed 16-token diagnostic run; previous bytes at this path were overwritten |
| `/private/tmp/bonsai-rollback-observations/kv0/chad.json` | `runs/kv0/chad.json` | Completed 16-token diagnostic run |
| `/private/tmp/bonsai-rollback-observations/kv8/mlx-dspark-aligned.json` | `runs/kv8/mlx-dspark-aligned.json` | Completed 16-token diagnostic run |
| `/private/tmp/bonsai-rollback-observations/no-fastpath-kv0/chad.json` | `runs/no-fastpath-kv0/chad.json` | Completed 16-token diagnostic run |
| `/private/tmp/bonsai_reference_probe.py` (attempted write; source file absent) | `probes/bonsai_reference_probe.py` | Interrupted marker only; no source bytes existed to copy |
| T001 control record | `controls.md` | Completed |
| T002 runtime record | `runtime-profiles.md` | Completed |
| T003 provenance record | `baseline.md` | Completed |
| Feature checklist edits | `../tasks.md` | Partial implementation; T001–T003 remain checked |
| T017 width-8 test edit | `../../../tests/test_bonsai.py` | Partial; accepted=0 case failed before migration |
| Investigation path guidance | `../plan.md` | Completed path migration |
| Investigation command guidance | `../quickstart.md` | Completed path migration |
| Unrelated ignore additions, now removed by direct edit | `../../../.gitignore` | Completed restoration of this run's additions |
| This migration report | `migration-report.md` | Completed |

All existing T001–T003 records and T017 code remain in place. The five JSON runs are diagnostic captures of the first three rounds under a 16-token limit, not the final 512-token golden benchmark or completed Gate A–E evidence. The exact historical user message and template settings are in `controls.md`; historical acceptances and margins remain observations to measure rather than forced inputs. `MLX_DSPARK_DFLASH_CHAD_PREFILL=1` remains a **DIAGNOSTIC** alignment mode and is **DISPROVED** as a production or performance fix.

The first `/private/tmp/bonsai-rollback-observations/mlx-dspark-aligned.json` run used a preallocated context cache that disabled the intended one-behind alignment. A corrected run overwrote that same path. The overwritten bytes cannot be migrated; the retained JSON is the corrected run. An attempted patch for `/private/tmp/bonsai_reference_probe.py` was interrupted before any file appeared at that path. The project file is an explicit, nonfunctional interruption marker, not recovered reference-probe code. No invented token IDs, accept counts, margins, or missing source bytes were inserted.

Model weights loaded into process memory, lazy MLX computation graphs, full unreduced logits and taps, full live KV/recurrent/convolution cache arrays, and the pre-overwrite JSON bytes were lost when diagnostic processes stopped. The retained JSON captures selected metadata, hashes, shapes, samples, and top-two values, but cannot reconstruct those full arrays. Command output shown only in the terminal was not saved as a file and therefore has no bytes to migrate.

**Exact resume point:** T004 is still open: finish the project-local boundary probe and its `ControlledRun`, `ProposalInputRecord`, `BoundarySnapshot`, and `GateResult` fields. Then implement T005's independent fresh same-runtime, same-width S=8 reference in `probes/bonsai_reference_probe.py`. T006–T007 controls and the continuous controlled gate trace remain outstanding. T017's accepted=0 test failure remains to resolve in the authorized investigation; no model, benchmark, or test was run during this migration.

Future diagnostic outputs should use `evidence/runs/` or `evidence/raw/`; the project probe now defaults to `evidence/runs/`. Policy trace environment variables should likewise point to project-local `evidence/runs/...` paths when runs resume. No runtime or model semantics were changed for this migration.
