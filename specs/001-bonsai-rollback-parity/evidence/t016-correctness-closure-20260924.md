# T016 correctness closure — 2026-09-24

## Outcome

**RESOLVED_NO_CHANGE.** No production correctness patch is authorized.

Both physical runtimes pass the required isolated B2 → C → D → E chain for both forced acceptance branches.

| Runtime | B2 | C keep=1 | C keep=2 | D keep=1 | D keep=2 | E keep=1 | E keep=2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| chad | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| mlx-dspark | PASS | PASS | PASS | PASS | PASS | PASS | PASS |

## Evidence

- Chad: `/Users/do/git/mlx-dspark/specs/001-bonsai-rollback-parity/evidence/runs/required-gates-chad-kv0-retry2-20260923T231232Z-fda745d7/run.json`
- Chad SHA-256: `13908e890c559d3a9e9dcac68f91d6581f95b0ad86bda6b9641309c0f6052189`
- mlx-dspark: `/Users/do/git/mlx-dspark/specs/001-bonsai-rollback-parity/evidence/runs/required-gates-mlx-prod-kv0-20260923T231814Z-246758b0/run.json`
- mlx-dspark SHA-256: `11d83a6d93c9d26e05eaa3ef4c0a46877264a9f99ea7b2ba1c768a71e07298d9`

B2 uses the exact frozen S=8 root and five-run same-root controls. C compares runtime rollback/reconcile state against an independent same-S8 accepted-prefix reconstruction. D executes the same next committed token from production and independent post-C states with controlled target-state comparisons. E independently reconstructs the equivalent fresh drafter/context state and next proposal.

Earlier `gate_results` fields that stopped before C/D/E are historical. For B2/C/D/E they are superseded by `required_gate_acquisition` from these later physical runs.

Natural acceptance histories still differ between Chad and mlx-dspark. That is no longer evidence of a rollback defect: both runtimes pass both forced state-transition branches.

## Next phase

Correctness is closed. The next investigation is performance/operating-point only: target KV precision, margin trajectory, adaptive width selection, acceptance, target-forward count, and decode throughput.
