# Tasks: Golden Prompt Panel

## Phase 1 — Runner and checks

- [X] T001 Implement a complete feature-local panel runner, frozen prompt text/order, T027 product request semantics, process-per-observation execution, immutable evidence, anchor schedule, candidate/repeat selection, validator, and per-prompt report generation.
- [X] T002 Add lightweight model-free checks for all 17 exact prompts, IDs/hash runtime validation path, generation controls, process isolation, no serial path, thresholds/statuses, anchors, repeat policy, and report row structure.
- [X] T003 Run model-free checks and cheap preflight for exact local Qwen/B-Q revisions/files, tokenization IDs/digests, relevant source hashes, fresh child/Engine construction, enabled prefix-cache initialization, plain KV and ordinary controller/no tuning. After correcting the pre-execution output-hash defect, checks/preflight passed again; frozen runner SHA: `f80ba3c06517f5faa007af3086fe651aeba4aa5a5ea58cb54da0f38df3d6b984`.

## Phase 2 — Prompt discovery

- [X] T004 Execute the 17 speculative-only H0+B-Q screen observations and the separate midpoint/end P01 anchors in fresh processes; retain every result and assign screen/anchor status. Attempts 1–2 failed before usable metrics and are preserved; corrected attempt 3 completed all 19 observations.
- [X] T005 Repeat screen GOLDEN_CANDIDATE prompts once in fresh processes and certify only when both observations independently meet >=410 generated tokens and >=40.0 tok/s. Nine candidates were repeated and all nine certified.
- [X] T006 If fewer than two new prompts certify, execute the prescribed repeats for the three fastest long-form non-goldens nearest 40 tok/s; no prompt edits. Not triggered: nine new prompts certified.
- [X] T007 Validate evidence completeness and generate per-prompt machine-readable and Markdown reports with every prompt visible, grouped by status and with no aggregate headline. Validation passed for 17 screen rows and nine repeats.

## Completion

The task concludes after T007. New prompts, physical runs, or architecture comparisons require a separate follow-up.
