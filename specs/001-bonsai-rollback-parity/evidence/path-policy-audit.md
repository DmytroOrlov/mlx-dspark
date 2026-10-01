# Evidence path policy audit

- **Normative occurrences fixed:** The plan, quickstart, T004, and active probe now direct serializable investigation evidence to the feature evidence tree. Probe runs default to a fresh unique directory and refuse an existing path; explicit output paths must remain under `evidence/runs/`. Policy-trace variables have no `/tmp` examples; future values belong in the selected unique run directory.
- **Historical occurrences retained:** Migration/loss descriptions, resume records, manifests' `original_path` values, and the interrupted `/private/tmp/bonsai_reference_probe.py` account remain unchanged as provenance.
- **Raw artifacts untouched:** `evidence/raw/bonsai_rollback_probe.original.py` and imported raw traces/bytecode were not modified.
- **Current-instruction check:** No active/current instruction writes canonical investigation evidence outside `specs/001-bonsai-rollback-parity/evidence/`.
