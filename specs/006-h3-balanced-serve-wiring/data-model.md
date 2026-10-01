# Data Model: H3 Balanced Serve Wiring

No persistent data is introduced. The CLI translates two strings into the existing immutable `TargetCompositionRequest`.

## Donor selection

- `donor_model`: required repository or path string when composition is enabled.
- `donor_blocks`: required block-selection string when composition is enabled.
- Pair invariant: both are absent (ordinary serve) or both are present (composition); exactly one is invalid.
- Grammar: comma-separated elements, each either an integer index or `start-end` inclusive range.
- Validation: decimal integer tokens only; each index in 0–63; ranges ascending; no blank list elements.
- Normalization: expand, sort ascending, and deduplicate indices.

## TargetCompositionRequest (existing)

- `donor_path`, `donor_repo`, `donor_revision`: resolved donor snapshot identity.
- `qwen_repo`, `qwen_revision`: resolved target snapshot identity.
- `donor_indices`: normalized tuple of selected decoder block indices.
- Existing constructor/path validations remain authoritative.

Ownership is represented by the existing composition loader: selected same-index decoder blocks come from the donor; embedding, unselected blocks, final norm, and LM head remain Qwen-owned.
