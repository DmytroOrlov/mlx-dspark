# Serve CLI Contract: Donor Blocks

```text
mlx-dspark serve [existing options] [--donor-model REPO-OR-PATH --donor-blocks SELECTION]
```

- The two options are optional and must be supplied together.
- `SELECTION` supports `62`, `56-63`, and comma-separated mixtures such as `12,20-23,62`.
- Indices are integers from 0 through 63. Ranges are ascending and inclusive.
- Empty elements and malformed tokens are errors before model resolution/loading.
- Duplicate indices are sorted and deduplicated.
- With neither option, existing ordinary serving behavior continues without a composition request.
- With both options, the CLI constructs the existing `TargetCompositionRequest` and passes it to `Engine.load(target_composition=...)`.
- Donor/drafter selection are independent.
- Startup reports donor ID, resolved donor revision when available from the resolved snapshot path, normalized selection, and Qwen ownership of embedding, final norm, and LM head.
- Composition requires donor and Qwen revisions to be identifiable from their resolved immutable snapshot paths; existing request and loader checks are preserved.
