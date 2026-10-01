# Research: H3 Balanced Serve Wiring

## Decision: Keep composition assembly in the serve CLI

**Rationale**: `cmd_serve` already parses serve-only arguments and assembles the kwargs passed to `Engine.load`. `Engine.load` already accepts `target_composition` and dispatches the existing donor block loading plus `load_target` block-replacement route. The default branch calls `load_target` without replacements.

**Alternatives considered**: Add a second loader or payload, or put a named preset in the product CLI. Both add unnecessary surface and conflict with the revised paired-option interface.

## Decision: Parse and validate selections before resolving model identifiers

**Rationale**: The CLI can reject an incomplete pair or malformed selection immediately after argparse parsing. A compact parser can support indices, ascending inclusive ranges, and comma-separated mixtures while bounding every integer to 0–63. Sorting and deduplicating gives deterministic duplicate behavior without another error class.

**Alternatives considered**: Reject duplicates; normalize them instead, as the existing request normalizes ordering and the feature asks for the simplest deterministic policy.

## Decision: Reuse existing resolution and immutable identity requirements

**Rationale**: `load._resolve` uses the existing local model lookup and Hugging Face snapshot resolution; `resolve_mode` resolves ordinary target/drafter selection, not filesystem paths or revisions. Hub snapshot paths carry their immutable SHA and can supply the request revision. `TargetCompositionRequest.create` requires immutable 40-character donor and Qwen revisions, and existing donor and Qwen loading checks require those revisions to occur in resolved snapshot paths. A local path without that SHA cannot satisfy the current request and must fail closed. `checkpoint_identity()` is a file-manifest digest, not a Hub revision. Preserve the current guards. Do not add a parallel resolver or temporary path.

**Alternatives considered**: Relax request revisions or path checks to accept mutable/unverifiable paths; rejected because it weakens existing integrity guarantees. Local paths without an immutable SHA in the resolved path remain ineligible for composition.

## Decision: No changes to loader or server layers

**Rationale**: `Engine.load(target_composition=...)` already requires the existing request type and loads donor blocks before calling `load_target` with block replacements and identity. `load_target` already verifies Qwen snapshot identity. `replace_blocks` keeps Qwen non-block components and every non-selected block by object identity.

**Alternatives considered**: Extend `server.py`, `hybrid_target.py`, or `load.py`; direct seam inspection found no required interface change.
