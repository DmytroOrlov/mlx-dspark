# Implementation Plan: Golden Prompt Panel

Use a feature-local Python runner modeled on feature-003's T027 product Engine path. Reuse only its Qwen/B-Q identity, `Engine.load` configuration, prompt encoding, freshness assertions, and immutable-write pattern. Do not import feature-002 benchmark request helpers or modify production runtime.

The runner has a frozen ordered prompt panel, model-free self-check, cheap local checkpoint/tokenization/source preflight, a child-process-per-observation speculative-only path, anchor scheduling, candidate certification and fallback repeat selection, immutable raw records, validation, and per-prompt Markdown/JSON reporting. All outputs live beneath `specs/004-golden-prompt-panel/evidence/`.

Physical protocol: 17 screen requests in panel order, with P01 screen result serving as start anchor, a separate P01 at the midpoint, and a separate P01 at the end. Then repeat candidates once in fresh children; only if fewer than two new prompts certify, repeat up to three fastest long-form non-goldens. P01 is retained as a known anchor and does not count among the two new prompts.

No serial control, architecture changes, prompt edits, speculative tuning, or aggregate qualification. Each observation stores exact rendered token IDs/hash, output hash/text, per-request metrics, runtime hashes, process identity/order and actual freshness. Raw files use create-only writes.
