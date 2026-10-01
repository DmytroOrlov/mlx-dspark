# Feature: Golden Prompt Panel

## Goal

Find at least two additional prompts that sustain at least 40 speculative decode tokens/sec and naturally generate at least 410 tokens on current H0+B-Q under feature-003 product request semantics. Every prompt remains an individual observation; aggregate throughput is not a qualification or headline.

## Fixed experiment

- Target `mlx-community/Qwen3.8-27B-4bit` revision `10c35caafbb80f7dc6a7a432cdd11af10a6d4818`.
- Drafter `incoai/Qwen3.8-27B-DFlash2` revision `015e795645c74b1a0eeef3b570031fb62e769bc5`.
- Feature-003 T027 product Engine semantics, fresh process/Engine and empty measured request, enabled normal prefix-cache initialization with zero reused prompt tokens, thinking off, temperature 0, top-p 1, top-k 0, max tokens 512, plain KV, ordinary automatic CapController, no WidthPolicy/KV8/tuning.
- Run the 17 exact prompt strings frozen in `evidence/probes/prompt_panel.py`; record rendered IDs and their SHA-256 for each prompt.
- Discovery is one speculative-only request per prompt, with P01 anchors at start, midpoint and end. Repeat each qualifying candidate once. If fewer than two new prompts certify, repeat the three fastest long-form non-goldens nearest 40 tok/s. No text changes or padding.

## Qualification

`GOLDEN_CANDIDATE` requires >=410 generated tokens and >=40.0 speculative tok/s on screen. `CERTIFIED_GOLDEN` requires both fresh observations independently satisfy both thresholds. Fewer than 410 tokens is `TOO_SHORT`, regardless of speed. Other statuses: `SLOW`, `FAILED`.

## Evidence and output

Write immutable per-observation records and machine-readable preflight/validation/repeat decisions under this feature's `evidence/`. Produce a per-prompt human table with every prompt visible and classified. Report screen/repeat rates separately; never qualify from averages. No production/runtime or feature-003 evidence is modified.
