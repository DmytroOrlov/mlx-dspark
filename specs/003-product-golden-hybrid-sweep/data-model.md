# Data Model: Product Golden Hybrid Sweep

## Product request

Represents the immutable request controls shared by every measured cell.

- `workload`: `Write a production-quality Python LRU cache with tests and type hints.`
- `input_ids`: exact T027 26-token list from `spec.md`
- `input_ids_sha256`: `8bda1dedde8e022452b7dafaa3677dca47dedc350258c0885c4d1e5fde155f59`
- `temperature`, `top_p`, `top_k`, `thinking`, `max_tokens`: `0`, `1`, `0`, disabled, `512`
- Runtime controls: production Engine, ordinary CapController, plain KV, normal T027 warmup and memory guard

Validation: rendered/tokenized IDs must equal the pinned sequence; every cell must use this same request digest and controls.

## Target composition

One of seven named target identities. H0/H1a/H1b/H1c/H2/H3 are Qwen hybrids: they retain Qwen embedding, final norm, and LM head, and replace only the specified decoder blocks with Bonsai2 blocks. B0 is the full official Bonsai2 endpoint: native Bonsai2 embedding, all decoder blocks, final norm, and LM head, matching the accepted feature-002 B0 endpoint control. B0 is not a 64-block hybrid or “H64”. `bonsai_owned_blocks` gives decoder-block ownership where relevant.

| Target | Bonsai-owned blocks |
|---|---|
| H0 | none |
| H1a | 63 |
| H1b | 62 |
| H1c | 62–63 |
| H2 | 60–63 |
| H3 | 56–63 |
| B0 | 0–63 (native full Bonsai2 target, including native non-block components) |

Validation: hybrid ownership must match the feature-002 declarative manifest and 64-block inventory; B0 identity must match the full official Bonsai2 endpoint. H1b is an interaction diagnostic, not a monotonic suffix point.

## Drafter checkpoint

One of two immutable drafter identities paired with a target composition:

- B-Q: `incoai/Qwen3.8-27B-DFlash2`, revision `015e795645c74b1a0eeef3b570031fb62e769bc5`.
- B-B: `naklitechie/Qwen3.8-27B-DFlash2-ternary-bonsai2`, revision `0059b38aa255698b1a87305eb3fbb5a3cfd616e2`; selected weight SHA-256 `eb141d0fb8621cf7a7c81047b83810fa19b8093543bd05529e9478d9555cb8f1`.

## Product cell observation

One fresh process/cache measurement for one target composition and one drafter. The first H0+B-Q observation is both the Golden-A canary and matrix cell 1. If it passes, retain it and run only the other 13 rows; the first-pass matrix has exactly 14 mandatory physical observations total. Required identity includes cell ID/order, revisions and verified files, target identity/ownership (native full-target identity for B0), runtime source hashes and git dirty state, machine, request digest, and explicit process/cache freshness assertion.

Required measurements: serial and speculative tok/s, speedup, delta versus fresh H0+B-Q, acceptance, mean accepted, target forwards, generated tokens and generated/forward, width/cap distribution, peak GiB, duration, and status/notes. With a passing canary included as cell 1, matrix cardinality is exactly 14.

## Canary adjudication

Records the one fresh H0+B-Q result, which is also matrix cell 1, and one decision: `STOP_MISMATCH` with concrete request/runtime differences when it remains around 32–36 tok/s / ~49% acceptance; or `CONTINUE_SWEEP` when it returns to the historical ~45 tok/s regime. On pass, retain this result as the delta reference and proceed with only 13 remaining rows. Ambiguous outcomes stop for adjudication and do not launch hybrid cells. An additional final fresh H0+B-Q drift bracket is optional only if substantial runtime duration/order drift is suspected or it is needed to adjudicate a product-competitive result; it is not a mandatory matrix cell.

## Historical context and report

Historical Golden A is 45.7788518 speculative tok/s, with the values and provenance in `spec.md`. Golden B is approximately 45.4 tok/s with 25/26 prompt tokens hot-cached and is context only. Neither historical row is a fresh paired delta reference; current deltas use the fresh H0+B-Q canary.

The report presents the human-readable path by drafter and a machine-readable artifact. The B-Q and B-B paths use H0 → H1a → H1c → H2 → H3 → B0; H1b is displayed as an interaction diagnostic. The frozen 14-row execution order is: H0+B-Q, H0+B-B, H1a+B-B, H1a+B-Q, H1b+B-Q, H1b+B-B, H1c+B-B, H1c+B-Q, H2+B-Q, H2+B-B, H3+B-B, H3+B-Q, B0+B-Q, B0+B-B. Each row uses a fresh process/cache.
