# Product Golden and Historical Prompt Sensitivity

## A. Where did 45 tok/s go?

| Observation | Speculative tok/s | Serial target tok/s | Speedup |
|---|---:|---:|---:|
| Historical Golden A | 45.7788518 | 15.4992848 | 2.95361× |
| Historical Golden B | ~45.4 | — | — |
| Fresh feature-003 H0+B-Q | 43.9179011 | 14.9187419 | 2.943807× |

Golden B is hot-prefix context only: 25 of its 26 prompt tokens were cached. It is not a fresh paired baseline. Fresh H0+B-Q is 1.8609507 tok/s (4.07%) below Golden A. Its serial target rate is 0.5805429 tok/s (3.75%) lower; speedup is lower by about 0.33%.

Descriptive arithmetic, not causal attribution: applying Golden A’s 2.95361× multiplier to today’s 14.9187419 tok/s serial observation gives about **44.0641 tok/s**. The measured rate is **43.9179 tok/s**. Of the 1.861 tok/s historical gap, about **1.715 tok/s (92.1%)** is accounted for arithmetically by the lower serial observation at the historical multiplier; about **0.146 tok/s** remains in the multiplier comparison. This decomposition does not establish why either measurement differs.

Acceptance changed only from 0.64823 to 0.63885; target forwards changed from 94 to 98. The current automatic controller used width/cap 2 for its first four rounds, then width/cap 7 for 93 rounds. Historical Golden A recorded width 7 in all 93 rounds.

The immutable canary adjudication’s `continuation_rows: 0` is a metadata-only naming/count defect: that value is the number of stopped rows, not the number of rows left to run. The 13 remaining cells were run once each and are present in the complete matrix.

**The ~45 tok/s regime did not disappear. A fresh exact Golden-A reproduction is 43.9 tok/s, about 4% below the historical 45.8 measurement, with nearly the same ~2.95× speculative multiplier. Most of that numerical difference tracks the currently slower serial target observation rather than a collapse of speculative decoding.** These are descriptive comparisons, not causal findings.

## B. Where did 40+ live?

The historical raw files contain **66** observations for p02 with SHA-256 `b8a64e59879ee812dd745bebd7e32e3411a1a976838b0e37095ac636fe907d97`: 56 Series-A observations and 10 Series-B observations. **57** raw p02 rates are at least 40 tok/s. The supplied threshold TSV has 57 rows; every row is p02 with that same SHA, and all 57 entries match the raw observations when rates are rounded to the TSV’s six decimal places. There is **one unique prompt SHA** among those >=40 hits.

The 40+ observations were concentrated in this one speculation-friendly coding prompt, not spread uniformly across prompts. The complete per-run ledger—including run ID, target, drafter and immutable revision, aggregate run rate, p02 rate, mean accept length, forwards, generated tokens, generation limit, and raw result path—is in [historical-prompt-sensitivity.json](historical-prompt-sensitivity.json).

Historical target/drafter combinations with at least one p02 observation at 40+ tok/s:

- **Series A, B-Q:** H0, H1a, H1b, H1c, H2, H3, and B0.
- **Series B, B-Q:** H0, H2, and B0.
- **Series B, B-B:** H0 and B0.

## C. Aggregate speed hid the fast prompt

These Series-B values were re-read from raw JSON. They are context only: this was a three-prompt, 128-new-token protocol, not the feature-003 LRU/512-token matrix.

| Target + drafter | Aggregate tok/s | p01 tok/s | p02 tok/s | p03 tok/s | p02 mean accepted | p02 forwards | p02 generated |
|---|---:|---:|---:|---:|---:|---:|---:|
| H0+B-Q | 32.134 | 28.731 | 44.335 | 27.996 | 6.095 | 22 | 128 |
| H0+B-B | 29.629 | 25.847 | 42.452 | 25.712 | 5.909 | 23 | 130 |
| H2+B-Q | 28.522 | 27.049 | 44.011 | 22.044 | 6.045 | 23 | 133 |
| B0+B-Q | 33.953 | 28.576 | 55.545 | 28.109 | 5.739 | 24 | 132 |
| B0+B-B | 35.656 | 31.522 | 58.366 | 28.211 | 6.000 | 23 | 132 |

The p02 prompt generated tokens faster and needed fewer target forwards, with higher mean accepted length, than p01 and p03 in these examples. The aggregate combines those different prompt behaviors, so a 28–35 tok/s aggregate does not mean the runtime had lost its ability to produce 40–50+ tok/s on a favorable prompt. These historical p02 measurements do not predict feature-003 LRU performance.

## D. Primary current product table: Golden-A LRU/512

Context rows (not part of fresh paired deltas):

| Context | Speculative tok/s | Note |
|---|---:|---|
| Historical Golden A | 45.779 | Original product observation |
| Historical Golden B | ~45.4 | Hot-prefix context only; 25/26 prompt tokens cached |
| Fresh H0+B-Q | 43.918 | Immutable current matrix row 1 and fresh denominator |

The matrix below uses the same fresh Golden-A request and controls. Retention is speculative tok/s divided by fresh H0+B-Q (43.9179011). Widths are draft widths by round.

### B-Q progression: H0 → H1a → H1c → H2 → H3 → B0

| Target | Bonsai blocks | Drafter | Serial tok/s | Spec tok/s | Δ vs H0 | Retention | Acceptance | Forwards | Generated/forward | Widths | Peak GiB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| H0 | none | B-Q | 14.919 | 43.918 | 0.000 | 100.00% | 0.639 | 98 | 5.296 | 2:4, 7:93 | 15.921 |
| H1a | 63 | B-Q | 14.949 | 41.145 | −2.773 | 93.69% | 0.589 | 103 | 4.971 | 2:4, 7:98 | 15.821 |
| H1c | 62–63 | B-Q | 15.197 | 37.618 | −6.300 | 85.65% | 0.522 | 113 | 4.531 | 2:4, 7:108 | 15.722 |
| H2 | 60–63 | B-Q | 15.383 | 39.008 | −4.910 | 88.82% | 0.548 | 109 | 4.697 | 2:4, 7:104 | 15.522 |
| H3 | 56–63 | B-Q | 15.783 | 40.745 | −3.173 | 92.78% | 0.568 | 107 | 4.832 | 2:4, 7:102 | 15.136 |
| B0 | 0–63, native full Bonsai2 | B-Q | 24.944 | 34.545 | −9.373 | 78.66% | 0.379 | 144 | 3.583 | 2:4, 7:139 | 8.925 |

### B-B progression: H0 → H1a → H1c → H2 → H3 → B0

| Target | Bonsai blocks | Drafter | Serial tok/s | Spec tok/s | Δ vs H0 | Retention | Acceptance | Forwards | Generated/forward | Widths | Peak GiB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| H0 | none | B-B | 14.987 | 43.003 | −0.915 | 97.92% | 0.618 | 100 | 5.160 | 2:4, 7:95 | 15.917 |
| H1a | 63 | B-B | 15.132 | 39.532 | −4.386 | 90.01% | 0.562 | 107 | 4.794 | 2:4, 7:102 | 15.821 |
| H1c | 62–63 | B-B | 15.184 | 39.276 | −4.642 | 89.43% | 0.553 | 109 | 4.734 | 2:4, 7:104 | 15.722 |
| H2 | 60–63 | B-B | 15.410 | 39.482 | −4.435 | 89.90% | 0.554 | 108 | 4.741 | 2:4, 7:103 | 15.522 |
| H3 | 56–63 | B-B | 15.772 | 37.908 | −6.010 | 86.31% | 0.525 | 115 | 4.461 | 2:8, 7:106 | 15.128 |
| B0 | 0–63, native full Bonsai2 | B-B | 24.952 | 35.161 | −8.757 | 80.06% | 0.389 | 142 | 3.648 | 2:4, 7:137 | 8.924 |

**H1b interaction diagnostic:** H1b+B-Q reached 43.897 tok/s (99.95% retention); H1b+B-B reached 44.859 tok/s (102.14%). The policy-triggered H1b+B-B repeat measured 45.080 tok/s. This is one initial observation plus one repeat, not a broad significance claim.

Across the main progression, adding Bonsai blocks did not produce a monotonic throughput gain. H1a and H1c were slower than fresh H0 for both drafters; H2 and H3 recovered some rate for B-Q, while B0 was slower than the Qwen control for both drafter pairings. The H1b diagnostic is the exception near or above the denominator and is retained as an interaction result.

## Follow-up

A new current-runtime p02 sweep would add prompt-sensitivity information: it could show whether today’s runtime still has a p02-specific high-throughput regime under the same feature-002 prompt protocol. It is not needed to answer the primary Golden-A product comparison and must be a separate follow-up; no p02 benchmark was run here.
