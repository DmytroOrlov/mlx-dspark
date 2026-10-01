# Correctness gates — matched-lifecycle diagnostics and production-semantics evidence

Recorded offline from saved physical captures on 2026-09-23. This document
contains both **matched-lifecycle diagnostic evidence** and later
**production-semantics mlx evidence**. The initial pair is a
matched-lifecycle diagnostic: its mlx-dspark capture used
`MLX_DSPARK_DFLASH_CHAD_PREFILL=1` (`--align-prefill`). That mode was disproved
as a production/performance fix. Neither this pair nor any cross-runtime raw
numeric difference authorizes a production change. No model or five-run control
was executed for this contract correction.

## Evidence roots and continuous trace

- Chad, normal installed fastpath, explicit plain target KV:
  [T008 run](runs/t008-r1-chad-kv0-fastpath-20260923T190924Z-3aeb5303/run.json),
  [manifest](manifests/t008-r1-chad-kv0-fastpath-20260923T190924Z-3aeb5303.json).
- mlx-dspark, aligned lifecycle diagnostic, explicit plain target KV:
  [T005 run](runs/t005-r1-mlx-kv0-aligned-20260923-c791/run.json).
- Both recorded attention target caches as plain `KVCache`, with `bits=None`
  and `group_size=None`. Chad recorded its active installed Prism fastpath and
  16 attention, 48 GDN, and 64 MLP projection replacements. mlx-dspark used its
  own Prism path. These are distinct valid target projection/op graphs.
- Both naturally ran R1 width 7 / S=8 / accepted 0, R2 width 7 / S=8 /
  accepted 1, then pre-R3 margin **0.125** and R3 width 1. The historical
  Chad 0.25 versus mlx-dspark 0.125 pre-R3 scheduler symptom did **not**
  reproduce with matched plain target KV. Both captures have identical first
  16 committed output IDs:
  `[2, 23288, 27325, 10417, 436, 34810, 18887, 303, 12654, 271, 36948, 369, 264, 4434, 11, 5492]`.

## A — proposal inputs before target verify

**Result: PASS for this matched-lifecycle diagnostic pair.** The corrected
`compare_gate_a()` checks target/sidecar/tokenizer/config/source fingerprints,
prompt and committed-prefix IDs, sampling, tap IDs/order, R1–R3 logical and
absolute positions, anchor/pending IDs, drafter input block IDs, context row
counts/positions, and target/drafter cache family, precision, semantic layout,
shapes, and live ranges. Those exact cross-runtime controls match. Each
runtime's `verify_ids` is exactly its own `[anchor] + drafts`; equality of the
two `verify_ids` lists is not required when the valid proposals differ.

Independent fresh R1 prefill/context/proposal reconstruction passed **within
each runtime**: all ten local input/position/structure checks passed, production
and independent pending-context SHA-256 matched, and all seven independent
proposal IDs matched the corresponding production proposal. The Chad context
SHA is `f15bae42a84f67ccaa771162c6fa91085a26088f76561891e5a78d582744341e`;
the mlx-dspark context SHA is
`3e75cee27df75e7cda7ed71224884d4a567b8637c835ba68f972b4fbe4280c0f`.
The source runs retain the observation IDs. Exact
context equality made R1 five-run numeric controls unnecessary for this local
comparison.

The first cross-runtime proposal difference is retained exactly at **R1 draft
IDs 6–7** (one-based):

| Runtime | R1 seven draft IDs |
|---|---|
| Chad | `[71093, 12305, 198, 21924, 54572, 198, 1445]` |
| mlx-dspark | `[71093, 12305, 198, 21924, 54572, 5492, 6971]` |

Classification:
`equivalent_exact_gate_a_fields_with_implementation_dependent_proposals`.
The difference is a **cross-runtime implementation-dependent diagnostic**
downstream of the distinct valid numerical target graphs, with both independent
same-runtime references passing. It is not an mlx-dspark defect and does not
route to a production drafter/context patch. Identical committed outputs alone
did not establish this Gate A result; the exact controls and local references
did. This diagnostic result does not establish production-semantics correctness.

## B1 — exact local S=8 roots

**PASS for naturally occurring R1 and R2 in both runtimes.** Each speculative
S=8 pre-verify state matched its frozen independent S=8 root in state ID,
semantic structure, positions, and exact eight verify IDs. Mutable cache
object identities were disjoint. Each frozen reference executed after the
continuous trace and was never reconciled. The separate fresh committed S1/S2
reference family is not the B1/B2 root.

## B2 — same-runtime S=8 verify/capture

The saved aligned Chad/mlx pair remains sample-exact for accepted=0 and
accepted=1, but those runs did not contain the required five-run same-width B2
controls.

A later **production-semantics mlx-dspark** run with `--measure-controls`,
plain target KV, and no aligned-prefill diagnostic naturally produced
accepted counts 1, 1, and 5. Its observed S=8 branches have independent frozen
pre-verify roots and stable five-run same-width controls; the speculative and
reference S=8 captures are exact for the observed branches. This establishes
controlled mlx-dspark B2 for the naturally observed accepted=1 branches.

Formal T010/B2 as a whole remains **INCONCLUSIVE**, because the required
accepted=0 controlled branch was not naturally observed in that production
run, while the saved Chad accepted=0/1 branches lack their required five-run
same-width B2 controls. Existing single-sample exact captures are retained and
must not be promoted into missing controls.

## C — post-reconcile rollback mechanics

The previous fresh-S1/S2 numeric adjudication is superseded as a **contract
false positive**. Production reconciliation in both implementations preserves
the semantics of the actual S=8 verify: GDN recurrence is replayed from
captured S=8 `q/k/v/a/b`, convolution history is rebuilt from captured S=8
`conv_input`, and attention KV is trimmed from the S=8 verify state. Fresh
S=1/S=2 executions use different projection widths and therefore cannot serve
as the raw numeric cache-state oracle for this S=8-originating state.

The corrected independent same-S8 accepted-prefix oracle never invokes
production rollback/reconcile. It reconstructs the accepted state only from
the frozen B2 S=8 capture.

Physical local results are exact:

- mlx-dspark aligned diagnostic accepted=0/keep=1: PASS;
- mlx-dspark aligned diagnostic accepted=1/keep=2: PASS;
- mlx-dspark production-semantics accepted=1/keep=2, two observed branches: PASS;
- Chad normal-fastpath accepted=0/keep=1: PASS;
- Chad normal-fastpath accepted=1/keep=2: PASS.

Every selected branch has exact accepted-prefix identity, exact corresponding
frozen-S8 fused identity, exact independent reconstruction across all 48 GDN
layers, and exact attention-KV reconstruction across all 16 attention layers.
The offline GDN replay is exactly repeatable.

The aligned mlx accepted=0 result establishes **local physical rollback
mechanics only**. It does not qualify `--align-prefill` as production semantics
or as a production/performance fix.

Because formal B2 accepted=0/Chad same-width control coverage is still missing,
overall Gate C remains **INCONCLUSIVE behind its B2 prerequisite**, even though
the local rollback-mechanics oracle shows no mlx-dspark defect. No rollback
patch is authorized.

Canonical records:

- `evidence/gate-c-contract-correction-20260923.md`
- `evidence/gate-c-same-s8-rollback-oracle-20260923.json`
- `evidence/gate-c-contract-correction-20260923.json`

## D, E, and T016

- **D subsequent committed target behavior:** not formally reached because
  overall B2/C prerequisites remain incomplete. Existing serial S=1/S=2
  references remain available for offline diagnostic review.
- **E drafter state:** not formally reached after D.
- **T016:** OPEN. Neither `FIX_PENDING` nor `RESOLVED_NO_CHANGE` is authorized.
  The evidence currently demonstrates no Gate-C rollback defect, but missing
  formal B2 coverage prevents closing the investigation.
- **Production patch:** not authorized.

Do not launch another broad model acquisition merely to repeat already exact
rollback-state evidence. Any future physical acquisition must target only the
remaining formal control/coverage gap and should be reviewed as a minimal
experiment first.

<!-- BEGIN T016 CORRECTNESS CLOSURE -->

## Final correctness closure — T016 — 2026-09-24

**Outcome: RESOLVED_NO_CHANGE.**

The final physical required-gate acquisitions pass B2, C, D, and E on both
Chad and production-semantics mlx-dspark for both isolated forced branches:

- accepted_count=0 / keep=1
- accepted_count=1 / keep=2

The production target rollback/reconcile state, the next ordinary target step,
and the next drafter/context proposal all agree with independent same-runtime
references under their predeclared five-run control contracts.

No production correctness patch is authorized.

The authoritative closure record is:

`evidence/t016-correctness-closure-20260924.json`

Earlier `run.json` `gate_results` entries that stopped before C/D/E are
historical. For these gates, the later `required_gate_acquisition` physical
evidence supersedes those stale statuses.

Natural scheduler histories may still differ between Chad and mlx-dspark.
That difference is now treated as a performance/operating-point question,
not evidence of a target rollback defect.

<!-- END T016 CORRECTNESS CLOSURE -->
