# Gate C numeric-oracle contract correction — 2026-09-23

## Why the previous Gate C numeric rule was invalid

The saved physical evidence shows that production rollback/reconciliation does
not recompute the accepted prefix through a fresh S=1 or S=2 projection path.
It preserves the semantics of the actual S=8 verify by replaying the accepted
prefix from captured S=8 GDN inputs, rebuilding the convolution window from
the captured S=8 convolution input, and trimming the actual S=8 attention KV.

Fresh S=1/S=2 execution changes projection execution width. Therefore a raw
post-reconcile-versus-fresh-S1/S2 cache delta is a cross-width diagnostic, not
by itself a rollback defect.

The correction does not use matching committed output as a substitute for
state validation. It replaces the invalid cross-width numeric oracle with an
independent same-runtime frozen-S8 accepted-prefix reconstruction.

## Independent same-S8 rollback oracle

For each selected branch the oracle:

- starts only from the frozen same-runtime S=8 B2 capture;
- independently replays captured `q/k/v/a/b[:keep]` from pre-round recurrent state;
- independently derives the convolution window from captured S=8 `conv_input`;
- independently trims S=8 attention KV to the accepted prefix;
- checks accepted-prefix IDs, cache structure, live length, offsets and positions;
- never calls production rollback/reconcile.

Physical results:

- mlx-dspark aligned diagnostic: accepted=0/keep=1 PASS, accepted=1/keep=2 PASS;
- mlx-dspark production semantics: both naturally observed accepted=1/keep=2 branches PASS;
- Chad normal fastpath: accepted=0/keep=1 PASS, accepted=1/keep=2 PASS.

Every reported branch has exact accepted-prefix identity, exact frozen-S8 fused
identity, exact 48-layer GDN reconstruction and exact 16-layer attention-KV
reconstruction. The offline GDN replay is also exactly repeatable.

## Role of fresh S=1/S=2 after correction

Fresh ordinary S=1 and fresh S=2 references remain required as serial
semantic/behavior and cross-width diagnostics. They are useful for Gate D and
for measuring width sensitivity. They are not the numeric cache-state oracle
for state whose projections originated in an S=8 verify.

The prior fresh-S2-derived Gate-C failure is therefore superseded as a contract
false positive and does not authorize a runtime patch.

## Formal status after correction

Local rollback mechanics are physically covered for keep=1 and keep=2 in both
implementations and show no mlx-dspark-only defect.

Formal T010/B2 is still incomplete: the saved Chad branches lack their required
five-run same-width B2 controls, and production-semantics mlx-dspark did not
naturally produce accepted=0 in the measured control run. The aligned mlx
accepted=0 branch is valid local rollback-mechanics evidence but does not
qualify the aligned prefill lifecycle as production semantics.

Therefore overall Gate C remains formally INCONCLUSIVE only behind that B2
prerequisite gap. D/E are not formally reached, T016 remains OPEN, and no
production patch is authorized.

Canonical oracle: `evidence/gate-c-same-s8-rollback-oracle-20260923.json`
Machine-readable correction record: `evidence/gate-c-contract-correction-20260923.json`
