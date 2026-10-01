# Remaining correctness gates — implementation brief

Date: 2026-09-23

## Current established evidence

Gate A:
- matched plain-KV diagnostic exact semantic/logical fields PASS;
- cross-runtime R1 draft IDs differ at positions 6–7;
- both runtimes independently reproduce their own R1 context and proposal exactly;
- proposal difference is implementation-dependent diagnostic, not an mlx defect.

Gate B1:
- naturally occurring S=8 roots are exact in both runtimes.

Gate B2:
- saved aligned Chad/mlx accepted=0 and accepted=1 samples are hash-exact;
- production-semantics mlx run with --measure-controls observed accepted 1,1,5;
- its naturally observed S=8 branches have stable five-run same-width controls and exact speculative/reference captures;
- formal coverage gap remains: production mlx accepted=0 controlled branch and Chad five-run same-width controls.

Gate C:
- prior fresh-S1/S2 numeric oracle was invalid because production rollback state originates from S=8 projections;
- corrected independent same-S8 accepted-prefix oracle reconstructs state without calling production rollback;
- exact PASS:
  * mlx aligned accepted=0/keep=1;
  * mlx aligned accepted=1/keep=2;
  * mlx production two accepted=1/keep=2 branches;
  * Chad accepted=0/keep=1;
  * Chad accepted=1/keep=2;
- every selected branch exactly matches all 48 GDN layers and all 16 attention layers;
- no evidence-backed rollback defect remains;
- C is formally blocked only by the B2 prerequisite coverage gap.

No production patch is authorized.

## Cost objective

Do not spend more physical loads than necessary.

Before any model run, extend the evidence probe so that ONE future load per
runtime can collect all remaining required evidence. Do not rely on naturally
obtaining accepted=0 in production mlx.

The desired final acquisition budget is at most:
- one normal-fastpath Chad load;
- one production-semantics mlx-dspark load.

No Sol reasoning model is needed to implement this plumbing.

## Required probe extension

Do not edit production runtime code and do not edit Chad.

Preserve the existing continuous run and all existing evidence fields.

Add isolated required-branch evaluation after continuous generation using
fresh/cloned state. The isolated branch mechanism must cover:

- verify width S=8 / seven drafts;
- keep=1 corresponding to accepted=0;
- keep=2 corresponding to accepted=1.

The isolated acceptance choice is a state-transition test input. It must be
clearly labeled as isolated/forced and must never be reported as natural
scheduler/acceptance behavior.

### B2

B2 is entirely pre-reconciliation.

For one exact frozen S=8 pre-verify root:
- run independent same-width S=8 verify reference;
- run exactly five same-width controls under the existing 3-calibration +
  2-validation rule;
- retain exact verify IDs, output/tap/cache hashes, dispatch and root IDs.

If keep=1 and keep=2 isolated branches share the identical frozen B2 verify
root, link both branches to that same B2 observation/control set instead of
duplicating expensive S=8 controls merely because the later forced acceptance
choice differs.

Do not transfer a bound to another logical root.

### C

For each isolated keep branch:
- production-under-test branch must invoke the actual runtime
  rollback/reconcile implementation on a cloned exact S=8 branch;
- independent oracle must NOT invoke production rollback/reconcile;
- independent oracle must reconstruct accepted state from the frozen S=8
  capture exactly as established by
  bonsai_same_s8_rollback_oracle.py:
  recurrence replay from captured q/k/v/a/b[:keep] + pre-state,
  convolution history from captured conv_input,
  attention KV trimmed to pre-live-length + keep;
- compare exact structural/logical fields and same-S8 state;
- keep fresh S1/S2 only as serial/cross-width diagnostics.

### D

After each isolated reconciled branch, establish the next ordinary width-1
target evaluation.

Use the S=8 verify row at index keep-1 to determine the deterministic next
committed token for the branch.

Construct two independent starting caches:
1. actual production rollback/reconcile result;
2. independent same-S8 accepted-prefix oracle state.

Feed the SAME next committed token through the ordinary one-token target path
from both states.

Capture before/after:
- exact input token and absolute/logical position;
- token/top-k identity;
- top-2 values and margin;
- taps/fused row;
- resulting cache structure/live lengths/offsets/positions;
- recurrent/convolution/attention state.

Use exact equality where the same-path execution is bit-repeatable. If a
numeric same-path observation requires a bound, use exactly five predeclared
same-root width-1 controls before judging the production sample.

Also retain the ordinary fresh S1 committed path as the serial-semantic
diagnostic. Do not use raw fresh-S1 cache equality as an oracle when its
pre-state came from a different projection-width history.

### E

Build an independent same-runtime drafter/context reference for each isolated
branch without reusing the production drafter cache.

For the first logical speculative round this can be reconstructed entirely
from durable inputs already available in the probe:
- initial prompt fused/context rows;
- exact accepted fused S=8 rows;
- their logical and absolute positions;
- next anchor token;
- fresh drafter cache.

Rebuild the drafter cache/context with the runtime's applicable projection and
append semantics, while normalizing the lifecycle difference:
- Chad may stage accepted fused rows as pending until the next proposal;
- mlx-dspark may append/consume them at its own reconcile/proposal boundary.

Compare semantic state at the equivalent logical point:
- accepted context rows and order;
- projected rows where available;
- drafter cache family/shape/live rows/offsets/absolute positions;
- exact next anchor/block;
- exact next proposal IDs.

For a numeric-only E difference, follow the existing spec requirement:
the fresh independent drafter/context reconstruction must be repeatable under
its own exactly five same-runtime controls, and production must exceed that
predeclared stable bound before E can fail.

Cross-runtime numeric differences remain diagnostic.

## Important implementation constraints

- Keep current normal probe behavior backward compatible.
- Add an explicit CLI switch for the new isolated gate acquisition rather than
  silently changing every diagnostic run.
- The switch name should clearly describe intent, e.g.
  --measure-required-gates or similarly precise wording.
- --measure-controls must remain explicit.
- Production mlx acquisition must NOT use --align-prefill.
- Chad must use normal installed fastpath and explicit plain target KV.
- Exact forced/natural provenance must be present in JSON.
- No model is to be loaded while implementing this patch.
- No benchmark is to be run while implementing this patch.
- Do not run /speckit.analyze.
- Do not change Git state.
- Do not modify production src/mlx_dspark code.
- Do not modify ~/git/chad.
- Do not weaken Gate A/B1/B2/C contracts to make existing evidence pass.

## Small documentation cleanup

The Gate C correction is accepted, but clean these stale phrasings while
touching investigation docs:

- plan.md high-level summary still says to compare post-reconcile raw state
  against ordinary S1/S2 in wording that can be read as the old numeric oracle;
  make it explicit that S1/S2 are serial/width diagnostics and same-S8 is C's
  numeric state oracle.
- evidence/gates.md title/preamble still frames the whole document only as the
  matched-lifecycle pair although it now also records production mlx evidence.
  Rename/reword without changing historical provenance.

Do not mark B2/C/D/E/T016 complete during implementation.

## Offline validation before STOP

Without loading a model:
- py_compile all changed probe files;
- run any pure helper/unit checks that do not load checkpoints;
- verify old saved JSON remains readable;
- verify the new schema distinguishes natural versus isolated/forced branches;
- git --no-pager diff --check.

Then STOP.

Report:
- files changed;
- exact new CLI;
- exact isolated branch lifecycle for mlx and Chad;
- how B2 controls are shared/linkable before reconciliation;
- D oracle construction;
- E independent drafter reconstruction;
- why there is no circular reuse of production rollback/drafter state;
- commands for exactly ONE future Chad acquisition and ONE future production
  mlx acquisition, but DO NOT execute them;
- any remaining reason that two loads would not be sufficient.

Produce a compact review bundle with source, relevant diff, schema example,
static-check output, and no tensor blobs.
