# Remaining-gates probe implementation report — 2026-09-23

## Implementation

Added opt-in `--measure-required-gates`. It requires `--measure-controls`,
production semantics (no `--align-prefill`), the normal Chad fastpath, and
plain target KV. These checks run before output setup and before a model load.
Without the new option, the normal continuous generation and existing
reference/control behavior remain unchanged.

After continuous generation is persisted, the runtime function selects the
first observed valid S=8/seven-draft root. The frozen cache is retained; the
ordinary B2/reference path receives a clone in required-gates mode so the root
remains available for the forced branches.

### mlx-dspark lifecycle

`run_mlx()` uses one independent S=8 reference and one five-run same-width
control matrix for the selected frozen root. Each control component retains
the existing three calibration plus two validation repetitions and its output,
tap, GDN capture, cache, dispatch, root, and verify-ID evidence. The two
isolated branches each clone that root, re-run the actual `Target.verify`, then
call the original `Target.rollback` with the source call convention from
`generate.py`: `(cache, 7 - accepted_count, verify_ids[1:1+accepted_count])`.

### Chad lifecycle

`run_chad()` likewise measures B2 once for the selected root. Each branch
clones it and executes an S=8 target forward while the original model call,
tap sink, and GDN collector are live. Chad has no callable target rollback
method: its hybrid target rollback is inline in
`Engine._generate_spec()`. The probe mirrors that source block's GDN
reconstruction and rejected-tail KV trimming, then calls the original
`_DFlashDrafter.reconcile(self, 7, accepted, anchor, draft, hid, fused)` with
the captured forward values. This applies keep=1 and keep=2 state transitions
without synthesizing post-state cache values.

Both runtimes record forced cases in
`doc["required_gate_acquisition"]["branches"]`, outside natural
`doc["rounds"]`, with `branch_kind=isolated_forced` and
`acceptance_source=forced_state_transition_test_input`.

## B2 linkage and C/D/E

The same five-run B2 control set is linked to both branches only after checking
the exact frozen cache state ID, frozen-root observation ID, logical position,
verify-ID SHA-256, and B2 control-set ID. The provenance helper also checks the
S=8 ID list against its recorded digest, keep against accepted_count, and same
runtime. It rejects changes in any of those root identities. No human-readable
root label authorizes sharing.

For C, the production branch uses runtime rollback/reconcile behavior. A
separate clone is independently verified at S=8, then the probe installs GDN
recurrent state from captured `q/k/v/a/b[:keep]` plus pre-state, derives
convolution history from captured `conv_input`, and trims the independently
executed attention KV to the accepted length. It never invokes rollback or
reconcile on the oracle branch. Cache content IDs, structure, accepted fused
rows, prefix IDs, and in-process object disjointness are recorded.

For D, the next token is greedy argmax from S=8 verify row `keep-1`, matching
each runtime's width-8 greedy path. The same token is run through ordinary
width-1 target execution from clones of the production and oracle post-states.
Each branch collects five same-root width-1 paired controls, calibrates on
repetitions 1–3, validates on 4–5, and records output/tap/cache state, top-2
IDs, values, margin, positions, and lengths. Fresh S=1 remains a serial
diagnostic only.

For E, each branch starts with fresh drafter caches. The production path uses
the runtime's proposal and accepted-context lifecycle. The independent path
projects and appends the initial prompt fused rows and accepted S=8 fused rows
through primitive projection/append APIs, then proposes at the same logical
point. It records context order, projected rows, logical/absolute positions,
cache structure, anchor/block, proposal IDs, and five independent same-runtime
projection/proposal controls. Cache object disjointness is checked in-process;
content and root IDs are retained for offline provenance.

There is no circular reuse of production rollback state for C/D: the oracle
cache is independently cloned, re-verified, and reconstructed from captured
S=8 inputs. There is no circular reuse of the continuous drafter cache for E:
both paths allocate fresh drafter caches, and the reference uses only
projection/append primitives.

## Files changed in this pass

- `specs/001-bonsai-rollback-parity/evidence/probes/bonsai_rollback_probe.py`
- `specs/001-bonsai-rollback-parity/evidence/probes/bonsai_probe_support.py`
- `specs/001-bonsai-rollback-parity/plan.md`
- `specs/001-bonsai-rollback-parity/evidence/gates.md`
- `specs/001-bonsai-rollback-parity/evidence/required-gate-provenance-schema-example-20260923.json`
- `specs/001-bonsai-rollback-parity/evidence/remaining-gates-implementation-report.md`

No B2/C/D/E/T016 completion status was changed. No production source or Chad
file was edited. No model, inference, benchmark, or ZIP was run or created.
The existing staged/index state was preserved.

## One future acquisition per runtime

Chad, normal installed fastpath, explicit plain KV:

```sh
python3 /Users/do/git/mlx-dspark/specs/001-bonsai-rollback-parity/evidence/probes/bonsai_rollback_probe.py --runtime chad --chad-kv-bits 0 --measure-controls --measure-required-gates
```

mlx-dspark, production semantics, plain KV, without aligned prefill:

```sh
python3 /Users/do/git/mlx-dspark/specs/001-bonsai-rollback-parity/evidence/probes/bonsai_rollback_probe.py --runtime mlx-dspark --mlx-kv-bits 0 --measure-controls --measure-required-gates
```

The intended budget is now one Chad load plus one production-semantics
mlx-dspark load. Those two loads are sufficient if each continuous run
produces the expected valid S=8/seven-draft source root and the configured
five-run controls complete. If the continuous trace has no valid S=8 root, the
probe fails explicitly before isolated acquisition rather than borrowing
controls from another logical root.

## Offline validation

- `py_compile` passed for all three probe files: the main probe, support module,
  and existing same-S8 oracle.
- CLI validation passed: required-gates without controls exits with argument
  error before output setup; valid mlx-dspark and Chad option combinations
  validate; Chad no-fastpath is rejected.
- Pure schema checks passed for natural/forced separation and acquisition
  construction. Shared linkage passed for one identical root and rejected
  different state ID, observation ID, position, verify-ID digest, or control
  set ID.
- All five existing saved `run.json` files parsed with the old schema. The
  saved Gate C correction JSON also remains readable.
- `git --no-pager diff --check` passed for working-tree changes. Existing
  staged review-bundle content has unrelated cached-diff whitespace findings;
  the index was not changed.

No future acquisition command was executed.

## 2026-09-24 control and E-root hardening

This pass tightens D/E control coverage without changing the acquisition
architecture. The two future acquisitions remain unexecuted.

### D component controls

Each keep branch now runs exactly five fresh LEFT/RIGHT width-1 pairs, each
starting from a clone of that branch's independent same-S8 oracle post-C root.
The declared numeric component set includes logits, fused output, attention
live keys/values, GDN convolution windows, and GDN recurrent state. Cache
structure, layer order, family, offsets, live lengths, and absolute positions
are exact comparisons. Top-2 identities must reproduce exactly; top-2 margin
repeatability is measured separately. An unstable top-2 identity or margin
makes D inconclusive. Component bounds use repetitions 1–3 for calibration
and 4–5 for validation. Control tensors use `EvidenceStore` transient arrays;
only hashes, component metadata, distances, bounds, and discrete results remain
durable.

### E numeric controls and source guard

Each keep branch has its own E root ID, incorporating prompt context, accepted
S=8 fused rows, logical position, keep, and next anchor. Five fresh
LEFT/RIGHT independent-reference pairs project/append those rows into new
drafter caches and capture proposal IDs and live drafter cache keys/values.
Projected prompt and accepted rows and live cache keys/values receive
component-specific calibration/validation bounds. Cache family, layer order,
live rows, offsets, and absolute positions remain exact. Bounds reject any
root ID mismatch, so keep=1 and keep=2 cannot share E controls.

The reconstructed production path records its first proposal and compares it
exactly with source `verify_ids[1:]`. A mismatch sets `next_proposal_comparable`
false and E numeric status to inconclusive. The required source must be round
1, `generated_before == 0`, have durable prompt context rows, and have an
empty captured drafter cache before proposal. A later S=8 row is rejected with
a specific error.

D records a pointer to the source row's fresh S1 keep=1 or keep=2 reference for
serial-semantic diagnostics only; that reference is not used as D's state
oracle.

### Validation update

- `py_compile` passed for the modified probe/support modules.
- CLI validation rejected `--measure-required-gates` without
  `--measure-controls` before model loading.
- Synthetic D/E component-control checks passed for outputs, KV, convolution,
  recurrent components, five-pair calibration/validation, top-2/margin
  repeatability, branch-root isolation, first-proposal mismatch, and the
  first-round/fresh-cache guard.
- Shared B2 linkage passed for identical keep=1/keep=2 roots and rejected
  altered state ID, observation ID, logical position, verify IDs, or control
  set ID.
- Five saved `run.json` files and the Gate C correction JSON parsed.
- The Chad project interpreter's probe `--help` completed without loading a
  checkpoint. Importing `mlx_lm` directly in this headless session failed
  while initializing MLX because no Metal device is available; `uv run`
  `--help` worked with an offline temporary cache. This is an execution-host
  limitation, not a probe dependency or checkpoint-load result.
- `git --no-pager diff --check` passed. No model, benchmark, or acquisition
  ran.

### Future commands and storage

mlx-dspark:

```sh
cd /Users/do/git/mlx-dspark
"$PWD/.venv/bin/python" \
  specs/001-bonsai-rollback-parity/evidence/probes/bonsai_rollback_probe.py \
  --runtime mlx-dspark --mlx-kv-bits 0 --measure-controls \
  --measure-required-gates --min-free-gib 9
```

Chad:

```sh
cd /Users/do/git/chad
uv run python \
  /Users/do/git/mlx-dspark/specs/001-bonsai-rollback-parity/evidence/probes/bonsai_rollback_probe.py \
  --runtime chad --chad-kv-bits 0 --measure-controls \
  --measure-required-gates --min-free-gib 9
```

The five-pair D/E control tensor arrays are transient and add no permanent
tensor blobs. Their result rows and hashes are expected to add under roughly
1 MB per runtime. The required production/oracle samples remain durable and
content-addressed as before; their exact storage depends on cache shapes and
deduplication in the physical run.

No known probe-contract gap remains before acquisition. The acquisition host
must provide a usable Metal device; this validation session did not.
