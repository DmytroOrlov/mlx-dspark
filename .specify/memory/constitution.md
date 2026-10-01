<!--
Sync Impact Report
- Version change: 1.0.0 → 2.0.0 (major governance change: the project goal changes from
  rollback-parity/correctness investigation to hybrid-target performance experimentation).
- Modified principles:
  - Parity Before Optimization → End-to-End Throughput Is the Decision Metric
  - Evidence Before Functional Changes → Runtime Integrity Is the Correctness Gate
  - One Hypothesis Per Patch → One Experiment Axis at a Time
  - Deterministic Correctness Before Benchmarks → Runtime Integrity Before Performance Claims
  - Preserve Existing mlx-dspark Behavior → Preserve the Vanilla Qwen Path
  - Controlled Comparisons Only → Matched Benchmark Discipline
  - Hardware Claims Require Hardware Evidence → Hardware-Backed, Reproducible Evidence
- Added sections: Experimental Constraints; Measurement and Adjudication; Quality Safeguard
- Removed sections: Evidence and Parity Constraints; Development Workflow
- Removed/demoted old objectives: the ordered rollback-parity investigation sequence, earliest-
  divergence priority, mandatory correctness-investigation stop conditions, Chad as behavioral
  oracle, broad hybrid-cache semantic mandate, and rollback-fix Definition of Done no longer govern
  this program. Cache integrity, controlled comparison, behavior preservation, hardware
  measurement, and causal experiments remain only where they directly support the hybrid experiment.
- Follow-up TODOs: none.
-->
# mlx-dspark Constitution

## Core Principles

### I. End-to-End Throughput Is the Decision Metric
The primary optimization target is speculative `decode_tokens_per_sec`. Serial target speed,
acceptance, accept length, target forwards, generated tokens per target forward, memory, and timing
breakdowns are explanatory measurements; none is a substitute for end-to-end throughput. The
experiment MUST allow a no-win result and MUST NOT presume that more Bonsai2 blocks improve the
outcome.

### II. One Experiment Axis at a Time
The initial hybrid series MUST keep the Qwen embedding, final norm, and LM head; the original
`incoai/Qwen3.8-27B-DFlash2` drafter and its weights and architecture; plain KV; ordinary production
`CapController`; speculative algorithm; sampling and generation settings; prompt corpus; benchmark
Engine path; target hardware; and normal runtime environment fixed. Drafter changes, controller or
speculative-width tuning, KV8, unrelated kernel tuning, rollback redesign, Chad parity work, and
historical-performance reconstruction are out of scope unless evidence motivates a separate,
explicit follow-up experiment. Each comparison MUST isolate one hypothesis or variable.

### III. Preserve the Vanilla Qwen Path
Hybrid support MUST be opt-in and structurally isolated. Loading or running
`mlx-community/Qwen3.8-27B-4bit` with `incoai/Qwen3.8-27B-DFlash2` and no hybrid options MUST
preserve existing behavior. Production-code changes MUST have a demonstrated requirement for
constructing or measuring a hybrid. A small experimental composition/loading layer is preferred to
scattered Bonsai-specific branches in ordinary Qwen execution. A valid outcome may require no
production-code change.

### IV. Establish Representation and Tap Semantics
Compatible transformer shapes do not establish that Qwen and Bonsai2 blocks can be substituted.
Before performance results are accepted, the experiment MUST establish the representation consumed
and returned by the existing Prism/Bonsai Hadamard/packing path and prove each Qwen↔Bonsai boundary
uses the canonical residual stream expected by its neighbor. If transforms persist across block
boundaries, a correct explicit bridge is required or that substitution is invalid. Packed or rotated
Bonsai weights MUST NOT be silently interpreted as ordinary Qwen linear weights. The implementation
MUST reuse the working Prism/Bonsai loader and kernels.

The repository's actual DFlash hook and index semantics MUST also be inspected and recorded,
including whether each target layer ID denotes a block input, block output, or hidden-state sequence
offset. For every variant, record how many hidden states consumed by the original drafter are
modified. Drafter changes to compensate are prohibited in this experiment.

### V. Runtime Integrity Is the Correctness Gate
A hybrid is intentionally not required to match vanilla Qwen logits. Runtime gates MUST establish
successful loading; exact block ownership against the manifest; valid tensor shapes and dtypes;
absence of NaN/Inf corruption; consistent cache advancement and rollback under the existing
speculative runtime; expected deterministic reproducibility; unchanged Qwen-only H0 behavior; and
repeatable composition for the same configuration. Logit or top-1 drift may be measured as a
diagnostic, but is not a hybrid correctness oracle.

## Experimental Constraints

The experiment tests whether a suffix of Bonsai2 ternary blocks can improve target inference speed
while preserving enough compatibility for the unchanged original Qwen DFlash2 drafter to improve
speculative throughput. It MUST NOT assume unchanged acceptance, a positive result, or that a deeper
suffix is better.

The initial family uses Qwen for embedding, final norm, LM head, and every block except explicitly
listed Bonsai2 donor blocks. The required variants are:

| Variant | Bonsai2 donor blocks | Qwen blocks |
| --- | --- | --- |
| H0 | none | 0–63 |
| H1a | 63 | 0–62 |
| H1b | 62 | 0–61, 63 |
| H1c | 62–63 | 0–61 |
| H2 | 60–63 | 0–59 |
| H3 | 56–63 | 0–55 |

Every variant uses the original `incoai/Qwen3.8-27B-DFlash2` drafter. H0 is the fresh Qwen control
for each benchmark series. Hybrid ownership MUST be represented declaratively as explicit donor
block indices, a range, or an equivalent configuration; runtime support MUST permit extending the
family backward without hardcoding each named variant. The initial search MUST stop at H3 until H0,
H1, H2, and H3 results have been adjudicated.

## Measurement and Adjudication

Every required variant MUST record serial target tokens/sec with speculation disabled, speculative
decode tokens/sec, speculative speedup over that variant's serial target, acceptance/accept length
using existing runtime definitions, target-forward count, generated tokens per target forward,
available draft-width/cap distribution, peak steady-state memory, generated token count, and benchmark
duration. Available target-verification, drafter-proposal, rollback/cache, host, and sampling timing
breakdowns SHOULD be retained when obtainable without invasive instrumentation.

Comparisons MUST be fresh and matched for benchmark corpus, generation settings, plain-KV path,
controller, original drafter, runtime revision, machine, and measurement method. Every series MUST
include a fresh H0 measurement; historical throughput (including approximately 45 tok/s) is context
or a regression warning only. Discovery runs may be inexpensive, but any apparent winner, regression,
or difference near measurement noise MUST be confirmed by repeated matched runs before adjudication.
One marginal run MUST NOT be called significant.

For each variant, report deltas against fresh H0 for serial target speed, speculative decode,
acceptance, generated tokens per target forward, target-forward count, and memory. Interpret results
as evidence for how much target-compute speed each amount of ternary substitution buys before reduced
compatibility with the unchanged drafter consumes that gain. Acceptance and forward counts explain
the result; they are not goals in themselves.

The experiment SHOULD include a fresh full-Bonsai2 target with the original incoai DFlash2 drafter,
the cleaned runtime, and ordinary `CapController` as endpoint context. The Bonsai-specific drafter
MUST NOT be used in the primary hybrid comparison.

Results MUST distinguish measured facts, derived metrics, hypotheses, and interpretations. Each
result MUST have a reproducibility manifest identifying the git revision and dirty state, exact
target checkpoint(s) and revisions where available, exact drafter checkpoint and revision where
available, block ownership and embedding/norm/head ownership, controller/KV/speculative settings,
prompt corpus, hardware, command or structured configuration, and raw result artifact.

## Quality Safeguard

The initial stage evaluates runtime integrity and performance, not full model quality. A hybrid that
appears to beat H0 MUST pass basic generation sanity checks before it is called viable. Before a
production recommendation, the winning hybrid MUST be evaluated on the relevant coding-agent/task-
quality workload intended for the product. Throughput, acceptance, and absence of runtime errors do
not establish model-quality equivalence.

After the required H0, H1a, H1b, H1c, H2, and H3 variants are adjudicated, later searches may propose
additional suffix boundaries, localization near a throughput cliff, projection-level substitutions,
or other strategies as separate evidence-driven stages. When a cliff appears, localizing its boundary
takes priority over unrelated optimizations. No expansion is automatic, and no positive result is
required.

## Governance

This constitution governs the hybrid-target performance experiment and related implementation
decisions in mlx-dspark. Amendments MUST be reviewed as governance changes and MUST include an
updated Sync Impact Report. The version uses semantic versioning: MAJOR for incompatible governance
or principle removals/redefinitions, MINOR for new principles or materially expanded guidance, and
PATCH for clarifications or non-semantic wording changes. Work proposals and reviews MUST identify
applicable principles, fixed experiment controls, and evidence required for the decision. Deviations
from the initial experiment contract MUST be specified as a separate experiment axis before they are
introduced. Hypotheses MUST NOT be presented as established causes without a discriminating
experiment. The program may conclude that no hybrid wins and that no production change is warranted.

**Version**: 2.0.0 | **Ratified**: 2026-09-23 | **Last Amended**: 2026-09-25
