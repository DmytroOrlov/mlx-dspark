<!--
Sync Impact Report
- Version change: 0.0.0 → 1.0.0 (initial constitution adopted; twelve principles establish governance)
- Modified principles: none (initial adoption)
- Added sections: Core Principles (12 principles), Evidence and Parity Constraints, Development Workflow
- Removed sections: none
-->
# mlx-dspark Constitution

## Core Principles

### I. Parity Before Optimization
For inference ports and integrations, establish semantic parity in this order where applicable:
serial target → hidden-state/tap → fixed-width speculation → proposal/acceptance →
cache/rollback/reconcile → cost model → adaptive scheduler → server integration → performance
optimization. Do not optimize a later stage while an earlier applicable stage remains unverified.

### II. Evidence Before Functional Changes
Every runtime behavior change MUST be justified by the earliest demonstrated semantic divergence or
by a deterministic failing test. Source-code differences alone are not evidence for a behavior
change.

### III. One Hypothesis Per Patch
Each correctness patch MUST make the smallest causally attributable change for one hypothesis. It
MUST NOT combine unrelated refactors, threshold tuning, model-math changes, quantization changes, or
kernel changes with a correctness fix.

### IV. Deterministic Correctness Before Benchmarks
Correctness claims MUST rely on deterministic evidence such as exact token IDs, logits, top-k
identity, top-2 margins, cache offsets, recurrent state, convolution state, accepted-prefix replay,
and rollback equivalence to serial execution. Performance benchmarks are not correctness tests.

### V. Hybrid Cache State Is Model Semantics
For Qwen3.5/Qwen3.8 hybrid targets, speculative rollback MUST correctly restore or replay every
stateful component: attention KV, recurrent/GDN state, convolution state, positions and offsets,
accepted-prefix state, and hidden-state taps. Implementations MUST NOT assume every cache supports
simple truncation.

### VI. Preserve Existing mlx-dspark Behavior
Changes MUST NOT silently regress existing targets or modes. Fixes MUST be scoped by actual model
capability or semantic requirement, and MUST avoid broad model-name special cases when a capability
based scope is available.

### VII. Controlled Comparisons Only
Parity comparisons MUST control target weights, drafter weights, tokenizer, prompt, sampling, draft
width, prefix-cache state, warmup assumptions, and relevant kernels. Comparisons SHOULD use the exact
same physical checkpoint files across runtimes.

### VIII. Earliest Divergence Wins
Once an earlier observable divergence is found, downstream differences MUST be treated as
consequences until evidence proves otherwise. Downstream symptoms MUST NOT be optimized while an
earlier semantic divergence remains unexplained.

### IX. Mandatory Stop Conditions
Every investigation MUST define conditions under which the suspected subsystem will not be patched.
When a hypothesis is disproved, investigators MUST report that result and continue to the next
earliest divergence.

### X. Hardware Claims Require Hardware Evidence
Claims about Apple Silicon throughput, latency, memory, or kernel crossover MUST be supported by
local measurements on the target machine. Static analysis may define a benchmark, but MUST NOT be
used to invent performance conclusions.

### XI. Chad Is a Read-Only Behavioral Oracle
For current Bonsai2/Qwen3.8 DFlash parity work, `~/git/chad` MAY be inspected and instrumented for
diagnosis. Product fixes MUST be made in mlx-dspark; Chad MUST NOT be changed to redefine parity.

### XII. Definition of Done
A runtime correctness fix is complete only when the earliest divergence is explained, a deterministic
regression test exists, the minimal fix is applied, relevant existing tests pass, and controlled
parity is re-checked. When performance is part of the goal, local hardware benchmarks MUST be
re-run. Temporary diagnostics and disproved experiments MUST be removed.

## Evidence and Parity Constraints

Investigation reports and change reviews MUST identify the earliest observable divergence, the
evidence used, the stop conditions, and any disproved hypotheses. Parity and performance results
MUST identify enough comparison controls to reproduce the result. A correctness fix MUST NOT be
presented as complete while any Definition of Done item applicable to its goal remains outstanding.

## Development Workflow

Changes MUST follow the applicable parity sequence and keep one causal hypothesis per correctness
patch. Reviewers MUST check the evidence, scope, deterministic regression coverage, controlled
parity results, and applicable hardware measurements against these principles. Benchmark results
MUST be reported separately from correctness evidence.

## Governance

This constitution governs inference-port, integration, runtime-correctness, and performance work in
mlx-dspark. Amendments MUST be reviewed as governance changes and MUST include an updated Sync Impact
Report. The version MUST use semantic versioning: MAJOR for incompatible governance or principle
removals/redefinitions, MINOR for new principles or materially expanded guidance, and PATCH for
clarifications or non-semantic wording changes. Reviewers MUST assess proposed work against the
principles and record applicable evidence and any unmet requirements. A conflict with a principle
MUST be resolved explicitly in the change proposal before implementation proceeds.

**Version**: 1.0.0 | **Ratified**: 2026-09-23 | **Last Amended**: 2026-09-23
