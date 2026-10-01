# T027 process-local probe contract (prepared, not executed)

`t027_product_ab.py` runs the production `mlx_dspark.server.Engine.load()` and
`Engine.generate()` path for the two locked plain-KV candidates. There is no
descriptive production-semantics switch. Candidate A uses the actual production
Qwen target and raw HF DFlash2 checkpoint through the production q4g64 loader.
Candidate B uses the validated local Bonsai target and existing Chad-built q4g64
generic sidecar. Both use adaptive DFlash and the same one-user-message request
IDs, sampling, token limit, server defaults, and direct Engine request path. The
loaded production tokenizer re-renders the selected prompt after Engine startup;
the runner aborts before timing if those IDs differ from the model-free
preflight IDs.

`Engine.load` is called with mode `dflash`, `drafter_bits=4`, derived cap,
`kv_bits=None`, default context, warmup enabled, memory guard enabled, and the
production two-slot prefix cache. Its normal small-M and SDPA split probes,
wide-GEMM and CPU co-prefill application, static/depth-aware adaptive cap path,
and 12-token warmup remain active. The runner records resolved cap, active
small-M/SDPA/CPU-split states, prefix settings, context, warmup, and memory guard.
It bypasses HTTP/socket and BatchEngine scheduling because this is one isolated
request; it does not claim to include those service-layer costs.

The primary physical stage is only Qwen/LRU/fresh and Bonsai/LRU/fresh. Stop
after those two runs and inspect the causal decomposition. The resolver prompt
is optional confirmation only if that pair does not explain the gap robustly.
It is a coding-shaped decode workload, not an agent task-quality result: the
runner neither edits the repository nor invokes tools nor executes pytest. The
existing `tests/test_resolve.py` test is provenance for the task's realism.

The result schema includes request/decode wall time, speculative and serial
throughput, speedup over warm-target serial, token counts, target-forward ratios,
accepted/proposed ratio, committed/forward ratio, round count, per-round proposed,
accepted, committed and source data, width/source distributions, prompt/prefix
state, actual target KV cache classes, peak-memory scope, physical identities,
configuration, and exact request settings. The JSON contains no composite score
or verdict. `PrefixCache.acquire()` is wrapped process-locally; the exact cache
returned to each request is inspected for layer class, attention status, bits,
and group size. A quantized attention cache, non-plain `target.kv_bits`, or no
observable attention KV cache fails the measured request closed.

MLX `metal.reset_peak_memory()` is available in the installed API. The runner
resets immediately before the speculative request and captures peak immediately
after it, then resets independently before the serial control. Each peak includes
the already-resident models and Engine state and has its own scope. The serial
control is warm-target `mlx_lm.stream_generate` after server warmup and the
speculative request. mlx-lm 0.31.3 exposes
`stream_generate(model, tokenizer, prompt, max_tokens=256, draft_model=None,
**kwargs)`; kwargs reach `generate_step`, whose valid sampling parameter is
`sampler`, not `temp`. The runner uses
`make_sampler(temp=0.0, top_p=1.0, top_k=0)`.

Primary timing uses request-boundary wall time and adds no per-round
synchronization. Verify/proposal device time and residual remain unavailable:
the relevant MLX work is lazy, so host-call duration is not device time and
forcing evaluation per round would perturb the primary run. Component microtiming
is not required if the counters explain the product gap.

Fresh-prefix isolates speculative/runtime behavior. To preserve the production
prefix-cache question without adding initial work, `--prefix reuse` is prepared
as a conditional control: issue the exact same full prompt once to seed the real
`Engine.prefix.acquire/checkpoint/store` mechanism, then repeat it and require an
observed cache hit. It does not shorten the prompt or delete tokens; it represents
an exact retry hit, not a complete multi-turn coding-agent history. Run only if
the fresh pair fails to reproduce or explain the observed Qwen approximately
45 tok/s regime.

The drafter relation is classified as **same source checkpoint but materially
different runtime representation**. The production checkpoint is raw BF16 and
the production loader dynamically quantizes Linear modules to q4g64 (excluding
`candidate_selector` under its default predicate); the Chad sidecar carries its
own packed quantized inventory/metadata and is reconstructed without requantizing.
This is a credible attribution confound, so the primary comparison preserves each
product's actual path. If the A/B leaves this as a plausible unresolved cause,
one optional later control is Bonsai target plus the production raw HF drafter
through the production q4g64 loader. Do not run it preemptively.

KV8 and target-KV quantization are absent from the runner's primary path. No
physical run is part of this preparation. T027 remains open until the first two
physical LRU/fresh plain-KV A/B records exist and are causally reviewed.
