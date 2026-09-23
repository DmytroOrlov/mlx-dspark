# Hardware validation preparation (T025)

Recorded offline on 2026-09-24. **Every command below is prepared only. None was
executed.** No model was loaded and no inference or benchmark was run for T025.

## Status and scope

- T016 correctness = **RESOLVED_NO_CHANGE**; T024 preserves the same final
  investigation outcome. See `t016-correctness-closure-20260924.{json,md}` and
  `regression.md`.
- T025 = **complete: preparation and offline dry-run validation only**.
- T026 = **COMPLETE (physical acquisition and offline adjudication)**; four physical arms already ran. See `t026-golden-adjudication-20260924.json`. Do not rerun.
- T027 = **OPEN / not executed**; prepare the primary plain-KV product A/B and causal probe described below.
- T025 does not execute hardware validation and makes no new throughput claim.
  Correctness evidence and performance measurements remain separate.

## Physical comparison identity

| Field | Resolved value |
|---|---|
| Physical target used for T025/T026 | Exact local snapshot below; this is the physical text target used by the completed correctness investigation and historical Chad performance work. No current Prism repository artifact is substituted. |
| Locally available HF snapshot | `/Users/do/.cache/huggingface/hub/models--nathansutton--Qwen3.8-27B-Ternary-Bonsai-2-DFlash2-MLX/snapshots/234cc925d77cd94683e47b1493bd964917aaf43b` |
| Snapshot/commit | `234cc925d77cd94683e47b1493bd964917aaf43b` (`refs/main` resolves to this value) |
| Target config SHA-256 | `20a7ccac3e519b5b5d7c4eaa6451352dd4188da44b4385ffa6cf87a4f2b8a94a` |
| Target weight SHA-256 | `68541bf9c72747df90764338fa966105b34d94e4e8751dd5d3403d732eedddcf` (snapshot symlink points to HF blob with this cryptographic name; no redundant multi-GB hash was needed) |
| Target Hadamard data | `hadamard.json` HF blob identity `65b947b711b0ba2f654ee6839075f80dfa321685` |
| Prism source revision | `3f926b415992eaa2ae9dd7b573706494d6bbf787` (from the local safetensors header metadata) |
| Physical pack metadata | `model_type=prism_hadamard_qwen35`; target is Prism Hadamard Bonsai2, 2-bit ternary pack. `hadamard.json` is present. Runtime profile records Prism loader and target MMA paths. |
| Tokenizer source | Same snapshot, `tokenizer.json` SHA-256 `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523`; `tokenizer_config.json` SHA-256 `95c557768e6b88a7128befc7bfd3c7de50e5d51af9b8b33a9f4dee0e04f99679`; snapshot `chat_template.jinja` present. |
| Target sidecar relationship | The target snapshot also contains `dflash/`; **do not use it** in the primary comparison. |

Target provenance: source model/build is **Prism ML Ternary Bonsai 2 27B**. The
`nathansutton` repository snapshot is a Chad-oriented repack. Its README documents
that the language-model tensors, sign vectors, and `hadamard.json` are byte-identical
to the Prism source revision recorded in the safetensors header. Packaging differs:
this repack is text-only, while the Prism public pack may include a vision component;
tokenizer and chat-template packaging also differs. T026 therefore compares the
exact local text snapshot already used for correctness and historical Chad work,
not a newly downloaded or current Prism repo artifact. No download is needed or
allowed. The requested `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` is source/build
provenance, not asserted to be an alias for this physical snapshot.

## Exact physical generic Chad-built sidecar

Required physical path (expanded):
`/Users/do/.cache/chad/dflash/015e795645c74b1a0eeef3b570031fb62e769bc5-q4g64`.

| Fingerprint/metadata | Value |
|---|---|
| `config.json` SHA-256 | `6fd15051e629eba87121298bce299f92f1ef99c14e01e144825bbf0aaeaca86f` |
| `model.safetensors` SHA-256 | `876c368b5abfdd5de52ab14fcd5d3cccb07f0059f3903c984e4f17dbcee8a552` |
| Encoded identity | `015e795645c74b1a0eeef3b570031fb62e769bc5-q4g64` is the generic Qwen3.8-27B DFlash2 sidecar built by Chad; it is not the target-specific Bonsai sidecar. |
| Quantization | 4-bit, group size 64 (`q4g64`); prior actual mlx-dspark load inventory: 47 b4g64 and 2 b8g64 quantized modules. |
| Selector / DFlash metadata | `DFlash2DraftModel`; block size 8; draft ceiling 7; taps/target layer IDs `[5,19,33,47,61]`; selector rank 256, selector top-k 16; mask token ID 248070; five context layers; conv group size 16, kernel size 2. |
| Exact fingerprint to bind T026/T027 | Sidecar directory identity plus both SHA-256 values above. Recheck both immediately before the future physical run. |

These values are from the exact sidecar `config.json`, file metadata and existing
`controls.md` actual-loader observation. The runner binds this path directly via
`CHAD_DFLASH_PATH` for Chad and the `load_dflash(path)` argument for mlx-dspark.

## Golden request, rendered prompt, and equality contract

Literal user text:

> Write a production-quality Python LRU cache with tests and type hints.

Request shape for both runtime APIs: one user message, exactly
`{"role":"user","content":"Write a production-quality Python LRU cache with tests and type hints."}`.
No system prompt, tools, prior turns, or prefix are included. Request settings:
thinking disabled, temperature `0`, top-p `1`, top-k `0`, and max tokens `512`.

Rendered by the target snapshot tokenizer's `apply_chat_template` with
`add_generation_prompt=True, enable_thinking=False` (the same mechanism used by
mlx-dspark `encode_prompt` and Chad's `benchmarks/spec_lru.py::render`):

```text
<|im_start|>user
Write a production-quality Python LRU cache with tests and type hints.<|im_end|>
<|im_start|>assistant
<think>

</think>

```

Tokenizer-only IDs from `controls.md` (26 tokens):

```text
[248045, 846, 198, 7734, 264, 5492, 21408, 12654, 436, 34810, 6297, 440, 6813, 321, 913, 29642, 13, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271]
```

The IDs are reproducible from the tokenizer identity/fingerprints above using
tokenizer-only code. The prompt ends in the template's assistant generation
suffix; it contains neither a generated EOS nor a separately added BOS. The
visible `<|actor_true|>...<|channel>final` suffix corresponds to the special token
IDs shown in the ID list. The run must regenerate these IDs locally and abort or
report a control mismatch if they differ.

The public CLIs do not establish equal inputs: mlx-dspark's generation CLI
accepts prompt text and internally uses `encode_prompt`; Chad's public CLI creates
an agent turn with its system/tool transcript and does not expose `max_tokens=512`
as a one-request option. For the primary exact comparison, use the two inspected
in-process runtime APIs with the exact same `prompt_ids` list above:

- Chad: `Engine.generate(prompt_ids, max_tokens=512, ...)` accepts already-rendered
  token IDs. The future harness must create one user-only template request with
  `Agent._template_ids(tok.apply_chat_template(..., add_generation_prompt=True,
  enable_thinking=False))`, assert equality with the recorded list, and call
  `Engine.generate` directly. Do not call `Agent.run_turn`.
- mlx-dspark: `dflash_generate(..., prompt_ids=ids, apply_chat_template=False,
  ...)` accepts those same exact IDs and bypasses a second template render.
- The shared target tokenizer and equality assertion establish semantic and
  token-wise matching; the harness records the IDs received by both APIs.

Chad's exact no-thinking mechanism for this low-level golden request is the
template kwarg `enable_thinking=False`, as confirmed in `Agent._template_ids` /
`_render` and the historical benchmark `render`. The engine API itself takes IDs
and therefore has no thinking flag. Do not substitute Chad `--no-think` for this
request: that CLI flag configures an agent and its sampler preset. mlx-dspark has
no generation-time reasoning switch; its chat template renderer defaults to
`enable_thinking=False`, and here the prepared harness explicitly renders it once
and passes IDs to generation.

## Exact-input runner and four prepared commands

The project-local evidence runner is
`evidence/probes/bonsai_golden_perf.py`. It accepts `--runtime {chad,mlx-dspark}`,
`--arm {fixed7,adaptive}`, `--output PATH`, and `--dry-run`. Defaults encode the
exact target, generic sidecar, prompt, 26 IDs, plain KV, greedy sampling, 512-token
budget, and no prefix reuse. Dry-run only reads local config/tokenizer metadata and
fails before any target or drafter weights are loaded. Normal mode is intentionally
not run during T025.

Both runtimes receive the same rendered one-user-message prompt IDs. The runner
re-renders once with `add_generation_prompt=True, enable_thinking=False`, compares
the result with the frozen 26 IDs, and aborts on mismatch. The low-level generation
call receives those IDs directly and does not apply a second chat template.

Chad controls are source-backed: fixed7 sets `dflash_num_draft=7` and adaptive off
(`CHAD_DFLASH_DRAFT=7 CHAD_DFLASH_ADAPTIVE=0`); adaptive sets the same ceiling
and adaptive on (`CHAD_DFLASH_ADAPTIVE=1`, with `CHAD_DFLASH_DRAFT` unset). Both
bind `CHAD_DFLASH_PATH` to the exact generic sidecar, `CHAD_KV_BITS=0`, and
`CHAD_NO_PREFIX_CACHE=1`.

mlx-dspark calls the inspected `load_target`, `load_dflash`, and `dflash_generate`
path. Adaptive uses the ordinary Prism WidthPolicy at ceiling 7 and explicitly
unsets `MLX_DSPARK_DFLASH_FORCE_WIDTH`. Fixed7 sets
`MLX_DSPARK_DFLASH_FORCE_WIDTH=7`: this is a **diagnostic fixed-width execution-cost
control only**, not a recommended production configuration and not correctness
evidence. The runner restores the variable after the measured call. No production
switch is added.

Every physical result is written under project-local
`specs/001-bonsai-rollback-parity/evidence/t026-results/`; `/tmp` is not used for
canonical evidence. The physical runner recomputes the current sidecar weight
SHA-256 before loading the model. It also records the runner SHA-256, tracked dirty
runtime paths and direct hashes of the runtime source files on the exercised path;
repository HEAD alone is not treated as sufficient provenance when a worktree is
dirty.

Each command starts a fresh process, runs one disposable 8-token warmup through the
selected arm to compile/ramp its kernels, then measures a request from fresh empty
target/drafter request caches. Chad calls `Engine.reset()` after warmup; mlx-dspark
creates a new request cache for each `dflash_generate` call. Warmup output/timing is
excluded. Persistent prefix reuse is disabled. Use the project environments and
capture the JSON output path as shown. These are the four final T026/T025 prepared
commands; **do not execute them until the maintainer starts T026**.

A. Chad fixed7:

```bash
cd /Users/do/git/chad
CHAD_MODEL=/Users/do/.cache/huggingface/hub/models--nathansutton--Qwen3.8-27B-Ternary-Bonsai-2-DFlash2-MLX/snapshots/234cc925d77cd94683e47b1493bd964917aaf43b CHAD_DFLASH_PATH=/Users/do/.cache/chad/dflash/015e795645c74b1a0eeef3b570031fb62e769bc5-q4g64 CHAD_KV_BITS=0 CHAD_NO_PREFIX_CACHE=1 CHAD_DFLASH_DRAFT=7 CHAD_DFLASH_ADAPTIVE=0 /Users/do/git/chad/.venv/bin/python /Users/do/git/mlx-dspark/specs/001-bonsai-rollback-parity/evidence/probes/bonsai_golden_perf.py --runtime chad --arm fixed7 --output /Users/do/git/mlx-dspark/specs/001-bonsai-rollback-parity/evidence/t026-results/chad-fixed7.json
```

B. mlx-dspark fixed7 diagnostic execution-cost control:

```bash
cd /Users/do/git/mlx-dspark
MLX_DSPARK_DFLASH_FORCE_WIDTH=7 /Users/do/git/mlx-dspark/.venv/bin/python /Users/do/git/mlx-dspark/specs/001-bonsai-rollback-parity/evidence/probes/bonsai_golden_perf.py --runtime mlx-dspark --arm fixed7 --output /Users/do/git/mlx-dspark/specs/001-bonsai-rollback-parity/evidence/t026-results/mlx-fixed7.json
```

C. Chad adaptive:

```bash
cd /Users/do/git/chad
env -u CHAD_DFLASH_DRAFT CHAD_MODEL=/Users/do/.cache/huggingface/hub/models--nathansutton--Qwen3.8-27B-Ternary-Bonsai-2-DFlash2-MLX/snapshots/234cc925d77cd94683e47b1493bd964917aaf43b CHAD_DFLASH_PATH=/Users/do/.cache/chad/dflash/015e795645c74b1a0eeef3b570031fb62e769bc5-q4g64 CHAD_KV_BITS=0 CHAD_NO_PREFIX_CACHE=1 CHAD_DFLASH_ADAPTIVE=1 /Users/do/git/chad/.venv/bin/python /Users/do/git/mlx-dspark/specs/001-bonsai-rollback-parity/evidence/probes/bonsai_golden_perf.py --runtime chad --arm adaptive --output /Users/do/git/mlx-dspark/specs/001-bonsai-rollback-parity/evidence/t026-results/chad-adaptive.json
```

D. mlx-dspark adaptive (force-width explicitly unset):

```bash
cd /Users/do/git/mlx-dspark
env -u MLX_DSPARK_DFLASH_FORCE_WIDTH /Users/do/git/mlx-dspark/.venv/bin/python /Users/do/git/mlx-dspark/specs/001-bonsai-rollback-parity/evidence/probes/bonsai_golden_perf.py --runtime mlx-dspark --arm adaptive --output /Users/do/git/mlx-dspark/specs/001-bonsai-rollback-parity/evidence/t026-results/mlx-adaptive.json
```

The final product golden pair for T026 is **Chad adaptive + mlx-dspark adaptive**.
The fixed7 pair is its controlled execution-cost companion. Chad's target-specific
bundled drafter is not used in any arm.

## T026 fields to record before any run

Each runtime/arm must write one immutable run record containing:

- runtime name/version, repository HEAD/source revision, dirty paths and runner
  hash;
- target ID, resolved snapshot/commit, config fingerprint, weight/index identity,
  Prism/quantization metadata;
- exact sidecar path, config and weight fingerprints, DFlash/quantization/selector
  metadata;
- tokenizer source/snapshot, tokenizer file fingerprints, literal prompt,
  message object, rendered prompt, exact prompt IDs/count, special-token/BOS/EOS
  behavior, and the equality assertion result;
- exact request settings and thinking-disabled mechanism;
- target KV cache class, bits and group size; width mode (fixed7/adaptive), full
  width trajectory/distribution, per-round margin when captured, accepted count
  and acceptance length, target forwards;
- output token IDs, output-text identity/hash, generated count, decode tok/s,
  prefill tok/s and TTFT when exposed, and round-latency distribution when
  exposed;
- peak memory, effective context window, prefix-cache state, warmup state,
  kernel/environment flags and all relevant `MLX_*`/`CHAD_*` variables;
- hardware model/chip/RAM, OS, Python, MLX and mlx-lm versions, plus command and
  timestamp.

Correctness remains the already-resolved T016/T024 result. Throughput, acceptance,
width, memory and latency are performance metadata and cannot change that result.

## Warmup and prefix-cache contract

Primary T026 comparisons start each runtime/arm in a fresh process with a fresh
empty target/drafter cache, no reusable prompt checkpoint, and no shared prefix
cache. Run each runtime's documented disposable kernel/GPU warmup before the
measured request, then reset to a new empty request cache; warmup tokens are not
included in output IDs or timing. Chad's exact runner must disable its persistent
on-disk cache (`CHAD_NO_PREFIX_CACHE=1`, as in `Engine.generate`) and avoid
`~/.cache/chad/kv` checkpoint reuse. mlx-dspark uses its one-shot generation
path (not a persistent server prefix cache), with no cache object passed in.
Record measured request as `prefix_cache=disabled/empty`, `warmup=completed`
along with the warmup method/duration. The historical benchmark measures warm
decode on real mid-session contexts; this golden case intentionally uses the
specified 26-token fresh prompt and cannot be presented as reproducing that
protocol exactly.

## Historical performance context only

The prior approximate observations supplied by the project are **historical /
background only**, not new T026 results: Chad adaptive about 47.8–47.9 tok/s;
Chad fixed7 about 33.9–34.0 tok/s; mlx-dspark fixed7 about 33.6 tok/s; serial
about 25–26 tok/s. The Chad benchmark source explains that its headline decode
measurements used warm mid-session contexts, ten prompts, 384 generated tokens,
and decode-only timing. Its separate `spec_lru.py` corpus helper measures a
golden seed prompt but modifies in-process arm fields and reports corpus summaries.
These contexts/protocols are not the fresh 512-token T026 golden case. No value in
this paragraph is attributed to T025 or claimed as a new measurement.

## T027 primary product A/B (prepared only; not executed)

Primary question: **Why is established `mlx-community/Qwen3.8-27B-4bit` + its
production generic `incoai/Qwen3.8-27B-DFlash2` + plain target KV + adaptive
DFlash + production serving/runtime faster or slower in real coding-agent use
than Bonsai2 ternary + its appropriate generic DFlash2 sidecar + plain target
KV + adaptive DFlash, and where exactly does the end-to-end time difference
come from?** The maintainer confirms Qwen 4-bit fits with plain KV at configured
262144 context. Bonsai2's ternary target is already an aggressive target-weight
compression tradeoff. No KV precision reduction is an acceptable product
optimization.

| Candidate | Target | Drafter | KV / runtime |
|---|---|---|---|
| A | production Qwen 4-bit local snapshot | production generic DFlash2 snapshot | plain target KV; adaptive DFlash; production semantics |
| B | validated local Bonsai2 ternary snapshot above | appropriate generic DFlash2 sidecar; establish relation to A drafter | plain target KV; adaptive DFlash; matched request/runtime semantics where possible |

Fresh/empty-prefix is the initial product pair. Run Qwen/LRU and Bonsai/LRU only,
then stop and inspect their causal decomposition before doing any further physical
work. A direct fresh-prefix result isolates the speculative/runtime path; it does
not represent steady-state coding-agent traffic when production prefix reuse hits.
Use the exact T026 LRU prompt for these first runs.

The local resolver task below is only a predeclared coding-shaped confirmation
workload if the LRU pair does not explain the gap robustly. It is not a task-quality
or test-graded agent-success measurement: this runner only generates text and does
not edit files, invoke tools, or run its test command.

> Review the target resolver in `src/mlx_dspark/load.py` for Qwen3.6 27B routing
> collisions with Ternary Bonsai and dense Qwen3. If a collision or missing
> guard exists, make the smallest correction. Preserve quantization-agnostic
> Qwen3.6 resolution and Bonsai's variant-specific mapping. Report the changed
> behavior and run the focused resolver tests.

Provenance criterion: the existing repository tests
`uv run pytest tests/test_resolve.py -q`, including
`test_qwen36_27b_no_cross_match_with_bonsai_or_dense_qwen3`, establish that this
is a real local resolver maintenance task. T027 does not claim that generation
passes these tests. The fixture is this checkout's `src/mlx_dspark/load.py` and
`tests/test_resolve.py`; no model or network access is needed.
The model-free preflight records repository HEAD
`1b8e5357ec9caec2ea8f92395d516333946697da`, resolver source SHA-256
`c7e21d6bb4d93947f02ec0830712065f1e4b75627f490fb14cdd7622350f9573`, and test
file SHA-256 `af87cf44f1d10e5c2743886f1abf9c9b29835e788547a63240184ed843234101`
to bind that local fixture.

Record exact prompts, prompt and generated token counts, temperature/top-p/top-k, and reasoning/thinking settings,
target/drafter physical identities and fingerprints, drafter tensor/packaging
equivalence classification, target format/precision, cache class/precision,
context and prefix-cache state. The runner collects serial target-only tok/s,
prefill where exposed, decode wall time/tok/s, peak memory, per-round
proposed/accepted/committed tokens and source, target-forward count,
accepted/proposed, committed/forward, and width distribution. Verify/proposal
latency, TTFT, accumulated component time and residual are explicitly
unavailable in the primary run; no host-call timings are mislabeled as device
execution time.

Offline drafter finding: the production generic checkpoint and Chad-built q4g64
sidecar have identical configs (production config SHA-256
`873e3556509b0da06e29654ba00d4944888d4b5e8a33afde25f7eb27d321e980`; sidecar
config SHA-256 `6fd15051e629eba87121298bce299f92f1ef99c14e01e144825bbf0aaeaca86f`).
The sidecar safetensors metadata names the exact production snapshot as its
source, with `bits=4`, `group_size=64`; header inventory is 81 BF16 production
tensors versus 179 packed sidecar tensors, including U32 selector weights.
`load_dflash` production defaults quantize raw upstream weights to q4g64 but
exclude `candidate_selector` modules; existing sidecar metadata instead rebuilds
its quantized module inventory and is not quantized again. Classification:
**same source checkpoint but materially different runtime representation**.
This is a credible confound for target-versus-drafter attribution. Optional
same-drafter control, only if the plain A/B leaves that specific ambiguity:
run Bonsai target with the production raw HF drafter passed through the same
production q4g64 loader. Do not run this control preemptively.

The runner uses the actual `mlx_dspark.server.Engine.load()` and
`Engine.generate()` path for both candidates; there is no descriptive
`--production-semantics` switch. `Engine.load(mode="dflash", drafter_bits=4,
max_draft_tokens=None, kv_bits=None, context_window=None, warmup=True,
memory_guard=True)` follows the server's defaults for target and drafter loading,
adaptive cap derivation/depth refinement, small-M and SDPA-split probes,
wide-GEMM and CPU co-prefill application, 12-token kernel warmup, memory guard,
and enabled two-slot prefix cache. It records the resolved cap and active
small-M/SDPA/CPU-split settings. The request is greedy with thinking disabled.
This is production Engine initialization and request handling, but one isolated
request bypasses HTTP/socket and BatchEngine queue/batching overhead. That
difference is recorded in each result; both candidates use the same direct
Engine request path.

For each acquired request, the runner wraps production `Engine.prefix.acquire`
and inspects the exact target cache returned to the measured request. It records
each layer-cache class,
attention-cache bits/group size, and fails closed if any measured attention
cache is quantized, if `target.kv_bits` is non-plain, or if no attention KV cache
was observed. This observes the cache actually supplied by prefix acquire, not
just the requested config.

The installed MLX `metal.pyi` declares `reset_peak_memory() -> None`, along with
`get_active_memory()` and `get_peak_memory()`. The runner records active baseline,
resets immediately before the speculative request, and captures peak immediately
after it. It resets again immediately before the serial control and records a
separate peak. Each peak includes model weights resident at the reset point; it
does not combine the two generation phases.

The runner uses warm-target serial `mlx_lm.stream_generate` with an explicit
`make_sampler(temp=0.0, top_p=1.0, top_k=0)` callable. In installed mlx-lm
0.31.3, `stream_generate(model, tokenizer, prompt, max_tokens=256, draft_model=None,
**kwargs)` forwards `kwargs` to `generate_step`; `generate_step` accepts `sampler`
but not a `temp` keyword. This avoids the old invalid `temp=0.0` call. Serial
throughput is a separate warm-target decode control after server warmup and the
speculative request, not cold-load throughput.

Primary observability is request-boundary/decode wall time, generated tokens,
target forwards, each round's width/source/accepted/committed counts, width and
source distributions, accepted/proposed, committed/forward,
target-forwards/generated-token, actual cache classes and speculative peak
memory. It adds no per-round synchronization. Verify/proposal GPU timings and
residual are unavailable: lazy MLX results make asynchronous host-call times
misleading, and forcing per-call evaluation would perturb execution. No
component microtiming run is required for this product decision.

Preflight commands (model-free; all four candidate/workload combinations):

```sh
.venv/bin/python specs/001-bonsai-rollback-parity/evidence/probes/t027_product_ab.py --candidate qwen --prefix empty --workload lru --kv plain --adaptive --preflight --overwrite --output /tmp/t027-qwen-lru.json
.venv/bin/python specs/001-bonsai-rollback-parity/evidence/probes/t027_product_ab.py --candidate bonsai --prefix empty --workload lru --kv plain --adaptive --preflight --overwrite --output /tmp/t027-bonsai-lru.json
.venv/bin/python specs/001-bonsai-rollback-parity/evidence/probes/t027_product_ab.py --candidate qwen --prefix empty --workload resolver --kv plain --adaptive --preflight --overwrite --output /tmp/t027-qwen-resolver.json
.venv/bin/python specs/001-bonsai-rollback-parity/evidence/probes/t027_product_ab.py --candidate bonsai --prefix empty --workload resolver --kv plain --adaptive --preflight --overwrite --output /tmp/t027-bonsai-resolver.json
```

First physical stage: only the fresh-prefix LRU pair below. After these two
results, stop and inspect the decomposition. The resolver pair is predeclared
confirmation only if the LRU pair does not robustly explain the gap. Fresh-prefix
isolates speculative/runtime behavior; it does not stand in for the user's
steady-state coding-agent throughput when production prefix reuse hits.

Exact prepared physical commands (one candidate per process; load weights and
run inference; **not executed**):

```sh
.venv/bin/python specs/001-bonsai-rollback-parity/evidence/probes/t027_product_ab.py --candidate qwen --prefix empty --workload lru --kv plain --adaptive --output specs/001-bonsai-rollback-parity/evidence/t027-results/qwen-lru.json
.venv/bin/python specs/001-bonsai-rollback-parity/evidence/probes/t027_product_ab.py --candidate bonsai --prefix empty --workload lru --kv plain --adaptive --output specs/001-bonsai-rollback-parity/evidence/t027-results/bonsai-lru.json
```

If and only if the fresh pair fails to reproduce or explain the observed Qwen
approximately 45 tok/s regime, the prepared reused-prefix diagnostic uses the
real production `PrefixCache.acquire()` mechanism. It seeds a cache through one
request using the exact full prompt, then repeats that exact request and requires
`reused_tokens > 0`; it does not shorten the prompt or delete tokens. This exact
retry hit is a narrow cache control, not a full multi-turn agent transcript.
Do not run it unless the fresh pair leaves the stated question unresolved:

```sh
.venv/bin/python specs/001-bonsai-rollback-parity/evidence/probes/t027_product_ab.py --candidate qwen --prefix reuse --workload lru --kv plain --adaptive --output specs/001-bonsai-rollback-parity/evidence/t027-results/qwen-lru-prefix-reuse.json
.venv/bin/python specs/001-bonsai-rollback-parity/evidence/probes/t027_product_ab.py --candidate bonsai --prefix reuse --workload lru --kv plain --adaptive --output specs/001-bonsai-rollback-parity/evidence/t027-results/bonsai-lru-prefix-reuse.json
```

The optional raw-production-drafter representation control, only if the primary
A/B leaves drafter representation as a plausible unresolved cause, is:

```sh
.venv/bin/python specs/001-bonsai-rollback-parity/evidence/probes/t027_product_ab.py --candidate bonsai --drafter-profile production --prefix empty --workload lru --kv plain --adaptive --output specs/001-bonsai-rollback-parity/evidence/t027-results/bonsai-production-drafter-control.json
```

The runner locks candidate paths internally and records config fingerprints
plus local HF weight blob identities. Preflight verifies the paths; when a
physical process loads the production tokenizer, it re-renders the exact prompt
and aborts before measurement unless those runtime IDs match the preflight IDs.
Candidate A
uses the production generic checkpoint and candidate B the Chad-built generic
q4g64 sidecar. Do not substitute the target-specific Bonsai sidecar.

KV8 is not a product candidate; no KV8 run is authorized here. Consider one
forensic run only if A/B leaves a specific causal ambiguity, it separates
hypotheses relevant to understanding/fixing the plain-KV path, and the result
could lead to an improvement that still ships with plain KV. Historical
plain→KV8 scheduler sensitivity already exists, so do not repeat it merely to
show margins/widths can change. Stop T027 once the plain-KV A/B decomposition
explains the product gap sufficiently to make the product decision; do not
expand the matrix for completeness. Component GPU microtiming is not mandatory
if counters already explain the gap; no async host-call timing is represented
as target or drafter device time.

## Offline validation and completion status

Offline validation completed for T025: runner compilation and `--help` passed;
all four runtime/arm dry-runs passed; each regenerated the exact 26 prompt IDs,
checked local paths/fingerprints, resolved its real source-backed arm control,
and emitted `weight_load_performed=false`. The dry-run tokenizer is loaded through
Transformers with `local_files_only=True`; importing `mlx_lm.utils` would initialize
Metal in this sandbox, so runtime mode independently repeats the same template/ID
assertion with the runtime's tokenizer before any generation. The existing sidecar
SHA-256 is reused from `controls.md`; dry-run checks its config hash and expected
file size/mtime without rereading the 1.1 GB weight file. `git --no-pager diff
--check` passed. T024 and T026 are checked; T026's observed overshoots and
cross-runtime output divergence are recorded in the adjudication JSON. T027
remains unchecked and unexecuted.

T027 preparation offline validation on 2026-09-24: runner `py_compile` and
`--help` passed. Four model-free preflights passed (Qwen/LRU 26 IDs,
Bonsai/LRU 26 IDs, Qwen/resolver 90 IDs, Bonsai/resolver 90 IDs); each asserted
the expected physical target and drafter paths, writable output, plain KV
requested, and `weight_load_performed=false`, `inference_performed=false`,
`benchmark_performed=false`. The tokenizer emitted expected Transformers
warnings because PyTorch is absent and Bonsai's custom model type is not a
Transformers model class; tokenization completed locally. Installed mlx-lm is
0.31.3: AST inspection confirmed `stream_generate(..., **kwargs)` forwards to
`generate_step`, whose sampling parameter is `sampler`; `temp` is not accepted.
Installed `mlx/core/metal.pyi` declares `get_active_memory()`,
`get_peak_memory()`, and `reset_peak_memory()`. `uv run pytest
tests/test_resolve.py -q` passed (30 tests), and
`uv run pytest tests/test_prefix_cache.py -q` passed (34 tests). No model
weights, inference, benchmark, or KV8 run was performed. `git diff --check`
passed for the preparation changes; T027 remains open.

<!-- T027_ACTUAL_START -->
## T027 actual plain-KV product A/B — final, 2026-09-24

**Status: FINAL.** The original two-run A/B and one justified target-specific
drafter control are complete. No further T027 operating points are requested.

Both candidates used the exact same 26-token T026 LRU prompt, greedy sampling,
fresh prefix, actual plain target KV, production `Engine.load` / `Engine.generate`
semantics, and a configured 262144 context window.

| Metric | Qwen3.8-27B 4-bit + plain KV | Bonsai2 + plain KV |
|---|---:|---:|
| Speculative decode | 45.779 tok/s | 30.389 tok/s |
| Serial target decode | 15.499 tok/s | 25.178 tok/s |
| Speculative speedup over serial | 2.954x | 1.207x |
| Generated tokens | 516 | 516 |
| Target forwards | 94 | 191 |
| Generated tokens / target forward | 5.489 | 2.702 |
| Draft acceptance | 64.8% | 36.2% |
| Round count | 93 | 190 |
| Width distribution | `{'7': 93}` | `{'7': 70, '1': 19, '0': 36, '6': 65}` |
| Source distribution | `{'drafter': 93}` | `{'drafter': 154, 'plain': 36}` |
| Speculative peak memory | 15.916 GiB | 8.751 GiB |

The fresh-prefix production-path Qwen result itself reproduces the previously
observed high-throughput regime: **45.779 tok/s**. Prefix reuse is therefore
not required to explain the approximately 45 tok/s Qwen behavior in this case.

The throughput gap is explained at the speculative-efficiency layer, not by a
slow Bonsai target. Bonsai serial target decode is **62.4% faster**
than Qwen serial, but Qwen speculative decode is **50.6% faster**
end-to-end.

For the same 516 generated tokens, Bonsai requires **191**
target forwards versus Qwen's **94** (2.03x).
Qwen obtains **5.489** generated
tokens/forward versus Bonsai's **2.702**.

Qwen remains at width 7 in all 93 recorded rounds and accepts
**64.8%** of proposed drafts. Bonsai accepts only
**36.2%** overall and spends 36 rounds at width 0,
19 at width 1, 65 at width 6 and 70 at width 7. Importantly, Bonsai's width-7
acceptance is only **33.9%**, versus Qwen's
**64.8%**, so the loss is not merely an adaptive-policy
choice to use narrower blocks: the drafter's proposals agree substantially less
with the Bonsai target even when both operate at width 7.

Aggregate decode time per reported target forward is
119.9 ms for Qwen versus
88.9 ms for Bonsai. This is an aggregate amortized
quantity, not isolated target-verify device latency. It shows that Bonsai rounds
are cheaper overall, but not cheap enough to compensate for roughly twice as
many target forwards.

Bonsai's clear measured product advantage is memory: speculative peak is
8.751 GiB versus
15.916 GiB for Qwen, about
45.0% lower.
On this machine that memory saving is not required to fit the stated Qwen
4-bit + plain-KV product configuration.

The Qwen production raw drafter and Bonsai Chad-built generic sidecar share the
same source checkpoint but use materially different runtime representations.
That remains a plausible deeper contributor to Bonsai's lower proposal
agreement. A same-drafter control is only relevant to a future attempt to rescue
Bonsai without changing target/KV precision; it is not needed to explain or
complete this product A/B.

Machine-readable adjudication:
`evidence/t027-product-ab-adjudication-20260924.json`.

### Final target-specific drafter control

The fresh production-path A/B localized the Bonsai product loss to speculative
proposal agreement. One additional control used the local Bonsai-specific
checkpoint with the same Bonsai target, plain target KV, adaptive DFlash, exact
26-token LRU prompt, thinking disabled, greedy sampling, 512 requested tokens,
fresh prefix, production context behavior (262144), and production
`Engine.load` / `Engine.generate` path. The control retained request-boundary
timing without per-round synchronization, measured-request cache inspection,
scoped speculative peak memory, all round/forward/proposal counters, width/source
distributions, and the warm serial target control.

Before its successful inference, the first load attempt exposed a loader format
assumption: this raw checkpoint stores selector codebooks as `nn.Embedding`
`.weight` tensors. The loader now selects that existing embedding-backed model
structure when those two checkpoint keys are present. After this narrow fix,
`py_compile`, `--help`, model-free preflight, and `git diff --check` all passed.
Preflight confirmed the exact target and drafter paths, locked prompt IDs,
plain KV, fresh prefix, and zero weight loads, inference, or benchmark execution.

Exact successful physical command:

```sh
.venv/bin/python specs/001-bonsai-rollback-parity/evidence/probes/t027_product_ab.py --candidate bonsai --drafter-profile bonsai-specific --target /Users/do/.cache/huggingface/hub/models--nathansutton--Qwen3.8-27B-Ternary-Bonsai-2-DFlash2-MLX/snapshots/234cc925d77cd94683e47b1493bd964917aaf43b --drafter /Users/do/models/Qwen3.8-27B-DFlash2-ternary-bonsai2 --prefix empty --workload lru --kv plain --adaptive --max-tokens 512 --output specs/001-bonsai-rollback-parity/evidence/t027-results/bonsai-specific-drafter-lru.json
```

| Metric | Qwen generic | Bonsai generic | Bonsai-specific drafter |
|---|---:|---:|---:|
| Speculative decode | 45.779 tok/s | 30.389 tok/s | 33.157 tok/s |
| Serial target decode | 15.499 tok/s | 25.178 tok/s | 25.121 tok/s |
| Speculative speedup over serial | 2.954x | 1.207x | 1.320x |
| Target forwards | 94 | 191 | 151 |
| Generated tokens / target forward | 5.489 | 2.702 | 3.404 |
| Target forwards / generated token | 0.182 | 0.370 | 0.294 |
| Proposed / accepted drafts | 651 / 422 | 899 / 325 | 924 / 363 |
| Accepted / proposed | 64.8% | 36.2% | 39.3% |
| Rounds | 93 | 190 | 150 |
| Width distribution | `{7: 93}` | `{7: 70, 6: 65, 1: 19, 0: 36}` | `{7: 121, 1: 17, 6: 10, 0: 2}` |
| Source distribution | `{drafter: 93}` | `{drafter: 154, plain: 36}` | `{drafter: 148, plain: 2}` |
| Speculative peak | 15.916 GiB | 8.751 GiB | 8.950 GiB |
| Actual attention KV class | `KVCache` | `KVCache` | `KVCache` |

The target-specific drafter materially reduced width-0/1 rounds and target
forwards versus the generic Bonsai pairing, and acceptance rose from 36.2% to
39.3%. It did not materially restore end-to-end throughput toward Qwen: 33.157
tok/s remains below the Qwen result of 45.779 tok/s and close to the generic
Bonsai regime. The existing Bonsai-specific drafter therefore does not rescue
the plain-KV product point. No Qwen rerun, other physical run, KV8, prefix reuse,
resolver, scheduler tuning, Chad benchmark, or GPT-Sol work occurred. The
initial loader failure occurred before inference; exactly one physical inference
completed successfully after the loader fix.
<!-- T027_ACTUAL_END -->
