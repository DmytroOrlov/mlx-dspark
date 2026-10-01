# Physical comparison controls (T001)

Recorded on 2026-09-23. The maintainer identified the historical `benchmarks/spec_lru.py` `seeds` / `greedy` / `schedule` input: one user message with content `Write a production-quality Python LRU cache with tests and type hints.`, rendered with `add_generation_prompt=True`, `enable_thinking=False`, and a 512 generated-token budget. No historical token-ID dump survived; the IDs below were regenerated from the fingerprinted target tokenizer. The historical R1/R2/pre-R3 acceptances and margins remain observations to reproduce, never forced expectations.

| Input | Exact identity / control |
|---|---|
| Target | `nathansutton/Qwen3.8-27B-Ternary-Bonsai-2-DFlash2-MLX`, local snapshot `234cc925d77cd94683e47b1493bd964917aaf43b` |
| Target `config.json` SHA-256 | `20a7ccac3e519b5b5d7c4eaa6451352dd4188da44b4385ffa6cf87a4f2b8a94a` |
| Target `model.safetensors` SHA-256 | `68541bf9c72747df90764338fa966105b34d94e4e8751dd5d3403d732eedddcf` |
| Tokenizer `tokenizer.json` SHA-256 | `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523` |
| Tokenizer `tokenizer_config.json` SHA-256 | `95c557768e6b88a7128befc7bfd3c7de50e5d51af9b8b33a9f4dee0e04f99679` |
| Physical sidecar | `/Users/do/.cache/chad/dflash/015e795645c74b1a0eeef3b570031fb62e769bc5-q4g64` |
| Sidecar `config.json` SHA-256 | `6fd15051e629eba87121298bce299f92f1ef99c14e01e144825bbf0aaeaca86f` |
| Sidecar `model.safetensors` SHA-256 | `876c368b5abfdd5de52ab14fcd5d3cccb07f0059f3903c984e4f17dbcee8a552` |
| Device | Local Apple Silicon, MLX `Device(gpu, 0)` with Metal access during approved runtime call |
| Target architecture / loader | `prism_hadamard_qwen35`; mlx-dspark `load_target` routes through `src/mlx_dspark/prism_pack.py`; Chad `Engine.load` routes through `src/chad/prism_pack.py` |
| Drafter shape | DFlash2 block size 8, draft width 7, target taps `[5, 19, 33, 47, 61]`, five context layers |
| Sampling | Greedy, temperature 0, thinking disabled, no prompt lookup; top-p/top-k in Chad greedy preset 0/0; exact generator flags must be captured with each run |
| Initial state | Fresh empty target and drafter caches; no reused prefix checkpoint; 26 prompt tokens before any generated token. Subsequent anchor, proposal, accepted-count and live-cache fields must be captured by the round probe, not guessed from this setup. |
| Kernel flags | No `MLX_*` or `CHAD_*` variables in the shell at inspection. Chad normal fastpath installed in an actual model load; mlx-dspark Prism MMA install/dispatch remains to be captured with its model-backed run. |

The no-thinking chat-template prompt IDs, from `mlx_lm.utils.load_tokenizer` against the exact snapshot, are `[248045, 846, 198, 7734, 264, 5492, 21408, 12654, 436, 34810, 6297, 440, 6813, 321, 913, 29642, 13, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271]`. These are regenerated from the actual user message and template settings, not invented or copied from an absent historical dump. The active run must independently regenerate them and record any mismatch.

The exact sidecar was loaded directly by `mlx_dspark.load.load_dflash(path)` with no registry lookup or conversion: the loader reported 47 b4g64 and 2 b8g64 quantized modules, `DFlashDraftModel`, block size 8 and taps `(5, 19, 33, 47, 61)`. This establishes that a direct adapter or substituted drafter is unnecessary for the local mlx-dspark runtime.
