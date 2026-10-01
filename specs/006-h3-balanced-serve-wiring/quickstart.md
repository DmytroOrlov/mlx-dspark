# Quickstart: H3 Serve Wiring

## Model-free validation

Run the focused CLI/hybrid tests after implementation:

```sh
python -m pytest -q tests/test_cli_pause.py tests/test_hybrid_target.py
```

Expected coverage: no donor options preserve the no-composition path; `56-63` expands exactly to indices 56–63; `62` selects one block; incomplete pairs and malformed/out-of-range/descending syntax fail before loading; and an explicit drafter remains independent. Cover comma-separated syntax if implemented.

## Optional Apple-Silicon smoke

With target/drafter/donor artifacts available, run the human-test command once and send one tiny OpenAI-compatible request to confirm startup and inference. This is not a benchmark or quality evaluation.

```sh
MLX_DSPARK_SLOW_ROUND_LOG_S=1 mlx-dspark serve \
  --api-key 'jA-s#q' \
  --model mlx-community/Qwen3.8-27B-4bit \
  --mode dflash \
  --drafter incoai/Qwen3.8-27B-DFlash2 \
  --donor-model prism-ml/Ternary-Bonsai-2-27B-mlx-2bit \
  --donor-blocks 56-63 \
  --host 0.0.0.0 \
  --port 13885 \
  --context-window 262144 \
  --reasoning-effort xhigh
```
