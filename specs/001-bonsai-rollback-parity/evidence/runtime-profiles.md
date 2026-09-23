# Runtime profiles (T002)

Captured 2026-09-23 from installed environments and an actual read-only Chad model load.

| Runtime | Python | MLX | mlx-lm | Repository HEAD |
|---|---|---|---|---|
| mlx-dspark | 3.12.13 | 0.32.2 | 0.31.3 | `d05f06f6e7dd9b2ae1409dfb7b4cd960c44ff07e` |
| Chad | 3.11.15 | 0.32.2 | 0.31.3 | `8d7c1d9c7dfa084893916a910c5bdb62a506397c` |

The installed `mlx_lm/models/` files are byte-identical across both environments: `qwen3_5.py` SHA-256 `f0daa30bba5cb521c8bdfa7093101a544c6a37bbba09bca582288219cb04ae3a`, `gated_delta.py` `79c8376a51c694b03e54d2f996ced6ea6c8c42868b8571529f97334db165a3e1`, and `cache.py` `819ed95dcbf755652363cfdb15a639890447abb534a06dcefd52c7fff5055750`.

Both runtime inspections confirmed `qwen3_5.gated_delta_update is gated_delta.gated_delta_update`. The callable is `mlx_lm.models.gated_delta.gated_delta_update`, defined in installed `gated_delta.py:262`, with signature `(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, use_kernel=True)`. In mlx-dspark the capture hook in `src/mlx_dspark/target.py` wraps the defining GDN module and rollback calls the original function; in Chad the installed GDN `__call__` in `src/chad/mlx_fastpath.py:264` invokes the same callable. Equality of the function object and source does not establish equality of its inputs or kernel flags.

An actual `Engine(model_id=<exact local target snapshot>, dflash=False).load()` in a fresh Chad process recorded `Engine.fastpath=True` and unset `CHAD_NO_FASTPATH`. Layer 0 GDN class remained `mlx_lm.models.qwen3_5.GatedDeltaNet`, but its installed `__call__` resolves to `src/chad/mlx_fastpath.py:264`; layer 0 has `_fused_w` on both GDN and MLP. `src/chad/mlx_fastpath.py:74-110,254-385` installs the Prism fused qkv|z and b|a GDN projections, fused q/k/v attention projections, sign folding/rotation, and S=1 compiled layer path. Those installed attributes and callable identity are actual run observations, while whether each branch is taken for a given width is a per-run diagnostic still to capture. The alternate `CHAD_NO_FASTPATH=1` process has not yet been run.
