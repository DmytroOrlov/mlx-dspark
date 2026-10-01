"""Opt-in Qwen/Bonsai block composition for feature 002.

This module owns composition and evidence helpers only. Prism tensors stay in the
existing Packed representation; the ordinary loader remains unchanged when no
replacement payload is supplied.
"""
from __future__ import annotations

import gc
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Mapping

NUM_LAYERS = 64


class UnsafeDonorResidencyError(RuntimeError):
    """Raised before Qwen loading when donor pruning did not release enough MLX memory."""

    def __init__(self, message: str, *, memory: Mapping[str, Any]):
        super().__init__(message)
        self.memory = memory


@dataclass(frozen=True, slots=True)
class TargetCompositionRequest:
    donor_path: str
    donor_repo: str
    donor_revision: str
    qwen_repo: str
    qwen_revision: str
    donor_indices: tuple[int, ...]

    @classmethod
    def create(cls, *, donor_path: str, donor_repo: str, donor_revision: str,
               qwen_repo: str, qwen_revision: str,
               donor_indices) -> "TargetCompositionRequest":
        indices = normalize_indices(donor_indices)
        if not donor_path or not donor_repo:
            raise ValueError("donor path and repository identity are required")
        if len(donor_revision) != 40 or any(c not in "0123456789abcdef" for c in donor_revision):
            raise ValueError("donor revision must be a 40-character immutable commit SHA")
        if len(qwen_revision) != 40 or any(c not in "0123456789abcdef" for c in qwen_revision):
            raise ValueError("Qwen revision must be a 40-character immutable commit SHA")
        if not qwen_repo:
            raise ValueError("Qwen repository identity is required")
        return cls(str(donor_path), str(donor_repo), donor_revision,
                   str(qwen_repo), qwen_revision, indices)


@dataclass(frozen=True, slots=True)
class OwnershipManifest:
    qwen_repo: str
    qwen_revision: str
    donor_repo: str
    donor_revision: str
    donor_indices: tuple[int, ...]
    block_owners: tuple[str, ...]
    layer_families: tuple[str, ...]
    embedding_owner: str = "qwen"
    final_norm_owner: str = "qwen"
    lm_head_owner: str = "qwen"
    qwen_nonblock_identity_preserved: bool = True

    def as_dict(self) -> dict:
        return {
            "schema": "qwen-bonsai-ownership/v1",
            "qwen": {"repo_id": self.qwen_repo, "revision": self.qwen_revision},
            "bonsai": {"repo_id": self.donor_repo, "revision": self.donor_revision},
            "donor_indices": list(self.donor_indices),
            "block_owners": [
                {"index": index, "owner": owner, "layer_family": self.layer_families[index]}
                for index, owner in enumerate(self.block_owners)
            ],
            "embedding_owner": self.embedding_owner,
            "final_norm_owner": self.final_norm_owner,
            "lm_head_owner": self.lm_head_owner,
            "qwen_nonblock_identity_preserved": self.qwen_nonblock_identity_preserved,
        }

    def canonical_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class DonorLoadResult:
    blocks: Mapping[int, Any]
    memory: Mapping[str, Any]
    selected_parameter_bytes: int
    module_paths: tuple[str, ...]


def normalize_indices(indices) -> tuple[int, ...]:
    normalized = tuple(sorted(int(index) for index in indices))
    if len(set(normalized)) != len(normalized):
        raise ValueError("donor block indices must be unique")
    if any(index < 0 or index >= NUM_LAYERS for index in normalized):
        raise ValueError(f"donor block indices must be in [0, {NUM_LAYERS - 1}]")
    return normalized


def _layers(model):
    layers = getattr(model, "layers", None)
    if layers is None and hasattr(model, "language_model"):
        inner = getattr(model.language_model, "model", None)
        layers = getattr(inner, "layers", None)
    if layers is None and hasattr(model, "model"):
        layers = getattr(model.model, "layers", None)
    if layers is None or len(layers) != NUM_LAYERS:
        raise ValueError("composition requires the expected 64-block Qwen3.8 model")
    return layers


def _nonblock_components(model) -> dict[str, Any]:
    language_model = getattr(model, "language_model", model)
    inner = getattr(language_model, "model", None)
    if inner is None:
        return {}
    result = {"embedding": getattr(inner, "embed_tokens", None),
              "final_norm": getattr(inner, "norm", None)}
    head = getattr(language_model, "lm_head", None)
    if head is not None:
        result["lm_head"] = head
    return result


def layer_family(layer) -> str:
    is_linear = getattr(layer, "is_linear", None)
    if isinstance(is_linear, bool):
        return "linear_attention" if is_linear else "full_attention"
    family = getattr(layer, "layer_type", None) or getattr(layer, "block_type", None)
    if family in ("linear_attention", "full_attention"):
        return family
    raise ValueError(f"cannot determine decoder layer family for {type(layer).__name__}")


def packed_module_paths(model, *, predicate=None) -> tuple[str, ...]:
    """Return live Prism Packed module names; predicate injection keeps fixtures model-free."""
    if predicate is None:
        from . import prism_pack

        predicate = prism_pack.is_packed
    return tuple(name for name, module in model.named_modules() if predicate(module))


def replace_blocks(qwen_model, donor_blocks: Mapping[int, Any], donor_indices,
                   *, qwen_repo: str, qwen_revision: str,
                   donor_repo: str, donor_revision: str) -> OwnershipManifest:
    """Replace complete blocks before Target construction and prove by object identity."""
    indices = normalize_indices(donor_indices)
    if set(donor_blocks) != set(indices):
        raise ValueError("donor block payload indices must exactly match the composition indices")
    qwen_layers = _layers(qwen_model)
    original = tuple(qwen_layers)
    original_roles = _nonblock_components(qwen_model)
    families = tuple(layer_family(layer) for layer in original)
    for index in indices:
        donor = donor_blocks[index]
        if layer_family(donor) != families[index]:
            raise ValueError(
                f"same-index layer family mismatch at {index}: "
                f"Qwen={families[index]} donor={layer_family(donor)}")
    for index in indices:
        qwen_layers[index] = donor_blocks[index]
    owners = tuple("bonsai" if i in donor_blocks else "qwen" for i in range(NUM_LAYERS))
    manifest = OwnershipManifest(
        qwen_repo=qwen_repo, qwen_revision=qwen_revision,
        donor_repo=donor_repo, donor_revision=donor_revision,
        donor_indices=indices, block_owners=owners, layer_families=families)
    for index, expected_owner in enumerate(owners):
        expected = donor_blocks[index] if expected_owner == "bonsai" else original[index]
        if qwen_layers[index] is not expected:
            raise AssertionError(f"block ownership identity mismatch at index {index}")
    composed_roles = _nonblock_components(qwen_model)
    if original_roles and any(composed_roles.get(key) is not value
                              for key, value in original_roles.items()):
        raise AssertionError("Qwen embedding/norm/head identity changed during block replacement")
    return manifest


def memory_snapshot(label: str) -> dict:
    import mlx.core as mx

    return {
        "label": label,
        "timestamp_unix": time.time(),
        "active_bytes": int(mx.get_active_memory()),
        "cache_bytes": int(mx.get_cache_memory()),
        "peak_bytes": int(mx.get_peak_memory()),
    }


def _tree_parameter_bytes(module) -> int:
    from mlx.utils import tree_flatten

    return sum(int(value.nbytes) for _, value in tree_flatten(module.parameters())
               if hasattr(value, "nbytes"))


def load_donor_blocks(request: TargetCompositionRequest) -> DonorLoadResult:
    """Load through Prism, retain requested block objects, release the donor, and measure."""
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from . import prism_pack

    if request.donor_revision not in os.path.realpath(request.donor_path):
        raise ValueError("donor snapshot path does not identify the requested immutable revision")

    mx.clear_cache()
    before = memory_snapshot("before_donor_load")
    mx.reset_peak_memory()
    model, config = prism_pack.load(request.donor_path)
    layers = _layers(model)
    donor_config = config or json.loads(
        __import__("pathlib").Path(request.donor_path, "config.json").read_text())
    packed_paths = tuple(record["path"] for record in donor_config.get("modules", []))
    selected = {index: layers[index] for index in request.donor_indices}
    for block in selected.values():
        mx.eval([value for _, value in tree_flatten(block.parameters())
                 if hasattr(value, "dtype")])
    selected_bytes = sum(_tree_parameter_bytes(block) for block in selected.values())
    peak = memory_snapshot("donor_peak")
    del layers, model, config, donor_config
    gc.collect()
    mx.clear_cache()
    after = memory_snapshot("after_pruning_release_cache_cleanup")
    evidence = {
        "before_donor_load": before,
        "donor_peak": peak,
        "after_pruning_release_cache_cleanup": after,
        "selected_donor_parameter_bytes": selected_bytes,
        "full_donor_active_reclaim_fraction": (
            (peak["active_bytes"] - after["active_bytes"]) / peak["active_bytes"]
            if peak["active_bytes"] else None),
        "cache_cleared": after["cache_bytes"] <= before["cache_bytes"] + 256 * 1024 * 1024,
        "load_path": "prism_pack.load",
    }
    return DonorLoadResult(selected, evidence, selected_bytes, packed_paths)


def require_safe_before_qwen(donor: DonorLoadResult) -> dict:
    """Fail closed before Qwen loading if the full donor remains materially resident."""
    after = donor.memory["after_pruning_release_cache_cleanup"]
    peak = donor.memory["donor_peak"]
    reclaim_fraction = donor.memory["full_donor_active_reclaim_fraction"]
    # Eight retained blocks should be a minority of the full donor. The 50% floor is a
    # conservative observable release gate; the selected parameter byte count is retained
    # alongside it to expose allocator/accounting anomalies instead of assuming GC worked.
    if (reclaim_fraction is None or reclaim_fraction < 0.50 or
            after["active_bytes"] > peak["active_bytes"] * 0.50 or
            not donor.memory["cache_cleared"]):
        raise UnsafeDonorResidencyError(
            "donor pruning did not demonstrate material MLX memory reclamation; "
            "Qwen loading is refused", memory=donor.memory)
    return memory_snapshot("immediately_before_qwen_load")
