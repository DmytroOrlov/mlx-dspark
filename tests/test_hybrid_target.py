"""Model-free contracts for the opt-in Qwen/Bonsai composition layer."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from mlx_dspark.hybrid_target import (
    NUM_LAYERS,
    TargetCompositionRequest,
    normalize_indices,
    packed_module_paths,
    replace_blocks,
)

ROOT = Path(__file__).resolve().parents[1]


class FakeLayer:
    def __init__(self, is_linear: bool, label: str):
        self.is_linear = is_linear
        self.label = label


class FakeModel:
    def __init__(self, *, label: str, families=None):
        families = families or [i % 4 != 3 for i in range(NUM_LAYERS)]
        self.layers = [FakeLayer(family, f"{label}-{i}") for i, family in enumerate(families)]

    def named_modules(self):
        yield "", self
        for index, layer in enumerate(self.layers):
            yield f"layers.{index}", layer
            packed = getattr(layer, "packed", None)
            if packed is not None:
                yield f"layers.{index}.proj", packed


class FakePacked:
    pass


def identity_args(indices=(63,)):
    return dict(qwen_repo="mlx-community/Qwen3.8-27B-4bit",
                qwen_revision="1" * 40,
                donor_repo="prism-ml/Ternary-Bonsai-2-27B-mlx-2bit",
                donor_revision="2" * 40)


@pytest.mark.parametrize("raw, expected", [
    ([63], (63,)),
    ([63, 60, 62], (60, 62, 63)),
    ([], ()),
])
def test_indices_are_canonical_sorted_sets(raw, expected):
    assert normalize_indices(raw) == expected


@pytest.mark.parametrize("raw", [[1, 1], [-1], [64], [0, 64]])
def test_duplicate_and_out_of_range_indices_reject(raw):
    with pytest.raises(ValueError):
        normalize_indices(raw)


def test_composition_replaces_arbitrary_complete_blocks_and_proves_identity():
    qwen = FakeModel(label="qwen")
    original = tuple(qwen.layers)
    donor = FakeModel(label="bonsai")
    indices = (17, 42, 63)
    donor_blocks = {index: donor.layers[index] for index in indices}
    manifest = replace_blocks(qwen, donor_blocks, reversed(indices), **identity_args())

    assert manifest.donor_indices == indices
    assert len(manifest.block_owners) == NUM_LAYERS
    for index in range(NUM_LAYERS):
        expected = donor.layers[index] if index in indices else original[index]
        assert qwen.layers[index] is expected
    assert [row["owner"] for row in manifest.as_dict()["block_owners"]].count("bonsai") == 3


@pytest.mark.parametrize("indices", [(63,), (62,), (62, 63), (60, 61, 62, 63),
                                      (56, 57, 58, 59, 60, 61, 62, 63)])
def test_required_suffix_mappings(indices):
    qwen = FakeModel(label="qwen")
    original = tuple(qwen.layers)
    donor = FakeModel(label="bonsai")
    replacements = {index: donor.layers[index] for index in indices}
    manifest = replace_blocks(qwen, replacements, indices, **identity_args())
    assert tuple(manifest.donor_indices) == indices
    assert all(qwen.layers[i] is donor.layers[i] for i in indices)
    assert all(qwen.layers[i] is original[i] for i in range(NUM_LAYERS) if i not in indices)


def test_layer_family_mismatch_fails_before_replacement():
    qwen = FakeModel(label="qwen")
    donor = FakeModel(label="bonsai")
    donor.layers[62] = FakeLayer(not qwen.layers[62].is_linear, "wrong-family")
    original = tuple(qwen.layers)
    with pytest.raises(ValueError, match="layer family mismatch at 62"):
        replace_blocks(qwen, {62: donor.layers[62]}, [62], **identity_args())
    assert all(actual is expected for actual, expected in zip(qwen.layers, original))


def test_ownership_uses_live_identity_not_shared_layer_class_name():
    class SameClassLayer:
        is_linear = False

    qwen = FakeModel(label="qwen")
    donor = FakeModel(label="bonsai")
    qwen.layers[63] = SameClassLayer()
    donor.layers[63] = SameClassLayer()
    manifest = replace_blocks(qwen, {63: donor.layers[63]}, [63], **identity_args())
    assert type(qwen.layers[63]) is type(donor.layers[63])
    assert qwen.layers[63] is donor.layers[63]
    assert manifest.block_owners[63] == "bonsai"


def test_b0_is_outside_partial_composition_and_packed_inventory_is_discoverable():
    qwen = FakeModel(label="qwen")
    donor = FakeModel(label="bonsai")
    donor.layers[0].packed = FakePacked()
    assert packed_module_paths(donor, predicate=lambda item: isinstance(item, FakePacked)) == (
        "layers.0.proj",)
    # B0 is represented by the existing full-donor loader, not a 64-entry replacement request.
    assert normalize_indices([]) == ()
    assert len(donor.layers) == NUM_LAYERS
    assert len(qwen.layers) == NUM_LAYERS


def test_request_requires_immutable_revisions_and_sorts_indices():
    request = TargetCompositionRequest.create(
        donor_path="/snapshot/" + "2" * 40,
        donor_repo="prism-ml/Ternary-Bonsai-2-27B-mlx-2bit",
        donor_revision="2" * 40,
        qwen_repo="mlx-community/Qwen3.8-27B-4bit",
        qwen_revision="1" * 40,
        donor_indices=[63, 62],
    )
    assert request.donor_indices == (62, 63)
    with pytest.raises(ValueError, match="immutable commit SHA"):
        TargetCompositionRequest.create(
            donor_path="/snapshot/main", donor_repo="official", donor_revision="main",
            qwen_repo="qwen", qwen_revision="1" * 40, donor_indices=[63])


def test_manifest_serialization_is_canonical_and_complete():
    qwen = FakeModel(label="qwen")
    donor = FakeModel(label="bonsai")
    manifest = replace_blocks(qwen, {62: donor.layers[62], 63: donor.layers[63]},
                              [63, 62], **identity_args())
    first = manifest.canonical_json()
    second = manifest.canonical_json()
    assert first == second
    record = manifest.as_dict()
    assert len(record["block_owners"]) == 64
    assert record["embedding_owner"] == record["final_norm_owner"] == record["lm_head_owner"] == "qwen"


def test_default_loader_and_engine_branches_preserve_no_composition_path():
    load_tree = ast.parse((ROOT / "src/mlx_dspark/load.py").read_text())
    load_fn = next(node for node in load_tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == "load_target")
    replacement_guard = next(node for node in ast.walk(load_fn)
                             if isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                             and isinstance(node.test.left, ast.Name)
                             and node.test.left.id == "_block_replacements")
    assert isinstance(replacement_guard.test.ops[0], ast.IsNot)
    assert isinstance(replacement_guard.test.comparators[0], ast.Constant)
    assert replacement_guard.test.comparators[0].value is None

    server_tree = ast.parse((ROOT / "src/mlx_dspark/server.py").read_text())
    load_method = next(node for node in ast.walk(server_tree) if isinstance(node, ast.FunctionDef)
                       and node.name == "load")
    default_guard = next(node for node in ast.walk(load_method)
                         if isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                         and isinstance(node.test.left, ast.Name)
                         and node.test.left.id == "target_composition")
    assert isinstance(default_guard.test.ops[0], ast.Is)
    assert default_guard.test.comparators[0].value is None
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
               and node.func.id == "load_target"
               for statement in default_guard.body for node in ast.walk(statement))
