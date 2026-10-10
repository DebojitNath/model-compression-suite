"""
Unit tests for Iterative Magnitude Pruning, binary masking, and LTH rewinding.
"""
import torch
import torch.nn as nn
import pytest
from compression_suite.pruning.masks import ParameterMask, MaskManager
from compression_suite.pruning.rewinding import WeightRewinder
from compression_suite.pruning.pruner import IterativeMagnitudePruner
from compression_suite.models.registry import build_model
from compression_suite.data.datasets import get_dataloaders


def test_parameter_mask_forward_and_gradient_zeroing():
    layer = nn.Linear(4, 2, bias=False)
    layer.weight.data = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

    p_mask = ParameterMask(layer, "weight")
    # Mask out indices (0, 0) and (1, 3)
    binary_mask = torch.tensor([[0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0]])
    p_mask.set_mask(binary_mask)

    # 1. Check weight zeroing
    assert layer.weight[0, 0].item() == 0.0
    assert layer.weight[1, 3].item() == 0.0

    # 2. Forward pass under mask
    x = torch.ones(1, 4)
    out = layer(x)
    loss = out.sum()
    loss.backward()

    # 3. Check gradient zeroing on masked weights
    assert layer.weight.grad[0, 0].item() == 0.0
    assert layer.weight.grad[1, 3].item() == 0.0
    assert layer.weight.grad[0, 1].item() != 0.0

    p_mask.remove()


def test_mask_manager_global_pruning_sparsity():
    model = build_model("fast_convnet_student", num_classes=10)
    manager = MaskManager(model)

    achieved_sparsity = manager.apply_global_magnitude_pruning(target_sparsity=0.50)
    # Check achieved sparsity is approximately 50%
    assert abs(achieved_sparsity - 50.0) < 2.0

    breakdown = manager.get_sparsity_breakdown()
    assert "overall" in breakdown
    assert abs(breakdown["overall"] - 50.0) < 2.0

    manager.remove_all()


def test_lth_weight_rewinding():
    model = nn.Sequential(nn.Linear(10, 10, bias=False))
    init_val = model[0].weight.data.clone()

    rewinder = WeightRewinder(model)
    manager = MaskManager(model)

    # Simulate training by mutating weights
    model[0].weight.data.add_(10.0)
    assert not torch.equal(model[0].weight.data, init_val)

    # Prune 40% of weights
    manager.apply_global_magnitude_pruning(0.40)
    mask = manager.masks["0.weight"].mask

    # Rewind
    rewinder.rewind(manager)

    # Survived weights should equal init_val; pruned weights should be zero
    survived_mask = mask.bool()
    pruned_mask = ~survived_mask

    assert torch.equal(model[0].weight.data[survived_mask], init_val[survived_mask])
    assert torch.all(model[0].weight.data[pruned_mask] == 0.0)

    manager.remove_all()


def test_auto_sparsity_sweet_spot_search():
    dataloaders = get_dataloaders("synthetic_vision", batch_size=32)
    model = build_model("fast_convnet_student", num_classes=10)

    # Test auto-search across [0.3, 0.6] with 1 round / 0 healing epoch for speed
    best_s, grid = IterativeMagnitudePruner.find_optimal_sparsity(
        model=model,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        candidates=[0.30, 0.60],
        max_accuracy_drop=15.0,  # Generous tolerance so highest candidate passes
        healing_epochs=0,
        verbose=False,
    )

    assert len(grid) == 2
    assert "sparsity_pct" in grid[0]
    assert "accuracy" in grid[0]
    assert "drop" in grid[0]
    assert "within_sla" in grid[0]
    # Since tolerance is 15.0%, both 0.3 and 0.6 are within SLA, so best should be 0.60
    assert best_s == 0.60

