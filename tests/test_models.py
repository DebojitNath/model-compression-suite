"""
Unit tests for model registry and compressible model architectures.
"""
import torch
import pytest
from compression_suite.models.registry import build_model, list_available_models, register_model
from compression_suite.models.base import BaseCompressibleModel


def test_registry_contains_expected_models():
    models = list_available_models()
    assert "resnet50" in models
    assert "resnet18" in models
    assert "fast_convnet_teacher" in models
    assert "fast_convnet_student" in models
    assert "bert_teacher" in models
    assert "minibert_student" in models


def test_build_and_forward_fast_convnet():
    teacher = build_model("fast_convnet_teacher", num_classes=10)
    student = build_model("fast_convnet_student", num_classes=10)

    x = torch.randn(2, 3, 32, 32)
    y_t = teacher(x)
    y_s = student(x)

    assert y_t.shape == (2, 10)
    assert y_s.shape == (2, 10)


def test_model_parameter_counting():
    student = build_model("fast_convnet_student", num_classes=10)
    total, active, sparsity = student.count_parameters()
    assert total > 0
    assert active > 0
    assert active <= total
    assert sparsity < 1.0  # Dense model has <1% zeros (from BatchNorm bias init)

    # Prunable module inspection
    prunables = student.get_prunable_modules()
    assert len(prunables) >= 3  # Conv2d and Linear layers


def test_custom_model_registration():
    @register_model("test_custom_net")
    def build_dummy(num_classes=5, **kwargs):
        return torch.nn.Linear(10, num_classes)

    assert "test_custom_net" in list_available_models()
    net = build_model("test_custom_net", num_classes=5)
    assert isinstance(net, torch.nn.Linear)
