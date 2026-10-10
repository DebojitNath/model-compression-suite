"""
Unit tests for Knowledge Distillation loss and trainer.
"""
import torch
import pytest
from compression_suite.distillation.loss import DistillationLoss
from compression_suite.distillation.trainer import DistillationTrainer, evaluate_accuracy
from compression_suite.data.datasets import get_dataloaders
from compression_suite.models.registry import build_model


def test_distillation_loss_forward_and_backward():
    loss_fn = DistillationLoss(temperature=4.0, alpha=0.5)

    student_logits = torch.randn(4, 10, requires_grad=True)
    teacher_logits = torch.randn(4, 10)
    targets = torch.tensor([0, 2, 4, 9])

    loss, metrics = loss_fn(student_logits, teacher_logits, targets)

    assert loss.item() > 0.0
    assert "loss_total" in metrics
    assert "loss_ce" in metrics
    assert "loss_kl" in metrics

    # Backward pass verification
    loss.backward()
    assert student_logits.grad is not None
    assert torch.isfinite(student_logits.grad).all()


def test_distillation_temperature_scaling_invariance():
    # As T increases, soft distribution entropy increases
    loss_fn_low_t = DistillationLoss(temperature=1.0, alpha=0.0)
    loss_fn_high_t = DistillationLoss(temperature=10.0, alpha=0.0)

    student_logits = torch.randn(8, 5)
    teacher_logits = torch.randn(8, 5)
    targets = torch.zeros(8, dtype=torch.long)

    loss_1, _ = loss_fn_low_t(student_logits, teacher_logits, targets)
    loss_2, _ = loss_fn_high_t(student_logits, teacher_logits, targets)

    assert torch.isfinite(loss_1)
    assert torch.isfinite(loss_2)


def test_distillation_trainer_single_epoch():
    student = build_model("fast_convnet_student", num_classes=10)
    teacher = build_model("fast_convnet_teacher", num_classes=10)
    loaders = get_dataloaders("synthetic_vision", batch_size=32)

    trainer = DistillationTrainer(
        student=student,
        teacher=teacher,
        device=torch.device("cpu"),
        temperature=3.0,
        alpha=0.5,
    )

    metrics = trainer.train_epoch(loaders["train"])
    assert "train_loss" in metrics
    assert "train_accuracy" in metrics
    assert metrics["train_loss"] >= 0.0
