"""
Knowledge Distillation module init.
"""
from compression_suite.distillation.loss import DistillationLoss
from compression_suite.distillation.trainer import DistillationTrainer, evaluate_accuracy

__all__ = [
    "DistillationLoss",
    "DistillationTrainer",
    "evaluate_accuracy",
]
