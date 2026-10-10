"""
Composite Knowledge Distillation Loss implementation.
"""
from __future__ import annotations
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """
    Composite Knowledge Distillation Loss:
    L_total = alpha * L_CE(targets, student_logits) + (1 - alpha) * T^2 * L_KL(Softmax(student/T), Softmax(teacher/T))
    
    Args:
        temperature: Temperature hyperparameter scaling soften logits (T >= 1.0)
        alpha: Weighting coefficient for ground-truth CrossEntropy loss (0.0 <= alpha <= 1.0)
    """
    def __init__(self, temperature: float = 4.0, alpha: float = 0.5):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"Temperature must be strictly positive, got {temperature}")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"Alpha must be in range [0.0, 1.0], got {alpha}")
        self.temperature = float(temperature)
        self.alpha = float(alpha)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculates composite distillation loss and individual loss components.

        Args:
            student_logits: Unnormalized logits from student model (B, num_classes)
            teacher_logits: Unnormalized logits from teacher model (B, num_classes)
            targets: Ground-truth class labels (B,)

        Returns:
            Tuple of (total_loss_tensor, dictionary_of_scalar_components)
        """
        # Hard ground-truth Cross-Entropy loss
        ce_loss = F.cross_entropy(student_logits, targets)

        # Soft teacher-student KL Divergence loss with temperature scaling
        # T^2 scaling is essential to match gradient magnitudes of softened logits
        t = self.temperature
        p_student = F.log_softmax(student_logits / t, dim=-1)
        q_teacher = F.softmax(teacher_logits / t, dim=-1)
        
        # Using reduction="batchmean" aligns mathematically with KL divergence definition
        kl_loss = F.kl_div(p_student, q_teacher, reduction="batchmean") * (t * t)

        total_loss = (self.alpha * ce_loss) + ((1.0 - self.alpha) * kl_loss)

        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_ce": float(ce_loss.item()),
            "loss_kl": float(kl_loss.item()),
        }
        return total_loss, metrics
