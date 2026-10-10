"""
Teacher-Student Knowledge Distillation training loop and evaluator.
"""
from __future__ import annotations
from typing import Dict, List, Optional
import copy
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from compression_suite.distillation.loss import DistillationLoss


def evaluate_accuracy(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluates Top-1 accuracy, Cross-Entropy loss, and latency on a dataloader.
    """
    model = model.to(device)
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch[0].to(device), batch[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = (correct / total) * 100.0 if total > 0 else 0.0
    avg_loss = total_loss / total if total > 0 else 0.0
    return {
        "accuracy_top1": acc,
        "loss": avg_loss,
        "total_samples": total,
    }


class DistillationTrainer:
    """
    Training harness for standard model training and teacher-student distillation.
    """
    def __init__(
        self,
        student: nn.Module,
        teacher: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        temperature: float = 4.0,
        alpha: float = 0.5,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.student = student.to(self.device)
        self.teacher = teacher.to(self.device) if teacher is not None else None

        if self.teacher is not None:
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False
            self.criterion = DistillationLoss(temperature=temperature, alpha=alpha)
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=10,
            eta_min=1e-5,
        )

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.student.train()
        total_loss = 0.0
        total_samples = 0
        correct = 0

        total_batches = len(dataloader)
        log_interval = max(10, min(50, total_batches // 4))

        for b_idx, batch in enumerate(dataloader, 1):
            inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
            self.optimizer.zero_grad()

            student_logits = self.student(inputs)

            if self.teacher is not None:
                with torch.no_grad():
                    teacher_logits = self.teacher(inputs)
                loss, _ = self.criterion(student_logits, teacher_logits, targets)
            else:
                loss = self.criterion(student_logits, targets)

            loss.backward()
            self.optimizer.step()

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            _, predicted = student_logits.max(1)
            correct += predicted.eq(targets).sum().item()

            if total_batches > 15 and (b_idx % log_interval == 0 or b_idx == total_batches):
                current_acc = (correct / total_samples) * 100.0
                print(f"    -> Progress: Batch [{b_idx:04d}/{total_batches:04d}] ({b_idx/total_batches*100:4.1f}%) | Batch Loss: {loss.item():.4f} | Running Acc: {current_acc:.2f}%", flush=True)

        self.scheduler.step()
        epoch_loss = total_loss / total_samples if total_samples > 0 else 0.0
        epoch_acc = (correct / total_samples) * 100.0 if total_samples > 0 else 0.0
        return {
            "train_loss": epoch_loss,
            "train_accuracy": epoch_acc,
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 5,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Runs the full training loop and keeps the best student checkpoint based on val accuracy.
        """
        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }
        best_acc = -1.0
        best_weights = copy.deepcopy(self.student.state_dict())

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_metrics = self.train_epoch(train_loader)
            val_metrics = evaluate_accuracy(self.student, val_loader, self.device)
            elapsed = time.time() - t0

            history["train_loss"].append(train_metrics["train_loss"])
            history["train_accuracy"].append(train_metrics["train_accuracy"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_accuracy"].append(val_metrics["accuracy_top1"])

            if val_metrics["accuracy_top1"] > best_acc:
                best_acc = val_metrics["accuracy_top1"]
                best_weights = copy.deepcopy(self.student.state_dict())

            if verbose:
                print(
                    f"Epoch [{epoch:02d}/{epochs:02d}] "
                    f"Train Loss: {train_metrics['train_loss']:.4f} Acc: {train_metrics['train_accuracy']:.2f}% | "
                    f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy_top1']:.2f}% | "
                    f"Time: {elapsed:.1f}s"
                )

        self.student.load_state_dict(best_weights)
        return history
