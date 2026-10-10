"""
Iterative Magnitude Pruner (IMP) with scheduled healing and LTH rewinding.
"""
from __future__ import annotations
from typing import Dict, List, Optional
import copy
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from compression_suite.pruning.masks import MaskManager
from compression_suite.pruning.rewinding import WeightRewinder
from compression_suite.distillation.trainer import evaluate_accuracy


class IterativeMagnitudePruner:
    """
    Coordinates multi-stage Iterative Magnitude Pruning:
    1. Computes scheduled sparsity at round t
    2. Updates dynamic parameter binary masks
    3. (Optionally) rewinds weights to W_0 (Lottery Ticket Hypothesis)
    4. Executes scheduled healing / fine-tuning epochs to recover lost accuracy
    """
    def __init__(
        self,
        model: nn.Module,
        target_sparsity: float = 0.5,
        prune_iterations: int = 3,
        healing_epochs: int = 2,
        lr: float = 1e-4,
        prune_mode: str = "global",
        use_lth_rewind: bool = False,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.target_sparsity = float(target_sparsity)
        self.prune_iterations = max(1, int(prune_iterations))
        self.healing_epochs = max(0, int(healing_epochs))
        self.lr = float(lr)
        self.prune_mode = prune_mode.lower()
        self.use_lth_rewind = bool(use_lth_rewind)

        self.mask_manager = MaskManager(self.model)
        self.rewinder = WeightRewinder(self.model) if self.use_lth_rewind else None

    def _train_healing_epoch(self, dataloader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        total_samples = 0

        for batch in dataloader:
            inputs, targets = batch[0].to(self.device), batch[1].to(self.device)
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        return total_loss / total_samples if total_samples > 0 else 0.0

    def prune_and_heal(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Executes the multi-round iterative magnitude pruning schedule.
        """
        history = {
            "round": [],
            "target_sparsity": [],
            "achieved_sparsity": [],
            "accuracy_pre_heal": [],
            "accuracy_post_heal": [],
        }

        # Initial baseline evaluation
        base_eval = evaluate_accuracy(self.model, val_loader, self.device)
        if verbose:
            print(f"--- IMP Initialization | Dense Accuracy: {base_eval['accuracy_top1']:.2f}% ---")

        for round_idx in range(1, self.prune_iterations + 1):
            t0 = time.time()
            # Geometric sparsity schedule: S_t = 1 - (1 - S_final)^(t / T)
            curr_target_sparsity = 1.0 - (1.0 - self.target_sparsity) ** (round_idx / self.prune_iterations)

            # Apply binary mask
            if self.prune_mode == "layerwise":
                achieved = self.mask_manager.apply_layerwise_magnitude_pruning(curr_target_sparsity)
            else:
                achieved = self.mask_manager.apply_global_magnitude_pruning(curr_target_sparsity)

            # Optional Lottery Ticket Hypothesis rewinding
            if self.use_lth_rewind and self.rewinder is not None:
                self.rewinder.rewind(self.mask_manager)

            # Accuracy immediately post-prune (pre-healing)
            pre_eval = evaluate_accuracy(self.model, val_loader, self.device)

            # Healing / Fine-tuning epochs
            if self.healing_epochs > 0:
                heal_params = [p for p in self.model.parameters() if p.requires_grad]
                if not heal_params:
                    for p in self.model.parameters():
                        p.requires_grad = True
                    heal_params = list(self.model.parameters())

                optimizer = torch.optim.AdamW(
                    heal_params,
                    lr=self.lr,
                    weight_decay=1e-4,
                )
                for _ in range(self.healing_epochs):
                    self._train_healing_epoch(train_loader, optimizer)

            # Accuracy post-healing
            post_eval = evaluate_accuracy(self.model, val_loader, self.device)
            elapsed = time.time() - t0

            history["round"].append(round_idx)
            history["target_sparsity"].append(curr_target_sparsity * 100.0)
            history["achieved_sparsity"].append(achieved)
            history["accuracy_pre_heal"].append(pre_eval["accuracy_top1"])
            history["accuracy_post_heal"].append(post_eval["accuracy_top1"])

            if verbose:
                print(
                    f"IMP Round [{round_idx}/{self.prune_iterations}] "
                    f"Target Sparsity: {curr_target_sparsity * 100.0:.1f}% | "
                    f"Achieved: {achieved:.2f}% | "
                    f"Acc Drop: {base_eval['accuracy_top1'] - pre_eval['accuracy_top1']:+.2f}% -> "
                    f"Healed Acc: {post_eval['accuracy_top1']:.2f}% | "
                    f"Time: {elapsed:.1f}s"
                )

        # Make pruning permanent in weights and remove dynamic hooks
        self.mask_manager.make_pruning_permanent()

        return history

    @classmethod
    def find_optimal_sparsity(
        cls,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        candidates: List[float] = [0.30, 0.50, 0.70, 0.85],
        max_accuracy_drop: float = 2.0,
        healing_epochs: int = 1,
        lr: float = 1e-4,
        device: Optional[torch.device] = None,
        verbose: bool = True,
    ) -> Tuple[float, List[Dict[str, float]]]:
        """
        Evaluates candidate sparsities and automatically identifies the maximum
        sparsity that remains within the allowed accuracy tolerance.
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_eval = evaluate_accuracy(model, val_loader, device)
        baseline_acc = base_eval["accuracy_top1"]

        if verbose:
            print(f"\n[Auto-Sparsity Search] Baseline Accuracy: {baseline_acc:.2f}% | Max Allowed Drop: {max_accuracy_drop:.2f}%")
            print(f"Testing Candidate Grid: {[f'{c*100:.0f}%' for c in candidates]}")

        grid_results = []
        best_sparsity = candidates[0]
        max_valid_sparsity = None

        for s in candidates:
            # Clone model to test each candidate independently
            test_model = copy.deepcopy(model).to(device)
            pruner = cls(
                model=test_model,
                target_sparsity=s,
                prune_iterations=2,
                healing_epochs=healing_epochs,
                lr=lr,
                device=device,
            )
            pruner.prune_and_heal(train_loader, val_loader, verbose=False)
            eval_res = evaluate_accuracy(test_model, val_loader, device)
            acc = eval_res["accuracy_top1"]
            drop = baseline_acc - acc
            within_sla = drop <= max_accuracy_drop

            if verbose:
                status = "PASS (Within SLA)" if within_sla else "FAIL (Exceeds SLA Drop)"
                print(f"  -> Target {s*100:.0f}%: Healed Acc {acc:.2f}% (Drop: {drop:+.2f}%) | {status}")

            grid_results.append({
                "sparsity": s,
                "sparsity_pct": s * 100.0,
                "accuracy": acc,
                "drop": drop,
                "within_sla": within_sla,
            })

            if within_sla:
                max_valid_sparsity = s

        if max_valid_sparsity is not None:
            best_sparsity = max_valid_sparsity
        else:
            # If all exceeded drop, pick the one with smallest drop
            best_sparsity = min(grid_results, key=lambda r: r["drop"])["sparsity"]

        if verbose:
            print(f"[Auto-Sparsity] >>> Optimal Sweet Spot Selected: {best_sparsity*100:.1f}% Sparsity <<<\n")

        return best_sparsity, grid_results



