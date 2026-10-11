"""
Combinatorial Compression Pipeline Runner.
Orchestrates the 12 compression permutations across KD, IMP, and PTQ.
"""
from __future__ import annotations
from typing import Dict, List, Optional
import os
import shutil
import re
import json
import copy
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from compression_suite.distillation.trainer import DistillationTrainer
from compression_suite.pruning.pruner import IterativeMagnitudePruner
from compression_suite.quantization.quantizer import PostTrainingQuantizer
from compression_suite.profiling.metrics import ModelProfiler
from compression_suite.profiling.pareto import ParetoFrontier


class PipelineRunner:
    """
    Executes and benchmarks the 12 combinatorial compression pipelines.
    """
    def __init__(
        self,
        teacher: nn.Module,
        student_factory: callable,
        dataloaders: Dict[str, DataLoader],
        sample_input: torch.Tensor,
        device: Optional[torch.device] = None,
        train_epochs: int = 3,
        prune_iterations: int = 2,
        prune_target_sparsity: float | str = 0.5,
        sparsity_candidates: Optional[List[float]] = None,
        max_accuracy_drop: float = 2.0,
        healing_epochs: int = 1,
        ptq_method: str = "percentile",
        ptq_dtype: str = "int8",
        output_dir: Optional[str] = None,
        verbose: bool = True,
        train_teacher: bool = False,
        teacher_epochs: Optional[int] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.teacher = teacher.to(self.device)
        self.student_factory = student_factory
        self.dataloaders = dataloaders
        self.sample_input = sample_input.to(self.device)
        self.train_epochs = train_epochs
        self.prune_iterations = prune_iterations
        self.prune_target_sparsity = prune_target_sparsity
        self.sparsity_candidates = sparsity_candidates or [0.30, 0.50, 0.70, 0.85]
        self.max_accuracy_drop = float(max_accuracy_drop)
        self.healing_epochs = healing_epochs
        self.ptq_method = ptq_method
        self.ptq_dtype = ptq_dtype
        self.output_dir = output_dir or "./outputs"
        self.verbose = verbose
        self.train_teacher = train_teacher
        self.teacher_epochs = teacher_epochs

        self.effective_sparsity = 0.5 if str(prune_target_sparsity).lower() == "auto" else float(prune_target_sparsity)
        self.sparsity_grid_results = []
        self.staging_dir = os.path.join(self.output_dir, ".staging")
        os.makedirs(self.staging_dir, exist_ok=True)
        self.staged_paths: Dict[str, str] = {}
        self.trained_models: Dict[str, nn.Module] = {}

        self.profiler = ModelProfiler(device=self.device)
        self.quantizer = PostTrainingQuantizer(
            method=ptq_method,
            dtype=ptq_dtype,
        )

    def _train_model(self, model: nn.Module, teacher: Optional[nn.Module] = None, epochs: Optional[int] = None) -> nn.Module:
        """Trains or distills a model."""
        trainer = DistillationTrainer(
            student=model,
            teacher=teacher,
            device=self.device,
        )
        trainer.fit(
            train_loader=self.dataloaders["train"],
            val_loader=self.dataloaders["val"],
            epochs=epochs or self.train_epochs,
            verbose=self.verbose,
        )
        return trainer.student

    def _prune_model(self, model: nn.Module, use_lth: bool = False) -> nn.Module:
        """Applies IMP with healing."""
        pruner = IterativeMagnitudePruner(
            model=model,
            target_sparsity=self.effective_sparsity,
            prune_iterations=self.prune_iterations,
            healing_epochs=self.healing_epochs,
            use_lth_rewind=use_lth,
            device=self.device,
        )
        pruner.prune_and_heal(
            train_loader=self.dataloaders["train"],
            val_loader=self.dataloaders["val"],
            verbose=self.verbose,
        )
        return pruner.model

    def _quantize_model(self, model: nn.Module) -> nn.Module:
        """Applies PTQ."""
        return self.quantizer.quantize(
            model=model,
            calib_loader=self.dataloaders.get("calib"),
            use_native_torch=True,
            device=self.device,
        )

    def run_all_pipelines(self) -> List[Dict[str, any]]:
        """
        Executes all 12 pipeline configurations sequentially, profiling each artifact.
        """
        results: List[Dict[str, any]] = []

        print("\n=======================================================")
        print("Starting 12-Pipeline Combinatorial Compression Sweep")
        print("=======================================================\n")

        # -------------------------------------------------------------------
        # 1. Baseline Teacher FP32
        # -------------------------------------------------------------------
        p1_path = os.path.join(self.staging_dir, "1_baseline_teacher_fp32.pt")
        if os.path.exists(p1_path):
            print(">>> [1/12] Loading staged Baseline Teacher FP32...")
            t_model = torch.load(p1_path, map_location=self.device)
        elif self.train_teacher:
            print("\n>>> [1/12] Training Baseline Teacher FP32...")
            t_model = copy.deepcopy(self.teacher).to(self.device)
            t_model = self._train_model(t_model, teacher=None, epochs=self.teacher_epochs or self.train_epochs)
            torch.save(t_model, p1_path)
        else:
            print(">>> [1/12] Profiling Baseline Teacher FP32...")
            t_model = copy.deepcopy(self.teacher)
            torch.save(t_model, p1_path)
        res_1 = self.profiler.profile_full(t_model, self.dataloaders["test"], self.sample_input, "1_baseline_teacher_fp32")
        results.append(res_1)
        self.staged_paths["1_baseline_teacher_fp32"] = p1_path
        self.teacher = copy.deepcopy(t_model)

        # -------------------------------------------------------------------
        # 2. Baseline Student FP32 (trained from scratch without KD)
        # -------------------------------------------------------------------
        p2_path = os.path.join(self.staging_dir, "2_baseline_student_fp32.pt")
        if os.path.exists(p2_path):
            print("\n>>> [2/12] Loading staged Baseline Student FP32...")
            s_base = torch.load(p2_path, map_location=self.device)
        else:
            print("\n>>> [2/12] Training Baseline Student FP32 (No KD)...")
            s_base = self.student_factory().to(self.device)
            s_base = self._train_model(s_base, teacher=None)
            torch.save(s_base, p2_path)
        res_2 = self.profiler.profile_full(s_base, self.dataloaders["test"], self.sample_input, "2_baseline_student_fp32")
        results.append(res_2)
        self.staged_paths["2_baseline_student_fp32"] = p2_path

        # -------------------------------------------------------------------
        # 3. Distilled Student (KD)
        # -------------------------------------------------------------------
        p3_path = os.path.join(self.staging_dir, "3_distilled_student.pt")
        if os.path.exists(p3_path):
            print("\n>>> [3/12] Loading staged Distilled Student (KD)...")
            s_kd = torch.load(p3_path, map_location=self.device)
        else:
            print("\n>>> [3/12] Training Distilled Student (KD)...")
            s_kd = self.student_factory().to(self.device)
            s_kd = self._train_model(s_kd, teacher=self.teacher)
            torch.save(s_kd, p3_path)
        res_3 = self.profiler.profile_full(s_kd, self.dataloaders["test"], self.sample_input, "3_distilled_student")
        results.append(res_3)
        self.staged_paths["3_distilled_student"] = p3_path

        # -------------------------------------------------------------------
        # Auto-Sparsity Search (if prune_target_sparsity == "auto")
        # -------------------------------------------------------------------
        if str(self.prune_target_sparsity).lower() == "auto":
            print("\n=======================================================")
            print(">>> Auto-Sparsity Optimization: Finding Maximum Viable Sparsity")
            print("=======================================================")
            self.effective_sparsity, self.sparsity_grid_results = IterativeMagnitudePruner.find_optimal_sparsity(
                model=s_kd,
                train_loader=self.dataloaders["train"],
                val_loader=self.dataloaders["val"],
                candidates=self.sparsity_candidates,
                max_accuracy_drop=self.max_accuracy_drop,
                healing_epochs=self.healing_epochs,
                device=self.device,
                verbose=self.verbose,
            )

        # -------------------------------------------------------------------
        # 4. Pruned Teacher (IMP)
        # -------------------------------------------------------------------
        p4_path = os.path.join(self.staging_dir, "4_pruned_teacher.pt")
        if not self.train_teacher and os.path.exists(p4_path):
            print(f"\n>>> [4/12] Loading staged Pruned Teacher ({self.effective_sparsity*100:.1f}% sparsity)...")
            t_pruned = torch.load(p4_path, map_location=self.device)
        else:
            print(f"\n>>> [4/12] Pruning Teacher (IMP at {self.effective_sparsity*100:.1f}% sparsity)...")
            t_pruned = copy.deepcopy(self.teacher)
            t_pruned = self._prune_model(t_pruned, use_lth=False)
            torch.save(t_pruned, p4_path)
        res_4 = self.profiler.profile_full(t_pruned, self.dataloaders["test"], self.sample_input, "4_pruned_teacher")
        results.append(res_4)
        self.staged_paths["4_pruned_teacher"] = p4_path
        del t_pruned

        # -------------------------------------------------------------------
        # 5. Pruned Student (IMP on Baseline Student)
        # -------------------------------------------------------------------
        p5_path = os.path.join(self.staging_dir, "5_pruned_student.pt")
        if os.path.exists(p5_path):
            print("\n>>> [5/12] Loading staged Pruned Baseline Student (IMP)...")
            s_pruned = torch.load(p5_path, map_location=self.device)
        else:
            print("\n>>> [5/12] Pruning Baseline Student (IMP)...")
            s_pruned = copy.deepcopy(s_base)
            s_pruned = self._prune_model(s_pruned, use_lth=False)
            torch.save(s_pruned, p5_path)
        res_5 = self.profiler.profile_full(s_pruned, self.dataloaders["test"], self.sample_input, "5_pruned_student")
        results.append(res_5)
        self.staged_paths["5_pruned_student"] = p5_path

        # -------------------------------------------------------------------
        # 6. Distilled + Pruned Student (KD + IMP)
        # -------------------------------------------------------------------
        p6_path = os.path.join(self.staging_dir, "6_distilled_pruned_student.pt")
        if not self.train_teacher and os.path.exists(p6_path):
            print("\n>>> [6/12] Loading staged Distilled + Pruned Student (KD + IMP)...")
            s_kd_pruned = torch.load(p6_path, map_location=self.device)
        else:
            print("\n>>> [6/12] Pruning Distilled Student (KD + IMP)...")
            s_kd_pruned = copy.deepcopy(s_kd)
            s_kd_pruned = self._prune_model(s_kd_pruned, use_lth=False)
            torch.save(s_kd_pruned, p6_path)
        res_6 = self.profiler.profile_full(s_kd_pruned, self.dataloaders["test"], self.sample_input, "6_distilled_pruned_student")
        results.append(res_6)
        self.staged_paths["6_distilled_pruned_student"] = p6_path

        # -------------------------------------------------------------------
        # 7. Quantized Teacher INT8 (PTQ)
        # -------------------------------------------------------------------
        print("\n>>> [7/12] Quantizing Teacher INT8 (PTQ)...")
        t_quant = self._quantize_model(self.teacher)
        res_7 = self.profiler.profile_full(t_quant, self.dataloaders["test"], self.sample_input, "7_quantized_teacher_int8")
        results.append(res_7)
        p7_path = os.path.join(self.staging_dir, "7_quantized_teacher_int8.pt")
        torch.save(t_quant, p7_path)
        self.staged_paths["7_quantized_teacher_int8"] = p7_path
        del t_quant

        # -------------------------------------------------------------------
        # 8. Quantized Student INT8 (PTQ)
        # -------------------------------------------------------------------
        print("\n>>> [8/12] Quantizing Baseline Student INT8 (PTQ)...")
        s_quant = self._quantize_model(s_base)
        res_8 = self.profiler.profile_full(s_quant, self.dataloaders["test"], self.sample_input, "8_quantized_student_int8")
        results.append(res_8)
        p8_path = os.path.join(self.staging_dir, "8_quantized_student_int8.pt")
        torch.save(s_quant, p8_path)
        self.staged_paths["8_quantized_student_int8"] = p8_path
        del s_quant

        # -------------------------------------------------------------------
        # 9. Distilled + Quantized Student (KD + PTQ)
        # -------------------------------------------------------------------
        print("\n>>> [9/12] Quantizing Distilled Student INT8 (KD + PTQ)...")
        s_kd_quant = self._quantize_model(s_kd)
        res_9 = self.profiler.profile_full(s_kd_quant, self.dataloaders["test"], self.sample_input, "9_distilled_quantized_student")
        results.append(res_9)
        p9_path = os.path.join(self.staging_dir, "9_distilled_quantized_student.pt")
        torch.save(s_kd_quant, p9_path)
        self.staged_paths["9_distilled_quantized_student"] = p9_path
        del s_kd_quant

        # -------------------------------------------------------------------
        # 10. Pruned + Quantized Student (IMP + PTQ)
        # -------------------------------------------------------------------
        print("\n>>> [10/12] Quantizing Pruned Student INT8 (IMP + PTQ)...")
        s_pruned_quant = self._quantize_model(s_pruned)
        res_10 = self.profiler.profile_full(s_pruned_quant, self.dataloaders["test"], self.sample_input, "10_pruned_quantized_student")
        results.append(res_10)
        p10_path = os.path.join(self.staging_dir, "10_pruned_quantized_student.pt")
        torch.save(s_pruned_quant, p10_path)
        self.staged_paths["10_pruned_quantized_student"] = p10_path
        del s_pruned, s_pruned_quant

        # -------------------------------------------------------------------
        # 11. Trifecta: Distilled + Pruned + Quantized Student
        # -------------------------------------------------------------------
        print("\n>>> [11/12] Generating Trifecta Student (KD + IMP + PTQ)...")
        s_trifecta = self._quantize_model(s_kd_pruned)
        res_11 = self.profiler.profile_full(s_trifecta, self.dataloaders["test"], self.sample_input, "11_trifecta_student_int8")
        results.append(res_11)
        p11_path = os.path.join(self.staging_dir, "11_trifecta_student_int8.pt")
        torch.save(s_trifecta, p11_path)
        self.staged_paths["11_trifecta_student_int8"] = p11_path
        del s_kd_pruned, s_trifecta

        # -------------------------------------------------------------------
        # 12. LTH Rewound Pruned + Quantized Student
        # -------------------------------------------------------------------
        print("\n>>> [12/12] Generating LTH Rewound Student (KD + IMP with Rewinding + PTQ)...")
        s_lth = copy.deepcopy(s_kd)
        s_lth = self._prune_model(s_lth, use_lth=True)
        s_lth_quant = self._quantize_model(s_lth)
        res_12 = self.profiler.profile_full(s_lth_quant, self.dataloaders["test"], self.sample_input, "12_lth_rewound_student_int8")
        results.append(res_12)
        p12_path = os.path.join(self.staging_dir, "12_lth_rewound_student_int8.pt")
        torch.save(s_lth_quant, p12_path)
        self.staged_paths["12_lth_rewound_student_int8"] = p12_path
        del s_lth, s_lth_quant

        # Annotate with Pareto efficiency flags
        annotated_results = ParetoFrontier.identify_pareto_points(results)
        return annotated_results

    def export_selected_models(
        self,
        selection_query: str | List[str],
        output_dir: str,
        annotated_results: List[Dict[str, any]],
        cleanup_staging: bool = True,
    ) -> Dict[str, str]:
        """
        Transfers user-selected model checkpoints from staging to output_dir/checkpoints/.
        Supports multi-selection:
          - By numbers: '3', '3, 9', '3, 6, 9'
          - By keywords: 'top-acc', 'top-speed', 'unpruned', 'pareto', 'all', 'none'
          - By name: '3_distilled_student'
        """
        query_str = ", ".join(str(x) for x in selection_query) if isinstance(selection_query, list) else str(selection_query).strip()
        tokens = [t.strip().lower() for t in re.split(r'[,;\s]+', query_str) if t.strip()]

        if not tokens or "none" in tokens:
            if cleanup_staging and os.path.exists(self.staging_dir):
                shutil.rmtree(self.staging_dir, ignore_errors=True)
            return {}

        pipeline_order = [r["pipeline"] for r in annotated_results]
        selected_names = set()

        for tok in tokens:
            if tok == "all":
                selected_names.update(pipeline_order)
            elif tok == "top-acc":
                best = max(annotated_results, key=lambda r: r["accuracy_top1"])
                selected_names.add(best["pipeline"])
            elif tok == "top-speed":
                fastest = min(annotated_results, key=lambda r: r["latency_p50_ms"])
                selected_names.add(fastest["pipeline"])
            elif tok == "unpruned":
                for r in annotated_results:
                    if r.get("sparsity_pct", 0.0) < 1.0:
                        selected_names.add(r["pipeline"])
            elif tok == "pareto":
                for r in annotated_results:
                    if r.get("is_pareto_latency") or r.get("is_pareto_size") or r.get("is_pareto_3d"):
                        selected_names.add(r["pipeline"])
            elif tok.isdigit():
                idx = int(tok) - 1
                if 0 <= idx < len(pipeline_order):
                    selected_names.add(pipeline_order[idx])
            else:
                for p in pipeline_order:
                    if tok in p.lower():
                        selected_names.add(p)

        checkpoints_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(checkpoints_dir, exist_ok=True)
        saved_files = {}
        manifest = {}

        for p_name in pipeline_order:
            if p_name in selected_names and p_name in self.staged_paths:
                src = self.staged_paths[p_name]
                dst = os.path.join(checkpoints_dir, f"{p_name}.pt")
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                    saved_files[p_name] = dst
                    p_res = next((r for r in annotated_results if r["pipeline"] == p_name), {})
                    manifest[p_name] = {
                        "checkpoint_file": f"{p_name}.pt",
                        "accuracy_top1": p_res.get("accuracy_top1"),
                        "latency_p50_ms": p_res.get("latency_p50_ms"),
                        "disk_size_mb": p_res.get("disk_size_mb"),
                        "active_params": p_res.get("active_params"),
                        "sparsity_pct": p_res.get("sparsity_pct"),
                    }

        if manifest:
            manifest_path = os.path.join(checkpoints_dir, "manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump({"exported_models": manifest}, f, indent=2)

        if cleanup_staging and os.path.exists(self.staging_dir):
            shutil.rmtree(self.staging_dir, ignore_errors=True)

        return saved_files

    def save_checkpoints(
        self,
        output_dir: str,
        annotated_results: List[Dict[str, any]],
        save_all: bool = False,
    ) -> Dict[str, str]:
        """Backward-compatible wrapper around export_selected_models."""
        choice = "all" if save_all else "pareto"
        return self.export_selected_models(choice, output_dir, annotated_results, cleanup_staging=False)


