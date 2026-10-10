"""
End-to-end integration test running the 12-pipeline sweep and verifying output artifacts.
"""
import os
import tempfile
import torch
import pytest

from compression_suite.models.registry import build_model
from compression_suite.data.datasets import get_dataloaders
from compression_suite.pipeline.runner import PipelineRunner
from compression_suite.pipeline.experiment import save_experiment_results
from compression_suite.visualization.plots import generate_all_plots


def test_full_12_pipeline_sweep_integration():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Load lightweight fast models
        teacher = build_model("fast_convnet_teacher", num_classes=10)
        student_fn = lambda: build_model("fast_convnet_student", num_classes=10)

        dataloaders = get_dataloaders("synthetic_vision", batch_size=32)
        sample_input = torch.randn(1, 3, 32, 32)

        runner = PipelineRunner(
            teacher=teacher,
            student_factory=student_fn,
            dataloaders=dataloaders,
            sample_input=sample_input,
            device=torch.device("cpu"),
            train_epochs=1,
            prune_iterations=1,
            prune_target_sparsity=0.30,
            healing_epochs=0,
            ptq_method="percentile",
            ptq_dtype="int8",
            verbose=False,
        )

        results = runner.run_all_pipelines()

        # 1. Verify all 12 pipelines completed
        assert len(results) == 12
        expected_pipelines = [
            "1_baseline_teacher_fp32",
            "2_baseline_student_fp32",
            "3_distilled_student",
            "4_pruned_teacher",
            "5_pruned_student",
            "6_distilled_pruned_student",
            "7_quantized_teacher_int8",
            "8_quantized_student_int8",
            "9_distilled_quantized_student",
            "10_pruned_quantized_student",
            "11_trifecta_student_int8",
            "12_lth_rewound_student_int8",
        ]
        actual_names = [r["pipeline"] for r in results]
        assert actual_names == expected_pipelines

        # 2. Check Pareto optimality annotations
        for r in results:
            assert "is_pareto_latency" in r
            assert "is_pareto_size" in r
            assert "is_pareto_3d" in r

        # 3. Save experiment outputs
        paths = save_experiment_results(results, tmp_dir)
        assert os.path.isfile(paths["json"])
        assert os.path.isfile(paths["csv"])
        assert os.path.isfile(paths["report"])

        # 4. Generate plots
        generate_all_plots(results, tmp_dir)
        assert os.path.isfile(os.path.join(tmp_dir, "pareto_accuracy_vs_latency.png"))
        assert os.path.isfile(os.path.join(tmp_dir, "pareto_accuracy_vs_size.png"))
        assert os.path.isfile(os.path.join(tmp_dir, "pareto_3d_frontier.png"))
        assert os.path.isfile(os.path.join(tmp_dir, "pipeline_comparison_bars.png"))

        # 5. Verify selective multi-model export and staging cleanup
        exported = runner.export_selected_models("3, 9", tmp_dir, results, cleanup_staging=True)
        assert len(exported) == 2
        assert "3_distilled_student" in exported
        assert "9_distilled_quantized_student" in exported
        assert os.path.isfile(exported["3_distilled_student"])
        assert os.path.isfile(exported["9_distilled_quantized_student"])
        assert os.path.isfile(os.path.join(tmp_dir, "checkpoints", "manifest.json"))
        # Verify staging directory was deleted
        assert not os.path.exists(runner.staging_dir)

