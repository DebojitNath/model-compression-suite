"""
Tests for input bundle resolution, checkpoint loading, generic image dataset,
smart output collision resolution, and bundle-driven CLI sweep.
"""
import os
import tempfile
import torch
import torch.nn as nn
from PIL import Image
import pytest

from compression_suite.utils.workspace import (
    resolve_input_bundle,
    resolve_output_dir,
    get_next_available_output_dir,
)
from compression_suite.models.loader import (
    load_model_from_checkpoint,
    resolve_model_from_directory,
)
from compression_suite.data.datasets import get_dataloaders, infer_sample_input
from compression_suite.models.registry import build_model
from compression_suite.cli import main


class TinyModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv(x)))
        x = torch.flatten(x, 1)
        return self.fc(x)


def create_dummy_image_dataset(root_dir: str, num_classes: int = 2, images_per_class: int = 4):
    for c in range(num_classes):
        class_dir = os.path.join(root_dir, f"class_{c}")
        os.makedirs(class_dir, exist_ok=True)
        for i in range(images_per_class):
            img = Image.new("RGB", (32, 32), color=(c * 100, i * 40, 120))
            img.save(os.path.join(class_dir, f"img_{i}.png"))


def test_resolve_input_bundle_validation():
    with tempfile.TemporaryDirectory() as tmp_dir:
        bundle_path = os.path.join(tmp_dir, "exp_1")
        os.makedirs(bundle_path)

        # Missing subdirs should raise FileNotFoundError
        with pytest.raises(FileNotFoundError, match="missing required subdirectories"):
            resolve_input_bundle(bundle_path)

        # Create teacher, student, dataset
        os.makedirs(os.path.join(bundle_path, "teacher"))
        os.makedirs(os.path.join(bundle_path, "student"))
        os.makedirs(os.path.join(bundle_path, "dataset"))

        bundle = resolve_input_bundle(bundle_path)
        assert bundle["bundle_name"] == "exp_1"
        assert os.path.isdir(bundle["teacher_dir"])
        assert os.path.isdir(bundle["student_dir"])
        assert os.path.isdir(bundle["dataset_dir"])


def test_smart_output_dir_resolution():
    with tempfile.TemporaryDirectory() as tmp_dir:
        base_out = os.path.join(tmp_dir, "my_run")
        os.makedirs(base_out)
        # Put a file so it's not empty
        with open(os.path.join(base_out, "dummy.txt"), "w") as f:
            f.write("hello")

        # Auto-increment candidate
        cand_1 = get_next_available_output_dir(base_out)
        assert cand_1 == f"{base_out}_1"

        # Resolve output non-interactive
        resolved = resolve_output_dir(base_out, interactive=False)
        assert resolved == f"{base_out}_1"

        # Explicit overwrite
        resolved_overwrite = resolve_output_dir(base_out, overwrite=True)
        assert resolved_overwrite == base_out


def test_load_model_from_checkpoint_full_and_statedict():
    with tempfile.TemporaryDirectory() as tmp_dir:
        model = TinyModel(num_classes=2)

        # 1. Full model save
        full_path = os.path.join(tmp_dir, "full_model.pt")
        torch.save(model, full_path)
        loaded_full = load_model_from_checkpoint(full_path)
        assert isinstance(loaded_full, nn.Module)
        out = loaded_full(torch.randn(1, 3, 32, 32))
        assert out.shape == (1, 2)

        # 2. State dict save with registered architecture
        state_path = os.path.join(tmp_dir, "state_dict.pt")
        fast_student = build_model("fast_convnet_student", num_classes=10)
        torch.save(fast_student.state_dict(), state_path)
        loaded_state = load_model_from_checkpoint(state_path, arch_name="fast_convnet_student", num_classes=10)
        assert isinstance(loaded_state, nn.Module)
        out_state = loaded_state(torch.randn(1, 3, 32, 32))
        assert out_state.shape == (1, 10)


def test_generic_image_folder_dataset_and_sample_input():
    with tempfile.TemporaryDirectory() as tmp_dir:
        data_dir = os.path.join(tmp_dir, "dataset")
        create_dummy_image_dataset(data_dir, num_classes=2, images_per_class=6)

        loaders = get_dataloaders(dataset_name=data_dir, batch_size=2)
        assert "train" in loaders
        assert "val" in loaders
        assert "test" in loaders
        assert "calib" in loaders

        sample = infer_sample_input(loaders)
        assert sample.shape == (1, 3, 32, 32)


def test_end_to_end_bundle_sweep(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp_dir:
        bundle_path = os.path.join(tmp_dir, "my_custom_experiment")
        teacher_dir = os.path.join(bundle_path, "teacher")
        student_dir = os.path.join(bundle_path, "student")
        dataset_dir = os.path.join(bundle_path, "dataset")

        os.makedirs(teacher_dir)
        os.makedirs(student_dir)
        os.makedirs(dataset_dir)

        # Save teacher and student models (.pt)
        teacher_model = build_model("fast_convnet_teacher", num_classes=2)
        student_model = build_model("fast_convnet_student", num_classes=2)
        torch.save(teacher_model, os.path.join(teacher_dir, "teacher.pt"))
        torch.save(student_model, os.path.join(student_dir, "student.pt"))

        # Create mini dataset
        create_dummy_image_dataset(dataset_dir, num_classes=2, images_per_class=4)

        output_dir = os.path.join(tmp_dir, "outputs", "my_custom_experiment")

        test_args = [
            "compression-suite",
            "sweep",
            "--input", bundle_path,
            "--output-dir", output_dir,
            "--epochs", "1",
            "--prune-iterations", "1",
            "--prune-sparsity", "0.20",
            "--healing-epochs", "0",
            "--batch-size", "2",
            "--device", "cpu",
            "--export", "none",
        ]

        monkeypatch.setattr("sys.argv", test_args)
        # Execute main CLI
        main()

        # Check outputs exist
        assert os.path.isfile(os.path.join(output_dir, "results.json"))
        assert os.path.isfile(os.path.join(output_dir, "results.csv"))
        assert os.path.isfile(os.path.join(output_dir, "model_compression_summary_report.md"))
        assert os.path.isfile(os.path.join(output_dir, "pareto_accuracy_vs_latency.png"))
