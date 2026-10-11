"""
Command-Line Interface (CLI) for Model Compression Suite.
"""
from __future__ import annotations
import argparse
import os
import sys
from typing import List, Dict, Any, Optional
import yaml
import torch

from compression_suite.models.registry import build_model, list_available_models
from compression_suite.models.loader import resolve_model_from_directory
from compression_suite.data.datasets import get_dataloaders, infer_sample_input
from compression_suite.pipeline.runner import PipelineRunner
from compression_suite.pipeline.experiment import save_experiment_results
from compression_suite.visualization.plots import generate_all_plots
from compression_suite.utils.workspace import resolve_input_bundle, resolve_output_dir

VISION_MODELS = {"fast_convnet_teacher", "fast_convnet_student", "resnet18", "resnet34", "resnet50"}
TEXT_MODELS = {"bert_teacher", "minibert_student"}


def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def handle_sweep(args):
    config = {}
    if args.config:
        config = load_config(args.config)

    # 1. Resolve Input Bundle if specified
    input_bundle = None
    input_dir = getattr(args, "input_dir", None) or config.get("input_dir") or config.get("input")
    if input_dir:
        input_bundle = resolve_input_bundle(input_dir)

    # CLI args override YAML config
    teacher_name = args.teacher or config.get("teacher_model", "fast_convnet_teacher")
    student_name = args.student or config.get("student_model", "fast_convnet_student")

    if input_bundle:
        dataset_name = input_bundle["dataset_dir"]
        default_out = f"./outputs/{input_bundle['bundle_name']}"
    else:
        dataset_name = args.dataset or config.get("dataset")
        default_out = "./outputs/sweep_results"

    if not dataset_name:
        print("\n[ERROR] Missing required dataset argument!")
        print("You must specify a dataset using '--dataset <name>' (e.g. --dataset cifar10), '--input <bundle_dir>', or define 'dataset: ...' in a YAML config.\n")
        sys.exit(1)

    raw_output_dir = args.output_dir or config.get("output_dir", default_out)
    overwrite_flag = getattr(args, "overwrite", None)
    output_dir = resolve_output_dir(raw_output_dir, overwrite=overwrite_flag)
    os.makedirs(output_dir, exist_ok=True)

    epochs = args.epochs if args.epochs is not None else config.get("train_epochs", 2)

    raw_sparsity = args.prune_sparsity if args.prune_sparsity is not None else config.get("prune_target_sparsity", "auto")
    if str(raw_sparsity).strip().lower() == "auto":
        prune_sparsity = "auto"
    else:
        prune_sparsity = float(raw_sparsity)

    sparsity_candidates = args.sparsity_candidates or config.get("sparsity_candidates", [0.30, 0.50, 0.70, 0.85])
    max_accuracy_drop = args.max_accuracy_drop if args.max_accuracy_drop is not None else config.get("max_accuracy_drop", 2.0)

    prune_iterations = args.prune_iterations if args.prune_iterations is not None else config.get("prune_iterations", 2)
    healing_epochs = args.healing_epochs if args.healing_epochs is not None else config.get("healing_epochs", 1)
    ptq_method = args.ptq_method or config.get("ptq_method", "percentile")
    ptq_dtype = args.ptq_dtype or config.get("ptq_dtype", "int8")
    batch_size = args.batch_size or config.get("batch_size", 64)

    device_str = args.device or config.get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    if prune_sparsity == "auto":
        sparsity_str = f"AUTO-TUNING (Candidates: {[f'{c*100:.0f}%' for c in sparsity_candidates]} | Max Drop <= {max_accuracy_drop:.1f}%)"
    else:
        sparsity_str = f"{prune_sparsity * 100:.1f}% (Hardcoded Fixed)"

    num_classes = config.get("num_classes")
    if num_classes is None:
        if dataset_name.lower() in ["sst2", "sst-2", "glue_sst2", "synthetic_text"] or teacher_name in TEXT_MODELS:
            num_classes = 2
        elif dataset_name.lower() == "cifar100":
            num_classes = 100
        else:
            num_classes = 10

    # 2. Prepare Data
    max_samples = getattr(args, "max_samples", None) or config.get("max_samples")
    dataloaders = get_dataloaders(
        dataset_name=dataset_name,
        batch_size=batch_size,
        num_classes=num_classes,
        max_samples=max_samples,
    )

    # 3. Build Models
    if input_bundle:
        teacher, _ = resolve_model_from_directory(
            input_bundle["teacher_dir"],
            role="teacher",
            default_arch=args.teacher or config.get("teacher_model"),
            device=device,
        )
        _, student_fn = resolve_model_from_directory(
            input_bundle["student_dir"],
            role="student",
            default_arch=args.student or config.get("student_model"),
            device=device,
        )
        teacher_label = f"Bundle [{getattr(teacher, '__class__', type(teacher)).__name__}]"
        student_label = f"Bundle [{getattr(student_fn(), '__class__', type(student_fn())).__name__}]"
    else:
        teacher = build_model(teacher_name, num_classes=num_classes)
        student_fn = lambda: build_model(student_name, num_classes=num_classes)
        teacher_label = teacher_name
        student_label = student_name

    # 4. Derive sample input
    sample_input = infer_sample_input(dataloaders, device=device)

    print("===============================================================")
    print("      Model Compression Suite - Combinatorial Sweep           ")
    print("===============================================================")
    if input_bundle:
        print(f"Input Bundle     : {input_bundle['bundle_name']} ({input_bundle['bundle_path']})")
    print(f"Teacher Model    : {teacher_label}")
    print(f"Student Model    : {student_label}")
    print(f"Dataset          : {dataset_name}")
    print(f"Device           : {device}")
    print(f"Train Epochs     : {epochs}")
    print(f"Prune Sparsity   : {sparsity_str} ({prune_iterations} iterations, {healing_epochs} healing ep)")
    print(f"PTQ Calibration  : {ptq_method.upper()} ({ptq_dtype.upper()})")
    print(f"Output Directory : {output_dir}")
    print("===============================================================\n")

    # Smart train_teacher resolution:
    # 1. If explicitly set via CLI flag (--train-teacher / --no-train-teacher), respect it.
    # 2. If defined in YAML config, respect it.
    # 3. Otherwise: Input bundles assume pre-trained checkpoint (False);
    #    Fresh code architectures without pre-trained checkpoints need training (True).
    cli_train_teacher = getattr(args, "train_teacher", None)
    if cli_train_teacher is not None:
        train_teacher = cli_train_teacher
    elif "train_teacher" in config:
        train_teacher = bool(config["train_teacher"])
    else:
        train_teacher = (input_bundle is None)

    teacher_epochs = getattr(args, "teacher_epochs", None) or config.get("teacher_epochs", None)

    # 5. Run Pipeline
    runner = PipelineRunner(
        teacher=teacher,
        student_factory=student_fn,
        dataloaders=dataloaders,
        sample_input=sample_input,
        device=device,
        train_epochs=epochs,
        prune_iterations=prune_iterations,
        prune_target_sparsity=prune_sparsity,
        sparsity_candidates=sparsity_candidates,
        max_accuracy_drop=max_accuracy_drop,
        healing_epochs=healing_epochs,
        ptq_method=ptq_method,
        ptq_dtype=ptq_dtype,
        output_dir=output_dir,
        verbose=True,
        train_teacher=train_teacher,
        teacher_epochs=teacher_epochs,
    )

    results = runner.run_all_pipelines()

    # 5. Persist Results
    meta = {
        "teacher_model": teacher_name,
        "student_model": student_name,
        "dataset": dataset_name,
        "device": str(device),
        "ptq_method": ptq_method,
        "ptq_dtype": ptq_dtype,
        "prune_target_sparsity_mode": "auto" if prune_sparsity == "auto" else "fixed",
        "effective_prune_sparsity": runner.effective_sparsity,
        "sparsity_candidates": sparsity_candidates if prune_sparsity == "auto" else None,
        "max_accuracy_drop": max_accuracy_drop if prune_sparsity == "auto" else None,
        "auto_sparsity_search_results": runner.sparsity_grid_results if prune_sparsity == "auto" else None,
    }
    saved_paths = save_experiment_results(results, output_dir, metadata=meta)

    # 6. Generate Pareto Visualizations
    print("\n>>> Generating Publication-Grade Pareto Frontier Visualizations...")
    generate_all_plots(
        results,
        output_dir,
        sparsity_grid_results=runner.sparsity_grid_results,
        optimal_sparsity=runner.effective_sparsity if prune_sparsity == "auto" else None,
    )

    # 7. Summary Table & Manual Model Export
    print("\n" + "=" * 85)
    print("                         COMPRESSION SWEEP SUMMARY TABLE")
    print("=" * 85)
    print(f"{'#':<4} {'Pipeline Name':<34} {'Top-1 Acc':<11} {'Lat (P50)':<12} {'Size':<10} {'Sparsity':<10} {'Pareto?'}")
    print("-" * 85)
    for idx, r in enumerate(results, 1):
        p_name = r["pipeline"]
        acc = f"{r['accuracy_top1']:.1f}%"
        lat = f"{r['latency_p50_ms']:.2f} ms"
        sz = f"{r['disk_size_mb']:.2f} MB"
        sp = f"{r['sparsity_pct']:.1f}%"
        is_p = "YES" if (r.get("is_pareto_latency") or r.get("is_pareto_size") or r.get("is_pareto_3d")) else "-"
        print(f"[{idx:<2}] {p_name:<32} {acc:<11} {lat:<12} {sz:<10} {sp:<10} {is_p}")
    print("=" * 85)

    export_choice = getattr(args, "export", None)
    if export_choice is None and "export_models" in config:
        export_choice = str(config.get("export_models"))
    if export_choice is None:
        if sys.stdin.isatty():
            print("\nSelect model(s) to export to disk (.pt):")
            print("  - Single or multi-choice numbers: e.g. '3' or '3, 9' or '3, 6, 9'")
            print("  - Or keywords: 'top-acc', 'top-speed', 'unpruned', 'pareto', 'all', 'none'")
            export_choice = input("Your selection ['none']: ").strip() or "none"
        else:
            export_choice = "none"

    saved_ckpts = runner.export_selected_models(
        selection_query=export_choice,
        output_dir=output_dir,
        annotated_results=results,
        cleanup_staging=True,
    )

    if saved_ckpts:
        print(f"\n>>> Exported {len(saved_ckpts)} Model Checkpoint(s) to: {os.path.join(output_dir, 'checkpoints')}")
        for name, path in saved_ckpts.items():
            print(f"  [+] {name} -> {path}")
    else:
        print("\n>>> No model checkpoints exported. Staging memory cleared.")

    print("\n===============================================================")
    print("                 Sweep Execution Completed!                   ")
    print("===============================================================")
    print(f"Metrics JSON   : {saved_paths['json']}")
    print(f"Metrics CSV    : {saved_paths['csv']}")
    print(f"Summary Report : {saved_paths['report']}")
    print(f"Pareto Plots   : {output_dir}")
    if saved_ckpts:
        print(f"Checkpoints    : {os.path.join(output_dir, 'checkpoints')}")
    print("===============================================================\n")




def handle_list_models(args):
    print("Available Registered Architectures in Model Compression Suite:")
    for name in list_available_models():
        print(f"  - {name}")


TEACHER_MODELS_VISION = [
    ("resnet50", "ResNet-50 (23.5M params - Deep ResNet)"),
    ("resnet34", "ResNet-34 (21.3M params - Medium ResNet)"),
    ("fast_convnet_teacher", "FastConvNet Teacher (2.8M params - Lightweight CNN)"),
]

STUDENT_MODELS_VISION = [
    ("resnet18", "ResNet-18 (11.2M params - Compact ResNet)"),
    ("fast_convnet_student", "FastConvNet Student (0.7M params - Tiny CNN)"),
]

TEACHER_MODELS_TEXT = [
    ("bert_teacher", "BERT-base (109.5M params - Transformer)"),
]

STUDENT_MODELS_TEXT = [
    ("minibert_student", "MiniBERT (14.4M params - 4-layer Transformer)"),
]


def prompt_user(prompt_text: str, default: str) -> str:
    val = input(f"{prompt_text} [{default}]: ").strip()
    return val if val else default


def select_from_menu(title: str, options: List[tuple[str, str]], default_idx: int = 1) -> str:
    print(f"\n{title}:")
    for idx, (name, desc) in enumerate(options, 1):
        print(f"  [{idx}] {name} ({desc})")
    raw = input(f"Select option [default: {default_idx}]: ").strip()
    if not raw:
        return options[default_idx - 1][0]
    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(options):
            return options[idx][0]
    for name, _ in options:
        if raw.lower() == name.lower():
            return name
    print(f"  [!] Invalid choice '{raw}'. Using default: {options[default_idx - 1][0]}")
    return options[default_idx - 1][0]


def handle_interactive(args):
    print("\n" + "=" * 65)
    print("      MODEL COMPRESSION SUITE - Interactive Setup")
    print("=" * 65)
    print("Press [Enter] at any prompt to accept the default option.\n")

    # 1. Domain Selection
    domain_options = [
        ("vision", "Computer Vision (Standard models: ResNet, FastConvNet on CIFAR-10)"),
        ("text", "Natural Language Processing (Standard models: BERT, MiniBERT on SST-2)"),
        ("custom_bundle", "Custom Input Bundle (Your own folder with teacher/, student/, dataset/)"),
    ]
    domain = select_from_menu("Step 1: Select Domain / Input Type", domain_options, default_idx=1)

    input_dir = None
    teacher_name = None
    student_name = None
    dataset_name = None

    if domain == "custom_bundle":
        input_dir = prompt_user("Enter path to your custom bundle directory", "./inputs/resnet_cifar10")
        print(f"\nConfiguration Summary:")
        print(f"  - Input Bundle  : {input_dir}")
    elif domain == "vision":
        teacher_name = select_from_menu("Step 2: Select Teacher Architecture", TEACHER_MODELS_VISION, default_idx=1)
        student_name = select_from_menu("Step 3: Select Student Architecture", STUDENT_MODELS_VISION, default_idx=1)

        dataset_options = [
            ("cifar10", "CIFAR-10 (Standard 10-class vision benchmark)"),
            ("synthetic", "Synthetic Images (Fast test, 100 samples)"),
            ("custom_dir", "Custom Image Folder (Specify your own directory path)"),
        ]
        chosen_ds = select_from_menu("Step 4: Select Dataset", dataset_options, default_idx=1)
        if chosen_ds == "custom_dir":
            dataset_name = prompt_user("Enter path to your image dataset folder", "./data/my_images")
        else:
            dataset_name = chosen_ds

        print(f"\nConfiguration Summary:")
        print(f"  - Teacher Model : {teacher_name}")
        print(f"  - Student Model : {student_name}")
        print(f"  - Dataset       : {dataset_name}")
    else:
        teacher_name = select_from_menu("Step 2: Select Teacher Architecture", TEACHER_MODELS_TEXT, default_idx=1)
        student_name = select_from_menu("Step 3: Select Student Architecture", STUDENT_MODELS_TEXT, default_idx=1)

        dataset_options = [
            ("sst2", "SST-2 (GLUE benchmark for binary sentiment)"),
            ("synthetic_text", "Synthetic Sentences (Fast test, 100 samples)"),
            ("custom_dir", "Custom Text Folder (Specify your own directory path)"),
        ]
        chosen_ds = select_from_menu("Step 4: Select Dataset", dataset_options, default_idx=1)
        if chosen_ds == "custom_dir":
            dataset_name = prompt_user("Enter path to your text dataset folder", "./data/my_texts")
        else:
            dataset_name = chosen_ds

        print(f"\nConfiguration Summary:")
        print(f"  - Teacher Model : {teacher_name}")
        print(f"  - Student Model : {student_name}")
        print(f"  - Dataset       : {dataset_name}")

    # 3. Sparsity Setting
    sparsity_menu = [
        ("auto", "Auto-tune (Search 30%, 50%, 70%, 85% for best accuracy)"),
        ("0.50", "Fixed 50% pruning (Fast, intermediate compression)"),
        ("0.85", "Fixed 85% pruning (Maximum parameter compression)"),
    ]
    raw_sparsity = select_from_menu("Step 5: Select Pruning Sparsity Mode", sparsity_menu, default_idx=1)

    if raw_sparsity == "auto":
        prune_sparsity = "auto"
        sparsity_candidates = [0.30, 0.50, 0.70, 0.85]
        max_drop = float(prompt_user("Max Allowed Accuracy Drop in %", "2.0"))
    else:
        prune_sparsity = float(raw_sparsity)
        sparsity_candidates = None
        max_drop = 2.0

    # 4. Training Epochs & Iterations
    print("\nStep 6: Training & Quantization Parameters:")
    epochs = int(prompt_user("Training / Distillation Epochs", "2"))
    prune_iters = int(prompt_user("Prune Iterations (IMP Rounds)", "2"))
    healing_ep = int(prompt_user("Healing Fine-tuning Epochs per round", "1"))
    ptq_method = prompt_user("Quantization Calibration (percentile / minmax)", "percentile")
    output_dir = prompt_user("Output Directory", "./outputs/interactive_sweep")

    # Construct dummy args and run
    sweep_args = argparse.Namespace(
        config=None,
        input_dir=input_dir,
        teacher=teacher_name,
        student=student_name,
        dataset=dataset_name,
        output_dir=output_dir,
        epochs=epochs,
        prune_sparsity=prune_sparsity,
        sparsity_candidates=sparsity_candidates,
        max_accuracy_drop=max_drop,
        prune_iterations=prune_iters,
        healing_epochs=healing_ep,
        ptq_method=ptq_method,
        ptq_dtype="int8",
        batch_size=64,
        device="auto",
    )
    handle_sweep(sweep_args)


def main():
    parser = argparse.ArgumentParser(
        prog="compression-suite",
        description="Model Compression Suite: Knowledge Distillation, Pruning, Quantization, and Pareto Profiling",
    )
    subparsers = parser.add_subparsers(dest="command")

    # Subcommand: sweep (CLI flags or YAML config)
    sweep_parser = subparsers.add_parser("sweep", help="Run full 12-pipeline combinatorial compression sweep via flags, YAML, or input bundle")
    sweep_parser.add_argument("--input", "--input-dir", dest="input_dir", type=str, default=None, help="Path to input experiment bundle containing teacher/, student/, and dataset/ subdirectories")
    sweep_parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    sweep_parser.add_argument("--teacher", type=str, default=None, help="Name of registered teacher model")
    sweep_parser.add_argument("--student", type=str, default=None, help="Name of registered student model")
    sweep_parser.add_argument("--dataset", type=str, default=None, help="Dataset name: synthetic_vision, cifar10, or custom directory")
    sweep_parser.add_argument("--output-dir", type=str, default=None, help="Directory to save artifacts and plots")
    sweep_parser.add_argument("--overwrite", action="store_true", default=None, help="Force overwrite output directory if it already exists")
    sweep_parser.add_argument("--epochs", type=int, default=None, help="Training / Distillation epochs")
    sweep_parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of dataset samples to load for rapid iteration")
    sweep_parser.add_argument("--prune-sparsity", type=str, default=None, help="Target pruning sparsity: 'auto' or float (e.g. 0.5)")
    sweep_parser.add_argument("--sparsity-candidates", type=float, nargs="+", default=None, help="Candidate sparsities for auto sweep (e.g. 0.3 0.5 0.7 0.85)")
    sweep_parser.add_argument("--max-accuracy-drop", type=float, default=None, help="Max acceptable accuracy drop in % for auto sparsity (default: 2.0)")
    sweep_parser.add_argument("--prune-iterations", type=int, default=None, help="Number of IMP rounds")
    sweep_parser.add_argument("--healing-epochs", type=int, default=None, help="Fine-tuning healing epochs per round")
    sweep_parser.add_argument("--ptq-method", type=str, default=None, choices=["percentile", "minmax"])
    sweep_parser.add_argument("--ptq-dtype", type=str, default=None, choices=["int8", "fp16"])
    sweep_parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    sweep_parser.add_argument("--device", type=str, default=None, help="'cpu', 'cuda', or 'auto'")
    sweep_parser.add_argument("--export", type=str, default=None, help="Comma-separated model(s) to export (e.g. '3', '3, 9', 'top-acc', 'all', 'none')")
    sweep_parser.add_argument("--train-teacher", dest="train_teacher", action="store_true", default=None, help="Force training the teacher model before sweeping")
    sweep_parser.add_argument("--no-train-teacher", dest="train_teacher", action="store_false", help="Skip teacher training (assumes pre-trained)")
    sweep_parser.add_argument("--teacher-epochs", type=int, default=None, help="Number of training epochs for the teacher model (default: same as --epochs)")
    sweep_parser.set_defaults(func=handle_sweep)

    # Subcommand: interactive (Interactive terminal setup)
    interactive_parser = subparsers.add_parser("interactive", help="Guided interactive terminal setup")
    interactive_parser.set_defaults(func=handle_interactive)

    # Subcommand: list-models
    list_parser = subparsers.add_parser("list-models", help="List registered compressible models")
    list_parser.set_defaults(func=handle_list_models)

    args = parser.parse_args()
    if args.command is None:
        # If user runs without subcommands, launch interactive setup by default
        handle_interactive(args)
    else:
        args.func(args)


if __name__ == "__main__":
    main()

