"""
Workspace management, input bundle resolution, and smart output collision handling.
"""
from __future__ import annotations
from typing import Dict, List, Optional
import os
import sys
import re


def get_next_available_output_dir(base_dir: str) -> str:
    """
    Finds the next available directory name by appending _1, _2, _3, etc.
    E.g., if outputs/my_exp exists, tries outputs/my_exp_1, outputs/my_exp_2...
    """
    clean_base = base_dir.rstrip("/\\")
    if not os.path.exists(clean_base):
        return clean_base

    counter = 1
    while True:
        candidate = f"{clean_base}_{counter}"
        if not os.path.exists(candidate):
            return candidate
        counter += 1


def resolve_output_dir(
    base_output_dir: str,
    overwrite: Optional[bool] = None,
    interactive: bool = True,
) -> str:
    """
    Resolves the final output directory path, handling collisions with user prompts
    or automatic version incrementing (_1, _2, etc.).

    Args:
        base_output_dir: Desired output directory path
        overwrite: If True, forces overwrite. If False, forces new version.
                   If None, prompts user (if interactive) or auto-increments.
        interactive: Whether to prompt via terminal if sys.stdin.isatty().

    Returns:
        Resolved absolute or relative directory path.
    """
    clean_dir = os.path.normpath(base_output_dir)

    if not os.path.exists(clean_dir):
        return clean_dir

    # Check if directory is empty
    try:
        if not os.listdir(clean_dir):
            return clean_dir
    except OSError:
        pass

    next_candidate = get_next_available_output_dir(clean_dir)

    # If explicit flag passed
    if overwrite is True:
        return clean_dir
    if overwrite is False:
        return next_candidate

    # Interactive prompt if running in an interactive terminal
    if interactive and sys.stdin.isatty():
        print(f"\n[!] Notice: Output directory '{clean_dir}' already exists and contains files.")
        print(f"    [1] Create new version: '{next_candidate}' [Default]")
        print(f"    [2] Overwrite existing directory: '{clean_dir}'")
        print(f"    [3] Cancel sweep")
        choice = input("Select an option [1]: ").strip()

        if choice == "2":
            return clean_dir
        elif choice == "3":
            print("Sweep cancelled by user.")
            sys.exit(0)
        else:
            return next_candidate

    # Non-interactive default: safe auto-increment
    return next_candidate


def resolve_input_bundle(bundle_path: str) -> Dict[str, any]:
    """
    Scans and validates a convention-based input bundle directory.
    Expected structure:
        <bundle_path>/
        ├── teacher/   (contains .pt/.pth model or state_dict)
        ├── student/   (contains .pt/.pth model or state_dict)
        └── dataset/   (contains image folders or dataset files)

    Returns:
        Dict with paths and discovered assets.
    """
    bundle_path = os.path.normpath(os.path.abspath(bundle_path))
    if not os.path.isdir(bundle_path):
        raise FileNotFoundError(f"Input bundle directory does not exist: {bundle_path}")

    bundle_name = os.path.basename(bundle_path)

    teacher_dir = os.path.join(bundle_path, "teacher")
    student_dir = os.path.join(bundle_path, "student")
    dataset_dir = os.path.join(bundle_path, "dataset")

    missing = []
    if not os.path.isdir(teacher_dir):
        missing.append("teacher/")
    if not os.path.isdir(student_dir):
        missing.append("student/")
    if not os.path.isdir(dataset_dir):
        missing.append("dataset/")

    if missing:
        missing_str = ", ".join(missing)
        raise FileNotFoundError(
            f"Input bundle '{bundle_path}' is missing required subdirectories: [{missing_str}].\n"
            f"An input bundle must contain:\n"
            f"  - {bundle_path}/teacher/\n"
            f"  - {bundle_path}/student/\n"
            f"  - {bundle_path}/dataset/"
        )

    # Scan for weight files (.pt, .pth, .bin)
    def find_checkpoints(directory: str) -> List[str]:
        ckpts = []
        for root, _, files in os.walk(directory):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in [".pt", ".pth", ".bin"]:
                    ckpts.append(os.path.join(root, f))
        return sorted(ckpts)

    def find_python_scripts(directory: str) -> List[str]:
        scripts = []
        for f in os.listdir(directory):
            if f.endswith(".py"):
                scripts.append(os.path.join(directory, f))
        return sorted(scripts)

    teacher_ckpts = find_checkpoints(teacher_dir)
    student_ckpts = find_checkpoints(student_dir)
    teacher_scripts = find_python_scripts(teacher_dir)
    student_scripts = find_python_scripts(student_dir)

    return {
        "bundle_name": bundle_name,
        "bundle_path": bundle_path,
        "teacher_dir": teacher_dir,
        "student_dir": student_dir,
        "dataset_dir": dataset_dir,
        "teacher_checkpoints": teacher_ckpts,
        "student_checkpoints": student_ckpts,
        "teacher_scripts": teacher_scripts,
        "student_scripts": student_scripts,
    }
