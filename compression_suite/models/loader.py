"""
Universal Model Loader: Loads arbitrary PyTorch models from checkpoints (.pt/.pth),
state dictionaries, dynamic Python modules, or the built-in registry.
"""
from __future__ import annotations
from typing import Callable, Optional, Tuple, Union
import os
import sys
import copy
import importlib.util
import torch
import torch.nn as nn

from compression_suite.models.registry import build_model, list_available_models


def load_model_from_python_file(file_path: str, num_classes: int = 10, **kwargs) -> nn.Module:
    """
    Dynamically loads an nn.Module from a standalone Python file.
    Looks for build_model(), get_model(), or the first nn.Module class.
    """
    spec = importlib.util.spec_from_file_location("custom_model_module", file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Python module from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["custom_model_module"] = mod
    spec.loader.exec_module(mod)

    if hasattr(mod, "build_model") and callable(mod.build_model):
        return mod.build_model(num_classes=num_classes, **kwargs)
    if hasattr(mod, "get_model") and callable(mod.get_model):
        return mod.get_model(num_classes=num_classes, **kwargs)

    # Search for nn.Module class definitions in the file
    for attr_name in dir(mod):
        attr = getattr(mod, attr_name)
        if isinstance(attr, type) and issubclass(attr, nn.Module) and attr is not nn.Module:
            try:
                return attr(num_classes=num_classes)
            except Exception:
                try:
                    return attr()
                except Exception:
                    pass

    raise ValueError(f"Could not find an instantiable model or build_model() in {file_path}")


def load_model_from_checkpoint(
    checkpoint_path: str,
    arch_name: Optional[str] = None,
    num_classes: int = 10,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """
    Loads a PyTorch model from a .pt / .pth / .bin checkpoint file.
    Handles both full serialized models (torch.save(model)) and
    state dictionaries (torch.save(model.state_dict())).
    """
    device = device or torch.device("cpu")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Load object with weights_only=False if supported to allow full module deserialization
    try:
        loaded = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        loaded = torch.load(checkpoint_path, map_location=device)

    # 1. Full serialized nn.Module
    if isinstance(loaded, nn.Module):
        model = loaded
        return model.to(device)

    # 2. State dictionary
    state_dict = None
    if isinstance(loaded, dict):
        if "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
            state_dict = loaded["state_dict"]
        elif "model_state_dict" in loaded and isinstance(loaded["model_state_dict"], dict):
            state_dict = loaded["model_state_dict"]
        elif "model" in loaded and isinstance(loaded["model"], dict):
            state_dict = loaded["model"]
        else:
            # Assume dictionary itself is state_dict
            state_dict = loaded

    if state_dict is not None:
        if not arch_name:
            raise ValueError(
                f"Checkpoint at '{checkpoint_path}' contains a state dictionary, not a full model.\n"
                f"You must specify the base architecture (e.g. --teacher resnet50 or an arch.txt) "
                f"to instantiate the model structure before loading weights.\n"
                f"Available registered models: [{', '.join(list_available_models())}]"
            )
        model = build_model(arch_name, num_classes=num_classes)
        # Strip common prefixes like 'module.' from DataParallel if present
        clean_state_dict = {}
        for k, v in state_dict.items():
            clean_k = k[7:] if k.startswith("module.") else k
            clean_state_dict[clean_k] = v
        model.load_state_dict(clean_state_dict, strict=False)
        return model.to(device)

    raise ValueError(f"Unsupported checkpoint object type: {type(loaded)} in {checkpoint_path}")


def resolve_model_from_directory(
    directory: str,
    role: str = "model",
    default_arch: Optional[str] = None,
    num_classes: int = 10,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Callable[[], nn.Module]]:
    """
    Discovers and loads a model from an experiment subfolder (e.g., bundle/teacher or bundle/student).
    Returns (model_instance, factory_function).
    """
    device = device or torch.device("cpu")
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found for {role}: {directory}")

    # Check for architecture hint in arch.txt
    arch_file = os.path.join(directory, "arch.txt")
    arch_hint = default_arch
    if os.path.isfile(arch_file):
        with open(arch_file, "r", encoding="utf-8") as f:
            line = f.read().strip()
            if line:
                arch_hint = line

    # Look for .pt/.pth/.bin checkpoint files
    ckpts = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if os.path.splitext(f)[1].lower() in [".pt", ".pth", ".bin"]
    ]

    # Look for .py model definition
    scripts = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".py")
    ]

    model: Optional[nn.Module] = None

    if ckpts:
        ckpt_path = ckpts[0]
        model = load_model_from_checkpoint(
            checkpoint_path=ckpt_path,
            arch_name=arch_hint,
            num_classes=num_classes,
            device=device,
        )
    elif scripts:
        script_path = scripts[0]
        model = load_model_from_python_file(
            file_path=script_path,
            num_classes=num_classes,
        ).to(device)
    elif arch_hint:
        model = build_model(arch_hint, num_classes=num_classes).to(device)
    else:
        raise FileNotFoundError(
            f"Could not resolve {role} from directory '{directory}'.\n"
            f"Expected either:\n"
            f"  - A model checkpoint file (.pt or .pth)\n"
            f"  - A Python model file (model.py)\n"
            f"  - An arch.txt specifying a registered architecture (e.g. 'resnet18')\n"
        )

    # Build a clean factory callable returning fresh copies
    model_template = copy.deepcopy(model)
    factory = lambda: copy.deepcopy(model_template)

    return model, factory
