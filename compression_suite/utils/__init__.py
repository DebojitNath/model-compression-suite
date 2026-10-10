"""
Utilities package for Model Compression Suite.
"""
from compression_suite.utils.workspace import (
    resolve_input_bundle,
    resolve_output_dir,
    get_next_available_output_dir,
)

__all__ = [
    "resolve_input_bundle",
    "resolve_output_dir",
    "get_next_available_output_dir",
]
