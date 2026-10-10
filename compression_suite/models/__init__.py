"""
Models module init.
Exposes base class, registry functions, and imports vision/text architectures.
"""
from compression_suite.models.base import BaseCompressibleModel
from compression_suite.models.registry import register_model, build_model, list_available_models
import compression_suite.models.vision_models
import compression_suite.models.text_models

__all__ = [
    "BaseCompressibleModel",
    "register_model",
    "build_model",
    "list_available_models",
]
