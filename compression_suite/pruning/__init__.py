"""
Pruning module init.
"""
from compression_suite.pruning.masks import ParameterMask, MaskManager
from compression_suite.pruning.rewinding import WeightRewinder
from compression_suite.pruning.pruner import IterativeMagnitudePruner

__all__ = [
    "ParameterMask",
    "MaskManager",
    "WeightRewinder",
    "IterativeMagnitudePruner",
]
