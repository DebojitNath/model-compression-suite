"""
Lottery Ticket Hypothesis (LTH) Weight Rewinder.
Stores early/initialization state and rewinds surviving sparse weights.
"""
from __future__ import annotations
from typing import Dict
import copy
import torch
import torch.nn as nn
from compression_suite.pruning.masks import MaskManager


class WeightRewinder:
    """
    Manages reference weight snapshots (W_0 or W_k) and rewinds unpruned weights
    under active binary masks as postulated in Frankle & Carbin (2018).
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.reference_state: Dict[str, torch.Tensor] = {}
        self.save_reference()

    def save_reference(self):
        """Saves a deep copy of the current model state_dict as the rewinding reference."""
        self.reference_state = {
            k: v.clone().detach().cpu()
            for k, v in self.model.state_dict().items()
        }

    def rewind(self, mask_manager: MaskManager):
        """
        Rewinds model parameters back to the saved reference state,
        then reapplies active binary masks so that only pruned weights remain zeroed.
        """
        device = next(self.model.parameters()).device
        model_state = self.model.state_dict()

        for k, v in self.reference_state.items():
            if k in model_state:
                model_state[k].copy_(v.to(device))

        # Re-enforce binary masks on the rewound weights
        for name, p_mask in mask_manager.masks.items():
            param = getattr(p_mask.module, p_mask.param_name)
            param.data.mul_(p_mask.mask)
