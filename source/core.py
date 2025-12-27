# source/core.py
from __future__ import annotations
import torch
import torch.nn.functional as F

def probs_from_logits(Z: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    """
    Convert unconstrained logits Z into relaxed assignments P via softmax.

    Z: (..., K) logits
    T: temperature (higher -> softer, lower -> sharper)
    returns: same shape as Z, normalized over last dimension
    """
    return F.softmax(Z / T, dim=-1)

def pretty_grid_from_probs(P: torch.Tensor) -> torch.Tensor:
    """
    Project relaxed assignments to a discrete grid via argmax.
    For 4x4: returns digits 1..4 (hence +1).
    For 9x9: returns digits 1..9 (also +1).
    """
    return P.argmax(dim=-1) + 1

def print_losses(tag: str, d: dict[str, torch.Tensor]) -> None:
    """
    Print a dict of scalar tensors as floats for readability.
    """
    fmt = {k: f"{v.detach().cpu().item():.3f}" for k, v in d.items()}
    print(tag, fmt, "\n")
