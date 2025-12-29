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

def check_grid(Z: torch.Tensor)-> None:
    grid_hat = pretty_grid_from_probs(probs_from_logits(Z, T=0.5))  # oder T_end
    print(grid_hat)
    
    # Check rows/cols contain 1..9 exactly once
    ok_rows = all(sorted(row.tolist()) == list(range(1,10)) for row in grid_hat)
    ok_cols = all(sorted(grid_hat[:,j].tolist()) == list(range(1,10)) for j in range(9))
    
    # Check 3x3 blocks
    ok_blks = True
    for r0 in range(0, 9, 3):
        for c0 in range(0, 9, 3):
            blk = grid_hat[r0:r0+3, c0:c0+3].reshape(-1).tolist()
            if sorted(blk) != list(range(1,10)):
                ok_blks = False
    
    print("rows ok:", ok_rows, "cols ok:", ok_cols, "blocks ok:", ok_blks)
