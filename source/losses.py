# source/losses.py
from __future__ import annotations
import torch

def entropy_loss(P: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Sum of entropies over all cells.
    P: (N,N,K) probabilities over digits (K=N)
    """
    ent = -(P * (P + eps).log()).sum(dim=-1)  # (N,N)
    return ent.sum()

def entropy_mean(P: torch.Tensor, eps: float = 1e-9) -> float:

    ent = -(P * (P + 1e-9).log()).sum(dim=2)   # (9,9)
    return ent.mean().item()

def sudoku_losses(
    P: torch.Tensor,
    givens_mask: torch.Tensor,
    givens_target: torch.Tensor,
    block_size: int,
    eps: float = 1e-9,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute unweighted constraint losses for Sudoku.

    P: (N, N, K) relaxed assignments, K should be N
    givens_mask: (N, N) bool, True where a clue is given
    givens_target: (N, N) long indices in [0..K-1] for given digits
    block_size: 2 for 4x4, 3 for 9x9

    Returns: (L_row, L_col, L_blk, L_giv, L_ent)
    """
    N, _, K = P.shape
    assert K == N, "Expected K == N (digits match grid size)."
    assert N % block_size == 0, "Grid size must be divisible by block_size."

    # Row uniqueness: for each row i and digit k: sum_j P[i,j,k] = 1
    row_sum = P.sum(dim=1)  # (N, K)
    L_row = ((row_sum - 1.0) ** 2).sum()

    # Column uniqueness: for each col j and digit k: sum_i P[i,j,k] = 1
    col_sum = P.sum(dim=0)  # (N, K)
    L_col = ((col_sum - 1.0) ** 2).sum()

    # Block uniqueness: for each block b and digit k: sum_{cells in block} P = 1
    L_blk = torch.tensor(0.0, device=P.device, dtype=P.dtype)
    for r0 in range(0, N, block_size):
        for c0 in range(0, N, block_size):
            blk = P[r0:r0+block_size, c0:c0+block_size, :]  # (B,B,K)
            blk_sum = blk.sum(dim=(0, 1))                   # (K,)
            L_blk = L_blk + ((blk_sum - 1.0) ** 2).sum()

    # Given constraint: given cells should match givens_target
    # For each given cell, encourage P[i,j,target] = 1
    if givens_mask.any():
        P_giv = P[givens_mask]                 # (G, K)
        tgt = givens_target[givens_mask]       # (G,)
        p_correct = P_giv[torch.arange(P_giv.shape[0], device=P.device), tgt]
        L_giv = ((p_correct - 1.0) ** 2).sum()
    else:
        L_giv = torch.tensor(0.0, device=P.device, dtype=P.dtype)

    # Entropy regularization
    L_ent = entropy_loss(P, eps=eps)

    return L_row, L_col, L_blk, L_giv, L_ent
