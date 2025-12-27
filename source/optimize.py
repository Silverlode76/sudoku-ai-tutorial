# source/optimize.py
from __future__ import annotations
import torch
from .core import probs_from_logits
from .losses import sudoku_losses

def loss_dict(
    P: torch.Tensor,
    givens_mask: torch.Tensor,
    givens_target: torch.Tensor,
    weights: tuple[float, float, float, float, float],
    block_size: int,
) -> dict[str, torch.Tensor]:
    w_row, w_col, w_blk, w_giv, w_ent = weights
    L_row, L_col, L_blk, L_giv, L_ent = sudoku_losses(P, givens_mask, givens_target, block_size=block_size)
    L_total = w_row*L_row + w_col*L_col + w_blk*L_blk + w_giv*L_giv + w_ent*L_ent
    return {
        "L_row": L_row, "L_col": L_col, "L_blk": L_blk,
        "L_giv": L_giv, "L_ent": L_ent, "L_total": L_total
    }

def optimize_sudoku(
    Z_init: torch.Tensor,
    givens_mask: torch.Tensor,
    givens_target: torch.Tensor,
    *,
    steps: int = 300,
    lr: float = 0.2,
    T_start: float = 1.5,
    T_end: float = 0.6,
    block_size: int = 2,
    w_row: float = 1.0,
    w_col: float = 1.0,
    w_blk: float = 1.2,
    w_giv: float = 2.0,
    w_ent: float = 0.01,
    save_P_every: int | None = None,
) -> tuple[torch.Tensor, dict]:
    """
    Optimize logits Z to minimize Sudoku constraint losses in a relaxed space.

    Returns:
      Z_final (detached),
      hist dict with losses, temperature, and optional P snapshots.
    """
    Z = Z_init.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([Z], lr=lr)

    hist = {
        "L_total": [], "L_row": [], "L_col": [], "L_blk": [],
        "L_giv": [], "L_ent": [], "T": []
    }
    if save_P_every is not None:
        hist["P_snapshots"] = {}

    weights = (w_row, w_col, w_blk, w_giv, w_ent)

    for t in range(steps):
        alpha = t / max(1, steps - 1)
        T = (1 - alpha) * T_start + alpha * T_end

        P = probs_from_logits(Z, T=T)
        d = loss_dict(P, givens_mask, givens_target, weights, block_size=block_size)
        L_total = d["L_total"]

        opt.zero_grad()
        L_total.backward()
        opt.step()

        # log
        hist["L_total"].append(float(d["L_total"].detach().cpu()))
        hist["L_row"].append(float(d["L_row"].detach().cpu()))
        hist["L_col"].append(float(d["L_col"].detach().cpu()))
        hist["L_blk"].append(float(d["L_blk"].detach().cpu()))
        hist["L_giv"].append(float(d["L_giv"].detach().cpu()))
        hist["L_ent"].append(float(d["L_ent"].detach().cpu()))
        hist["T"].append(float(T))

        if save_P_every is not None and (t % save_P_every == 0 or t == steps - 1):
            hist["P_snapshots"][t] = P.detach().cpu()

    return Z.detach(), hist
