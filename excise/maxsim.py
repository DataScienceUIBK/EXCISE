"""The late-interaction score: each query token's best match over the document tokens, summed."""
from __future__ import annotations
import torch


def weighted_maxsim_matrix(q_vec, q_w, d_vec, d_mask):
    """ALL-PAIRS weighted MaxSim.  q_vec:(Bq,Lq,P) q_w:(Bq,Lq) d_vec:(Bd,Td,P) d_mask:(Bd,Td) -> (Bq,Bd)."""
    sims = torch.einsum("ilp,jtp->ijlt", q_vec, d_vec)          # (Bq,Bd,Lq,Td)
    sims = sims.masked_fill(~d_mask[None, :, None, :], float("-inf"))
    per_q = sims.max(dim=-1).values                             # (Bq,Bd,Lq)
    return (per_q * q_w[:, None, :]).sum(dim=-1)                 # (Bq,Bd)
