"""The adapter's loss: PyLate's contrastive loss plus a margin on MaxSim(Z, d), the quantity the operator
demotes by, normalised by the number of Z tokens as the operator normalises it."""
from __future__ import annotations
import torch
from torch import nn


def _zsim(qv, qm, dv, dm):
    """Per-pair (diagonal) normalised MaxSim: for each row, its OWN document, not all pairs."""
    sims = torch.einsum("blp,btp->blt", qv, dv)                  # (B,Lq,Td)
    sims = sims.masked_fill(~dm.bool().unsqueeze(1), float("-inf"))
    per_q = sims.max(dim=-1).values                              # (B,Lq) best doc token per Z token
    qmf = qm.float()
    nz = qmf.sum(dim=-1).clamp(min=1.0)
    return (per_q * qmf).sum(dim=-1) / nz


class ContrastiveWithZMargin(nn.Module):
    """`base` (pylate losses.Contrastive) unchanged, plus the Z margin."""

    def __init__(self, model, base, margin: float = 0.1, weight: float = 1.0, mask: bool = True):
        super().__init__()
        self.model = model
        self.base = base
        self.margin = float(margin)
        self.weight = float(weight)
        self.mask = bool(mask)

    def forward(self, sentence_features, labels=None):
        feats = list(sentence_features)
        if len(feats) < 4:                       # no z_query column -> plain contrastive, unchanged
            return self.base(feats, labels)
        q, pos, neg, zq = feats[0], feats[1], feats[2], feats[3]
        loss = self.base([q, pos, neg], labels)  # the untouched objective

        def emb(f):
            return torch.nn.functional.normalize(
                self.model(f)["token_embeddings"], p=2, dim=-1)
        ez, ep, en = emb(zq), emb(pos), emb(neg)
        sp = _zsim(ez, zq["attention_mask"], ep, pos["attention_mask"])
        sn = _zsim(ez, zq["attention_mask"], en, neg["attention_mask"])
        # The NEGATIVE is the document about Z, so it must score HIGHER on this channel than the gold.
        hinge = torch.relu(self.margin - (sn - sp))
        if labels is not None and self.mask:
            w = labels.to(hinge.dtype).to(hinge.device).view(-1)
            lz = (hinge * w).sum() / w.sum().clamp(min=1.0)
        else:
            lz = hinge.mean()
        return loss + self.weight * lz

    def get_config_dict(self):
        return {"margin": self.margin, "weight": self.weight}
