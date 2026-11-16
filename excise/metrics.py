"""The metrics: success@10, right-rank, nDCG@10, xnDCG, graded nDCG and recall."""
from __future__ import annotations
import math

_ABSENT = 10 ** 9


def success10(order_ids, gold_set, excl_set):
    pos = {int(d): r + 1 for r, d in enumerate(order_ids)}
    G, E = set(gold_set), set(excl_set)
    recall = len([g for g in G if pos.get(g, 999) <= 10]) / max(len(G), 1)
    excl = sum(1 for x in E if pos.get(x, 999) <= 10) / max(len(E), 1)
    succ = int(any(pos.get(g, 999) <= 10 for g in G) and not any(pos.get(x, 999) <= 10 for x in E))
    return recall, excl, succ


def right_rank(order_ids, pos_id, neg_ids):
    """1 iff the positive is retrieved AND outranks every excluded negative."""
    rank = {int(d): r for r, d in enumerate(order_ids)}
    rp = rank.get(int(pos_id))
    if rp is None: return 0                                        # positive not retrieved at all
    return int(all(rank.get(int(n), _ABSENT) > rp for n in neg_ids))


def ndcg10(order_ids, gold_set):
    G = set(int(g) for g in gold_set)
    gains = [1.0 if int(d) in G else 0.0 for d in order_ids[:10]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(G), 10)))
    return dcg / ideal if ideal > 0 else 0.0


def ndcg10_excl(order_ids, gold_set, excl_set, k=10, beta=1.0):
    """Exclusion-aware nDCG@k: ONE ranking number that credits the positive's placement AND penalises"""
    G = set(int(g) for g in gold_set)
    E = set(int(x) for x in excl_set)
    def rel(d):
        d = int(d)
        return 1.0 if d in G else (-beta if d in E else 0.0)
    dcg = sum(rel(d) / math.log2(i + 2) for i, d in enumerate(order_ids[:k]))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(G), k)))   # golds at top, no negatives
    return dcg / idcg if idcg > 0 else 0.0


def ndcg_at_k_graded(order_ids, rel, k=10):
    """Standard graded nDCG@k. `rel` maps corpus index -> relevance grade (BEIR qrels are graded 0/1/2)."""
    def gain(g): return (2 ** g - 1) if g > 0 else 0.0
    dcg = sum(gain(rel.get(int(d), 0)) / math.log2(i + 2) for i, d in enumerate(order_ids[:k]))
    ideal = sorted((g for g in rel.values() if g > 0), reverse=True)[:k]
    idcg = sum(gain(g) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(order_ids, rel, k=100):
    """Fraction of relevant (rel>0) docs present in the top-k."""
    R = {d for d, g in rel.items() if g > 0}
    if not R: return 0.0
    top = {int(d) for d in order_ids[:k]}
    return len(R & top) / len(R)
