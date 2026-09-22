"""Data mining helpers: a compact Apriori, K-Means with recorded iterations,
and an Isolation-tree trace for animation."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def apriori(transactions: list[set[str]], min_support: float = 0.05, max_len: int = 3) -> pd.DataFrame:
    """Frequent itemsets by level-wise Apriori (downward closure pruning)."""
    n = len(transactions)
    counts: dict[frozenset, int] = {}
    for t in transactions:
        for item in t:
            k = frozenset([item])
            counts[k] = counts.get(k, 0) + 1
    current = {k for k, v in counts.items() if v / n >= min_support}
    frequent = {k: counts[k] / n for k in current}
    size = 1
    while current and size < max_len:
        size += 1
        items = sorted({i for s in current for i in s})
        candidates = set()
        for combo in combinations(items, size):
            fs = frozenset(combo)
            if all(frozenset(sub) in current for sub in combinations(combo, size - 1)):
                candidates.add(fs)
        cand_counts = {c: 0 for c in candidates}
        for t in transactions:
            for c in candidates:
                if c <= t:
                    cand_counts[c] += 1
        current = {c for c, v in cand_counts.items() if v / n >= min_support}
        frequent.update({c: cand_counts[c] / n for c in current})
    rows = [{"itemset": ", ".join(sorted(k)), "size": len(k), "support": v} for k, v in frequent.items()]
    out = pd.DataFrame(rows).sort_values(["size", "support"], ascending=[True, False]) if rows else pd.DataFrame(
        columns=["itemset", "size", "support"])
    out.attrs["frequent"] = frequent
    return out


def association_rules(freq: pd.DataFrame, min_confidence: float = 0.3) -> pd.DataFrame:
    frequent: dict[frozenset, float] = freq.attrs.get("frequent", {})
    rows = []
    for itemset, sup in frequent.items():
        if len(itemset) < 2:
            continue
        for r in range(1, len(itemset)):
            for ante in combinations(itemset, r):
                a = frozenset(ante)
                c = itemset - a
                if a not in frequent or c not in frequent:
                    continue
                conf = sup / frequent[a]
                if conf < min_confidence:
                    continue
                lift = conf / frequent[c]
                leverage = sup - frequent[a] * frequent[c]
                conviction = np.inf if conf == 1 else (1 - frequent[c]) / (1 - conf)
                rows.append({"antecedent": ", ".join(sorted(a)), "consequent": ", ".join(sorted(c)),
                             "support": sup, "confidence": conf, "lift": lift, "leverage": leverage,
                             "conviction": conviction})
    if not rows:
        return pd.DataFrame(columns=["antecedent", "consequent", "support", "confidence", "lift", "leverage",
                                     "conviction"])
    return pd.DataFrame(rows).sort_values("lift", ascending=False).reset_index(drop=True)


def kmeans_trace(X: np.ndarray, k: int, seed: int = 0, max_iter: int = 15) -> list[dict]:
    """Lloyd's algorithm, recording centroids and assignments at every step."""
    rng = np.random.default_rng(seed)
    centroids = X[rng.choice(len(X), k, replace=False)].copy()
    history = []
    for it in range(max_iter):
        d = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = d.argmin(axis=1)
        inertia = float(d[np.arange(len(X)), labels].sum())
        history.append({"iter": it, "phase": "assign", "centroids": centroids.copy(), "labels": labels.copy(),
                        "inertia": inertia})
        new = np.array([X[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j] for j in range(k)])
        moved = float(np.abs(new - centroids).max())
        centroids = new
        history.append({"iter": it, "phase": "update", "centroids": centroids.copy(), "labels": labels.copy(),
                        "inertia": inertia, "moved": moved})
        if moved < 1e-6:
            break
    return history


def isolation_trace(X: np.ndarray, target: int, seed: int = 0, max_depth: int = 30) -> list[dict]:
    """Random axis-aligned splits until ``target`` point is isolated (one tree)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    lo = X.min(axis=0) - 0.5
    hi = X.max(axis=0) + 0.5
    box = [lo.copy(), hi.copy()]
    steps = []
    for depth in range(max_depth):
        if len(idx) <= 1:
            break
        feat = int(rng.integers(0, X.shape[1]))
        vals = X[idx, feat]
        if vals.min() == vals.max():
            break
        split = float(rng.uniform(vals.min(), vals.max()))
        go_left = X[target, feat] < split
        keep = idx[(X[idx, feat] < split) == go_left]
        if go_left:
            box[1][feat] = split
        else:
            box[0][feat] = split
        steps.append({"depth": depth + 1, "feature": feat, "split": split, "remaining": len(keep),
                      "box": (box[0].copy(), box[1].copy())})
        idx = keep
    return steps
