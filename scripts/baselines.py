#!/usr/bin/env python3
"""
Baseline comparison for the Jarir recommender.

Measures ItemKNN's uplift over trivial and non-trivial baselines through the
same evaluation path as scripts/evaluate.py: the same sequences_<split>.parquet
queries, the same recall_at_k / ndcg_at_k / coverage_at_k from
src/evaluation/metrics.py, and the same ranked set (all 1,735 items). Nothing
here trains, tunes or rewrites the split.

Models (every item-item model is refit per query date on purchases strictly
before it, and all three share the same recency-weighted query profile, so the
only thing that differs between them is the item-item matrix):

  random        10 items drawn uniformly from the catalogue, fixed seed.
  random_unseen as above but drawn from items the customer has not bought.
  popularity    the 10 items with the most purchase lines dated strictly before
                the split's first timestamp, the same list for every query.
                This is row A of scripts/evaluate.py.
  popularity_t  the same count refit per query date, so it sees exactly as much
                history as the item-item models do.
  item_knn      cosine over the binarised purchase matrix. Row A3 of
                scripts/evaluate.py.
  item_knn_bm25 the same, with BM25 term weighting (K1=1.2, B=0.75) applied
                before the cosine.
  ease          Steck's closed-form item-item model, l2=250. Both defaults are
                the literature's, untuned on this data.

Seen items are not filtered by default, because the served retriever does not
filter them either (src/inference/item_knn.py keeps self-similarity on purpose,
and ~9% of test positives are repeat purchases). --grid reports every model
both ways.

Sections (all run by default, pick with --sections):
  main     one table per split, uplift over MostPopular, history-length
           breakdown, leakage diagnostics.
  grid     Recall/NDCG under {same-day history kept, dropped} x {seen items
           kept, filtered}.
  rolling  the same protocol over consecutive temporal windows, pooled, with
           bootstrap CIs on both the ratio and the difference.

Usage:
    python scripts/baselines.py
    python scripts/baselines.py --sections main --splits val test
    python scripts/baselines.py --sections rolling --windows 4
"""

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from src.data.dataset import load_id_mappings, load_processed_sequences, parse_history_string
from src.data.features import SequenceBuilder
from src.evaluation.metrics import coverage_at_k, ndcg_at_k, recall_at_k
from src.inference.item_knn import ItemKNNRetriever
from src.utils.io import read_parquet
from src.utils.seed import set_seed

MAX_HIST = 15
HIST_BUCKETS = [("1", 1, 1), ("2", 2, 2), ("3-5", 3, 5), ("6+", 6, 10 ** 9)]
ITEM_ITEM = ("item_knn", "item_knn_bm25", "ease")


# --------------------------------------------------------------------------- #
# Item-item models. Each mirrors ItemKNNRetriever's fit/score interface so the
# time-aware refit loop below can drive any of them.
# --------------------------------------------------------------------------- #
def purchase_matrix(interactions: pd.DataFrame, n_items: int) -> csr_matrix:
    """Binarised customer x item matrix; repeat purchases count once."""
    users = interactions["customer_id"].astype("category").cat.codes.to_numpy()
    items = interactions["item_idx"].astype(int).to_numpy()
    mat = csr_matrix(
        (np.ones(len(items), dtype=np.float32), (users, items)),
        shape=(int(users.max()) + 1, n_items),
    )
    mat.data[:] = 1.0
    return mat


def bm25_weight(mat: csr_matrix, k1: float = 1.2, b: float = 0.75) -> csr_matrix:
    """
    BM25 term weighting over a customer x item matrix: customers are documents,
    items are terms. Damps customers with long baskets and rare-item inflation,
    which plain cosine does not.
    """
    n_users = mat.shape[0]
    df = np.bincount(mat.indices, minlength=mat.shape[1]).astype(np.float32)
    idf = np.log(1.0 + (n_users - df + 0.5) / (df + 0.5))
    lengths = np.asarray(mat.sum(axis=1)).ravel().astype(np.float32)
    avgdl = float(lengths.mean()) if lengths.size else 1.0
    coo = mat.tocoo()
    denom = coo.data + k1 * (1.0 - b + b * lengths[coo.row] / max(avgdl, 1e-8))
    coo.data = coo.data * (k1 + 1.0) / denom * idf[coo.col]
    return coo.tocsr()


class BM25ItemKNN(ItemKNNRetriever):
    """ItemKNN with BM25 weighting applied before the cosine."""

    def __init__(self, n_items: int, k1: float = 1.2, b: float = 0.75, **kwargs):
        super().__init__(n_items, **kwargs)
        self.k1, self.b = k1, b

    def fit(self, interactions: pd.DataFrame) -> "BM25ItemKNN":
        if len(interactions) == 0:
            self.similarity = np.zeros((self.n_items, self.n_items), dtype=np.float32)
            return self
        mat = bm25_weight(purchase_matrix(interactions, self.n_items), self.k1, self.b)
        self.similarity = cosine_similarity(mat.T, dense_output=True).astype(np.float32)
        return self


class EASE(ItemKNNRetriever):
    """
    Steck (2019), "Embarrassingly Shallow Autoencoders". Closed form:
    P = (X'X + lI)^-1, B = I - P diag(1 / diag(P)), diagonal zeroed.

    Zeroing the diagonal is constitutive of the model, so unlike ItemKNN this
    one cannot re-recommend an item purely on self-similarity. It still can via
    a co-purchased neighbour.
    """

    def __init__(self, n_items: int, l2: float = 250.0, **kwargs):
        super().__init__(n_items, **kwargs)
        self.l2 = l2

    def fit(self, interactions: pd.DataFrame) -> "EASE":
        if len(interactions) == 0:
            self.similarity = np.zeros((self.n_items, self.n_items), dtype=np.float32)
            return self
        X = purchase_matrix(interactions, self.n_items)
        G = np.asarray((X.T @ X).todense(), dtype=np.float64)
        G[np.diag_indices_from(G)] += self.l2
        P = np.linalg.inv(G)
        B = -P / np.diag(P)
        B[np.diag_indices_from(B)] = 0.0
        self.similarity = B.astype(np.float32)
        return self


MODELS: Dict[str, Callable[[int], ItemKNNRetriever]] = {
    "item_knn": lambda n: ItemKNNRetriever(n),
    "item_knn_bm25": lambda n: BM25ItemKNN(n),
    "ease": lambda n: EASE(n),
}


def time_aware_scores(
    factory: Callable[[int], ItemKNNRetriever],
    interactions: pd.DataFrame,
    timestamps: Sequence,
    histories: List[List[int]],
    n_items: int,
) -> np.ndarray:
    """
    Refit per distinct query date on purchases strictly before it and score
    every item. Returns (n_queries, n_items). Same leak-free contract as
    src.inference.item_knn.time_aware_topk, but keeps the scores so that
    several rankings (filtered and not) can be derived from one fit.
    """
    ts = pd.Series(pd.to_datetime(list(timestamps))).reset_index(drop=True)
    out = np.zeros((len(histories), n_items), dtype=np.float32)
    dates = interactions["invoice_date"]
    for t, pos in ts.groupby(ts).indices.items():
        model = factory(n_items).fit(interactions[dates < t])
        out[pos] = model.score([histories[i] for i in pos])
    return out


# --------------------------------------------------------------------------- #
# Ranking. One tie-break rule for every model: score desc, then item_idx asc.
# --------------------------------------------------------------------------- #
def rank(scores: np.ndarray, k: int, exclude: Optional[List[set]] = None) -> List[List[int]]:
    order = np.argsort(-scores, axis=1, kind="stable")
    if exclude is None:
        return order[:, :k].tolist()
    return [[int(j) for j in row if j not in ex][:k] for row, ex in zip(order, exclude)]


def popularity_scores(past: pd.DataFrame, n_items: int, n_queries: int) -> np.ndarray:
    """Purchase-line count per item, broadcast to every query."""
    counts = np.zeros(n_items, dtype=np.float32)
    if len(past):
        vc = past["item_idx"].astype(int).value_counts()
        counts[vc.index.to_numpy()] = vc.to_numpy()
    return np.tile(counts, (n_queries, 1))


def popularity_scores_time_aware(
    interactions: pd.DataFrame, timestamps: Sequence, n_items: int
) -> np.ndarray:
    ts = pd.Series(pd.to_datetime(list(timestamps))).reset_index(drop=True)
    out = np.zeros((len(ts), n_items), dtype=np.float32)
    dates = interactions["invoice_date"]
    for t, pos in ts.groupby(ts).indices.items():
        out[pos] = popularity_scores(interactions[dates < t], n_items, 1)[0]
    return out


def random_preds(
    n_items: int, histories: List[List[int]], k: int, seed: int, exclude_seen: bool = False
) -> List[List[int]]:
    """k distinct items per query, drawn uniformly without replacement."""
    rng = np.random.default_rng(seed)
    out = []
    for h in histories:
        pool = np.arange(n_items)
        if exclude_seen:
            pool = np.setdiff1d(pool, np.unique(h))
        out.append(rng.choice(pool, size=min(k, len(pool)), replace=False).tolist())
    return out


# --------------------------------------------------------------------------- #
# Split helpers
# --------------------------------------------------------------------------- #
def histories_of(df: pd.DataFrame) -> List[List[int]]:
    return [parse_history_string(str(h))[-MAX_HIST:] for h in df["history_idx"]]


def same_day_items(interactions: pd.DataFrame) -> pd.Series:
    """item_idx set per (customer_id, invoice_date)."""
    return interactions.groupby(["customer_id", "invoice_date"])["item_idx"].apply(
        lambda s: set(s.astype(int))
    )


def same_day_mask(
    df: pd.DataFrame, histories: List[List[int]], gt: List[int], by_cust_day: pd.Series
) -> np.ndarray:
    """
    True where the query history holds an item the customer bought on the
    target's own invoice_date. Dates carry no time of day, so SequenceBuilder
    orders a single day's lines arbitrarily and part of the current basket ends
    up in the history.
    """
    out = np.zeros(len(df), dtype=bool)
    for i, (cust, ts, h, g) in enumerate(zip(df["customer_id"], df["ts"], histories, gt)):
        out[i] = bool((by_cust_day.get((cust, ts), set()) & set(h)) - {g})
    return out


def strip_same_day(
    df: pd.DataFrame, histories: List[List[int]], by_cust_day: pd.Series
) -> List[List[int]]:
    """Drop history items the customer also bought on the target's own date."""
    return [
        [i for i in h if i not in by_cust_day.get((cust, ts), set())]
        for cust, ts, h in zip(df["customer_id"], df["ts"], histories)
    ]


def build_window(
    interactions: pd.DataFrame,
    item_map: pd.DataFrame,
    customer_map: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Queries whose target falls in [start, end), built by the repo's own
    SequenceBuilder with the configs/data.yaml settings. Reproduces
    sequences_test.parquet exactly when handed the 0.90 quantile and max+1s.
    """
    sb = SequenceBuilder(max_history_length=MAX_HIST, min_history_length=2, random_seed=42)
    raw = interactions.drop(columns=["item_idx", "user_idx"], errors="ignore")
    merged = (
        sb._add_id_mappings(raw, item_map, customer_map)
        .sort_values(["customer_id", "invoice_date"])
        .reset_index(drop=True)
    )
    return sb._build_sequences_for_split(merged, start, end, "window").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Metrics and uncertainty
# --------------------------------------------------------------------------- #
def hit_vectors(preds: List[List[int]], gt: List[int], k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Per-query Recall@k and NDCG@k contributions, matching src.evaluation.metrics."""
    rec = np.array([float(g in p[:k]) for p, g in zip(preds, gt)])
    ndcg = np.array(
        [1.0 / np.log2(p[:k].index(g) + 2) if g in p[:k] else 0.0 for p, g in zip(preds, gt)]
    )
    return rec, ndcg


def bootstrap(
    a: np.ndarray, b: np.ndarray, users: np.ndarray, n_resamples: int, seed: int
) -> Dict[str, float]:
    """
    Customer-clustered percentile bootstrap for mean(a)/mean(b) and
    mean(a)-mean(b). Customers are drawn with replacement and bring all of
    their queries, which is the unit the CI should be over. `dropped` counts
    resamples where the denominator came out zero and the ratio was undefined;
    the difference is defined for every resample.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(users)
    by_user = {u: np.flatnonzero(users == u) for u in uniq}
    ratios, diffs, dropped = [], [], 0
    for _ in range(n_resamples):
        idx = np.concatenate([by_user[u] for u in rng.choice(uniq, size=len(uniq), replace=True)])
        ma, mb = a[idx].mean(), b[idx].mean()
        diffs.append(ma - mb)
        if mb == 0:
            dropped += 1
        else:
            ratios.append(ma / mb)
    pct = lambda v, q: float(np.percentile(v, q)) if len(v) else float("nan")
    return {
        "ratio": a.mean() / b.mean() if b.mean() > 0 else float("nan"),
        "ratio_lo": pct(ratios, 2.5),
        "ratio_hi": pct(ratios, 97.5),
        "diff": float(a.mean() - b.mean()),
        "diff_lo": pct(diffs, 2.5),
        "diff_hi": pct(diffs, 97.5),
        "dropped": dropped,
    }


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def section_main(ctx: dict, split: str) -> None:
    K, n_items = ctx["K"], ctx["n_items"]
    df, gt, users, hists = ctx["df"], ctx["gt"], ctx["users"], ctx["hists"]
    preds, hits, n = ctx["preds"], ctx["hits"], len(df)
    args = ctx["args"]

    print("=" * 94)
    print(f"{split.upper()}  -  {n} queries, {len(np.unique(users))} customers, "
          f"ranked set: all {n_items} items, seen items not filtered")
    print("=" * 94)
    hdr = (f"{'model':<16} {'Recall@' + str(K):>10} {'NDCG@' + str(K):>10} "
           f"{'coverage@' + str(K):>12} {'hits':>6}")
    print(hdr)
    print("-" * len(hdr))
    for m, p in preds.items():
        print(f"{m:<16} {recall_at_k(p, gt, K):>10.4f} {ndcg_at_k(p, gt, K):>10.4f} "
              f"{coverage_at_k(p, n_items, K):>12.4f} {int(hits[m][0].sum()):>6}")

    # Random sanity check. One seed expects n*K/n_items hits (under 1 here), so
    # a single draw tests the draw, not the evaluator. Average over seeds.
    expected = K / n_items
    many = np.array([
        hit_vectors(random_preds(n_items, hists, K, s), gt, K)[0].mean()
        for s in range(args.random_seeds)
    ])
    se = many.std(ddof=1) / np.sqrt(args.random_seeds)
    flag = "OK" if abs(many.mean() - expected) < 3 * se else "CHECK THE EVALUATOR"
    print(f"\nRandom sanity check: expected Recall@{K} ~ {expected:.4f} (= {K}/{n_items}); "
          f"seed {args.seed} gave {hits['random'][0].mean():.4f} "
          f"({int(hits['random'][0].sum())} hits, {n * expected:.1f} expected)")
    print(f"  mean over {args.random_seeds} seeds: {many.mean():.4f} +/- {se:.4f} (SE)  [{flag}]")

    print(f"\nUplift over MostPopular ({args.bootstrap} bootstrap resamples over "
          f"{len(np.unique(users))} customers, seed {args.seed})")
    for model in ITEM_ITEM:
        for j, metric in enumerate([f"Recall@{K}", f"NDCG@{K}"]):
            s = bootstrap(hits[model][j], hits["popularity"][j], users, args.bootstrap, args.seed)
            note = f"  [{s['dropped']} resamples dropped]" if s["dropped"] else ""
            print(f"  {metric:<10} {model:<14} ratio {s['ratio']:>6.2f}x "
                  f"[{s['ratio_lo']:.2f}, {s['ratio_hi']:>6.2f}]   "
                  f"diff {s['diff']:+.4f} [{s['diff_lo']:+.4f}, {s['diff_hi']:+.4f}]{note}")

    print(f"\nRecall@{K} by query history length")
    hdr2 = (f"{'history':<9} {'N':>5} {'custs':>6} {'item_knn':>10} {'bm25':>8} "
            f"{'ease':>8} {'popularity':>12} {'random':>9}")
    print(hdr2)
    print("-" * len(hdr2))
    hist_len = np.array([len(h) for h in hists])
    for label, lo_b, hi_b in HIST_BUCKETS:
        mask = (hist_len >= lo_b) & (hist_len <= hi_b)
        if not mask.any():
            print(f"{label:<9} {0:>5} {0:>6} {'-':>10} {'-':>8} {'-':>8} {'-':>12} {'-':>9}")
            continue
        cells = [f"{hits[m][0][mask].mean():>{w}.4f}" for m, w in
                 (("item_knn", 10), ("item_knn_bm25", 8), ("ease", 8),
                  ("popularity", 12), ("random", 9))]
        print(f"{label:<9} {int(mask.sum()):>5} {len(np.unique(users[mask])):>6} " + " ".join(cells))

    in_hist = np.array([g in h for g, h in zip(gt, hists)])
    knn_hit = hits["item_knn"][0].astype(bool)
    seen_slots = np.mean([len(set(p) & set(h)) / K for p, h in zip(preds["item_knn"], hists)])
    sd = ctx["sd"]
    print(f"\nrepeat purchases: the positive is already in the history for {in_hist.mean():.1%} of "
          f"queries; {int((knn_hit & in_hist).sum())} of item_knn's {int(knn_hit.sum())} hits are repeats")
    print(f"item_knn top-{K} slots holding an already-purchased item: {seen_slots:.1%}")
    print(f"same-day basket: {sd.mean():.1%} of queries carry a history item bought on the "
          f"target's own date (invoice_date has no time of day)")
    hdr3 = f"{'stratum':<26} {'N':>5} {'item_knn':>10} {'popularity':>12} {'ratio':>8}"
    print(hdr3)
    print("-" * len(hdr3))
    for label, mask in (("includes same-day lines", sd), ("strictly earlier only", ~sd)):
        if not mask.any():
            continue
        rk, rp = hits["item_knn"][0][mask].mean(), hits["popularity"][0][mask].mean()
        print(f"{label:<26} {int(mask.sum()):>5} {rk:>10.4f} {rp:>12.4f} "
              f"{(f'{rk / rp:.2f}x' if rp > 0 else 'inf'):>8}")
    print()


def section_grid(ctx: dict, split: str) -> None:
    """{same-day history kept, dropped} x {seen items kept, filtered}."""
    K, n_items = ctx["K"], ctx["n_items"]
    df, gt, hists, inter_idx = ctx["df"], ctx["gt"], ctx["hists"], ctx["inter_idx"]
    hists_clean = strip_same_day(df, hists, ctx["by_cust_day"])
    # Dropping same-day lines empties some histories entirely. ItemKNN then
    # scores every item 0 and the ranking degenerates to item_idx order, which
    # is not a recommendation and would not be comparable to the other cells.
    # All four cells are therefore reported on the queries that survive both
    # rules; the full-split reference row is printed above them.
    keep = np.array([bool(h) for h in hists_clean])
    gt_k = [g for g, m in zip(gt, keep) if m]

    print("=" * 94)
    print(f"{split.upper()}  -  Recall@{K} / NDCG@{K} under both history and seen-item rules")
    print("=" * 94)
    print(f"dropping same-day history items empties {int((~keep).sum())} of {len(df)} histories "
          f"(their history was nothing but same-day lines); the grid below uses the "
          f"{int(keep.sum())} queries that survive both rules")
    full = rank(ctx["scores"]["item_knn"], K)
    print(f"reference, all {len(df)} queries, same-day kept / seen kept: "
          f"Recall@{K}={recall_at_k(full, gt, K):.4f}, NDCG@{K}={ndcg_at_k(full, gt, K):.4f}\n")

    hdr = (f"{'model':<12} {'same-day history':<18} {'seen items':<12} {'N':>5} "
           f"{'Recall@' + str(K):>10} {'NDCG@' + str(K):>10}")
    print(hdr)
    print("-" * len(hdr))
    for hist_label, hh in (("kept", hists), ("dropped", hists_clean)):
        knn = time_aware_scores(MODELS["item_knn"], inter_idx, df["ts"].tolist(), hh, n_items)
        scores = {"item_knn": knn, "popularity": ctx["scores"]["popularity"]}
        seen = [set(h) for h in hh]
        for model in ("item_knn", "popularity"):
            for seen_label, ex in (("kept", None), ("filtered", seen)):
                p = [q for q, m in zip(rank(scores[model], K, ex), keep) if m]
                print(f"{model:<12} {hist_label:<18} {seen_label:<12} {len(p):>5} "
                      f"{recall_at_k(p, gt_k, K):>10.4f} {ndcg_at_k(p, gt_k, K):>10.4f}")
    print()


def section_rolling(ctx: dict) -> None:
    """The same protocol over consecutive temporal windows, then pooled."""
    K, n_items, args = ctx["K"], ctx["n_items"], ctx["args"]
    inter_idx, item_map, customer_map = ctx["inter_idx"], ctx["item_map"], ctx["customer_map"]
    n_win = args.windows

    dates = inter_idx["invoice_date"]
    # Windows tile the tail of the year by interaction quantile, so the last one
    # is exactly the shipped test split and the others are its predecessors.
    qs = [1.0 - (n_win - i) * 0.10 for i in range(n_win)] + [1.0]
    edges = [dates.quantile(q) for q in qs[:-1]] + [dates.max() + pd.Timedelta(seconds=1)]

    print("=" * 94)
    print(f"ROLLING TEMPORAL EVALUATION  -  {n_win} consecutive windows, "
          f"every model refit on purchases strictly before each query")
    print("=" * 94)
    hdr = (f"{'window':<8} {'dates':<25} {'N':>5} {'custs':>6} {'item_knn':>10} "
           f"{'popularity':>12} {'knn NDCG':>10} {'pop NDCG':>10}")
    print(hdr)
    print("-" * len(hdr))

    pooled: Dict[str, List[np.ndarray]] = {"knn_r": [], "pop_r": [], "knn_n": [], "pop_n": []}
    pooled_users, per_window = [], []
    for w in range(n_win):
        start, end = edges[w], edges[w + 1]
        df = build_window(inter_idx, item_map, customer_map, start, end)
        if df.empty:
            continue
        gt = df["pos_item_idx"].astype(int).tolist()
        hists = histories_of(df)
        users = df["user_idx"].astype(int).to_numpy()
        knn = rank(time_aware_scores(MODELS["item_knn"], inter_idx, df["ts"].tolist(),
                                     hists, n_items), K)
        pop = rank(popularity_scores(inter_idx[dates < start], n_items, len(df)), K)
        kr, kn = hit_vectors(knn, gt, K)
        pr, pn = hit_vectors(pop, gt, K)
        label = f"W{w + 1}"
        print(f"{label:<8} {str(start.date()) + '..' + str(min(end, dates.max()).date()):<25} "
              f"{len(df):>5} {len(np.unique(users)):>6} {kr.mean():>10.4f} {pr.mean():>12.4f} "
              f"{kn.mean():>10.4f} {pn.mean():>10.4f}")
        s = bootstrap(kr, pr, users, args.bootstrap, args.seed)
        per_window.append((label, s))
        pooled["knn_r"].append(kr), pooled["pop_r"].append(pr)
        pooled["knn_n"].append(kn), pooled["pop_n"].append(pn)
        # Namespace users per window so pooling resamples customers, not
        # customer-windows; a customer seen twice keeps both windows together.
        pooled_users.append(users)

    cat = {k: np.concatenate(v) for k, v in pooled.items()}
    users_all = np.concatenate(pooled_users)
    print(f"{'POOLED':<8} {str(edges[0].date()) + '..' + str(dates.max().date()):<25} "
          f"{len(cat['knn_r']):>5} {len(np.unique(users_all)):>6} {cat['knn_r'].mean():>10.4f} "
          f"{cat['pop_r'].mean():>12.4f} {cat['knn_n'].mean():>10.4f} {cat['pop_n'].mean():>10.4f}")

    print(f"\nUplift, {args.bootstrap} resamples clustered on customer (seed {args.seed})")
    hdr2 = (f"{'window':<8} {'metric':<10} {'ratio':>7} {'95% CI':>18} {'width':>8}   "
            f"{'diff':>8} {'95% CI':>20} {'width':>8}")
    print(hdr2)
    print("-" * len(hdr2))
    rows = per_window + [("POOLED", None)]
    for label, s in rows:
        for j, metric in enumerate([f"Recall@{K}", f"NDCG@{K}"]):
            if label == "POOLED":
                a = cat["knn_r"] if j == 0 else cat["knn_n"]
                b = cat["pop_r"] if j == 0 else cat["pop_n"]
                s2 = bootstrap(a, b, users_all, args.bootstrap, args.seed)
            elif j == 0:
                s2 = s
            else:
                continue  # per-window NDCG CI is in the pooled block only
            note = f"  [{s2['dropped']} dropped]" if s2["dropped"] else ""
            ratio_ci = "[{:.2f}, {:.2f}]".format(s2["ratio_lo"], s2["ratio_hi"])
            diff_ci = "[{:+.4f}, {:+.4f}]".format(s2["diff_lo"], s2["diff_hi"])
            print(f"{label:<8} {metric:<10} {s2['ratio']:>6.2f}x {ratio_ci:>18} "
                  f"{s2['ratio_hi'] - s2['ratio_lo']:>8.2f}   {s2['diff']:>+8.4f} "
                  f"{diff_ci:>20} {s2['diff_hi'] - s2['diff_lo']:>8.4f}{note}")
    print()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ItemKNN uplift over MostPopular / Random / EASE")
    p.add_argument("--data-dir", default="models/data")
    p.add_argument("--splits", nargs="+", default=["test"], choices=["train", "val", "test"])
    p.add_argument("--sections", nargs="+", default=["main", "grid", "rolling"],
                   choices=["main", "grid", "rolling"])
    p.add_argument("--k", type=int, default=10, help="Cut-off for Recall@K / NDCG@K / coverage@K")
    p.add_argument("--windows", type=int, default=4, help="Rolling temporal windows")
    p.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap resamples over customers")
    p.add_argument("--random-seeds", type=int, default=200,
                   help="Seeds averaged for the Random sanity check")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    K = args.k
    data_dir = Path(args.data_dir)

    seqs = load_processed_sequences(data_dir, splits=args.splits)
    item_map, customer_map = load_id_mappings(data_dir)
    interactions = read_parquet(data_dir / "interactions_clean.parquet")
    inter_idx = interactions.merge(item_map, on="stock_code", how="inner")
    n_items = len(item_map)
    by_cust_day = same_day_items(inter_idx)

    for split in args.splits:
        df = seqs[split].reset_index(drop=True)
        gt = df["pos_item_idx"].astype(int).tolist()
        users = df["user_idx"].astype(int).to_numpy()
        hists = histories_of(df)
        ts = df["ts"].tolist()

        scores = {m: time_aware_scores(f, inter_idx, ts, hists, n_items) for m, f in MODELS.items()}
        scores["popularity"] = popularity_scores(
            inter_idx[inter_idx["invoice_date"] < df["ts"].min()], n_items, len(df)
        )
        scores["popularity_t"] = popularity_scores_time_aware(inter_idx, ts, n_items)

        preds = {m: rank(s, K) for m, s in scores.items()}
        preds["random"] = random_preds(n_items, hists, K, args.seed)
        preds["random_unseen"] = random_preds(n_items, hists, K, args.seed, exclude_seen=True)
        order = ["random", "random_unseen", "popularity", "popularity_t", *ITEM_ITEM]
        preds = {m: preds[m] for m in order}

        ctx = dict(
            args=args, K=K, n_items=n_items, df=df, gt=gt, users=users, hists=hists,
            preds=preds, hits={m: hit_vectors(p, gt, K) for m, p in preds.items()},
            scores=scores, inter_idx=inter_idx, item_map=item_map, customer_map=customer_map,
            by_cust_day=by_cust_day, sd=same_day_mask(df, hists, gt, by_cust_day),
        )
        if "main" in args.sections:
            section_main(ctx, split)
        if "grid" in args.sections:
            section_grid(ctx, split)

    if "rolling" in args.sections:
        section_rolling(dict(args=args, K=K, n_items=n_items, inter_idx=inter_idx,
                             item_map=item_map, customer_map=customer_map))


if __name__ == "__main__":
    main()
