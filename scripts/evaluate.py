#!/usr/bin/env python3
"""
Evaluate the saved Jarir recommender artifacts on the val and test splits.

Everything is computed from artifacts already on disk; nothing is trained or
tuned here:
  - models/data/*.parquet            (sequences, candidates, id maps, catalog)
  - models/reranker/best_ranker.pt   (src MLPRanker, written by train_reranker.py)
  - models/retriever/*.npy           (Two-Tower embeddings, for rows B/C and the
                                      reranker dot_uv / max_sim_recent features)

Every row is Recall@10 / NDCG@10 with one relevant item per query, so Recall@10
is a hit rate and NDCG@10 is 1/log2(rank+1) when hit. The "ranked set" column
is the denominator, and it is the number to read first: a metric over 200
candidates is not comparable to one over the whole catalogue.

  A   Popularity: the 10 items with the most purchase lines dated before the
      split starts, the same list for every query.        Ranked set: all items.
  A2  Repeat purchase: the customer's own recent distinct items, oldest dropped
      first, padded with A.                                Ranked set: all items.
  A3  ItemKNN retriever: item-item cosine over the purchase matrix, refitted per
      query date on purchases strictly before it.          Ranked set: all items.
  B   Two-Tower retriever using the user-ID embedding.     Ranked set: all items.
  C   Two-Tower retriever using the history-mean embedding.Ranked set: all items.
  D   Full pipeline: --retriever retrieves --cand-k candidates, the reranker
      scores them, top 10 kept.                            Ranked set: all items.
  E   Hybrid: the customer's recent items first, then D.   Ranked set: all items.
  F   Reranker over the saved candidates_<split>.parquet lists.
                                          Ranked set: the stored candidate list.
  G   Reranker over freshly retrieved candidates, restricted to the queries
      where the retriever actually returned the positive.
                                          Ranked set: --cand-k candidates.
  G0  The same queries as G in the retriever's own order, with no reranking.
      G below G0 means the reranker is destroying ranking quality.
                                          Ranked set: --cand-k candidates.

Rows F and G are candidate-restricted and will always look better than rows over
the catalogue. The README once quoted a row-F-style number (~0.69) as the
system's performance, at a time when the ground-truth item was inserted into
every candidate list. The diagnostics under the table report how many lists had
the positive inserted rather than retrieved.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --retriever two_tower --cand-k 100 --splits val test
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.data.dataset import load_id_mappings, load_processed_sequences, parse_history_string
from src.data.features import RankingFeatureBuilder, build_item_side_features, positive_in_candidates
from src.evaluation.metrics import ndcg_at_k, recall_at_k
from src.inference.ann_index import build_index_from_embeddings
from src.inference.item_knn import interactions_with_item_idx, time_aware_topk
from src.models.reranker.mlp_ranker import MLPRanker
from src.utils.config import load_config
from src.utils.io import read_numpy, read_parquet
from src.utils.seed import set_seed

FEATURE_COLS = ["dot_uv", "max_sim_recent", "pop", "hist_len", "price_z", "retr_rr"]
MAX_HIST = 15


# --------------------------------------------------------------------------- #
# Reranker checkpoints. Two layouts exist:
#   - src MLPRanker (scripts/train_reranker.py -> best_ranker.pt), keys "layers.*"
#   - legacy notebook-04 "RankerMLP" (ranker_best.pt), keys "fc1.*"; kept so the
#     old checkpoint can still be evaluated.
# --------------------------------------------------------------------------- #
class RankerMLP(nn.Module):
    def __init__(self, d_in: int = 5, hidden: int = 384, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(d_in, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden // 2)
        self.out = nn.Linear(hidden // 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.ln1 = nn.LayerNorm(hidden)
        self.ln2 = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.ln1(self.dropout(self.act(self.fc1(x))))
        h2 = self.ln2(self.dropout(self.act(self.fc2(h1))))
        h = self.dropout(self.act(self.fc3(h1 + h2)))
        return self.out(h).squeeze(-1)


def load_ranker(path: Path, config_path: Path = Path("configs/reranker.yaml")) -> nn.Module:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "layers.0.weight" in state:
        model_cfg = load_config(config_path).get("model", {}) if config_path.exists() else {}
        fd = model_cfg.get("feature_dropout", {})
        model = MLPRanker(
            input_dim=model_cfg.get("input_dim", 6),
            hidden_dims=model_cfg.get("hidden_dims", [384, 384, 192]),
            dropout=model_cfg.get("dropout", 0.3),
            feature_dropout_indices=(fd.get("feature_indices", [0]) if fd.get("enabled", True) else None),
            feature_dropout_prob=fd.get("dropout_prob", 0.3),
            use_residual=model_cfg.get("use_residual", True),
            use_layer_norm=model_cfg.get("use_layer_norm", True),
            activation=model_cfg.get("activation", "relu"),
        )
    else:
        model = RankerMLP()
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def histories_of(df: pd.DataFrame) -> List[List[int]]:
    return [parse_history_string(str(h))[-MAX_HIST:] for h in df["history_idx"].tolist()]


def history_mean_vectors(item_emb: np.ndarray, histories: List[List[int]]) -> np.ndarray:
    """Mean of the (last MAX_HIST) history item embeddings, L2-normalised."""
    out = np.zeros((len(histories), item_emb.shape[1]), dtype=np.float32)
    for i, h in enumerate(histories):
        if h:
            v = item_emb[h].mean(axis=0)
            out[i] = v / (np.linalg.norm(v) + 1e-8)
    return out


def repeat_purchase_lists(hists: List[List[int]], pop_top: List[int], k: int) -> List[List[int]]:
    """Most recent distinct history items first, padded with the popularity list."""
    out = []
    for h in hists:
        seen, rec = set(), []
        for it in reversed(h):
            if it not in seen:
                seen.add(it)
                rec.append(it)
        for it in pop_top:
            if len(rec) >= k:
                break
            if it not in seen:
                seen.add(it)
                rec.append(it)
        out.append(rec[:k])
    return out


def rerank_lists(
    model: nn.Module,
    cand_df: pd.DataFrame,
    item_emb: np.ndarray,
    pop_vec: np.ndarray,
    price_z: np.ndarray,
    k: int,
) -> List[List[int]]:
    """Score every candidate with the reranker; return the top-k item ids per query."""
    fb = RankingFeatureBuilder(
        embedding_dim=item_emb.shape[1],
        max_history_length=MAX_HIST,
        hard_negatives=False,
        n_negatives_per_query=None,  # keep the full candidate list
    )
    feats = fb.build_features(
        cand_df,
        user_embeddings=np.zeros((1, item_emb.shape[1]), dtype=np.float32),  # unused by the builder
        item_embeddings=item_emb,
        popularity_scores=pop_vec,
        price_features=price_z,
        device="cpu",
        batch_size=1024,
    )
    X = np.stack([feats[c] for c in FEATURE_COLS], axis=-1).astype(np.float32)  # (Q, K, 5)
    Q, K, F_ = X.shape
    with torch.no_grad():
        scores = model(torch.from_numpy(X.reshape(-1, F_))).view(Q, K)
    top = torch.topk(scores, k=min(k, K), dim=1).indices.numpy()
    cands = feats["item_idx"].astype(np.int64)  # (Q, K)
    return [cands[i, top[i]].tolist() for i in range(Q)]


def make_cand_df(seq_df: pd.DataFrame, cand_idx: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "history_idx": seq_df["history_idx"].astype(str).values,
            "pos_item_idx": seq_df["pos_item_idx"].astype(int).values,
            "cands": [" ".join(map(str, row.tolist())) for row in cand_idx],
        }
    )


def metrics(preds: List[List[int]], gt: List[int], k: int) -> Tuple[float, float]:
    return recall_at_k(preds, gt, k), ndcg_at_k(preds, gt, k)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate saved retriever / reranker artifacts")
    p.add_argument("--data-dir", default="models/data")
    p.add_argument("--retriever-dir", default="models/retriever")
    p.add_argument("--ranker", default="models/reranker/best_ranker.pt")
    p.add_argument("--splits", nargs="+", default=["val", "test"], choices=["train", "val", "test"])
    p.add_argument("--k", type=int, default=10, help="Cut-off for Recall@K / NDCG@K")
    p.add_argument("--cand-k", type=int, default=200, help="Candidates retrieved before reranking")
    p.add_argument("--retriever", default="item_knn", choices=["item_knn", "two_tower"],
                   help="Candidate generator for the full pipeline (rows D-G)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch.set_num_threads(1)

    data_dir = Path(args.data_dir)
    K = args.k

    # ---- data ------------------------------------------------------------- #
    seqs = load_processed_sequences(data_dir, splits=["train", "val", "test"])
    item_map, customer_map = load_id_mappings(data_dir)
    items_clean = read_parquet(data_dir / "items_clean.parquet")
    customers_clean = read_parquet(data_dir / "customers_clean.parquet")
    interactions = read_parquet(data_dir / "interactions_clean.parquet")
    inter_idx = interactions.merge(item_map, on="stock_code", how="inner")
    n_items, n_users = len(item_map), len(customer_map)

    print("=" * 96)
    print("DATASET (models/data/*.parquet)")
    print("=" * 96)
    print(f"customers_clean.parquet      : {len(customers_clean):>6} customers "
          f"(customer_id_map: {n_users})")
    print(f"items_clean.parquet          : {len(items_clean):>6} items "
          f"(item_id_map: {n_items})")
    print(f"interactions_clean.parquet   : {len(interactions):>6} purchase lines, "
          f"{interactions['customer_id'].nunique()} customers, {interactions['stock_code'].nunique()} items, "
          f"{interactions['invoice_date'].min().date()} .. {interactions['invoice_date'].max().date()}")
    for s in ["train", "val", "test"]:
        d = seqs[s]
        print(f"sequences_{s:<5}.parquet     : {len(d):>6} sequences, "
              f"{d['user_idx'].nunique():>4} customers, ts {d['ts'].min().date()} .. {d['ts'].max().date()}")
    cand_files = {s: data_dir / f"candidates_{s}.parquet" for s in ["train", "val", "test"]}
    for s, p in cand_files.items():
        if p.exists():
            c = read_parquet(p)
            kk = c["cands"].str.split().str.len()
            inj = float(np.mean([str(a) in b.split() for a, b in zip(c["pos_item_idx"], c["cands"])]))
            print(f"candidates_{s:<5}.parquet    : {len(c):>6} queries, {int(kk.min())}..{int(kk.max())} "
                  f"candidates each, positive present in {inj:.1%} of lists")

    # ---- artifacts -------------------------------------------------------- #
    user_emb = read_numpy(Path(args.retriever_dir) / "user_embeddings.npy").astype(np.float32)
    item_emb = read_numpy(Path(args.retriever_dir) / "item_embeddings.npy").astype(np.float32)
    assert user_emb.shape[0] == n_users and item_emb.shape[0] == n_items, "embedding / id-map size mismatch"
    ranker = load_ranker(Path(args.ranker))
    index = build_index_from_embeddings(item_emb, index_type="flat", metric="cosine")  # exact search
    pop_vec, price_z = build_item_side_features(seqs["train"], items_clean, item_map, n_items)

    print()
    print(f"user_embeddings {user_emb.shape}, item_embeddings {item_emb.shape}, "
          f"ranker {args.ranker}, seed {args.seed}, K={K}, cand_k={args.cand_k}")

    # ---- evaluation ------------------------------------------------------- #
    rows = []
    diag = []
    for split in args.splits:
        df = seqs[split].reset_index(drop=True)
        gt = df["pos_item_idx"].astype(int).tolist()
        users = df["user_idx"].astype(int).values
        hists = histories_of(df)
        n = len(df)
        assert all(len(h) > 0 for h in hists), "empty history in split"

        # A. popularity, fit strictly before the first timestamp of the split
        split_start = df["ts"].min()
        past = inter_idx[inter_idx["invoice_date"] < split_start]
        counts = past.groupby("item_idx").size()
        pop_top = [i for i, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:K]]
        preds_A = [pop_top] * n
        rows.append((split, "A", "Popularity top-10 (purchase lines before split start), same list for all",
                     f"all {n_items} items", n, *metrics(preds_A, gt, K)))

        # A2. repeat purchase: the customer's own recent items, padded with popularity
        preds_A2 = repeat_purchase_lists(hists, pop_top, K)
        rows.append((split, "A2", "Repeat purchase: recent distinct history items, then popularity",
                     f"all {n_items} items", n, *metrics(preds_A2, gt, K)))

        # A3. ItemKNN, fitted on purchases strictly before each query's date
        big_k = max(args.cand_k, 200)
        idx_knn_big = time_aware_topk(inter_idx, df["ts"].tolist(), hists, n_items, big_k)
        preds_A3 = idx_knn_big[:, :K].tolist()
        rows.append((split, "A3", "ItemKNN retriever (recency-weighted, fit on purchases before query date)",
                     f"all {n_items} items", n, *metrics(preds_A3, gt, K)))

        # B. retriever with user-ID embedding
        _, idx_B = index.search(user_emb[users], k=K)
        preds_B = idx_B.tolist()
        rows.append((split, "B", "Retriever, user-ID embedding (user_embeddings.npy)",
                     f"all {n_items} items", n, *metrics(preds_B, gt, K)))

        # C. retriever with history-mean embedding
        U = history_mean_vectors(item_emb, hists)
        _, idx_C_big = index.search(U, k=big_k)
        preds_C = idx_C_big[:, :K].tolist()
        rows.append((split, "C", "Two-Tower retriever, history-mean embedding",
                     f"all {n_items} items", n, *metrics(preds_C, gt, K)))
        in_hist = float(np.mean([g in h for g, h in zip(gt, hists)]))
        diag.append(f"[{split}] positive item already in the query history: {in_hist:.1%} of queries")
        for name, big in (("Two-Tower history-mean", idx_C_big), ("ItemKNN", idx_knn_big)):
            r100 = recall_at_k(big[:, :100].tolist(), gt, 100)
            r200 = recall_at_k(big[:, :200].tolist(), gt, 200)
            diag.append(f"[{split}] {name} candidate recall: Recall@100={r100:.4f}, Recall@200={r200:.4f}")

        # D. full pipeline: retrieve cand_k (no injection) -> rerank -> top K
        cand_big = idx_knn_big if args.retriever == "item_knn" else idx_C_big
        retr_name = "ItemKNN" if args.retriever == "item_knn" else "Two-Tower"
        cand_idx = cand_big[:, : args.cand_k]
        cand_df = make_cand_df(df, cand_idx)
        preds_D = rerank_lists(ranker, cand_df, item_emb, pop_vec, price_z, K)
        rows.append((split, "D", f"Pipeline: {retr_name} retrieves {args.cand_k} -> {Path(args.ranker).name} -> top {K}",
                     f"all {n_items} items", n, *metrics(preds_D, gt, K)))

        # E. hybrid: the customer's own recent items first, then the pipeline's picks
        preds_E = [r[:K] for r in (
            [it for it in rp if it in set(h)] + [it for it in d if it not in set(h)]
            for rp, d, h in zip(preds_A2, preds_D, hists)
        )]
        rows.append((split, "E", "Hybrid: recent history items first, then pipeline picks",
                     f"all {n_items} items", n, *metrics(preds_E, gt, K)))

        # F. reranker over the saved candidate lists in models/data
        if cand_files[split].exists():
            saved = read_parquet(cand_files[split])
            assert len(saved) == n, "candidates file / sequences length mismatch"
            kk = int(saved["cands"].str.split().str.len().iloc[0])
            present = positive_in_candidates(saved)
            preds_F = rerank_lists(ranker, saved, item_emb, pop_vec, price_z, K)
            gt_F = saved["pos_item_idx"].astype(int).tolist()
            rows.append((split, "F", f"Reranker over saved candidates_{split}.parquet",
                         f"{kk} cands, positive present in {present.mean():.0%}", n, *metrics(preds_F, gt_F, K)))
            # Was the positive genuinely retrieved (in fresh top-kk) or inserted into the file?
            fresh = cand_big[:, :kk].tolist()
            injected = np.array([g not in row for g, row in zip(gt_F, fresh)]) & present
            hit_F = np.array([g in p for g, p in zip(gt_F, preds_F)])
            if injected.any():
                r_inj = hit_F[injected].mean()
                r_ret = hit_F[present & ~injected].mean() if (present & ~injected).any() else float("nan")
                diag.append(f"[{split}] row F breakdown: positive force-inserted in {injected.mean():.1%} of lists "
                            f"(hit rate {r_inj:.4f}, n={int(injected.sum())}); positive genuinely retrieved in "
                            f"{(present & ~injected).mean():.1%} (hit rate {r_ret:.4f}, n={int((present & ~injected).sum())})")

        # G. reranker over fresh candidates, only queries where the positive was retrieved
        hit_mask = np.array([g in row for g, row in zip(gt, cand_idx.tolist())])
        if hit_mask.any():
            preds_G = [p for p, m in zip(preds_D, hit_mask) if m]
            gt_G = [g for g, m in zip(gt, hit_mask) if m]
            rows.append((split, "G", f"Reranker over fresh {retr_name} {args.cand_k} cands, queries where positive was retrieved",
                         f"{args.cand_k} cands, positive present", int(hit_mask.sum()), *metrics(preds_G, gt_G, K)))
            # Same queries, retriever's own order: does the reranker beat it?
            preds_G0 = [row[:K] for row, m in zip(cand_idx.tolist(), hit_mask) if m]
            rows.append((split, "G0", f"Same queries as G, {retr_name} order without reranking",
                         f"{args.cand_k} cands, positive present", int(hit_mask.sum()), *metrics(preds_G0, gt_G, K)))

    # ---- table ------------------------------------------------------------ #
    print()
    print("=" * 96)
    print(f"RESULTS  (Recall@{K} = hit rate of the single held-out next purchase; NDCG@{K} = 1/log2(rank+1) on hit)")
    print("=" * 96)
    hdr = f"{'split':<5} {'row':<3} {'model':<78} {'ranked set':<36} {'N':>5} {'Recall@'+str(K):>10} {'NDCG@'+str(K):>9}"
    print(hdr)
    print("-" * len(hdr))
    for split, tag, name, denom, n, r, nd in rows:
        print(f"{split:<5} {tag:<3} {name:<78} {denom:<36} {n:>5} {r:>10.4f} {nd:>9.4f}")
    print()
    for line in diag:
        print(line)


if __name__ == "__main__":
    main()
