#!/usr/bin/env python3
"""
Build candidate lists from Two-Tower embeddings (notebook-aligned).

This script loads sequences_{train,val,test}.parquet and the exported
user/item embeddings, computes user vectors from history, retrieves
top-K items per query, and writes candidates_{split}.parquet.

The positive item is NOT inserted into the list when the retriever misses it.
Doing so (as earlier versions did) teaches the reranker to spot the inserted
item and makes candidate-level metrics meaningless.

Example:
  python scripts/build_candidates.py --data-dir models/data --k 200
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.utils.io import read_numpy
from src.utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build candidate lists from embeddings",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-dir", type=str, default="models/data", help="Processed data directory")
    p.add_argument("--emb-dir", type=str, default="models/retriever", help="Directory with user/item embeddings")
    p.add_argument("--k", type=int, default=200, help="Candidates per query")
    p.add_argument("--batch", type=int, default=4096, help="Batch size for retrieval")
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    return p.parse_args()


def history_to_tensor(series: pd.Series, L: int) -> torch.Tensor:
    H = torch.full((len(series), L), -1, dtype=torch.long)
    for i, s in enumerate(series.astype(str).tolist()):
        if not s or s == "nan":
            continue
        h = [int(x) for x in s.split() if x.strip()]
        if len(h) > L:
            h = h[-L:]
        if h:
            H[i, -len(h):] = torch.tensor(h, dtype=torch.long)
    return H


def main() -> None:
    args = parse_args()
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)

    data_dir = Path(args.data_dir)
    emb_dir = Path(args.emb_dir)

    seq_paths = {s: data_dir / f"sequences_{s}.parquet" for s in ["train","val","test"]}
    for s, p in seq_paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing sequences file: {p}")

    # Load sequences
    seq = {s: pd.read_parquet(p) for s, p in seq_paths.items()}
    logger.info("Loaded sequences: train=%d, val=%d, test=%d", len(seq['train']), len(seq['val']), len(seq['test']))

    # Load embeddings
    user_emb_path = emb_dir / "user_embeddings.npy"
    item_emb_path = emb_dir / "item_embeddings.npy"
    if not user_emb_path.exists() or not item_emb_path.exists():
        # Allow alternative: embeddings saved to processed dir (notebook parity)
        user_emb_path = data_dir / "user_embeddings.npy"
        item_emb_path = data_dir / "item_embeddings.npy"
    user_emb = read_numpy(user_emb_path).astype("float32")
    item_emb = read_numpy(item_emb_path).astype("float32")
    logger.info("Embeddings loaded: users %s, items %s", user_emb.shape, item_emb.shape)

    # Torch tensors
    device = torch.device("cpu")
    ITEM = torch.from_numpy(item_emb).to(device)
    ITEM = F.normalize(ITEM, dim=-1)

    L = 15
    K = int(args.k)
    BQ = int(args.batch)

    def build_for_split(df: pd.DataFrame) -> pd.DataFrame:
        if len(df) == 0:
            return pd.DataFrame(columns=["history_idx","pos_item_idx","cands","ts"]) 
        H = history_to_tensor(df['history_idx'], L)
        rows = []
        with torch.no_grad():
            for i in range(0, len(df), BQ):
                Hb = H[i:i+ BQ].to(device)
                safe = Hb.clamp(min=0)
                Hbv = ITEM.index_select(0, safe.view(-1)).view(Hb.size(0), Hb.size(1), -1)
                mask = (Hb >= 0).float().unsqueeze(-1)
                U = (Hbv * mask).sum(1) / mask.sum(1).clamp_min(1e-6)
                U = F.normalize(U, dim=-1)
                sims = U @ ITEM.t()
                _, topk = torch.topk(sims, k=K, dim=1)
                topk = topk.cpu().numpy()
                for j, (pos, h) in enumerate(zip(df['pos_item_idx'].iloc[i:i+ BQ].astype(int).tolist(),
                                                 df['history_idx'].iloc[i:i+ BQ].astype(str).tolist())):
                    cands = topk[j].tolist()
                    rows.append((h, int(pos), " ".join(map(str,cands)), str(df['ts'].iloc[i+j]) if 'ts' in df.columns else ""))
        return pd.DataFrame(rows, columns=["history_idx","pos_item_idx","cands","ts"]) 

    for split in ["train","val","test"]:
        logger.info("Building candidates for %s", split)
        out = build_for_split(seq[split])
        out_path = data_dir / f"candidates_{split}.parquet"
        out.to_parquet(out_path, index=False)
        present = float(np.mean([str(p) in c.split() for p, c in zip(out["pos_item_idx"], out["cands"])])) if len(out) else 0.0
        logger.info("Saved %s (%d rows); positive retrieved in %.1f%% of lists (Recall@%d)", out_path, len(out), 100 * present, K)


if __name__ == "__main__":
    main()
