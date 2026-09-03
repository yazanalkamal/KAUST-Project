"""
Guards against the failure modes that produced the original inflated metrics.

The headline numbers in this project were once ~0.69 Recall@10 because the
ground-truth item was inserted into every candidate list and the reranker
learned to spot it. These tests fail if that ever comes back.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.features import RankingFeatureBuilder, build_item_side_features, positive_in_candidates
from src.evaluation.metrics import ndcg_at_k, recall_at_k
from src.inference.item_knn import ItemKNNRetriever, interactions_with_item_idx, time_aware_topk

DATA_DIR = Path("models/data")
pytestmark = pytest.mark.skipif(not DATA_DIR.exists(), reason="models/data not available")


def _read(name):
    return pd.read_parquet(DATA_DIR / name)


def test_candidates_do_not_contain_every_positive():
    """A candidate file where the positive is always present means it was inserted."""
    for split in ["train", "val", "test"]:
        cand = _read(f"candidates_{split}.parquet")
        rate = positive_in_candidates(cand).mean()
        assert rate < 0.95, (
            f"candidates_{split}.parquet contains the positive item in {rate:.1%} of lists. "
            "Candidate lists must come from retrieval only; see scripts/build_candidates.py."
        )
        assert rate > 0.0, f"candidates_{split}.parquet never retrieves the positive"


def test_candidate_lists_are_ordered_best_first():
    """retr_rr assumes position in the list is the retriever's ranking."""
    item_map = _read("item_id_map.parquet")
    inter = interactions_with_item_idx(_read("interactions_clean.parquet"), item_map)
    seq = _read("sequences_test.parquet").head(20)
    hists = [[int(x) for x in str(h).split()][-15:] for h in seq["history_idx"]]
    knn = ItemKNNRetriever(n_items=len(item_map)).fit(inter[inter["invoice_date"] < seq["ts"].min()])
    scores = knn.score(hists)
    top = knn.retrieve(hists, k=20)
    for i in range(len(hists)):
        ordered = scores[i][top[i]]
        assert np.all(np.diff(ordered) <= 1e-6), "retrieved candidates are not sorted best-first"


def test_retrieval_is_time_aware():
    """Candidates for a query must not be built from purchases at or after it."""
    item_map = _read("item_id_map.parquet")
    inter = interactions_with_item_idx(_read("interactions_clean.parquet"), item_map)
    seq = _read("sequences_test.parquet").head(5)
    hists = [[int(x) for x in str(h).split()][-15:] for h in seq["history_idx"]]
    n_items = len(item_map)

    early = time_aware_topk(inter, seq["ts"].tolist(), hists, n_items, k=10)
    # Poisoning the future must not change the result.
    future = inter.copy()
    future["invoice_date"] = future["invoice_date"] + pd.Timedelta(days=3650)
    poisoned = time_aware_topk(pd.concat([inter, future]), seq["ts"].tolist(), hists, n_items, k=10)
    assert np.array_equal(early, poisoned), "retrieval leaked purchases dated at or after the query"


def test_reranker_features_expose_retrieval_rank():
    """Without retr_rr the reranker cannot see the first-stage ordering."""
    fb = RankingFeatureBuilder(embedding_dim=4, max_history_length=3, hard_negatives=False,
                               n_negatives_per_query=None)
    item_emb = np.eye(6, 4, dtype=np.float32)
    df = pd.DataFrame({"history_idx": ["0 1"], "pos_item_idx": [3], "cands": ["2 3 4"]})
    feats = fb.build_features(df, np.zeros((1, 4), np.float32), item_emb,
                              np.zeros(6, np.float32), np.zeros(6, np.float32), device="cpu")
    assert "retr_rr" in feats
    rr = feats["retr_rr"][0]
    assert np.all(np.diff(rr) < 0), "retr_rr must decrease with candidate position"
    assert feats["label"][0].tolist() == [0.0, 1.0, 0.0]


def test_item_side_features_are_not_all_zero():
    """The reranker was once trained and served with zeroed pop and price_z."""
    item_map = _read("item_id_map.parquet")
    pop, price = build_item_side_features(_read("sequences_train.parquet"), _read("items_clean.parquet"),
                                          item_map, n_items=len(item_map))
    assert (pop > 0).sum() > 50, "popularity feature is empty"
    assert (price != 0).sum() > 50, "price_z feature is empty"


def test_metrics_agree_with_hand_computed_values():
    preds = [[7, 3, 9], [1, 2, 3]]
    gt = [3, 9]
    assert recall_at_k(preds, gt, 3) == pytest.approx(0.5)
    # hit at rank 2 in the first list, miss in the second
    assert ndcg_at_k(preds, gt, 3) == pytest.approx(0.5 / np.log2(3))
