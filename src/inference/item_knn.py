"""
Item-based nearest-neighbour retriever.

Scores an item for a customer as the recency-weighted sum of cosine
similarities between that item and the items in the customer's purchase
history, where similarity is computed over the customer-item purchase matrix.
No neural weights, no training loop; fitting is one sparse matrix product and
takes well under a second on this dataset.

Self-similarity is kept (an item is maximally similar to itself), so an item
the customer bought recently scores highly. About 9% of next purchases in this
dataset are repeats, and dropping the diagonal costs recall.

Defaults (recency_decay=0.5, max_history_length=15) were chosen on the
validation split only; see scripts/evaluate.py for test numbers.

On the Jarir data this beats the Two-Tower retriever by a wide margin, which
is the usual outcome for a catalogue of a few thousand purchase lines.
"""

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from ..utils.logging import get_logger

logger = get_logger(__name__)


class ItemKNNRetriever:
    """
    Item-item cosine similarity retriever.

    Args:
        n_items: Size of the item index space (item_idx in [0, n_items)).
        recency_decay: Weight of the i-th most recent history item is
            recency_decay ** i. 1.0 gives every history item equal weight.
        max_history_length: Only the most recent items of the history are used.
    """

    def __init__(self, n_items: int, recency_decay: float = 0.5, max_history_length: int = 15):
        self.n_items = n_items
        self.recency_decay = recency_decay
        self.max_history_length = max_history_length
        self.similarity: Optional[np.ndarray] = None

    def fit(self, interactions: pd.DataFrame) -> "ItemKNNRetriever":
        """
        Fit from a purchase table with columns customer_id and item_idx.

        Every (customer, item) pair counts once; repeat purchases of the same
        item by the same customer do not increase the weight.
        """
        if len(interactions) == 0:
            self.similarity = np.zeros((self.n_items, self.n_items), dtype=np.float32)
            return self
        users = interactions["customer_id"].astype("category").cat.codes.to_numpy()
        items = interactions["item_idx"].astype(int).to_numpy()
        mat = csr_matrix(
            (np.ones(len(items), dtype=np.float32), (users, items)),
            shape=(int(users.max()) + 1, self.n_items),
        )
        mat.data[:] = 1.0  # binarise duplicates
        self.similarity = cosine_similarity(mat.T, dense_output=True).astype(np.float32)
        return self

    def _profile(self, history: Sequence[int]) -> np.ndarray:
        prof = np.zeros(self.n_items, dtype=np.float32)
        recent = list(history)[-self.max_history_length:]
        for age, item in enumerate(reversed(recent)):
            if 0 <= item < self.n_items:
                prof[item] += self.recency_decay ** age
        return prof

    def score(self, histories: Iterable[Sequence[int]]) -> np.ndarray:
        """Score every item for every history. Returns an (n_queries, n_items) array."""
        if self.similarity is None:
            raise RuntimeError("ItemKNNRetriever.fit must be called first")
        profiles = np.stack([self._profile(h) for h in histories]).astype(np.float32)
        return profiles @ self.similarity

    def retrieve(self, histories: Iterable[Sequence[int]], k: int) -> np.ndarray:
        """Top-k item indices per history, best first. Returns an (n_queries, k) array."""
        scores = self.score(histories)
        k = min(k, self.n_items)
        part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1, kind="stable")
        return np.take_along_axis(part, order, axis=1)


def interactions_with_item_idx(interactions: pd.DataFrame, item_map: pd.DataFrame) -> pd.DataFrame:
    """Attach item_idx to interactions_clean rows via the item id map."""
    return interactions.merge(item_map[["stock_code", "item_idx"]], on="stock_code", how="inner")


def time_aware_topk(
    interactions: pd.DataFrame,
    timestamps: Sequence,
    histories: List[List[int]],
    n_items: int,
    k: int,
    **knn_kwargs,
) -> np.ndarray:
    """
    Leak-free retrieval for a batch of dated queries: for every distinct query
    date, fit ItemKNN on purchases strictly before that date and retrieve top-k
    for the queries on that date.

    Args:
        interactions: purchase rows with invoice_date, customer_id, item_idx.
        timestamps: one query timestamp per history.
        histories: one item-index list per query.
    Returns:
        (n_queries, k) array of item indices, best first.
    """
    ts = pd.Series(pd.to_datetime(list(timestamps))).reset_index(drop=True)
    out = np.zeros((len(histories), min(k, n_items)), dtype=np.int64)
    groups: Dict = ts.groupby(ts).indices
    dates = interactions["invoice_date"]
    for t, pos in groups.items():
        knn = ItemKNNRetriever(n_items, **knn_kwargs).fit(interactions[dates < t])
        out[pos] = knn.retrieve([histories[i] for i in pos], k)
    logger.info(f"Time-aware ItemKNN retrieval: {len(histories)} queries over {len(groups)} dates")
    return out
