#!/usr/bin/env python3
"""
Train MLP reranker on retrieved candidates and engineered features.

This script loads candidates (built from the retriever) and item/user embeddings,
builds training features, trains an MLP ranker, and persists the best checkpoint.

Example usage:
    python scripts/train_reranker.py --config configs/reranker.yaml \
      --data-dir data/processed/jarir --model-dir models/reranker
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure src is importable when running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.dataset import load_id_mappings
from src.data.features import RankingFeatureBuilder, build_item_side_features, positive_in_candidates
from src.models.reranker.mlp_ranker import MLPRanker
from src.training.trainer import RankingTrainer, setup_training
from src.training.callbacks import EarlyStopping, ModelCheckpoint, MetricsLogger
from src.utils.config import load_config
from src.utils.io import read_numpy, write_json
from src.utils.logging import setup_logging
from src.utils.seed import set_seed, get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MLP reranker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True, help="Path to reranker YAML config")
    parser.add_argument("--data-dir", type=str, help="Processed data dir (overrides config)")
    parser.add_argument("--model-dir", type=str, help="Output dir for ranker artifacts")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], help="Device override")
    parser.add_argument("--epochs", type=int, help="Epochs override")
    parser.add_argument("--batch-size", type=int, help="Batch size override")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--seed", type=int, help="Random seed override")
    return parser.parse_args()


def load_candidates(data_dir: Path) -> dict[str, pd.DataFrame]:
    cand = {}
    for split in ["train", "val", "test"]:
        p = data_dir / f"candidates_{split}.parquet"
        if p.exists():
            cand[split] = pd.read_parquet(p)
        else:
            logging.getLogger(__name__).warning(f"Missing candidates file for {split}: {p}")
    return cand


def make_feature_tensors(
    df: pd.DataFrame,
    feature_builder: RankingFeatureBuilder,
    user_emb: np.ndarray,
    item_emb: np.ndarray,
    pop: np.ndarray | None,
    price: np.ndarray | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    feats = feature_builder.build_features(
        df, user_emb, item_emb, pop, price, device=str(device), batch_size=1024,
    )
    cols = ["dot_uv", "max_sim_recent", "pop", "hist_len", "price_z"]
    # Stack features into (Q, K, F) then flatten to (Q*K, F)
    X_np = np.stack([feats[c] for c in cols], axis=-1)
    X_np = X_np.reshape(-1, X_np.shape[-1])
    y_np = feats["label"].reshape(-1)
    X = torch.from_numpy(X_np).to(device)
    y = torch.from_numpy(y_np).to(device)
    return X.float(), y.float()


def _recall_from_scores(scores: np.ndarray, cand_lists: list[list[int]], pos_list: list[int], topk: int) -> float:
    B, K = scores.shape
    hits = 0
    for bi in range(B):
        order = np.argsort(scores[bi])[::-1][:min(topk, K)]
        preds = [cand_lists[bi][j] for j in order]
        hits += int(pos_list[bi] in preds)
    return float(hits / max(B, 1))


class ValRecallCallback:
    def __init__(self, model: MLPRanker, X_val_flat: torch.Tensor, y_val_flat: torch.Tensor,
                 batch_count: int, k_per_q: int, device: torch.device, topk: int = 10):
        self.model = model
        self.X_val_flat = X_val_flat
        self.y_val_flat = y_val_flat
        self.batch_count = batch_count
        self.k_per_q = k_per_q
        self.device = device
        self.topk = topk

    def on_train_begin(self, logs=None):
        return None

    def on_train_end(self, logs=None):
        return None

    def on_epoch_begin(self, epoch: int, logs=None):
        return None

    def on_epoch_end(self, epoch: int, logs=None) -> bool:
        with torch.no_grad():
            scores = self.model(self.X_val_flat).view(self.batch_count, self.k_per_q).cpu()
        # Labels shaped (B,K2) with exactly one positive per row
        y = self.y_val_flat.view(self.batch_count, self.k_per_q).cpu()
        topk_idx = torch.topk(scores, k=min(self.topk, self.k_per_q), dim=1).indices
        # For each row, check if any top-k position corresponds to label==1
        rows = torch.arange(self.batch_count).unsqueeze(1)
        hits = (y[rows, topk_idx] > 0.5).any(dim=1).float().mean().item()
        rec = float(hits)
        logging.getLogger(__name__).info(f"Epoch {epoch+1}: val_recall@{self.topk}={rec:.4f}")
        # Do not stop training
        return False


def main() -> None:
    args = parse_args()
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)

    cfg = load_config(args.config)
    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    if args.model_dir:
        cfg["model_dir"] = args.model_dir
    if args.epochs:
        cfg.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size:
        cfg.setdefault("training", {})["batch_size"] = args.batch_size
    if args.seed:
        cfg["random_seed"] = args.seed

    seed = int(cfg.get("random_seed", 42))
    set_seed(seed)

    device_str = cfg.get("hardware", {}).get("device", "auto")
    if args.device:
        device_str = args.device
    device = torch.device(get_device() if device_str == "auto" else device_str)
    logger.info(f"Using device: {device}")

    data_dir = Path(cfg.get("data_dir", "models/data"))
    model_dir = Path(cfg.get("model_dir", "models/reranker"))
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load embeddings from retriever stage (try models/retriever, fallback to data/processed/jarir)
    retriever_dir_primary = Path("models/retriever")
    retriever_dir_fallback = Path("data/processed/jarir")
    user_path_primary = retriever_dir_primary / "user_embeddings.npy"
    item_path_primary = retriever_dir_primary / "item_embeddings.npy"
    if user_path_primary.exists() and item_path_primary.exists():
        user_emb = read_numpy(user_path_primary)
        item_emb = read_numpy(item_path_primary)
    else:
        user_emb = read_numpy(retriever_dir_fallback / "user_embeddings.npy")
        item_emb = read_numpy(retriever_dir_fallback / "item_embeddings.npy")
    logger.info(f"Loaded embeddings: users {user_emb.shape}, items {item_emb.shape}")

    # Item-side features. The same function is used at inference time (app, evaluate.py)
    # so the model is scored on the features it was trained with.
    item_map, _ = load_id_mappings(data_dir)
    seq_train = pd.read_parquet(data_dir / "sequences_train.parquet")
    items_clean = pd.read_parquet(data_dir / "items_clean.parquet")
    pop, price = build_item_side_features(seq_train, items_clean, item_map, n_items=item_emb.shape[0])
    logger.info(f"Item features: pop non-zero for {int((pop > 0).sum())} items, price_z for {int((price != 0).sum())} items")

    # Load candidates
    cand = load_candidates(data_dir)
    if "train" not in cand or "val" not in cand:
        raise FileNotFoundError("candidates_train.parquet and candidates_val.parquet are required")

    # A pointwise ranker needs one positive per list to learn from. Keep only the
    # queries whose positive the retriever actually returned; never insert it.
    for split in list(cand.keys()):
        mask = positive_in_candidates(cand[split])
        logger.info(f"{split}: positive retrieved in {int(mask.sum())}/{len(mask)} candidate lists; keeping those")
        cand[split] = cand[split][mask].reset_index(drop=True)
    if len(cand["train"]) == 0 or len(cand["val"]) == 0:
        raise ValueError("No candidate lists contain the positive item; check the retriever")

    # Feature builder
    fb_cfg = cfg.get("features", {})
    feat_builder = RankingFeatureBuilder(
        embedding_dim=item_emb.shape[1],
        max_history_length=fb_cfg.get("max_history_length", 15),
        hard_negatives=fb_cfg.get("hard_negatives", True),
        n_negatives_per_query=fb_cfg.get("neg_per_query", 20),
    )

    # Build tensors
    logger.info("Building training features (this may take a moment)...")
    X_tr, y_tr = make_feature_tensors(cand["train"], feat_builder, user_emb, item_emb, pop, price, device)
    # For validation, build full-candidate features (no negative subsampling) to match notebook metrics
    val_feat_builder = RankingFeatureBuilder(
        embedding_dim=item_emb.shape[1],
        max_history_length=fb_cfg.get("max_history_length", 15),
        hard_negatives=False,
        n_negatives_per_query=None,
    )
    feats_val = val_feat_builder.build_features(cand["val"], user_emb, item_emb, pop, price, device=str(device), batch_size=1024)
    cols = ["dot_uv", "max_sim_recent", "pop", "hist_len", "price_z"]
    X_va_np = np.stack([feats_val[c] for c in cols], axis=-1)
    X_va_np = X_va_np.reshape(-1, X_va_np.shape[-1])
    y_va_np = feats_val["label"].reshape(-1)
    X_va = torch.from_numpy(X_va_np).to(device).float()
    y_va = torch.from_numpy(y_va_np).to(device).float()

    # Precompute structures for fast Recall@K during training (avoid per-epoch feature building)
    # Infer K per query from the validation labels size
    # y_va is flattened (B*K,), so k_per_q = y_va.numel() // B
    batch_count_va = len(cand["val"]) if len(cand["val"]) > 0 else 0
    k_per_q_va = int(y_va.numel() // max(batch_count_va, 1)) if batch_count_va > 0 else 1

    # Prepare loaders
    # Batch size (notebook default)
    bs = int(cfg.get("training", {}).get("batch_size", 2048))
    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=bs, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_va, y_va), batch_size=bs, shuffle=False)

    # Model
    model_cfg = cfg.get("model", {})
    model = MLPRanker(
        input_dim=model_cfg.get("input_dim", 5),
        hidden_dims=model_cfg.get("hidden_dims", [384, 384, 192]),
        dropout=model_cfg.get("dropout", 0.3),
        feature_dropout_indices=(model_cfg.get("feature_dropout", {}).get("feature_indices", [0]) if model_cfg.get("feature_dropout", {}).get("enabled", True) else None),
        feature_dropout_prob=model_cfg.get("feature_dropout", {}).get("dropout_prob", 0.3),
        use_residual=model_cfg.get("use_residual", True),
        use_layer_norm=model_cfg.get("use_layer_norm", True),
        activation=model_cfg.get("activation", "relu"),
    ).to(device)

    # Training setup
    optimizer, scheduler, loss_fn = setup_training(model, cfg, device)
    trainer = RankingTrainer(
        model=model,
        optimizer=optimizer,
        loss_function=loss_fn,
        device=device,
        scheduler=scheduler,
        gradient_clip_value=cfg.get("training", {}).get("gradient_clip_value", 1.0),
        accumulation_steps=1,
        mixed_precision=(device.type == "cuda" and cfg.get("training", {}).get("mixed_precision", True)),
    )

    # Callbacks
    # Notebook default: save best to models/reranker/best_ranker.pt
    ckpt = ModelCheckpoint(filepath=model_dir / "best_ranker.pt", monitor="val_loss", mode="min", save_best_only=True, save_weights_only=True)
    ckpt.set_model(model)
    trainer.add_callback(ckpt)
    trainer.add_callback(EarlyStopping(monitor="val_loss", patience=cfg.get("training", {}).get("patience", 5), mode="min"))
    trainer.add_callback(MetricsLogger(save_path=model_dir / "training_metrics.json", save_frequency=5))
    # Add validation Recall@K printer (fast, precomputed features)
    eval_topk = int(cfg.get("evaluation", {}).get("eval_topk", 10))
    trainer.add_callback(ValRecallCallback(model, X_va, y_va, batch_count_va, k_per_q_va, device, topk=eval_topk))

    # Train
    epochs = int(cfg.get("training", {}).get("epochs", 20))
    history = trainer.fit(train_loader=train_loader, validation_loader=val_loader, epochs=epochs, save_dir=model_dir)

    # Save final
    torch.save(model.state_dict(), model_dir / "final_ranker.pt")
    write_json({"history": history}, model_dir / "training_history.json")

    logger.info("Reranker training complete")


if __name__ == "__main__":
    main()
