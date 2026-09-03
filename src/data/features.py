"""
Feature engineering utilities for the recommendation system.

This module provides functions for creating user-item interaction sequences,
generating features for ranking models, and handling candidate generation.
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..utils.io import write_parquet
from ..utils.logging import get_logger

logger = get_logger(__name__)


class SequenceBuilder:
    """
    Builder for creating user interaction sequences from transaction data.
    
    Converts transaction data into sequences suitable for training
    recommendation models, with support for temporal splitting.
    """
    
    def __init__(
        self,
        max_history_length: int = 15,
        min_history_length: int = 2,
        train_split_quantile: float = 0.80,
        val_split_quantile: float = 0.90,
        random_negatives: bool = False,
        n_negatives_train: int = 50,
        n_negatives_val: int = 100,
        random_seed: int = 42,
    ):
        """
        Initialize the sequence builder.
        
        Args:
            max_history_length: Maximum length of user history.
            min_history_length: Minimum length of user history.
            train_split_quantile: Quantile for train/val split.
            val_split_quantile: Quantile for val/test split.
            random_negatives: Whether to add random negative samples.
            n_negatives_train: Number of negatives per training sample.
            n_negatives_val: Number of negatives per validation/test sample.
            random_seed: Random seed for reproducibility.
        """
        self.max_history_length = max_history_length
        self.min_history_length = min_history_length
        self.train_split_quantile = train_split_quantile
        self.val_split_quantile = val_split_quantile
        self.random_negatives = random_negatives
        self.n_negatives_train = n_negatives_train
        self.n_negatives_val = n_negatives_val
        self.random_seed = random_seed
        
        # Initialize random number generator
        self.rng = np.random.default_rng(random_seed)
    
    def build_sequences(
        self,
        interactions_df: pd.DataFrame,
        item_map: pd.DataFrame,
        customer_map: pd.DataFrame,
        output_dir: Union[str, Path],
    ) -> Dict[str, pd.DataFrame]:
        """
        Build user interaction sequences from transaction data.
        
        Args:
            interactions_df: Clean interactions DataFrame.
            item_map: Item ID mapping DataFrame.
            customer_map: Customer ID mapping DataFrame.
            output_dir: Directory to save sequence files.
            
        Returns:
            Dictionary with train/val/test sequence DataFrames.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Building interaction sequences")
        
        # Merge with ID mappings
        interactions_with_ids = self._add_id_mappings(
            interactions_df, item_map, customer_map
        )
        
        # Sort by customer and date
        interactions_with_ids = interactions_with_ids.sort_values(
            ['customer_id', 'invoice_date']
        ).reset_index(drop=True)
        
        # Determine split timestamps
        timestamps = interactions_with_ids['invoice_date']
        train_cutoff = timestamps.quantile(self.train_split_quantile)
        val_cutoff = timestamps.quantile(self.val_split_quantile)
        
        logger.info(f"Train cutoff: {train_cutoff}")
        logger.info(f"Validation cutoff: {val_cutoff}")
        
        # Build sequences for each split
        sequences = {}
        
        sequences['train'] = self._build_sequences_for_split(
            interactions_with_ids,
            start_time=timestamps.min(),
            end_time=train_cutoff,
            split_name='train'
        )
        
        sequences['val'] = self._build_sequences_for_split(
            interactions_with_ids,
            start_time=train_cutoff,
            end_time=val_cutoff,
            split_name='val'
        )
        
        sequences['test'] = self._build_sequences_for_split(
            interactions_with_ids,
            start_time=val_cutoff,
            end_time=timestamps.max() + pd.Timedelta(seconds=1),
            split_name='test'
        )
        
        # Add random negatives if requested
        if self.random_negatives:
            item_pool = item_map['item_idx'].values
            
            sequences['train'] = self._add_random_negatives(
                sequences['train'], item_pool, self.n_negatives_train
            )
            sequences['val'] = self._add_random_negatives(
                sequences['val'], item_pool, self.n_negatives_val
            )
            sequences['test'] = self._add_random_negatives(
                sequences['test'], item_pool, self.n_negatives_val
            )
        
        # Save sequences
        for split_name, seq_df in sequences.items():
            output_path = output_dir / f'sequences_{split_name}.parquet'
            write_parquet(seq_df, output_path)
            logger.info(f"Saved {len(seq_df)} {split_name} sequences to {output_path}")
        
        return sequences
    
    def _add_id_mappings(
        self,
        interactions_df: pd.DataFrame,
        item_map: pd.DataFrame,
        customer_map: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add item and customer ID mappings to interactions."""
        # Merge with item mapping
        interactions_with_ids = interactions_df.merge(
            item_map, on='stock_code', how='inner'
        )
        
        # Merge with customer mapping
        interactions_with_ids = interactions_with_ids.merge(
            customer_map, on='customer_id', how='inner'
        )
        
        logger.info(f"Mapped {len(interactions_with_ids)} interactions to IDs")
        return interactions_with_ids
    
    def _build_sequences_for_split(
        self,
        interactions_df: pd.DataFrame,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        split_name: str,
    ) -> pd.DataFrame:
        """Build sequences for a specific time split."""
        sequences = []
        
        for customer_id, group in interactions_df.groupby('customer_id', sort=False):
            # Sort by time
            group = group.sort_values('invoice_date')
            
            timestamps = group['invoice_date'].values
            item_indices = group['item_idx'].values
            user_indices = group['user_idx'].values
            country = group['country'].iloc[-1]
            
            # Build sequences for this user
            for i in range(1, len(item_indices)):
                # Check if target interaction is in the split time range
                if not (start_time <= timestamps[i] < end_time):
                    continue
                
                # Extract history
                history_start = max(0, i - self.max_history_length)
                history = item_indices[history_start:i]
                
                # Skip if history is too short
                if len(history) < self.min_history_length:
                    continue
                
                # Create sequence record
                sequence = {
                    'customer_id': customer_id,
                    'user_idx': user_indices[i],
                    'ts': timestamps[i],
                    'history_idx': ' '.join(map(str, history)),
                    'pos_item_idx': int(item_indices[i]),
                    'country': country,
                }
                
                sequences.append(sequence)
        
        seq_df = pd.DataFrame(sequences)
        logger.info(f"Built {len(seq_df)} sequences for {split_name} split")
        
        return seq_df
    
    def _add_random_negatives(
        self,
        sequences_df: pd.DataFrame,
        item_pool: np.ndarray,
        n_negatives: int,
    ) -> pd.DataFrame:
        """Add random negative samples to sequences."""
        sequences_df = sequences_df.copy()
        negative_samples = []
        
        for _, row in sequences_df.iterrows():
            # Parse history and positive item
            history = [int(x) for x in row['history_idx'].split()]
            positive_item = int(row['pos_item_idx'])
            
            # Items to exclude from negatives
            forbidden_items = set(history + [positive_item])
            
            # Sample negatives
            available_items = item_pool[~np.isin(item_pool, list(forbidden_items))]
            
            if len(available_items) == 0:
                negative_samples.append('')
            else:
                n_sample = min(n_negatives, len(available_items))
                sampled_negatives = self.rng.choice(
                    available_items, size=n_sample, replace=False
                )
                negative_samples.append(' '.join(map(str, sampled_negatives)))
        
        sequences_df['neg_idx'] = negative_samples
        logger.info(f"Added random negatives to {len(sequences_df)} sequences")
        
        return sequences_df


def create_id_mappings(
    interactions_df: pd.DataFrame,
    output_dir: Union[str, Path],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create item and customer ID mappings from interactions.
    
    Args:
        interactions_df: Clean interactions DataFrame.
        output_dir: Directory to save mapping files.
        
    Returns:
        Tuple of (item_map, customer_map) DataFrames.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Creating ID mappings")
    
    # Create item mapping
    unique_items = interactions_df['stock_code'].unique()
    item_map = pd.DataFrame({
        'stock_code': unique_items,
        'item_idx': np.arange(len(unique_items), dtype=np.int64)
    })
    
    # Create customer mapping
    unique_customers = interactions_df['customer_id'].unique()
    customer_map = pd.DataFrame({
        'customer_id': unique_customers,
        'user_idx': np.arange(len(unique_customers), dtype=np.int64)
    })
    
    # Save mappings
    item_map_path = output_dir / 'item_id_map.parquet'
    customer_map_path = output_dir / 'customer_id_map.parquet'
    
    write_parquet(item_map, item_map_path)
    write_parquet(customer_map, customer_map_path)
    
    logger.info(f"Created {len(item_map)} item mappings")
    logger.info(f"Created {len(customer_map)} customer mappings")
    
    return item_map, customer_map


def build_item_side_features(
    seq_train: pd.DataFrame,
    items_clean: pd.DataFrame,
    item_map: pd.DataFrame,
    n_items: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Item-level feature vectors used by the reranker: popularity and price z-score.

    - pop: number of times each item is the positive in the training sequences,
      min-max normalised to [0, 1]; 0 for items never purchased in training.
    - price_z: z-score of items_clean.price_median over the catalogue; 0 if missing.

    The same function must be used at training and at inference time so the
    reranker sees the features it was trained on.

    Returns:
        Tuple of (pop, price_z), each a float32 array of length n_items indexed by item_idx.
    """
    pop_counts = seq_train['pos_item_idx'].value_counts()
    pop = np.zeros(n_items, dtype=np.float32)
    if len(pop_counts) > 0:
        pop_norm = (pop_counts - pop_counts.min()) / (pop_counts.max() - pop_counts.min() + 1e-9)
        pop[pop_counts.index.values.astype(int)] = pop_norm.values.astype(np.float32)

    price_z = np.zeros(n_items, dtype=np.float32)
    if 'price_median' in items_clean.columns:
        m = items_clean[['stock_code', 'price_median']].dropna().merge(item_map, on='stock_code', how='inner')
        if len(m) > 0:
            mu, sigma = m['price_median'].mean(), m['price_median'].std() + 1e-6
            z = ((m['price_median'] - mu) / sigma).astype(float)
            price_z[m['item_idx'].astype(int).values] = z.values.astype(np.float32)

    return pop, price_z


def positive_in_candidates(candidates_df: pd.DataFrame) -> np.ndarray:
    """Boolean mask: True where the query's positive item appears in its candidate list."""
    return np.array(
        [str(int(p)) in c.split() for p, c in zip(candidates_df['pos_item_idx'], candidates_df['cands'])],
        dtype=bool,
    )


class RankingFeatureBuilder:
    """
    Builder for ranking features used in the MLP reranker.
    
    Creates features from user embeddings, item embeddings, and metadata
    for training the ranking model.
    """
    
    def __init__(
        self,
        embedding_dim: int = 256,
        max_history_length: int = 15,
        hard_negatives: bool = True,
        n_negatives_per_query: int = 20,
    ):
        """
        Initialize the ranking feature builder.
        
        Args:
            embedding_dim: Dimension of user/item embeddings.
            max_history_length: Maximum length of user history.
            hard_negatives: Whether to use hard negative sampling.
            n_negatives_per_query: Number of negatives per query.
        """
        self.embedding_dim = embedding_dim
        self.max_history_length = max_history_length
        self.hard_negatives = hard_negatives
        self.n_negatives_per_query = n_negatives_per_query
    
    def build_features(
        self,
        candidates_df: pd.DataFrame,
        user_embeddings: np.ndarray,
        item_embeddings: np.ndarray,
        popularity_scores: Optional[np.ndarray] = None,
        price_features: Optional[np.ndarray] = None,
        device: str = 'cpu',
        batch_size: int = 1024,
    ) -> Dict[str, np.ndarray]:
        """
        Build ranking features from candidates and embeddings.
        
        Args:
            candidates_df: DataFrame with candidate lists.
            user_embeddings: User embedding matrix.
            item_embeddings: Item embedding matrix.
            popularity_scores: Optional popularity scores for items.
            price_features: Optional price features for items.
            device: Device for tensor computations.
            batch_size: Batch size for processing.
            
        Returns:
            Dictionary of feature arrays.
        """
        logger.info(f"Building ranking features for {len(candidates_df)} queries")
        
        # Convert to tensors
        user_emb_tensor = torch.from_numpy(user_embeddings).to(device)
        item_emb_tensor = torch.from_numpy(item_embeddings).to(device)
        
        # Initialize feature lists
        features = {
            'dot_uv': [],
            'max_sim_recent': [],
            'pop': [],
            'hist_len': [],
            'price_z': [],
            'retr_rr': [],
            'label': [],
            'item_idx': [],
        }
        
        # Process in batches
        for i in range(0, len(candidates_df), batch_size):
            batch_df = candidates_df.iloc[i:i + batch_size]
            batch_features = self._process_batch(
                batch_df,
                user_emb_tensor,
                item_emb_tensor,
                popularity_scores,
                price_features,
                device,
            )
            
            # Accumulate features
            for key, values in batch_features.items():
                features[key].append(values)
        
        # Concatenate all batches
        final_features = {}
        for key, value_list in features.items():
            if value_list:
                final_features[key] = np.concatenate(value_list, axis=0)
            else:
                final_features[key] = np.array([])
        
        logger.info(f"Built features with shape: {final_features['dot_uv'].shape}")
        return final_features
    
    def _process_batch(
        self,
        batch_df: pd.DataFrame,
        user_emb_tensor: torch.Tensor,
        item_emb_tensor: torch.Tensor,
        popularity_scores: Optional[np.ndarray],
        price_features: Optional[np.ndarray],
        device: str,
    ) -> Dict[str, np.ndarray]:
        """Process a single batch of candidates."""
        with torch.no_grad():
            # Parse batch data
            history_tensors = []
            candidate_tensors = []
            positive_items = []
            
            for _, row in batch_df.iterrows():
                # Parse history
                history = [int(x) for x in row['history_idx'].split()]
                if len(history) > self.max_history_length:
                    history = history[-self.max_history_length:]
                
                # Pad history to fixed length
                padded_history = [-1] * self.max_history_length
                padded_history[-len(history):] = history
                history_tensors.append(padded_history)
                
                # Parse candidates
                candidates = [int(x) for x in row['cands'].split()]
                candidate_tensors.append(candidates)
                
                # Store positive item
                positive_items.append(int(row['pos_item_idx']))
            
            # Convert to tensors
            H = torch.tensor(history_tensors, dtype=torch.long, device=device)
            C = torch.tensor(candidate_tensors, dtype=torch.long, device=device)
            P = torch.tensor(positive_items, dtype=torch.long, device=device)
            
            # Compute user vectors from history
            U = self._compute_user_vectors_from_history(H, item_emb_tensor)
            
            # Compute features
            batch_features = self._compute_features(
                U, H, C, P, item_emb_tensor, popularity_scores, price_features
            )
            
            # Apply negative sampling if configured
            if self.n_negatives_per_query is not None:
                batch_features = self._select_negatives(batch_features)
            
            # Convert to numpy
            numpy_features = {}
            for key, tensor in batch_features.items():
                numpy_features[key] = tensor.cpu().numpy()
            
            return numpy_features
    
    def _compute_user_vectors_from_history(
        self,
        history_tensor: torch.Tensor,
        item_emb_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Compute user vectors by averaging history item embeddings."""
        B, L = history_tensor.shape
        
        # Handle negative indices (padding)
        safe_indices = history_tensor.clamp(min=0)
        
        # Get history embeddings
        H_emb = item_emb_tensor.index_select(0, safe_indices.view(-1)).view(B, L, -1)
        
        # Create mask for valid history items
        mask = (history_tensor >= 0).float().unsqueeze(-1)
        
        # Average valid embeddings
        masked_embeddings = H_emb * mask
        user_vectors = masked_embeddings.sum(1) / mask.sum(1).clamp_min(1e-6)
        
        # Normalize user vectors
        return F.normalize(user_vectors, p=2, dim=1)
    
    def _compute_features(
        self,
        user_vectors: torch.Tensor,
        history_tensor: torch.Tensor,
        candidate_tensor: torch.Tensor,
        positive_tensor: torch.Tensor,
        item_emb_tensor: torch.Tensor,
        popularity_scores: Optional[np.ndarray],
        price_features: Optional[np.ndarray],
    ) -> Dict[str, torch.Tensor]:
        """Compute all ranking features."""
        B, K = candidate_tensor.shape
        L = history_tensor.size(1)
        d = user_vectors.size(1)
        
        # Get candidate embeddings
        candidate_emb = item_emb_tensor.index_select(0, candidate_tensor.view(-1)).view(B, K, d)
        
        # Feature 1: Dot product between user and candidate vectors
        dot_uv = (user_vectors.unsqueeze(1) * candidate_emb).sum(-1)
        
        # Feature 2: Maximum similarity with recent history
        history_emb = item_emb_tensor.index_select(0, history_tensor.clamp(min=0).view(-1)).view(B, L, d)
        history_emb = F.normalize(history_emb, p=2, dim=-1)
        candidate_emb_norm = F.normalize(candidate_emb, p=2, dim=-1)
        
        # Compute similarities
        similarities = torch.matmul(history_emb, candidate_emb_norm.transpose(1, 2))
        
        # Mask invalid history positions
        history_mask = (history_tensor >= 0).unsqueeze(-1).float()
        similarities = similarities + (history_mask - 1.0) * 1e9
        
        max_sim_recent = similarities.max(dim=1).values
        
        # Feature 3: Popularity scores
        if popularity_scores is not None:
            pop_tensor = torch.from_numpy(popularity_scores).to(candidate_tensor.device)
            pop = pop_tensor.index_select(0, candidate_tensor.view(-1)).view(B, K)
        else:
            pop = torch.zeros((B, K), device=candidate_tensor.device)
        
        # Feature 4: History length (normalized)
        hist_len = (history_tensor >= 0).float().sum(1) / float(self.max_history_length)
        hist_len = hist_len.unsqueeze(1).expand(B, K)
        
        # Feature 5: Price features
        if price_features is not None:
            price_tensor = torch.from_numpy(price_features).to(candidate_tensor.device)
            price_z = price_tensor.index_select(0, candidate_tensor.view(-1)).view(B, K)
        else:
            price_z = torch.zeros((B, K), device=candidate_tensor.device)
        
        # Feature 6: first-stage retrieval rank, as a discounted reciprocal rank.
        # Candidate lists are stored best-first, so a candidate's position in the
        # list is the retriever's ranking of it. Without this the reranker cannot
        # see the first-stage score and reorders candidates blind, which loses
        # more ranking quality than its own five features recover.
        ranks = torch.arange(K, device=candidate_tensor.device, dtype=torch.float32)
        retr_rr = (1.0 / torch.log2(ranks + 2.0)).unsqueeze(0).expand(B, K).contiguous()

        # Labels: 1 for positive items, 0 for negatives
        labels = (candidate_tensor == positive_tensor.view(-1, 1)).float()
        
        return {
            'dot_uv': dot_uv,
            'max_sim_recent': max_sim_recent,
            'pop': pop,
            'hist_len': hist_len,
            'price_z': price_z,
            'retr_rr': retr_rr,
            'label': labels,
            'item_idx': candidate_tensor.float(),
        }
    
    def _select_negatives(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Select negatives using hard or random sampling."""
        B, K = features['label'].shape
        
        # Find positive items
        positive_mask = features['label'] > 0.5
        positive_indices = positive_mask.nonzero(as_tuple=True)[1].view(B, 1)
        
        # Select negatives
        if self.hard_negatives:
            # Hard negatives: highest scoring negatives
            negative_scores = features['dot_uv'].clone()
            negative_scores[positive_mask] = -1e9
            
            n_negatives = min(self.n_negatives_per_query, K - 1)
            _, negative_indices = torch.topk(negative_scores, k=n_negatives, dim=1)
        else:
            # Random negatives
            random_scores = torch.rand_like(features['dot_uv'])
            random_scores[positive_mask] = 1e9
            
            n_negatives = min(self.n_negatives_per_query, K - 1)
            _, negative_indices = torch.topk(-random_scores, k=n_negatives, dim=1)
        
        # Combine positive and negative indices
        keep_indices = torch.cat([positive_indices, negative_indices], dim=1)
        
        # Select features
        selected_features = {}
        for key, tensor in features.items():
            selected_features[key] = torch.gather(tensor, 1, keep_indices)
        
        return selected_features
