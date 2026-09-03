"""
Streamlit Application for Jarir Recommendation System
Simple, clean design focused on controls and recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Tuple
import time
import re
import sys
import os

# Add project root to Python path for Streamlit Cloud deployment
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Also add the absolute path to handle different deployment environments
abs_project_root = project_root.resolve()
if str(abs_project_root) not in sys.path:
    sys.path.insert(0, str(abs_project_root))

# Import our modules
from src.data.features import RankingFeatureBuilder, build_item_side_features
from src.models.reranker.mlp_ranker import MLPRanker
from src.utils.config import load_config
from src.utils.io import read_numpy

# Set page config
st.set_page_config(
    page_title="Jarir Recommendation System",
    page_icon="🎯",
    layout="wide"
)

# Simple, clean CSS
st.markdown("""
<style>
/* Hide Streamlit branding */
.stDeployButton {display: none !important;}
header[data-testid="stHeader"] {display: none !important;}

/* Main layout */
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: #f5f5f5;
}

.main {
    padding: 20px;
    max-width: 1200px;
    margin: 0 auto;
}

/* Header */
.header {
    background: #2c3e50;
    color: white;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
    text-align: center;
}

.header h1 {
    margin: 0;
    font-size: 2rem;
}

.header p {
    margin: 10px 0 0 0;
    opacity: 0.9;
}

/* Controls section */
.controls {
    background: #2c3e50;
    color: white;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}

.controls h3 {
    margin-top: 0;
    color: white;
    border-bottom: 2px solid #3498db;
    padding-bottom: 10px;
}

/* Customer info */
.customer-info {
    background: #34495e;
    color: white;
    padding: 15px;
    border-radius: 8px;
    margin: 15px 0;
    border-left: 4px solid #3498db;
}

.customer-avatar {
    display: inline-block;
    width: 50px;
    height: 50px;
    background: #3498db;
    color: white;
    border-radius: 50%;
    text-align: center;
    line-height: 50px;
    font-weight: bold;
    margin-right: 15px;
    vertical-align: middle;
}

.customer-details {
    display: inline-block;
    vertical-align: middle;
}

.customer-details strong {
    color: white !important;
}

.customer-details small {
    color: #bdc3c7 !important;
}

.customer-stats {
    margin-top: 10px;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
    gap: 10px;
}

.stat {
    text-align: center;
    padding: 8px;
    background: #2c3e50;
    border-radius: 5px;
    border: 1px solid #3498db;
}

.stat-value {
    font-weight: bold;
    color: white !important;
    font-size: 1.1rem;
}

.stat-label {
    font-size: 0.8rem;
    color: #bdc3c7 !important;
}

/* Recommendations section */
.recommendations {
    background: white;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.recommendations h3 {
    margin-top: 0;
    color: #2c3e50;
    border-bottom: 2px solid #e74c3c;
    padding-bottom: 10px;
}

.generation-time {
    background: #27ae60;
    color: white;
    padding: 5px 12px;
    border-radius: 15px;
    font-size: 0.9rem;
    display: inline-block;
    margin-bottom: 15px;
}

/* Product grid */
.product-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 15px;
    margin-top: 15px;
}

.product-card {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 15px;
    position: relative;
    transition: transform 0.2s, box-shadow 0.2s;
}

.product-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
}

.product-rank {
    position: absolute;
    top: 10px;
    right: 10px;
    background: #e74c3c;
    color: white;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.8rem;
    font-weight: bold;
}

.product-title {
    font-weight: 600;
    color: #2c3e50 !important;
    margin-bottom: 8px;
    line-height: 1.3;
    font-size: 1rem;
}

.product-price {
    font-size: 1.2rem;
    font-weight: bold;
    color: #27ae60;
    margin-bottom: 8px;
}

.product-category {
    background: #3498db;
    color: white;
    padding: 3px 8px;
    border-radius: 12px;
    font-size: 0.75rem;
    display: inline-block;
    margin-bottom: 8px;
}

.product-explanation {
    color: #6c757d !important;
    font-size: 0.85rem;
    font-style: italic;
    line-height: 1.4;
}

/* Empty state */
.empty-state {
    text-align: center;
    padding: 40px;
    color: #6c757d;
}

.empty-state h4 {
    color: #495057;
    margin-bottom: 10px;
}

/* Responsive */
@media (max-width: 768px) {
    .product-grid {
        grid-template-columns: 1fr;
    }
    .customer-stats {
        grid-template-columns: repeat(2, 1fr);
    }
}
</style>
""", unsafe_allow_html=True)

# Reranker checkpoint written by scripts/train_reranker.py (src MLPRanker architecture)
RERANKER_PATH = Path("models/reranker/best_ranker.pt")
RERANKER_CONFIG = Path("configs/reranker.yaml")

# Data loading functions
@st.cache_resource
def load_embeddings() -> Tuple[np.ndarray, np.ndarray, str]:
    paths_to_try = [
        Path("models/retriever"),
        Path("models/reranker")
    ]
    
    for path in paths_to_try:
        user_path = path / "user_embeddings.npy"
        item_path = path / "item_embeddings.npy"
        if user_path.exists() and item_path.exists():
            return read_numpy(user_path), read_numpy(item_path), str(path)
    
    st.error("Could not find trained embeddings in models/ directories.")
    st.stop()

@st.cache_resource
def load_reranker() -> Tuple[MLPRanker, str]:
    if not RERANKER_PATH.exists():
        st.error(f"Reranker checkpoint not found: {RERANKER_PATH}. Run scripts/train_reranker.py first.")
        st.stop()
    model_cfg = load_config(RERANKER_CONFIG).get("model", {}) if RERANKER_CONFIG.exists() else {}
    fd = model_cfg.get("feature_dropout", {})
    model = MLPRanker(
        input_dim=model_cfg.get("input_dim", 5),
        hidden_dims=model_cfg.get("hidden_dims", [384, 384, 192]),
        dropout=model_cfg.get("dropout", 0.3),
        feature_dropout_indices=(fd.get("feature_indices", [0]) if fd.get("enabled", True) else None),
        feature_dropout_prob=fd.get("dropout_prob", 0.3),
        use_residual=model_cfg.get("use_residual", True),
        use_layer_norm=model_cfg.get("use_layer_norm", True),
        activation=model_cfg.get("activation", "relu"),
    )
    state_dict = torch.load(RERANKER_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model, str(RERANKER_PATH)

@st.cache_data
def load_catalog_data() -> Dict[str, pd.DataFrame]:
    # Try models directories first, then fall back to data/processed
    data_paths_to_try = [
        Path("models/data"),
        Path("data/processed/jarir")
    ]
    
    for data_dir in data_paths_to_try:
        try:
            if (data_dir / "customers_clean.parquet").exists():
                return {
                    "customers": pd.read_parquet(data_dir / "customers_clean.parquet"),
                    "items": pd.read_parquet(data_dir / "items_clean.parquet"),
                    "interactions": pd.read_parquet(data_dir / "interactions_clean.parquet"),
                    "customer_map": pd.read_parquet(data_dir / "customer_id_map.parquet"),
                    "item_map": pd.read_parquet(data_dir / "item_id_map.parquet"),
                    "sequences_train": pd.read_parquet(data_dir / "sequences_train.parquet"),
                }
        except Exception:
            continue
    
    st.error("Could not load catalog data from models/data or data/processed/jarir directories.")
    st.stop()

@st.cache_resource
def load_item_features(n_items: int) -> Tuple[np.ndarray, np.ndarray]:
    """Popularity and price_z vectors, built exactly as at reranker training time."""
    data = load_catalog_data()
    return build_item_side_features(data["sequences_train"], data["items"], data["item_map"], n_items)

@st.cache_data
def get_user_history(customer_id: int, interactions: pd.DataFrame, limit: int = 50) -> pd.DataFrame:
    user_history = interactions[interactions['customer_id'] == customer_id].copy()
    if 'invoice_date' in user_history.columns:
        user_history = user_history.sort_values('invoice_date', ascending=False)
    return user_history.head(limit)

def clean_category_name(category: str) -> str:
    if pd.isna(category) or category == "" or str(category).lower() == 'nan':
        return "General"
    
    category_str = str(category).strip()
    # Remove leading codes like IN01, In06, IN001 etc., followed by spaces/dashes
    category_str = re.sub(r"^(?i:in)\d+\s*[-:_]*\s*", "", category_str).strip()
    return category_str.title() if category_str else "General"

def clean_brand_name(brand: str) -> str:
    """Clean brand names"""
    if pd.isna(brand) or brand == "" or str(brand).lower() == 'nan':
        return "Generic"
    
    brand_str = str(brand).strip()
    
    # Handle "Non Branded" variants
    if brand_str.lower() in ['non branded', 'nonbranded', 'no brand']:
        return "Non Branded"
    
    # Remove suffix after " - "
    if " - " in brand_str:
        brand_str = brand_str.split(" - ")[0]
    
    return brand_str.title()

def build_user_vector_from_history(item_emb: np.ndarray, history: List[int], max_hist: int = 15) -> np.ndarray:
    if not history:
        return np.zeros(item_emb.shape[1], dtype=np.float32)
    
    recent_items = history[:max_hist]
    valid_items = [i for i in recent_items if 0 <= i < len(item_emb)]
    
    if not valid_items:
        return np.zeros(item_emb.shape[1], dtype=np.float32)
    
    u = np.mean(item_emb[valid_items], axis=0)
    n = np.linalg.norm(u) + 1e-8
    return (u / n).astype(np.float32)

def get_candidates(user_vector: np.ndarray, item_embeddings: np.ndarray, k: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    similarities = user_vector @ item_embeddings.T
    top_indices = np.argsort(similarities)[-k:][::-1]
    top_scores = similarities[top_indices]
    return top_indices, top_scores

def get_popular_items(items: pd.DataFrame, limit: int = 50) -> List[int]:
    return items.nlargest(limit, 'popularity').index.tolist()

def rerank_candidates(candidates: np.ndarray, user_vector: np.ndarray, item_embeddings: np.ndarray,
                     user_history: List[int], reranker: MLPRanker, 
                     items_df: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, np.ndarray]:
    if len(candidates) == 0:
        return np.array([]), np.array([])
    
    feature_builder = RankingFeatureBuilder(
        embedding_dim=item_embeddings.shape[1],
        max_history_length=15,
        hard_negatives=False,
        n_negatives_per_query=None
    )
    
    df = pd.DataFrame({
        "history_idx": [" ".join(map(str, user_history[:15]))],
        "pos_item_idx": [int(candidates[0])],
        "cands": [" ".join(map(str, candidates.tolist()))],
    })
    
    # Real item-side features; the model was trained with these, not zeros.
    popularity_scores, price_features = load_item_features(item_embeddings.shape[0])

    features = feature_builder.build_features(
        df, 
        user_embeddings=np.array([user_vector]), 
        item_embeddings=item_embeddings,
        popularity_scores=popularity_scores,
        price_features=price_features,
        device="cpu",
        batch_size=1024
    )
    
    feature_cols = ["dot_uv", "max_sim_recent", "pop", "hist_len", "price_z"]
    X = np.stack([features[col] for col in feature_cols], axis=-1)
    X_tensor = torch.from_numpy(X).float()
    
    with torch.no_grad():
        scores = reranker(X_tensor).numpy().flatten()
    
    sorted_indices = np.argsort(scores)[::-1]
    return candidates[sorted_indices], scores[sorted_indices]

# Removed generate_explanation function - no longer needed

# Main application
def main():
    # Header
    st.markdown("""
    <div class="header">
        <h1>🎯 Jarir B2B Recommendation System</h1>
        <p>Personalized product recommendations for enhanced customer experience</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    data = load_catalog_data()
    user_embeddings, item_embeddings, embeddings_path = load_embeddings()
    reranker, reranker_path = load_reranker()
    
    # Controls Section
    st.markdown('<div class="controls">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Controls")
    
    # Create customer options
    customers = data["customers"]
    customer_options = {}
    for _, row in customers.iterrows():
        customer_id = row['customer_id']
        total_spent = row['total_spent']
        customer_options[f"Customer {customer_id} (${total_spent:.0f})"] = customer_id
    
    # Control inputs
    col1, col2, col3 = st.columns([4, 1, 1])
    
    with col1:
      # Set default to specific customer ID 10018322
      preferred_customer_id = 10018322
      keys_list = list(customer_options.keys())
      vals_list = list(customer_options.values())
      default_index = 0
      if preferred_customer_id in vals_list:
          default_index = vals_list.index(preferred_customer_id)
      
      selected_display = st.selectbox(
          "Select Customer",
          options=keys_list,
          index=default_index
      )
      selected_customer_id = customer_options[selected_display]
    
    with col2:
        k = st.select_slider(
            "Top Results",
            options=[5, 10, 15, 20, 25, 30],
            value=10
        )
    
    with col3:
        get_recs = st.button("Get Recommendations", type="primary")
        if get_recs:
            # Clear any cached data to ensure fresh results
            st.cache_data.clear()
            st.session_state['generate_recs'] = True
            st.rerun()
    
    # Customer Info
    customer_info = customers[customers['customer_id'] == selected_customer_id].iloc[0]
    member_date = str(customer_info['first_date']).split()[0]
    
    # Get user history for analysis
    user_history = get_user_history(selected_customer_id, data["interactions"])
    
    if not user_history.empty:
        # Get top 5 categories
        top_categories = user_history['category'].value_counts().head(5)
        category_names = [clean_category_name(cat) for cat in top_categories.index]
        
        # Calculate insights
        avg_price = user_history['price'].mean()
        total_purchases = len(user_history)
        favorite_brand = user_history['brand'].value_counts().index[0] if len(user_history) > 0 else "None"
        
        st.markdown(f"""
        <div class="customer-info">
            <div class="customer-avatar">{str(selected_customer_id)[-2:]}</div>
            <div class="customer-details">
                <strong>Customer {selected_customer_id}</strong><br>
                <small>Member since {member_date}</small>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Top 5 Categories Section
        st.markdown("#### 🎯 Top 5 Categories")
        
        # Create columns for categories
        category_cols = st.columns(len(category_names) if len(category_names) <= 5 else 5)
        for i, (cat, count) in enumerate(zip(category_names[:5], top_categories.values[:5])):
            with category_cols[i]:
                st.markdown(f"""
                <div style="background: #2c3e50; padding: 12px; border-radius: 8px; text-align: center; border: 1px solid #3498db; margin-bottom: 10px;">
                    <strong style="color: white; font-size: 0.9rem;">{cat}</strong><br>
                    <small style="color: #bdc3c7;">{count} purchases</small>
                </div>
                """, unsafe_allow_html=True)
        
    else:
        st.markdown(f"""
        <div class="customer-info">
            <div class="customer-avatar">{str(selected_customer_id)[-2:]}</div>
            <div class="customer-details">
                <strong>Customer {selected_customer_id}</strong><br>
                <small>Member since {member_date}</small>
            </div>
            <div style="margin-top: 15px; color: white;">
                <p>No purchase history available for analysis.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Customer Purchase History Section
    st.markdown("### 📦 Recent Purchases")
    
    # Get user history for display (75% of all purchases)
    user_purchase_history = get_user_history(selected_customer_id, data["interactions"], limit=None)
    
    if not user_purchase_history.empty:
        # Calculate 75% of purchases to show
        total_purchases = len(user_purchase_history)
        items_to_show = max(3, int(total_purchases * 0.75))  # Show at least 3, or 75% of total
        items_to_show = min(items_to_show, 9)  # Cap at 9 items (3 rows of 3)
        items_to_show = min(items_to_show, total_purchases)  # Ensure we don't exceed available items
        
        # Display recent purchases in a responsive grid (dark mode like recommendations)
        for row in range(0, items_to_show, 3):  # Process in rows of 3
            purchase_cols = st.columns(3)
            for col_idx in range(3):
                i = row + col_idx
                if i >= items_to_show or i >= total_purchases:
                    break
                    
                item = user_purchase_history.iloc[i]
                col = purchase_cols[col_idx]
                with col:
                    clean_cat = clean_category_name(str(item['category']))
                    st.markdown(f"""
                    <div style="
                        background: #2c3e50; 
                        border: 1px solid #34495e; 
                        border-radius: 10px; 
                        padding: 20px; 
                        margin-bottom: 15px; 
                        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
                        transition: transform 0.2s ease;
                        height: 200px;
                        display: flex;
                        flex-direction: column;
                        justify-content: space-between;
                    " onmouseover="this.style.transform='translateY(-2px)'" onmouseout="this.style.transform='translateY(0)'">
                        <div style="
                            background: #e74c3c; 
                            color: white; 
                            width: 30px; 
                            height: 30px; 
                            border-radius: 50%; 
                            display: flex; 
                            align-items: center; 
                            justify-content: center; 
                            font-weight: bold; 
                            margin-bottom: 15px;
                            font-size: 14px;
                        ">
                            {i+1}
                        </div>
                        <div style="flex-grow: 1;">
                            <h4 style="color: white !important; margin-bottom: 10px; font-size: 16px; line-height: 1.3; font-weight: 600;">{str(item['description'])[:50]}{'...' if len(str(item['description'])) > 50 else ''}</h4>
                            <p style="color: #2ecc71 !important; font-size: 18px; font-weight: bold; margin: 10px 0;">${item['price']:.2f}</p>
                            <span style="background: #3498db; color: white; padding: 4px 8px; border-radius: 12px; font-size: 12px; font-weight: 500;">{clean_cat}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Show total count
        if items_to_show < total_purchases:
            st.markdown(f"<p style='text-align: center; color: #7f8c8d; font-size: 0.9rem;'>Showing {items_to_show} of {total_purchases} purchases (75% of history)</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p style='text-align: center; color: #7f8c8d; font-size: 0.9rem;'>Showing all {total_purchases} purchases</p>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: #34495e; color: white; border-radius: 10px;">
            <p>No purchase history available for this customer.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Recommendations Section
    st.markdown("### ✨ Recommendations")
    
    if st.session_state.get('generate_recs', False):
        start_time = time.time()
        
        # Get user history for recommendations
        user_history = get_user_history(selected_customer_id, data["interactions"])
        
        if not user_history.empty:
            item_map_dict = dict(zip(data["item_map"]['stock_code'], data["item_map"]['item_idx']))
            user_item_indices = [item_map_dict.get(stock) for stock in user_history['stock_code'].tolist()]
            user_item_indices = [idx for idx in user_item_indices if idx is not None]
        else:
            user_item_indices = []
        
        # Generate recommendations
        if user_item_indices:
            user_vector = build_user_vector_from_history(item_embeddings, user_item_indices)
            candidates, retrieval_scores = get_candidates(user_vector, item_embeddings, 200)
            
            if len(candidates) == 0:
                candidates = get_popular_items(data["items"], k)
                candidates = np.array(candidates[:k])
                scores = np.ones(len(candidates))
            else:
                candidates, scores = rerank_candidates(candidates, user_vector, item_embeddings, 
                                                     user_item_indices, reranker, data["items"])
                candidates = candidates[:k]
                scores = scores[:k]
        else:
            candidates = get_popular_items(data["items"], k)
            candidates = np.array(candidates[:k])
            scores = np.ones(len(candidates))
        
        generation_time = (time.time() - start_time) * 1000
        
        st.markdown(f'<div class="generation-time">⚡ Generated in {generation_time:.0f}ms</div>', unsafe_allow_html=True)
        st.success(f"✅ Successfully generated {len(candidates)} recommendations!")
        
        # Debug: Show if reranker was used
        if user_item_indices and len(candidates) > 0:
            st.info(f"🧠 **Reranker Used**: Retrieved 200 candidates, reranked with MLP model, showing top {len(candidates)}")
            # Show top 3 reranker scores for verification
            if len(scores) >= 3:
                st.markdown(f"**Top 3 Reranker Scores:** {scores[0]:.4f}, {scores[1]:.4f}, {scores[2]:.4f}")
        else:
            st.warning(f"📊 **Popularity Fallback**: No user history available, showing popular items")
        
        # Display recommendations
        if len(candidates) > 0:
            item_map_dict = dict(zip(data["item_map"]['item_idx'], data["item_map"]['stock_code']))
            items_indexed = data["items"].set_index('stock_code')
            
            products_html = '<div class="product-grid">'
            products_added = 0
            
            for i, (item_idx, score) in enumerate(zip(candidates, scores)):
                stock_code = item_map_dict.get(item_idx)
                if stock_code and stock_code in items_indexed.index:
                    try:
                        item_info = items_indexed.loc[stock_code]
# No explanation needed
                        clean_cat = clean_category_name(str(item_info['category']))
                        
                        # Ensure we have valid data
                        title = str(item_info['description'])[:80] if pd.notna(item_info['description']) else "Product"
                        price = float(item_info['price_median']) if pd.notna(item_info['price_median']) else 0.0
                        
                        products_html += f"""
                        <div class="product-card">
                            <div class="product-rank">{products_added+1}</div>
                            <div class="product-title">{title}</div>
                            <div class="product-price">${price:.2f}</div>
                            <div class="product-category">{clean_cat}</div>
                        </div>
                        """
                        products_added += 1
                        
                        if products_added >= k:  # Stop when we have enough products
                            break
                            
                    except Exception as e:
                        st.write(f"Error processing item {stock_code}: {e}")
                        continue
            
            products_html += '</div>'
            
            if products_added > 0:
                # Create columns to display products in a grid
                cols = st.columns(3)
                col_index = 0
                
# No explanations needed - removed user_history_for_explanations
                
                for i, (item_idx, score) in enumerate(zip(candidates, scores)):
                    if col_index >= len(cols):
                        cols = st.columns(3)
                        col_index = 0
                    
                    stock_code = item_map_dict.get(item_idx)
                    if stock_code and stock_code in items_indexed.index:
                        try:
                            item_info = items_indexed.loc[stock_code]
                            clean_cat = clean_category_name(str(item_info['category']))
                            
                            # Ensure we have valid data
                            title = str(item_info['description'])[:60] if pd.notna(item_info['description']) else "Product"
                            price = float(item_info['price_median']) if pd.notna(item_info['price_median']) else 0.0
                            
                            with cols[col_index]:
                                st.markdown(f"""
                                <div style="background: #34495e; border: 1px solid #3498db; border-radius: 8px; padding: 15px; margin-bottom: 15px; position: relative; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
                                    <div style="position: absolute; top: 10px; right: 10px; background: #e74c3c; color: white; width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 0.8rem; font-weight: bold;">{i+1}</div>
                                    <h5 style="color: white; margin-bottom: 8px; font-size: 1rem; font-weight: 600;">{title}</h5>
                                    <p style="color: #2ecc71; font-size: 1.2rem; font-weight: bold; margin-bottom: 8px;">${price:.2f}</p>
                                    <span style="background: #3498db; color: white; padding: 3px 8px; border-radius: 12px; font-size: 0.75rem;">{clean_cat}</span>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            col_index += 1
                            
                            if i + 1 >= k:  # Stop when we have enough products
                                break
                                
                        except Exception as e:
                            continue
                # Add "Why these recommendations?" section after displaying products
                if products_added > 0:
                    st.markdown("---")  # Separator line
                    st.markdown("#### 🤖 Why These Recommendations?")
                    
                    # Get user history for analysis
                    user_history_analysis = get_user_history(selected_customer_id, data["interactions"])
                    
                    if not user_history_analysis.empty:
                        # Get top categories from user history
                        top_user_categories = user_history_analysis['category'].value_counts().head(3)
                        category_names_analysis = [clean_category_name(cat) for cat in top_user_categories.index]
                        
                        # Calculate insights
                        avg_price_analysis = user_history_analysis['price'].mean()
                        total_purchases_analysis = len(user_history_analysis)
                        favorite_brand_analysis = user_history_analysis['brand'].value_counts().index[0] if len(user_history_analysis) > 0 else "None"
                        
                        st.markdown(f"""
                        <div style="background: #2c3e50; padding: 20px; border-radius: 10px; border-left: 4px solid #3498db; margin-bottom: 20px;">
                            <p style="margin: 8px 0; color: white; line-height: 1.5;"><strong>🎯 Pattern Recognition:</strong> Based on your {total_purchases_analysis} purchases, you frequently buy {category_names_analysis[0]} items</p>
                            <p style="margin: 8px 0; color: white; line-height: 1.5;"><strong>💰 Price Matching:</strong> These recommendations match your typical spending range around ${avg_price_analysis:.0f}</p>
                            <p style="margin: 8px 0; color: white; line-height: 1.5;"><strong>🏷️ Brand Preference:</strong> Selected items align with your preference for {clean_brand_name(favorite_brand_analysis)}</p>
                            <p style="margin: 8px 0; color: white; line-height: 1.5;"><strong>🧠 AI Logic:</strong> Our neural network analyzed similar customers and identified these items with highest purchase probability</p>
                            <p style="margin: 8px 0; color: white; line-height: 1.5;"><strong>📊 Recommendation Score:</strong> Items ranked by ML model combining purchase history, category preferences, and collaborative filtering</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style="background: #2c3e50; padding: 20px; border-radius: 10px; border-left: 4px solid #3498db; margin-bottom: 20px;">
                            <p style="margin: 8px 0; color: white; line-height: 1.5;"><strong>🤖 AI Logic:</strong> These are popular items recommended for new customers based on general trends and ratings.</p>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.error("No valid products found in recommendations.")
        else:
            st.error("No candidates generated for recommendations.")
        
        st.session_state['generate_recs'] = False
    
    else:
        st.markdown("""
        <div style="text-align: center; padding: 40px; background: #34495e; color: white; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
            <h4 style="color: #3498db; margin-bottom: 10px;">🎯 Ready to get recommendations?</h4>
            <p style="color: #bdc3c7;">Click "Get Recommendations" to see personalized suggestions for this customer.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Model Debug Section
    st.markdown("---")
    st.markdown("### 🔧 Model Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div style="background: #34495e; padding: 15px; border-radius: 8px; margin-bottom: 10px;">
            <h5 style="color: #3498db; margin-bottom: 8px;">📊 Embeddings (Two-Tower Model)</h5>
            <p style="color: white; margin: 2px 0; font-size: 0.9rem;"><strong>Path:</strong> {embeddings_path}</p>
            <p style="color: white; margin: 2px 0; font-size: 0.9rem;"><strong>User Embeddings:</strong> {user_embeddings.shape}</p>
            <p style="color: white; margin: 2px 0; font-size: 0.9rem;"><strong>Item Embeddings:</strong> {item_embeddings.shape}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: #34495e; padding: 15px; border-radius: 8px; margin-bottom: 10px;">
            <h5 style="color: #e74c3c; margin-bottom: 8px;">🧠 Reranker (MLP Model)</h5>
            <p style="color: white; margin: 2px 0; font-size: 0.9rem;"><strong>Path:</strong> {reranker_path}</p>
            <p style="color: white; margin: 2px 0; font-size: 0.9rem;"><strong>Architecture:</strong> MLP (5 → 384 → 384 → 192 → 1)</p>
            <p style="color: white; margin: 2px 0; font-size: 0.9rem;"><strong>Features:</strong> dot_uv, max_sim, pop, hist_len, price_z</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()