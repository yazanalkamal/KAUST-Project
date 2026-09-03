# Jarir B2B Recommendation System

Next-purchase recommendations for Jarir B2B customers, with a Streamlit
interface for browsing them per customer.

## 🎯 Overview

Recommendations are served in two stages:

1. **Retrieval**: an item-item nearest-neighbour model (ItemKNN) scores every
   product by its cosine similarity to the customer's recent purchases and
   returns the top 200.
2. **Reranking (optional, off by default)**: an MLP scores those 200 candidates
   on six features and reorders them.

The reranker is off by default because it does not beat the retriever on the
validation split. The numbers for both are in [Evaluation](#-evaluation).

A Two-Tower neural retriever is also in the repository. Its embeddings supply
two of the reranker's features, but it is not used for retrieval: it scores
0.056 Recall@10 on test against 0.200 for ItemKNN. With 2,901 purchase lines
there is not enough data to fit its 690k embedding parameters.

## 🚀 Live Demo

**Option 1: Streamlit Cloud (Recommended)**
- Visit the live application: [Jarir B2B Recommendation System](https://jarir-project.streamlit.app/)
- No installation required - ready to use immediately

**Option 2: Local Development**
- Clone and run locally for development and customization

## 🏗️ Architecture

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Customer        │ -> │  ItemKNN         │ -> │  MLP reranker    │
│  purchase        │    │  retriever       │    │  (optional)      │
│  history         │    │                  │    │                  │
│                  │    │  item-item       │    │  Features:       │
│  most recent 15  │    │  cosine over     │    │  • retrieval rank│
│  items, recency  │    │  the purchase    │    │  • embedding sim │
│  weighted (0.5^i)│    │  matrix          │    │  • max sim to    │
│                  │    │                  │    │    recent items  │
│                  │    │  top 200         │    │  • popularity    │
│                  │    │  candidates      │    │  • history length│
│                  │    │                  │    │  • price z-score │
└──────────────────┘    └──────────────────┘    └──────────────────┘
                                 │                        │
                                 │  default path          │
                                 v                        v
                          ┌─────────────────────────────────┐
                          │        Top-K products           │
                          └─────────────────────────────────┘
```

Candidate lists are built from retrieval only. The ground-truth item is never
inserted into them; `tests/test_evaluation_integrity.py` fails if it is.

## 📁 Project Structure

```
KAUST-Project/
├── app/
│   └── streamlit_app.py          # Interactive web application
├── models/
│   ├── data/                     # Clean dataset (Parquet files)
│   │   ├── customers_clean.parquet
│   │   ├── items_clean.parquet
│   │   ├── interactions_clean.parquet
│   │   └── *_id_map.parquet
│   │   ├── sequences_*.parquet   # train/val/test queries
│   │   └── candidates_*.parquet  # 200 retrieved candidates per query
│   ├── retriever/                # Two-Tower artifacts (reranker features)
│   │   ├── user_embeddings.npy
│   │   ├── item_embeddings.npy
│   │   └── training_metrics.json
│   └── reranker/                 # MLP reranker artifacts
│       ├── best_ranker.pt
│       └── training_metrics.json
├── src/                          # Core ML modules
│   ├── data/                     # Data processing & feature engineering
│   ├── models/                   # Two-Tower and MLP architectures
│   ├── training/                 # Training loops and callbacks
│   ├── inference/                # item_knn.py (served retriever), rerank, ANN
│   └── evaluation/               # Recall@K / NDCG@K
├── configs/                      # YAML configuration files
├── scripts/                      # Training & data preparation CLIs
├── tests/                        # Unit tests and evaluation-integrity guards
└── requirements.txt              # Dependencies
```

## 📊 Data

One year of Jarir B2B purchase lines, cleaned into `models/data/*.parquet`.

| Table | Rows |
|---|---|
| `customers_clean.parquet` | 929 customers |
| `items_clean.parquet` | 1,735 items |
| `interactions_clean.parquet` | 2,901 purchase lines, 2024-01-01 to 2024-12-31 |

Sequences are one row per purchase event that has at least two prior purchases
by the same customer. Each row holds the customer's history and the single next
item, which is the prediction target. Splits are by time, so no split sees a
purchase made at or after the queries it is evaluated on.

| Split | Sequences | Customers | Date range |
|---|---|---|---|
| `sequences_train.parquet` | 1,108 | 296 | 2024-01-02 to 2024-10-12 |
| `sequences_val.parquet` | 169 | 64 | 2024-10-14 to 2024-11-20 |
| `sequences_test.parquet` | 160 | 62 | 2024-11-21 to 2024-12-31 |

`candidates_{train,val,test}.parquet` hold 200 retrieved candidates per query.
The target is among them for 34% of train, 37% of val and 33% of test queries;
the rest are queries the retriever misses, and they count as misses everywhere
in the evaluation.

This is a small dataset. The test split is 160 predictions, so one extra hit
moves Recall@10 by 0.006 and differences under about 0.05 are not meaningful.

## 🛠️ Local Setup

### Prerequisites
- Python 3.10+
- 8GB+ RAM recommended
- Windows/macOS/Linux

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yazanalkamal/KAUST-Project
cd KAUST-Project
```

2. **Create virtual environment**
```bash
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify model artifacts**
Ensure these files exist:
- `models/retriever/{user_embeddings.npy, item_embeddings.npy}`
- `models/reranker/best_ranker.pt`
- `models/data/*.parquet`

5. **Launch the application**
```bash
streamlit run app/streamlit_app.py
```

The app will be available at `http://localhost:8501`

## 📊 Features

### Interface
- Customer selection with search and profile summary
- Recommendations with purchase-history context
- Toggle between retrieval-only and reranked results
- Light and dark themes

### Models
- ItemKNN retriever over the customer-item purchase matrix
- MLP reranker on six features, optional
- Two-Tower retriever with 256-dimensional embeddings, used for reranker features
- Evaluation with Recall@K and NDCG@K over the full catalogue

## 📈 Evaluation

Reproduce with `python scripts/evaluate.py` (seed 42, no tuning).

Each query has exactly one correct item, the customer's actual next purchase.
Recall@10 is therefore the share of queries whose next purchase appears in the
top 10, and NDCG@10 is 1/log2(rank+1) when it does.

**Ranked over all 1,735 items.** This is the number that describes the product:
the model picks 10 items out of the whole catalogue.

| Model | Split | Recall@10 | NDCG@10 |
|---|---|---|---|
| Popularity: 10 most purchased items, same list for everyone | test | 0.050 | 0.032 |
| Repeat purchase: the customer's own recent items | test | 0.125 | 0.068 |
| Two-Tower retriever, history-mean embedding | test | 0.056 | 0.049 |
| Two-Tower retriever, user-ID embedding | test | 0.038 | 0.028 |
| **ItemKNN retriever (served by default)** | **test** | **0.200** | **0.112** |
| ItemKNN 200 candidates + MLP reranker | test | 0.188 | 0.082 |
| Popularity: 10 most purchased items, same list for everyone | val | 0.095 | 0.058 |
| Repeat purchase: the customer's own recent items | val | 0.112 | 0.061 |
| Two-Tower retriever, history-mean embedding | val | 0.047 | 0.034 |
| Two-Tower retriever, user-ID embedding | val | 0.095 | 0.060 |
| **ItemKNN retriever (served by default)** | **val** | **0.201** | **0.117** |
| ItemKNN 200 candidates + MLP reranker | val | 0.189 | 0.102 |

The reranker is behind the retriever on both splits, so the app serves
retrieval only and offers reranking through a checkbox.

**Ranked over the 200 retrieved candidates.** A much easier task, listed
separately because the two denominators are not comparable. Restricted to the
queries whose target the retriever found (53 of 160 on test, 62 of 169 on val).

| Model | Split | Ranked set | Recall@10 | NDCG@10 |
|---|---|---|---|---|
| ItemKNN order, no reranking | test | 200 candidates | 0.604 | 0.338 |
| MLP reranker | test | 200 candidates | 0.566 | 0.249 |
| ItemKNN order, no reranking | val | 200 candidates | 0.548 | 0.318 |
| MLP reranker | val | 200 candidates | 0.516 | 0.279 |

Retrieval quality sets the ceiling for the whole pipeline: ItemKNN puts the
target in its top 200 for 33% of test queries, against 15% for the Two-Tower
retriever. Nothing downstream can recover the other 67%.

### An earlier claim of 0.69

Previous versions of this README reported Recall@10 ~0.69 and NDCG@10 ~0.55.
That measurement reranked 100-candidate lists into which the target had been
inserted whenever retrieval missed it, which was 89% of test lists. The
reranker learned to identify the inserted item, since it was always the
lowest-similarity entry in its list. Over the full catalogue that same pipeline
scored 0.006 Recall@10. The insertion is gone, and a test fails if it returns.

## 🔬 What was fixed

| Problem | Effect | Where |
|---|---|---|
| Ground-truth item inserted into candidate lists | Reranker learned the artifact; headline metric inflated from 0.006 to 0.64 | `scripts/build_candidates.py` |
| `scripts/evaluate.py` returned hard-coded zeros for reranking | No working end-to-end evaluation existed | `scripts/evaluate.py` |
| `src/evaluation/metrics.py` raised NameError on import | Metrics module and its test could not run | `src/evaluation/metrics.py` |
| Popularity and price features passed as None in training and zeros in the app | Two of five features were constant | `scripts/train_reranker.py`, `app/streamlit_app.py` |
| App rebuilt the reranker under a different architecture and remapped weights into it | Served network differed from the trained one | `app/streamlit_app.py` |
| Reranker could not see the retrieval score | Reranking cut candidate-set Recall@10 from 0.604 to 0.226 | `src/data/features.py` (`retr_rr`) |
| Checkpoints selected on BCE val_loss | Selection was near-blind to ranking quality | `scripts/train_reranker.py` |
| Purchase history passed most-recent-first into features expecting chronological order | Recency features inverted at serving time | `app/streamlit_app.py` |

## 🔄 Training Pipeline (Optional)

The served ItemKNN retriever has no training step; it is fitted from the
purchase table at startup. To rebuild the rest:

1. **Data preparation**
```bash
python scripts/prepare_data.py --config configs/data.yaml
```

2. **Train the Two-Tower retriever** (optional; supplies reranker features)
```bash
python scripts/train_retriever.py --config configs/retriever.yaml
```

3. **Build candidate lists**
```bash
python scripts/build_candidates.py --data-dir models/data --k 200
```

4. **Train the reranker**
```bash
python scripts/train_reranker.py --config configs/reranker.yaml --data-dir models/data
```

5. **Evaluate**
```bash
python scripts/evaluate.py
```

Outputs are written to `models/`.

## 🚀 Deployment

### Streamlit Community Cloud
1. Fork this repository
2. Connect to Streamlit Cloud
3. Deploy directly - no additional configuration needed

### Custom Infrastructure
- **Docker**: Use provided containerization (optional)
- **Cloud platforms**: AWS, GCP, Azure compatible
- **Requirements**: Python 3.10+, 2GB+ RAM, 1 CPU core

## 🧪 Testing

```bash
pytest tests/ -v
```

`tests/test_evaluation_integrity.py` guards the evaluation itself. It fails if
the target item reappears in nearly every candidate list, if candidates stop
being ordered best-first, if retrieval sees purchases dated at or after the
query, or if reranker features go constant. Each of those once produced a
wrong headline number here.

## 🛠️ Tech Stack

- **Core ML**: Python, NumPy, Pandas, scikit-learn, PyTorch
- **Models**: ItemKNN retriever (served), MLP reranker, Two-Tower retriever
- **Data**: PyArrow and fastparquet, YAML configuration
- **Search**: FAISS, used by the evaluation script for exact top-k
- **UI**: Streamlit with custom CSS, light and dark themes
- **Development**: pytest, rich logging
- **Deployment**: model artifacts committed under `models/`, Streamlit Community Cloud

## 🔍 Troubleshooting

### Common Issues

**Missing model files**
- Verify all required files exist in `models/` directory
- Check file permissions and paths

**Model loading errors**
- Clear Streamlit cache: Settings → Clear cache
- Restart the application

**Performance issues**
- Ensure sufficient RAM (8GB+ recommended)
- Check CPU usage during inference

**Import errors on Streamlit Cloud**
- The app automatically handles Python path configuration
- Verify all dependencies are in `requirements.txt`

## 🤝 Contributing

**Yazan Alkamal** | Software Engineering at UQU
- Designed and built the two-stage recommendation pipeline
- Built the Streamlit application and the customer-facing interface
- Set up the models-first layout used for deployment
- Data cleaning, sequence construction and evaluation
