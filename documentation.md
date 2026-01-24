# Tourist Recommendation System Documentation

A personalized tourist recommendation system that integrates MBTI personality prediction, multimodal topic modeling, graph neural networks, and hybrid ranking for venue recommendations.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Project Structure](#project-structure)
5. [Pipeline Phases](#pipeline-phases)
6. [Usage](#usage)
7. [API Reference](#api-reference)
8. [Configuration](#configuration)
9. [Evaluation Metrics](#evaluation-metrics)

---

## Overview

This system provides personalized venue recommendations by understanding:
- **Who** the user is (MBTI personality from reviews)
- **What** venues offer (multimodal topic modeling)
- **How** users and venues relate (graph neural networks)
- **Why** recommendations are relevant (hybrid ranking with calibration)

### Key Features

- BERT-based MBTI personality prediction from user reviews
- Multimodal BERTopic with CLIP image embeddings
- Linear-Time GNN (LTGNN) for scalable graph learning
- XGBoost ranking with gBCE loss for calibrated predictions
- DCI Closed itemset mining for behavioral patterns

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Yelp Dataset                            │
│              (Business, Review, User, Image)                    │
└─────────────────────────────────────────────────────────────────┘
                              │
           ┌──────────────────┼──────────────────┐
           ▼                  ▼                  ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   Phase 2       │  │   Phase 3       │  │   Phase 4       │
│   BERT MBTI     │  │   BERTopic      │  │   LTGNN         │
│   Classifier    │  │   + CLIP        │  │   Graph         │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                    │
         ▼                    ▼                    ▼
   User MBTI            Venue Topics         GNN Embeddings
   Embeddings           + Regions            (User & Venue)
         │                    │                    │
         └────────────────────┼────────────────────┘
                              ▼
                 ┌─────────────────────────┐
                 │      Phase 5            │
                 │   Hybrid Recommender    │
                 │   (DCI + XGBoost)       │
                 └────────────┬────────────┘
                              ▼
                   Personalized Rankings
```

---

## Installation

### Requirements

- Python 3.10+
- CUDA 11.8+ (for GPU support)

### Setup

```bash
# Clone repository
git clone <repository-url>
cd draft-koko

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Data Setup

1. Download the [Yelp Dataset](https://www.yelp.com/dataset)
2. Extract to `data/raw/`:
   ```
   data/raw/
   ├── yelp_academic_dataset_business.json
   ├── yelp_academic_dataset_review.json
   ├── yelp_academic_dataset_user.json
   └── photos/
   ```

---

## Project Structure

```
draft-koko/
├── src/
│   ├── config/
│   │   └── settings.py          # All configuration classes
│   ├── data/
│   │   ├── yelp_loader.py       # Yelp dataset loader
│   │   ├── preprocessor.py      # Text cleaning, tokenization
│   │   └── sampler.py           # Train/val/test splits, negative sampling
│   ├── models/
│   │   ├── bert_mbti/
│   │   │   ├── model.py         # MBTIClassifier
│   │   │   ├── trainer.py       # Training loop
│   │   │   └── inference.py     # Batch prediction
│   │   ├── bertopic/
│   │   │   ├── multimodal.py    # Text + Image embeddings
│   │   │   ├── topic_extractor.py  # BERTopic pipeline
│   │   │   └── geo_cluster.py   # Geographic clustering
│   │   ├── gnn/
│   │   │   ├── graph_builder.py # Heterogeneous graph
│   │   │   ├── ltgnn.py         # Linear-Time GNN
│   │   │   ├── evr_sampler.py   # Variance-reduced sampling
│   │   │   └── trainer.py       # GNN training
│   │   └── hybrid/
│   │       ├── dci_closed.py    # Closed itemset mining
│   │       ├── xgboost_ranker.py # XGBoost ranking
│   │       └── gbce_loss.py     # Calibration losses
│   ├── utils/
│   │   ├── metrics.py           # Evaluation metrics
│   │   └── helpers.py           # Utilities
│   └── api/                     # FastAPI endpoints (Phase 7)
├── scripts/
│   ├── prepare_data.py          # Data preprocessing
│   ├── train_mbti.py            # MBTI classifier training
│   ├── extract_topics.py        # Topic extraction
│   ├── train_gnn.py             # GNN training
│   └── train_hybrid.py          # Hybrid model training
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_mbti_training.ipynb
│   ├── 03_topic_modeling.ipynb
│   ├── 04_gnn_training.ipynb
│   └── 05_hybrid_evaluation.ipynb
├── data/
│   ├── raw/                     # Original Yelp data
│   └── processed/               # Preprocessed data
├── models/
│   ├── checkpoints/             # Training checkpoints
│   ├── bertopic/                # Topic model outputs
│   ├── gnn/                     # GNN outputs
│   └── hybrid/                  # Final model
├── requirements.txt
├── setup.py
└── claude.md                    # Implementation plan
```

---

## Pipeline Phases

### Phase 1: Data Preparation

```bash
python scripts/prepare_data.py --filter-tourism
```

**Components:**
- `YelpDataLoader`: Loads Business, Review, User, Photo datasets
- `TextPreprocessor`: Cleans text (HTML, URLs, special chars)
- `DataSampler`: Train/val/test splits with stratification

**Outputs:**
- `data/processed/train_reviews.parquet`
- `data/processed/val_reviews.parquet`
- `data/processed/test_reviews.parquet`
- `data/processed/businesses.parquet`

---

### Phase 2: MBTI Personality Prediction

```bash
python scripts/train_mbti.py --epochs 10 --batch-size 16
```

**Components:**
- `MBTIClassifier`: BERT + classification head for 16 MBTI types
- `MBTITrainer`: AdamW, OneCycleLR, early stopping
- `MBTIInference`: Batch prediction and user profiling

**Model Architecture:**
```
BERT (bert-base-uncased)
    ↓
[CLS] Token Embedding (768-dim)
    ↓
Dropout (0.1)
    ↓
Linear (768 → 16)
    ↓
Softmax → MBTI Type
```

**Outputs:**
- `models/checkpoints/bert_mbti/best_model.pt`
- `data/processed/user_mbti_profiles.parquet`
- `data/processed/user_mbti_embeddings.npy`

---

### Phase 3: Multimodal Topic Modeling

```bash
python scripts/extract_topics.py --n-topics 50 --visualize
```

**Components:**
- `TextEmbedder`: Sentence-BERT embeddings
- `ImageEmbedder`: CLIP vision embeddings
- `VenueTopicExtractor`: BERTopic with UMAP + K-Means
- `GeoClusterer`: HDBSCAN on lat/lon coordinates

**Pipeline:**
```
Reviews → Sentence-BERT → Embeddings
                              ↓
                           UMAP (dim reduction)
                              ↓
                         K-Means (clustering)
                              ↓
                      c-TF-IDF (topic words)
```

**Outputs:**
- `models/bertopic/topic_model/`
- `models/bertopic/venue_topics.parquet`
- `models/bertopic/venue_embeddings.npy`
- `models/bertopic/venue_regions.parquet`

---

### Phase 4: Graph Neural Network

```bash
python scripts/train_gnn.py --epochs 100 --hidden-dim 128
```

**Components:**
- `HeteroGraphBuilder`: Builds heterogeneous graph
- `LTGNN`: Linear-Time GNN with fixed-point iteration
- `EVRSampler`: Variance-reduced neighbor sampling
- `GNNTrainer`: Training with gBCE loss

**Graph Structure:**
```
Node Types:
  - User (features: MBTI embedding)
  - Venue (features: topic embedding)
  - Topic (features: topic centroid)
  - Region (features: geo embedding)

Edge Types:
  - (User) --visits--> (Venue)
  - (Venue) --has_topic--> (Topic)
  - (Venue) --in_region--> (Region)
```

**LTGNN Architecture:**
```
User/Venue Features
        ↓
  Input Projection (Linear → ReLU → Linear)
        ↓
  Fixed-Point Iteration (10 iterations)
    └── LightGCN Conv + Residual
        ↓
  Output Projection
        ↓
  Link Predictor (MLP)
```

**Outputs:**
- `models/gnn/graph/`
- `models/gnn/user_gnn_embeddings.npy`
- `models/gnn/venue_gnn_embeddings.npy`
- `models/checkpoints/gnn/best_model.pt`

---

### Phase 5: Hybrid Recommendation

```bash
python scripts/train_hybrid.py --num-rounds 100 --calibration-t 0.8
```

**Components:**
- `DCIClosed`: Frequent closed itemset mining
- `UserProfileMiner`: Behavioral pattern extraction
- `XGBoostRanker`: Final ranking model
- `HybridRecommender`: Complete recommendation pipeline

**Feature Synthesis:**
```
User Features:
  - GNN embedding (64-dim)
  - Itemset patterns (10-dim)

Venue Features:
  - GNN embedding (64-dim)

Pair Features:
  - User embedding
  - Venue embedding
  - Element-wise product
  - Cosine similarity
```

**gBCE Loss:**
```
gBCE(p, y) = -[y·log(p^(1/t)) + (1-y)·log(1-p^(1/t))]

where t = 0.8 (temperature for calibration)
```

**Outputs:**
- `models/hybrid/xgboost_ranker.json`
- `models/hybrid/feature_importance.csv`
- `models/hybrid/test_results.csv`

---

## Usage

### Quick Start

```python
from src.models.hybrid import HybridRecommender
import numpy as np

# Load pre-trained recommender
recommender = HybridRecommender()
recommender.ranker.load("models/hybrid/xgboost_ranker.json")

# Get recommendations
recommendations = recommender.recommend(
    user_id="user123",
    k=10,
    exclude_visited=True
)

for venue_id, score in recommendations:
    print(f"{venue_id}: {score:.4f}")
```

### MBTI Prediction

```python
from src.models.bert_mbti import MBTIInference

inference = MBTIInference(
    model_path="models/checkpoints/bert_mbti/best_model.pt"
)

result = inference.predict_single(
    "I love quiet cafes for reading and deep conversations..."
)

print(f"MBTI Type: {result['mbti_type']}")
print(f"Probabilities: {result['probabilities']}")
```

### Topic Extraction

```python
from src.models.bertopic import VenueTopicExtractor

extractor = VenueTopicExtractor()
extractor.load("models/bertopic/topic_model")

# Get topic for a venue
topics, probs = extractor.transform(["Great pizza and cozy atmosphere!"])
print(f"Topic: {topics[0]}, Probability: {probs[0]}")
```

---

## API Reference

### Data Module

#### `YelpDataLoader`
```python
loader = YelpDataLoader()
data = loader.load_all(limit=10000, categories_filter=["Restaurant"])
```

#### `TextPreprocessor`
```python
preprocessor = TextPreprocessor(lowercase=True, remove_urls=True)
clean_text = preprocessor.clean_text("Raw text with <html> tags...")
```

#### `ReviewDataset`
```python
dataset = ReviewDataset.from_dataframe(
    df, text_column="text", label_column="mbti"
)
```

### Models Module

#### `MBTIClassifier`
```python
model = MBTIClassifier(config=BERTConfig())
outputs = model(input_ids, attention_mask, labels)
loss = outputs["loss"]
logits = outputs["logits"]
```

#### `VenueTopicExtractor`
```python
extractor = VenueTopicExtractor(config=BERTopicConfig())
extractor.build_model(use_kmeans=True)
topics, probs = extractor.fit(documents, embeddings)
```

#### `LTGNN`
```python
model = LTGNN(
    user_input_dim=64,
    venue_input_dim=64,
    hidden_dim=128,
    embedding_dim=64,
    num_iterations=10,
)
user_emb, venue_emb = model.encode(user_x, venue_x, edge_index)
```

#### `HybridRecommender`
```python
recommender = HybridRecommender(config=HybridConfig())
recommender.fit(
    train_edges, user_embeddings, venue_embeddings,
    user_ids, venue_ids, user_extra
)
recommendations = recommender.recommend(user_id, k=10)
```

---

## Configuration

All configuration is centralized in `src/config/settings.py`:

### DataConfig
```python
@dataclass
class DataConfig:
    max_review_length: int = 512
    min_review_length: int = 10
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    min_reviews_per_user: int = 5
    min_reviews_per_business: int = 3
```

### BERTConfig
```python
@dataclass
class BERTConfig:
    model_name: str = "bert-base-uncased"
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 10
    num_labels: int = 16  # 16 MBTI types
```

### GNNConfig
```python
@dataclass
class GNNConfig:
    hidden_dim: int = 128
    embedding_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    fixed_point_iterations: int = 10
    evr_sample_size: int = 20
```

### HybridConfig
```python
@dataclass
class HybridConfig:
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    xgb_n_estimators: int = 100
    gbce_calibration_t: float = 0.8
    min_support: float = 0.01
```

---

## Evaluation Metrics

### Recommendation Metrics

| Metric | Description |
|--------|-------------|
| Precision@K | Fraction of recommended items that are relevant |
| Recall@K | Fraction of relevant items that are recommended |
| NDCG@K | Normalized Discounted Cumulative Gain |
| MRR | Mean Reciprocal Rank |
| Hit Rate@K | Whether any relevant item appears in top-K |

### Calibration Metrics

| Metric | Description |
|--------|-------------|
| ECE | Expected Calibration Error |
| MCE | Maximum Calibration Error |

### Usage

```python
from src.utils.metrics import RecommendationMetrics

metrics = RecommendationMetrics()
results = metrics.evaluate_all(
    user_recommendations,
    user_ground_truth,
    k_values=[5, 10, 20]
)

print(f"NDCG@10: {results['ndcg@10']:.4f}")
print(f"Recall@10: {results['recall@10']:.4f}")
```

---

## References

1. **BERT**: Devlin et al. "BERT: Pre-training of Deep Bidirectional Transformers" (2019)
2. **BERTopic**: Grootendorst. "BERTopic: Neural topic modeling with a class-based TF-IDF procedure" (2022)
3. **LTGNN**: Zhang et al. "Linear-Time Graph Neural Networks for Scalable Recommendations" WWW (2024)
4. **gBCE**: "Reducing Overconfidence in Sequential Recommendation Trained with Negative Sampling" RecSys (2023)
5. **DCI Closed**: Lucchese et al. "DCI_Closed: A Fast and Memory Efficient Algorithm for Mining Frequent Closed Itemsets"

---

## License

This project is for educational and research purposes.
