# Tourist Recommendation System Implementation Plan

## Project Overview

Build a personalized tourist recommendation system that integrates:
- MBTI personality prediction from user reviews (BERT)
- Multimodal topic modeling (BERTopic + CLIP)
- Graph Neural Networks for user-venue relationships (LTGNN)
- Hybrid recommendation engine (XGBoost + gBCE loss)

**Data Sources**: Yelp Dataset (Business, Review, User, Image) + Kaggle MBTI Dataset
**Thesis Reference**: Omer Amac's MSc thesis (methodology corrections applied)

---

## Critical Methodology Fix (vs. Omer's Thesis)

Omer's original approach had **data leakage**: upsampling BEFORE train/test split inflated MBTI accuracy to ~94%. Our corrected approach:

| Step | Omer (Flawed) | Corrected |
|------|---------------|-----------|
| 1 | Upsample all data | Split first (80/10/10) |
| 2 | Then split | Upsample training set only |
| Result | ~94% accuracy (inflated) | ~69% accuracy (honest) |

Script: `scripts/compare_methodology.py` generates LaTeX comparison tables.

---

## Phase 1: Project Setup & Data Pipeline [COMPLETED]

### 1.1 Environment Setup
- [x] Python virtual environment (Python 3.10+)
- [x] All dependencies installed (torch, transformers, torch-geometric, xgboost, etc.)
- [x] Project structure established

### 1.2 Data Acquisition & Preprocessing
- [x] Yelp Dataset loaded (2.8M training reviews, 354K val reviews)
- [x] 235,643 unique users, 85,814 unique venues
- [x] Text cleaning pipeline with 512-token limit
- [x] Train/val/test splits (80/10/10)
- [x] Kaggle MBTI dataset (`data/raw/mbti_1.csv`) for BERT training

---

## Phase 2: BERT MBTI Personality Prediction [COMPLETED]

### Training Details
- **Data**: Kaggle MBTI dataset (8,675 users, ~411K posts after `|||` split)
- **Split**: train_test_split FIRST, then upsample training set only
- **Architecture**: `bert-base-uncased` + multi-label classification head (4 binary: E/I, S/N, T/F, J/P)
- **Hardware optimizations** (RTX 4060 Laptop, 8GB VRAM):
  - Gradient checkpointing on BERT backbone
  - `num_workers=0` (Windows multiprocessing limitation)
  - Plain Python loop for post explosion (avoids pyarrow 6GB realloc)

### Training Results
- Epoch 1: Val Acc 0.6554
- Epoch 2: Val Acc 0.6803
- Epoch 3: Val Acc 0.6788
- **Epoch 4: Val Acc 0.6926** (best, early stopping triggered)
- Training time: ~2h per epoch at ~4.6 it/s
- Checkpoint: `models/checkpoints/bert_mbti/best_model.pt`

### Key Scripts
- `scripts/train_mbti.py` - Training pipeline (rewritten for Kaggle data)
- `src/models/bert_mbti/trainer.py` - Training loop with gradient checkpointing
- `src/models/bert_mbti/model.py` - MBTIMultiLabelClassifier

---

## Phase 3: Multimodal BERTopic Modeling [COMPLETED]

### Components
- [x] Sentence-BERT review embeddings with per-venue aggregation
- [x] CLIP image embeddings with missing-image fallback
- [x] Multimodal fusion (text + image concatenation)
- [x] UMAP dimensionality reduction
- [x] K-Means topic clustering with c-TF-IDF labels
- [x] HDBSCAN geospatial clustering

### MBTI-Informed Embeddings (New)
- `MBTIEmbedder` replaces generic sentence-transformers with BERT-MBTI CLS tokens
- Implements BERTopic's `.encode()` protocol
- Config flags: `use_mbti_embeddings`, `mbti_checkpoint_path`

### Key Scripts
- `scripts/extract_topics_mbti.py` - Topic extraction with MBTI embeddings
- `src/models/bertopic/mbti_embedder.py` - MBTIEmbedder class
- `src/models/bertopic/topic_extractor.py` - BERTopic pipeline (updated)

---

## Phase 4: Graph Construction & Linear-Time GNN [IN PROGRESS]

### 4.1 Graph Construction [COMPLETED]
- [x] Heterogeneous graph with User (235K) and Venue (86K) nodes
- [x] 2.8M user-venue edges weighted by star ratings
- [x] Embedding reconciliation (pad/truncate to match node counts)
- [x] Vectorized validation edge construction

### 4.2 LTGNN Architecture [COMPLETED - REWRITTEN]

**Critical optimization**: Original dense gather/scatter OOM'd on 8GB GPU.

| Component | Before (OOM) | After (Fixed) |
|-----------|-------------|---------------|
| Convolution | `x[row]` creates [5.6M, 128] = 2.9GB | `torch.sparse.mm(adj, x)` = ~70MB |
| Per-batch encodes | 3,460 full-graph passes/epoch | 2 passes/epoch (1 cached + 1 gradient) |
| Epoch time | ~4 hours (estimated) | ~40 seconds |
| Memory per conv | ~2.9GB | ~70MB |

Architecture:
- `LightGCNConv`: Sparse matrix multiplication (no learnable weights)
- `FixedPointLayer`: 5 iterations with learnable weighted combination
- `LTGNN.encode()`: Projects user/venue features -> sparse GNN -> output projection
- Sparse adjacency matrix cached after first build

### 4.3 GNN Training [IN PROGRESS]

Training strategy (per epoch):
1. **Phase 1**: Encode all 321K nodes ONCE with `no_grad` (sparse conv, ~1-2 sec)
2. **Phase 2**: Train link predictor on 692 mini-batches using cached embeddings (~30 sec)
3. **Phase 3**: One full forward+backward to update GNN encoder weights (~2 sec)

Config: 15 epochs max, patience=5 early stopping, batch_size=4096, 5 fixed-point iterations

### Key Files
- `src/models/gnn/ltgnn.py` - LTGNN with sparse convolution
- `src/models/gnn/trainer.py` - Cached-embedding training loop
- `src/models/gnn/graph_builder.py` - Heterogeneous graph construction
- `src/models/gnn/evr_sampler.py` - EVR sampling and mini-batch loader
- `src/models/gnn/hetero_gnn.py` - HeteroSAGEConv (alternative architecture)
- `src/models/gnn/hetero_trainer.py` - Heterogeneous GNN trainer
- `scripts/train_gnn.py` - GNN training script
- `scripts/train_hetero_gnn.py` - Heterogeneous GNN training script

---

## Phase 5: Hybrid Recommendation Engine [COMPLETED - CODE READY]

### Components
- [x] DCI Closed itemset mining for behavioral patterns
- [x] Feature synthesis (GNN embeddings + MBTI + topics + context)
- [x] XGBoost ranker with gBCE loss (t=0.8)
- [x] PersonalityScorer for MBTI-to-vector conversion
- [x] Hard negative mining for discrimination

### Key Files
- `src/models/hybrid/dci_closed.py` - Frequent closed itemset mining
- `src/models/hybrid/xgboost_ranker.py` - XGBoost with personality features
- `src/models/hybrid/gbce_loss.py` - gBCE calibration loss
- `src/models/hybrid/personality_scorer.py` - MBTI vector conversion
- `scripts/evaluate_hybrid.py` - Full evaluation pipeline with ablation studies

---

## Phase 6: Evaluation & Optimization [COMPLETED - CODE READY]

### Metrics Implemented
- Precision@K, Recall@K, NDCG@K, MRR, Hit Rate
- Coverage, Intra-list diversity
- Expected Calibration Error (ECE)

### Ablation Studies (in `scripts/evaluate_hybrid.py`)
- With/without MBTI personality features
- With/without multimodal topics
- With/without gBCE loss
- Methodology comparison (Omer vs. corrected)
- Generates LaTeX tables for thesis

---

## Phase 7: API & Deployment [NOT STARTED]

- [ ] FastAPI endpoints (predict-mbti, recommend, venue topics)
- [ ] ONNX model export
- [ ] Docker container

---

## Pipeline Execution Order

```bash
# Step 1: Train BERT MBTI classifier (DONE - ~8 hours)
python scripts/train_mbti.py

# Step 2: Extract topics with MBTI embeddings
python scripts/extract_topics_mbti.py

# Step 3: Train LTGNN (IN PROGRESS - ~10 minutes with sparse conv)
python scripts/train_gnn.py

# Step 4: Evaluate hybrid system with ablation studies
python scripts/evaluate_hybrid.py

# Step 5: Generate methodology comparison tables
python scripts/compare_methodology.py
```

---

## Hardware Constraints & Optimizations

**Target**: NVIDIA RTX 4060 Laptop GPU (8GB VRAM), Windows

| Issue | Solution |
|-------|----------|
| BERT OOM on forward pass | `gradient_checkpointing_enable()` |
| CUDA memory fragmentation | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| Windows multiprocessing crash | `num_workers=0` in DataLoader |
| pyarrow 6GB realloc on explode | Plain Python loop instead of `df.explode()` |
| GNN dense gather 2.9GB tensor | `torch.sparse.mm(adj, x)` (40x reduction) |
| 3,460 full-graph encodes/epoch | Cache embeddings, encode once per epoch |
| Embedding/node count mismatch | `_reconcile_embeddings()` pads or truncates |

---

## Project Structure

```
src/
  data/
    __init__.py
    yelp_loader.py              # Load Yelp dataset
    preprocessor.py             # Text cleaning, tokenization
    sampler.py                  # Upsampling, negative sampling
  models/
    __init__.py
    bert_mbti/
      model.py                  # BERT multi-label MBTI classifier
      trainer.py                # Training loop with gradient checkpointing
      inference.py              # Batch prediction
    bertopic/
      __init__.py
      multimodal.py             # Text + Image fusion
      topic_extractor.py        # BERTopic pipeline (MBTI-aware)
      geo_cluster.py            # HDBSCAN for locations
      mbti_embedder.py          # MBTI CLS token embedder for BERTopic
    gnn/
      __init__.py
      graph_builder.py          # Heterogeneous graph construction
      ltgnn.py                  # LTGNN with sparse convolution
      evr_sampler.py            # EVR sampling and mini-batch loader
      trainer.py                # Cached-embedding training loop
      hetero_gnn.py             # HeteroSAGEConv (alternative)
      hetero_trainer.py         # Heterogeneous GNN trainer
    hybrid/
      __init__.py
      dci_closed.py             # Frequent closed itemset mining
      xgboost_ranker.py         # XGBoost with personality features
      gbce_loss.py              # gBCE calibration loss
      personality_scorer.py     # MBTI-to-vector conversion
  utils/
    metrics.py                  # Evaluation metrics (Precision@K, NDCG@K, etc.)
    helpers.py                  # Setup logging, seeding, checkpoints
  config/
    settings.py                 # All hyperparameters and paths

scripts/
  train_mbti.py                 # BERT MBTI training (Kaggle data)
  train_gnn.py                  # LTGNN training (sparse, cached)
  train_hetero_gnn.py           # Heterogeneous GNN training
  extract_topics_mbti.py        # BERTopic with MBTI embeddings
  evaluate_hybrid.py            # Full evaluation + ablation + LaTeX tables
  compare_methodology.py        # Omer vs. corrected methodology comparison

results/                        # Output directory for evaluation results
```

---

## Technical Specifications

| Component | Technology | Key Detail |
|-----------|------------|------------|
| MBTI Classifier | BERT (`bert-base-uncased`) | 4 binary heads, gradient checkpointing |
| Topic Modeling | BERTopic + CLIP | MBTI-informed embeddings optional |
| Graph Convolution | Sparse LightGCN | `torch.sparse.mm`, cached adjacency |
| Fixed-Point Iteration | 5 iterations, alpha=0.5 | Learnable iteration weights |
| Final Ranking | XGBoost + gBCE (t=0.8) | Personality features + GNN embeddings |
| Pattern Mining | DCI Closed | Compact behavioral profiles |
| Clustering | K-Means (topics), HDBSCAN (geo) | Grouping and segmentation |
