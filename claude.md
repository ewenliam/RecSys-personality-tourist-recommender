# Tourist Recommendation System Implementation Plan

> **NOTE (final state):** The phase-by-phase plan below is the original roadmap
> and is kept for history. Several components were revised after empirical
> evidence; the accurate description of the system that produced the final
> results is in **"Final Architecture & Results (as built)"** immediately below.
> Where the phase notes and the as-built section disagree, the as-built section
> is correct.

## Project Overview

Build a personalized tourist recommendation system that integrates:
- MBTI personality prediction from user reviews (BERT)
- Personality-informed topic modeling (BERTopic with BERT-MBTI CLS embeddings)
- Graph Neural Network for user-venue relationships (heterogeneous GraphSAGE)
- Parameter-free hybrid ranking (Reciprocal Rank Fusion of four signals)

**Data Sources**: Yelp Dataset (Business, Review, User) + Kaggle MBTI Dataset
**Thesis Reference**: Omer Amac's MSc thesis (methodology corrections applied)

---

## Final Architecture & Results (as built)

This section overrides any stale claims in the phase notes below.

### What changed from the original plan
| Original plan | Final system | Why |
|---------------|--------------|-----|
| GNN: **LTGNN** (homogeneous LightGCN) | **Heterogeneous GraphSAGE** (`hetero_gnn.py`, `train_hetero_gnn.py`) | LightGCN's conv is parameter-free; GraphSAGE learns per-edge-type transforms so personality/topic features are first-class |
| Ranker: **XGBoost + gBCE** | **Reciprocal Rank Fusion (RRF)** of 4 signals | Parameter-free: nothing to overfit, no popularity shortcut |
| GNN loss: **gBCE (t=0.8)** | **BPR** + temperature ($\tau$=0.1) + hard negatives | gBCE retained only for AUC/calibration reporting |
| MBTI fix = split-then-upsample (~69%) | **User-disjoint split + per-dimension class weights**, balanced-acc selection, per-user eval | Found post-level leakage (100% val users in train) and class-prior collapse |
| Venue personality from review text | **Visitor-mean** MBTI profile | Review-text venue embeddings barely vary (cosine std ~0.04); visitor-mean is discriminative |

### Final MBTI classifier (honest, user-disjoint test)
- Per-user accuracy **0.801**, mean balanced accuracy **0.752**, exact 16-type **0.445**.
- All four axes genuinely learned (balanced acc: E/I 0.696, S/N 0.782, T/F 0.836, J/P 0.693).

### Final recommender (5 seeds, K=10, mean)
- Full hybrid (KNN content + GNN + MBTI + popularity) best on all 5 metrics:
  NDCG@10 **0.0153**, MRR **0.0238**, Hit Rate@10 **0.0524** (2.4x popularity baseline).

### The four RRF signals
1. **Content (KNN)**: cosine(user BERTopic profile, venue BERTopic PCA-64).
2. **Collaborative (GNN)**: cosine(GraphSAGE user, GraphSAGE venue).
3. **Personality (MBTI)**: cosine(user BERT-MBTI CLS, venue visitor-mean profile).
4. **Popularity**: small log-degree tie-breaker (alpha=0.01), applied after fusion.

### Reproduce the final pipeline
```bash
python scripts/train_mbti.py --epochs 4          # honest classifier (~8h)
python scripts/rebuild_pipeline.py               # topics -> user_mbti -> gnn -> robustness
```
Environment pin: **numpy==1.26.4** and `NUMBA_DISABLE_INTEL_SVML=1` (for UMAP).
Thesis sources: `docs/thesis/` (outline, WUT LaTeX project, tables, figures).

---

## Critical Methodology Fix (vs. Omer's Thesis)

Omer's original approach had **data leakage**: upsampling BEFORE train/test split inflated MBTI accuracy to ~94%. A later audit (`scripts/diagnose_mbti.py`) found a **second** leak and a class-imbalance artefact, so the final correction goes further than the original two-step fix:

| Aspect | Omer (Flawed) | Final correction |
|--------|---------------|------------------|
| Upsampling | Upsample all data, then split | Split first; **class-weighted loss** instead of upsampling |
| Split unit | Post level (100% val users also in train) | **User-disjoint** split |
| Model selection | Val loss | **Mean balanced accuracy** |
| Evaluation | Per-post accuracy | **Per-user** (majority vote) + balanced accuracy |
| Result | ~94% (inflated/leaked) | **80.1% acc / 75.2% balanced** (honest, all 4 axes learned) |

> The ~69% figure in the phase notes was an intermediate result before the full audit.

Scripts: `scripts/diagnose_mbti.py` (audit), `scripts/generate_thesis_assets.py` (LaTeX tables/figures).

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

> Reflects the **final** system (see "Final Architecture & Results" at the top).

| Component | Technology | Key Detail |
|-----------|------------|------------|
| MBTI Classifier | BERT (`bert-base-uncased`) | 4 binary heads, class-weighted loss, user-disjoint eval |
| Topic Modeling | BERTopic (BERT-MBTI CLS) | personality-informed venue embeddings |
| Graph Model | Heterogeneous GraphSAGE | per-edge-type learned transforms, 1 layer |
| GNN Training | BPR + temperature (0.1) | hard negatives; gBCE kept for calibration reporting only |
| Final Ranking | Reciprocal Rank Fusion (RRF) | parameter-free fusion of 4 signals, k=60 |
| Personality signal | Visitor-mean MBTI profile | venue = mean MBTI of its visitors |

> Earlier-explored components not in the final pipeline: LTGNN (`ltgnn.py`),
> XGBoost ranker, gBCE training loss, DCI-Closed mining, CLIP multimodal,
> HDBSCAN geo-clustering. Kept in the repo as alternatives/history.
