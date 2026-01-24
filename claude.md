# Tourist Recommendation System Implementation Plan

## Project Overview

Build a personalized tourist recommendation system that integrates:
- MBTI personality prediction from user reviews (BERT)
- Multimodal topic modeling (BERTopic + CLIP)
- Graph Neural Networks for user-venue relationships (LTGNN)
- Hybrid recommendation engine (XGBoost + gBCE loss)

**Data Source**: Yelp Dataset (Business, Review, User, Image)

---

## Phase 1: Project Setup & Data Pipeline [COMPLETED]

### 1.1 Environment Setup
- [x] Create Python virtual environment (Python 3.10+)
- [x] Install core dependencies:
  - `torch`, `transformers` (BERT, CLIP)
  - `bertopic`, `umap-learn`, `hdbscan`, `scikit-learn`
  - `torch-geometric` (GNN)
  - `xgboost`
  - `pandas`, `numpy`, `matplotlib`, `seaborn`
- [x] Set up project structure:
  ```
  src/
  ├── data/           # Data loading and preprocessing
  ├── models/         # Model implementations
  │   ├── bert_mbti/  # MBTI classifier
  │   ├── bertopic/   # Topic modeling
  │   ├── gnn/        # Graph neural network
  │   └── hybrid/     # Final recommendation engine
  ├── utils/          # Helper functions
  ├── config/         # Configuration files
  └── notebooks/      # Experimentation notebooks
  ```

### 1.2 Data Acquisition & Preprocessing
- [x] Download Yelp Dataset (Business, Review, User, Image subsets)
- [x] Create data loaders for each subset
- [x] Implement text cleaning pipeline for reviews:
  - Remove HTML tags, special characters
  - Normalize text, handle encoding issues
  - Tokenization with 512-token limit for BERT
- [x] Create train/validation/test splits (80/10/10)
- [x] Handle class imbalance via upsampling for MBTI labels

---

## Phase 2: BERT MBTI Personality Prediction [COMPLETED]

### 2.1 MBTI Classifier Architecture
- [x] Load pre-trained BERT model (`bert-base-uncased`)
- [x] Add classification head for 16 MBTI types
- [x] Implement data collator with padding/truncation

### 2.2 Training Pipeline
- [x] Implement training loop with:
  - AdamW optimizer
  - Learning rate scheduler (warmup + linear decay)
  - Gradient clipping
- [x] Add validation metrics: accuracy, F1-score (macro/weighted)
- [x] Implement early stopping based on validation loss
- [x] Save best model checkpoint

### 2.3 Inference Module
- [x] Create batch inference pipeline for all users
- [x] Generate MBTI embeddings and predictions
- [x] Store user MBTI profiles for downstream use

---

## Phase 3: Multimodal BERTopic Modeling [COMPLETED]

### 3.1 Text Embedding Pipeline
- [x] Use Sentence-BERT for review embeddings
- [x] Aggregate multiple reviews per venue

### 3.2 Image Embedding Pipeline
- [x] Load CLIP model for image embeddings
- [x] Process venue images through CLIP vision encoder
- [x] Handle venues with missing images (fallback strategy)

### 3.3 Multimodal Fusion
- [x] Concatenate/fuse text and image embeddings
- [x] Apply UMAP for dimensionality reduction (preserve local structure)

### 3.4 Topic Extraction
- [x] Apply K-Means clustering for semantic topic groups
- [x] Extract topic representations using c-TF-IDF
- [x] Generate human-readable topic labels

### 3.5 Geospatial Clustering
- [x] Encode venue coordinates as vectors
- [x] Apply HDBSCAN for location-based clustering
- [x] Create region identifiers (e.g., "Central Paris", "Downtown")

---

## Phase 4: Graph Construction & Linear-Time GNN [COMPLETED]

### 4.1 Heterogeneous Graph Construction
- [x] Define node types:
  - `User` nodes (features: MBTI embedding, user metadata)
  - `Venue` nodes (features: topic embedding, business attributes)
  - `Context` nodes (Time, Weather, Location)
- [x] Define edge types:
  - `User-Venue`: weighted by visit frequency/ratings
  - `Venue-Topic`: association strength
  - `Venue-Context`: seasonal/temporal relevance
- [x] Build graph using PyTorch Geometric

### 4.2 LTGNN Implementation
- [x] Implement single propagation layer (avoid over-smoothing)
- [x] Implement fixed-point iteration for multi-hop information
- [x] Implement EVR (Expected Variance Reduction) sampling:
  - Neighbor sampling with variance reduction
  - Control neighbor size during aggregation
- [x] Ensure linear complexity O(|E|)

### 4.3 GNN Training
- [x] Define link prediction task (user-venue interaction)
- [x] Implement negative sampling strategy
- [x] Train with BCE loss initially
- [x] Generate node embeddings for downstream use

---

## Phase 5: Hybrid Recommendation Engine [COMPLETED]

### 5.1 DCI Closed Algorithm
- [x] Implement frequent closed itemset mining
- [x] Mine user visit history for behavioral patterns
- [x] Create compact user profiles (no superset with same support)

### 5.2 Feature Synthesis
- [x] Combine features for XGBoost:
  - GNN user/venue embeddings
  - MBTI personality features
  - Topic embeddings
  - Context features (current time, weather, location)
  - DCI closed itemset patterns

### 5.3 XGBoost Refinement Model
- [x] Train XGBoost for final ranking
- [x] Implement gBCE (generalized BCE) loss:
  ```
  gBCE = BCE + calibration_penalty(t=0.8)
  ```
- [x] Tune calibration parameter `t` to reduce overconfidence
- [x] Implement hard negative mining for improved discrimination

### 5.4 Uncertainty Estimation (Optional Enhancement)
- [x] Add Monte Carlo dropout for uncertainty quantification
- [x] Diversify recommendations based on confidence scores

---

## Phase 6: Evaluation & Optimization [COMPLETED]

### 6.1 Evaluation Metrics
- [x] Implement standard metrics:
  - Precision@K, Recall@K, NDCG@K
  - MRR (Mean Reciprocal Rank)
  - Hit Rate
- [x] Implement diversity metrics:
  - Coverage (catalog coverage)
  - Intra-list diversity
- [x] Implement calibration metrics:
  - Expected Calibration Error (ECE)

### 6.2 Ablation Studies
- [x] Compare with baseline (collaborative filtering)
- [x] Measure impact of each component:
  - With/without MBTI
  - With/without multimodal topics
  - With/without gBCE loss

### 6.3 Hyperparameter Tuning
- [x] Tune BERT learning rate, batch size
- [x] Tune UMAP n_neighbors, min_dist
- [x] Tune GNN hidden dimensions, propagation iterations
- [x] Tune XGBoost depth, learning rate, n_estimators

---

## Phase 7: API & Deployment

### 7.1 Inference API
- [ ] Create FastAPI endpoints:
  - `POST /predict-mbti`: Predict user personality from reviews
  - `POST /recommend`: Get personalized venue recommendations
  - `GET /venue/{id}/topics`: Get venue topic profile
- [ ] Implement caching for embeddings and predictions

### 7.2 Model Serving
- [ ] Export models to ONNX for faster inference
- [ ] Set up model versioning
- [ ] Create Docker container for deployment

---

## Technical Specifications

| Component | Technology | Purpose |
|-----------|------------|---------|
| MBTI Classifier | BERT (bert-base-uncased) | User personality prediction |
| Topic Modeling | BERTopic + CLIP | Multimodal venue understanding |
| Graph Learning | PyTorch Geometric (LTGNN) | Relational pattern learning |
| Final Ranking | XGBoost + gBCE | Confidence-calibrated recommendations |
| Pattern Mining | DCI Closed | Compact behavioral profiles |
| Dimensionality Reduction | UMAP | Preserve semantic structure |
| Clustering | K-Means (topics), HDBSCAN (geo) | Grouping and segmentation |

---

## Execution Order

1. **Start with Phase 1**: Set up environment and data pipeline
2. **Phase 2 & 3 in parallel**: MBTI and BERTopic can be developed independently
3. **Phase 4 depends on 2 & 3**: Graph construction needs embeddings
4. **Phase 5 depends on 4**: Hybrid engine needs GNN outputs
5. **Phase 6 throughout**: Continuous evaluation as components are built
6. **Phase 7 last**: Deployment after system is validated

---

## Key Files to Create

```
src/
├── data/
│   ├── __init__.py
│   ├── yelp_loader.py          # Load Yelp dataset
│   ├── preprocessor.py         # Text cleaning, tokenization
│   └── sampler.py              # Upsampling, negative sampling
├── models/
│   ├── __init__.py
│   ├── bert_mbti/
│   │   ├── model.py            # BERT classifier
│   │   ├── trainer.py          # Training loop
│   │   └── inference.py        # Batch prediction
│   ├── bertopic/
│   │   ├── multimodal.py       # Text + Image fusion
│   │   ├── topic_extractor.py  # BERTopic pipeline
│   │   └── geo_cluster.py      # HDBSCAN for locations
│   ├── gnn/
│   │   ├── graph_builder.py    # Heterogeneous graph
│   │   ├── ltgnn.py            # Linear-Time GNN
│   │   └── evr_sampler.py      # Variance-reduced sampling
│   └── hybrid/
│       ├── dci_closed.py       # Itemset mining
│       ├── xgboost_ranker.py   # Final ranking model
│       └── gbce_loss.py        # Confidence calibration
├── utils/
│   ├── metrics.py              # Evaluation metrics
│   └── config.py               # Hyperparameters
├── api/
│   └── main.py                 # FastAPI endpoints
└── notebooks/
    ├── 01_data_exploration.ipynb
    ├── 02_mbti_training.ipynb
    ├── 03_topic_modeling.ipynb
    ├── 04_gnn_training.ipynb
    └── 05_hybrid_evaluation.ipynb
```

---

## Dependencies (requirements.txt)

```
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.0
bertopic>=0.15.0
umap-learn>=0.5.0
hdbscan>=0.8.0
scikit-learn>=1.3.0
torch-geometric>=2.3.0
xgboost>=1.7.0
pandas>=2.0.0
numpy>=1.24.0
fastapi>=0.100.0
uvicorn>=0.22.0
python-multipart>=0.0.6
pillow>=10.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
```
