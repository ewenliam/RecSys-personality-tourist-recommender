# Tourist Recommendation System - Complete Technical Documentation

This document provides a comprehensive explanation of every component in the system, including architecture details, hyperparameters, tuning strategies, and implementation decisions.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Project Structure](#2-project-structure)
3. [Configuration & Hyperparameters](#3-configuration--hyperparameters)
4. [BERT MBTI Personality Classifier](#4-bert-mbti-personality-classifier)
5. [Multimodal BERTopic Modeling](#5-multimodal-bertopic-modeling)
6. [Linear-Time Graph Neural Network (LTGNN)](#6-linear-time-graph-neural-network-ltgnn)
7. [Hybrid Recommendation Engine](#7-hybrid-recommendation-engine)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [Training Pipeline](#9-training-pipeline)
10. [Tuning Guide](#10-tuning-guide)

---

## 1. System Overview

This system combines multiple machine learning paradigms to deliver personalized tourist venue recommendations:

```
User Reviews ──► BERT MBTI ──► Personality Embeddings ──┐
                                                        │
Venue Reviews ──► BERTopic ──► Topic Embeddings ────────┼──► Heterogeneous Graph ──► GNN ──► XGBoost ──► Recommendations
                                                        │
Venue Images ──► CLIP ──► Image Embeddings ─────────────┤
                                                        │
User History ──► DCI Closed ──► Behavioral Patterns ────┘
```

**Key Innovation Points:**
- **MBTI-based personalization**: Users are profiled by personality type
- **Multimodal understanding**: Venues are represented by both text and images
- **Graph-based learning**: Captures complex user-venue-context relationships
- **Calibrated predictions**: gBCE loss prevents overconfident recommendations

---

## 2. Project Structure

```
src/
├── config/
│   └── settings.py              # All hyperparameters and configurations
│
├── data/
│   ├── yelp_loader.py           # Yelp dataset loading utilities
│   ├── preprocessor.py          # Text cleaning and tokenization
│   └── sampler.py               # Upsampling and negative sampling
│
├── models/
│   ├── bert_mbti/
│   │   ├── model.py             # BERT classifier architecture
│   │   ├── trainer.py           # Training loop with early stopping
│   │   └── inference.py         # Batch prediction pipeline
│   │
│   ├── bertopic/
│   │   ├── multimodal.py        # Text + Image embedding fusion
│   │   ├── topic_extractor.py   # BERTopic pipeline with c-TF-IDF
│   │   └── geo_cluster.py       # HDBSCAN geospatial clustering
│   │
│   ├── gnn/
│   │   ├── ltgnn.py             # Linear-Time GNN architecture
│   │   ├── graph_builder.py     # Heterogeneous graph construction
│   │   ├── trainer.py           # GNN training with link prediction
│   │   └── evr_sampler.py       # Expected Variance Reduction sampling
│   │
│   └── hybrid/
│       ├── xgboost_ranker.py    # Final ranking model
│       ├── dci_closed.py        # Frequent closed itemset mining
│       └── gbce_loss.py         # Calibrated loss function
│
└── utils/
    └── metrics.py               # Precision@K, NDCG, diversity metrics
```

---

## 3. Configuration & Hyperparameters

**File:** `src/config/settings.py`

### 3.1 Data Configuration

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `max_review_length` | 512 | Maximum tokens per review (BERT limit) |
| `min_review_length` | 10 | Filter out too-short reviews |
| `train_split` | 0.8 | 80% for training |
| `val_split` | 0.1 | 10% for validation |
| `test_split` | 0.1 | 10% for testing |
| `min_reviews_per_user` | 5 | Users with fewer reviews are excluded |
| `min_reviews_per_business` | 3 | Venues with fewer reviews are excluded |
| `upsample_minority` | True | Handle MBTI class imbalance |

### 3.2 BERT Configuration

| Parameter | Value | Tuning Notes |
|-----------|-------|--------------|
| `model_name` | "bert-base-uncased" | Standard 12-layer BERT |
| `max_length` | 512 | Full context window |
| `batch_size` | 16 | Fits on 16GB GPU |
| `learning_rate` | 2e-5 | Standard for fine-tuning BERT |
| `weight_decay` | 0.01 | Regularization |
| `num_epochs` | 10 | Usually converges by epoch 5-7 |
| `warmup_ratio` | 0.1 | 10% warmup prevents early divergence |
| `gradient_clip` | 1.0 | Prevents exploding gradients |
| `early_stopping_patience` | 3 | Stop if no improvement for 3 epochs |
| `num_labels` | 16 | 16 MBTI personality types |

### 3.3 BERTopic Configuration

| Parameter | Value | Tuning Notes |
|-----------|-------|--------------|
| `embedding_model` | "all-MiniLM-L6-v2" | Fast, 384-dim embeddings |
| `clip_model` | "ViT-B-32" | Standard CLIP, 512-dim |
| `umap_n_neighbors` | 15 | Local structure preservation |
| `umap_n_components` | 5 | Reduced dimensionality |
| `umap_min_dist` | 0.0 | Tight clusters preferred |
| `hdbscan_min_cluster_size` | 10 | Minimum points per cluster |
| `hdbscan_min_samples` | 5 | Core point threshold |
| `kmeans_n_clusters` | 50 | Alternative to HDBSCAN |

### 3.4 GNN Configuration

| Parameter | Value | Tuning Notes |
|-----------|-------|--------------|
| `hidden_dim` | 128 | Internal representation size |
| `embedding_dim` | 64 | Final user/venue embedding |
| `num_layers` | 2 | Prevents over-smoothing |
| `dropout` | 0.2 | Regularization |
| `learning_rate` | 0.001 | Higher than BERT (simpler model) |
| `batch_size` | 1024 | Mini-batch for large graphs |
| `num_epochs` | 100 | GNNs need more epochs |
| `num_neighbors` | [10, 5] | Sampling neighbors per layer |
| `fixed_point_iterations` | 10 | Multi-hop message passing |
| `evr_sample_size` | 20 | Variance-reduced sampling |

### 3.5 Hybrid Configuration

| Parameter | Value | Tuning Notes |
|-----------|-------|--------------|
| `xgb_max_depth` | 6 | Tree depth (prevents overfitting) |
| `xgb_learning_rate` | 0.1 | Shrinkage rate |
| `xgb_n_estimators` | 100 | Number of boosting rounds |
| `xgb_subsample` | 0.8 | Row sampling |
| `xgb_colsample_bytree` | 0.8 | Column sampling |
| `gbce_calibration_t` | 0.8 | Temperature for calibration |
| `min_support` | 0.01 | 1% minimum pattern frequency |
| `max_itemset_size` | 5 | Maximum items in a pattern |

---

## 4. BERT MBTI Personality Classifier

**Files:** `src/models/bert_mbti/model.py`, `trainer.py`, `inference.py`

### 4.1 Architecture

```
Input Text (Review)
       │
       ▼
┌─────────────────────────────────────┐
│  BERT-base-uncased                  │
│  - 12 Transformer layers            │
│  - 768 hidden units                 │
│  - 12 attention heads               │
└─────────────────────────────────────┘
       │
       ▼ [CLS] token embedding (768-dim)
       │
┌─────────────────────────────────────┐
│  Dropout (p=0.1)                    │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Linear Layer (768 → 16)            │
│  16 MBTI types output               │
└─────────────────────────────────────┘
       │
       ▼
   Softmax → MBTI Type
```

### 4.2 MBTI Types Predicted

The 16 Myers-Briggs personality types:
- **Analysts**: INTJ, INTP, ENTJ, ENTP
- **Diplomats**: INFJ, INFP, ENFJ, ENFP
- **Sentinels**: ISTJ, ISFJ, ESTJ, ESFJ
- **Explorers**: ISTP, ISFP, ESTP, ESFP

### 4.3 Multi-Label Variant

For better granularity, a multi-label classifier predicts each MBTI dimension independently:

| Dimension | Options | Description |
|-----------|---------|-------------|
| E/I | Extraversion / Introversion | Energy source |
| S/N | Sensing / Intuition | Information processing |
| T/F | Thinking / Feeling | Decision making |
| J/P | Judging / Perceiving | Lifestyle |

### 4.4 Training Details

**Optimizer:** AdamW with weight decay
```python
optimizer = AdamW(
    model.parameters(),
    lr=2e-5,
    weight_decay=0.01,
    betas=(0.9, 0.999),
    eps=1e-8
)
```

**Learning Rate Schedule:** OneCycleLR
- Warmup: First 10% of steps (linear increase)
- Decay: Remaining 90% (cosine annealing)

**Loss Function:** CrossEntropyLoss
```python
loss = F.cross_entropy(logits, labels)
```

**Gradient Clipping:**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Early Stopping:** Training stops if validation loss doesn't improve for 3 consecutive epochs.

### 4.5 Inference Modes

1. **Single Review:** Direct prediction from one review
2. **Multi-Review Aggregation:**
   - `majority`: Most frequent predicted type across reviews
   - `mean_probs`: Average probabilities, then argmax

**Output:**
- MBTI type (string, e.g., "INTJ")
- Confidence score (0-1)
- Embedding (768-dim vector for downstream use)

---

## 5. Multimodal BERTopic Modeling

**Files:** `src/models/bertopic/multimodal.py`, `topic_extractor.py`, `geo_cluster.py`

### 5.1 Text Embedding Pipeline

**Model:** Sentence-Transformers `all-MiniLM-L6-v2`
- Output dimension: 384
- Speed: ~14,000 sentences/sec on GPU

**Aggregation Strategies:**
```python
def aggregate(self, texts: List[str], strategy: str):
    embeddings = self.model.encode(texts)
    if strategy == "mean":
        return embeddings.mean(axis=0)
    elif strategy == "max":
        return embeddings.max(axis=0)
    elif strategy == "first":
        return embeddings[0]
```

### 5.2 Image Embedding Pipeline

**Model:** CLIP ViT-B/32
- Output dimension: 512
- Input: PIL Image or image path

**Processing:**
```python
image = preprocess(Image.open(path)).unsqueeze(0)
with torch.no_grad():
    embedding = model.encode_image(image)
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # L2 normalize
```

**Fallback:** Zero vectors for failed image loads (missing/corrupted images)

### 5.3 Multimodal Fusion

**Method 1: Concatenation (Default)**
```
Text (384-dim) + Image (512-dim) → Fused (896-dim)
```

**Method 2: Weighted Sum**
```python
fused = text_weight * text_emb + (1 - text_weight) * image_emb
# Requires projection to same dimension first
```

**Method 3: Attention Fusion**
```python
# Learnable attention weights
attention = softmax(W @ [text_emb, image_emb])
fused = attention[0] * text_emb + attention[1] * image_emb
```

### 5.4 Dimensionality Reduction (UMAP)

```python
umap_model = UMAP(
    n_neighbors=15,       # Local neighborhood size
    n_components=5,       # Target dimensions
    min_dist=0.0,         # Tighter clusters
    metric='cosine',      # Semantic similarity
    random_state=42       # Reproducibility
)
```

**Why these values?**
- `n_neighbors=15`: Balances local vs global structure
- `n_components=5`: Enough for clustering, reduces noise
- `min_dist=0.0`: Allows tight clusters for distinct topics

### 5.5 Topic Clustering

**Primary: HDBSCAN**
```python
hdbscan_model = HDBSCAN(
    min_cluster_size=10,  # Minimum topic size
    min_samples=5,        # Core sample threshold
    metric='euclidean',
    cluster_selection_method='eom'
)
```

**Alternative: K-Means**
```python
kmeans_model = KMeans(
    n_clusters=50,
    random_state=42,
    n_init=10
)
```

### 5.6 Topic Representation (c-TF-IDF)

Class-based TF-IDF creates topic labels:

```
c-TF-IDF = tf(word, class) × log(1 + A / freq(word))

Where:
- tf(word, class) = frequency in topic
- A = average words per topic
- freq(word) = documents containing word
```

**Topic Labeling:** KeyBERT with Maximal Marginal Relevance
```python
representation_model = KeyBERTInspired()
mmr = MaximalMarginalRelevance(diversity=0.3)
```

### 5.7 Geospatial Clustering

**Coordinate Transformation:** Lat/Lon → 3D Cartesian
```python
def to_cartesian(lat, lon):
    lat_rad, lon_rad = radians(lat), radians(lon)
    x = R * cos(lat_rad) * cos(lon_rad)
    y = R * cos(lat_rad) * sin(lon_rad)
    z = R * sin(lat_rad)
    return x, y, z
```

**Distance:** Haversine formula (great-circle distance)
```python
def haversine(lat1, lon1, lat2, lon2):
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))
```

**Clustering:** HDBSCAN on 3D coordinates
- Output: Region IDs, center coordinates, radius estimates

### 5.8 Context Encoding

**Time Encoding:**
| Period | Hours | Embedding |
|--------|-------|-----------|
| Morning | 6-11 | 16-dim random |
| Afternoon | 11-17 | 16-dim random |
| Evening | 17-21 | 16-dim random |
| Night | 21-6 | 16-dim random |

**Season Encoding:**
| Season | Months | Embedding |
|--------|--------|-----------|
| Spring | Mar-May | 16-dim random |
| Summer | Jun-Aug | 16-dim random |
| Fall | Sep-Nov | 16-dim random |
| Winter | Dec-Feb | 16-dim random |

**Weather Encoding:**
- Weather types: sunny, cloudy, rainy, snowy, windy, foggy
- Temperature bins: freezing (<0°C), cold (0-10), mild (10-20), warm (20-30), hot (>30)

---

## 6. Linear-Time Graph Neural Network (LTGNN)

**Files:** `src/models/gnn/ltgnn.py`, `graph_builder.py`, `trainer.py`, `evr_sampler.py`

### 6.1 Graph Structure

**Node Types:**

| Type | Features | Dimension |
|------|----------|-----------|
| User | MBTI embedding | 768 |
| Venue | Topic embedding | 896 (fused) |
| Topic | c-TF-IDF vector | Variable |
| Region | Geo embedding | 3 (x, y, z) |

**Edge Types:**

| Source | Relation | Target | Weight |
|--------|----------|--------|--------|
| User | visits | Venue | Rating / frequency |
| Venue | has_topic | Topic | Topic probability |
| Venue | in_region | Region | Binary (1.0) |
| Venue | rev_visits | User | (reverse edge) |

### 6.2 LightGCN Convolution

Simplified graph convolution without feature transformation:

```python
class LightGCNConv(MessagePassing):
    def forward(self, x, edge_index, edge_weight=None):
        # Compute normalization: D^(-1/2)
        deg = degree(edge_index[0], x.size(0))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        # Symmetric normalization
        norm = deg_inv_sqrt[edge_index[0]] * deg_inv_sqrt[edge_index[1]]

        if edge_weight is not None:
            norm = norm * edge_weight

        return self.propagate(edge_index, x=x, norm=norm)
```

### 6.3 Fixed-Point Layer

Multi-hop information aggregation via iteration:

```python
class FixedPointLayer(nn.Module):
    def forward(self, x, edge_index, edge_weight=None):
        h = x
        for i in range(self.num_iterations):  # 10 iterations
            # Aggregate from neighbors
            h_new = self.conv(h, edge_index, edge_weight)
            h_new = self.dropout(h_new)

            # Weighted update with residual
            weight = softmax(self.iteration_weights)[i]
            h = self.alpha * x + (1 - self.alpha) * (weight * h_new + (1 - weight) * h)

        return h
```

**Why Fixed-Point?**
- Avoids over-smoothing (unlike stacking many GNN layers)
- Linear complexity O(|E|)
- Learnable iteration weights adapt to graph structure

### 6.4 Full LTGNN Architecture

```
User Features (768-dim) ──► Linear ──► 128-dim ──┐
                                                 │
                                                 ▼
                                        ┌─────────────────┐
                                        │  FixedPointLayer │
                                        │  (10 iterations) │
                                        └─────────────────┘
                                                 │
Venue Features (896-dim) ──► Linear ──► 128-dim ──┘
                                                 │
                                                 ▼
                                        ┌─────────────────┐
                                        │  Output Linear   │
                                        │  (128 → 64-dim)  │
                                        └─────────────────┘
                                                 │
                                                 ▼
                                          User Embeddings (64-dim)
                                          Venue Embeddings (64-dim)
```

### 6.5 EVR Sampling (Expected Variance Reduction)

Variance-reduced neighbor sampling for efficient training:

```python
class EVRSampler:
    def sample(self, node_idx, num_samples):
        neighbors = self.adj[node_idx]
        degrees = self.degree[neighbors]

        # Importance weights: inverse degree
        importance = 1.0 / degrees
        probs = importance / importance.sum()

        # Sample with importance weights
        sampled = np.random.choice(neighbors, size=num_samples, p=probs)

        # Importance weights for unbiased estimation
        weights = 1.0 / (num_samples * probs[sampled])

        return sampled, weights
```

**Benefits:**
- Reduces variance in gradient estimates
- Maintains unbiased gradients
- Linear complexity

### 6.6 Loss Functions

**gBCE Loss (Generalized BCE):**
```python
class gBCELoss(nn.Module):
    def __init__(self, calibration_t=0.8):
        self.t = calibration_t

    def forward(self, predictions, targets):
        # Temperature scaling for calibration
        calibrated_preds = torch.pow(torch.sigmoid(predictions), 1/self.t)
        return F.binary_cross_entropy(calibrated_preds, targets)
```

**BPR Loss (Bayesian Personalized Ranking):**
```python
class BPRLoss(nn.Module):
    def forward(self, pos_scores, neg_scores):
        return -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
```

### 6.7 Training Configuration

```python
trainer = GNNTrainer(
    model=ltgnn,
    optimizer=AdamW(lr=0.001, weight_decay=0.01),
    scheduler=CosineAnnealingLR(T_max=100, eta_min=1e-6),
    loss_fn=gBCELoss(calibration_t=0.8),
    num_negative_samples=4,
    early_stopping_patience=10
)
```

---

## 7. Hybrid Recommendation Engine

**Files:** `src/models/hybrid/xgboost_ranker.py`, `dci_closed.py`, `gbce_loss.py`

### 7.1 DCI Closed Algorithm (Pattern Mining)

Mines frequent closed itemsets from user visit history:

**Closed Itemset:** An itemset is closed if no superset has the same support.

```python
class DCIClosed:
    def mine(self, transactions, min_support=0.01, max_size=5):
        """
        transactions: List of user visit histories
        min_support: Minimum frequency (1%)
        max_size: Maximum items per pattern
        """
        # Count item frequencies
        item_counts = Counter(item for t in transactions for item in t)

        # Filter by support
        min_count = len(transactions) * min_support
        frequent_items = {item for item, count in item_counts.items()
                         if count >= min_count}

        # Generate closed itemsets via DCI
        closed_itemsets = self._dci_closed(transactions, frequent_items, max_size)

        return closed_itemsets
```

**Output:** Compact user profiles representing behavioral patterns

### 7.2 Feature Synthesis

Combines all feature sources for XGBoost:

```python
class FeatureSynthesizer:
    def synthesize(self, user_id, venue_id):
        features = []

        # GNN embeddings (64-dim each)
        user_emb = self.gnn_embeddings['user'][user_id]
        venue_emb = self.gnn_embeddings['venue'][venue_id]
        features.extend(user_emb)
        features.extend(venue_emb)

        # Interaction features
        features.append(cosine_similarity(user_emb, venue_emb))
        features.extend(user_emb * venue_emb)  # Element-wise product

        # MBTI embedding (768-dim, reduced to 64)
        mbti_emb = self.mbti_embeddings[user_id]
        features.extend(self.mbti_reducer.transform(mbti_emb))

        # Topic embedding (from venue)
        topic_emb = self.topic_embeddings[venue_id]
        features.extend(topic_emb)

        # DCI patterns (binary features)
        user_patterns = self.dci_patterns[user_id]
        pattern_features = self.encode_patterns(user_patterns)
        features.extend(pattern_features)

        # Context features
        context = self.encode_context(time, weather, location)
        features.extend(context)

        return np.array(features)
```

### 7.3 XGBoost Ranking Model

```python
class XGBoostRanker:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='hist',
            device='cuda',
            eval_metric=['auc', 'logloss'],
            early_stopping_rounds=10
        )

    def fit(self, X_train, y_train, X_val, y_val):
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            verbose=True
        )
```

### 7.4 gBCE Calibration

Prevents overconfident predictions:

```python
class gBCELoss:
    def __init__(self, calibration_t=0.8):
        """
        calibration_t < 1.0: Reduces overconfidence
        calibration_t > 1.0: Increases confidence
        calibration_t = 1.0: Standard BCE
        """
        self.t = calibration_t

    def calibrate(self, predictions):
        # Temperature scaling
        return np.power(predictions, 1/self.t)
```

**Why t=0.8?**
- Reduces probability of top predictions slightly
- Better diversity in recommendations
- Prevents filter bubbles

### 7.5 Focal Loss (Alternative)

For handling class imbalance:

```python
class FocalLoss:
    def __init__(self, gamma=2.0, alpha=0.25):
        self.gamma = gamma  # Focusing parameter
        self.alpha = alpha  # Class weight

    def forward(self, predictions, targets):
        p_t = predictions * targets + (1 - predictions) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        return -self.alpha * focal_weight * log(p_t)
```

### 7.6 Calibration Methods

**1. Temperature Scaling:**
```python
calibrated = sigmoid(logits / temperature)
```

**2. Platt Scaling:**
```python
# Fit logistic regression on validation set
platt = LogisticRegression()
platt.fit(logits.reshape(-1, 1), labels)
calibrated = platt.predict_proba(logits)[:, 1]
```

**3. Isotonic Regression:**
```python
isotonic = IsotonicRegression(out_of_bounds='clip')
isotonic.fit(predictions, labels)
calibrated = isotonic.predict(predictions)
```

### 7.7 Hybrid Recommender

```python
class HybridRecommender:
    def recommend(self, user_id, context, top_k=10):
        # Get candidate venues (exclude visited)
        candidates = self.get_candidates(user_id)

        # Score each candidate
        scores = []
        for venue_id in candidates:
            features = self.synthesizer.synthesize(user_id, venue_id, context)
            score = self.xgb_model.predict_proba([features])[0, 1]
            scores.append((venue_id, score))

        # Calibrate and rank
        scores = [(v, self.calibrate(s)) for v, s in scores]
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]
```

---

## 8. Evaluation Metrics

**File:** `src/utils/metrics.py`

### 8.1 Ranking Metrics

**Precision@K:**
```python
def precision_at_k(recommended, relevant, k):
    hits = len(set(recommended[:k]) & set(relevant))
    return hits / k
```

**Recall@K:**
```python
def recall_at_k(recommended, relevant, k):
    hits = len(set(recommended[:k]) & set(relevant))
    return hits / len(relevant) if relevant else 0
```

**NDCG@K (Normalized Discounted Cumulative Gain):**
```python
def ndcg_at_k(recommended, relevant, k):
    def dcg(items, relevant):
        return sum(1/log2(i+2) for i, item in enumerate(items) if item in relevant)

    dcg_k = dcg(recommended[:k], relevant)
    idcg_k = dcg(sorted(relevant, key=lambda x: x in recommended[:k], reverse=True)[:k], relevant)

    return dcg_k / idcg_k if idcg_k > 0 else 0
```

**MRR (Mean Reciprocal Rank):**
```python
def mrr(recommended, relevant):
    for i, item in enumerate(recommended):
        if item in relevant:
            return 1 / (i + 1)
    return 0
```

**Hit Rate@K:**
```python
def hit_rate_at_k(recommended, relevant, k):
    return 1 if any(item in relevant for item in recommended[:k]) else 0
```

### 8.2 Diversity Metrics

**Intra-List Diversity:**
```python
def intra_list_diversity(items, embeddings):
    """Average pairwise distance within recommendation list"""
    distances = []
    for i, item1 in enumerate(items):
        for item2 in items[i+1:]:
            dist = 1 - cosine_similarity(embeddings[item1], embeddings[item2])
            distances.append(dist)
    return np.mean(distances) if distances else 0
```

**Catalog Coverage:**
```python
def catalog_coverage(all_recommendations, catalog):
    """Fraction of catalog items recommended at least once"""
    recommended_items = set(item for rec in all_recommendations for item in rec)
    return len(recommended_items) / len(catalog)
```

### 8.3 Calibration Metrics

**Expected Calibration Error (ECE):**
```python
def ece(predictions, labels, n_bins=10):
    """Measures calibration gap across confidence bins"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece_score = 0

    for i in range(n_bins):
        mask = (predictions >= bin_boundaries[i]) & (predictions < bin_boundaries[i+1])
        if mask.sum() > 0:
            bin_accuracy = labels[mask].mean()
            bin_confidence = predictions[mask].mean()
            bin_weight = mask.sum() / len(predictions)
            ece_score += bin_weight * abs(bin_accuracy - bin_confidence)

    return ece_score
```

---

## 9. Training Pipeline

### 9.1 Complete Training Flow

```
┌────────────────────────────────────────────────────────────────────┐
│ Phase 1: Data Preparation                                          │
│ ├── Load Yelp dataset (Business, Review, User, Image)              │
│ ├── Clean text (HTML, special chars, encoding)                     │
│ ├── Split data (80/10/10)                                          │
│ └── Upsample minority MBTI classes                                 │
└────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────┐
│ Phase 2: BERT MBTI Training (can run in parallel with Phase 3)     │
│ ├── Fine-tune BERT on MBTI-labeled reviews                         │
│ ├── Early stopping on validation loss (patience=3)                 │
│ ├── Save best checkpoint                                           │
│ └── Generate user MBTI embeddings (768-dim)                        │
└────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────┐
│ Phase 3: BERTopic + Geo Clustering (can run in parallel with 2)    │
│ ├── Generate text embeddings (Sentence-BERT)                       │
│ ├── Generate image embeddings (CLIP)                               │
│ ├── Fuse multimodal embeddings                                     │
│ ├── UMAP dimensionality reduction                                  │
│ ├── HDBSCAN topic clustering                                       │
│ ├── c-TF-IDF topic representations                                 │
│ └── HDBSCAN geospatial clustering                                  │
└────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────┐
│ Phase 4: Graph Construction & GNN Training                         │
│ ├── Build heterogeneous graph (User, Venue, Topic, Region nodes)   │
│ ├── Add edges (visits, has_topic, in_region)                       │
│ ├── Train LTGNN with link prediction task                          │
│ ├── Use gBCE loss for calibration                                  │
│ ├── EVR sampling for efficient training                            │
│ └── Generate user/venue embeddings (64-dim)                        │
└────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────┐
│ Phase 5: Hybrid Engine Training                                    │
│ ├── DCI Closed mining on user visit histories                      │
│ ├── Synthesize features (GNN + MBTI + topics + patterns)           │
│ ├── Train XGBoost ranker                                           │
│ ├── Apply gBCE calibration                                         │
│ └── Save final model                                               │
└────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────┐
│ Phase 6: Evaluation                                                │
│ ├── Compute Precision@K, Recall@K, NDCG@K                          │
│ ├── Compute diversity metrics                                      │
│ ├── Compute calibration metrics (ECE)                              │
│ └── Ablation studies                                               │
└────────────────────────────────────────────────────────────────────┘
```

### 9.2 Negative Sampling Strategy

```python
def sample_negatives(user_id, positive_venues, all_venues, num_negatives=4):
    """Sample venues user hasn't visited"""
    negative_pool = all_venues - positive_venues
    negatives = random.sample(list(negative_pool), num_negatives)
    return negatives
```

### 9.3 Mini-Batch Training (GNN)

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        user_ids, pos_venue_ids, neg_venue_ids = batch

        # Forward pass
        user_emb = model.get_user_embeddings(user_ids)
        pos_emb = model.get_venue_embeddings(pos_venue_ids)
        neg_emb = model.get_venue_embeddings(neg_venue_ids)

        # Compute scores
        pos_scores = (user_emb * pos_emb).sum(dim=1)
        neg_scores = (user_emb * neg_emb).sum(dim=1)

        # Compute loss
        loss = gbce_loss(pos_scores, neg_scores)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 10. Tuning Guide

### 10.1 Critical Parameters to Tune

| Component | Parameter | Default | Tuning Range | Impact |
|-----------|-----------|---------|--------------|--------|
| BERT | learning_rate | 2e-5 | 1e-5 to 5e-5 | Too high = instability, too low = slow convergence |
| BERT | batch_size | 16 | 8-32 | Memory vs gradient noise tradeoff |
| UMAP | n_neighbors | 15 | 5-50 | Local vs global structure |
| UMAP | min_dist | 0.0 | 0.0-0.5 | Cluster tightness |
| HDBSCAN | min_cluster_size | 10 | 5-50 | Number of topics |
| GNN | hidden_dim | 128 | 64-256 | Model capacity |
| GNN | fixed_point_iterations | 10 | 5-20 | Multi-hop range |
| XGBoost | max_depth | 6 | 3-10 | Complexity vs overfitting |
| XGBoost | learning_rate | 0.1 | 0.01-0.3 | Training speed vs precision |
| gBCE | calibration_t | 0.8 | 0.5-1.0 | Confidence calibration strength |

### 10.2 Tuning Strategies

**Grid Search (for small parameter spaces):**
```python
param_grid = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1, 0.2],
    'n_estimators': [50, 100, 200]
}
```

**Bayesian Optimization (for larger spaces):**
```python
from optuna import create_study

def objective(trial):
    params = {
        'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256]),
        'learning_rate': trial.suggest_loguniform('lr', 1e-4, 1e-2),
        'dropout': trial.suggest_uniform('dropout', 0.1, 0.5)
    }
    return train_and_evaluate(params)

study = create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

### 10.3 Ablation Study Setup

```python
ablation_configs = {
    'full': {'mbti': True, 'topics': True, 'gnn': True, 'dci': True, 'gbce': True},
    'no_mbti': {'mbti': False, 'topics': True, 'gnn': True, 'dci': True, 'gbce': True},
    'no_topics': {'mbti': True, 'topics': False, 'gnn': True, 'dci': True, 'gbce': True},
    'no_gbce': {'mbti': True, 'topics': True, 'gnn': True, 'dci': True, 'gbce': False},
    'baseline_cf': {'mbti': False, 'topics': False, 'gnn': False, 'dci': False, 'gbce': False}
}

for name, config in ablation_configs.items():
    model = build_model(config)
    metrics = evaluate(model, test_data)
    print(f"{name}: NDCG@10={metrics['ndcg@10']:.4f}")
```

### 10.4 Common Issues and Solutions

| Issue | Symptom | Solution |
|-------|---------|----------|
| BERT overfitting | Val loss increases early | Reduce epochs, increase dropout |
| GNN over-smoothing | All embeddings similar | Reduce layers, use residual connections |
| HDBSCAN too few clusters | Most points = noise (-1) | Decrease min_cluster_size |
| XGBoost overfitting | Train AUC >> Val AUC | Reduce max_depth, increase regularization |
| Poor calibration | High ECE | Tune calibration_t, use isotonic regression |

---

## Summary

This Tourist Recommendation System integrates multiple state-of-the-art techniques:

1. **BERT MBTI Classifier**: Personality-aware user modeling
2. **Multimodal BERTopic**: Rich venue understanding from text + images
3. **LTGNN**: Efficient graph-based relationship learning
4. **XGBoost + gBCE**: Calibrated final ranking

Key design principles:
- **Modularity**: Each component can be trained/tuned independently
- **Scalability**: Linear-time GNN, efficient sampling strategies
- **Calibration**: gBCE loss prevents overconfident predictions
- **Diversity**: Multiple embedding sources ensure rich representations

The system achieves personalized recommendations by combining personality insights, multimodal venue understanding, graph-based relationship patterns, and calibrated ranking.
