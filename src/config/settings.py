"""
Configuration settings for the Tourist Recommendation System.
"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"


@dataclass
class DataConfig:
    """Data processing configuration."""
    # Yelp dataset paths
    business_path: Path = RAW_DATA_DIR / "yelp_academic_dataset_business.json"
    review_path: Path = RAW_DATA_DIR / "yelp_academic_dataset_review.json"
    user_path: Path = RAW_DATA_DIR / "yelp_academic_dataset_user.json"
    photos_path: Path = RAW_DATA_DIR / "photos.json"

    # Processing params
    max_review_length: int = 512
    min_review_length: int = 10
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    random_seed: int = 42

    # Sampling
    upsample_minority: bool = True
    min_reviews_per_user: int = 5
    min_reviews_per_business: int = 3


@dataclass
class BERTConfig:
    """BERT MBTI classifier configuration."""
    model_name: str = "bert-base-uncased"
    max_length: int = 128
    batch_size: int = 32         # Updated to match Table 6.3
    optimizer: str = "AdamW"      # New: Explicitly state the optimizer
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 10
    warmup_ratio: float = 0.1
    gradient_clip: float = 1.0
    early_stopping_patience: int = 3
    num_labels: int = 16  # 16 MBTI types


@dataclass
class BERTopicConfig:
    """BERTopic configuration."""
    embedding_model: str = "all-MiniLM-L6-v2"
    clip_model: str = "ViT-B-32"
    umap_n_neighbors: int = 15
    umap_n_components: int = 5
    umap_min_dist: float = 0.0
    umap_metric: str = "cosine"
    hdbscan_min_cluster_size: int = 10
    hdbscan_min_samples: int = 5
    kmeans_n_clusters: int = 50
    nr_topics: Optional[int] = None  # Auto-determine
    # MBTI-informed embeddings (Phase 2)
    use_mbti_embeddings: bool = False
    mbti_checkpoint_path: Optional[str] = None
    # Outlier reduction (reduces topic -1 assignment rate)
    reduce_outliers: bool = True
    outlier_confidence: float = 0.1  # Confidence for ex-outlier venues


@dataclass
class GNNConfig:
    """Graph Neural Network configuration."""
    hidden_dim: int = 128
    embedding_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 1024
    num_epochs: int = 100
    num_neighbors: list = field(default_factory=lambda: [10, 5])
    # LTGNN specific
    fixed_point_iterations: int = 10
    evr_sample_size: int = 20
    # Heterogeneous GNN (Phase 3)
    use_heterogeneous: bool = True
    num_sage_layers: int = 2


@dataclass
class HybridConfig:
    """Hybrid recommendation engine configuration."""
    # XGBoost params (late-fusion ensemble, only 4 features)
    xgb_max_depth: int = 4
    xgb_learning_rate: float = 0.1
    xgb_n_estimators: int = 100
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.5

    # Post-hoc calibration (replaces raw gBCE temperature scaling)
    calibration_method: str = "platt"  # "platt", "isotonic", "temperature"
    gbce_calibration_t: float = 0.8    # Only used if method="temperature"

    # DCI Closed
    min_support: float = 0.01
    max_itemset_size: int = 5


@dataclass
class Config:
    """Main configuration container."""
    data: DataConfig = field(default_factory=DataConfig)
    bert: BERTConfig = field(default_factory=BERTConfig)
    bertopic: BERTopicConfig = field(default_factory=BERTopicConfig)
    gnn: GNNConfig = field(default_factory=GNNConfig)
    hybrid: HybridConfig = field(default_factory=HybridConfig)

    # Device
    device: str = "cuda"  # or "cpu", "mps"


def get_config() -> Config:
    """Get the default configuration."""
    return Config()
