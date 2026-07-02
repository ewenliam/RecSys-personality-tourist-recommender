#!/usr/bin/env python3
"""
Train BERT MBTI Classifier.

Loads the Kaggle MBTI dataset (mbti_1.csv), explodes posts, cleans text,
splits into train/val/test, upsamples the training set only (robust
methodology), and trains a MBTIMultiLabelClassifier (4 binary heads).

Usage:
    python scripts/train_mbti.py --epochs 3 --batch-size 32
    python scripts/train_mbti.py --resume checkpoint_epoch_5.pt
"""
import argparse
import gc
import logging
import os
import sys
from pathlib import Path

# Prevent CUDA memory fragmentation on Windows (must be set before torch import)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.config.settings import PROCESSED_DATA_DIR, CHECKPOINT_DIR, DATA_DIR
from src.data.preprocessor import ReviewDataset, TextPreprocessor
from src.models.bert_mbti import (
    MBTIMultiLabelClassifier,
    MBTITrainer,
    create_data_loaders,
)
from src.utils.helpers import setup_logging, set_seed, get_device

logger = logging.getLogger(__name__)


def load_raw_users() -> pd.DataFrame:
    """
    Load the Kaggle MBTI dataset (mbti_1.csv) at the USER level (one row per
    user), keeping a stable user_id.  Posts are exploded later, AFTER the
    user-level split, so a user's posts never span train/val/test.
    """
    mbti_csv_path = DATA_DIR / "raw" / "mbti_1.csv"
    if not mbti_csv_path.exists():
        raise FileNotFoundError(
            f"MBTI dataset not found at {mbti_csv_path}\n"
            "Download it from Kaggle: https://www.kaggle.com/datasnaek/mbti-type\n"
            "Place mbti_1.csv in data/raw/"
        )
    df = pd.read_csv(mbti_csv_path, dtype_backend="numpy_nullable").reset_index(drop=True)
    df["user_id"] = df.index
    df["type"] = df["type"].astype(str).str.upper()
    logger.info(f"Loaded {len(df)} MBTI-labeled users")
    return df


def explode_and_clean(users_df: pd.DataFrame) -> pd.DataFrame:
    """Explode each user's '|||'-separated posts into rows and clean text."""
    rows = []
    for _, row in users_df.iterrows():
        uid = int(row["user_id"])
        mbti_type = str(row["type"])
        for post in str(row["posts"]).split("|||"):
            post = post.strip()
            if post:
                rows.append({"user_id": uid, "mbti": mbti_type, "text": post})
    exploded = pd.DataFrame(rows)
    preprocessor = TextPreprocessor()
    exploded = preprocessor.process_dataframe(exploded, text_column="text")
    return exploded


def _stratified_user_split(users_df, test_size, seed):
    """User-level split, stratified by type with a safe non-stratified fallback."""
    try:
        return train_test_split(
            users_df, test_size=test_size, stratify=users_df["type"], random_state=seed)
    except ValueError:
        logger.warning("Stratified user split failed (rare classes); using random split")
        return train_test_split(users_df, test_size=test_size, random_state=seed)


def _upsample_minority(train_df: pd.DataFrame) -> pd.DataFrame:
    """Upsample minority MBTI types in the training set to the majority size."""
    majority = train_df["mbti"].value_counts().max()
    parts = []
    for mbti_type in train_df["mbti"].unique():
        cls = train_df[train_df["mbti"] == mbti_type]
        if len(cls) < majority:
            cls = resample(cls, replace=True, n_samples=majority, random_state=42)
        parts.append(cls)
    return pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)


def compute_class_weights(train_df: pd.DataFrame) -> dict:
    """
    Balanced per-dimension class weights from the TRAIN labels.

    For each dimension, weight_c = n_total / (2 * n_c) so the loss penalises
    errors on the minority class proportionally more.  Returned as
    {dim: tensor([w_first_letter, w_second_letter])} matching the label
    convention (label 1 == second letter I/N/F/P).
    """
    mapping = {"EI": ("I", 0), "SN": ("N", 1), "TF": ("F", 2), "JP": ("P", 3)}
    weights = {}
    for dim, (second_letter, pos) in mapping.items():
        labels = train_df["mbti"].str[pos].eq(second_letter).astype(int).values
        n = len(labels)
        n1 = int(labels.sum())          # class 1 (second letter)
        n0 = n - n1                     # class 0 (first letter)
        w0 = n / (2 * n0) if n0 > 0 else 1.0
        w1 = n / (2 * n1) if n1 > 0 else 1.0
        weights[dim] = torch.tensor([w0, w1], dtype=torch.float)
        logger.info(
            f"  class weights {dim}: w0={w0:.3f} w1={w1:.3f} "
            f"(n0={n0}, n1={n1})"
        )
    return weights


def split_users_and_explode(
    users_df: pd.DataFrame,
    test_size: float = 0.2,
    val_ratio: float = 0.5,
    upsample: bool = False,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    USER-LEVEL split, then explode posts within each split.

    Splitting users (not posts) before exploding guarantees that all of a
    user's posts stay in a single split.  This removes the identity leakage
    of the old post-level split, where ~100% of validation users also had
    posts in training, so val accuracy partly measured recognising a known
    user rather than generalising to a new person.

    Args:
        users_df: One row per user (with 'type', 'posts', 'user_id').
        test_size: Fraction of USERS held out for val+test.
        val_ratio: Fraction of the held-out users used for validation.
        upsample: If True, upsample minority types in the (post-level) train
            set.  Prefer per-dimension class weights instead (see model).

    Returns:
        (train_df, val_df, test_df) post-level frames, user-disjoint, each
        carrying 'user_id' for per-user evaluation.
    """
    train_u, temp_u = _stratified_user_split(users_df, test_size, seed=seed)
    val_u, test_u = _stratified_user_split(temp_u, val_ratio, seed=seed)
    logger.info(
        f"User-level split: train={len(train_u)} val={len(val_u)} "
        f"test={len(test_u)} users (disjoint)"
    )

    train_df = explode_and_clean(train_u)
    val_df = explode_and_clean(val_u)
    test_df = explode_and_clean(test_u)
    logger.info(
        f"After exploding posts: train={len(train_df)} val={len(val_df)} "
        f"test={len(test_df)} posts"
    )

    if upsample:
        train_df = _upsample_minority(train_df)
        logger.info(f"After upsampling train posts: {len(train_df)}")

    return train_df, val_df, test_df


@torch.no_grad()
def evaluate_test_per_user(model, test_df, bert_config, device, batch_size=64):
    """
    Honest final evaluation on the held-out TEST users.

    Runs the model on every test post, then aggregates per user by majority
    vote.  Reports per-dimension accuracy AND balanced accuracy plus the
    exact 16-type per-user accuracy.  Because the split is user-disjoint,
    these numbers reflect generalisation to unseen people - no leakage.
    """
    from transformers import AutoTokenizer
    from sklearn.metrics import balanced_accuracy_score

    dims = ["EI", "SN", "TF", "JP"]
    second = {"EI": "I", "SN": "N", "TF": "F", "JP": "P"}
    pos = {"EI": 0, "SN": 1, "TF": 2, "JP": 3}

    model.eval()
    tok = AutoTokenizer.from_pretrained(bert_config.model_name)
    texts = test_df["clean_text"].astype(str).tolist()
    post_pred = {d: [] for d in dims}
    for i in tqdm(range(0, len(texts), batch_size), desc="Test inference"):
        enc = tok(texts[i:i + batch_size], max_length=bert_config.max_length,
                  padding="max_length", truncation=True, return_tensors="pt")
        out = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        for d in dims:
            post_pred[d].extend(torch.argmax(out["logits"][d], dim=-1).cpu().numpy().tolist())

    tdf = test_df.reset_index(drop=True)
    # Per-user majority vote
    user_pred = {d: {} for d in dims}
    user_gold = {d: {} for d in dims}
    from collections import defaultdict
    votes = defaultdict(lambda: {d: [] for d in dims})
    gold_type = {}
    for i, (_, r) in enumerate(tdf.iterrows()):
        uid = r["user_id"]
        gold_type[uid] = r["mbti"]
        for d in dims:
            votes[uid][d].append(post_pred[d][i])

    logger.info("=" * 60)
    logger.info("HONEST TEST METRICS (per-user, user-disjoint split)")
    logger.info("=" * 60)
    accs, bals = [], []
    exact = 0
    uids = list(votes.keys())
    for d in dims:
        yp, yt = [], []
        for uid in uids:
            vote = int(round(np.mean(votes[uid][d])))
            gold = 1 if gold_type[uid][pos[d]] == second[d] else 0
            yp.append(vote)
            yt.append(gold)
        acc = np.mean(np.array(yp) == np.array(yt))
        bal = balanced_accuracy_score(yt, yp)
        accs.append(acc)
        bals.append(bal)
        logger.info(f"  {d}: per-user acc={acc:.3f}  balanced_acc={bal:.3f}")
    for uid in uids:
        ok = all(int(round(np.mean(votes[uid][d]))) ==
                 (1 if gold_type[uid][pos[d]] == second[d] else 0) for d in dims)
        exact += ok
    logger.info(f"  MEAN per-user accuracy          = {np.mean(accs):.4f}")
    logger.info(f"  MEAN per-user balanced accuracy = {np.mean(bals):.4f}")
    logger.info(f"  EXACT 16-type per-user accuracy = {exact/len(uids):.4f} (random=0.0625)")
    return {"mean_acc": float(np.mean(accs)), "mean_balanced_acc": float(np.mean(bals)),
            "exact": exact / len(uids)}


def main(args):
    """Main training function."""
    setup_logging(level=logging.INFO)
    config = get_config()

    # Override config with command line args
    if args.epochs:
        config.bert.num_epochs = args.epochs
    if args.batch_size:
        config.bert.batch_size = args.batch_size
    if args.learning_rate:
        config.bert.learning_rate = args.learning_rate

    if args.seed is not None:
        config.data.random_seed = args.seed
    set_seed(config.data.random_seed)
    device = get_device(args.device)

    # Load Kaggle MBTI data at the user level, then split USERS before
    # exploding posts (prevents identity leakage across splits).
    logger.info("Loading MBTI data from mbti_1.csv...")
    users_df = load_raw_users()

    # Upsampling backfired on the imbalanced E/I and S/N heads (the model
    # dropped below the majority-class baseline).  Prefer per-dimension class
    # weights instead; only upsample if class weights are disabled.
    use_class_weights = getattr(config.bert, "use_class_weights", True)
    use_upsample = config.data.upsample_minority and not use_class_weights
    train_df, val_df, test_df = split_users_and_explode(
        users_df, upsample=use_upsample, seed=config.data.random_seed)

    # Determine text column
    text_col = "clean_text" if "clean_text" in train_df.columns else "text"

    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = ReviewDataset.from_dataframe(
        train_df,
        text_column=text_col,
        label_column="mbti",
        config=config.bert,
    )
    val_dataset = ReviewDataset.from_dataframe(
        val_df,
        text_column=text_col,
        label_column="mbti",
        config=config.bert,
    )

    # Create data loaders (num_workers=0 for Windows compatibility)
    train_loader, val_loader = create_data_loaders(
        train_dataset,
        val_dataset,
        batch_size=config.bert.batch_size,
        num_workers=0,
    )

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")

    # Create model
    logger.info("Creating model...")
    model = MBTIMultiLabelClassifier(config=config.bert)

    # Per-dimension class weights (counter imbalance without upsampling).
    if use_class_weights:
        logger.info("Computing per-dimension class weights from train labels:")
        model.set_class_weights(compute_class_weights(train_df))

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")

    # Create trainer
    trainer = MBTITrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config.bert,
        device=device,
    )

    # Resume from checkpoint if specified
    if args.resume:
        checkpoint_path = CHECKPOINT_DIR / "bert_mbti" / args.resume
        trainer.load_checkpoint(checkpoint_path)

    # Free any stale allocations before training
    gc.collect()
    torch.cuda.empty_cache()

    # Train
    logger.info("Starting training...")
    history = trainer.train()

    # Print results
    logger.info(f"\nBest validation loss: {history['best_val_loss']:.4f}")
    logger.info(
        f"Best val mean-balanced-acc: "
        f"{history.get('best_mean_balanced_acc', 0.0):.4f}"
    )
    logger.info(f"Checkpoints saved to: {trainer.checkpoint_dir}")

    # Honest final evaluation: load the best checkpoint and score the
    # held-out TEST users with per-user majority vote (no leakage).
    best_path = CHECKPOINT_DIR / "bert_mbti" / "best_model.pt"
    if best_path.exists():
        ck = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ck.get("model_state_dict", ck))
        logger.info("Loaded best checkpoint for final test evaluation")
    evaluate_test_per_user(model, test_df, config.bert, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BERT MBTI classifier")

    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=None,
        help="Learning rate",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        choices=["cuda", "mps", "cpu"],
        help="Device to train on",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Checkpoint filename to resume from",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed controlling the user split AND init "
             "(for multi-seed robustness runs)",
    )

    args = parser.parse_args()
    main(args)
