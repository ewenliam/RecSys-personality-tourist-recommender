#!/usr/bin/env python3
"""
Extract per-user MBTI representations from Yelp review text.

For every user we run the fine-tuned BERT-MBTI classifier over a short sample
of their OWN reviews and keep two outputs:

  1. user_mbti_embeddings.npy [n_users, 768]
       L2-normalised CLS embedding - lives in the SAME space as the venue
       MBTI-CLS embeddings (models/bertopic_mbti/venue_embeddings.npy), so
       cosine(user, venue) is a direct personality-compatibility score.

  2. user_mbti_traits.npy [n_users, 4]
       P(I), P(N), P(F), P(P) - the four binary-dimension probabilities.
       Used for the methodology table and optional trait-level matching.

Only TRAIN reviews are used so nothing from the test split leaks into the
user representation that later ranks test venues.

This signal is distinct from the KNN content profile: KNN profiles a user by
the venues they VISITED, whereas this profiles them by how they WRITE - their
personality as expressed in their own text.

Usage:
    python scripts/extract_user_mbti.py
    python scripts/extract_user_mbti.py --reviews-per-user 5 --batch-size 64
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import get_config, PROCESSED_DATA_DIR, MODEL_DIR, CHECKPOINT_DIR
from src.models.bert_mbti.model import MBTIMultiLabelClassifier
from src.utils.helpers import setup_logging, get_device

logger = logging.getLogger(__name__)


class _UserDocDataset(Dataset):
    """Tokenises one concatenated review-document per user."""

    def __init__(self, docs: list[str], tokenizer, max_length: int):
        self.docs = docs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.docs[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
        }


def build_user_documents(
    reviews_path: Path,
    reviews_per_user: int,
) -> pd.DataFrame:
    """
    Build one text document per user from their first-N training reviews.

    Returns a DataFrame with columns [user_id, doc] in stable user order.
    """
    logger.info(f"Reading {reviews_path.name} (user_id, clean_text)...")
    df = pd.read_parquet(reviews_path, columns=["user_id", "clean_text"])
    logger.info(f"Loaded {len(df):,} reviews")

    # First-N reviews per user keeps cost bounded; personality is stable so a
    # short sample is enough to characterise a user's writing.
    df = df.groupby("user_id", sort=False).head(reviews_per_user)

    docs = (
        df.groupby("user_id", sort=False)["clean_text"]
        .apply(lambda s: " ".join(str(x) for x in s.values))
        .reset_index()
    )
    docs.columns = ["user_id", "doc"]

    # Guard against empty / whitespace-only docs (tokenizer needs non-empty).
    docs["doc"] = docs["doc"].fillna("").str.strip()
    n_empty = int((docs["doc"] == "").sum())
    if n_empty:
        logger.warning(f"{n_empty} users have empty docs; using placeholder token")
        docs.loc[docs["doc"] == "", "doc"] = "[UNK]"

    del df
    import gc
    gc.collect()

    logger.info(f"Built documents for {len(docs):,} users")
    return docs


@torch.no_grad()
def encode_users(
    docs: list[str],
    model: MBTIMultiLabelClassifier,
    tokenizer,
    max_length: int,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run BERT-MBTI over user documents.

    Returns (cls_embeddings [N, 768] L2-normalised, trait_probs [N, 4]) where
    trait columns are P(I), P(N), P(F), P(P).
    """
    dataset = _UserDocDataset(docs, tokenizer, max_length)
    # pin_memory disabled: on Windows it can trigger a native crash with CUDA
    # at scale.  num_workers=0 avoids Windows multiprocessing issues.
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    cls_chunks = []
    trait_chunks = []
    dims = MBTIMultiLabelClassifier.DIMENSIONS  # ["EI","SN","TF","JP"]

    for batch in tqdm(loader, desc="Encoding users"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # One BERT pass -> CLS embedding + the four dimension heads.
        bert_out = model.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = bert_out.last_hidden_state[:, 0, :]              # [B, 768]
        pooled = model.dropout(cls)

        # P(second class) per dimension = P(I), P(N), P(F), P(P).
        probs = []
        for dim in dims:
            logits = model.classifiers[dim](pooled)            # [B, 2]
            probs.append(F.softmax(logits, dim=-1)[:, 1:2])    # [B, 1]
        trait = torch.cat(probs, dim=1)                        # [B, 4]

        cls = F.normalize(cls, p=2, dim=-1)                    # cosine-ready
        cls_chunks.append(cls.cpu().numpy())
        trait_chunks.append(trait.cpu().numpy())

    return np.vstack(cls_chunks), np.vstack(trait_chunks)


def main(args):
    setup_logging(level=logging.INFO)
    device = get_device(args.device)
    config = get_config().bert

    # --- Load model -----------------------------------------------------
    ckpt_path = CHECKPOINT_DIR / "bert_mbti" / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"BERT-MBTI checkpoint not found: {ckpt_path}")

    logger.info(f"Loading BERT-MBTI from {ckpt_path}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = MBTIMultiLabelClassifier(config=config)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # --- Build per-user documents (TRAIN only) --------------------------
    train_path = PROCESSED_DATA_DIR / "train_reviews.parquet"
    docs_df = build_user_documents(train_path, args.reviews_per_user)

    # --- Inference ------------------------------------------------------
    cls_embs, trait_probs = encode_users(
        docs_df["doc"].tolist(), model, tokenizer,
        config.max_length, device, args.batch_size,
    )

    # --- Save -----------------------------------------------------------
    out_dir = MODEL_DIR / "bert_mbti"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "user_mbti_embeddings.npy", cls_embs.astype(np.float32))
    np.save(out_dir / "user_mbti_traits.npy", trait_probs.astype(np.float32))
    docs_df[["user_id"]].to_parquet(out_dir / "user_mbti_ids.parquet", index=False)

    # Quick sanity report: trait base rates and embedding spread.
    rates = trait_probs.mean(axis=0)
    logger.info(
        f"Trait base rates  P(I)={rates[0]:.3f} P(N)={rates[1]:.3f} "
        f"P(F)={rates[2]:.3f} P(P)={rates[3]:.3f}"
    )
    # Cosine spread: std of pairwise sim on a small sample (collapse check).
    sample = cls_embs[np.random.RandomState(0).choice(len(cls_embs),
              size=min(2000, len(cls_embs)), replace=False)]
    sims = sample @ sample.T
    off = sims[~np.eye(len(sample), dtype=bool)]
    logger.info(
        f"User MBTI embedding cosine: mean={off.mean():.3f} std={off.std():.3f} "
        f"(higher std = more discriminative)"
    )
    logger.info(f"Saved user MBTI embeddings: {cls_embs.shape} -> {out_dir}")
    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract per-user MBTI representations")
    parser.add_argument("--reviews-per-user", type=int, default=5,
                        help="Max training reviews concatenated per user (default 5)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu", "mps"])
    args = parser.parse_args()
    main(args)
