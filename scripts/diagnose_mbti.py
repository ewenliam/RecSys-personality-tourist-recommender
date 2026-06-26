#!/usr/bin/env python3
"""
Diagnose the BERT-MBTI classifier: reproduce the exact validation split the
model trained against, then dissect where the ~69% accuracy comes from.

Checks performed:
  1. Dataset shape - users, posts/user, per-dimension label balance.
  2. SPLIT-LEAKAGE CHECK - the pipeline explodes posts and splits at the POST
     level, so a single user's posts can land in BOTH train and val. We
     quantify how many val users also appear in train (identity leakage).
  3. Per-post accuracy, per dimension, vs a majority-class baseline
     (detects a head that has collapsed to always predicting one class).
  4. Per-USER accuracy via majority vote (the fairer, deployment-style number).

Read-only: loads the saved checkpoint, trains nothing.
"""
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, balanced_accuracy_score
from transformers import AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import get_config, DATA_DIR, CHECKPOINT_DIR
from src.data.preprocessor import TextPreprocessor
from src.models.bert_mbti.model import MBTIMultiLabelClassifier
from src.utils.helpers import setup_logging, get_device

logger = logging.getLogger(__name__)
DIMS = ["EI", "SN", "TF", "JP"]
SECOND = {"EI": "I", "SN": "N", "TF": "F", "JP": "P"}  # label==1 letter


def label_of(mbti: str, dim: str) -> int:
    pos = DIMS.index(dim)
    return 1 if mbti[pos] == SECOND[dim] else 0


def build_exploded_with_user() -> pd.DataFrame:
    """Reproduce load_mbti_data() but keep the originating user id."""
    df = pd.read_csv(DATA_DIR / "raw" / "mbti_1.csv", dtype_backend="numpy_nullable")
    rows = []
    for uid, row in df.iterrows():                       # uid = original user row
        mbti = str(row["type"]).upper()
        for post in str(row["posts"]).split("|||"):
            post = post.strip()
            if post:
                rows.append({"user_id": uid, "mbti": mbti, "text": post})
    ex = pd.DataFrame(rows)
    ex = TextPreprocessor().process_dataframe(ex, text_column="text")  # same filter
    return ex


def reproduce_split(ex: pd.DataFrame):
    """Same split calls as scripts/train_mbti.py (post-level, seed 42)."""
    train_df, temp_df = train_test_split(
        ex, test_size=0.2, stratify=ex["mbti"], random_state=42)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["mbti"], random_state=42)
    return train_df, val_df, test_df


@torch.no_grad()
def predict(df, model, tokenizer, max_length, device, batch_size=64):
    """Return dict dim -> predicted labels array for each row of df."""
    texts = df["clean_text"].astype(str).tolist()
    preds = {d: [] for d in DIMS}
    for i in tqdm(range(0, len(texts), batch_size), desc="infer"):
        chunk = texts[i:i + batch_size]
        enc = tokenizer(chunk, max_length=max_length, padding="max_length",
                        truncation=True, return_tensors="pt")
        out = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        for d in DIMS:
            preds[d].extend(torch.argmax(out["logits"][d], dim=-1).cpu().numpy())
    return {d: np.array(v) for d, v in preds.items()}


def main():
    setup_logging(level=logging.INFO)
    device = get_device("cuda")
    cfg = get_config().bert

    logger.info("Reproducing dataset + split (this matches training)...")
    ex = build_exploded_with_user()
    train_df, val_df, test_df = reproduce_split(ex)

    n_users = ex["user_id"].nunique()
    posts_per_user = ex.groupby("user_id").size()
    print("\n" + "=" * 70)
    print("1) DATASET SHAPE")
    print("=" * 70)
    print(f"users={n_users}  posts={len(ex)}  posts/user: "
          f"mean={posts_per_user.mean():.1f} median={posts_per_user.median():.0f}")
    print(f"split posts -> train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    print("\nPer-dimension label balance (fraction == second letter I/N/F/P):")
    for d in DIMS:
        frac = ex["mbti"].apply(lambda m: label_of(m, d)).mean()
        print(f"  {d}: P({SECOND[d]})={frac:.3f}  (majority-class baseline acc="
              f"{max(frac, 1-frac):.3f})")

    print("\n" + "=" * 70)
    print("2) SPLIT-LEAKAGE CHECK (post-level split -> users span splits)")
    print("=" * 70)
    tr_u, va_u = set(train_df["user_id"]), set(val_df["user_id"])
    overlap = tr_u & va_u
    print(f"val users={len(va_u)}  | also present in train={len(overlap)} "
          f"({100*len(overlap)/len(va_u):.1f}%)")
    print("  -> if this is high, val accuracy partly measures recognising a")
    print("     KNOWN user's other posts, not generalising to NEW users.")

    logger.info("Loading checkpoint best_model.pt ...")
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    model = MBTIMultiLabelClassifier(config=cfg)
    ck = torch.load(CHECKPOINT_DIR / "bert_mbti" / "best_model.pt",
                    map_location=device, weights_only=False)
    model.load_state_dict(ck.get("model_state_dict", ck))
    model.to(device).eval()

    preds = predict(val_df, model, tok, cfg.max_length, device)
    gold = {d: val_df["mbti"].apply(lambda m: label_of(m, d)).values for d in DIMS}

    print("\n" + "=" * 70)
    print("3) PER-POST ACCURACY (reproduces the reported ~69%)")
    print("=" * 70)
    per_dim_acc = {}
    flat_correct = 0
    flat_total = 0
    for d in DIMS:
        acc = (preds[d] == gold[d]).mean()
        bal = balanced_accuracy_score(gold[d], preds[d])
        f1 = f1_score(gold[d], preds[d], average="macro")
        base = max(gold[d].mean(), 1 - gold[d].mean())
        pred_rate = preds[d].mean()
        per_dim_acc[d] = acc
        flat_correct += (preds[d] == gold[d]).sum()
        flat_total += len(gold[d])
        flag = "  <-- near baseline (weak)" if acc - base < 0.03 else ""
        print(f"  {d}: acc={acc:.3f}  balanced_acc={bal:.3f}  macroF1={f1:.3f}  "
              f"baseline={base:.3f}  pred_rate(P=1)={pred_rate:.3f}{flag}")
    print(f"\n  MEAN per-dimension (flattened) accuracy = {flat_correct/flat_total:.4f}")

    print("\n" + "=" * 70)
    print("4) PER-USER ACCURACY (majority vote over each user's val posts)")
    print("=" * 70)
    vdf = val_df.reset_index(drop=True)
    by_user = defaultdict(lambda: {d: [] for d in DIMS})
    user_gold = {}
    for i, (_, r) in enumerate(vdf.iterrows()):
        for d in DIMS:
            by_user[r["user_id"]][d].append(preds[d][i])
        user_gold[r["user_id"]] = r["mbti"]
    per_dim_user_correct = {d: 0 for d in DIMS}
    exact_correct = 0
    n_u = len(by_user)
    for uid, dimpreds in by_user.items():
        g = user_gold[uid]
        all_ok = True
        for d in DIMS:
            vote = int(round(np.mean(dimpreds[d])))   # majority vote
            ok = vote == label_of(g, d)
            per_dim_user_correct[d] += ok
            all_ok = all_ok and ok
        exact_correct += all_ok
    print(f"  users in val = {n_u}")
    for d in DIMS:
        print(f"  {d}: per-user acc={per_dim_user_correct[d]/n_u:.3f}")
    mean_user = np.mean([per_dim_user_correct[d]/n_u for d in DIMS])
    print(f"\n  MEAN per-user per-dimension accuracy = {mean_user:.4f}")
    print(f"  EXACT 16-type per-user accuracy       = {exact_correct/n_u:.4f}  "
          f"(random=0.0625)")
    print("\nDone.")


if __name__ == "__main__":
    main()
