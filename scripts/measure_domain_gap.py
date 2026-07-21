#!/usr/bin/env python3
"""
Measure the train/deploy domain gap of the BERT-MBTI classifier.

The classifier is trained on Kaggle MBTI forum posts but applied to Yelp
review text.  This script quantifies that shift using the classifier's own
behaviour (confidence, predictive entropy, predicted class balance), because
Yelp users carry no ground-truth MBTI label and accuracy cannot be measured
out of domain.

IN-DOMAIN sample : posts of the HELD-OUT TEST USERS of the Kaggle split,
                   reproduced exactly as scripts/train_mbti.py builds it
                   (user-level stratified split, seed 42, posts exploded and
                   cleaned only AFTER the user split).
OUT-OF-DOMAIN    : clean_text from data/processed/test_reviews.parquet.

Inference only.  Nothing is trained or modified.

Usage:
    python scripts/measure_domain_gap.py [--n 2000] [--batch-size 64]
"""
import argparse
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import (
    get_config, PROCESSED_DATA_DIR, CHECKPOINT_DIR, PROJECT_ROOT,
)

RESULTS_DIR = PROJECT_ROOT / "results"
from src.models.bert_mbti.model import MBTIMultiLabelClassifier
from src.utils.helpers import setup_logging, get_device
from scripts.train_mbti import load_raw_users, explode_and_clean, _stratified_user_split

logger = logging.getLogger(__name__)

DIMS = ["EI", "SN", "TF", "JP"]
POS_LETTER = {"EI": "I", "SN": "N", "TF": "F", "JP": "P"}


def build_in_domain_sample(n: int, seed: int, sample_seed: int = 0) -> pd.DataFrame:
    """
    Reproduce the user-disjoint Kaggle split of scripts/train_mbti.py and
    return a random sample of posts belonging to the TEST users only.

    train_mbti.py calls split_users_and_explode(users_df, seed=42) which does:
        train_u, temp_u = stratified_split(users_df, test_size=0.2, seed)
        val_u,   test_u = stratified_split(temp_u,   test_size=0.5, seed)
    Both calls are sklearn train_test_split stratified on 'type' with
    random_state=seed, so re-running them on the same user frame is exact.
    """
    users_df = load_raw_users()
    train_u, temp_u = _stratified_user_split(users_df, 0.2, seed=seed)
    val_u, test_u = _stratified_user_split(temp_u, 0.5, seed=seed)
    logger.info(
        f"Reproduced user split: train={len(train_u)} val={len(val_u)} "
        f"test={len(test_u)} users"
    )
    # Sanity: the three user sets must be disjoint.
    tr, va, te = set(train_u["user_id"]), set(val_u["user_id"]), set(test_u["user_id"])
    assert not (tr & te) and not (va & te) and not (tr & va), "splits overlap"

    test_posts = explode_and_clean(test_u)
    logger.info(f"Test users exploded to {len(test_posts):,} cleaned posts")
    take = min(n, len(test_posts))
    return test_posts.sample(n=take, random_state=sample_seed).reset_index(drop=True)


def build_out_domain_sample(n: int, sample_seed: int = 0) -> pd.DataFrame:
    """Random sample of cleaned Yelp review texts from the test split."""
    df = pd.read_parquet(PROCESSED_DATA_DIR / "test_reviews.parquet",
                         columns=["clean_text"])
    df = df[df["clean_text"].astype(str).str.strip().str.len() > 0]
    logger.info(f"Loaded {len(df):,} non-empty Yelp test reviews")
    take = min(n, len(df))
    return df.sample(n=take, random_state=sample_seed).reset_index(drop=True)


@torch.no_grad()
def predict_probs(texts, model, tokenizer, max_length, device, batch_size):
    """Return [N, 4] array of P(I), P(N), P(F), P(P) for each text."""
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i + batch_size], max_length=max_length,
                        padding="max_length", truncation=True,
                        return_tensors="pt")
        res = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
        cols = [F.softmax(res["logits"][d], dim=-1)[:, 1:2] for d in DIMS]
        out.append(torch.cat(cols, dim=1).cpu().numpy())
        if (i // batch_size) % 10 == 0:
            logger.info(f"  {min(i + batch_size, len(texts))}/{len(texts)}")
    return np.vstack(out)


def summarise(probs: np.ndarray, domain: str) -> list[dict]:
    """Per-axis confidence, predictive entropy (nats) and positive rate."""
    rows = []
    eps = 1e-12
    for j, dim in enumerate(DIMS):
        p = probs[:, j].astype(np.float64)
        conf = np.maximum(p, 1.0 - p)
        ent = -(p * np.log(p + eps) + (1 - p) * np.log(1 - p + eps))
        rows.append({
            "domain": domain,
            "axis": f"{dim} (P({POS_LETTER[dim]}))",
            "mean_confidence": float(conf.mean()),
            "mean_entropy": float(ent.mean()),
            "positive_rate": float((p > 0.5).mean()),
            "n": int(len(p)),
        })
    return rows


def make_figure(df: pd.DataFrame, out_pdf: Path, copy_to: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "lines.linewidth": 2.5,
    })

    labels = [f"{d} (P({POS_LETTER[d]}))" for d in DIMS]
    ind = df[df["domain"] == "kaggle_forum_in_domain"].set_index("axis")
    ood = df[df["domain"] == "yelp_reviews_out_of_domain"].set_index("axis")
    v_in = [ind.loc[a, "mean_confidence"] for a in labels]
    v_out = [ood.loc[a, "mean_confidence"] for a in labels]

    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w / 2, v_in, w, label="In-domain (Kaggle forum posts)",
                color="#1565c0")
    b2 = ax.bar(x + w / 2, v_out, w, label="Out-of-domain (Yelp reviews)",
                color="#e65100")
    for bars in (b1, b2):
        for bar in bars:
            ax.annotate(f"{bar.get_height():.3f}",
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([d for d in DIMS])
    ax.set_xlabel("MBTI axis")
    ax.set_ylabel("Mean confidence  max(p, 1-p)")
    ax.set_ylim(0.5, 1.0)
    ax.axhline(0.5, color="grey", linewidth=1.0, linestyle=":")
    ax.legend(loc="upper right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), dpi=200, bbox_inches="tight")
    copy_to.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out_pdf, copy_to)
    plt.close(fig)
    logger.info(f"Figure written to {out_pdf} and {copy_to}")


def main(args):
    setup_logging(level=logging.INFO)
    config = get_config()
    device = get_device(args.device)
    logger.info(f"Device: {device}")

    ckpt_path = CHECKPOINT_DIR / "bert_mbti" / "best_model.pt"
    tokenizer = AutoTokenizer.from_pretrained(config.bert.model_name)
    model = MBTIMultiLabelClassifier(config=config.bert)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck.get("model_state_dict", ck))
    model.to(device).eval()
    logger.info(f"Loaded checkpoint {ckpt_path} (max_length={config.bert.max_length})")

    in_df = build_in_domain_sample(args.n, seed=config.data.random_seed)
    out_df = build_out_domain_sample(args.n)

    logger.info("Running in-domain inference")
    p_in = predict_probs(in_df["clean_text"].astype(str).tolist(), model,
                         tokenizer, config.bert.max_length, device, args.batch_size)
    logger.info("Running out-of-domain inference")
    p_out = predict_probs(out_df["clean_text"].astype(str).tolist(), model,
                          tokenizer, config.bert.max_length, device, args.batch_size)

    rows = (summarise(p_in, "kaggle_forum_in_domain")
            + summarise(p_out, "yelp_reviews_out_of_domain"))
    res = pd.DataFrame(rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "domain_gap.csv"
    res.to_csv(csv_path, index=False)
    logger.info(f"Wrote {csv_path}")

    print("\n" + res.to_string(index=False))
    mc_in = res[res.domain == "kaggle_forum_in_domain"]["mean_confidence"].mean()
    mc_out = res[res.domain == "yelp_reviews_out_of_domain"]["mean_confidence"].mean()
    print(f"\nMean confidence over 4 axes: in-domain={mc_in:.4f} "
          f"out-of-domain={mc_out:.4f} delta={mc_out - mc_in:+.4f}")
    for d in DIMS:
        a = f"{d} (P({POS_LETTER[d]}))"
        r_in = res[(res.domain == "kaggle_forum_in_domain") & (res.axis == a)]["positive_rate"].iloc[0]
        r_out = res[(res.domain == "yelp_reviews_out_of_domain") & (res.axis == a)]["positive_rate"].iloc[0]
        print(f"positive-rate shift {d}: {r_in:.4f} -> {r_out:.4f} "
              f"(abs {abs(r_out - r_in):.4f})")

    # Raw probability arrays kept for the write-up / reproducibility.
    np.save(RESULTS_DIR / "domain_gap_probs_in.npy", p_in.astype(np.float32))
    np.save(RESULTS_DIR / "domain_gap_probs_out.npy", p_out.astype(np.float32))

    make_figure(
        res,
        PROJECT_ROOT / "docs" / "thesis" / "figures" / "fig_domain_gap.pdf",
        PROJECT_ROOT / "docs" / "thesis" / "wut" / "tex" / "img" / "fig_domain_gap.pdf",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure BERT-MBTI domain gap")
    parser.add_argument("--n", type=int, default=2000,
                        help="Texts sampled per domain (default 2000)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu", "mps"])
    args = parser.parse_args()
    main(args)
