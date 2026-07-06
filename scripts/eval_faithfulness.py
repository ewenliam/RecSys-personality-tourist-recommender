#!/usr/bin/env python3
"""
Faithfulness evaluation (deletion test) for the ONLY approximate segment of
the explanation trace: the word-level occlusion attributions behind the
personality inference.

Deletion test: for each persona and their dominant axis, delete the top-k
attributed words (k = 1..10) and measure the drop in that axis probability;
compare against deleting k random words (mean of 10 random orders). If the
attribution-ordered curve drops faster, the attributions point at words the
model genuinely relies on. Reported as area-over-the-curve (AOC) of the
probability drop; higher = more faithful, random ~ 0.

Usage:
    python scripts/eval_faithfulness.py [--device cuda]
"""
import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.explain.text_attrib import MBTIExplainer
from src.utils.helpers import setup_logging
from scripts.run_itineraries import PERSONAS

logger = logging.getLogger(__name__)
K_MAX = 10
N_RANDOM = 10


def axis_prob(explainer, text: str, axis_i: int) -> float:
    _, probs = explainer._forward([text])
    return float(probs[0, axis_i])


def deletion_curve(explainer, words: list[str], order: list[int],
                   axis_i: int, base: float) -> np.ndarray:
    """Prob drop after deleting the first k words of `order`, k=1..K_MAX."""
    drops = []
    removed: set[int] = set()
    for k in range(min(K_MAX, len(order))):
        removed.add(order[k])
        text = " ".join(w for i, w in enumerate(words) if i not in removed)
        drops.append(base - axis_prob(explainer, text, axis_i))
    return np.array(drops)


def main(args):
    setup_logging(level=logging.INFO)
    explainer = MBTIExplainer(device=args.device)
    rng = np.random.default_rng(42)

    rows = []
    for persona in PERSONAS:
        words = re.findall(r"\S+", persona.text)
        attributions = explainer.occlusion(persona.text)
        _, probs = explainer._forward([persona.text])

        # Dominant axis = the one the persona leans to most strongly.
        lean = np.abs(probs[0] - 0.5)
        axis_i = int(np.argmax(lean))
        axis = explainer.dims[axis_i]
        base = float(probs[0, axis_i])
        sign = 1.0 if base >= 0.5 else -1.0

        # Attribution order: words whose deletion most reduces the leaned-to
        # probability (sign-adjusted so it works for either class).
        attr_order = sorted(
            range(len(words)),
            key=lambda i: -sign * attributions[i]["deltas"][axis])
        attr_drop = sign * deletion_curve(
            explainer, words, attr_order, axis_i, base)

        rand_drops = []
        for _ in range(N_RANDOM):
            order = rng.permutation(len(words)).tolist()
            rand_drops.append(sign * deletion_curve(
                explainer, words, order, axis_i, base))
        rand_drop = np.mean(rand_drops, axis=0)

        rows.append({
            "persona": persona.persona_id,
            "axis": axis, "base_prob": round(base, 3),
            "aoc_attribution": round(float(attr_drop.mean()), 4),
            "aoc_random": round(float(rand_drop.mean()), 4),
            "gap": round(float(attr_drop.mean() - rand_drop.mean()), 4),
        })
        logger.info(f"{persona.persona_id}: axis={axis} "
                    f"attr AOC={attr_drop.mean():.4f} "
                    f"random AOC={rand_drop.mean():.4f}")

    df = pd.DataFrame(rows)
    out = Path("results") / "faithfulness_eval.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nmean AOC gap (attribution - random): {df['gap'].mean():.4f} "
          f"(positive = attributions faithful)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    main(parser.parse_args())
