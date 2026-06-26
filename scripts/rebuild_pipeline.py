#!/usr/bin/env python3
"""
Rebuild the downstream pipeline after retraining the BERT-MBTI classifier.

The classifier checkpoint (models/checkpoints/bert_mbti/best_model.pt) feeds
every downstream artifact, so when it changes they all must be regenerated:

    classifier ─► [topics]  BERTopic venue embeddings ─► [gnn] HeteroGNN ─┐
               └► [user_mbti] per-user MBTI embeddings ──────────────────┴─► [robustness]

This script runs the stages in dependency order, with the correct flags, the
checkpoint-path fix, and the robustness-cache clear baked in.  It stops on the
first failure (downstream stages would only consume stale/missing inputs).

Usage:
    # after you have retrained the classifier (best_model.pt exists):
    python scripts/rebuild_pipeline.py

    # also retrain the classifier first (~8h), then everything:
    python scripts/rebuild_pipeline.py --with-classifier --epochs 4

    # run a subset:
    python scripts/rebuild_pipeline.py --only topics,gnn
    python scripts/rebuild_pipeline.py --skip user_mbti

    # preview without running:
    python scripts/rebuild_pipeline.py --dry-run
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PY = sys.executable
BEST_MODEL = PROJECT_ROOT / "models" / "checkpoints" / "bert_mbti" / "best_model.pt"
ROBUSTNESS_CACHE = PROJECT_ROOT / "results" / "_robustness"

# Stage order matters: each consumes the outputs of earlier ones.
STAGE_ORDER = ["classifier", "topics", "user_mbti", "gnn", "robustness"]


def build_stages(args):
    """Return {id: {desc, cmd, needs_best_model, pre}} for the requested run."""
    return {
        "classifier": {
            "desc": "Retrain BERT-MBTI classifier (~8h) -> best_model.pt",
            "cmd": [PY, "scripts/train_mbti.py", "--epochs", str(args.epochs)],
            "needs_best_model": False,
            "pre": None,
        },
        "topics": {
            "desc": "BERTopic MBTI-informed venue embeddings",
            # NOTE the explicit --checkpoint: the script's default points at a
            # non-existent bert_mbti_robust/ path.
            "cmd": [PY, "scripts/extract_topics_mbti.py",
                    "--checkpoint", str(BEST_MODEL)],
            "needs_best_model": True,
            "pre": None,
        },
        "user_mbti": {
            "desc": "Per-user BERT-MBTI embeddings + traits",
            "cmd": [PY, "scripts/extract_user_mbti.py",
                    "--reviews-per-user", "5", "--batch-size", "64"],
            "needs_best_model": True,
            "pre": None,
        },
        "gnn": {
            "desc": "Retrain HeteroGNN (uses BERTopic venue embeddings)",
            "cmd": [PY, "scripts/train_hetero_gnn.py",
                    "--epochs", "25", "--hidden-dim", "128", "--num-layers", "1"],
            "needs_best_model": False,  # consumes topics output, not the ckpt
            "pre": None,
        },
        "robustness": {
            "desc": "Multi-seed robustness evaluation (clears stale cache first)",
            "cmd": [PY, "scripts/robustness_eval.py", "--seeds", args.seeds],
            "needs_best_model": False,
            "pre": "clear_robustness_cache",
        },
    }


def clear_robustness_cache():
    if ROBUSTNESS_CACHE.exists():
        shutil.rmtree(ROBUSTNESS_CACHE, ignore_errors=True)
        print(f"  cleared stale cache: {ROBUSTNESS_CACHE}")


def select_stages(args):
    stages = STAGE_ORDER.copy()
    if not args.with_classifier:
        stages.remove("classifier")
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        stages = [s for s in STAGE_ORDER if s in wanted]
    if args.skip:
        skip = {s.strip() for s in args.skip.split(",")}
        stages = [s for s in stages if s not in skip]
    return stages


def main(args):
    all_stages = build_stages(args)
    plan = select_stages(args)

    print("=" * 70)
    print("PIPELINE REBUILD PLAN")
    print("=" * 70)
    for i, sid in enumerate(plan, 1):
        print(f"  {i}. [{sid}] {all_stages[sid]['desc']}")
        print(f"       $ {' '.join(all_stages[sid]['cmd'])}")
    print("=" * 70)

    # Fail fast: any downstream stage needs the classifier checkpoint.
    downstream_needs_ckpt = any(
        all_stages[s]["needs_best_model"] for s in plan)
    classifier_will_run = "classifier" in plan
    if downstream_needs_ckpt and not classifier_will_run and not BEST_MODEL.exists():
        print(f"\nERROR: {BEST_MODEL} not found and classifier stage not in plan.")
        print("Retrain first (--with-classifier) or restore the checkpoint.")
        return 1

    if args.dry_run:
        print("\nDry run - nothing executed.")
        return 0

    results = []
    for sid in plan:
        stage = all_stages[sid]
        print("\n" + "#" * 70)
        print(f"# STAGE [{sid}] {stage['desc']}")
        print("#" * 70)

        if stage["pre"] == "clear_robustness_cache":
            clear_robustness_cache()

        if stage["needs_best_model"] and not BEST_MODEL.exists():
            print(f"ERROR: required checkpoint missing: {BEST_MODEL}")
            results.append((sid, "FAILED (missing checkpoint)", 0))
            break

        t0 = time.time()
        proc = subprocess.run(stage["cmd"], cwd=str(PROJECT_ROOT))
        dt = time.time() - t0

        if proc.returncode != 0:
            print(f"\nERROR: stage [{sid}] failed (exit {proc.returncode}) "
                  f"after {dt/60:.1f} min. Aborting (downstream depends on it).")
            results.append((sid, f"FAILED (exit {proc.returncode})", dt))
            break
        results.append((sid, "ok", dt))
        print(f"\n[{sid}] done in {dt/60:.1f} min")

    print("\n" + "=" * 70)
    print("REBUILD SUMMARY")
    print("=" * 70)
    for sid, status, dt in results:
        print(f"  {sid:12s} {status:24s} {dt/60:6.1f} min")
    ok = all(s == "ok" for _, s, _ in results) and len(results) == len(plan)
    print("=" * 70)
    print("All stages completed." if ok else "Pipeline stopped before completion.")
    return 0 if ok else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Rebuild pipeline after classifier retrain")
    p.add_argument("--with-classifier", action="store_true",
                   help="Also retrain the classifier first (~8h)")
    p.add_argument("--epochs", type=int, default=4,
                   help="Classifier epochs if --with-classifier (default 4)")
    p.add_argument("--seeds", type=str, default="42,123,7,2024,99",
                   help="Seeds for the robustness stage")
    p.add_argument("--only", type=str, default=None,
                   help="Comma list of stage ids to run exclusively")
    p.add_argument("--skip", type=str, default=None,
                   help="Comma list of stage ids to skip")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan without executing")
    args = p.parse_args()
    sys.exit(main(args))
