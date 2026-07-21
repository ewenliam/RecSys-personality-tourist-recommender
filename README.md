# Personality-Aware Tourist Venue Recommender

MSc thesis project (Warsaw University of Technology). A tourist venue
recommender that infers user personality (MBTI) from review text and fuses it
with content, collaborative, and popularity signals. It extends and corrects an
earlier MSc thesis on the same task: the prior work's headline MBTI accuracy
(~94%) was inflated by data leakage, and the honest replication is the core
contribution.

> This README documents the system **as built**. Several components differ from
> the original proposal (see history): the classifier uses four binary heads
> rather than 16-way classification, the graph model is heterogeneous GraphSAGE
> rather than LTGNN, and the final ranker is parameter-free Reciprocal Rank
> Fusion rather than an XGBoost learner.

## What it does

1. **Infers MBTI from text.** A fine-tuned `bert-base-uncased` classifier with
   four binary heads (E/I, S/N, T/F, J/P) predicts personality from a user's
   reviews.
2. **Ranks venues** by fusing four signals with Reciprocal Rank Fusion (RRF):
   personality-informed content (BERTopic), collaborative (GraphSAGE),
   direct personality compatibility (visitor-mean venue profiles), and a
   popularity tie-breaker.
3. **Plans and explains** (proof of concept). An orienteering solver turns the
   ranked list into a feasible single-day itinerary, and each stop carries a
   faithful reasoning trace across the text encoder, graph model, and planner.

## Headline results (honest, corrected protocol)

- **Classifier** (user-disjoint split, per-user majority vote): 80.1% accuracy,
  75.2% balanced accuracy, 44.5% exact 16-type. The prior work's ~94% was a
  leakage artefact.
- **Backbone ablation**: BERT, RoBERTa, and DeBERTa-v3 all land within
  0.74-0.76 balanced accuracy. The dataset, not model capacity, is the ceiling.
- **Recommender** (5 seeds, paired evaluation, K=10): the full four-signal
  hybrid is best on every metric (Hit Rate@10 0.038, ~91x a random ranker) and
  wins on all or nearly all seeds. A paired Wilcoxon test finds the personality
  signal a significant contributor; the graph signal and the margin over a
  popularity baseline are positive but not robustly significant.

Two data-leakage audits underpin these numbers: one of the prior work's
classifier, and one of this project's own graph encoder (which had propagated
over held-out interactions). Both corrections lowered reported figures and are
documented rather than absorbed.

## Architecture

```
BERT-MBTI classifier (4 binary heads, class-weighted loss, user-disjoint split)
    |
    +-- BERTopic on MBTI CLS embeddings -> PCA-64 content vectors
    +-- per-user personality embeddings + visitor-mean venue profiles
    |
Heterogeneous GraphSAGE (1 layer, BPR + temperature 0.1 + hard negatives)
    |
Reciprocal Rank Fusion (k=60) + post-fusion popularity tie-breaker (alpha=0.01)
```

Venue personality is the mean personality of a venue's visitors, not the
embedding of its review text (review-text venue embeddings are near-identical,
pairwise cosine std ~0.04, and rank close to random).

## Repository layout

- `src/` - models (BERT-MBTI, BERTopic, hetero-GNN, hybrid), planner, explain
- `scripts/` - training, evaluation, asset building, figure generation
- `demo/` - Streamlit demo (existing-user, cold-start, and itinerary modes)
- `hpc/` - SLURM scripts for the faculty HPC cluster
- `notebooks/` - exploration and an editable thesis-figure notebook

Data, trained models, and results are not tracked (they are large and, for the
thesis, kept private). A fresh clone contains code only.

## Setup

```bash
python -m venv venv && venv/Scripts/activate   # Windows; use bin/activate on Linux
pip install -r requirements.txt                # keep numpy==1.26.4
export NUMBA_DISABLE_INTEL_SVML=1              # required for UMAP
```

NumPy must stay at 1.26.4; version 2.x breaks numba, UMAP, and the topic
pipeline. Verify with `python -c "import numpy; print(numpy.__version__)"`.

## Commands

```bash
# Train the MBTI classifier
python scripts/train_mbti.py --epochs 4 [--seed N]

# Rebuild the full downstream pipeline (topics -> GNN -> fusion -> robustness)
python scripts/rebuild_pipeline.py --seeds 42,123,7,2024,99

# Recommender evaluation (paired, honest GNN split) and 5-seed robustness
python scripts/evaluate_hybrid.py
python scripts/robustness_eval.py --seeds 42,123,7,2024,99

# Interactive demo (build assets once, then run)
python scripts/build_demo_assets.py && python scripts/build_planner_assets.py
streamlit run demo/app.py

# Explainable itinerary proof of concept
python scripts/run_itineraries.py
python scripts/eval_planning.py
python scripts/eval_faithfulness.py
```

The graph encoder trains message passing on training edges only and saves its
split, which `evaluate_hybrid.py` reuses so that every evaluated edge is one the
encoder never propagated over.

## Acknowledgements

Supervisor: dr inz. Robert Bembenik. Built on and correcting an earlier MSc
thesis (Amac, 2024) on the same task.
