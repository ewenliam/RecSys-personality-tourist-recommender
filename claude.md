# Tourist Recommender (MSc thesis, WUT)

Personality-aware tourist venue recommender. Extends and corrects Omer Amac's
2024 thesis (his ~94% MBTI accuracy was leaked; the honest replication is the
core contribution). Supervisor: dr inz. Robert Bembenik.

**Deadline: thesis submission ~August 2026 (1 month). Priority order: thesis
text/figures > defense demo stability > anything else. Do not start new
experiments unless I ask.**

## Environment (check before doing anything)
- Canonical interpreter: `venv\Scripts\activate` at repo root (fixed
  2026-07-07: numpy restored to 1.26.4, streamlit added; earlier results were
  produced on system Python 3.12, which also holds a correct stack).
- Before trusting any interpreter, verify:
  `python -c "import numpy; print(numpy.__version__)"` must print `1.26.4`.
  (Reason: numpy 2.x breaks numba/UMAP/scipy - the topic pipeline dies.)
- Required env var for UMAP work: `NUMBA_DISABLE_INTEL_SVML=1`.
- GPU: RTX 4060 Laptop, 8 GB. Gradient checkpointing + AMP for BERT training.

## Canonical results - settled, do not reopen
- Classifier: bert-base-uncased/128 tokens. Honest, user-disjoint, per-user:
  acc 0.801 / balanced 0.752 / exact-16 0.445.
- Backbone ablation is CLOSED: roberta/256 (0.740), deberta-v3/128 (0.743),
  deberta-v3/256 (0.761) all fall inside bert's own 3-seed spread
  (0.742-0.772). The dataset is the ceiling, not model capacity.
- Recommender (5 seeds, K=10): NDCG 0.0153, MRR 0.0238, HR 0.0524. Full
  4-signal RRF hybrid is best on every metric.
- Never claim a ranking-metric win over Omer - he reported no ranking
  metrics. The comparison is methodological only.

## Architecture (as built)
BERT-MBTI (4 binary heads, class-weighted loss, user-disjoint split)
-> BERTopic on MBTI CLS embeddings -> heterogeneous GraphSAGE (1 layer,
BPR + temperature 0.1 + popularity-weighted hard negatives)
-> RRF fusion (k=60) + popularity tie-breaker (alpha=0.01, post-fusion).
Venue personality = visitor-mean profile, NOT review-text embedding.
(Reason: review-text venue embeddings are near-identical, cosine std ~0.04.)
Full detail: explain.md and the thesis. README.md now documents the as-built
system (rewritten 2026-07; the recommender numbers there are the paired
protocol: HR@10 0.038, personality ablation significant).

## Commands
- Classifier: `python scripts/train_mbti.py --epochs 4 [--seed N]` (~8 h)
- Full rebuild: `python scripts/rebuild_pipeline.py --seeds 42,123,7,2024,99`
- Demo assets:  `python scripts/build_demo_assets.py` then
  `python scripts/build_planner_assets.py`
- Demo:         `streamlit run demo/app.py` (3 modes; planner mode needs both
  asset scripts run first)
- Itinerary PoC: `python scripts/run_itineraries.py`, then
  `eval_planning.py` and `eval_faithfulness.py`
- Thesis assets: `python scripts/generate_thesis_assets.py` and
  `generate_extra_figures.py` (see docs/thesis/CLAUDE.md for sync rules)

## Layout and what is public
- Pushed to GitHub: code only (src/, scripts/, hpc/, demo/, requirements).
- Local only (gitignored): data/, models/, results/, docs/ (the thesis!),
  venv/. Never try to commit these; never assume a fresh clone has them.
- src/planner/ + src/explain/ = itinerary PoC. Its per-signal ranks are the
  explanation evidence - keep them flowing through any refactor.

## Gotchas that cost real time
- deberta-v3 loads fp16 by default and crashes AMP GradScaler - the fp32
  load in src/models/bert_mbti/model.py must stay.
- argparse defaults silently override intended config (a default num-layers=2
  once undid the 1-layer fix). When changing a default, grep for the flag.
- Three different index orders exist (BERTopic rows, GNN id_mappings, pandas
  order). Always align through models/gnn_hetero id mappings.
- tables.tex is auto-generated - edit scripts/generate_thesis_assets.py,
  never the .tex output.

## HPC (hpc.ii.pw.edu.pl) - only if I re-open it
- Guided mode per global rules. Full setup guide: hpc/README.md.
- Node h86 (96 GB) is fast but flaky: launch faults (RaisedSignal:53) and
  silent hangs. titan3 (12 GB) is slow but reliable; shrink SBATCH to
  --mem=48G --cpus-per-task=8 for small nodes. Compute nodes are offline -
  pre-download HF models on the login node.
