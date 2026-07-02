# Running the pipeline on the II/WUT HPC (SLURM)

This directory holds everything needed to run the recommender pipeline on
`hpc.ii.pw.edu.pl` with the **upgraded configuration**: a stronger backbone
(`roberta-base`, optionally `deberta-v3-base`) and a longer context
(`max_length=256`), made feasible by the 96 GB GPU on the `h86` node.

The model configuration is read from environment variables (set in
[`env.sh`](env.sh)), so the *same code* runs the laptop config (defaults) and
the HPC config (upgraded) with no source edits.

---

## 0. One-time access setup
1. **Account**: email `labadm.ii@pw.edu.pl` with **"HPC"** in the subject. Verify
   group membership with `id` (look for `hpc`).
2. **Initialise home on obelix** (mandatory, easy to miss): open
   `https://obelix.ii.pw.edu.pl` once, log in, accept the license. This creates
   your `$HOME`; without it, HPC login fails.
3. **Network**: connect from inside the II/WEiTI network, or via the institute
   **VPN**, or by tunnelling port 443.
4. **Log in**: `ssh <user>@hpc.ii.pw.edu.pl`.

## 1. Get the code
```bash
git clone https://github.com/ewenliam/RecSys-personality-tourist-recommender.git
cd RecSys-personality-tourist-recommender
```

## 2. Environment
```bash
module avail                       # discover real module names/versions
module load python/3.11 cuda/12.1  # then put these into hpc/env.sh
python -m venv ~/recsys-env && source ~/recsys-env/bin/activate
pip install -r requirements.txt    # keep numpy==1.26.4
# For the deberta-v3-base option only:
# pip install sentencepiece
```
Edit [`env.sh`](env.sh) so the `module load` lines match what `module avail`
shows on this cluster.

## 3. Pre-download the backbone (login node has internet, compute nodes do not)
```bash
python - <<'PY'
from transformers import AutoModel, AutoTokenizer
m = "roberta-base"   # or microsoft/deberta-v3-base
AutoTokenizer.from_pretrained(m); AutoModel.from_pretrained(m)
print("cached", m)
PY
```
This populates `~/.cache/huggingface`; jobs then run with
`TRANSFORMERS_OFFLINE=1`.

## 4. Get the data
The dataset and any pre-trained artifacts are **not** in git. Either:
- `rsync` your local `data/processed/*.parquet` and `data/raw/mbti_1.csv` up:
  ```bash
  rsync -avP data/ <user>@hpc.ii.pw.edu.pl:~/RecSys-.../data/
  ```
- or re-download the Yelp dataset on the cluster and run the preprocessing.

⚠️ Check your obelix home quota (`quota` or `du -sh ~`). The processed parquet
files are several GB; if the quota is tight, place `data/` and `models/` in
scratch space and update the paths in `src/config/settings.py`.

## 5. Submit the jobs
```bash
mkdir -p logs
# Stage 1: train the classifier (hours) with the upgraded backbone/context
JID=$(sbatch --parsable hpc/01_train_mbti.slurm)
# Stage 2: rebuild everything downstream once stage 1 succeeds
sbatch --dependency=afterok:$JID hpc/02_rebuild.slurm
```
Monitor with `squeue -u $USER`, inspect `logs/*.out`, and watch cluster load on
Grafana (`http://194.29.167.205/hpc`, internal network only).

## 6. Get results back
The honest classifier metrics print at the end of stage 1's log; the recommender
robustness tables land in `results/phase4_robustness_*.csv`. Copy them down with
`rsync`/`scp` and regenerate the thesis tables/figures locally with
`python scripts/generate_thesis_assets.py`.

---

## Tuning knobs (in `env.sh`)
| Variable | Canonical | Ablation alternates | Notes |
|----------|-----------|---------------------|-------|
| `MBTI_MODEL_NAME` | `bert-base-uncased` | `roberta-base`, `microsoft/deberta-v3-base` | deberta needs sentencepiece + fp32 load |
| `MBTI_MAX_LENGTH` | 128 | 256 | 512 also fits on 96 GB |
| `MBTI_BATCH_SIZE` | 32 | 64 | 16 on the 11-12 GB titan nodes |

## Outcome: the ablation is done, bert stays canonical
The backbone/context ablation was run on this cluster (see the thesis section
"Backbone and context ablation"). Honest user-disjoint per-user balanced
accuracy: bert/128 0.752, roberta/256 0.740, deberta-v3/128 0.743,
deberta-v3/256 0.761 -- and bert's own 3-seed spread is 0.742-0.772
(`hpc/multiseed_bert.slurm`, `results/bertvar_metrics.csv`). Every alternative
lands inside that spread, so **bert-base-uncased/128 remains the canonical
configuration** and no downstream (stage 2) rerun is needed. The laptop-trained
canonical pipeline and its thesis numbers stand.

Practical notes from the runs: compute nodes are offline (pre-download
backbones on the login node); deberta-v3 must be loaded in fp32 or the AMP
GradScaler fails; node `h86` once stalled a job silently for hours, so
multi-run scripts wrap each run in `timeout` (see `multiseed_bert.slurm`).
