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
| Variable | Laptop | HPC (upgraded) | Notes |
|----------|--------|----------------|-------|
| `MBTI_MODEL_NAME` | `bert-base-uncased` | `roberta-base` | `deberta-v3-base` is stronger (needs sentencepiece) |
| `MBTI_MAX_LENGTH` | 128 | 256 | 512 also fits on 96 GB |
| `MBTI_BATCH_SIZE` | 32 | 64 | raise further if VRAM allows |

## Important: the numbers will change
Running with a different backbone and context length will produce **different**
(expected: better, more honest) classifier and recommender numbers than the
laptop run currently written into the thesis. After the HPC run, re-copy the
result CSVs, regenerate the tables/figures, and **update the thesis numbers**
accordingly. Keep a note of which configuration produced which results.
