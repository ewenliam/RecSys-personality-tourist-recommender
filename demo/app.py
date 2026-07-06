#!/usr/bin/env python3
"""
Streamlit demo for the personality-aware tourist venue recommender.

Serves the exact final system from the thesis: Reciprocal Rank Fusion of four
signals (content KNN, collaborative GNN, MBTI personality, popularity
tie-breaker) over precomputed embeddings. No model in the serving path except
optional live BERT-MBTI inference for the cold-start ("new user") mode.

Prerequisite:  python scripts/build_demo_assets.py
Run:           streamlit run demo/app.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEMO_DIR = PROJECT_ROOT / "models" / "demo"
GNN_DIR = PROJECT_ROOT / "models" / "gnn_hetero"
CKPT_PATH = PROJECT_ROOT / "models" / "checkpoints" / "bert_mbti" / "best_model.pt"

RRF_K = 60          # smoothing constant, same as the thesis (k=60)
TRAIT_NAMES = [("E", "I"), ("S", "N"), ("T", "F"), ("J", "P")]


# ---------------------------------------------------------------------------
# Asset loading (cached across reruns)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading precomputed embeddings...")
def load_assets():
    a = {}
    a["venue_meta"] = pd.read_parquet(DEMO_DIR / "venue_meta.parquet")
    a["user_index"] = pd.read_parquet(DEMO_DIR / "user_index.parquet")
    a["venue_pca"] = np.load(DEMO_DIR / "venue_bertopic_pca64.npy")
    a["venue_mbti"] = np.load(DEMO_DIR / "venue_mbti_profiles.npy").astype(np.float32)
    a["venue_logdeg"] = np.load(DEMO_DIR / "venue_log_degree.npy")
    a["mbti_center"] = np.load(DEMO_DIR / "mbti_center.npy")
    # Big per-user matrices stay memory-mapped; rows are sliced on demand.
    a["knn_user"] = np.load(DEMO_DIR / "knn_user_profiles.npy", mmap_mode="r")
    a["user_mbti"] = np.load(DEMO_DIR / "user_mbti_centered.npy", mmap_mode="r")
    a["user_traits"] = np.load(DEMO_DIR / "user_traits.npy", mmap_mode="r")
    a["visits_indptr"] = np.load(DEMO_DIR / "visits_indptr.npy")
    a["visits_venues"] = np.load(DEMO_DIR / "visits_venues.npy")
    a["user_gnn"] = np.load(GNN_DIR / "user_embeddings.npy", mmap_mode="r")
    a["venue_gnn"] = np.load(GNN_DIR / "venue_embeddings.npy").astype(np.float32)
    return a


@st.cache_resource(show_spinner="Loading BERT-MBTI (first cold-start use)...")
def load_mbti_model():
    import torch
    from transformers import AutoTokenizer
    from src.config.settings import get_config
    from src.models.bert_mbti.model import MBTIMultiLabelClassifier

    config = get_config().bert
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = MBTIMultiLabelClassifier(config=config)
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()
    return model, tokenizer, config


# ---------------------------------------------------------------------------
# Core ranking (mirrors RRFRanker in scripts/evaluate_hybrid.py)
# ---------------------------------------------------------------------------

def cosine_scores(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    d = min(query.shape[0], matrix.shape[1])
    q = query[:d].astype(np.float32)
    m = matrix[:, :d]
    dots = m @ q
    return dots / ((np.linalg.norm(q) + 1e-8)
                   * (np.linalg.norm(m, axis=1) + 1e-8))


def scores_to_ranks(scores: np.ndarray) -> np.ndarray:
    ranks = np.empty(len(scores), dtype=np.int32)
    ranks[np.argsort(-scores)] = np.arange(1, len(scores) + 1)
    return ranks


def rrf_recommend(signal_queries: dict, assets: dict, k: int,
                  pop_alpha: float, exclude: np.ndarray | None):
    """
    signal_queries: {"content": vec64 | None, "gnn": vec | None,
                     "mbti": vec768 | None}
    Returns (top_idx, fused_scores, per_signal_ranks dict).
    """
    n_venues = assets["venue_gnn"].shape[0]
    fused = np.zeros(n_venues, dtype=np.float64)
    per_ranks = {}

    keys = {"content": "venue_pca", "gnn": "venue_gnn", "mbti": "venue_mbti"}
    for name, venue_key in keys.items():
        q = signal_queries.get(name)
        if q is None:
            continue
        ranks = scores_to_ranks(cosine_scores(q, assets[venue_key]))
        per_ranks[name] = ranks
        fused += 1.0 / (RRF_K + ranks)

    if pop_alpha > 0:
        fused *= 1.0 + pop_alpha * assets["venue_logdeg"]
        per_ranks["popularity"] = scores_to_ranks(assets["venue_logdeg"])

    if exclude is not None and len(exclude):
        fused[exclude] = -np.inf

    top = np.argsort(-fused)[:k]
    return top, fused[top], per_ranks


def visited_venues(assets: dict, user_idx: int) -> np.ndarray:
    lo, hi = assets["visits_indptr"][user_idx], assets["visits_indptr"][user_idx + 1]
    return assets["visits_venues"][lo:hi]


def traits_to_type(traits: np.ndarray) -> str:
    """traits = [P(I), P(N), P(F), P(P)] -> 4-letter type string."""
    return "".join(second if p >= 0.5 else first
                   for (first, second), p in zip(TRAIT_NAMES, traits))


def rec_table(assets: dict, top: np.ndarray, per_ranks: dict) -> pd.DataFrame:
    meta = assets["venue_meta"].iloc[top]
    out = pd.DataFrame({
        "#": np.arange(1, len(top) + 1),
        "Venue": meta["name"].values,
        "City": meta["city"].values,
        "Stars": meta["stars"].values,
        "Visits": meta["n_visits"].values,
        "Categories": [str(c)[:60] for c in meta["categories"].values],
    })
    label = {"content": "Content rank", "gnn": "GNN rank",
             "mbti": "MBTI rank", "popularity": "Pop. rank"}
    for name, ranks in per_ranks.items():
        out[label[name]] = ranks[top]
    return out


def show_traits(traits: np.ndarray):
    mbti = traits_to_type(traits)
    st.metric("Inferred MBTI type", mbti)
    cols = st.columns(4)
    for col, (first, second), p in zip(cols, TRAIT_NAMES, traits):
        p = float(p)
        letter, conf = (second, p) if p >= 0.5 else (first, 1 - p)
        col.progress(conf, text=f"{first}/{second} -> {letter} ({conf:.0%})")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Personality-aware venue recommender",
                   layout="wide")
st.title("Personality-aware tourist venue recommender")
st.caption(
    "Reciprocal Rank Fusion of four signals: content (BERTopic), "
    "collaborative (GraphSAGE), personality (BERT-MBTI, visitor-mean venue "
    "profiles), and a popularity tie-breaker. Serving path: cosine "
    "similarities + rank merge over precomputed embeddings."
)

if not DEMO_DIR.exists():
    st.error("Demo assets missing - run `python scripts/build_demo_assets.py` first.")
    st.stop()

assets = load_assets()

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Mode", ["Existing user", "New user (cold start)"])
    top_k = st.slider("Top-K", 5, 30, 10)
    st.subheader("Signals")
    use_content = st.checkbox("Content (BERTopic KNN)", value=True)
    use_gnn = st.checkbox("Collaborative (GNN)", value=True)
    use_mbti = st.checkbox("Personality (MBTI)", value=True)
    use_pop = st.checkbox("Popularity tie-breaker (alpha=0.01)", value=True)
    pop_alpha = 0.01 if use_pop else 0.0

# ---------------------------------------------------------------- existing --
if mode == "Existing user":
    ui = assets["user_index"]
    eligible = ui[ui["n_visits"] >= 5]
    st.subheader("Pick a user")
    c1, c2 = st.columns([2, 1])
    min_visits = c2.number_input("Min. visits", 5, 500, 20)
    pool = eligible[eligible["n_visits"] >= min_visits]
    sample = pool.sample(min(200, len(pool)), random_state=0).sort_values(
        "n_visits", ascending=False)
    choice = c1.selectbox(
        "User (id - visit count)",
        sample.index,
        format_func=lambda i: f"{ui.loc[i, 'user_id'][:18]}...  "
                              f"({ui.loc[i, 'n_visits']} visits)",
    )
    user_idx = int(choice)

    traits = np.asarray(assets["user_traits"][user_idx], dtype=np.float32)
    if not np.isnan(traits[0]):
        show_traits(traits)

    vis = visited_venues(assets, user_idx)
    with st.expander(f"Visit history ({len(vis)} venues)"):
        st.dataframe(assets["venue_meta"].iloc[vis][
            ["name", "city", "stars", "categories"]], height=240)

    queries = {
        "content": np.asarray(assets["knn_user"][user_idx], np.float32)
        if use_content else None,
        "gnn": np.asarray(assets["user_gnn"][user_idx], np.float32)
        if use_gnn else None,
        "mbti": np.asarray(assets["user_mbti"][user_idx], np.float32)
        if use_mbti else None,
    }
    exclude = vis if st.checkbox("Exclude already-visited venues", True) else None

    if st.button("Recommend", type="primary"):
        top, scores, per_ranks = rrf_recommend(
            queries, assets, top_k, pop_alpha, exclude)
        st.subheader("Recommendations (fused)")
        st.dataframe(rec_table(assets, top, per_ranks), hide_index=True)
        st.caption(
            "Per-signal rank columns show where each venue stood in each "
            "individual ranking (1 = best of 85,857) - venues surface when "
            "several signals independently rank them high."
        )

# -------------------------------------------------------------- cold start --
else:
    st.subheader("New user: personality from text")
    st.write(
        "Paste a few sentences the user has written (reviews, posts). The "
        "BERT-MBTI classifier infers a personality profile, which drives "
        "recommendations even with **zero interaction history**."
    )
    text = st.text_area("User text", height=160, placeholder=(
        "e.g. I usually avoid crowded places - give me a quiet coffee shop "
        "with a bookshelf and I am happy for hours. Big group dinners drain "
        "me, but I love discovering a hidden gallery or a tiny ramen bar."))

    if st.button("Infer personality & recommend", type="primary") and text.strip():
        import torch
        import torch.nn.functional as F

        model, tokenizer, config = load_mbti_model()
        enc = tokenizer(text, max_length=config.max_length,
                        padding="max_length", truncation=True,
                        return_tensors="pt")
        with torch.no_grad():
            out = model.bert(input_ids=enc["input_ids"],
                             attention_mask=enc["attention_mask"])
            cls = out.last_hidden_state[:, 0, :]
            pooled = model.dropout(cls)
            probs = [F.softmax(model.classifiers[d](pooled), dim=-1)[0, 1].item()
                     for d in model.DIMENSIONS]
            cls = F.normalize(cls, p=2, dim=-1)[0].numpy()

        # Identical transform to the precomputed users: center, renormalise.
        q_mbti = cls - assets["mbti_center"]
        q_mbti /= np.linalg.norm(q_mbti) + 1e-8

        show_traits(np.array(probs, dtype=np.float32))

        queries = {"content": None, "gnn": None,
                   "mbti": q_mbti if use_mbti else None}
        if not use_mbti:
            st.warning("Cold-start needs the MBTI signal - enable it in the sidebar.")
        else:
            top, scores, per_ranks = rrf_recommend(
                queries, assets, top_k, pop_alpha, None)
            st.subheader("Recommendations (personality + popularity only)")
            st.dataframe(rec_table(assets, top, per_ranks), hide_index=True)
            st.caption(
                "Content and collaborative signals need interaction history; "
                "for a brand-new user the personality pathway alone already "
                "produces a personalised ranking - the cold-start argument."
            )
