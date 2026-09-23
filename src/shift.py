"""SHIFT: removing language bias from a multilingual embedding space.

Method (Qdrant blog, "Solving Multilingual RAG Language Bias"):

    offset[lang] = mean( embed(target_lang) - embed(pivot_lang) )   over parallel pairs
    indexed      = embed(doc)   - alpha * offset[doc.lang]
    queried      = embed(query) - alpha * offset[query.lang]

Corpus: the Google Product Taxonomy, published by Google in 8 languages with
identical category IDs. That gives a genuinely parallel corpus and exact
cross-lingual ground truth -- a category is "relevant" iff the ID matches.

E5 models need SEPARATE offsets for the "query:" and "passage:" prefixes.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import torch

# see note in server.py -- macOS Accelerate emits spurious warnings on big matmuls
warnings.filterwarnings("ignore", message=".*encountered in matmul",
                        category=RuntimeWarning)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ART = ROOT / "artifacts"
MODEL = "intfloat/multilingual-e5-small"
PIVOT = "en-US"
LANGS = ["en-US", "es-ES", "de-DE", "nl-NL", "pl-PL", "ja-JP", "fr-FR", "pt-BR", "it-IT"]
LANG_LABEL = {
    "en-US": "English", "es-ES": "Spanish", "de-DE": "German", "nl-NL": "Dutch",
    "pl-PL": "Polish", "ja-JP": "Japanese", "fr-FR": "French",
    "pt-BR": "Portuguese", "it-IT": "Italian",
}


# --------------------------------------------------------------------------
def load_parallel():
    """{lang: {category_id: (leaf_name, full_path)}} for every language file present.

    Queries use the leaf name, documents use the full path, so a same-language
    hit is a real retrieval result rather than an identical-string match.
    """
    out = {}
    for lang in LANGS:
        path = DATA / f"taxonomy.{lang}.txt"
        if not path.exists():
            continue
        d = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            raw_id, _, full = line.partition(" - ")
            parts = [p.strip() for p in full.split(">")]
            d[int(raw_id)] = (parts[-1], " > ".join(parts))
        out[lang] = d
    # keep only ids present in every language, so pairs are truly parallel
    common = set.intersection(*(set(d) for d in out.values()))
    return {lang: {k: v for k, v in d.items() if k in common} for lang, d in out.items()}, sorted(common)


def get_model():
    from sentence_transformers import SentenceTransformer
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return SentenceTransformer(MODEL, device=device)


def embed_all(force=False):
    """Embed every category name in every language, under both E5 prefixes."""
    ART.mkdir(exist_ok=True, parents=True)
    parallel, ids = load_parallel()
    cache = ART / "shift_ids.json"
    need = force or not cache.exists()
    if not need:
        for lang in parallel:
            for prefix in ("query", "passage"):
                if not (ART / f"e5_{lang}_{prefix}.npy").exists():
                    need = True
    if not need:
        return parallel, json.load(open(cache))

    model = get_model()
    for lang, d in parallel.items():
        # query = leaf name, passage = full category path
        for prefix, pick in (("query", 0), ("passage", 1)):
            texts = [f"{prefix}: {d[i][pick]}" for i in ids]
            vecs = model.encode(
                texts, batch_size=256, normalize_embeddings=True,
                show_progress_bar=False, convert_to_numpy=True,
            )
            np.save(ART / f"e5_{lang}_{prefix}.npy", vecs.astype(np.float32))
        print(f"  embedded {lang} ({len(ids)})")
    json.dump(ids, open(cache, "w"))
    return parallel, ids


def load_embeddings(langs, prefix):
    return {l: np.load(ART / f"e5_{l}_{prefix}.npy") for l in langs}


# --------------------------------------------------------------------------
def compute_offsets(emb, ids, rows, pivot=PIVOT):
    """offset[lang] = mean(embed(lang) - embed(pivot)) over the given rows."""
    return {
        lang: (emb[lang][rows] - emb[pivot][rows]).mean(axis=0)
        for lang in emb
    }


def apply_shift(vecs, offset, alpha, renormalize=True):
    out = vecs - alpha * offset
    if renormalize:
        n = np.linalg.norm(out, axis=-1, keepdims=True)
        out = out / np.clip(n, 1e-12, None)
    return out


# --------------------------------------------------------------------------
def build_corpus(emb_p, offsets_p, alpha, eval_rows):
    """Concatenate every language's shifted documents into one flat index."""
    langs = list(emb_p)
    vecs, cats, lang_of = [], [], []
    for lang in langs:
        vecs.append(apply_shift(emb_p[lang][eval_rows], offsets_p[lang], alpha))
        cats.append(np.asarray(eval_rows))
        lang_of += [lang] * len(eval_rows)
    return (np.concatenate(vecs).astype(np.float32),
            np.concatenate(cats),
            np.array(lang_of))


def evaluate(alpha, emb_q, emb_p, offsets_q, offsets_p, ids, eval_rows,
             query_langs=None, k=10, n_queries=300, seed=0):
    """Cross-lingual retrieval over the 9-language corpus.

    A document is relevant iff its category id matches the query, in any
    language. Every category has exactly one copy per language, so:

      overall recall@k  = relevant docs retrieved / 9
      same-language     = 1 if the query-language copy is in the top k
      cross-language    = other-language copies retrieved / 8
    """
    rng = np.random.default_rng(seed)
    langs = list(emb_p)
    query_langs = query_langs or langs
    n_lang = len(langs)

    doc_vecs, doc_cat, doc_lang = build_corpus(emb_p, offsets_p, alpha, eval_rows)

    picks = rng.choice(len(eval_rows), size=min(n_queries, len(eval_rows)), replace=False)
    rows = np.asarray(eval_rows)[picks]
    agg = {"overall": [], "same": [], "cross": [], "share": []}
    per_lang = {}

    for lang in query_langs:
        q = apply_shift(emb_q[lang][rows], offsets_q[lang], alpha).astype(np.float32)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            sims = q @ doc_vecs.T
        top = np.argpartition(-sims, k, axis=1)[:, :k]
        ordered = np.take_along_axis(
            top, np.argsort(-np.take_along_axis(sims, top, 1), axis=1), 1)

        o, s, c, sh = [], [], [], []
        for i, row in enumerate(rows):
            hit_cat = doc_cat[ordered[i]] == row
            hl = doc_lang[ordered[i]]
            o.append(int(hit_cat.sum()) / n_lang)
            s.append(float((hit_cat & (hl == lang)).any()))
            c.append(int((hit_cat & (hl != lang)).sum()) / (n_lang - 1))
            sh.append(float((hl == lang).mean()))
        per_lang[lang] = {
            "overall": float(np.mean(o)), "same": float(np.mean(s)),
            "cross": float(np.mean(c)), "same_lang_share": float(np.mean(sh)),
        }
        agg["overall"] += o; agg["same"] += s; agg["cross"] += c; agg["share"] += sh

    return {
        "alpha": alpha,
        "recall@10": float(np.mean(agg["overall"])),
        "same_language_recall@10": float(np.mean(agg["same"])),
        "cross_language_recall@10": float(np.mean(agg["cross"])),
        "same_language_share@10": float(np.mean(agg["share"])),
        "per_lang": per_lang,
    }
