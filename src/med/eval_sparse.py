"""Sparse neural retrieval on the cross-border lookup, measured against dense.

Medicine names are an unusually good case for lexical matching and an unusually
bad one, depending on the product, and the split is not subtle:

  ibuprofen / ibuprofeno / ibuprofeen    INN, near-identical across borders
  Apap      / ANTIDOL                    brand, no shared characters at all

A dense multilingual encoder plus SHIFT handles the second and is wasteful on
the first. An inverted index handles the first perfectly and cannot touch the
second. That is the textbook argument for hybrid retrieval, and this measures
whether it holds on real registry data rather than assuming it.

Four scorers, same 300 Polish queries against the same 1,248 Spanish documents:

  bm25       Qdrant/bm25, IDF-weighted, Spanish stemmer on the document side
  minicoil   Qdrant/minicoil-v1, term weighting with word-sense resolution
  splade     prithivida/Splade_PP_en_v1, term expansion (English-trained)
  dense      arctic + SHIFT at alpha=0.75, dim=256 -- the current pipeline

then RRF fusion of dense with each sparse scorer. RRF is used rather than a
weighted score sum because BM25 scores and cosines are not on a shared scale
and normalising them per query list would make the fusion depend on how many
candidates happened to be retrieved.

Measured on the three-language corpus (n=300, alpha=0.75, dim=256):

  dense          0.827 / 0.913 / 0.953
  bm25           0.230 / 0.260 / 0.307     160/300 queries share no term at all
  minicoil       0.180 / 0.207 / 0.253     168/300 share no term
  splade         0.770 / 0.833 / 0.930     0/300 share no term
  dense+bm25     0.340 / 0.473 / 0.807     fusion far below dense
  dense+splade   0.837 / 0.920 / 0.970     fusion above dense at every cut

The split is not the one the argument above predicts. Exact matching fails the
way it should. Term expansion does not: SPLADE finds a shared term on every
single query, and fusing it with dense is the best configuration measured here.
That holds despite the model being English-trained, which makes it a stronger
result than it looks and not a safe one to generalise from.

A caveat that has to stay attached to any BM25 number here: fastembed ships
Snowball stopwords and stemmers for 18 languages and Polish is not one of them.
Spanish, German, Dutch and Portuguese all are. So the Polish query side runs
unstemmed, which is the weaker half of the pair, and Polish is a heavily
inflected language -- this understates BM25 for Polish specifically, and the
same code would do better for the languages being added.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "artifacts" / "med"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shift_med import PIVOT, SHIFTED, load_corpus, shift_all  # noqa: E402

QLANG = "pl"
ALPHA = 0.75
DIM = 256
K = 10
N_QUERIES = 300
RRF_K = 60
SEED = 0

SPARSE = [
    ("bm25", "Qdrant/bm25", {"language": "spanish"}),
    ("minicoil", "Qdrant/minicoil-v1", {}),
    ("splade", "prithivida/Splade_PP_en_v1", {}),
]


def truncate(v, d):
    out = np.atleast_2d(v)[:, :d].astype(np.float32)
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.clip(n, 1e-12, None)


def to_dict(se):
    return dict(zip(se.indices.tolist(), se.values.tolist()))


def sparse_dot(q, d):
    """Inverted-index scoring, written out: shared terms only."""
    if len(q) > len(d):
        q, d = d, q
    return sum(v * d[i] for i, v in q.items() if i in d)


def rrf(*rank_lists):
    """Reciprocal rank fusion. Rank-based, so unscaled scores can be combined."""
    agg = {}
    for ranks in rank_lists:
        for pos, j in enumerate(ranks):
            agg[j] = agg.get(j, 0.0) + 1.0 / (RRF_K + pos + 1)
    return [j for j, _ in sorted(agg.items(), key=lambda kv: -kv[1])]


def score_at(order, es_atc, gold):
    got = [es_atc[j] for j in order[:K]]
    return int(gold in got[:1]), int(gold in got[:3]), int(gold in got[:K])


def main():
    np.seterr(all="ignore")
    pairs, docs = load_corpus()
    dv = np.load(OUT / "arctic_docs.npy")
    qv = np.load(OUT / "arctic_queries.npy")
    off_d = {l: np.load(OUT / f"arctic_offset_doc_{l}.npy") for l in SHIFTED}
    off_q = {l: np.load(OUT / f"arctic_offset_query_{l}.npy") for l in SHIFTED}

    langs = np.array([d["lang"] for d in docs])
    atcs = np.array([d["atc"] for d in docs])

    D = truncate(shift_all(dv, off_d, ALPHA, langs), DIM)
    Q = truncate(shift_all(qv, off_q, ALPHA, langs), DIM)

    es = np.where(langs == PIVOT)[0]
    D_es = D[es]
    es_atc = atcs[es]
    es_text = [docs[i]["text"] for i in es]

    has_es = set(es_atc.tolist())
    cand = [i for i in np.where(langs == QLANG)[0] if atcs[i] in has_es]
    rng = np.random.default_rng(SEED)
    qidx = rng.choice(cand, size=min(N_QUERIES, len(cand)), replace=False)
    q_texts = [docs[i]["text"] for i in qidx]
    golds = [atcs[i] for i in qidx]

    n = len(qidx)
    dense_orders = [np.argsort(-(D_es @ Q[i])).tolist() for i in qidx]
    res = {"dense": [score_at(o, es_atc, g) for o, g in zip(dense_orders, golds)]}

    from fastembed import SparseTextEmbedding
    for label, name, kw in SPARSE:
        print(f"loading {label} ...", flush=True)
        try:
            m = SparseTextEmbedding(name, **kw)
            dsp = [to_dict(x) for x in m.embed(es_text, batch_size=32)]
            qsp = [to_dict(x) for x in m.query_embed(q_texts)]
        except Exception as e:                      # noqa: BLE001
            print(f"  {label} unavailable: {e}")
            continue

        rows, fused = [], []
        empty = 0
        for i in range(n):
            s = np.array([sparse_dot(qsp[i], d) for d in dsp])
            if not s.any():
                empty += 1                          # no shared term with anything
            order = np.argsort(-s).tolist()
            rows.append(score_at(order, es_atc, golds[i]))
            fused.append(score_at(rrf(dense_orders[i][:100], order[:100]),
                                  es_atc, golds[i]))
        res[label] = rows
        res[f"dense+{label}"] = fused
        print(f"  {label}: {empty}/{n} queries shared no term with any document")

    print(f"\nalpha={ALPHA} dim={DIM} n={n} fusion=RRF(k={RRF_K})\n")
    print(f"{'config':<16} {'top1':>7} {'top3':>7} {'top10':>7}")
    for name in ("dense", "bm25", "minicoil", "splade",
                 "dense+bm25", "dense+minicoil", "dense+splade"):
        if name not in res:
            continue
        a = np.array(res[name])
        print(f"{name:<16} {a[:, 0].mean():>7.3f} {a[:, 1].mean():>7.3f} "
              f"{a[:, 2].mean():>7.3f}")

    (OUT / "sparse_eval.json").write_text(json.dumps(
        {k: [int(x[0]) for x in v] for k, v in res.items()}))


if __name__ == "__main__":
    main()
