"""Does late interaction actually earn its place in this pipeline?

The demo already retrieves with a single dense vector per document. ColBERT
replaces that with one vector per token and scores by MaxSim -- for every query
token, the best-matching document token, summed. The claim is that this helps
where a single pooled vector blurs: "500 MG/200 MG" versus "1 G", "paracetamol"
versus "paracetamol/ibuprofeno". Those distinctions live in a couple of tokens
and pooling averages them away.

That claim is testable here and it is worth testing rather than asserting,
because a first candidate model failed it badly. answerai-colbert-small-v1
scored plain paracetamol and the paracetamol+ibuprofen combination at an
identical 30.00, and an unrelated antihistamine 1.97 below an exact self-match.
A reranker that cannot separate those would cost latency and buy nothing.

Scored on the real task, not on hand-picked strings: Polish query text against
Spanish documents only, which is what the cross-border lookup does. Four
configurations share one candidate list so the comparison is like for like:

  dense            what the server does today at alpha=0.75, dim=256
  dense+hyp        plus the ATC-tree rerank already in serve_med.py
  colbert          dense prefetch, reordered by MaxSim
  colbert+hyp      both reranks, MaxSim first

The prefetch depth matters and is reported: ColBERT can only reorder what dense
handed it, so recall@PREFETCH is the ceiling on every reranked row.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "artifacts" / "med"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shift_med import (MODEL, PIVOT, QUERY_PREFIX, SHIFTED,  # noqa: E402
                       load_corpus, shift_all)

QLANG = "pl"

ALPHA = 0.75
DIM = 256
PREFETCH = 50
K = 10
N_QUERIES = 300
HYP_SCALE = 13.0
HYP_BETA = 0.08
SEED = 0


def truncate(v, d):
    out = np.atleast_2d(v)[:, :d].astype(np.float32)
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.clip(n, 1e-12, None)


def maxsim(q, d):
    """MaxSim: each query token takes its best document token, then sum."""
    return float(np.max(q @ d.T, axis=1).sum())


def hyp_all_pairs(p):
    sq = np.sum(p ** 2, axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * (p @ p.T), 0.0)
    den = np.clip((1 - sq)[:, None] * (1 - sq)[None, :], 1e-12, None)
    return np.arccosh(np.clip(1 + 2 * d2 / den, 1.0, None))


def main(model_name):
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

    # Only ATC codes with a Spanish product are answerable at all.
    has_es = set(es_atc.tolist())
    cand = [i for i in np.where(langs == QLANG)[0] if atcs[i] in has_es]
    rng = np.random.default_rng(SEED)
    qidx = rng.choice(cand, size=min(N_QUERIES, len(cand)), replace=False)

    # Hyperbolic tree distances, same 10D embedding the server reranks with.
    codes = json.loads((OUT / "atc_codes.json").read_text())
    code_idx = {c: i for i, c in enumerate(codes)}
    HYP = hyp_all_pairs(np.load(OUT / "atc_poincare_10.npy"))

    from fastembed import LateInteractionTextEmbedding
    print(f"loading {model_name} ...", flush=True)
    cb = LateInteractionTextEmbedding(model_name)
    print(f"encoding {len(es_text)} spanish docs ...", flush=True)
    CB_D = list(cb.embed(es_text, batch_size=32))
    q_texts = [docs[i]["text"] for i in qidx]
    print(f"encoding {len(q_texts)} queries ...", flush=True)
    CB_Q = list(cb.query_embed(q_texts))

    rows = {"dense": [], "dense+hyp": [], "colbert": [], "colbert+hyp": []}
    recall_pf = 0
    for n, i in enumerate(qidx):
        gold = atcs[i]
        sims = D_es @ Q[i]
        pf = np.argpartition(-sims, PREFETCH)[:PREFETCH]
        pf = pf[np.argsort(-sims[pf])]
        recall_pf += int(gold in es_atc[pf])

        # The anchor is the query's own molecule resolved in its own language;
        # here that is known exactly, so the rerank is measured without the
        # anchor step's own 0.930 accuracy folded in.
        gi = code_idx.get(gold)
        hyp = np.array([HYP[gi, code_idx[c]] if gi is not None
                        and c in code_idx else 0.0 for c in es_atc[pf]])

        cos = sims[pf]
        cbs = np.array([maxsim(CB_Q[n], CB_D[j]) for j in pf])
        # MaxSim is an unbounded sum over query tokens, so it cannot be added to
        # a cosine. Rank-normalise to [0,1] within the candidate list instead.
        cbn = (cbs - cbs.min()) / max(cbs.max() - cbs.min(), 1e-9)

        order = {
            "dense": np.argsort(-cos),
            "dense+hyp": np.argsort(-(cos - HYP_BETA * hyp / HYP_SCALE)),
            "colbert": np.argsort(-cbn),
            "colbert+hyp": np.argsort(-(cbn - HYP_BETA * hyp / HYP_SCALE)),
        }
        for name, o in order.items():
            got = es_atc[pf][o][:K].tolist()
            rows[name].append((int(gold in got[:1]), int(gold in got[:3]),
                               int(gold in got[:K])))

    n = len(qidx)
    print(f"\nmodel={model_name}")
    print(f"alpha={ALPHA} dim={DIM} prefetch={PREFETCH} n={n}")
    print(f"recall@{PREFETCH} = {recall_pf / n:.3f}   (ceiling for every rerank)\n")
    print(f"{'config':<14} {'top1':>7} {'top3':>7} {'top10':>7}")
    for name, r in rows.items():
        a = np.array(r)
        print(f"{name:<14} {a[:, 0].mean():>7.3f} {a[:, 1].mean():>7.3f} "
              f"{a[:, 2].mean():>7.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "colbert-ir/colbertv2.0")
