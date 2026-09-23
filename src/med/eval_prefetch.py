"""How much slack does the cheap Matryoshka stage need before it stops losing?

Prefetch is the production shape of the Matryoshka claim. The chart says the
first 256 coordinates retain the quality of all 1,024; the query plan says
therefore search a truncated copy first and only rescore the survivors at full
width. Qdrant does both stages server-side, so the small vector never leaves
the database and the saving is real rather than notional.

What has to be measured is the overfetch factor. The cheap stage returns
limit * FACTOR candidates and the expensive stage can only reorder those, so
any document the 64-dim search misses is lost no matter how good the reranker
is. Too small a factor silently caps quality; too large and the two-stage plan
costs more than the single-stage one it replaced.

Reported as agreement with the full-width top-10 -- the question is not whether
prefetch finds the right molecule, but whether it returns the same answer the
expensive search would have, since that is what makes it a safe substitution.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "artifacts" / "med"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shift_med import PIVOT, SHIFTED, load_corpus, shift_all  # noqa: E402

ALPHA = 0.75
FULL_DIM = 1024
PREFETCH_DIM = 64
K = 10
FACTORS = (1, 2, 4, 8, 16, 32)
N_QUERIES = 400
SEED = 0


def truncate(v, d):
    out = np.atleast_2d(v)[:, :d].astype(np.float32)
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.clip(n, 1e-12, None)


def main():
    np.seterr(all="ignore")
    pairs, docs = load_corpus()
    dv = np.load(OUT / "arctic_docs.npy")
    qv = np.load(OUT / "arctic_queries.npy")
    off_d = {l: np.load(OUT / f"arctic_offset_doc_{l}.npy") for l in SHIFTED}
    off_q = {l: np.load(OUT / f"arctic_offset_query_{l}.npy") for l in SHIFTED}

    langs = np.array([d["lang"] for d in docs])
    atcs = np.array([d["atc"] for d in docs])
    D = shift_all(dv, off_d, ALPHA, langs)
    Q = shift_all(qv, off_q, ALPHA, langs)

    es = np.where(langs == PIVOT)[0]
    D_full = truncate(D, FULL_DIM)[es]
    D_small = truncate(D, PREFETCH_DIM)[es]
    es_atc = atcs[es]

    has_es = set(es_atc.tolist())
    cand = [i for i in range(len(docs))
            if docs[i]["lang"] != PIVOT and atcs[i] in has_es]
    rng = np.random.default_rng(SEED)
    qidx = rng.choice(cand, size=min(N_QUERIES, len(cand)), replace=False)

    Q_full = truncate(Q, FULL_DIM)
    Q_small = truncate(Q, PREFETCH_DIM)

    print(f"alpha={ALPHA}  {PREFETCH_DIM}d prefetch -> {FULL_DIM}d rescore, "
          f"k={K}, n={len(qidx)}")
    print(f"bytes/vector: {PREFETCH_DIM * 4} vs {FULL_DIM * 4} "
          f"({FULL_DIM // PREFETCH_DIM}x smaller)\n")
    print(f"{'factor':>7} {'cands':>6} {'agree@10':>9} {'top1 same':>10} "
          f"{'recall_atc':>11}")

    base_top, base_hit = [], 0
    for i in qidx:
        s = D_full @ Q_full[i]
        t = np.argpartition(-s, K)[:K]
        t = t[np.argsort(-s[t])]
        base_top.append(t)
        base_hit += int(atcs[i] in es_atc[t])
    print(f"{'full':>7} {len(es):>6} {1.0:>9.3f} {1.0:>10.3f} "
          f"{base_hit / len(qidx):>11.3f}")

    for f in FACTORS:
        n_cand = K * f
        agree, same1, hit = 0, 0, 0
        for n, i in enumerate(qidx):
            s_small = D_small @ Q_small[i]
            if n_cand < len(s_small):
                pf = np.argpartition(-s_small, n_cand)[:n_cand]
            else:
                pf = np.arange(len(s_small))
            # Stage two: rescore only the shortlist, at full width.
            s_full = D_full[pf] @ Q_full[i]
            top = pf[np.argsort(-s_full)][:K]
            agree += len(set(top.tolist()) & set(base_top[n].tolist())) / K
            same1 += int(top[0] == base_top[n][0])
            hit += int(atcs[i] in es_atc[top])
        m = len(qidx)
        print(f"{f:>7} {n_cand:>6} {agree / m:>9.3f} {same1 / m:>10.3f} "
              f"{hit / m:>11.3f}")

    (OUT / "prefetch_eval.json").write_text(json.dumps(
        {"prefetch_dim": PREFETCH_DIM, "full_dim": FULL_DIM,
         "k": K, "n": len(qidx)}, indent=1))


if __name__ == "__main__":
    main()
