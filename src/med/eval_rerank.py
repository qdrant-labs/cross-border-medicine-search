"""Does hyperbolic reranking actually help, and at what beta?

Runs offline against the saved vectors -- no server, no Qdrant -- so the sweep
is reproducible and fast. The question it answers is narrow and specific:

  A Polish box is photographed. We want the Spanish product with the *same*
  active ingredient. Cosine alone cannot separate "paracetamol" (N02BE01) from
  "paracetamol + ibuprofen" (N02BE51), because the two records are nearly the
  same string. Does distance in the ATC tree separate them, and does adding it
  cost anything on the queries that were already right?

The metric is exact-ATC accuracy at rank 1 over Spanish-filtered results. Two
extra numbers are reported because the headline can hide the trade:

  combo_errors  top-1 is a combination product (level-5 code ending 5x) when
                the gold code is a plain single molecule. This is the failure
                being targeted, so it should fall.
  broken        queries that were right at beta=0 and are wrong after. This is
                the cost, and if it climbs faster than combo_errors falls, the
                rerank is not worth shipping.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "artifacts" / "med"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shift_med import PIVOT, SHIFTED, load_corpus, shift_all  # noqa: E402

BETAS = (0.0, 0.04, 0.08, 0.12, 0.16, 0.24, 0.40)
QLANG = "pl"
ALPHA = 0.75
DIM = 256
K = 10
OVERFETCH = 5
HYP_SCALE = 13.0


def truncate(v, d):
    out = np.atleast_2d(v)[:, :d].astype(np.float32)
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.clip(n, 1e-12, None)


def hyp_matrix(p):
    """All-pairs Poincare distance for the 534 ATC nodes. Small enough to be
    dense, and precomputing it makes the beta sweep almost free."""
    sq = np.sum(p ** 2, axis=1)
    den = np.outer(1 - sq, 1 - sq)
    num = np.sum((p[:, None, :] - p[None, :, :]) ** 2, axis=2)
    return np.arccosh(1 + 2 * num / np.clip(den, 1e-12, None))


def is_combo(code):
    """ATC level 5 marks combinations by a 5 in the 6th character: N02BE51 is
    paracetamol with something else, N02BE01 is paracetamol alone."""
    return len(code) >= 7 and code[5] == "5"


def main():
    pairs, docs = load_corpus()
    dv = np.load(OUT / "arctic_docs.npy")
    qv = np.load(OUT / "arctic_queries.npy")
    off_d = {l: np.load(OUT / f"arctic_offset_doc_{l}.npy") for l in SHIFTED}
    off_q = {l: np.load(OUT / f"arctic_offset_query_{l}.npy") for l in SHIFTED}

    codes = json.loads((OUT / "atc_codes.json").read_text())
    cidx = {c: i for i, c in enumerate(codes)}
    H = hyp_matrix(np.load(OUT / "atc_poincare_10.npy"))

    langs = np.array([d["lang"] for d in docs])
    atcs = np.array([d["atc"] for d in docs])

    D = truncate(shift_all(dv, off_d, ALPHA, langs), DIM)
    Q = truncate(shift_all(qv, off_q, ALPHA, langs), DIM)

    # The corpus is three languages now, so the target side has to be selected
    # by name. "not the query language" would sweep Dutch in with Spanish.
    es = np.where(langs == PIVOT)[0]
    pl = np.where(langs == QLANG)[0]
    D_es, D_pl = D[es], D[pl]

    # Candidates and the anchor are computed once; only the blend changes with
    # beta, so the sweep costs one matmul in total rather than one per beta.
    # Accelerate's BLAS leaves FP status flags set on these matmuls; the inputs
    # are unit-norm and the outputs are in range. Same artifact as shift_med.
    np.seterr(all="ignore")

    cand, anchors, golds = [], [], []
    for qi in pl:
        sims = D_es @ Q[qi]
        top = np.argpartition(-sims, K * OVERFETCH)[:K * OVERFETCH]
        top = top[np.argsort(-sims[top])]
        cand.append((top, sims[top]))
        # Anchor: identify the box against its own language side, and never
        # against itself.
        s_pl = D_pl @ Q[qi]
        s_pl[np.where(pl == qi)[0][0]] = -1e9
        anchors.append(atcs[pl[int(np.argmax(s_pl))]])
        golds.append(atcs[qi])

    print(f"{len(pl)} {QLANG} queries -> {len(es)} {PIVOT} docs, "
          f"alpha={ALPHA}, dim={DIM}, overfetch={K * OVERFETCH}")
    anchor_ok = sum(a == g for a, g in zip(anchors, golds)) / len(pl)
    print(f"anchor identified correctly: {anchor_ok:.3f}\n")

    base_right = None
    print(f"{'beta':>6} {'top1':>7} {'top3':>7} {'combo_err':>10} {'broken':>7}")
    for b in BETAS:
        right, combo, top3 = [], 0, 0
        for n, (top, sims) in enumerate(cand):
            a = anchors[n]
            score = sims.copy()
            if b and a in cidx:
                d = np.array([H[cidx[a], cidx[atcs[es[t]]]]
                              if atcs[es[t]] in cidx else 0.0 for t in top])
                score = score - b * (d / HYP_SCALE)
            order = top[np.argsort(-score)]
            hit = atcs[es[order[0]]]
            right.append(hit == golds[n])
            top3 += int(golds[n] in [atcs[es[t]] for t in order[:3]])
            if hit != golds[n] and is_combo(hit) and not is_combo(golds[n]):
                combo += 1
        right = np.array(right)
        if base_right is None:
            base_right = right
        broken = int((base_right & ~right).sum())
        print(f"{b:>6} {right.mean():>7.3f} {top3 / len(cand):>7.3f} "
              f"{combo:>10} {broken:>7}")


if __name__ == "__main__":
    main()
