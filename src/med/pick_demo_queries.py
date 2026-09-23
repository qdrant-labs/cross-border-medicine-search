"""Find queries where the SHIFT off/on contrast is actually visible.

Runs per source language, because a query that demonstrates the effect in one
language says nothing about another. Dutch and Spanish share far more surface
vocabulary than Polish and Spanish do, so a Dutch query often reaches the right
Spanish product with no correction at all -- which makes the off/on panels look
identical and the method look pointless, even though the underlying retrieval
is fine (Dutch top-1 is 0.910, better than Polish at 0.869). The queries worth
showing are the ones where the gap is real.


The mixed-index panel is the one that tells the story: SHIFT off returns ten
Polish products, SHIFT on returns Spanish ones you could buy in Barcelona. But
the contrast is not equally strong for every query, and picking a bad one on
stage makes a working method look broken.

The failure mode to avoid is querying with a string that is a near-verbatim
copy of a Polish record. The self-match then scores ~0.87 while the best
Spanish product scores ~0.50, no amount of offset closes that, and both panels
show ten Polish rows. That is not SHIFT failing -- it is lexical identity
winning -- but it is indistinguishable from failure on a projector.

So this ranks queries by how many Spanish products enter the top 10 when the
offset is applied, and reports the gain over alpha=0. Queries whose own record
dominates are exactly the ones that score badly here and get filtered out.
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
DIM = 256
K = 10
TOP = 15


def truncate(v, d):
    out = np.atleast_2d(v)[:, :d].astype(np.float32)
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.clip(n, 1e-12, None)


def tgt_in_top(q, D, langs, atcs, gold, target, k=K):
    sims = D @ q
    top = np.argpartition(-sims, k)[:k]
    top = top[np.argsort(-sims[top])]
    tg = [t for t in top if langs[t] == target]
    return len(tg), sum(1 for t in tg if atcs[t] == gold), top[0]


def main():
    np.seterr(all="ignore")
    pairs, docs = load_corpus()
    dv = np.load(OUT / "arctic_docs.npy")
    qv = np.load(OUT / "arctic_queries.npy")
    off_d = {l: np.load(OUT / f"arctic_offset_doc_{l}.npy") for l in SHIFTED}
    off_q = {l: np.load(OUT / f"arctic_offset_query_{l}.npy") for l in SHIFTED}

    langs = np.array([d["lang"] for d in docs])
    atcs = np.array([d["atc"] for d in docs])

    D0 = truncate(dv, DIM)
    D1 = truncate(shift_all(dv, off_d, ALPHA, langs), DIM)
    Q0 = truncate(qv, DIM)
    Q1 = truncate(shift_all(qv, off_q, ALPHA, langs), DIM)

    target = PIVOT
    # Only ATC codes that actually have a product in the target language can
    # show the effect at all.
    has_tgt = {atcs[i] for i in np.where(langs == target)[0]}

    out = {}
    for src in SHIFTED:
        rows = []
        for i in np.where(langs == src)[0]:
            if atcs[i] not in has_tgt:
                continue
            n0, _, _ = tgt_in_top(Q0[i], D0, langs, atcs, atcs[i], target)
            n1, right, _ = tgt_in_top(Q1[i], D1, langs, atcs, atcs[i], target)
            rows.append((n1 - n0, n1, right, i))

        rows.sort(key=lambda r: (-r[0], -r[2]))
        print(f"\nalpha={ALPHA} dim={DIM}  {src}->{target}, "
              f"ranked by target products gained in top {K}\n")
        print(f"{'gain':>5} {'tgt@10':>7} {'correct':>8}  query")
        seen = set()
        for gain, n1, right, i in rows:
            atc = atcs[i]
            if atc in seen:      # one example per molecule, or the list is
                continue         # ten spellings of the same paracetamol
            seen.add(atc)
            print(f"{gain:>5} {n1:>7} {right:>8}  [{atc}] {docs[i]['text'][:54]}")
            if len(seen) >= TOP:
                break

        out[src] = [{"atc": atcs[i], "text": docs[i]["text"], "lang": src,
                     "gain": int(g), "tgt_at_10": int(n1), "correct": int(r)}
                    for g, n1, r, i in rows[:200]]

    (OUT / "demo_queries.json").write_text(json.dumps(out, ensure_ascii=False,
                                                      indent=1))


if __name__ == "__main__":
    main()
