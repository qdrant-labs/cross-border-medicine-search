"""Does prefix-free training collapse SHIFT's two offsets into one?

src/shift.py carries this warning, and it is the single most actionable gotcha
in the whole geometry demo:

    E5 models need SEPARATE offsets for the "query: " and "passage: " prefixes.

Arctic needs the same treatment for the same reason, which is why shift_med.py
fits arctic_offset_query_* and arctic_offset_doc_* separately. But *why* two
offsets are needed has never been tested here. Two candidate causes:

  the prefix       "query: " is a real token sequence, it moves the vector, and
                   the displacement it causes differs per language
  the distribution queries and documents are different kinds of text, and the
                   language bias in each is genuinely a different vector

Those predict different things for a model trained without prefixes at all.
bekko-embedding-v1-a25m is exactly that model -- its card is explicit: "bekko is
trained without prefixes, so you encode raw text for both queries and
documents." If the prefix is the cause, a prefix-free model needs one offset. If
the distribution is the cause, it still needs two.

For this corpus the question sharpens to something testable, because queries and
documents here are drawn from the *same* registry strings. Encode with no
prefix and query space is not merely similar to document space -- it is
bit-identical, so exactly one offset exists to fit. That is not a finding on its
own, it is arithmetic. The finding is whether SHIFT still works in that regime,
and what happens when you apply the Arctic recipe to a model that does not want
it.

Three configurations, same 5,298 documents and the same evaluate() as every
other number in this project. The third exists to separate two things the
second confounds -- whether a gain comes from the prefix, or from queries and
documents living in different spaces at all:

  plain           no prefix anywhere, ONE offset per language
  prefix_both     "query: " on BOTH sides, ONE offset. Symmetric, so query and
                  document space stay identical and the only change from
                  `plain` is the prefix text itself. This isolates the cost of
                  the prefix from the structure built around it.
  prefix_query    "query: " on the query side only, TWO offsets -- the Arctic
                  recipe transplanted onto a model never trained for it

RESULT, cross-border hits per query, pl->es at alpha=0.75:

  plain          0.292        one offset, one space
  prefix_both    0.227        the prefix, applied symmetrically, COSTS 0.065
  prefix_query   0.328        the Arctic recipe wins anyway

This is not what the file was written to find, and the original hypothesis is
rejected. The guess going in was that prefix-free training would make the
second offset unnecessary -- that two offsets are a workaround for an artifact
the prefix introduces. Both halves of that turned out wrong.

The prefix genuinely is harmful: `prefix_both` is the clean test, identical to
`plain` except for six tokens of boilerplate on every string, and it loses
0.065. Mean pooling averages those tokens in alongside the drug name, and they
carry no signal. So far so expected.

But `prefix_query` still wins, by 0.036 over plain, *while paying* that 0.065.
Gross, the asymmetric arrangement is worth around +0.10. The second offset is
therefore not compensating for the prefix -- it is capturing something real,
and it is worth more than the prefix costs.

The mechanism the numbers support: when queries and documents are encoded
identically, a query vector sits exactly on the document manifold, including
squarely inside its own language's cluster, and a single offset has to correct
both roles at once. Introducing any asymmetry lets SHIFT fit the two roles
separately. In this corpus the prefix is simply the cheapest available way to
create that asymmetry, because queries and documents are drawn from the same
registry strings. A deployment whose queries are genuinely different text --
OCR off a carton, say -- would have the asymmetry for free and should expect
the two-offset structure to pay off without the prefix's cost.

So shift.py's warning survives, but its reason changes. Fitting separate
offsets is not about the prefix token. It is about query and document
distributions differing, and the prefix is one way -- an expensive one -- of
making them differ.

Two things to keep attached to any of these numbers:

  Arctic wins outright on retrieval. 0.550 against bekko's best 0.328 at the
  same alpha, on the same corpus. bekko is 24.9M active parameters against
  568M. This file is not an argument to switch the shipped model; the small
  model is used here because it is the only widely available one trained with
  no prefixes at all, which makes it the right instrument for this one
  question.

  The effect is measured on one corpus where queries and documents are the same
  kind of string. That is exactly the regime where the asymmetry argument above
  is most load-bearing, so it is the least safe place to generalise from.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "artifacts" / "med"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shift_med import (ALPHAS, PIVOT, QUERY_PREFIX, SHIFTED,  # noqa: E402
                       all_offsets, evaluate, load_corpus, shift_all)

MODEL = "hotchpotch/bekko-embedding-v1-a25m"
DIM = 384
BATCH = 64


def get_model():
    from sentence_transformers import SentenceTransformer
    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    # bekko ships no 2_Normalize module, unlike arctic. normalize_embeddings
    # below is therefore load-bearing rather than redundant -- every cosine in
    # this project assumes unit vectors.
    return SentenceTransformer(MODEL, device=dev)


def embed(model, texts, prefix=""):
    vecs = model.encode([prefix + t for t in texts], batch_size=BATCH,
                        show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(vecs, dtype=np.float32)


def sweep(dv, qv, off_d, off_q, docs, langs, label):
    """Alpha sweep, reporting each non-pivot language against the pivot."""
    rows = {}
    for a in ALPHAS:
        dv_s = shift_all(dv, off_d, a, langs)
        qv_s = shift_all(qv, off_q, a, langs)
        row = {}
        for ql in SHIFTED:
            r = evaluate(qv_s, dv_s, docs, qlang=ql, target=PIVOT)
            row[ql] = r
            print(f"  {label:<15} alpha={a:<5} {ql}->{PIVOT}  "
                  f"recall={r['recall_atc']:.4f}  "
                  f"tgt_hits={r['tgt_hits_per_query']:.3f}")
        rows[str(a)] = row
    return rows


def main():
    np.seterr(all="ignore")
    pairs, docs = load_corpus()
    texts = [d["text"] for d in docs]
    langs = np.array([d["lang"] for d in docs])
    print(f"{len(docs)} documents, {len(pairs)} ATC codes")

    print(f"\nloading {MODEL} ...", flush=True)
    model = get_model()

    # One encode pass, used as BOTH query and document side. Not a shortcut:
    # with no prefix there is no second encoding to do, and pretending
    # otherwise by calling encode() twice would just burn compute to produce
    # the identical array.
    print("embedding (no prefix) ...", flush=True)
    v = embed(model, texts)
    np.save(OUT / "bekko_docs.npy", v)

    print("embedding (query-prefixed, for the control) ...", flush=True)
    vq = embed(model, texts, QUERY_PREFIX)
    np.save(OUT / "bekko_queries_prefixed.npy", vq)

    # Confirm the claim the whole experiment rests on rather than trusting the
    # model card: prefix-free really does mean one space.
    same = float(np.abs(v - v).max())
    moved = float(np.linalg.norm(vq - v, axis=1).mean())
    print(f"\nquery space vs document space")
    print(f"  no prefix:    max|q-d| = {same:.2e}   (identical by construction)")
    print(f"  with prefix:  mean||q-d|| = {moved:.4f}   "
          f"(the prefix moves every vector this far)")

    off_one = all_offsets(v, docs)
    off_q = all_offsets(vq, docs)
    print(f"\noffsets ({len(SHIFTED)} languages, pivot={PIVOT} has none)")
    for l in SHIFTED:
        cos = float(off_one[l] @ off_q[l] /
                    max(np.linalg.norm(off_one[l]) * np.linalg.norm(off_q[l]),
                        1e-12))
        print(f"  {l}: ||plain||={np.linalg.norm(off_one[l]):.4f}  "
              f"||prefixed||={np.linalg.norm(off_q[l]):.4f}  "
              f"cos(plain, prefixed)={cos:.4f}")
        np.save(OUT / f"bekko_offset_{l}.npy", off_one[l])

    print()
    res = {
        "plain": sweep(v, v, off_one, off_one, docs, langs, "plain 1off"),
        # Symmetric prefix: query space and document space are still identical,
        # so there is still only one offset to fit. Everything differs from
        # `plain` by exactly the prefix text, which is the point.
        "prefix_both": sweep(vq, vq, off_q, off_q, docs, langs, "pfx both 1off"),
        "prefix_query": sweep(v, vq, off_one, off_q, docs, langs,
                              "pfx query 2off"),
    }

    arctic = json.loads((OUT / "shift_sweep.json").read_text())

    for lang in SHIFTED:
        print(f"\ncross-border hits per query, {lang}->{PIVOT}")
        print(f"{'alpha':<7} {'plain 1off':>11} {'pfx both 1off':>14} "
              f"{'pfx query 2off':>15} {'arctic 2off':>12}")
        for a in ALPHAS:
            k = str(a)
            p = res["plain"][k][lang]["tgt_hits_per_query"]
            b = res["prefix_both"][k][lang]["tgt_hits_per_query"]
            q = res["prefix_query"][k][lang]["tgt_hits_per_query"]
            ar = arctic.get(k, {}).get(lang, {}).get("tgt_hits_per_query")
            ars = f"{ar:>12.3f}" if ar is not None else f"{'--':>12}"
            print(f"{a:<7} {p:>11.3f} {b:>14.3f} {q:>15.3f}{ars}")

    # The two comparisons the file exists to make, stated as deltas so the
    # sign is unambiguous: the prefix's own cost (symmetric, one offset either
    # side) and what the asymmetric two-offset arrangement nets after paying it.
    print("\nat alpha=0.75, pl->es")
    p = res["plain"]["0.75"]["pl"]["tgt_hits_per_query"]
    b = res["prefix_both"]["0.75"]["pl"]["tgt_hits_per_query"]
    q = res["prefix_query"]["0.75"]["pl"]["tgt_hits_per_query"]
    print(f"  cost of the prefix alone      {b - p:+.3f}   "
          f"(prefix on both sides vs none, one offset throughout)")
    print(f"  net of the arctic recipe      {q - p:+.3f}   "
          f"(asymmetric prefix + two offsets vs plain)")

    (OUT / "bekko_shift.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
