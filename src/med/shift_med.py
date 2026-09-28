"""SHIFT offsets for the bilingual medicine corpus, on arctic-embed-l-v2.0.

The problem this exists to solve is visible without any maths: embed 2,600
Polish product records and 1,300 Spanish ones into one index, query it with the
text off a Polish box, and the top ten come back almost entirely Polish. The
embedding has encoded "this is Polish" alongside "this is paracetamol", and the
language signal is strong enough to crowd out the clinical one. That is the
language-spillover failure, and on a cross-border lookup it is fatal: the whole
point is to surface the *Spanish* products.

SHIFT estimates a single per-language offset vector and subtracts it:

    offset[lang] = mean( embed(lang record) - embed(pivot record) )
    indexed      = embed(doc) - alpha * offset[doc lang]

Estimating the offset needs aligned pairs. Literal translations of these records
do not exist, so pairs are formed through the ATC code: the Polish and Spanish
records under N02BE01 both denote paracetamol, even though neither is a
translation of the other. That is weaker supervision than a parallel corpus, but
it is the supervision a real deployment would actually have, and the offset only
needs to capture the direction the language shifts the embedding -- not the
meaning of any individual pair.

Arctic takes a "query: " prefix on queries and nothing on documents, so query
space and document space are offset separately, for the same reason E5 needs
separate offsets per prefix.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "artifacts" / "med"

MODEL = "Snowflake/snowflake-arctic-embed-l-v2.0"
QUERY_PREFIX = "query: "
PIVOT = "es"
# Every non-pivot language needs its own offset. The pivot has none by
# definition: its offset would be the mean of (es - es), which is zero.
LANGS = ("pl", "es", "nl")
SHIFTED = tuple(l for l in LANGS if l != PIVOT)
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
BATCH = 64


def load_corpus():
    """Flatten pairs.json into per-language document lists tagged with ATC."""
    pairs = json.loads((OUT / "pairs.json").read_text())
    docs = []
    for atc, v in pairs.items():
        for p in v.get("pl", []):
            bits = [p["brand"], p.get("inn", ""), p.get("form", "")]
            docs.append({"atc": atc, "lang": "pl", "brand": p["brand"],
                         "text": ", ".join(b for b in bits if b)})
        for p in v.get("es", []):
            # The INN has to be in here. Half the Spanish register is branded
            # -- "ANTIDOL 1 G COMPRIMIDOS" -- and a brand name carries no
            # signal about the molecule in any language. Without it the only
            # Spanish records a Polish query could reach were the generics
            # that happen to be named after their own active ingredient.
            bits = [p["brand"], p.get("inn", ""), p.get("form", "")]
            docs.append({"atc": atc, "lang": "es", "brand": p["brand"],
                         "otc": p.get("otc", ""), "nreg": p.get("nreg"),
                         "text": ", ".join(b for b in bits if b)})
        for p in v.get("nl", []):
            bits = [p["brand"], p.get("inn", ""), p.get("form", "")]
            docs.append({"atc": atc, "lang": "nl", "brand": p["brand"],
                         "rvg": p.get("rvg", ""),
                         "text": ", ".join(b for b in bits if b)})
    return pairs, docs


def get_model():
    from sentence_transformers import SentenceTransformer
    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    return SentenceTransformer(MODEL, device=dev)


def embed(model, texts, prefix=""):
    vecs = model.encode([prefix + t for t in texts], batch_size=BATCH,
                        show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(vecs, dtype=np.float32)


def aligned_pairs(docs, lang):
    """One (lang, pivot) index pair per ATC code that has both.

    Each non-pivot language is aligned to the pivot independently rather than
    to each other. That keeps every offset an estimate of the same quantity --
    displacement from Spanish -- so a Dutch query and a Polish query land in
    the same corrected space without a Dutch-to-Polish offset ever existing.
    """
    first = {}
    for i, d in enumerate(docs):
        first.setdefault((d["atc"], d["lang"]), i)
    out = []
    for atc in {d["atc"] for d in docs}:
        a, b = first.get((atc, lang)), first.get((atc, PIVOT))
        if a is not None and b is not None:
            out.append((a, b))
    return out


def offsets_from(vecs, pairs_idx):
    """Mean displacement from pivot to the other language."""
    diff = np.stack([vecs[a] - vecs[b] for a, b in pairs_idx])
    return diff.mean(axis=0)


def all_offsets(vecs, docs):
    """{lang: offset} for every non-pivot language."""
    return {l: offsets_from(vecs, aligned_pairs(docs, l)) for l in SHIFTED}


def apply_shift(vecs, offset, alpha):
    out = vecs - alpha * offset
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.clip(n, 1e-12, None)


def shift_all(vecs, offsets, alpha, langs):
    """Apply each language's own offset in place; pivot rows are left alone."""
    out = vecs.copy()
    for lang, off in offsets.items():
        m = langs == lang
        if m.any():
            out[m] = apply_shift(vecs[m], off, alpha)
    return out


def evaluate(qv, dv, docs, k=10, qlang="pl", target=PIVOT):
    """Query with one language's records; measure hits in the target language.

    recall_atc  -- did we find the right molecule at all
    tgt_hits    -- of the correct-molecule hits, how many were in the target
                   language. This is the spillover number: a spillover-bound
                   index keeps answering in the query's own language and scores
                   near zero here, however good its recall_atc looks.

    ndcg20 / tlr20 are the SHIFT write-up's own two metrics, so the claim there
    ("nDCG@20 rises 0.633 to 0.737, and TLR@20 jumps the most") can be checked
    against this corpus in the units it was made in rather than by analogy.
    Relevance is binary and comes from the ATC code, which is the only gold
    label here -- there is no graded judgement to discount, so nDCG reduces to
    rank-weighting a set of equally correct answers. Reported at 20 while the
    two metrics above stay at k=10: every number already published from this
    file was produced at 10 and silently moving it would rewrite them all.
    """
    # NumPy on Apple's Accelerate BLAS reports spurious "divide by zero" and
    # "overflow" from this matmul. The inputs are unit-norm with absmax ~0.18
    # and the outputs land in [0.003, 0.87]; float64 reproduces these metrics
    # exactly. The BLAS kernel is leaving FP status flags set, nothing more.
    #
    # Accelerate is also not bit-reproducible across processes -- its threaded
    # reduction order varies -- so a query sitting on a near-tie at the rank-10
    # boundary can fall either side of it. Measured on e5-large: 666 or 667
    # target hits over 2,568 queries, run to run. That is +-0.0004 on
    # tgt_hits_per_query, so treat the last published digit as noise rather
    # than chasing a one-hit difference between two runs.
    with np.errstate(all="ignore"):
        return _eval_loop(qv, dv, docs, k, qlang, target)


# The write-up's cutoff. Kept separate from `k` so the two are free to differ.
K20 = 20
# Discounts for ranks 1..20: 1/log2(rank+1), the standard DCG weighting.
_DISC = 1.0 / np.log2(np.arange(2, K20 + 2))


def _eval_loop(qv, dv, docs, k, qlang, target):
    langs = np.array([d["lang"] for d in docs])
    atcs = np.array([d["atc"] for d in docs])

    # How many relevant documents exist at all, per ATC code. Needed for both
    # the ideal DCG and the recall denominator, and constant across queries, so
    # counted once here rather than per query.
    n_atc, n_atc_tgt = {}, {}
    for a, l in zip(atcs, langs):
        n_atc[a] = n_atc.get(a, 0) + 1
        if l == target:
            n_atc_tgt[a] = n_atc_tgt.get(a, 0) + 1

    # argpartition needs kth < len. The real corpus is 5,298 documents so this
    # only binds in tests, but an IndexError there is a worse failure than a
    # shorter list.
    kk = min(K20, len(docs) - 1)
    disc = _DISC[:kk]

    hits, tgt_hits, n = 0, 0, 0
    ndcg_sum, tlr_sum, ceil_sum, n_tlr = 0.0, 0.0, 0.0, 0
    for qi in range(len(qv)):
        if docs[qi]["lang"] != qlang:
            continue
        atc = docs[qi]["atc"]
        sims = dv @ qv[qi]
        sims[qi] = -1e9                      # never retrieve the query itself
        top = np.argpartition(-sims, k)[:k]
        top = top[np.argsort(-sims[top])]
        same = atcs[top] == atc
        hits += int(same.any())
        tgt_hits += int(((langs[top] == target) & same).sum())
        n += 1

        # --- the write-up's two metrics, at 20 ---
        t20 = np.argpartition(-sims, kk)[:kk]
        t20 = t20[np.argsort(-sims[t20])]
        rel = atcs[t20] == atc
        # Relevant docs excluding the query itself, which was scored out above.
        R = n_atc[atc] - 1
        if R > 0:
            idcg = disc[:min(R, kk)].sum()
            ndcg_sum += float(disc[rel].sum() / idcg)

        R_t = n_atc_tgt.get(atc, 0) - (1 if docs[qi]["lang"] == target else 0)
        if R_t > 0:
            # Plain recall: found in the top 20, over all that exist. Not
            # capped at 20, so an ATC with 40 Spanish products cannot score 1
            # however good the ranking is -- `tlr20_ceiling` below is the mean
            # best-possible score and is what tlr20 should be read against.
            tlr_sum += float(((langs[t20] == target) & rel).sum()) / R_t
            ceil_sum += min(R_t, kk) / R_t
            n_tlr += 1

    return {"n_queries": n,
            "recall_atc": round(hits / max(n, 1), 4),
            "tgt_hits_per_query": round(tgt_hits / max(n, 1), 3),
            "ndcg20": round(ndcg_sum / max(n, 1), 4),
            "tlr20": round(tlr_sum / max(n_tlr, 1), 4),
            "tlr20_ceiling": round(ceil_sum / max(n_tlr, 1), 4)}


def main():
    pairs, docs = load_corpus()
    texts = [d["text"] for d in docs]
    counts = {l: sum(1 for d in docs if d["lang"] == l) for l in LANGS}
    print(f"{len(docs)} documents  ({counts}), {len(pairs)} ATC codes")

    model = get_model()
    print(f"embedding with {MODEL} ...")
    dv = embed(model, texts)                        # document space
    qv = embed(model, texts, QUERY_PREFIX)          # query space
    np.save(OUT / "arctic_docs.npy", dv)
    np.save(OUT / "arctic_queries.npy", qv)

    off_d = all_offsets(dv, docs)
    off_q = all_offsets(qv, docs)
    for lang in SHIFTED:
        np.save(OUT / f"arctic_offset_doc_{lang}.npy", off_d[lang])
        np.save(OUT / f"arctic_offset_query_{lang}.npy", off_q[lang])
        print(f"  {lang}->{PIVOT}: {len(aligned_pairs(docs, lang))} aligned pairs, "
              f"norm doc={np.linalg.norm(off_d[lang]):.4f} "
              f"query={np.linalg.norm(off_q[lang]):.4f}")
    print()

    langs = np.array([d["lang"] for d in docs])
    sweep = {}
    for a in ALPHAS:
        dv_s = shift_all(dv, off_d, a, langs)
        qv_s = shift_all(qv, off_q, a, langs)
        row = {}
        for ql in SHIFTED:
            r = evaluate(qv_s, dv_s, docs, qlang=ql, target=PIVOT)
            row[ql] = r
            print(f"alpha={a:<5} {ql}->{PIVOT}  recall_atc={r['recall_atc']:.4f}  "
                  f"target hits/query={r['tgt_hits_per_query']:.3f}")
        sweep[str(a)] = row

    (OUT / "shift_sweep.json").write_text(json.dumps(sweep, indent=1))
    (OUT / "docs.json").write_text(json.dumps(docs, ensure_ascii=False))


if __name__ == "__main__":
    main()
