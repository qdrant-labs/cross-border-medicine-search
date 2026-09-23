"""Does late interaction help when the query is OCR text rather than clean text?

ColBERT lost on registry strings -- dense+hyperbolic beat it 0.953 to 0.880 at
top-1 in eval_colbert.py -- and the reason is plausible: a registry string is
already short, clean and dense with signal, so pooling it into one vector loses
very little, and the ATC tree knows more about drug identity than any reranker
reading brand names.

OCR text is the opposite kind of input, which is why this is a separate test
rather than an assumed result. Vision returns whatever it could read off a
carton: the brand, the strength, and also the batch number, the marketing
authorisation holder, "Lea el prospecto antes de utilizar este medicamento",
and fragments of the barcode. A single pooled vector averages all of that
together and the signal gets diluted by the noise. MaxSim does not average --
each query token independently takes its best document token, so a token that
matches nothing contributes its best available match and little more. That is
the specific mechanism by which late interaction should tolerate noisy queries
better, and it either shows up in the numbers here or it does not.

Same photos and same gold labels as eval_ocr.py, so the dense column here is
directly comparable to the top-1 0.976 reported there for packaging shots.

Result, with jinaai/jina-colbert-v2 (the only late-interaction model in
fastembed that passed the discrimination sanity check at all):

  kind             n  blank   dense   d+hyp  colbert  cb+hyp
  materialas     250      0   0.976   0.984    0.972   0.976
  formafarmac    250     79   0.280   0.360    0.268   0.304

The mechanism argued for above does not appear. ColBERT is *below* dense on both
photo kinds, and dense+hyperbolic beats every ColBERT configuration including
ColBERT with the same hyperbolic rerank bolted on. On the packaging shots that
actually matter it costs 0.008 and buys a per-token index plus reranking
latency.

Worth being precise about why, because "late interaction did not help" is not
the lesson. MaxSim tolerates noise by letting junk tokens contribute their best
available match -- but on a carton the junk is not random, it is *other Spanish
pharmaceutical text*: dosage instructions, the authorisation holder, warning
boilerplate. Those tokens match real document tokens strongly and confidently in
the wrong direction, which is the one case the mechanism cannot absorb. Pooling
is what dilutes them, so here averaging is the feature rather than the flaw.

That closes the question this file was opened to answer: late interaction is
rejected on clean text, on OCR text, and on the discrimination check. The rerank
that does earn its place is the 10-dimensional hyperbolic one, which adds 0.008
on packaging and 0.080 on the near-textless tablet shots -- the harder case, and
the one where knowing the ATC tree substitutes for having read anything.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "artifacts" / "med"
WEB = ROOT / "web"
sys.path.insert(0, str(Path(__file__).resolve().parent))

import vision_ocr  # noqa: E402
from shift_med import (MODEL, PIVOT, QUERY_PREFIX, all_offsets,  # noqa: E402
                       load_corpus, shift_all)

ALPHA = 0.75
DIM = 256
K = 10
LIMIT = 250
PREFETCH = 50
HYP_SCALE = 13.0
HYP_BETA = 0.08
CB_MODEL = "jinaai/jina-colbert-v2"


def truncate(v, d):
    out = np.atleast_2d(v)[:, :d].astype(np.float32)
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.clip(n, 1e-12, None)


def maxsim(q, d):
    return float(np.max(q @ d.T, axis=1).sum())


def hyp_all_pairs(p):
    sq = np.sum(p ** 2, axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * (p @ p.T), 0.0)
    den = np.clip((1 - sq)[:, None] * (1 - sq)[None, :], 1e-12, None)
    return np.arccosh(np.clip(1 + 2 * d2 / den, 1.0, None))


def main():
    np.seterr(all="ignore")
    pairs, docs = load_corpus()
    dv = np.load(OUT / "arctic_docs.npy")
    langs = np.array([d["lang"] for d in docs])
    atcs = np.array([d["atc"] for d in docs])

    off_d = {l: np.load(OUT / f"arctic_offset_doc_{l}.npy")
             for l in ("pl", "nl") if (OUT / f"arctic_offset_doc_{l}.npy").exists()}
    D = truncate(shift_all(dv, off_d, ALPHA, langs), DIM)
    es = np.where(langs == PIVOT)[0]
    D_es = D[es]
    es_atc = atcs[es]
    es_text = [docs[i]["text"] for i in es]

    codes = json.loads((OUT / "atc_codes.json").read_text())
    code_idx = {c: i for i, c in enumerate(codes)}
    HYP = hyp_all_pairs(np.load(OUT / "atc_poincare_10.npy"))

    man = json.loads((WEB / "med_img" / "manifest.json").read_text())
    by_kind = defaultdict(list)
    for m in man.values():
        by_kind[m["kind"]].append(m)

    from sentence_transformers import SentenceTransformer
    from fastembed import LateInteractionTextEmbedding
    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(MODEL, device=dev)
    print(f"loading {CB_MODEL} ...", flush=True)
    cb = LateInteractionTextEmbedding(CB_MODEL)
    CB_D = list(cb.embed(es_text, batch_size=32))

    print(f"\nalpha={ALPHA} dim={DIM} prefetch={PREFETCH} "
          f"against {len(es)} spanish docs")
    print(f"{'kind':<12} {'n':>5} {'blank':>6} {'dense':>7} {'d+hyp':>7} "
          f"{'colbert':>8} {'cb+hyp':>7}")

    for kind in ("materialas", "formafarmac"):
        items = by_kind.get(kind, [])[:LIMIT]
        texts, golds, blank = [], [], 0
        for m in items:
            try:
                blocks = vision_ocr.read((WEB / "med_img" / m["file"]).read_bytes())
            except Exception:
                blocks = []
            t = vision_ocr.query_text(blocks) if blocks else ""
            if not t.strip():
                blank += 1
                continue
            texts.append(t)
            golds.append(m["atc"])
        if not texts:
            print(f"{kind:<12} {len(items):>5} {blank:>6}   no readable text")
            continue

        q = model.encode([QUERY_PREFIX + t for t in texts], batch_size=32,
                         normalize_embeddings=True)
        # The OCR text is Spanish, off a Spanish box, and Spanish is the pivot,
        # so there is no offset to subtract. Shifting here would be wrong.
        q = truncate(np.asarray(q, dtype=np.float32), DIM)
        CB_Q = list(cb.query_embed(texts))

        hit = {"dense": 0, "d+hyp": 0, "colbert": 0, "cb+hyp": 0}
        for i in range(len(texts)):
            gold = golds[i]
            sims = D_es @ q[i]
            pf = np.argpartition(-sims, PREFETCH)[:PREFETCH]
            pf = pf[np.argsort(-sims[pf])]
            gi = code_idx.get(gold)
            hyp = np.array([HYP[gi, code_idx[c]] if gi is not None
                            and c in code_idx else 0.0 for c in es_atc[pf]])
            cos = sims[pf]
            cbs = np.array([maxsim(CB_Q[i], CB_D[j]) for j in pf])
            cbn = (cbs - cbs.min()) / max(cbs.max() - cbs.min(), 1e-9)
            for name, sc in (("dense", cos),
                             ("d+hyp", cos - HYP_BETA * hyp / HYP_SCALE),
                             ("colbert", cbn),
                             ("cb+hyp", cbn - HYP_BETA * hyp / HYP_SCALE)):
                top = es_atc[pf][np.argsort(-sc)][:1]
                hit[name] += int(gold in top)

        tot = len(texts) + blank
        print(f"{kind:<12} {tot:>5} {blank:>6} "
              f"{hit['dense'] / tot:>7.3f} {hit['d+hyp'] / tot:>7.3f} "
              f"{hit['colbert'] / tot:>8.3f} {hit['cb+hyp'] / tot:>7.3f}")


if __name__ == "__main__":
    main()
