"""How well does the real camera path work, end to end, on real packaging?

Every other number in this project is measured on clean registry text. That
flatters the demo: on stage the query is whatever Apple Vision managed to read
off a carton under bad lighting, which is a different and much worse string.

The cached CIMA photos make this measurable without a camera. Each photo is
tagged with the nregistro it belongs to, and that maps to a known ATC, so the
gold answer is known for every image. Two photo kinds are scored separately
because they are not the same task:

  materialas    the packaging shot -- a box with the brand printed on it. This
                is what someone actually points a phone at, so it is the number
                that matters.
  formafarmac   a loose tablet on a white background. There is usually no text
                at all, so this is the floor: what happens when OCR finds
                nothing useful.

Reported against the same retrieval the server runs, so the comparison against
the clean-text baseline (top-1 0.870 on registry strings) is like for like.
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
from shift_med import (MODEL, PIVOT, QUERY_PREFIX, SHIFTED,  # noqa: E402
                       load_corpus, shift_all)

ALPHA = 0.75
DIM = 256
K = 10
LIMIT = 250          # photos per kind; the whole set takes far longer than it informs


def truncate(v, d):
    out = np.atleast_2d(v)[:, :d].astype(np.float32)
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.clip(n, 1e-12, None)


def main():
    np.seterr(all="ignore")
    pairs, docs = load_corpus()
    dv = np.load(OUT / "arctic_docs.npy")
    off_d = {l: np.load(OUT / f"arctic_offset_doc_{l}.npy") for l in SHIFTED}

    langs = np.array([d["lang"] for d in docs])
    atcs = np.array([d["atc"] for d in docs])
    D = truncate(shift_all(dv, off_d, ALPHA, langs), DIM)
    es = np.where(langs == PIVOT)[0]
    D_es = D[es]

    man = json.loads((WEB / "med_img" / "manifest.json").read_text())
    by_kind = defaultdict(list)
    for m in man.values():
        by_kind[m["kind"]].append(m)

    from sentence_transformers import SentenceTransformer
    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(MODEL, device=dev)

    print(f"alpha={ALPHA} dim={DIM}  scoring against {len(es)} Spanish docs\n")
    print(f"{'kind':<13} {'n':>5} {'no_text':>8} {'top1':>7} {'top5':>7} {'top10':>7}")
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
            print(f"{kind:<13} {len(items):>5} {blank:>8}  no readable text")
            continue

        q = model.encode([QUERY_PREFIX + t for t in texts],
                         batch_size=32, normalize_embeddings=True)
        q = np.asarray(q, dtype=np.float32)
        # The OCR text is Spanish, off a Spanish box, and Spanish is the pivot,
        # so there is no offset to subtract. Shifting here would be wrong.
        q = truncate(q, DIM)

        hits = {1: 0, 5: 0, 10: 0}
        for i in range(len(texts)):
            sims = D_es @ q[i]
            top = np.argpartition(-sims, K)[:K]
            top = top[np.argsort(-sims[top])]
            got = [atcs[es[t]] for t in top]
            for n in hits:
                hits[n] += int(golds[i] in got[:n])
        n = len(texts)
        # blank photos count against the total: a camera that reads nothing is
        # a failure of the pipeline, not a query to be excluded from it.
        tot = n + blank
        print(f"{kind:<13} {tot:>5} {blank:>8} "
              f"{hits[1] / tot:>7.3f} {hits[5] / tot:>7.3f} {hits[10] / tot:>7.3f}")


if __name__ == "__main__":
    main()
