"""Can the country of origin be read off the box, instead of picked from a menu?

The demo's language picker is a stage hazard: SHIFT applies a per-language
offset, so the wrong button does not degrade the result, it destroys it. This
measures whether langid.py can remove the button.

Two tests, because they are not equally hard and only one of them is the real
task:

  registry   Held-out corpus strings, all three languages. The detector is
             fitted on the remaining documents, so the test items contribute
             nothing to their own vote. This is the easy case and mostly
             confirms the method is not broken: registry text always carries a
             dosage form, and dosage forms are language stamps.

  carton     Apple Vision OCR on the cached CIMA packaging photos, which is the
             exact string the camera path would produce. Spanish only -- the
             photo set is Spanish, and there are no Polish or Dutch carton
             images here. So this measures one thing honestly and leaves the
             other two unmeasured rather than guessing at them, which is worth
             saying out loud before any number from it goes on a slide.

Reported alongside a do-nothing baseline (always guess the majority language),
because with three unbalanced registries an accuracy number means little on its
own.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "artifacts" / "med"
WEB = ROOT / "web"
sys.path.insert(0, str(Path(__file__).resolve().parent))

import vision_ocr  # noqa: E402
from langid import MARGIN_CONFIDENT, LangID  # noqa: E402
from shift_med import MODEL, QUERY_PREFIX, load_corpus  # noqa: E402

DIM = 256
EMB_TOPK = 5
N_PER_LANG = 400
PHOTO_LIMIT = 150
SEED = 0


def truncate(v, d):
    out = np.atleast_2d(v)[:, :d].astype(np.float32)
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.clip(n, 1e-12, None)


def emb_scores(qvec, D, masks, langs, k=EMB_TOPK):
    """Top-k mean cosine into each language's documents, unshifted."""
    out = {}
    for l in langs:
        s = D[masks[l]] @ qvec
        kk = min(k, s.shape[0])
        out[l] = float(np.mean(-np.partition(-s, kk - 1)[:kk]))
    return out


def abstain(gold, pred, margins):
    """What the confidence threshold buys: accuracy on the queries it keeps.

    Swept rather than asserted, because the whole value of the threshold is
    that it is the one number in this module that was fitted to data.
    """
    m = np.asarray(margins)
    ok = np.array([g == p for g, p in zip(gold, pred)])
    print(f"  {'-- abstain sweep':<22}")
    for th in (0.0, 0.1, MARGIN_CONFIDENT, 0.3):
        keep = m >= th
        tag = "  <- MARGIN_CONFIDENT" if th == MARGIN_CONFIDENT else ""
        print(f"     margin>={th:<4} keeps {keep.mean():>6.1%}  "
              f"acc {ok[keep].mean() if keep.any() else float('nan'):.4f}{tag}")


def report(name, gold, pred, langs):
    n = len(gold)
    acc = sum(g == p for g, p in zip(gold, pred)) / n
    major = Counter(gold).most_common(1)[0][0]
    base = sum(g == major for g in gold) / n
    print(f"  {name:<22} {acc:>7.3f}   (majority baseline {base:.3f}, n={n})")
    return acc


def main():
    np.seterr(all="ignore")
    pairs, docs = load_corpus()
    dv = np.load(OUT / "arctic_docs.npy")
    qv = np.load(OUT / "arctic_queries.npy")
    langs_arr = np.array([d["lang"] for d in docs])
    L = sorted(set(langs_arr.tolist()))
    masks = {l: np.where(langs_arr == l)[0] for l in L}

    # alpha=0 on purpose: the languages are only separable while SHIFT has not
    # been applied, which is what makes this work as a detector at all.
    D = truncate(dv, DIM)
    Q = truncate(qv, DIM)

    rng = np.random.default_rng(SEED)
    sel = {l: rng.choice(masks[l], size=min(N_PER_LANG, len(masks[l])),
                         replace=False) for l in L}
    held = set(np.concatenate([sel[l] for l in L]).tolist())

    # Fitted on everything the test does not contain. Holding out matters more
    # than it looks: a document votes for its own language with its own rare
    # brand name, which would make any accuracy here meaningless.
    fit = LangID([d for i, d in enumerate(docs) if i not in held])

    print(f"dim={DIM} emb_topk={EMB_TOPK}\n\nregistry text (held out)")
    gold, p_lex, p_emb, margins = [], [], [], []
    for l in L:
        for i in sel[l]:
            lex = fit(docs[i]["text"])
            es = emb_scores(Q[i], D, {m: masks[m][masks[m] != i] for m in L}, L)
            gold.append(l)
            p_lex.append(lex["lang"])
            p_emb.append(max(es, key=es.get))
            margins.append(lex["margin"])
    report("lexical", gold, p_lex, L)
    report("embedding (alpha=0)", gold, p_emb, L)
    abstain(gold, p_lex, margins)

    # ---------------------------------------------------------------- carton
    man = json.loads((WEB / "med_img" / "manifest.json").read_text())
    shots = [m for m in man.values() if m["kind"] == "materialas"][:PHOTO_LIMIT]
    if not shots:
        print("\nno packaging photos cached; skipping carton test")
        return

    print(f"\ncarton OCR ({len(shots)} spanish packaging photos)")
    full = LangID(docs)
    # Vision on 150 photos costs minutes and the strings never change, so the
    # OCR output is cached. Threshold sweeps re-read this instead of the images.
    cache_path = OUT / "ocr_carton.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    texts, blank = [], 0
    for m in shots:
        if m["file"] in cache:
            t = cache[m["file"]]
        else:
            try:
                blocks = vision_ocr.read(
                    (WEB / "med_img" / m["file"]).read_bytes())
            except Exception:                                   # noqa: BLE001
                blocks = []
            t = vision_ocr.query_text(blocks) if blocks else ""
            cache[m["file"]] = t
        if not t.strip():
            blank += 1
            continue
        texts.append(t)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False))

    from sentence_transformers import SentenceTransformer
    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(MODEL, device=dev)
    qe = truncate(np.asarray(model.encode([QUERY_PREFIX + t for t in texts],
                                          batch_size=32,
                                          normalize_embeddings=True),
                             dtype=np.float32), DIM)

    g, pl_, pe_, margins = [], [], [], []
    cov = []
    by_pred = defaultdict(int)
    for i, t in enumerate(texts):
        lex = full(t)
        es = emb_scores(qe[i], D, masks, L)
        cov.append(lex["coverage"])
        margins.append(lex["margin"])
        g.append("es")
        pl_.append(lex["lang"])
        pe_.append(max(es, key=es.get))
        by_pred[lex["lang"]] += 1
    report("lexical", g, pl_, L)
    report("embedding (alpha=0)", g, pe_, L)
    abstain(g, pl_, margins)
    print(f"  {'mean token coverage':<22} {np.mean(cov):>7.3f}   "
          f"(lexical found no known token in "
          f"{sum(1 for c in cov if c == 0)}/{len(texts)})")
    print(f"  {'unreadable photos':<22} {blank:>7}   (excluded; no text to judge)")
    print(f"  lexical predictions: {dict(by_pred)}")


if __name__ == "__main__":
    main()
