"""Truncate the embeddings and watch what happens -- with and without MRL.

Slicing a vector to its first N dimensions is the same line of code whichever
model produced it. The difference is whether the model was trained to tolerate
it. Arctic v2.0 was, via Matryoshka Representation Learning, and Snowflake
document the trained range as 256 dimensions. E5 was not, and serves here as
the control: same operation, same corpus, no MRL objective behind it.

The reason to care is the camera. A phone scanning medicine boxes wants the
index resident on device, and 1024 float32 dimensions is 4KB per vector. At 256
that is 1KB, and at int4 it is 128 bytes -- which is the difference between
shipping the index and calling a server.

Truncating below 256 is included deliberately. MRL is not magic past the range
it was trained for, and the curve should show that.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "artifacts" / "med"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shift_med import (SHIFTED, all_offsets, embed, evaluate,  # noqa: E402
                       load_corpus, shift_all)

DIMS = (1024, 512, 256, 128, 64)
E5 = "intfloat/multilingual-e5-small"     # 384-dim, no MRL -- the control
E5_DIMS = (384, 256, 128, 64)


def truncate(v, d):
    """First d dimensions, renormalised. This is all MRL asks of the consumer."""
    out = v[:, :d].astype(np.float32)
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.clip(n, 1e-12, None)


def truncate_offset(off, d):
    """Slice only -- no renormalisation.

    The offset is a displacement, not a direction: its magnitude (~0.44) is
    what makes alpha=1 mean "remove one language's worth of shift". Pushing it
    through truncate() would rescale it to unit norm, applying a correction
    more than twice too large and wrecking retrieval.
    """
    return off[:d].astype(np.float32)


def sweep(dv, qv, docs, dims, langs=None, off_d=None, off_q=None, alpha=1.0,
          qlang="pl"):
    rows = {}
    for d in dims:
        dvt, qvt = truncate(dv, d), truncate(qv, d)
        if langs is not None:
            dvt = shift_all(dvt, {l: truncate_offset(o, d)
                                  for l, o in off_d.items()}, alpha, langs)
            qvt = shift_all(qvt, {l: truncate_offset(o, d)
                                  for l, o in off_q.items()}, alpha, langs)
        r = evaluate(qvt, dvt, docs, qlang=qlang)
        r["bytes_per_vector"] = d * 4
        rows[str(d)] = r
        print(f"  dim={d:<5} recall_atc={r['recall_atc']:.4f}  "
              f"tgt_hits={r['tgt_hits_per_query']:.3f}  {d * 4}B/vec")
    return rows


def main():
    pairs, docs = load_corpus()
    langs = np.array([d["lang"] for d in docs])

    dv = np.load(OUT / "arctic_docs.npy")
    qv = np.load(OUT / "arctic_queries.npy")
    off_d = {l: np.load(OUT / f"arctic_offset_doc_{l}.npy") for l in SHIFTED}
    off_q = {l: np.load(OUT / f"arctic_offset_query_{l}.npy") for l in SHIFTED}

    print("arctic-embed-l-v2.0  (MRL-trained, documented range 256)")
    arctic = sweep(dv, qv, docs, DIMS, langs, off_d, off_q)

    print(f"\n{E5}  (no MRL -- control)")
    texts = [d["text"] for d in docs]
    from sentence_transformers import SentenceTransformer
    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    e5 = SentenceTransformer(E5, device=dev)
    e5_d = embed(e5, [f"passage: {t}" for t in texts])
    e5_q = embed(e5, [f"query: {t}" for t in texts])
    np.save(OUT / "e5_docs.npy", e5_d)
    np.save(OUT / "e5_queries.npy", e5_q)

    # The control only means anything if E5 is doing the same job as Arctic:
    # same SHIFT correction, estimated the same way, then truncated the same
    # way. Without its own offsets E5 would score ~0 Spanish hits at every
    # dimension and we would be measuring the absence of SHIFT, not the
    # absence of MRL.
    e5_off_d = all_offsets(e5_d, docs)
    e5_off_q = all_offsets(e5_q, docs)
    control = sweep(e5_d, e5_q, docs, E5_DIMS, langs, e5_off_d, e5_off_q)

    res = {"arctic_mrl": arctic, "e5_control": control}
    (OUT / "matryoshka.json").write_text(json.dumps(res, indent=1))

    # recall_atc is reported but it is not the metric to judge truncation by:
    # it sits at ~0.99 for every model at every dimension, because finding
    # *some* record of the right molecule in a 5,298-document index is easy
    # and stays easy. tgt_hits_per_query is the number under load -- it asks
    # whether the cross-border hits survived, and it is what degrades first.
    print("\nretention at 256 dimensions")
    for label, r, dims in (("arctic (MRL) ", arctic, DIMS),
                           ("e5   (no MRL)", control, E5_DIMS)):
        f, t = r[str(dims[0])], r["256"]
        print(f"  {label}  recall {t['recall_atc'] / f['recall_atc']:6.1%}   "
              f"cross-border hits {t['tgt_hits_per_query'] / f['tgt_hits_per_query']:6.1%}")


if __name__ == "__main__":
    main()
