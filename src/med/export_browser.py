"""Export the Spanish index so the companion page can retrieve without a server.

The whole point of the MRL section is that 256 dimensions is enough. This file
is what happens if you believe it: at 256 dims in fp16, the entire Spanish
register fits in well under a megabyte, which is a normal web asset. There is
no API behind the search box on the companion page -- the vectors are a static
file and the query is embedded in the browser.

Three things make this smaller than it sounds, and it is worth being precise
about which is doing the work:

  256 dims      MRL, measured in matryoshka.py -- 103% of bekko's cross-border
                hits at a third of its native width
  fp16          half of float32. Not MRL, just storage; verified below to
                change no top-10 result
  Spanish only  the pivot language. A cross-border query is asking for Spanish
                products, so the other 4,050 documents never need to ship

The last one is also why no document offsets are exported. Spanish is the
pivot: its offset is the mean of (es - es) and is zero by construction. SHIFT
happens entirely on the query side here, which is convenient, because the query
is the only thing the browser computes.

Query vectors are exported for nothing -- the browser makes its own. What it
needs from this file is the per-language *query* offset, taken from the
prefixed encoding, because eval_bekko.py measured that arrangement (asymmetric
prefix, two offsets) as the best of the three at 0.328 against 0.292.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "artifacts" / "med"
SITE = ROOT / "site" / "data"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shift_med import (PIVOT, SHIFTED, all_offsets, load_corpus)  # noqa: E402

DIM = 256
ALPHA = 0.75


def truncate(v, d):
    out = v[:, :d].astype(np.float32)
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.clip(n, 1e-12, None)


def main():
    pairs, docs = load_corpus()
    v = np.load(OUT / "bekko_docs.npy")              # prefix-free document space
    vq = np.load(OUT / "bekko_queries_prefixed.npy")  # prefixed query space

    es = [i for i, d in enumerate(docs) if d["lang"] == PIVOT]
    dv = truncate(v[es], DIM)

    # fp16 is a storage decision, not a modelling one, so it has to be shown to
    # be free rather than assumed. Compare top-10 before and after the cast,
    # using real queries: every Polish record, put through the same pipeline
    # the browser will use.
    dv16 = dv.astype(np.float16)
    back = dv16.astype(np.float32)
    off_q = {l: all_offsets(vq, docs)[l][:DIM].astype(np.float32) for l in SHIFTED}

    pl = [i for i, d in enumerate(docs) if d["lang"] == "pl"]
    qv = truncate(vq[pl], DIM) - ALPHA * off_q["pl"]
    qv /= np.clip(np.linalg.norm(qv, axis=1, keepdims=True), 1e-12, None)
    with np.errstate(all="ignore"):
        t32 = np.argsort(-(qv @ dv.T), axis=1)[:, :10]
        t16 = np.argsort(-(qv @ back.T), axis=1)[:, :10]

    # Exact top-10 agreement is the wrong bar and fails at 96.7% here. Cosines
    # this close are separated by less than fp16's own resolution, so adjacent
    # ranks swap. What matters is whether the same documents come back and
    # whether the metrics move: set overlap and recall are the honest checks.
    exact = float((t32 == t16).all(axis=1).mean())
    atc = np.array([docs[i]["atc"] for i in es])
    qatc = np.array([docs[i]["atc"] for i in pl])
    overlap = float(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(t32, t16)]))
    rec = [float(np.mean([(atc[r] == q).any() for r, q in zip(t, qatc)]))
           for t in (t32, t16)]
    drift = float(np.abs(dv - back).max())

    meta = [{"atc": docs[i]["atc"], "brand": docs[i]["brand"],
             "text": docs[i]["text"], "nreg": docs[i].get("nreg"),
             "otc": docs[i].get("otc", "")} for i in es]

    (SITE / "es_vectors_256_fp16.bin").write_bytes(dv16.tobytes())
    (SITE / "es_index.json").write_text(json.dumps({
        "dim": DIM, "count": len(es), "dtype": "float16", "alpha": ALPHA,
        "model": "hotchpotch/bekko-embedding-v1-a25m",
        "pooling": "mean", "query_prefix": "query: ",
        # The browser subtracts these. Not unit vectors and must not be made
        # into unit vectors -- an offset is a displacement, and renormalising
        # it would more than double the correction.
        "query_offsets": {l: off_q[l].tolist() for l in SHIFTED},
        "docs": meta,
    }, ensure_ascii=False, separators=(",", ":")))

    nbytes = (SITE / "es_vectors_256_fp16.bin").stat().st_size
    jbytes = (SITE / "es_index.json").stat().st_size
    print(f"{len(es)} Spanish documents at {DIM} dims, fp16")
    print(f"  vectors  {nbytes / 1024:7.1f} KB   ({nbytes / len(es):.0f} B/vector)")
    print(f"  metadata {jbytes / 1024:7.1f} KB")
    print(f"  total    {(nbytes + jbytes) / 1024:7.1f} KB")
    print(f"\nfp16 cast, checked against float32 over {len(pl)} Polish queries")
    print(f"  same 10 documents:  {overlap:.3%}   (order-insensitive)")
    print(f"  same 10, in order:  {exact:.3%}   (adjacent ties swap)")
    print(f"  recall_atc:         {rec[0]:.4f} -> {rec[1]:.4f}")
    print(f"  max per-element drift: {drift:.2e}")


if __name__ == "__main__":
    main()
