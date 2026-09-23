"""Build the collection behind the retrieval-method comparison panel.

This is a teaching artifact, not the production path -- but only one half of it
turned out to be the teaching it was built for.

BM25 loses exactly as predicted: 0.230 at top-1 against dense+SHIFT's 0.827, and
the reason is visible rather than statistical. On 160 of 300 Polish queries the
query shares *no term at all* with any Spanish document, so the result is empty
rather than badly ranked. Apap and ANTIDOL are the same molecule and have no
characters in common. That is the vocabulary gap, and a panel that runs it live
makes it land in a way a slide cannot.

SPLADE does not lose. It scores 0.770, and RRF fusion of dense with SPLADE beats
dense alone at every cut: 0.837 / 0.920 / 0.970 against 0.827 / 0.913 / 0.953.
Term expansion reaches across the vocabulary gap that exact matching cannot.
This was not the expected result -- Splade_PP is English-trained and these are
Spanish and Polish registry strings -- and the panel should report it rather than
bury it, because an unexpected measurement is worth more airtime than a
confirmed one.

Nothing sparse ships, and after this that is a cost decision rather than an
accuracy one: +0.010 at top-1 does not pay for a second index and a 500 MB model
in a demo. Worth saying in those terms, because "we measured it and it won, and
we still said no" is a more honest engineering story than "sparse lost".

Kept in its own collection rather than added to med_shift for two reasons: the
main collection carries twenty named dense vectors per point and rebuilding it
is slow, and the comparison holds SHIFT fixed at the measured best alpha, so it
does not need the slider's twenty variants at all -- one dense vector is enough.

Two caveats that have to stay attached to any number the panel shows:

  bm25    fastembed ships Snowball stemmers for 18 languages and Polish is not
          one of them. Spanish and Dutch are. So the Polish query side runs
          unstemmed, which understates BM25 for Polish specifically.
  splade  Splade_PP_en_v1 is English-trained. Its term expansion on Spanish and
          Polish registry strings is out of domain. There is no multilingual
          SPLADE in fastembed, so this is the honest ceiling for the demo, and
          the panel should say so rather than let SPLADE look beaten fairly.

Stop the server before running this. Qdrant in local mode takes an exclusive
lock on the storage directory, so the build and serve_med.py cannot both hold
artifacts/qdrant_med_sparse at once.
"""
import json
import sys
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "artifacts" / "med"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shift_med import SHIFTED, load_corpus, shift_all  # noqa: E402

COLLECTION = "med_sparse"
ALPHA = 0.75
DIM = 256

# (name, fastembed model, kwargs). Spanish stemming on the document side
# because Spanish is the pivot and the target language for every query.
SPARSE = (
    ("bm25", "Qdrant/bm25", {"language": "spanish"}),
    ("splade", "prithivida/Splade_PP_en_v1", {}),
)


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
    texts = [d["text"] for d in docs]

    dense = truncate(shift_all(dv, off_d, ALPHA, langs), DIM)

    from fastembed import SparseTextEmbedding
    cols = {}
    for name, model, kw in SPARSE:
        print(f"embedding {len(texts)} docs with {name} ...", flush=True)
        m = SparseTextEmbedding(model, **kw)
        cols[name] = list(m.embed(texts, batch_size=32))

    client = QdrantClient(path=str(ROOT / "artifacts" / "qdrant_med_sparse"))
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        COLLECTION,
        vectors_config={"dense": models.VectorParams(
            size=DIM, distance=models.Distance.COSINE)},
        # IDF is a property of the corpus, not of the text, so Qdrant computes
        # it server-side from what is actually indexed. fastembed only sends
        # term frequencies.
        sparse_vectors_config={
            "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF),
            "splade": models.SparseVectorParams(),
        },
    )

    points = []
    for i, x in enumerate(docs):
        points.append(models.PointStruct(
            id=i,
            vector={
                "dense": dense[i].tolist(),
                **{name: models.SparseVector(
                    indices=cols[name][i].indices.tolist(),
                    values=cols[name][i].values.tolist())
                   for name in cols},
            },
            payload={"atc": x["atc"], "lang": x["lang"], "brand": x["brand"],
                     "text": x["text"], "nreg": x.get("nreg"),
                     "rvg": x.get("rvg", "")},
        ))
    client.upload_points(COLLECTION, points, batch_size=128, wait=True)

    # The term ids that actually occur in each language's documents. This is
    # what lets the panel say "3 of your 7 words exist anywhere in the Spanish
    # index" instead of only "BM25 returned nothing" -- the same fact, but
    # counted, and still true when the count is small rather than zero.
    vocab = {}
    for lang in sorted(set(langs.tolist())):
        ids = set()
        for i in np.where(langs == lang)[0]:
            ids.update(cols["bm25"][i].indices.tolist())
        vocab[lang] = sorted(ids)

    (OUT / "sparse_index.json").write_text(json.dumps(
        {"collection": COLLECTION, "alpha": ALPHA, "dim": DIM,
         "n": len(points), "methods": [n for n, _, _ in SPARSE],
         "bm25_vocab": vocab}, indent=1))
    print(f"  {len(points)} points in {COLLECTION}")
    for lang, ids in vocab.items():
        print(f"  bm25 vocabulary {lang}: {len(ids)} terms")


if __name__ == "__main__":
    main()
