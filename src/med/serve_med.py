"""Cross-border medicine lookup: point a camera at a box, get the local equivalent.

Runs on its own port and its own Qdrant collection so the geometry demo stays
untouched as the fallback. Nothing here imports from src/server.py.

Three ideas are on screen at once, and each maps to one control:

  SHIFT (alpha)     subtract the language offset so a Polish query stops
                    preferring Polish documents. This is the whole demo: at
                    alpha=0 the top ten come back Polish and useless, at
                    alpha=1 they come back Spanish and actionable.
  Matryoshka (dim)  truncate the vector. 1024 -> 256 is free, 256 -> 128
                    falls off a cliff, because 256 is where Snowflake stopped
                    training the MRL objective.
  Hyperbolic        the ATC tree embedded in the Poincare ball. Radius tracks
                    how specific a code is, which is what lets the UI answer
                    "same molecule" and "same chemical subgroup" differently.

The collection carries one named vector per (alpha, dim) stop rather than
re-embedding on each request: the point of the demo is to move a slider and
watch the result list change, and that has to be instant.

Safety: results are joined on ATC and never on brand name, and the UI is given
the full ATC path so the claim is auditable on screen. The wording is "same
active ingredient, same ATC class" -- never "equivalent", which is a
therapeutic judgement no vector search is entitled to make.
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from qdrant_client import QdrantClient, models

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "artifacts" / "med"
WEB = ROOT / "web"
sys.path.insert(0, str(Path(__file__).resolve().parent))

import vision_ocr  # noqa: E402
from langid import MARGIN_CONFIDENT, LangID  # noqa: E402
from shift_med import (LANGS, MODEL, PIVOT, QUERY_PREFIX,  # noqa: E402
                       SHIFTED, apply_shift, load_corpus, shift_all)

COLLECTION = "med_shift"
ALPHAS = (0.0, 0.5, 0.75, 1.0)
DIMS = (1024, 512, 256, 128, 64)

# The sweep peaks at 0.75, not at 1.0 -- removing slightly less than one full
# language's worth of offset beats removing all of it. So "SHIFT on" in the
# side-by-side is the measured best setting rather than the round number.
BEST_ALPHA = 0.75

# Hyperbolic rerank. HYP_SCALE is the widest distance in this tree (two leaves
# under different level-1 letters), so d/HYP_SCALE lands in [0, 1] and beta is
# readable as "how much cosine a full cross-branch jump is allowed to cost".
# beta is swept in eval_rerank.py, not guessed.
HYP_SCALE = 13.0
HYP_OVERFETCH = 5

# Swept in eval_rerank.py over all 2,568 Polish queries. 0.12 scores a hair
# higher at top-1 (0.917 vs 0.912) but breaks 58 previously-correct queries
# against 38, and is worse at top-3. On a medicine lookup an answer that used
# to be right and now is not costs more than one that was never right, and the
# UI shows several results, so the gentler weight wins.
HYP_BETA = 0.08

# Prefetch stage. 64 dims is 256 bytes per vector against 4,096 at full width,
# and PREFETCH_FACTOR is how much slack the cheap stage gets before the
# expensive one rescores. Swept in eval_prefetch.py against the single-stage
# result, because the failure mode is silent: anything the 64-dim shortlist
# misses is gone, and the reranker cannot tell you it is missing.
#   factor  4   recall 0.938, agrees with full-width top-10 on 69%
#   factor  8   recall 0.970, 81%
#   factor 16   recall 0.988, 91%   <- full-width recall is 0.990
# 16 is where recall stops improving, at 160 candidates out of 1,248 Spanish
# documents: an eighth of the corpus rescored at full width instead of all of it.
PREFETCH_DIM = 64
PREFETCH_FACTOR = 16

# Retrieval-method comparison panel. Built by build_sparse.py into its own
# collection, and entirely optional -- if it has not been built the panel is
# hidden and the rest of the demo is unaffected. Nothing sparse is on the
# production path: eval_sparse.py measured dense+SHIFT at 0.827 top-1 against
# BM25 at 0.243, and fusing them makes dense worse, not better. The panel
# exists to show the vocabulary gap, not to retrieve with it.
SPARSE_COLLECTION = "med_sparse"
SPARSE_ALPHA = 0.75
SPARSE_DIM = 256
RRF_K = 60

app = FastAPI(title="Cross-border medicine lookup")

# Local origins only. The UI is normally served by this same app, but it has to
# be servable from a separate static server too -- a browser sandbox may be
# allowed to reach one directory and not another, and on the day the slides and
# the API may not sit on the same port.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
S = {}


def vname(alpha, dim):
    return f"a{int(alpha * 100):03d}_d{dim}"


def truncate(v, d):
    out = np.atleast_2d(v)[:, :d].astype(np.float32)
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.clip(n, 1e-12, None)


def build(client, dv, off_d, docs):
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        COLLECTION,
        vectors_config={
            vname(a, d): models.VectorParams(size=d,
                                             distance=models.Distance.COSINE)
            for a in ALPHAS for d in DIMS
        },
    )
    langs = np.array([x["lang"] for x in docs])

    # Shift at full width first, then truncate. Doing it the other way round
    # would need a separately estimated offset per dimension, and the whole
    # Matryoshka claim is that the leading coordinates of the *same* vector
    # still work -- so the vector has to be built once and sliced.
    vecs = {}
    for a in ALPHAS:
        shifted = shift_all(dv, off_d, a, langs)
        for d in DIMS:
            vecs[vname(a, d)] = truncate(shifted, d)

    # Photo filenames are deliberately NOT stored here. They are resolved from
    # the manifest at query time, so re-running the photo harvest never
    # invalidates the index -- and the index can be built before the images
    # have finished downloading.
    points = []
    for i, x in enumerate(docs):
        points.append(models.PointStruct(
            id=i,
            vector={k: vecs[k][i].tolist() for k in vecs},
            payload={
                "atc": x["atc"], "lang": x["lang"], "brand": x["brand"],
                "text": x["text"], "otc": x.get("otc", ""),
                "nreg": x.get("nreg"), "rvg": x.get("rvg", ""),
            },
        ))
    client.upload_points(COLLECTION, points, batch_size=256, wait=True)
    return len(points)


@app.on_event("startup")
def startup():
    pairs, docs = load_corpus()
    S["docs"] = docs
    S["tree"] = json.loads((OUT / "atc_tree.json").read_text())
    # One offset per non-pivot language, each estimated against Spanish. The
    # pivot has no file because its offset is zero by construction.
    S["off_q"] = {l: np.load(OUT / f"arctic_offset_query_{l}.npy")
                  for l in SHIFTED}
    reload_photos()

    dv = np.load(OUT / "arctic_docs.npy")
    off_d = {l: np.load(OUT / f"arctic_offset_doc_{l}.npy") for l in SHIFTED}

    client = QdrantClient(path=str(ROOT / "artifacts" / "qdrant_med"))
    S["client"] = client
    if not client.collection_exists(COLLECTION):
        print("building medicine collection ...")
        n = build(client, dv, off_d, docs)
        print(f"  {n} points")

    from sentence_transformers import SentenceTransformer
    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    S["model"] = SentenceTransformer(MODEL, device=dev)

    # 2D coordinates for the geometry panel, keyed by ATC code.
    codes = json.loads((OUT / "atc_codes.json").read_text())
    S["codes"] = codes
    S["code_idx"] = {c: i for i, c in enumerate(codes)}
    S["poincare2"] = np.load(OUT / "atc_poincare_2.npy")
    S["euclid2"] = np.load(OUT / "atc_euclidean_2.npy")
    # The 2D embedding is for drawing; reranking uses the 10D one, which is
    # what the MAP numbers were measured on (0.8314 vs 0.7369 Euclidean).
    S["poincare10"] = np.load(OUT / "atc_poincare_10.npy")

    # Fitted on the same corpus the search runs over, so adding a registry
    # teaches the detector that language for free. It is a token-frequency
    # count over 5k short strings -- cheap enough to do at every startup rather
    # than carry as another artifact that can drift out of sync with docs.json.
    S["langid"] = LangID(docs)

    load_sparse()
    print(f"ready: {len(docs)} docs, {len(codes)} ATC nodes")


def load_sparse():
    """Open the comparison collection, if build_sparse.py has made one.

    Loaded eagerly rather than on first request because SPLADE is a 500 MB
    download and the first click on this panel happens on stage. Every failure
    here is swallowed: the comparison is a talking point, and it must never be
    the reason the medicine lookup will not start.
    """
    path = ROOT / "artifacts" / "qdrant_med_sparse"
    S["sparse"] = None
    if not path.exists():
        return
    try:
        from fastembed import SparseTextEmbedding
        client = QdrantClient(path=str(path))
        if not client.collection_exists(SPARSE_COLLECTION):
            return
        meta = json.loads((OUT / "sparse_index.json").read_text())
        S["sparse"] = {
            "client": client,
            "bm25": SparseTextEmbedding("Qdrant/bm25", language="spanish"),
            "splade": SparseTextEmbedding("prithivida/Splade_PP_en_v1"),
            "vocab": {l: set(v) for l, v in meta["bm25_vocab"].items()},
        }
        print("  retrieval comparison: bm25 + splade ready")
    except Exception as e:                                      # noqa: BLE001
        print(f"  retrieval comparison unavailable: {e}")


def reload_photos():
    """Manifest -> {nreg: file} and {atc: [file]}, re-readable while serving."""
    path = WEB / "med_img" / "manifest.json"
    man = json.loads(path.read_text()) if path.exists() else {}
    S["photos"] = man
    by_nreg, by_atc = {}, {}
    for m in man.values():
        # The packaging shot is what someone points a camera at; the
        # dosage-form shot is a loose pill on a white background and is only
        # a fallback.
        rank = 0 if m["kind"] == "materialas" else 1
        cur = by_nreg.get(m["nreg"])
        if cur is None or rank < cur[0]:
            by_nreg[m["nreg"]] = (rank, m["file"])
        by_atc.setdefault(m["atc"], []).append(m["file"])
    S["photo_of_nreg"] = {k: v[1] for k, v in by_nreg.items()}
    S["photo_of_atc"] = by_atc
    return len(man)


def photo_for(payload):
    f = S["photo_of_nreg"].get(payload.get("nreg"))
    if f:
        return f
    # Polish records have no photo of their own -- the Polish registry does not
    # publish packaging shots. Falling back to any Spanish box under the same
    # ATC is honest here: it is a picture of the molecule's local packaging,
    # which is exactly what the user is being sent to the shelf to find.
    same = S["photo_of_atc"].get(payload.get("atc"))
    return same[0] if same else None


def atc_path(code):
    """The auditable chain: N -> N02 -> N02B -> N02BE -> N02BE01."""
    tree, out = S["tree"], []
    cuts = (1, 3, 4, 5, 7)
    for c in cuts:
        if len(code) >= c:
            node = tree.get(code[:c])
            out.append({"code": code[:c], "level": cuts.index(c) + 1,
                        "name": (node or {}).get("name", "")})
    return out


def hyp_dist(a, b):
    """Poincare distance between two ATC codes. Unknown code -> None."""
    i, j = S["code_idx"].get(a), S["code_idx"].get(b)
    if i is None or j is None:
        return None
    p = S["poincare10"]
    u, v = p[i], p[j]
    num = float(np.sum((u - v) ** 2))
    den = (1 - float(np.sum(u ** 2))) * (1 - float(np.sum(v ** 2)))
    return float(np.arccosh(1 + 2 * num / max(den, 1e-12)))


def anchor_atc(text, dim, qlang):
    """Which ATC is the product the query names? Answered in its own language.

    This has to be resolved against the language the query is written in, not
    against the candidates -- reranking Spanish hits by their distance to the
    best Spanish hit would just re-affirm whatever cosine already picked.
    Identifying a box in its own national registry is a separate piece of
    evidence, and it is the easy half of the problem: recall on the monolingual
    side is 0.995.

    Searching the wrong registry quietly poisons the rerank. A Spanish query
    anchored against Polish records returned ketoprofen for an ibuprofen box --
    same ATC subgroup, so the distances looked plausible and nothing failed
    loudly.
    """
    hits = search(text, 0.0, dim, 1, qlang, rerank=False, qlang=qlang)
    return hits[0]["atc"] if hits else None


def rerank_hyp(hits, anchor, beta=HYP_BETA):
    """Blend cosine with tree proximity to the anchor code.

    The failure this fixes: for a paracetamol box, cosine ranks ANTIDOL DUAL
    (N02BE51, paracetamol *plus* ibuprofen) above ANTIDOL (N02BE01, plain
    paracetamol). The two texts are near-identical strings, so no embedding
    separates them -- but they are different codes, and handing someone a
    combination product when they asked for a single molecule is a real error.

    A hard filter to anchor == candidate would also fix that case and would be
    wrong in general: plenty of ATC codes have no Spanish product at all, and
    the honest answer there is the nearest sibling, not an empty list. The
    hyperbolic distance is graded, so the ranking degrades by tree steps --
    same molecule, then same chemical subgroup, then same therapeutic class --
    which is the order a pharmacist would walk the shelf in.
    """
    if not anchor:
        return hits
    for h in hits:
        d = hyp_dist(anchor, h["atc"])
        h["hyp_dist"] = None if d is None else round(d, 3)
        h["cosine"] = h["score"]
        # Unknown code keeps its cosine rank rather than being pushed to the
        # bottom: a missing tree node is our gap, not the product's fault.
        if d is not None:
            h["score"] = round(h["score"] - beta * (d / HYP_SCALE), 4)
    hits.sort(key=lambda h: -h["score"])
    return hits


def embed_query(text):
    v = S["model"].encode([QUERY_PREFIX + text], normalize_embeddings=True)
    return np.asarray(v, dtype=np.float32)[0]


def search(text, alpha, dim, k, target, rerank=False, beta=HYP_BETA,
           qlang="pl", prefetch=False):
    q = embed_query(text)
    # SHIFT is a per-language correction and each language has its own offset,
    # estimated as "that language minus Spanish". Spanish is the pivot, so a
    # Spanish query is already in the shared space and has no entry here --
    # subtracting someone else's offset would push it away from every document.
    # Query space and document space have separate offsets because Arctic
    # prefixes queries and not documents.
    if qlang in S["off_q"]:
        q = apply_shift(q[None, :], S["off_q"][qlang], alpha)[0]
    q = truncate(q, dim)[0]

    flt = None
    if target:
        flt = models.Filter(must=[models.FieldCondition(
            key="lang", match=models.MatchValue(value=target))])
    # Overfetch before reranking, or the rerank can only reorder a list cosine
    # already chose -- the right product is often just outside the top k.
    limit = k * HYP_OVERFETCH if rerank else k

    if prefetch:
        # Two-stage retrieval, done inside Qdrant rather than in Python: the
        # 64-dim truncation picks a wide candidate set cheaply, and only those
        # candidates are rescored at full width. This is the Matryoshka panel
        # turned into an actual query plan -- the small vector is not a
        # separate model or a separate index, it is the first 64 coordinates
        # of the same vector, which is the entire point of MRL.
        qs = truncate(embed_query(text) if qlang not in S["off_q"] else
                      apply_shift(embed_query(text)[None, :],
                                  S["off_q"][qlang], alpha)[0], PREFETCH_DIM)[0]
        res = S["client"].query_points(
            COLLECTION, query=q.tolist(), using=vname(alpha, dim),
            prefetch=models.Prefetch(query=qs.tolist(),
                                     using=vname(alpha, PREFETCH_DIM),
                                     limit=limit * PREFETCH_FACTOR,
                                     filter=flt),
            limit=limit, query_filter=flt, with_payload=True).points
    else:
        res = S["client"].query_points(
            COLLECTION, query=q.tolist(), using=vname(alpha, dim),
            limit=limit, query_filter=flt, with_payload=True).points

    hits = [{"score": round(p.score, 4), **p.payload,
             "photo": photo_for(p.payload)} for p in res]
    if rerank:
        hits = rerank_hyp(hits, anchor_atc(text, dim, qlang), beta)[:k]
    return hits


@app.get("/api/med/meta")
def meta():
    docs = S["docs"]
    return {
        "n_docs": len(docs),
        "langs": list(LANGS),
        "pivot": PIVOT,
        "n_by_lang": {l: sum(1 for d in docs if d["lang"] == l) for l in LANGS},
        "n_atc": len({d["atc"] for d in docs}),
        "n_photos": len(S["photos"]),
        "alphas": list(ALPHAS), "dims": list(DIMS),
        "model": MODEL,
        "prefetch": {"dim": PREFETCH_DIM, "factor": PREFETCH_FACTOR},
        "sparse": bool(S.get("sparse")),
        "langid": {"margin": MARGIN_CONFIDENT},
        "sweep": json.loads((OUT / "shift_sweep.json").read_text()),
        "matryoshka": json.loads((OUT / "matryoshka.json").read_text()),
        "geometry": json.loads((OUT / "atc_results.json").read_text()),
    }


@app.get("/api/med/detect")
def api_detect(q: str):
    """Which registry wrote this string? Same detector the camera path uses.

    Exposed separately so the typed query gets the same treatment as the
    photographed one -- otherwise the text box would still need the picker and
    the two paths would disagree about what language means.
    """
    return S["langid"](q) if S.get("langid") else {"lang": None}


@app.get("/api/med/suggestions")
def api_suggestions(n: int = 8, qlang: str = "pl"):
    """Queries in one source language where the SHIFT contrast is strongest.

    Ranked offline by pick_demo_queries.py. Having these one click away matters
    on stage: the obvious thing to type is a famous brand like "Apap", and that
    is close to the worst possible choice -- it matches a Polish record almost
    verbatim, the self-match outscores every Spanish product, and both panels
    come back all-Polish. The method is fine; the query was bad.

    Per language, because the trap is language-specific. Dutch shares enough
    vocabulary with Spanish that many Dutch queries work with no correction at
    all, so a hand-picked Dutch example is likely to show two identical panels.
    """
    path = OUT / "demo_queries.json"
    if not path.exists():
        return {"queries": []}
    data = json.loads(path.read_text())
    rows = data.get(qlang, []) if isinstance(data, dict) else data
    out, seen = [], set()
    for r in rows:
        if r["atc"] in seen:
            continue
        seen.add(r["atc"])
        out.append({"text": r["text"], "atc": r["atc"], "gain": r["gain"]})
        if len(out) >= n:
            break
    return {"queries": out, "qlang": qlang}


@app.get("/api/med/search")
def api_search(q: str, alpha: float = 1.0, dim: int = 1024,
               k: int = 12, target: str = "es", rerank: bool = False,
               qlang: str = "pl", prefetch: bool = False):
    hits = search(q, alpha, dim, k, target or None, rerank, qlang=qlang,
                  prefetch=prefetch)
    for h in hits:
        h["path"] = atc_path(h["atc"])
    return {"query": q, "alpha": alpha, "dim": dim, "rerank": rerank,
            "qlang": qlang, "prefetch": prefetch, "hits": hits}


@app.get("/api/med/compare")
def api_compare(q: str, dim: int = 1024, k: int = 10, target: str = "",
                gold: str = "", rerank: bool = False, qlang: str = "pl",
                prefetch: bool = False):
    """Same query, SHIFT off versus on. This is the side-by-side money shot.

    Two views, and they answer different questions:

      target=""    the mixed index. Shows language spillover directly -- a
                   Polish query pulls back Polish records and buries the
                   Spanish ones, which is the failure the article is about.
      target=es    the actual product question, "what do I buy in Barcelona".
                   Filtering to Spanish is not cheating: the user already
                   told us which country they are standing in.

    'gold' is the expected ATC. When the caller knows it, the UI can mark
    which hits are genuinely the same molecule rather than just plausible.

    'rerank' and 'prefetch' apply to both sides, so toggling either never
    changes what the off/on pair is comparing.
    """
    tgt = target or None
    off = search(q, 0.0, dim, k, tgt, rerank, qlang=qlang, prefetch=prefetch)
    on = search(q, BEST_ALPHA, dim, k, tgt, rerank, qlang=qlang,
                prefetch=prefetch)

    def wrap(alpha, hits):
        for h in hits:
            h["path"] = atc_path(h["atc"])
            h["match"] = bool(gold) and h["atc"] == gold
        # Share of results in the language being searched *for*. With three
        # languages in one index this is no longer "how Spanish is the list" --
        # it is the spillover number for whichever country was asked about.
        want = tgt or PIVOT
        return {
            "alpha": alpha, "hits": hits,
            "tgt_share": round(sum(1 for x in hits if x["lang"] == want)
                               / max(len(hits), 1), 3),
            "top_atc": hits[0]["atc"] if hits else None,
        }

    return {"query": q, "dim": dim, "target": target, "gold": gold,
            "rerank": rerank, "qlang": qlang, "prefetch": prefetch,
            "off": wrap(0.0, off), "on": wrap(BEST_ALPHA, on)}


@app.get("/api/med/retrieval_compare")
def api_retrieval_compare(q: str, k: int = 6, target: str = "es",
                          gold: str = "", qlang: str = "pl"):
    """One query, four retrievers: dense+SHIFT, BM25, SPLADE, and RRF fusion.

    The point of the panel is the vocabulary gap. A Polish box says "Apap" and
    the Spanish product says "ANTIDOL"; they are the same paracetamol and share
    no characters, so an inverted index has nothing to match on and returns
    either noise or nothing at all. matched_terms reports that directly -- how
    many of the query's terms occur anywhere in the target-language corpus --
    and when it is zero the sparse column is empty on screen, which is a more
    honest picture of lexical search across a border than any accuracy number.

    Fusion is RRF rather than a weighted sum of scores. BM25 scores and cosines
    are not on a shared scale, and normalising them per result list would make
    the blend depend on how many candidates each side happened to return.

    Measured, so the panel can be labelled rather than left to imply: over 300
    Polish queries dense scores 0.827 at top-1, BM25 0.243, SPLADE 0.310, and
    RRF of dense with either drops below dense alone. Sparse is being shown
    here because it loses in an instructive way.
    """
    sp = S.get("sparse")
    if not sp:
        return JSONResponse({"available": False}, status_code=503)

    flt = None
    if target:
        flt = models.Filter(must=[models.FieldCondition(
            key="lang", match=models.MatchValue(value=target))])

    qv = embed_query(q)
    if qlang in S["off_q"]:
        qv = apply_shift(qv[None, :], S["off_q"][qlang], SPARSE_ALPHA)[0]
    qv = truncate(qv, SPARSE_DIM)[0]

    def sparse_query(name):
        e = next(sp[name].query_embed(q))
        return models.SparseVector(indices=e.indices.tolist(),
                                   values=e.values.tolist())

    bm25_q, splade_q = sparse_query("bm25"), sparse_query("splade")

    def run(query, using):
        return sp["client"].query_points(
            SPARSE_COLLECTION, query=query, using=using, limit=k,
            query_filter=flt, with_payload=True).points

    runs = {
        "dense": run(qv.tolist(), "dense"),
        "bm25": run(bm25_q, "bm25"),
        "splade": run(splade_q, "splade"),
    }
    # Fusion is one server-side query: both branches prefetch wider than k so
    # RRF has ranks to work with, and Qdrant does the combining.
    runs["fusion"] = sp["client"].query_points(
        SPARSE_COLLECTION,
        prefetch=[
            models.Prefetch(query=qv.tolist(), using="dense",
                            limit=k * 5, filter=flt),
            models.Prefetch(query=splade_q, using="splade",
                            limit=k * 5, filter=flt),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=k, query_filter=flt, with_payload=True).points

    def wrap(points):
        return [{"score": round(p.score, 4), "atc": p.payload["atc"],
                 "lang": p.payload["lang"], "brand": p.payload["brand"],
                 "text": p.payload["text"],
                 "match": bool(gold) and p.payload["atc"] == gold}
                for p in points]

    return {
        "available": True, "query": q, "qlang": qlang, "target": target,
        "gold": gold, "alpha": SPARSE_ALPHA, "dim": SPARSE_DIM,
        "rrf_k": RRF_K,
        "n_query_terms": len(bm25_q.indices),
        # How many of the query's terms occur anywhere in the target-language
        # documents. Zero is the headline -- the words are simply not in the
        # Spanish index, so BM25 has nothing to rank and comes back empty.
        "matched_terms": len(set(bm25_q.indices)
                             & sp["vocab"].get(target or PIVOT, set())),
        "bm25_empty": len(runs["bm25"]) == 0,
        "methods": {n: wrap(p) for n, p in runs.items()},
    }


@app.post("/api/med/scan")
async def api_scan(frames: list[UploadFile] = File(...),
                   alpha: float = Query(1.0), dim: int = Query(1024),
                   k: int = Query(12), target: str = Query("es"),
                   rerank: bool = Query(False), qlang: str = Query("")):
    """Camera frames -> OCR -> country of origin -> retrieval.

    Takes a list of frames, not a single image, because a phone pointed at a
    carton is a video and not a photograph. Glare moves, focus hunts, and the
    brand line is legible in maybe one frame out of five. Merging the blocks
    across frames and keeping the best reading of each distinct string is what
    makes the live demo survive a badly lit room.

    qlang is an override here, not an input. Empty means "read it off the box",
    which is the honest version of the task: nobody holding a carton tells the
    search engine what country they are standing in first. It matters more than
    a convenience, because SHIFT subtracts a per-language offset -- the wrong
    language does not soften the result, it applies a Dutch-shaped correction to
    a Polish vector and lands nowhere near the molecule.
    """
    merged = {}
    per_frame = []
    for f in frames:
        try:
            blocks = vision_ocr.read(await f.read())
        except Exception as e:
            per_frame.append({"error": str(e)})
            continue
        per_frame.append({"n_blocks": len(blocks)})
        for b in blocks:
            key = b["text"].strip().lower()
            prev = merged.get(key)
            if prev is None or b["confidence"] * b["area"] > prev["confidence"] * prev["area"]:
                merged[key] = b

    blocks = list(merged.values())
    if not blocks:
        return JSONResponse(
            {"error": "no text found", "frames": per_frame}, status_code=422)

    text = vision_ocr.query_text(blocks)
    det = S["langid"](text) if S.get("langid") else None

    # An unconfident guess is worse than no guess, because acting on it applies
    # a wrong offset rather than none. Every detection error measured in
    # eval_langid.py sat below the margin threshold, so falling back to the
    # pivot -- which by construction has no offset -- degrades to plain
    # unshifted retrieval instead of to actively misdirected retrieval.
    used = qlang or (det["lang"] if det and det["confident"] else PIVOT)

    hits = search(text, alpha, dim, k, target or None, rerank, qlang=used)
    for h in hits:
        h["path"] = atc_path(h["atc"])
    return {"ocr": sorted(blocks, key=lambda b: -b["area"])[:12],
            "query": text, "frames": per_frame, "rerank": rerank,
            "qlang": used, "detected": det, "override": bool(qlang),
            "alpha": alpha, "dim": dim, "hits": hits}


@app.get("/api/med/geometry")
def api_geometry(atc: str = "", limit: int = 600):
    """2D Poincare and Euclidean coordinates for the ATC tree panel."""
    tree, idx = S["tree"], S["code_idx"]
    codes = S["codes"][:limit]
    hi = {c["code"] for c in atc_path(atc)} if atc else set()
    p, e = S["poincare2"], S["euclid2"]
    out = []
    for c in codes:
        i = idx[c]
        node = tree.get(c, {})
        out.append({
            "code": c, "level": node.get("level"), "name": node.get("name", ""),
            "poincare": [round(float(p[i][0]), 4), round(float(p[i][1]), 4)],
            "euclidean": [round(float(e[i][0]), 4), round(float(e[i][1]), 4)],
            "radius": round(float(2 * np.arctanh(
                min(np.linalg.norm(p[i]), 1 - 1e-12))), 3),
            "on_path": c in hi,
        })
    return {"atc": atc, "path": atc_path(atc) if atc else [], "nodes": out}


@app.get("/med")
def index():
    return FileResponse(WEB / "med.html")


app.mount("/med_img", StaticFiles(directory=WEB / "med_img"), name="med_img")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8079)
