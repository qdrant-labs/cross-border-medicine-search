"""Serve hyperbolic embeddings from Qdrant.

Strategy from the Qdrant article: do NOT convert Poincare vectors and index them
directly with HNSW. Instead store the Poincare coordinates as an ordinary
Euclidean vector, keep each point's squared norm in the payload, use HNSW to
fetch candidates, then rescore with the true geodesic using a Formula Query.

Formula Query has no acosh, but acosh(x) = ln(x + sqrt(x^2 - 1)), and it does
have ln and sqrt -- so the real hyperbolic distance is expressible server side.
"""
import json
import os
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models

from taxonomy import Taxonomy

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
COLLECTION = "taxonomy_geometry"


def get_client():
    """Local in-process Qdrant by default; set QDRANT_URL to use a real server
    (which is what you want if you care about true HNSW behaviour)."""
    url = os.environ.get("QDRANT_URL")
    if url:
        return QdrantClient(url=url, api_key=os.environ.get("QDRANT_API_KEY")), True
    return QdrantClient(path=str(ART / "qdrant_local")), False


# --------------------------------------------------------------------------
# the formula: -acosh( 1 + 2*d_E^2 / ((1-|q|^2)(1-|p|^2)) )
# --------------------------------------------------------------------------
def hyperbolic_formula(query_sq_norm: float):
    inner = {
        "sum": [
            1.0,
            {"div": {
                "left": {"mult": [2.0, {"pow": {"base": "$score", "exponent": 2.0}}]},
                "right": {"mult": [
                    1.0 - query_sq_norm,
                    {"sum": [1.0, {"neg": "sq_norm"}]},
                ]},
            }},
        ]
    }
    # negated so that higher score = nearer, which is what Qdrant ranks by
    return {"neg": {"ln": {"sum": [
        inner,
        {"sqrt": {"sum": [{"pow": {"base": inner, "exponent": 2.0}}, -1.0]}},
    ]}}}


def poincare_distance(u, v):
    """Reference implementation, used to verify what Qdrant returns."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    sq = ((u - v) ** 2).sum(-1)
    un = np.clip((u ** 2).sum(-1), 0, 1 - 1e-9)
    vn = np.clip((v ** 2).sum(-1), 0, 1 - 1e-9)
    x = np.clip(1 + 2 * sq / ((1 - un) * (1 - vn)), 1.0, None)
    return np.arccosh(x)


# --------------------------------------------------------------------------
def build(dim=5, recreate=True):
    tax = Taxonomy()
    ids = sorted(tax.by_id)

    hyp = np.load(ART / f"emb_poincare_{dim}.npy").astype(np.float64)
    euc = np.load(ART / f"emb_euclidean_{dim}.npy").astype(np.float64)
    hyp2 = np.load(ART / "emb_poincare_2.npy").astype(np.float64)

    # keep every point strictly inside the ball
    def clamp(v):
        n = np.linalg.norm(v, axis=1, keepdims=True)
        scale = np.minimum(1.0, (1 - 1e-6) / np.maximum(n, 1e-12))
        return v * scale

    hyp, hyp2 = clamp(hyp), clamp(hyp2)
    sq_norm = (hyp ** 2).sum(1)

    client, is_server = get_client()
    if recreate and client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)

    if not client.collection_exists(COLLECTION):
        client.create_collection(
            COLLECTION,
            vectors_config={
                "hyperbolic": models.VectorParams(size=dim, distance=models.Distance.EUCLID),
                "euclidean": models.VectorParams(size=dim, distance=models.Distance.EUCLID),
                "disk2d": models.VectorParams(size=2, distance=models.Distance.EUCLID),
            },
        )
        # without this payload index the rescore degrades into a full scan
        client.create_payload_index(
            COLLECTION, "sq_norm", field_schema=models.PayloadSchemaType.FLOAT)

    points = []
    for i, cid in enumerate(ids):
        node = tax.by_id[cid]
        points.append(models.PointStruct(
            id=i,
            vector={
                "hyperbolic": hyp[i].tolist(),
                "euclidean": euc[i].tolist(),
                "disk2d": hyp2[i].tolist(),
            },
            payload={
                "sq_norm": float(sq_norm[i]),
                "category_id": cid,
                "name": node["name"],
                "full": node["full"],
                "depth": node["depth"],
                "radius": float(np.linalg.norm(hyp[i])),
                "radius2d": float(np.linalg.norm(hyp2[i])),
            },
        ))
    client.upload_points(COLLECTION, points, batch_size=512, wait=True)
    return client, ids, hyp, is_server


# --------------------------------------------------------------------------
def search(client, vec, prefetch=1000, limit=10, rescore=True, using="hyperbolic"):
    qsq = float(np.dot(vec, vec))
    if not rescore:
        r = client.query_points(COLLECTION, query=list(map(float, vec)),
                                using=using, limit=limit, with_payload=True)
        return r.points
    r = client.query_points(
        COLLECTION,
        prefetch=[models.Prefetch(query=list(map(float, vec)), using=using, limit=prefetch)],
        query=models.FormulaQuery(formula=hyperbolic_formula(qsq)),
        limit=limit, with_payload=True,
    )
    return r.points


def prefetch_sweep(client, hyp, ids, n_probe=120, k=10, seed=0,
                   sizes=(10, 50, 100, 300, 1000)):
    """Reproduce the article's prefetch table: Euclidean-only vs with rescore."""
    rng = np.random.default_rng(seed)
    probes = rng.choice(len(ids), size=n_probe, replace=False)

    # ground truth: exact hyperbolic nearest neighbours
    truth = {}
    for p in probes:
        d = poincare_distance(hyp[p][None, :], hyp)
        d[p] = np.inf
        truth[p] = set(np.argsort(d)[:k].tolist())

    rows = []
    for size in sizes:
        eu, re = [], []
        for p in probes:
            v = hyp[p]
            got_e = {pt.id for pt in search(client, v, limit=k, rescore=False)} - {int(p)}
            got_r = {pt.id for pt in search(client, v, prefetch=size, limit=k, rescore=True)} - {int(p)}
            eu.append(len(got_e & truth[p]) / k)
            re.append(len(got_r & truth[p]) / k)
        rows.append({"prefetch": size,
                     "euclidean_only": float(np.mean(eu)),
                     "with_rescore": float(np.mean(re))})
        print(f"  prefetch {size:>5}  euclid-only {rows[-1]['euclidean_only']:.3f}"
              f"   with rescore {rows[-1]['with_rescore']:.3f}")
    return rows


def verify_against_numpy(client, hyp, n=25, seed=1):
    """Confirm Qdrant's formula score equals locally computed hyperbolic distance."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for p in rng.choice(len(hyp), size=n, replace=False):
        for pt in search(client, hyp[p], prefetch=200, limit=5):
            local = poincare_distance(hyp[p], hyp[pt.id])
            worst = max(worst, abs(local - (-pt.score)))
    return float(worst)


if __name__ == "__main__":
    client, ids, hyp, is_server = build()
    print(f"indexed {len(ids)} categories "
          f"({'remote server' if is_server else 'local mode: exact prefetch, no HNSW approximation'})")
    print("\nprefetch sweep:")
    rows = prefetch_sweep(client, hyp, ids)
    delta = verify_against_numpy(client, hyp)
    print(f"\nmax |qdrant_score - numpy_distance| = {delta:.2e}")
    json.dump({"prefetch_sweep": rows, "max_abs_error": delta, "server": is_server},
              open(ART / "hyperbolic_results.json", "w"), indent=2)
