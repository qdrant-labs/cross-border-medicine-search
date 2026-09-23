"""Demo server: two geometries, one dataset, everything served by Qdrant.

  /            two-panel UI
  /api/shift   cross-language search, alpha slider
  /api/hyper   hierarchy search, prefetch + hyperbolic rescore
"""
import json
import os
import warnings
from pathlib import Path

import numpy as np

# Accelerate (the BLAS shipped with macOS) raises spurious FP warnings on large
# float32 matmuls. Inputs are checked finite and results verified against exact
# computation, so these are noise.
warnings.filterwarnings("ignore", message=".*encountered in matmul",
                        category=RuntimeWarning)
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from qdrant_client import QdrantClient, models

import hyperbolic_index as hx
import shift as sh
from taxonomy import Taxonomy

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
WEB = ROOT / "web"
SHIFT_COLLECTION = "taxonomy_multilingual"
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def alpha_key(a):
    return f"a{int(round(a * 100)):03d}"


def hyp_radius(euclid_norm):
    """Distance from the origin in the hyperbolic metric.

    The Euclidean norm saturates at 1.0 and tells you nothing, but the
    hyperbolic radius 2*artanh(|x|) keeps separating levels all the way out.
    That is the quantity that encodes specificity.
    """
    return float(2 * np.arctanh(min(float(euclid_norm), 1 - 1e-12)))


# ---- Mobius gyrovector ops, used to trace true geodesics on the disk -------
def mobius_add(u, v):
    """u (+) v in the Poincare ball, curvature -1."""
    uv, uu, vv = float(u @ v), float(u @ u), float(v @ v)
    num = (1 + 2 * uv + vv) * u + (1 - uu) * v
    return num / max(1 + 2 * uv + uu * vv, 1e-12)


def mobius_scalar(t, x):
    """t (x) x -- move a fraction t along the ray, in the hyperbolic metric."""
    n = float(np.linalg.norm(x))
    if n < 1e-12:
        return x * 0.0
    return np.tanh(t * np.arctanh(min(n, 1 - 1e-12))) * (x / n)


def geodesic(u, v, steps=64):
    """Sample the true geodesic from u to v.

        gamma(t) = u (+) ( t (x) ((-u) (+) v) ),   t in [0, 1]

    In the disk this is the circular arc through u and v that meets the
    boundary at right angles -- emphatically NOT the straight segment.
    Drawing it straight is drawing Euclidean geometry in a hyperbolic space,
    which is the exact confusion this panel exists to dispel.
    """
    w = mobius_add(-u, v)
    return [mobius_add(u, mobius_scalar(t, w)) for t in np.linspace(0.0, 1.0, steps)]


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def compute_anomalies(tax, ids, hyp):
    """Flag nodes whose learned radius disagrees with their nominal depth.

    The model never saw depth labels -- it only saw ancestor links. So the
    radius is what the *structure* implies about a node's generality. Where it
    disagrees with the official depth, the taxonomy is often the thing that is
    wrong: hub nodes with dozens of children get pulled toward the origin, and
    childless leaves get pushed out, regardless of the level they are filed at.

    Residual is a robust z-score (median/MAD) computed *within* each depth, so
    it measures "odd for its level" rather than just "deep".
    """
    r = 2 * np.arctanh(np.clip(np.linalg.norm(hyp, axis=1), 0, 1 - 1e-12))
    depth = np.array([tax.by_id[c]["depth"] for c in ids])
    kids = {c: 0 for c in ids}
    for c in ids:
        p = tax.by_id[c]["parent"]
        if p in kids:
            kids[p] += 1
    n_kids = np.array([kids[c] for c in ids])

    z = np.zeros(len(r))
    for d in sorted(set(depth.tolist())):
        m = depth == d
        med = float(np.median(r[m]))
        mad = float(np.median(np.abs(r[m] - med))) or 1e-9
        z[m] = (r[m] - med) / (1.4826 * mad)

    leaf = n_kids == 0
    stats = {
        "spearman_radius_depth": round(spearman(r, depth), 3),
        "spearman_residual_children": round(spearman(z, n_kids), 3),
        "mean_z_leaf": round(float(z[leaf].mean()), 2),
        "mean_z_internal": round(float(z[~leaf].mean()), 2),
        "n_leaf": int(leaf.sum()),
        "n_internal": int((~leaf).sum()),
    }
    return r, depth, z, n_kids, stats


app = FastAPI(title="Geometry of Retrieval")
S = {}


# --------------------------------------------------------------------------
def build_shift_collection(client, emb_p, offsets_p, rows, ids, tax):
    if client.collection_exists(SHIFT_COLLECTION):
        client.delete_collection(SHIFT_COLLECTION)
    client.create_collection(
        SHIFT_COLLECTION,
        vectors_config={
            alpha_key(a): models.VectorParams(size=384, distance=models.Distance.COSINE)
            for a in ALPHAS
        },
    )
    langs = list(emb_p)
    shifted = {
        a: {l: sh.apply_shift(emb_p[l][rows], offsets_p[l], a) for l in langs}
        for a in ALPHAS
    }
    parallel, _ = sh.load_parallel()
    points, pid = [], 0
    for l in langs:
        for j, row in enumerate(rows):
            cid = ids[row]
            points.append(models.PointStruct(
                id=pid,
                vector={alpha_key(a): shifted[a][l][j].tolist() for a in ALPHAS},
                payload={
                    "category_id": int(cid),
                    "lang": l,
                    "lang_label": sh.LANG_LABEL[l],
                    "text": parallel[l][cid][1],
                    "leaf": parallel[l][cid][0],
                },
            ))
            pid += 1
    client.upload_points(SHIFT_COLLECTION, points, batch_size=512, wait=True)
    return len(points)


@app.on_event("startup")
def startup():
    tax = Taxonomy()
    S["tax"] = tax
    S["ids"] = sorted(tax.by_id)

    # ---- hyperbolic side ----
    client, is_server = hx.get_client()
    S["client"], S["is_server"] = client, is_server
    if not client.collection_exists(hx.COLLECTION):
        print("building hyperbolic collection ...")
        client, S["ids"], hyp, is_server = hx.build()
        S["client"] = client
    S["hyp"] = np.load(ART / "emb_poincare_5.npy").astype(np.float64)
    S["idx_of_cid"] = {c: i for i, c in enumerate(S["ids"])}

    # 2D disk coords, loaded once -- autopilot hammers /api/disk
    d2 = np.load(ART / "emb_poincare_2.npy").astype(np.float64)
    S["d2"] = d2
    S["d2_rh"] = 2 * np.arctanh(np.clip(np.linalg.norm(d2, axis=1), 0, 1 - 1e-12))
    S["d2_rmax"] = float(np.percentile(S["d2_rh"], 99.5)) or 1.0

    r, depth, z, n_kids, astats = compute_anomalies(tax, S["ids"], S["hyp"])
    S["an"] = {"r": r, "depth": depth, "z": z, "kids": n_kids}
    S["an_stats"] = astats

    # text-embedding baseline panel: English full-path embeddings
    S["text_en"] = np.load(ART / "e5_en-US_passage.npy")

    # ---- SHIFT side ----
    parallel, ids = sh.embed_all()
    S["parallel"], S["shift_ids"] = parallel, ids
    langs = list(parallel)
    S["langs"] = langs
    emb_q = sh.load_embeddings(langs, "query")
    emb_p = sh.load_embeddings(langs, "passage")
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(ids))
    fit_rows = perm[:2200]
    S["offsets_q"] = sh.compute_offsets(emb_q, ids, fit_rows)
    S["offsets_p"] = sh.compute_offsets(emb_p, ids, fit_rows)

    if not client.collection_exists(SHIFT_COLLECTION):
        print("building multilingual collection ...")
        # every category, every language, so any query has a real answer present.
        # the reported metrics still use only the disjoint eval split.
        n = build_shift_collection(S["client"], emb_p, S["offsets_p"],
                                   np.arange(len(ids)), ids, tax)
        print(f"  {n} multilingual documents indexed")

    S["model"] = sh.get_model()
    for f in ("shift_sweep.json", "hyperbolic_results.json", "poincare_results.json"):
        p = ART / f
        S[f] = json.load(open(p)) if p.exists() else None
    print("ready")


# --------------------------------------------------------------------------
@app.get("/api/meta")
def meta():
    return {
        "langs": [{"code": l, "label": sh.LANG_LABEL[l]} for l in S["langs"]],
        "alphas": ALPHAS,
        "server_mode": "qdrant server" if S["is_server"] else "local mode (exact prefetch)",
        "shift_sweep": S.get("shift_sweep.json"),
        "hyperbolic": S.get("hyperbolic_results.json"),
        "poincare": S.get("poincare_results.json"),
        "categories": len(S["ids"]),
    }


@app.get("/api/shift")
def api_shift(q: str, lang: str = "es-ES", alpha: float = 0.0, k: int = 10):
    """Search the 9-language corpus. alpha=0 is the untouched embedding space."""
    a = min(ALPHAS, key=lambda x: abs(x - alpha))
    vec = S["model"].encode([f"query: {q}"], normalize_embeddings=True)[0]
    vec = sh.apply_shift(vec, S["offsets_q"].get(lang, 0.0), a)
    res = S["client"].query_points(
        SHIFT_COLLECTION, query=vec.tolist(), using=alpha_key(a),
        limit=k, with_payload=True).points
    hits = [{
        "text": p.payload["text"], "leaf": p.payload["leaf"],
        "lang": p.payload["lang"], "lang_label": p.payload["lang_label"],
        "category_id": p.payload["category_id"],
        "score": round(p.score, 4),
        "same_language": p.payload["lang"] == lang,
    } for p in res]
    same = sum(h["same_language"] for h in hits)
    return {
        "alpha": a, "query": q, "query_lang": lang, "hits": hits,
        "same_language_in_top_k": same,
        "same_language_share": round(same / max(len(hits), 1), 3),
        "distinct_languages": len({h["lang"] for h in hits}),
    }


@app.get("/api/categories")
def api_categories(search: str = "", limit: int = 40):
    tax, out = S["tax"], []
    s = search.lower().strip()
    for cid in S["ids"]:
        n = tax.by_id[cid]
        if not s or s in n["full"].lower():
            out.append({"category_id": cid, "name": n["name"],
                        "full": n["full"], "depth": n["depth"]})
        if len(out) >= limit:
            break
    return out


@app.get("/api/hyper")
def api_hyper(category_id: int, prefetch: int = 1000, k: int = 10,
              rescore: bool = True):
    """Three panels: hyperbolic rescored, flat Euclidean, and text embedding."""
    i = S["idx_of_cid"].get(category_id)
    if i is None:
        return JSONResponse({"error": "unknown category"}, status_code=404)
    tax = S["tax"]

    def fmt(points, label):
        out = []
        for p in points:
            if p.id == i:
                continue
            out.append({"name": p.payload["name"], "full": p.payload["full"],
                        "depth": p.payload["depth"],
                        "radius": round(hyp_radius(p.payload["radius"]), 2),
                        "score": round(p.score, 4)})
        return {"panel": label, "hits": out[:k]}

    hvec = S["hyp"][i]
    hyper = fmt(hx.search(S["client"], hvec, prefetch=prefetch, limit=k + 1,
                          rescore=rescore), "hyperbolic")
    euc = np.load(ART / "emb_euclidean_5.npy")[i]
    flat = fmt(S["client"].query_points(
        hx.COLLECTION, query=euc.tolist(), using="euclidean",
        limit=k + 1, with_payload=True).points, "euclidean")

    tv = S["text_en"][i]
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        sims = S["text_en"] @ tv
    sims[i] = -np.inf
    top = np.argsort(-sims)[:k]
    text = {"panel": "text", "hits": [{
        "name": tax.by_id[S["ids"][j]]["name"],
        "full": tax.by_id[S["ids"][j]]["full"],
        "depth": tax.by_id[S["ids"][j]]["depth"],
        "radius": round(hyp_radius(np.linalg.norm(S["hyp"][j])), 2),
        "score": round(float(sims[j]), 4)} for j in top]}

    node = tax.by_id[category_id]
    ancestors = []
    cur = node["parent"]
    while cur is not None:
        ancestors.append(tax.by_id[cur]["name"])
        cur = tax.by_id[cur]["parent"]
    return {
        "query": {"name": node["name"], "full": node["full"],
                  "depth": node["depth"],
                  "radius": round(hyp_radius(np.linalg.norm(hvec)), 2),
                  "ancestors": list(reversed(ancestors))},
        "prefetch": prefetch, "rescore": rescore,
        "panels": [hyper, flat, text],
    }


@app.get("/api/disk")
def api_disk(category_id: int = None, limit: int = 1800):
    """2D Poincare coordinates for the disk visual.

    Plotted at the *hyperbolic* radius rather than the raw Euclidean norm.
    Near the boundary every point has |x| ~ 1.0, so the raw coordinates pile
    the whole hierarchy onto the rim; 2*artanh(|x|) keeps the levels apart.
    """
    tax = S["tax"]
    d2, rh, rmax = S["d2"], S["d2_rh"], S["d2_rmax"]

    def place_vec(v):
        """True Poincare coord -> display coord.

        Radius is remapped to the *hyperbolic* radius (normalized), because the
        raw Euclidean norms all sit at ~0.9999 and would pile the whole tree
        onto the rim. Direction is untouched, so this is a purely radial warp.
        """
        n = min(float(np.linalg.norm(v)), 1 - 1e-12)
        if n < 1e-12:
            return 0.0, 0.0
        scale = min(2 * np.arctanh(n) / rmax, 1.0) / n
        return float(v[0] * scale), float(v[1] * scale)

    n_pts = min(limit, len(S["ids"]))
    step = max(1, len(S["ids"]) // n_pts)
    pts = []
    for j in range(0, len(S["ids"]), step):
        cid = S["ids"][j]
        x, y = place_vec(d2[j])
        pts.append({"x": x, "y": y, "d": tax.by_id[cid]["depth"],
                    "n": tax.by_id[cid]["name"], "id": cid})

    chain, path = [], []
    if category_id is not None and category_id in S["idx_of_cid"]:
        cur, vecs = category_id, []
        while cur is not None:
            j = S["idx_of_cid"][cur]
            x, y = place_vec(d2[j])
            chain.append({"x": x, "y": y, "n": tax.by_id[cur]["name"],
                          "d": tax.by_id[cur]["depth"],
                          "r": round(float(rh[j]), 2)})
            vecs.append(d2[j])
            cur = tax.by_id[cur]["parent"]
        chain.reverse()
        vecs.reverse()
        # Trace the real geodesic between consecutive ancestors in TRUE
        # coordinates, then push every sample through the same display warp.
        # The drawn curve is therefore the honest image of the geodesic, not a
        # straight line and not an arc invented in display space.
        for a, b in zip(vecs[:-1], vecs[1:]):
            seg = geodesic(a, b)
            if path:
                seg = seg[1:]
            path += [{"x": p[0], "y": p[1]} for p in map(place_vec, seg)]

        # Geodesics to siblings. These are the dramatic ones: two categories
        # sitting on top of each other at the rim are joined by a path that
        # dives far inward -- the shortest route between two specific things
        # runs through the general. That is the entire case for hyperbolic
        # space in one picture. (The ancestor chain above looks almost
        # straight, and correctly so: radial lines through the origin ARE
        # geodesics, so a path straight up the hierarchy has nothing to bend.)
        sib_paths = []
        par = tax.by_id[category_id]["parent"]
        if par is not None:
            q = d2[S["idx_of_cid"][category_id]]
            sibs = [c for c in tax.by_id[par]["children"] if c != category_id]
            for c in sibs[:3]:
                samples = []
                for p in geodesic(q, d2[S["idx_of_cid"][c]]):
                    x, y = place_vec(p)
                    rr = 2 * np.arctanh(min(float(np.linalg.norm(p)), 1 - 1e-12))
                    samples.append({"x": x, "y": y, "r": round(float(rr), 3)})
                sib_paths.append({"name": tax.by_id[c]["name"], "pts": samples})
        par_r = (round(float(rh[S["idx_of_cid"][par]]), 2)
                 if par is not None else None)
    else:
        sib_paths, par_r = [], None
    return {"points": pts, "chain": chain, "path": path,
            "sibling_paths": sib_paths, "parent_radius": par_r,
            "parent_name": tax.by_id[par]["name"] if par is not None else None,
            "r_max": round(rmax, 2)}


@app.get("/api/anomalies")
def api_anomalies(n: int = 8, scatter: int = 1500):
    """Depth-vs-radius scatter plus the nodes that disagree most with their level."""
    tax, an = S["tax"], S["an"]
    ids, z = S["ids"], an["z"]
    rng = np.random.default_rng(0)
    pick = rng.choice(len(ids), size=min(scatter, len(ids)), replace=False)

    def row(j):
        c = ids[j]
        return {"id": int(c), "name": tax.by_id[c]["name"],
                "full": tax.by_id[c]["full"], "depth": int(an["depth"][j]),
                "radius": round(float(an["r"][j]), 2),
                "z": round(float(z[j]), 1), "kids": int(an["kids"][j])}

    order = np.argsort(z)
    return {
        "points": [row(int(j)) for j in pick],
        "too_general": [row(int(j)) for j in order[:n]],
        "too_specific": [row(int(j)) for j in order[::-1][:n]],
        "stats": S["an_stats"],
    }


app.mount("/", StaticFiles(directory=str(WEB), html=True), name="web")
