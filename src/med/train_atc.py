"""Embed the ATC tree in the Poincare ball, and in Euclidean space for contrast.

The tree is small -- 529 nodes -- but it is a real five-level regulatory
hierarchy, not a synthetic one, and the levels carry clinical meaning:

    N          anatomical main group     (nervous system)
    N02        therapeutic subgroup      (analgesics)
    N02B       pharmacological subgroup  (other analgesics and antipyretics)
    N02BE      chemical subgroup         (anilides)
    N02BE01    chemical substance        (paracetamol)

Training follows Nickel & Kiela: each node is pulled toward its ancestors and
away from sampled negatives. Depth labels are never shown to the model, so when
the learned radius lines up with ATC level afterwards, that is the geometry
recovering the hierarchy rather than memorising it.

Two dimensionalities are produced for the same reason as the geometry demo:
dim 2 is what the disk visual can actually draw, dim 10 is what retrieval uses.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "artifacts" / "med"

from poincare import train, mean_average_precision  # noqa: E402

DIMS = (2, 10)


def build_pairs(tree):
    """(child, ancestor) index pairs over the transitive closure.

    Transitive rather than parent-only: linking a substance to every level
    above it, not just its immediate parent, is what makes radius track
    generality instead of just local tree position.
    """
    codes = sorted(tree)
    idx = {c: i for i, c in enumerate(codes)}
    pairs = []
    for c in codes:
        node, up = tree[c], []
        p = node["parent"]
        while p:
            up.append(p)
            p = tree[p]["parent"] if p in tree else None
        for a in up:
            if a in idx:
                pairs.append((idx[c], idx[a]))
    return codes, idx, pairs


def radii(emb):
    n = np.clip(np.linalg.norm(emb, axis=1), 0, 1 - 1e-12)
    return 2 * np.arctanh(n)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    tree = json.loads((OUT / "atc_tree.json").read_text())
    codes, idx, pairs = build_pairs(tree)
    levels = np.array([tree[c]["level"] for c in codes])
    print(f"{len(codes)} ATC nodes, {len(pairs)} ancestor pairs")
    print(f"levels: {np.bincount(levels)[1:]}\n")

    results = {}
    for dim in DIMS:
        for geom in ("poincare", "euclidean"):
            emb_t = train(pairs, len(codes), dim, geom, epochs=600,
                          n_neg=50, lr=15.0, verbose=False)
            mapk = mean_average_precision(emb_t.detach(), pairs, geom)
            emb = emb_t.detach().numpy().astype(np.float64)
            np.save(OUT / f"atc_{geom}_{dim}.npy", emb)

            key = f"{geom}_{dim}"
            results[key] = {"map": round(float(mapk), 4)}
            line = f"{geom:10s} dim={dim:2d}  MAP={mapk:.4f}"
            if geom == "poincare":
                r = radii(emb)
                rho = spearman(r, levels)
                results[key]["spearman_radius_level"] = round(rho, 3)
                results[key]["radius_by_level"] = {
                    int(l): round(float(r[levels == l].mean()), 2)
                    for l in sorted(set(levels.tolist()))
                }
                line += f"  spearman(radius, ATC level)={rho:+.3f}"
            print(line)

    (OUT / "atc_results.json").write_text(json.dumps(results, indent=1))
    (OUT / "atc_codes.json").write_text(json.dumps(codes))

    best = results.get("poincare_10", {})
    if "radius_by_level" in best:
        print("\nmean hyperbolic radius by ATC level (dim 10):")
        for lvl, r in best["radius_by_level"].items():
            print(f"   L{lvl}  {r}")


if __name__ == "__main__":
    main()
