"""Does hyperbolic beat Euclidean on this tree, or only at one question?

The headline in the README is that 5 hyperbolic dimensions beat 50 Euclidean
ones, measured as MAP on the ancestor task. That number is real and reproduced.
It is also directional in a way the metric's name does not admit, and the demo
puts both geometries on screen side by side, so the asymmetry is visible to
anyone in the room who clicks a hub node. Better to measure it than to be asked
about it.

Three measurements, same embeddings, same distance functions:

  up      child -> its ancestors. The published task. Hyperbolic wins.

  down    node -> its descendants. The same pairs, reversed. This is not a
          trick: "given a category, find the drugs in it" is at least as
          natural a retrieval query as "given a drug, find its category", and
          nothing about the stored vectors prefers one direction.

  kin@k   symmetric and label-free: of the k nearest nodes, how many are
          taxonomic relatives (ancestor, descendant, or sibling) at all? This
          is what the neighbour lists in the demo actually show, so it is the
          measurement that corresponds to what the audience sees.

The expected shape of the result, written down before running it: hyperbolic
should win `up`, lose `down`, and tie `kin@k`, because hyperbolic space makes
room by pushing outward. A child has exactly one parent sitting inward of it,
which is easy to rank first. A parent with forty children cannot be nearest to
all forty, because the circumference available at its radius is bounded while
the number of descendants is not. The geometry is good at looking up and
structurally bad at looking down, and `up` is the direction the literature
reports.

Hub sizes are reported alongside, because if that explanation is right the gap
should be worst at the nodes with the most descendants, and that is a check the
mean can hide.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
A = ROOT / "artifacts" / "med"
sys.path.insert(0, str(ROOT / "src"))

from poincare import mean_average_precision  # noqa: E402

DIM = 10
KS = (6, 10)
N_HUBS = 5


def load():
    codes = json.loads((A / "atc_codes.json").read_text())
    tree = json.loads((A / "atc_tree.json").read_text())
    emb = {
        "poincare": torch.tensor(np.load(A / f"atc_poincare_{DIM}.npy"),
                                 dtype=torch.float64),
        "euclidean": torch.tensor(np.load(A / f"atc_euclidean_{DIM}.npy"),
                                  dtype=torch.float64),
    }
    return codes, tree, emb


def ancestor_pairs(codes, tree):
    """(child, ancestor) over the transitive closure -- same pairs as training."""
    idx = {c: i for i, c in enumerate(codes)}
    out = []
    for c in codes:
        p = (tree.get(c) or {}).get("parent")
        while p:
            if p in idx:
                out.append((idx[c], idx[p]))
            p = (tree.get(p) or {}).get("parent")
    return idx, out


def dists(emb, geom):
    """Full pairwise distance matrix in the given geometry.

    The errstate is not papering over a degenerate input: macOS ships numpy on
    Accelerate, whose vectorised BLAS leaves floating-point flags set, so
    `matmul` reports divide-by-zero and overflow on inputs that contain
    neither. The assert is the actual check -- the max Poincare norm here is
    0.9987, so 1-|x|^2 never approaches zero and no term is near singular.
    """
    X = emb.numpy()
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        gram = X @ X.T
        n2 = np.sum(X ** 2, 1)
        num = n2[:, None] + n2[None, :] - 2 * gram
        if geom == "euclidean":
            out = np.sqrt(np.maximum(num, 0))
        else:
            den = np.outer(1 - n2, 1 - n2)
            out = np.arccosh(1 + 2 * np.maximum(num, 0) / np.maximum(den, 1e-12))
    assert np.isfinite(out).all(), f"non-finite distances in {geom}"
    return out


def kin_at_k(codes, tree, D, k):
    """Share of the k nearest nodes that are taxonomic relatives.

    Strict: two level-5 substances under different level-4 parents do not
    count, even if they are clinically close. So this is a floor, and the same
    floor for both geometries, which is all the comparison needs.
    """
    par = {c: (tree.get(c) or {}).get("parent") for c in codes}
    tot = 0
    for i, c in enumerate(codes):
        d = D[i].copy()
        d[i] = np.inf
        for j in np.argsort(d)[:k]:
            o = codes[j]
            rel = (c.startswith(o) or o.startswith(c)
                   or (par[c] is not None and par[o] == par[c]))
            tot += bool(rel)
    return tot / (len(codes) * k)


def hub_map(codes, tree, D, pairs_down, hubs):
    """Average precision on the descendant task, per hub, not averaged away."""
    pos = {}
    for u, v in pairs_down:
        pos.setdefault(u, set()).add(v)
    out = {}
    for i in hubs:
        d = D[i].copy()
        d[i] = np.inf
        order = np.argsort(d)
        p, hits, precs = pos[i], 0, []
        for rank, cand in enumerate(order, 1):
            if cand in p:
                hits += 1
                precs.append(hits / rank)
                if hits == len(p):
                    break
        out[codes[i]] = round(sum(precs) / len(p), 4)
    return out


def main():
    codes, tree, emb = load()
    idx, up = ancestor_pairs(codes, tree)
    down = [(v, u) for u, v in up]

    ndesc = {}
    for _, a in up:
        ndesc[a] = ndesc.get(a, 0) + 1
    hubs = sorted(ndesc, key=lambda i: -ndesc[i])[:N_HUBS]

    res = {"dim": DIM, "n_nodes": len(codes), "n_pairs": len(up)}
    print(f"ATC tree, dim={DIM}, {len(codes)} nodes, {len(up)} ancestor pairs\n")

    D = {g: dists(emb[g], g) for g in emb}

    print(f"{'':<26}{'poincare':>10}{'euclidean':>11}{'delta':>9}")
    for name, pairs in (("MAP up (child->anc)", up),
                        ("MAP down (node->desc)", down)):
        vals = {g: mean_average_precision(emb[g], pairs, g) for g in emb}
        res[name] = {g: round(vals[g], 4) for g in vals}
        d = vals["poincare"] - vals["euclidean"]
        print(f"  {name:<24}{vals['poincare']:>10.4f}{vals['euclidean']:>11.4f}"
              f"{d:>+9.4f}")

    for k in KS:
        vals = {g: kin_at_k(codes, tree, D[g], k) for g in D}
        res[f"kin@{k}"] = {g: round(vals[g], 4) for g in vals}
        d = vals["poincare"] - vals["euclidean"]
        print(f"  {'kin@' + str(k) + ' (symmetric)':<24}{vals['poincare']:>10.4f}"
              f"{vals['euclidean']:>11.4f}{d:>+9.4f}")

    # If the circumference argument is right, the descendant gap is worst
    # exactly where the fan-out is largest. Printed per hub so that claim is
    # falsifiable rather than folded into a mean.
    print(f"\ndescendant AP at the {N_HUBS} largest hubs")
    hp = hub_map(codes, tree, D["poincare"], down, hubs)
    he = hub_map(codes, tree, D["euclidean"], down, hubs)
    res["hubs"] = {codes[i]: {"n_descendants": ndesc[i],
                              "poincare": hp[codes[i]],
                              "euclidean": he[codes[i]]} for i in hubs}
    print(f"  {'code':<8}{'desc':>6}{'poincare':>10}{'euclidean':>11}{'delta':>9}")
    for i in hubs:
        c = codes[i]
        print(f"  {c:<8}{ndesc[i]:>6}{hp[c]:>10.4f}{he[c]:>11.4f}"
              f"{hp[c] - he[c]:>+9.4f}")

    out = A / "hierarchy_direction.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
