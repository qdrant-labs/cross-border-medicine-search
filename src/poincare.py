"""Poincare-ball and Euclidean embeddings of a hierarchy (Nickel & Kiela, 2017).

Both models share data, loss and optimizer. The only difference is the geometry --
which is the comparison the Qdrant article makes.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from taxonomy import Taxonomy

EPS = 1e-5
ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# distances
# --------------------------------------------------------------------------
def acosh(x):
    x = torch.clamp(x, min=1.0 + 1e-7)
    return torch.log(x + torch.sqrt(x * x - 1.0))


def poincare_dist(u, v):
    """Geodesic distance in the Poincare ball."""
    sq = ((u - v) ** 2).sum(-1)
    un = torch.clamp((u ** 2).sum(-1), max=1 - EPS)
    vn = torch.clamp((v ** 2).sum(-1), max=1 - EPS)
    return acosh(1 + 2 * sq / ((1 - un) * (1 - vn)))


def euclid_dist(u, v):
    return torch.sqrt(torch.clamp(((u - v) ** 2).sum(-1), min=1e-12))


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
def train(pairs, n_items, dim, geometry, epochs=600, n_neg=50, lr=15.0,
          burnin=20, batch_size=128, seed=0, verbose=True):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)

    scale = 1e-3
    emb = torch.empty(n_items, dim).uniform_(-scale, scale, generator=g)
    emb.requires_grad_(True)

    dist = poincare_dist if geometry == "poincare" else euclid_dist
    pairs_t = torch.tensor(pairs, dtype=torch.long)

    # positives, so negatives are never accidental true ancestors
    positives = [set() for _ in range(n_items)]
    for u, v in pairs:
        positives[u].add(v)

    for epoch in range(epochs):
        cur_lr = lr * 0.1 if epoch < burnin else lr
        perm = torch.randperm(len(pairs_t), generator=g)
        total = 0.0

        for start in range(0, len(perm), batch_size):
            idx = perm[start:start + batch_size]
            u = pairs_t[idx, 0]
            v = pairs_t[idx, 1]
            B = len(u)

            neg = torch.randint(0, n_items, (B, n_neg), generator=g)

            cand = torch.cat([v.unsqueeze(1), neg], dim=1)          # (B, 1+n_neg)
            u_e = emb[u].unsqueeze(1).expand(-1, cand.shape[1], -1)
            d = dist(u_e, emb[cand])                                 # (B, 1+n_neg)

            # softmax over -distance; index 0 is the true ancestor
            loss = torch.nn.functional.cross_entropy(
                -d, torch.zeros(B, dtype=torch.long)
            )

            if emb.grad is not None:
                emb.grad.zero_()
            loss.backward()
            total += loss.item() * B

            with torch.no_grad():
                grad = emb.grad
                if geometry == "poincare":
                    # Riemannian gradient rescaling for the Poincare ball
                    n2 = (emb ** 2).sum(-1, keepdim=True)
                    grad = grad * ((1 - n2) ** 2 / 4)
                emb -= cur_lr * grad
                if geometry == "poincare":
                    # project back inside the ball
                    norm = emb.norm(dim=-1, keepdim=True)
                    factor = torch.clamp(norm, min=1e-12)
                    over = norm >= 1 - EPS
                    emb[over.squeeze(-1)] = (
                        emb[over.squeeze(-1)]
                        / factor[over.squeeze(-1)]
                        * (1 - EPS - 1e-6)
                    )

        if verbose and (epoch % 25 == 0 or epoch == epochs - 1):
            print(f"    epoch {epoch:4d}  loss {total / len(pairs_t):.4f}")

    return emb.detach()


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def mean_average_precision(emb, pairs, geometry, chunk=256):
    """MAP over the given relationships: how near the top do true targets land?"""
    dist = poincare_dist if geometry == "poincare" else euclid_dist
    n = emb.shape[0]

    positives = {}
    for u, v in pairs:
        positives.setdefault(u, set()).add(v)

    nodes = sorted(positives)
    aps = []
    with torch.no_grad():
        for start in range(0, len(nodes), chunk):
            batch = nodes[start:start + chunk]
            u_e = emb[torch.tensor(batch)].unsqueeze(1).expand(-1, n, -1)
            d = dist(u_e, emb.unsqueeze(0).expand(len(batch), -1, -1))
            for row, u in enumerate(batch):
                dd = d[row].clone()
                dd[u] = float("inf")                    # never rank self
                order = torch.argsort(dd).tolist()
                pos = positives[u]
                hits, precs = 0, []
                for rank, cand in enumerate(order, start=1):
                    if cand in pos:
                        hits += 1
                        precs.append(hits / rank)
                        if hits == len(pos):
                            break
                aps.append(sum(precs) / len(pos))
    return float(np.mean(aps))


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, nargs="+", default=[2, 5, 10, 20, 50])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--out", default=str(ROOT / "artifacts"))
    args = ap.parse_args()

    tax = Taxonomy()
    print("taxonomy:", tax.stats())

    ids = sorted(tax.by_id)
    idx = {cid: i for i, cid in enumerate(ids)}
    pairs = [(idx[a], idx[b]) for a, b in tax.ancestor_pairs()]
    parent_pairs = [(idx[a], idx[b]) for a, b in tax.parent_pairs()]

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    results = {}

    for dim in args.dims:
        for geometry in ("poincare", "euclidean"):
            t0 = time.time()
            print(f"\n[{geometry} {dim}d]")
            emb = train(pairs, len(ids), dim, geometry, epochs=args.epochs)
            m = mean_average_precision(emb, pairs, geometry)
            results[f"{geometry}_{dim}"] = {"map_ancestors": m}
            print(f"    MAP(ancestors) = {m:.3f}   [{time.time()-t0:.0f}s]")

            if geometry == "poincare":
                mp = mean_average_precision(emb, parent_pairs, geometry)
                results[f"{geometry}_{dim}"]["map_direct_parent"] = mp
                print(f"    MAP(direct parent) = {mp:.3f}")

            np.save(outdir / f"emb_{geometry}_{dim}.npy", emb.numpy())

    json.dump({"ids": ids, "results": results},
              open(outdir / "poincare_results.json", "w"), indent=2)
    print("\nsaved ->", outdir)


if __name__ == "__main__":
    main()
