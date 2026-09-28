"""Can you change the geometry of a multilingual embedding you already have?

The question came up because this repo runs two geometries side by side -- a
flat text space (arctic, cosine) and a hyperbolic tree space (Poincare, ATC) --
and the obvious next thought is to merge them: take the multilingual vectors
that already exist and reinterpret them in hyperbolic space, getting hierarchy
for free without retraining anything.

This measures whether that is possible. Three tiers, because "change the
geometry" turns out to mean three different operations with three different
answers.

  tier 1  change the METRIC      rank Euclidean / cosine / angular / spherical
                                 geodesic against each other on the same vectors

  tier 2  change the CONTAINER   exponential-map the vectors into the Poincare
                                 ball and rank by Mobius distance, sweeping
                                 curvature

  tier 3  change the OBJECTIVE   retrain. Out of scope here -- it is the only
                                 tier that is not a free lunch, and train_atc.py
                                 is already what it looks like.

Tiers 1 and 2 are predicted to be *ranking-identical* to the baseline, for one
reason: arctic normalises. Every stored vector has norm 1.000000 (sigma 3e-08),
so the exponential map at the origin sends all 5,298 of them to the same radius,
and radius is the entire mechanism by which the Poincare ball encodes hierarchy.
The prediction is therefore not "hyperbolic is worse" but "hyperbolic is the
same function of the angle", which is a stronger and more falsifiable claim.

The escape hatch, and the only part of this whose answer was not known in
advance: arctic's *pre-normalisation* norms. If raw norm tracks generality --
if "nervous system" comes out systematically longer or shorter than
"paracetamol" -- then there is a real radial coordinate to map and tier 2
reopens. Test C measures that, with a text-length control, because ATC names at
different levels also differ in length and a raw correlation would not separate
the two.

Test E is the one alternative that is not ranking-neutral by construction:
SHIFT currently subtracts a flat offset and renormalises, which is the retraction
approximation of moving along the sphere's geodesic. The two agree to first order
in alpha and diverge as alpha grows, and this corpus runs at alpha=0.75.

Everything here reads cached artifacts except tests C and D, which need raw
unnormalised vectors and so re-encode with the model.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "artifacts" / "med"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shift_med import (PIVOT, SHIFTED, aligned_pairs, all_offsets,  # noqa: E402
                       load_corpus)

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
KS = (1, 3, 10)
N_PROBE = 400          # queries used for the rank-invariance tests
CURVATURES = (0.25, 0.5, 1.0, 2.0, 4.0)
SEED = 0


# --------------------------------------------------------------------- metrics
def rank_docs(Q, D, metric, c=1.0):
    """Return the argsort of every query against every document.

    All four metrics are computed from scratch in float64 rather than derived
    from one another, so that "identical" below is a measurement and not a
    restatement of the algebra.
    """
    Q = Q.astype(np.float64)
    D = D.astype(np.float64)
    if metric == "cosine":
        return np.argsort(-(Q @ D.T), axis=1)
    if metric == "euclidean":
        d = np.linalg.norm(Q[:, None, :] - D[None, :, :], axis=2)
        return np.argsort(d, axis=1)
    if metric == "angular":
        return np.argsort(np.arccos(np.clip(Q @ D.T, -1, 1)), axis=1)
    if metric == "poincare":
        # exp_0^c(v) = tanh(sqrt(c)||v||) * v/(sqrt(c)||v||); ||v||=1 here.
        r = np.tanh(np.sqrt(c)) / np.sqrt(c)
        X, Y = Q * r, D * r
        sq = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2)
        dx = 1 - np.sum(X ** 2, axis=1)
        dy = 1 - np.sum(Y ** 2, axis=1)
        return np.argsort(np.arccosh(
            1 + 2 * sq / np.maximum(dx[:, None] * dy[None, :], 1e-300)), axis=1)
    raise ValueError(metric)


def topk_agree(a, b, k):
    """Fraction of queries whose top-k list is identical, and total disagreeing
    positions. Both reported: one list differing in one slot is a different
    failure from every list differing everywhere."""
    same_rows = np.all(a[:, :k] == b[:, :k], axis=1)
    return float(same_rows.mean()), int((a[:, :k] != b[:, :k]).sum())


# ----------------------------------------------------------------------- shift
def shift_flat(vecs, offset, alpha):
    """What the repo does today: subtract, then project back to the sphere."""
    out = vecs - alpha * offset
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.clip(n, 1e-12, None)


def shift_geodesic(vecs, offset, alpha):
    """Move along the sphere's own geodesic instead of stepping off and back.

    The offset is a displacement in the ambient space, so it is first projected
    into the tangent space at each point -- the component that actually moves
    you along the sphere -- and then exponential-mapped. shift_flat is the
    first-order (retraction) approximation of this; they agree as alpha -> 0.
    """
    d = -offset[None, :]
    t = d - (vecs @ d[0])[:, None] * vecs            # tangent component at v
    tn = np.linalg.norm(t, axis=1, keepdims=True)
    tn = np.clip(tn, 1e-12, None)
    w = alpha * tn
    return np.cos(w) * vecs + np.sin(w) * (t / tn)


def retrieval_scores(qv, dv, docs, qlang, target, ks=KS):
    """hit@k and MRR@3 for one language pair. Correct == shares the ATC code.

    ATC is the label only. Nothing in the ranking path reads it.
    """
    langs = np.array([d["lang"] for d in docs])
    atcs = np.array([d["atc"] for d in docs])
    qi = np.where(langs == qlang)[0]
    di = np.where(langs == target)[0]
    if not len(qi) or not len(di):
        return {}
    # Only score queries whose molecule is actually present on the target
    # shelf; the rest are unanswerable and would just dilute every config
    # by the same constant.
    have = set(atcs[di].tolist())
    qi = np.array([i for i in qi if atcs[i] in have])

    S = qv[qi] @ dv[di].T
    order = np.argsort(-S, axis=1)
    gold = atcs[di][order] == atcs[qi][:, None]

    out = {"n": int(len(qi))}
    for k in ks:
        out[f"hit@{k}"] = round(float(gold[:, :k].any(axis=1).mean()), 4)
    rr = np.zeros(len(qi))
    for r in range(3):
        hit = gold[:, r] & (rr == 0)
        rr[hit] = 1.0 / (r + 1)
    out["mrr@3"] = round(float(rr.mean()), 4)
    return out


# ------------------------------------------------------------------- reporting
def main():
    np.seterr(all="ignore")
    rng = np.random.default_rng(SEED)
    res = {}

    pairs, docs = load_corpus()
    dv = np.load(OUT / "arctic_docs.npy")
    qv = np.load(OUT / "arctic_queries.npy")
    langs = np.array([d["lang"] for d in docs])

    n = np.linalg.norm(dv, axis=1)
    print(f"arctic_docs {dv.shape}  norm mean {n.mean():.6f} "
          f"sigma {n.std():.3e}   <- normalised on disk")
    res["stored_norm_sigma"] = float(n.std())

    es = np.where(langs == PIVOT)[0]
    probe = rng.choice(np.where(langs == "pl")[0],
                       size=min(N_PROBE, int((langs == "pl").sum())),
                       replace=False)
    Q, D = qv[probe], dv[es]

    # ------------------------------------------------- tier 1: change the metric
    print(f"\ntier 1 -- change the metric ({len(probe)} pl queries "
          f"-> {len(es)} es docs)")
    base = rank_docs(Q, D, "cosine")
    res["tier1"] = {}
    for m in ("euclidean", "angular"):
        r = rank_docs(Q, D, m)
        row = {}
        for k in KS:
            frac, bad = topk_agree(base, r, k)
            row[f"top{k}_identical"] = round(frac, 6)
            row[f"top{k}_positions_differing"] = bad
        res["tier1"][m] = row
        print(f"  cosine vs {m:<10} " + "  ".join(
            f"top{k} {row[f'top{k}_identical']:.4f} "
            f"({row[f'top{k}_positions_differing']} pos)" for k in KS))

    # --------------------------------------------- tier 2: change the container
    print("\ntier 2 -- change the container (exp-map into the Poincare ball)")
    res["tier2"] = {}
    for c in CURVATURES:
        r = rank_docs(Q, D, "poincare", c=c)
        row = {}
        for k in KS:
            frac, bad = topk_agree(base, r, k)
            row[f"top{k}_identical"] = round(frac, 6)
            row[f"top{k}_positions_differing"] = bad
        res["tier2"][f"c={c}"] = row
        print(f"  cosine vs poincare c={c:<5} " + "  ".join(
            f"top{k} {row[f'top{k}_identical']:.4f} "
            f"({row[f'top{k}_positions_differing']} pos)" for k in KS))

    # radius spread after the map: the reason tier 2 is degenerate
    r1 = np.tanh(1.0)
    res["tier2_radius"] = {"all_points_at": round(float(r1), 6),
                           "radius_spread": 0.0}
    print(f"  every point lands at radius {r1:.6f}; spread 0 -- the radial "
          f"coordinate carries no information")

    # -------------------------------- tier 2 escape hatch: raw norms (tests C/D)
    print("\ntest C -- do arctic's PRE-normalisation norms encode generality?")
    tree = json.loads((OUT / "atc_tree.json").read_text())
    codes = sorted(tree)
    names = [tree[c].get("name") or c for c in codes]
    levels = np.array([tree[c]["level"] for c in codes])

    from sentence_transformers import SentenceTransformer

    from shift_med import get_model
    model = get_model()
    # normalize_embeddings=False is not enough: arctic's modules.json ends in a
    # Normalize layer, so the flag only skips a second normalisation of an
    # already-unit vector. The first run of this test reported sd 0.0000 at
    # every level, which was the pipeline speaking and not the model. Rebuild
    # it from Transformer + Pooling so the pooled norm survives.
    trunc = SentenceTransformer(modules=[model[0], model[1]],
                                device=model.device)
    raw = np.asarray(trunc.encode(names, batch_size=64,
                                  normalize_embeddings=False,
                                  show_progress_bar=False), dtype=np.float64)
    rn = np.linalg.norm(raw, axis=1)
    if rn.std() < 1e-6:
        print("  WARNING: norms still constant -- pooling layer is normalising")
    # Length control. ATC names get longer and more specific together, so a raw
    # norm-vs-level correlation cannot tell the two apart on its own.
    toklen = np.array([len(x.split()) for x in names], dtype=np.float64)
    chlen = np.array([len(x) for x in names], dtype=np.float64)

    def spearman(a, b):
        ra = np.argsort(np.argsort(a)).astype(np.float64)
        rb = np.argsort(np.argsort(b)).astype(np.float64)
        ra -= ra.mean(); rb -= rb.mean()
        return float((ra @ rb) / np.sqrt((ra @ ra) * (rb @ rb)))

    def partial(a, b, ctrl):
        """Spearman of a,b with ctrl regressed out of both (on ranks)."""
        R = [np.argsort(np.argsort(x)).astype(np.float64)
             for x in (a, b, ctrl)]
        R = [x - x.mean() for x in R]
        cz = R[2] / np.sqrt(R[2] @ R[2])
        ra = R[0] - (R[0] @ cz) * cz
        rb = R[1] - (R[1] @ cz) * cz
        return float((ra @ rb) / np.sqrt((ra @ ra) * (rb @ rb)))

    rho_level = spearman(rn, levels.astype(np.float64))
    rho_tok = spearman(rn, toklen)
    rho_ch = spearman(rn, chlen)
    rho_part = partial(rn, levels.astype(np.float64), chlen)
    res["testC"] = {
        "n_nodes": len(codes),
        "rho_norm_vs_atc_level": round(rho_level, 4),
        "rho_norm_vs_token_count": round(rho_tok, 4),
        "rho_norm_vs_char_count": round(rho_ch, 4),
        "partial_rho_norm_vs_level_given_charcount": round(rho_part, 4),
        "norm_by_level": {int(l): round(float(rn[levels == l].mean()), 4)
                          for l in sorted(set(levels.tolist()))},
        "norm_std_by_level": {int(l): round(float(rn[levels == l].std()), 4)
                              for l in sorted(set(levels.tolist()))},
    }
    # How much of the norm's variance the level actually explains. The
    # correlations above can be significant and still useless if the levels
    # overlap almost completely, which is the thing that decides whether there
    # is a radial coordinate worth mapping.
    grand = rn.mean()
    ss_between = sum(int((levels == l).sum())
                     * (rn[levels == l].mean() - grand) ** 2
                     for l in set(levels.tolist()))
    ss_total = float(((rn - grand) ** 2).sum())
    eta2 = float(ss_between / ss_total)
    monotone = [float(rn[levels == l].mean())
                for l in sorted(set(levels.tolist()))]
    is_mono = all(b >= a for a, b in zip(monotone, monotone[1:])) or \
        all(b <= a for a, b in zip(monotone, monotone[1:]))
    res["testC"]["eta_squared_level_explains_norm"] = round(eta2, 4)
    res["testC"]["monotone_in_level"] = bool(is_mono)
    print(f"  n={len(codes)} ATC nodes")
    print(f"  spearman(norm, atc level)              {rho_level:+.4f}")
    print(f"  spearman(norm, token count)            {rho_tok:+.4f}   <- control")
    print(f"  spearman(norm, char count)             {rho_ch:+.4f}   <- control")
    print(f"  partial(norm, level | char count)      {rho_part:+.4f}")
    print(f"  eta^2 (level explains norm)            {eta2:.4f}")
    print(f"  monotone in level?                     {is_mono}")
    for l in sorted(set(levels.tolist())):
        m = levels == l
        print(f"    level {l}  n={m.sum():<4} mean norm {rn[m].mean():.4f} "
              f"sd {rn[m].std():.4f}")

    # ------------------------------- test F: take the escape hatch and measure
    #
    # Test C says the raw norm is not flat, so the exponential map is no longer
    # degenerate and tier 2 is formally reopened. That is a statement about
    # variance, not about retrieval. This settles it operationally: embed
    # without the Normalize layer, exp-map the real norms into the ball, rank by
    # Mobius distance, and score the same top-3 task. alpha=0 throughout, so the
    # only thing that differs from the cosine baseline is the geometry.
    print("\ntest F -- exp-map the UNNORMALISED vectors (tier 2, hatch taken)")
    raw_d_path, raw_q_path = OUT / "arctic_raw_docs.npy", OUT / "arctic_raw_queries.npy"
    if raw_d_path.exists() and raw_q_path.exists():
        RD = np.load(raw_d_path).astype(np.float64)
        RQ = np.load(raw_q_path).astype(np.float64)
        print(f"  reusing cached raw vectors {RD.shape}")
    else:
        from shift_med import QUERY_PREFIX
        texts = [d["text"] for d in docs]
        print(f"  encoding {len(texts)} docs + queries without Normalize...")
        RD = np.asarray(trunc.encode(texts, batch_size=64,
                                     normalize_embeddings=False,
                                     show_progress_bar=False), dtype=np.float64)
        RQ = np.asarray(trunc.encode([QUERY_PREFIX + t for t in texts],
                                     batch_size=64, normalize_embeddings=False,
                                     show_progress_bar=False), dtype=np.float64)
        np.save(raw_d_path, RD.astype(np.float32))
        np.save(raw_q_path, RQ.astype(np.float32))

    rdn = np.linalg.norm(RD, axis=1)
    print(f"  raw doc norms: min {rdn.min():.3f} max {rdn.max():.3f} "
          f"mean {rdn.mean():.3f} sd {rdn.std():.3f}")

    def ball(V, s):
        """exp_0 with the real norms, scaled so the ball is actually used."""
        nv = np.linalg.norm(V, axis=1, keepdims=True)
        r = np.tanh(s * nv / rdn.mean())
        return r * (V / np.clip(nv, 1e-12, None))

    def poincare_scores(qlang, s, chunk=256):
        di = np.where(langs == PIVOT)[0]
        atcs = np.array([d["atc"] for d in docs])
        have = set(atcs[di].tolist())
        qi = np.array([i for i in np.where(langs == qlang)[0]
                       if atcs[i] in have])
        X, Y = ball(RQ[qi], s), ball(RD[di], s)
        dx = 1 - np.sum(X ** 2, axis=1)
        dy = 1 - np.sum(Y ** 2, axis=1)
        y2 = np.sum(Y ** 2, axis=1)
        hits = {k: 0 for k in KS}
        # Chunked over queries: the full difference tensor here would be
        # 2508 x 1248 x 1024 in float64, which is ~25 GB.
        for a in range(0, len(qi), chunk):
            b = min(a + chunk, len(qi))
            Xc = X[a:b]
            sq = (np.sum(Xc ** 2, axis=1)[:, None] + y2[None, :]
                  - 2 * (Xc @ Y.T))
            dist = np.arccosh(np.maximum(
                1 + 2 * np.maximum(sq, 0)
                / np.maximum(dx[a:b, None] * dy[None, :], 1e-300), 1.0))
            order = np.argsort(dist, axis=1)
            gold = atcs[di][order] == atcs[qi[a:b]][:, None]
            for k in KS:
                hits[k] += int(gold[:, :k].any(axis=1).sum())
        return {f"hit@{k}": round(hits[k] / len(qi), 4) for k in KS}

    res["testF"] = {}
    for qlang in SHIFTED:
        base = retrieval_scores(qv, dv, docs, qlang, PIVOT)
        res["testF"][qlang] = {"cosine_baseline_alpha0": base}
        print(f"  {qlang}->{PIVOT} cosine baseline   " + "  ".join(
            f"hit@{k} {base[f'hit@{k}']:.4f}" for k in KS))
        for s in (0.5, 1.0, 2.0):
            sc = poincare_scores(qlang, s)
            res["testF"][qlang][f"scale={s}"] = sc
            print(f"  {qlang}->{PIVOT} poincare scale={s:<4} " + "  ".join(
                f"hit@{k} {sc[f'hit@{k}']:.4f}" for k in KS))

    # ------------------------------------ test E: spherical vs flat SHIFT
    #
    # SHIFT is applied to the QUERY, not to the candidate documents. The first
    # run of this file shifted documents and measured a dead-flat +0.0000 at
    # every alpha, which was correct arithmetic answering the wrong question:
    # the target shelf is Spanish, Spanish is the pivot, and the pivot's offset
    # is the mean of (es - es) = 0 by construction. Shifting the corpus
    # therefore never moved a single candidate. export_browser.py says the same
    # thing in prose -- it exports query offsets only.
    print("\ntest E -- spherical geodesic SHIFT vs subtract-and-renormalise")
    off_q = all_offsets(qv, docs)
    res["testE"] = {}

    def run(fn, alpha, qlang):
        q = qv.copy()
        m = langs == qlang
        q[m] = fn(qv[m], off_q[qlang], alpha)
        return retrieval_scores(q, dv, docs, qlang, PIVOT)

    for qlang in SHIFTED:
        res["testE"][qlang] = {}
        for a in ALPHAS:
            row = {"flat": run(shift_flat, a, qlang),
                   "geodesic": run(shift_geodesic, a, qlang)}
            m = langs == qlang
            sep = float(np.linalg.norm(
                shift_flat(qv[m], off_q[qlang], a)
                - shift_geodesic(qv[m], off_q[qlang], a), axis=1).mean())
            row["mean_displacement_between_operators"] = round(sep, 5)
            res["testE"][qlang][f"alpha={a}"] = row
            f3 = row["flat"].get("hit@3", float("nan"))
            g3 = row["geodesic"].get("hit@3", float("nan"))
            print(f"  {qlang}->{PIVOT} a={a:<5} hit@3 flat {f3:.4f}  "
                  f"geodesic {g3:.4f}  delta {g3 - f3:+.4f}   "
                  f"|flat-geo| {sep:.5f}")

    # ------------------------------------------- top-3 headline, SHIFT on/off
    print("\nheadline -- top 3, ATC as ground truth only")
    res["topk"] = {}
    for qlang in SHIFTED:
        res["topk"][qlang] = {}
        for a in (0.0, 0.75):
            sc = run(shift_flat, a, qlang)
            res["topk"][qlang][f"alpha={a}"] = sc
            print(f"  {qlang}->{PIVOT} a={a:<5} n={sc['n']:<5} " + "  ".join(
                f"hit@{k} {sc[f'hit@{k}']:.4f}" for k in KS)
                + f"  mrr@3 {sc['mrr@3']:.4f}")
        d = res["topk"][qlang]
        for k in KS:
            gap = d["alpha=0.75"][f"hit@{k}"] - d["alpha=0.0"][f"hit@{k}"]
            print(f"    delta hit@{k}  {gap:+.4f}")

    (OUT / "offset_geometry.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT / 'offset_geometry.json'}")


if __name__ == "__main__":
    main()
