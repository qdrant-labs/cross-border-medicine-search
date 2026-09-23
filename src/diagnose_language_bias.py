"""Measure language-clustering bias in an existing Qdrant collection.

Point this at your own index. It needs nothing but a payload field naming each
document's language. No eval set, no labels, no ground truth.

Idea: use documents already in the collection as queries. For each sampled
document, retrieve its neighbours and ask what share of them share its language.
Compare that to the language's share of the corpus. The gap is the bias.

    same-language share in results
    ------------------------------  =  lift
    same-language share in corpus

lift ~1.0  -> the space is not sorting by language
lift  >1.0 -> your retrieval is biased toward the query's own language

    python3 diagnose_language_bias.py --collection my_docs --lang-field lang \
        --url http://localhost:6333 [--vector-name dense] [--sample 300]
"""
import argparse
import collections
import random

from qdrant_client import QdrantClient, models


def scroll_sample(client, collection, lang_field, sample, seed=0):
    """Grab a sample of points with their vectors and language."""
    got, offset = [], None
    while len(got) < sample * 8:
        pts, offset = client.scroll(
            collection, limit=min(1024, sample * 8 - len(got)), offset=offset,
            with_payload=[lang_field], with_vectors=True,
        )
        got.extend(pts)
        if offset is None:
            break
    random.Random(seed).shuffle(got)
    return got


def pick_vector(point, vector_name):
    v = point.vector
    if isinstance(v, dict):
        if vector_name is None:
            if len(v) != 1:
                raise SystemExit(
                    f"collection has named vectors {list(v)}; pass --vector-name")
            return next(iter(v.values())), next(iter(v))
        return v[vector_name], vector_name
    return v, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", required=True)
    ap.add_argument("--lang-field", default="lang")
    ap.add_argument("--url", default="http://localhost:6333")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--vector-name", default=None)
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("-k", type=int, default=10)
    args = ap.parse_args()

    client = QdrantClient(url=args.url, api_key=args.api_key)
    pts = scroll_sample(client, args.collection, args.lang_field, args.sample)
    pts = [p for p in pts if p.payload and p.payload.get(args.lang_field)]
    if not pts:
        raise SystemExit(f"no points carried payload field '{args.lang_field}'")

    corpus_share = collections.Counter(p.payload[args.lang_field] for p in pts)
    total = sum(corpus_share.values())

    per_lang = collections.defaultdict(list)
    for p in pts[:args.sample]:
        vec, name = pick_vector(p, args.vector_name)
        lang = p.payload[args.lang_field]
        res = client.query_points(
            args.collection, query=vec, using=name, limit=args.k + 1,
            with_payload=[args.lang_field],
        ).points
        neigh = [r for r in res if r.id != p.id][:args.k]
        if not neigh:
            continue
        same = sum(1 for r in neigh
                   if (r.payload or {}).get(args.lang_field) == lang)
        per_lang[lang].append(same / len(neigh))

    print(f"\ncollection: {args.collection}   sampled: {sum(len(v) for v in per_lang.values())}   k={args.k}\n")
    print(f"{'language':<12}{'corpus %':>10}{'in results %':>14}{'lift':>8}")
    print("-" * 44)
    overall_r, overall_c = [], []
    for lang, vals in sorted(per_lang.items(), key=lambda x: -len(x[1])):
        c = corpus_share[lang] / total
        r = sum(vals) / len(vals)
        overall_r.append(r * len(vals)); overall_c.append(c * len(vals))
        print(f"{lang:<12}{c*100:>9.1f}%{r*100:>13.1f}%{(r/c if c else 0):>8.2f}")
    n = sum(len(v) for v in per_lang.values())
    R, C = sum(overall_r) / n, sum(overall_c) / n
    print("-" * 44)
    print(f"{'OVERALL':<12}{C*100:>9.1f}%{R*100:>13.1f}%{(R/C if C else 0):>8.2f}\n")
    if C and R / C > 1.5:
        print(f"Your retrieval over-returns the query's own language by "
              f"{R/C:.1f}x. Cross-language answers are being crowded out.")
        print("SHIFT is worth testing: qdrant.tech/blog/shift-multilingual-rag/")
    else:
        print("No strong language clustering detected at this k.")


if __name__ == "__main__":
    main()
