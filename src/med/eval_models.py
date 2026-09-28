"""Does SHIFT help every multilingual encoder, or only the one we happened to ship?

The SHIFT write-up reports multilingual-e5-large going from 0.633 to 0.737 nDCG@20
on the paper's benchmarks, with the cross-language recall moving most. That is one
model on someone else's data. This file runs the same intervention across five
encoders on *this* corpus -- 5,298 medicine records, three registries -- so the
demo can show a measured number next to the claim instead of borrowing one.

The comparison is only worth anything if each model is run the way its own card
says to run it, so the protocol is per-model rather than uniform:

  arctic-l-v2   "query: " on queries, nothing on documents. Two offsets.
  e5-large      "query: " / "passage: ". Two offsets -- the asymmetric prefix is
                the whole reason E5 needs them, and src/shift.py has carried that
                warning from the start.
  bekko a8m/a25m  no prefix anywhere. The card is explicit that bekko is trained
                prefix-free, and with queries and documents drawn from the same
                registry strings that makes the two spaces bit-identical, so
                there is exactly ONE offset to fit. eval_bekko.py established
                that bolting the arctic recipe on anyway scores higher (0.328 vs
                0.292); it is deliberately not done here, because this table is
                about what each model does when used as designed.
  jina-v5-nano  task adapters rather than text prefixes. Which mechanism actually
                engaged is recorded in the output as "protocol", because a silent
                fallback to plain encoding would look like a fair run and not be
                one.

Everything else is held fixed: same documents, same ATC-derived alignment pairs,
same alpha grid, and the same evaluate() that produces every other number in this
project. tgt_hits_per_query is the one to read -- recall_atc is near 1.0 for all
of these and hides the effect entirely, which is exactly the trap the SHIFT post
describes when it says the overall metric moves less than the cross-language one.

Licences and parameter counts are read off the loaded model and the hub card, not
typed in here, because the jina v5 family is CC-BY-NC-4.0 and that is the kind of
fact that should not rot quietly in a comment on a conference slide.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "artifacts" / "med"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shift_med import (ALPHAS, PIVOT, SHIFTED,  # noqa: E402
                       all_offsets, evaluate, load_corpus, shift_all)

BATCH = 64

# query/doc prompts are the model's own documented convention. `one_space` marks
# the prefix-free models, where the two encodings would be identical arrays and
# there is a single offset to fit rather than two.
MODELS = [
    {"key": "arctic-l-v2", "repo": "Snowflake/snowflake-arctic-embed-l-v2.0",
     "short": "arctic-embed-l-v2.0", "q": "query: ", "d": ""},
    {"key": "e5-large", "repo": "intfloat/multilingual-e5-large",
     "short": "multilingual-e5-large", "q": "query: ", "d": "passage: "},
    {"key": "bekko-a25m", "repo": "hotchpotch/bekko-embedding-v1-a25m",
     "short": "bekko-v1-a25m", "q": "", "d": "", "one_space": True},
    {"key": "bekko-a8m", "repo": "hotchpotch/bekko-embedding-v1-a8m",
     "short": "bekko-v1-a8m", "q": "", "d": "", "one_space": True},
    # jina separates the two things every other model here conflates: `task`
    # selects a LoRA adapter (and the model refuses to encode without one),
    # while the prompt names are ordinary text prefixes, "Query: "/"Document: ".
    # Both are read off the loaded model rather than assumed -- an earlier guess
    # at "retrieval.query" was wrong and would have silently produced untasked
    # embeddings labelled as retrieval ones.
    {"key": "jina-v5-nano", "repo": "jinaai/jina-embeddings-v5-text-nano",
     "short": "jina-v5-text-nano", "q": "", "d": "", "trust": True,
     "task": "retrieval", "prompts": ("query", "document")},
]


def device():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load(spec):
    from sentence_transformers import SentenceTransformer
    kw = {"device": device()}
    if spec.get("trust"):
        kw["trust_remote_code"] = True
    return SentenceTransformer(spec["repo"], **kw)


def encode(model, texts, spec, side):
    """Encode one side, returning (vectors, what_was_actually_used).

    The second value is not decoration. It is written into the table so the
    protocol each number was produced under is visible next to the number, and
    deliberately there is no fallback path: if a model advertises a retrieval
    adapter and the call fails, this raises and the model is recorded as failed.
    Quietly degrading to a generic embedding would produce a plausible row that
    is measuring something other than what its label claims.
    """
    if spec.get("task") or spec.get("prompts"):
        kw = {}
        if spec.get("task"):
            kw["task"] = spec["task"]
        if spec.get("prompts"):
            kw["prompt_name"] = spec["prompts"][0 if side == "q" else 1]
        v = model.encode(texts, batch_size=BATCH, normalize_embeddings=True,
                         show_progress_bar=False, **kw)
        return (np.asarray(v, dtype=np.float32),
                ", ".join(f"{k}={x}" for k, x in kw.items()))
    pre = spec[side]
    v = model.encode([pre + t for t in texts], batch_size=BATCH,
                     normalize_embeddings=True, show_progress_bar=False)
    return (np.asarray(v, dtype=np.float32),
            f'prefix "{pre}"' if pre else "no prefix")


def meta(spec, model):
    """Dimension and parameter counts off the loaded object; licence off the hub.

    Two counts, because the demo already prints "24.9M active" for bekko while
    the total is 123M and a second panel contradicting the first is worse than no
    panel. The difference is the token embedding matrix: bekko is a pruned mmBERT
    whose vocabulary dominates its parameter count but costs nothing per token at
    inference, so its card quotes the non-embedding figure. Arctic quotes the
    total. Both numbers are measured here and the UI can show the one it means.
    """
    import torch.nn as nn
    n = sum(p.numel() for p in model.parameters())
    emb = sum(m.weight.numel() for m in model.modules()
              if isinstance(m, nn.Embedding))
    lic = None
    try:
        from huggingface_hub import model_info
        lic = (model_info(spec["repo"]).cardData or {}).get("license")
    except Exception:
        pass
    # No dim here: jina's custom module returns None from
    # get_sentence_embedding_dimension() because its width depends on the task
    # adapter that has not been selected yet. Taken from the encoded array in
    # run() instead, which is the width actually used either way.
    return {"params": int(n), "params_m": round(n / 1e6, 1),
            "active_m": round((n - emb) / 1e6, 1), "license": lic}


def run(spec, docs, texts, langs, reuse=True):
    key = spec["key"]
    dpath, qpath = OUT / f"ms_{key}_d.npy", OUT / f"ms_{key}_q.npy"
    info = OUT / f"ms_{key}_meta.json"

    if reuse and dpath.exists() and qpath.exists() and info.exists():
        print(f"  reusing cached embeddings for {key}")
        dv, qv = np.load(dpath), np.load(qpath)
        m = json.loads(info.read_text())
    else:
        t0 = time.time()
        print(f"  loading {spec['repo']} ...", flush=True)
        model = load(spec)
        m = meta(spec, model)
        print(f"  {m['params_m']}M params, licence {m['license']}", flush=True)
        dv, dnote = encode(model, texts, spec, "d")
        m["dim"] = int(dv.shape[1])
        if spec.get("one_space"):
            # Prefix-free and both sides are the same strings, so the query
            # encoding IS the document encoding. Calling encode twice would burn
            # a minute to produce a bit-identical array.
            qv, qnote = dv, dnote
        else:
            qv, qnote = encode(model, texts, spec, "q")
        m["protocol"] = (f"doc: {dnote}; query: {qnote}"
                         + ("; one shared space" if spec.get("one_space")
                            else "; separate offsets"))
        m["encode_s"] = round(time.time() - t0, 1)
        np.save(dpath, dv)
        np.save(qpath, qv)
        info.write_text(json.dumps(m, indent=1))
        del model
        print(f"  encoded in {m['encode_s']}s -- {m['protocol']}", flush=True)

    one = bool(spec.get("one_space"))
    off_d = all_offsets(dv, docs)
    off_q = off_d if one else all_offsets(qv, docs)

    sweep = {}
    for a in ALPHAS:
        dv_s = shift_all(dv, off_d, a, langs)
        qv_s = shift_all(qv, off_q, a, langs)
        sweep[str(a)] = {ql: evaluate(qv_s, dv_s, docs, qlang=ql, target=PIVOT)
                         for ql in SHIFTED}

    # Best alpha by mean cross-border hits over the non-pivot languages. Chosen
    # per model rather than fixing 0.75 from arctic: the whole question is
    # whether the setting transfers, and hard-coding it would answer it by fiat.
    def score(a):
        r = sweep[str(a)]
        return sum(r[l]["tgt_hits_per_query"] for l in SHIFTED) / len(SHIFTED)

    best = max(ALPHAS, key=score)
    m.update({"key": key, "repo": spec["repo"], "short": spec["short"],
              "best_alpha": best, "sweep": sweep})
    for l in SHIFTED:
        off = sweep["0.0"][l]["tgt_hits_per_query"]
        on = sweep[str(best)][l]["tgt_hits_per_query"]
        print(f"  {l}->{PIVOT}  off={off:.3f}  on(a={best})={on:.3f}  "
              f"{on - off:+.3f}   recall {sweep['0.0'][l]['recall_atc']:.4f}"
              f" -> {sweep[str(best)][l]['recall_atc']:.4f}")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="model keys to run")
    ap.add_argument("--fresh", action="store_true", help="ignore cached embeddings")
    args = ap.parse_args()

    np.seterr(all="ignore")
    pairs, docs = load_corpus()
    texts = [d["text"] for d in docs]
    langs = np.array([d["lang"] for d in docs])
    print(f"{len(docs)} documents, {len(pairs)} ATC codes, pivot={PIVOT}\n")

    dest = OUT / "model_shift.json"
    prev = json.loads(dest.read_text()) if dest.exists() else {}
    rows = {r["key"]: r for r in prev.get("models", [])}

    todo = [s for s in MODELS if not args.only or s["key"] in args.only]
    for spec in todo:
        print(f"{spec['key']}")
        try:
            rows[spec["key"]] = run(spec, docs, texts, langs, reuse=not args.fresh)
        except Exception as e:
            # One model failing to download or to expose its adapter should not
            # discard the others' results, which take minutes each to produce.
            print(f"  FAILED: {type(e).__name__}: {e}")
        print()

    out = {"corpus": {"n_docs": len(docs), "n_atc": len(pairs), "pivot": PIVOT,
                      "langs": list(SHIFTED)},
           "alphas": list(ALPHAS),
           "models": [rows[s["key"]] for s in MODELS if s["key"] in rows]}
    dest.write_text(json.dumps(out, indent=1))

    print(f"{'model':<22}{'dim':>6}{'params':>9}{'a*':>6}"
          + "".join(f"{l + ' off':>9}{l + ' on':>9}{'delta':>8}" for l in SHIFTED))
    for r in out["models"]:
        line = (f"{r['short']:<22}{r['dim']:>6}{r['params_m']:>8.1f}M"
                f"{r['best_alpha']:>6}")
        for l in SHIFTED:
            o = r["sweep"]["0.0"][l]["tgt_hits_per_query"]
            n = r["sweep"][str(r["best_alpha"])][l]["tgt_hits_per_query"]
            line += f"{o:>9.3f}{n:>9.3f}{n - o:>+8.3f}"
        print(line)
    print(f"\nwrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
