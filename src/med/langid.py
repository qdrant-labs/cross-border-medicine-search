"""Which national registry is this box from?

The demo asks the user to pick the query language from a row of buttons. That is
a lie about the product -- someone holding a carton does not first tell the
search engine what country they are in -- and on stage it is a step where the
wrong button silently breaks the result. SHIFT subtracts a per-language offset,
so a Dutch box read while "polish" is selected gets the Polish correction
applied to a Dutch vector and lands nowhere near its own molecule. The language
has to be read off the box.

The method that won is embarrassingly simple. Registry text is "brand, INN,
dosage form", and the dosage form is effectively a language stamp -- "Tabletki
powlekane" is Polish the way "COMPRIMIDO" is Spanish. Each token votes in
proportion to how much more often it occurs in one language's documents than in
the others, so "ibuprofen" (all three registries) votes for nobody and
"powlekane" (one) votes decisively. Self-normalising, no stopword list, no
model, and it explains itself: the winning tokens can be shown on screen.

The interesting part is what lost. The obvious clever answer is to reuse the
embedding: score the text against each language's documents with NO shift
applied, since alpha=0 is precisely the configuration in which the languages
have not yet been merged, so the separation SHIFT exists to remove is what
identifies the language. It is a much better story and it is worse at the job.
Measured in eval_langid.py:

                       held-out registry    spanish carton OCR
  lexical                    0.999                0.987
  embedding (alpha=0)        0.958                0.947
  coverage-gated blend       0.998                0.960

Every blend threshold tested was worse than lexical alone, and the fallback was
never actually needed: across 150 carton photos the lexical vote never once ran
out of known tokens. So the embedding route is not in this module. It stays in
the eval as the evidence for why.

What replaces it is an abstain rule rather than a second signal. MARGIN_CONFIDENT
is set at the vote's lead over the runner-up, not its absolute score, because two
registries that both use "tablet" should read as uncertain however many tokens
agree. At 0.2 it catches every error in both tests -- the single registry miss
sat at 0.18 and both carton misses at 0.12 -- while flagging only 0.4% of clean
queries. Below it the caller should keep asking the human, which is what the
language picker is still there for.

The carton number is Spanish-only: the cached photo set is Spanish and there are
no Polish or Dutch packaging images here. It measures the false-positive rate on
one language and leaves the other two unmeasured. That belongs next to the
number anywhere it is quoted.
"""
import re
from collections import Counter

# Letters only, 3+ chars. Digits and dosages ("400", "mg") are identical in
# every registry and would only add noise to a vote that is already normalised.
TOKEN = re.compile(r"[^\W\d_]{3,}", re.UNICODE)

# Lead over the runner-up below which the detection should not be acted on
# without asking. Fitted in eval_langid.py, not chosen.
MARGIN_CONFIDENT = 0.2


def tokens(text):
    return TOKEN.findall(text.lower())


class LangID:
    """Lexical language vote, fitted on the corpus the search runs over."""

    def __init__(self, docs):
        self.langs = sorted({d["lang"] for d in docs})
        self.df = {l: Counter() for l in self.langs}
        self.n = Counter()
        for d in docs:
            self.n[d["lang"]] += 1
            for w in set(tokens(d["text"])):
                self.df[d["lang"]][w] += 1

    def __call__(self, text, top_evidence=6):
        toks = set(tokens(text))
        votes = Counter()
        evidence = []
        known = 0
        for w in toks:
            # Document frequency within each language as a rate, not a count,
            # or the largest registry would win every vote by size alone.
            rate = {l: self.df[l][w] / max(self.n[l], 1) for l in self.langs}
            tot = sum(rate.values())
            if tot <= 0:
                continue
            known += 1
            share = {l: rate[l] / tot for l in self.langs}
            for l in self.langs:
                votes[l] += share[l]
            best = max(share, key=share.get)
            evidence.append({"token": w, "lang": best,
                             "share": round(share[best], 3)})

        if not known:
            return {"lang": None, "scores": {}, "coverage": 0.0, "margin": 0.0,
                    "confident": False, "evidence": []}

        scores = {l: votes[l] / known for l in self.langs}
        ranked = sorted(scores, key=lambda l: -scores[l])
        margin = scores[ranked[0]] - (scores[ranked[1]] if len(ranked) > 1 else 0)
        # Sorted by how one-sided each token's vote was, so the explanation on
        # screen leads with the word that actually decided it.
        evidence.sort(key=lambda e: -e["share"])
        return {
            "lang": ranked[0],
            "scores": {l: round(scores[l], 3) for l in self.langs},
            "coverage": round(known / max(len(toks), 1), 3),
            "margin": round(margin, 3),
            "confident": margin >= MARGIN_CONFIDENT,
            "evidence": evidence[:top_evidence],
        }
