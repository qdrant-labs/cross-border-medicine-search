# Personalization at Agent Scale — Barcelona

Two live demos for a 20-minute Qdrant talk, in one repo.

| | demo | what it shows | run it |
|---|---|---|---|
| **1** | [Cross-border medicine lookup](#demo-1--cross-border-medicine-lookup) | point a camera at a box, get the equivalent medicine in another country | `python3 -m uvicorn src.med.serve_med:app --port 8079` → `/med` |
| **2** | [The geometry of retrieval](#demo-2--the-geometry-of-retrieval) | SHIFT + hyperbolic embeddings on the Google Product Taxonomy | `python3 run.py` → `:8077` |

Demo 1 is the talk. **Demo 2 is the fallback** and is deliberately untouched by
demo 1 — separate port, separate Qdrant collection, no shared imports — so a
camera failure on stage costs a tab switch rather than the session.

There is also [`site/`](#the-companion-page-site), a dependency-free static page
of the measured results, meant for Vercel so the audience has a link that keeps
working after the laptop closes.

![Cross-border medicine lookup: a Spanish carton scanned on the left, SHIFT off
against SHIFT on in the middle, and the encoder sweep and Matryoshka controls on
the right](docs/demo.png)

Demo 1 mid-query, from a cached carton. Left: the still, the language the
detector read off the box and the tokens that decided it. Middle: the same query
answered with the language offset subtracted and without it, then the alpha sweep
across five encoders. Right: the coarse-to-fine controls, which act on the two
panels in the middle.

---

## Demo 1 — Cross-border medicine lookup

Show a medicine box to the camera. The page OCRs the carton on-device, works out
which country's registry the text came from, and returns the equivalent product
in another country — joined on **ATC code**, never on brand name, because brand
names collide across borders and sometimes carry different active ingredients.

Three registries, all open government data: Poland (RPL), Spain (CIMA AEMPS),
Netherlands (CBG-MEB). **5,298 products, 220 ATC codes, 1,898 packaging photos.**

```bash
python3 src/med/build_corpus.py      # join the registries on ATC
python3 src/med/shift_med.py         # embed + fit offsets
python3 src/med/train_atc.py         # Poincare coords for the ATC tree
python3 src/med/fetch_photos.py      # CIMA packaging shots (optional)
python3 -m uvicorn src.med.serve_med:app --host 127.0.0.1 --port 8079
```

Then open `http://127.0.0.1:8079/med`.

### What it demonstrates

**Language is detected, not selected.** A token-frequency vote over the three
registries, where each token votes in proportion to its document-frequency
*rate* rather than its count — otherwise the largest registry wins every vote by
size alone. No stopword list, no model, and it shows its evidence on screen.

| test | accuracy | majority baseline |
|---|---:|---:|
| registry text, held out (n=1,200) | **0.999** | 0.333 |
| carton OCR, Spanish packaging (n=150) | **0.987** | 1.000 |

**It abstains rather than guess.** The rule is the *margin* over the runner-up,
not the absolute score, and 0.2 was fitted rather than chosen: every error in
both tests sits below it. At `margin >= 0.2` the detector keeps 99.6% / 98.7% of
queries at accuracy **1.0000**. When it abstains it falls back to the pivot
language, which by construction has no offset — so uncertainty degrades to plain
unshifted retrieval instead of a confident correction in the wrong direction.

`NUROFEN` on its own abstains (margin 0.181, pl 0.59 vs nl 0.41) because it is
genuinely sold in both registries. `Nurofen voor kinderen` commits to Dutch at
0.606.

**It says which product it thinks it is, and how sure.** A box is drawn over
the carton around the text that produced the identification, labelled with the
brand and a score. Two numbers, because they fail independently: `ocr` is
Vision's confidence in the characters, and `p` is the leader's share of a
softmax over the shortlist — how much daylight there is to the next candidate.
A crisp reading of a product that is not in the registry gives a high `ocr` and
a thin `p`; a blurred reading of an unambiguous box gives the reverse.

Identification runs against the packaging's **own** registry, not the target
one. "Which Polish box is this" is a different question from "what is its
Spanish equivalent", and only the first one can be answered by looking at the
carton. Below `p = 0.55` the box turns amber and names the runner-up instead of
committing. The usual reason is two strengths of one brand — a carton reading
only `VFEND®` scores 0.49 against VFEND 50 mg and VFEND 200 mg, which is a real
ambiguity on the box rather than a retrieval failure, so the label asks *which
strength?* rather than guessing.

`p` is a relative confidence over the shortlist and never a calibrated
probability: it cannot know the right product is absent from the registry
entirely, and in that case it will still hand the leader a high share. That is
why the runner-up is on screen next to it.

**Scan memory** sits under the camera: every distinct carton seen this session,
with a crop of the brand region, its ATC code, and which registry it was read
from. Unique by brand-and-registry rather than by ATC — two brands of one
molecule are two different boxes off the shelf, and collapsing them would hide
the very join the rest of the page argues for. Repeat scans increment a `×n`
counter. A language shown as `es?` means the detector abstained and fell back to
the pivot, so the registry is a default rather than a reading; the strip keeps
that distinction rather than presenting a guess as evidence. State is
client-side in `localStorage` — it is a property of this session, not of the
corpus, and it survives a mistimed reload on stage.

**Result photos are never borrowed.** A row shows a packaging shot only when the
shot belongs to that exact registration number; otherwise it shows a `no photo`
placeholder. An earlier cut fell back to any photo sharing the row's ATC code,
which looked like better coverage and was in fact the one thing this demo must
not do: same ATC means same *molecule*, not same product. Because every photo in
the corpus comes from the Spanish registry (CIMA) and the Polish and Dutch
registries publish no registration number to match on, that fallback fired on
all 4,050 non-Spanish records — a Polish `Apap` row illustrated with a Spanish
ANTIDOL carton. The placeholder is the honest answer, and its absence is itself
the point: the join is on the code, never on the picture.

**The two geometry panels are a viewer, not a chart.** Hover reads a node, click
traces its chain to the root, scroll zooms about the cursor, drag pans, and the
search box finds a node by ATC prefix or substance name and centres it in both
panels at once. The ⤢ button expands either disk to full screen — 534 nodes in a
300px sidebar is a thumbnail, and at that size adjacent level-5 codes are a pixel
apart. The two panels share a selection but keep independent zoom, so the same
chain can be examined at different scales side by side. In the Poincaré panel the
chain follows the true geodesic and in the Euclidean panel it is drawn straight,
because in that space a straight line *is* the geodesic: the visual difference
between the panels is the honest one, not a stylistic choice.

**Four retrievers, same 300 Polish queries against 1,248 Spanish documents:**

| config | top-1 | top-3 | top-10 | |
|---|---:|---:|---:|---|
| dense + SHIFT | 0.827 | 0.913 | 0.953 | the shipped path |
| bm25 | 0.230 | 0.260 | 0.307 | 160/300 queries share no term at all |
| miniCOIL | 0.180 | 0.207 | 0.253 | 168/300 share no term |
| SPLADE | 0.770 | 0.833 | 0.930 | 0/300 share no term |
| dense + bm25 (RRF) | 0.340 | 0.473 | 0.807 | fusion far below dense |
| **dense + SPLADE (RRF)** | **0.837** | **0.920** | **0.970** | best measured |

Exact matching fails the way the vocabulary-gap argument predicts — `Apap` and
`ANTIDOL` are the same molecule and share no characters. Term expansion does
not: SPLADE finds a shared term on every query and fusing it with dense wins at
every cut, **despite being an English-trained model on Spanish and Polish
registry strings**. Nothing sparse ships anyway, and that is now a cost decision
rather than an accuracy one: +0.010 at top-1 does not pay for a second index and
a 500 MB model in a demo. "We measured it, it won, and we still said no" is a
more honest engineering story than "sparse lost".

**Matryoshka:** Arctic v2.0 truncated to 256 dims retains **105%** of its
1024-dim accuracy. A second MRL-trained model, bekko-v1-a25m, retains **103%**
at a twentieth of the parameters. An e5 control with no MRL training drops to
86% — the same operation, and it only works because the model was trained for
it.

The tempting explanation for scoring *above* 100% is that truncation is
stripping the language signal, doing SHIFT's job for it. Measured rather than
assumed (`src/med/matryoshka.py`): dropping bekko 384→64 dims lifts cross-border
hits from 0.090 to 0.136 with SHIFT off, so that effect is real but small.
SHIFT's own contribution is +0.229 at 384 and +0.209 at 64 — essentially
unchanged across a 6× reduction. **The two are independent.** A 256-byte vector
still needs its offset.

**Two offsets beat one, and not for the reason the code claimed.** `src/shift.py`
warns that E5 needs separate offsets per prefix, and `shift_med.py` fits Arctic
the same way. `src/med/eval_bekko.py` tests *why*, using a prefix-free model
where query and document space are bit-identical and exactly one offset exists
to fit. The hypothesis — that the second offset is a workaround for a prefix
artifact — **was rejected**:

| configuration | cross-border hits, pl→es, α=0.75 |
|---|---|
| `plain` — no prefix, one offset | 0.292 |
| `prefix_both` — prefix on both sides, one offset | 0.227 |
| `prefix_query` — Arctic's recipe, two offsets | **0.328** |

`prefix_both` is the control that separates the two effects, and it says the
prefix genuinely costs 0.065 — mean pooling averages six tokens of boilerplate
in alongside the drug name. Yet the asymmetric two-offset arrangement still wins
by 0.036 *while paying* that cost, so gross it is worth about +0.10. The second
offset is capturing query/document **asymmetry**, not correcting for the prefix.
The prefix is merely the cheapest way to create asymmetry in a corpus where
queries and documents are drawn from the same registry strings. A deployment
whose queries are genuinely different text — OCR off a carton — gets that
asymmetry for free. The warning survives; its stated reason changes.

**Hyperbolic rerank:** a 10-dimensional Poincaré embedding of the ATC tree,
MAP 0.8314 against 0.7369 for Euclidean at the same dimension. It adds 0.008 on
packaging shots and **0.080** on near-textless tablet blisters — the harder case,
where knowing the drug-classification tree substitutes for having read anything.

### Turning the server off

The companion page ([`site/`](#the-companion-page-site)) runs the same query with
no server in the loop. The whole Spanish shelf — 1,248 products at 256 dims in
fp16, 512 bytes a product — is **624 KB of vectors plus 299 KB of metadata**,
exported by `src/med/export_browser.py` into `site/data/`. The browser fetches it
once, encodes the query with bekko's ONNX build, applies the same α=0.75 SHIFT
offset, and scans all 1,248 — encode included — in **66–156 ms warm** (four
consecutive runs on one laptop; the code asks for WebGPU and falls back to WASM
on Safari and Firefox, and the measurement does not record which it got). The
first click is slower: it downloads the fp32 ONNX export, 199 MB, once.

This was also a panel in `web/med.html`, side by side with the server's answer so
the comparison was visible rather than asserted. That panel is gone — cut to keep
the talk inside 20 minutes. The numbers below are unaffected: they were measured
against the same export the static page still loads, not read off the panel.

Three things that comparison is there to make honest:

- **It loses.** Arctic is 568M parameters against bekko's 24.9M active, and wins
  outright on this corpus: 0.550 cross-border hits against 0.328 at the same α.
  On the Polish metformin query the on-device list puts vildagliptin combos
  above plain metformin and surfaces an ibuprofen at rank 5.
- **ONNX is not the weights every other number came from.** A query vector from
  the ONNX export and from PyTorch agree at cosine 0.961, and over a 120-query
  sample the top-1 document differs on **11%**. The page says so rather than
  implying parity.
- **fp16 is not the cost.** Against fp32 the same 10 documents come back 99.93%
  of the time and `recall_atc` is identical to four decimals (0.9229). The
  exact-order figure is 96.69%, which is adjacent-rank tie-breaking at cosine
  differences below fp16 resolution — not a retrieval loss.

No photos on the on-device side either; those are served by the process this
exists to switch off.

### Demo queries that actually show SHIFT working

Verified end to end, at **both** 1024 and 256 dims, so they hold wherever the
slider is left. Note `Apap, Paracetamolum, Tabletki powlekane` finds ANTIDOL
*with SHIFT off as well*, so it makes a poor stage example despite being the
cleanest illustration of the vocabulary gap.

| query (Polish) | SHIFT off | SHIFT on |
|---|---|---|
| `Glucophage XR, Metformini hydrochloridum, Tabletki o przedluzonym uwalnianiu` | N06AX16, an antidepressant | metformin ✓ |
| `Metformax 500, Metformini hydrochloridum, Tabletki` | metformin **+ sitagliptin** combo (1024) / **metamizole**, a painkiller (256) | metformin ✓ |
| `Cialis, Tadalafilum, Tabletki powlekane` | G04BE03 sildenafil — wrong molecule | tadalafil ✓ |

Glucophage is the strongest: SHIFT off returns a psychiatric drug for a diabetes
query. Dutch equivalents that flip at 256 dims:
`Neoclarityn 0,5 mg/ml, drank, Desloratadine, Drank` and
`Diclofenacnatrium Auro 100 mg, zetpillen`.

Avoid `Quetiapine Viatris 150 mg, filmomhulde tabletten` — it misses at both
dims. `Controloc` and `Noacid` succeed with SHIFT off at 1024, so they show
nothing.

### What was rejected, and why that is in the repo

**ColBERT / late interaction.** Rejected three times over: on clean registry
text (0.880 vs 0.953 for dense+hyperbolic), on OCR text (0.972 vs 0.984), and on
a basic discrimination check. The mechanism matters more than the verdict —
MaxSim tolerates noise by letting junk query tokens take their best available
match, but on a medicine carton the junk is not random, it is *other Spanish
pharmaceutical text*: dosage instructions, the authorisation holder, warning
boilerplate. Those tokens match real document tokens strongly and confidently in
the wrong direction, which is the one case the mechanism cannot absorb. Pooling
is what dilutes them. Here averaging is the feature, not the flaw.

**Germany and Portugal.** Investigated and dropped, recorded in
`src/med/build_corpus.py` so the search does not get repeated. Germany's AMIce
carries ATC but its terms permit machine-readable storage only transiently, so a
derived file cannot be redistributed; the one genuinely open BfArM product file
is 415 bytes of counts with no products in it; and the email-gated
Referenzdatenbank has no ATC field at all. Portugal's INFOMED publishes no bulk
export and no public API.

### Caveats to keep attached

1. **Carton detection is measured on Spanish only.** All 1,898 cached photos come
   from CIMA. Polish and Dutch packaging is unmeasured — scanning a Polish box
   live is a genuine first run. The same caveat applies to the brand box and to
   the scan-memory strip: every cached carton is Spanish, so the strip will read
   `spanish N` until a real Polish or Dutch box is held up to the camera. The
   same single-registry origin is why Polish and Dutch result rows show `no
   photo` — the corpus has none to show, and inventing one from a shared ATC
   code is the failure described above.
2. **BM25 runs unstemmed on the Polish side.** fastembed ships Snowball stemmers
   for 18 languages and Polish is not one of them; Spanish and Dutch are. Polish
   is heavily inflected, so this understates BM25 for Polish specifically.
3. **SPLADE is English-trained.** `Splade_PP_en_v1` on Spanish and Polish
   registry strings is out of domain. There is no multilingual SPLADE in
   fastembed, so this is the honest ceiling — do not let SPLADE look beaten
   fairly.
4. **OCR is macOS-only** by design. Apple's Vision framework runs on-device, so
   the camera path works with the wifi off.

### Layout

```
src/med/build_corpus.py     join PL/ES/NL registries on ATC
src/med/shift_med.py        embeddings, per-language offsets, alpha sweep
src/med/langid.py           token-frequency language vote with abstain
src/med/vision_ocr.py       Apple Vision OCR, on-device (macOS)
src/med/train_atc.py        Poincare + Euclidean embeddings of the ATC tree
src/med/build_sparse.py     second collection for the retriever comparison
src/med/serve_med.py        FastAPI, port 8079
src/med/eval_*.py           every number above, reproducible
web/med.html                camera UI, zero dependencies
```

Each `eval_*.py` carries its result and its interpretation in the module
docstring, including the ones that argue against the code they sit next to.

---

## Demo 2 — The geometry of retrieval

Two ideas, one dataset, one index:

1. **SHIFT** — remove language bias from a multilingual embedding space with a
   vector subtraction. (`qdrant.tech/blog/shift-multilingual-rag/`)
2. **Hyperbolic embeddings** — put a hierarchy in a space whose geometry
   actually has room for it, and serve it from Qdrant.
   (`qdrant.tech/articles/hyperbolic-embeddings-qdrant/`)

The dataset is the **Google Product Taxonomy**. It is the reason the two halves
fit together in one talk: Google publishes it in 9 languages with *identical
category IDs*, so the same file gives you

- a genuinely parallel multilingual corpus with exact cross-lingual ground truth
  (a result is relevant iff the ID matches), **and**
- a 7-level, 5,595-node hierarchy for the hyperbolic half.

No synthetic data, no hand-labelled relevance judgements.


## Quickstart

```bash
pip install -r requirements.txt
python3 run.py            # -> http://127.0.0.1:8077
```

First run embeds the corpus and builds both collections (a few minutes on an
M-series Mac via MPS). After that, startup is seconds.

Runs against **Qdrant local mode** by default — no Docker needed, which matters
when conference wifi is the enemy. To point at a real server:

```bash
QDRANT_URL=http://localhost:6333 python3 run.py
```

The UI reports which mode it is in.


## What's here

```
src/taxonomy.py               parse the taxonomy into a tree
src/poincare.py               train Poincare + Euclidean embeddings (shared loss/optimizer)
src/shift.py                  parallel corpus, offsets, cross-lingual eval
src/hyperbolic_index.py       Qdrant collection + Formula Query rescoring
src/diagnose_language_bias.py standalone: measure language bias in YOUR collection
src/server.py                 FastAPI, 6 endpoints
web/index.html                three-tab UI, zero dependencies
```

Press **space** anywhere in the UI to run the whole demo hands-free: a 13-step
scripted reel that drives the sliders, switches tabs and captions itself. Press
space again to stop. If the wifi dies or your hands shake, the reel still runs.


## Reproduced numbers

Everything below was measured on this machine, not copied from the articles.

### Hierarchy: Poincare vs Euclidean (MAP, ancestor task)

Identical data, loss, optimizer, epochs. **Only the geometry differs.**

| dims | Euclidean (here / article) | Poincare (here / article) |
|-----:|---------------------------:|--------------------------:|
| 2    | 0.158 / 0.140 | 0.548 / 0.501 |
| 5    | 0.247 / 0.239 | **0.911 / 0.905** |
| 10   | 0.395 / 0.354 | 0.930 / 0.925 |
| 20   | 0.674 / 0.551 | 0.932 / 0.932 |
| 50   | 0.802 / 0.658 | 0.935 / 0.934 |

**5 hyperbolic dimensions beat 50 Euclidean ones.** That is the headline.

**The headline is directional, and the direction is not stated in the metric's
name.** MAP on the ancestor task scores exactly one question: given a node, are
its *ancestors* ranked highly? Run the same pairs backwards and the result
inverts (`src/med/eval_hierarchy_direction.py`, dim 10, 534 nodes):

| task | Poincare | Euclidean | delta |
|---|---:|---:|---:|
| MAP up (child → ancestors) | **0.8314** | 0.7369 | +0.0945 |
| MAP down (node → descendants) | 0.5911 | **0.7220** | −0.1309 |
| kin@6, symmetric | 0.8168 | 0.8118 | +0.0050 |
| kin@10, symmetric | 0.5654 | **0.5852** | −0.0199 |

`kin@k` asks the label-free version — of the k nearest nodes, how many are
taxonomic relatives at all — and is what the demo's neighbour lists actually
display. At k=6 it is a tie. The famous gap lives entirely in the *up*
direction.

Per-hub, the descendant result is not close:

| hub | descendants | Poincare | Euclidean |
|---|---:|---:|---:|
| N nervous system | 96 | 0.5179 | **1.0000** |
| A alimentary | 81 | 0.3544 | **1.0000** |
| C cardiovascular | 71 | 0.3709 | **1.0000** |
| R respiratory | 54 | 0.5497 | **1.0000** |
| J anti-infectives | 46 | 0.6578 | **1.0000** |

This is the space working as designed, not a bad fit, and the per-hub numbers
are what confirm it: the gap grows monotonically with fan-out (−0.342 at 46
descendants, −0.646 at 81). A child has one parent sitting inward of it, which
is trivial to rank first. A parent with 96 children *cannot* be nearest to all
96 — the circumference available at its radius is bounded while the number of
descendants is not. Hyperbolic space buys room by pushing outward, so looking up
is cheap and looking down is structurally expensive. Euclidean space has no such
constraint near the origin, which is why it scores a perfect 1.0.

So the defensible claim is "hyperbolic wins at generalisation — one direction,
on a hierarchy", not "hyperbolic wins". The prediction was written into the
module docstring before the script was run.

### Serving: prefetch + rescore (recall@10 vs exact hyperbolic)

Poincare coords stored as ordinary vectors, `sq_norm` in the payload, geodesic
distance reconstructed in a Formula Query at rescore time.

| prefetch | Euclidean only | with hyperbolic rescore |
|---------:|---------------:|------------------------:|
| 10   | 0.336 | 0.336 |
| 50   | 0.336 | 0.532 |
| 100  | 0.336 | 0.618 |
| 300  | 0.336 | 0.759 |
| 1000 | 0.336 | **0.868** |

Naive ANN on the converted vectors is not an option: the norms span ~600x, so
HNSW built directly on hyperbolic distance collapses to recall@10 ≈ 0.020.

Formula Query has no `acosh`, so it is rebuilt from `ln` and `sqrt`:

```
acosh(x) = ln(x + sqrt(x^2 - 1))
```

Max |Qdrant score − exact numpy distance| = **4.28e-07** (article: 4.7e-07).

### Language bias: SHIFT

Offsets fitted on a disjoint split (2,200 categories), evaluated on the rest.

| alpha | recall@10 | same-lang | cross-lang | % of top-10 in query language |
|------:|----------:|----------:|-----------:|------------------------------:|
| 0.00  | 0.230 | 0.992 | 0.134 | **61.5%** |
| 1.00  | 0.295 | 0.991 | 0.208 | **38.3%** |

61.5% independently reproduces the article's 61% figure.

Cross-lingual recall, alpha 0 → 1, by query language:

| language | 0 → 1 |
|---|---|
| Polish | 0.048 → 0.142 (**3x**) |
| Japanese | 0.075 → 0.149 |
| Spanish | 0.178 → 0.239 |
| English | 0.207 → 0.305 |

The lower-resource the language, the more it gains. Offsets estimated on two
disjoint halves of the corpus agree at **mean cosine 0.9955** — the offset is a
property of the model, not of the sample.

### Can you change the geometry of an embedding you already have?

Asked of arctic-embed-l-v2.0, the model the medicine demo retrieves with, since
the Poincare panel sitting next to it invites exactly the wrong conclusion.
`src/med/eval_offset_geometry.py`. Ground truth is the ATC code throughout and
nothing in the ranking path reads it.

"Change the geometry" turns out to be three different asks, with three
different answers.

**1 — The metric. Free, and it changes nothing.** Swapping cosine for
Euclidean, angular, or Poincare distance on the stored vectors leaves the
ranking **bit-identical**: 0 positions differ at top-1, top-3 or top-10, at
every curvature from c=0.25 to c=4. The vectors are unit-norm, so
‖u−v‖² = 2 − 2⟨u,v⟩ and every one of these distances is a monotone function of
the dot product — they *cannot* disagree about order. Computed independently in
float64 rather than derived from one another, so "identical" is a measurement
and not a restatement of the algebra. If a switch of metric on normalised
vectors reportedly improved recall, something else changed.

**2 — The container. Possible, and it loses.** On the stored vectors it is
degenerate: the exp-map at the origin sends every unit vector to the same
radius (0.761594, spread **0.0**). The normalisation layer already destroyed
the coordinate hyperbolic geometry needs.

The escape hatch is to drop that layer and keep the raw norms. Those norms are
*not* flat — norm tracks ATC depth even after regressing out string length
(partial ρ = **+0.42**, η² = 0.15, n=534). **This refuted my prediction**, which
was that they would be uninformative. So the question had to be settled
operationally rather than by correlation:

| retrieval | PL→ES hit@3 | NL→ES hit@3 |
|---|---:|---:|
| cosine, normalised | **0.922** | **0.955** |
| Poincare, scale 0.5 | 0.900 | 0.934 |
| Poincare, scale 1.0 | 0.855 | 0.910 |
| Poincare, scale 2.0 | 0.662 | 0.761 |

Monotonically worse. The radial variance is real, but it encodes *how much text
was embedded*, not where the molecule sits in the tree — and curvature
amplifies whatever is in the radius. Tier 2 closes for a better reason than
predicted.

**3 — The objective. The only real answer.** Hyperbolic structure has to be
trained in, which is what the ATC panel is: same tree, same 10 dimensions, same
budget, hyperbolic loss instead of Euclidean. That is a different model, not a
different view of this one.

Practical upshot: use the tree embedding for the **taxonomy** and the
normalised multilingual embedding for the **text**. They are not substitutes,
and the demo runs both.

**And SHIFT itself is a geometric operation.** Subtract-then-renormalise is a
flat approximation to moving along the sphere. Doing it properly — project the
offset into the tangent space at each point, then exponential-map — displaces
the vectors by a mean of **0.005** at α=0.75 and moves hit@3 by **±0.001** in
either direction. At this offset size the curved operator and the flat one are
the same operator, so the flat one stays, because it is two lines. What
actually moves the number is α:

| | α=0 | α=0.5 | α=0.75 | α=1.0 |
|---|---:|---:|---:|---:|
| PL→ES hit@3 | 0.922 | **0.941** | 0.940 | 0.935 |
| NL→ES hit@3 | 0.955 | 0.969 | **0.970** | 0.966 |

Both peak in the middle, not at α=1 — the offset is worth applying but not
worth applying fully.


### Radius learns generality, not depth (tab 3)

The Poincare model never sees a depth label — only `is-an-ancestor-of` links. So
its radius is what the *link structure* implies about how general a category is.
Residual below is a robust z-score (median/MAD) computed **within** each depth,
i.e. "odd for its level" rather than "deep".

| signal | value |
|---|---|
| Spearman(radius, depth) | **+0.602** |
| Spearman(residual, #children) | **−0.296** |
| mean residual, leaf nodes (n=4,719) | **+0.21** |
| mean residual, hub nodes (n=876) | **−2.34** |

Flagged *too general for its level* — every one a hub:
`Kitchen Tools & Utensils` (depth 2, **77 children**, z = −8.4),
`Fresh & Frozen Vegetables` (depth 3, **58 children**, z = −9.3).

Flagged *too specific for its level* — every one childless:
`Water > Distilled Water` (depth 3, z = +4.5),
`Telephony > Satellite Phones` (depth 3, z = +4.2).

Use case: **hierarchy QA.** Point it at a hand-maintained category tree and get a
ranked list of nodes to re-file, with no labels and no eval set.

⚠️ −0.296 is a moderate correlation — a triage signal, not an oracle. The strong,
defensible number is the leaf/hub separation (+0.21 vs −2.34). Lead with that.


## Honest deltas vs the published articles

Worth knowing before anyone in the audience checks.

1. **Local mode does exact prefetch.** The prefetch sweep above is therefore the
   *offline upper bound*. A real server with HNSW will land slightly lower. The
   article says the same; don't claim the number is a production measurement.

2. **Same-language recall does not degrade here** (0.992 → 0.991). In the
   article it does. The reason is that the same-language task in this corpus is
   easier, so the crowding-out effect surfaces in the *share* metric
   (61.5% → 38.3%) instead of the recall metric. Same phenomenon, different
   place on the scoreboard. Show the share number.

3. **Euclidean MAP at 20 and 50 dims is higher here** than in the article
   (0.674 vs 0.551, 0.802 vs 0.658). Tuning difference. It makes the comparison
   *more* conservative, not less — the Poincare win at 5 dims still holds.

4. **Direct-parent MAP is 0.644 here vs 0.539** in the article, on the harder
   parent-only eval.

5. **Catalan (`ca-ES`) is not published.** Google's taxonomy 404s for it. This
   is real, not a prop — and it is the single most on-topic fact available for a
   Barcelona audience. Don't fabricate a Catalan file to fill the gap; the
   absence *is* the point about language-resource asymmetry.


## Demo beats (20 minutes)

| min | beat | what's on screen |
|----:|------|------------------|
| 0–2 | Two failure modes of one embedding space | title |
| 2–3 | Gaudi aside (20s, see caveat below) | Sagrada Familia columns |
| 3–8 | **SHIFT live**: `calzado`, alpha 0 → 1, language lanes rebalance | tab 1 |
| 8–9 | Catalan lane is empty at every alpha | tab 1 |
| 9–10 | Run `diagnose_language_bias.py` on a collection | terminal |
| 10–15 | **Hyperbolic live**: espresso pots, prefetch slider, radius profile | tab 2 |
| 15–16 | 5 dims > 50 dims; the acosh trick | results table |
| 16–18 | **Radius finds bad data**: hubs filed too deep | tab 3 |
| 18–20 | What to take back to work Monday | three artifacts |

### Demo queries that land

**SHIFT tab.** `calzado` (es-ES). At alpha 0 you get 9/10 Spanish results.
At alpha 1 you get 4/10 Spanish and 4 distinct languages — same index, same
query, one slider.

**Hyperbolic tab.** Category **1647**, *Electric & Stovetop Espresso Pots*
(r = 9.57). Best single screen in the talk:

- **hyperbolic** returns r = 4.3, 2.51, 7.2, 0.78 — it walks *inward*, up the
  ancestor chain
- **euclidean** returns r = 9.4, 9.39, 9.51, 9.94 — siblings
- **text embedding** returns r = 9.94, 9.77, 9.39 — also siblings

Three panels, one query. Hyperbolic search retrieves *generalizations*; the
other two retrieve *things at the same level*. Watch the `r` column.

### Why radius is the whole story

Euclidean norms all saturate at ≈1.0000 near the ball boundary and tell you
nothing. The **hyperbolic radius** `2·artanh(||x||)` keeps separating levels all
the way out, and it stratifies monotonically by depth (median radius 3.12 at
depth 0 rising to 10.28 at depth 6). It is not a display trick — it is the
quantity that encodes specificity. The disk visual and the `r` column both use it.

### Two geometry bugs that were in the first cut

Worth knowing, because both are easy to reproduce in any hyperbolic viewer:

1. **The disk drew straight lines between ancestors.** Geodesics in the Poincare
   disk are circular arcs meeting the boundary at right angles. The panel built
   to teach hyperbolic geometry was rendering Euclidean geometry. Now traced
   with Mobius operations: `γ(t) = u ⊕ (t ⊗ ((−u) ⊕ v))`.

2. **Display coordinates are radially warped** (radius remapped to normalized
   hyperbolic radius, so the tree doesn't pile onto the rim). That means you
   cannot draw a textbook geodesic arc through the *displayed* points and call
   it correct. The fix: sample the geodesic in **true** coordinates, then push
   every sample through the same warp. The drawn curve is the honest image of
   the geodesic.

Note the ancestor chain still looks nearly straight after the fix — **correctly
so**, because radial lines through the origin *are* geodesics, so a path straight
up the hierarchy has nothing to bend. Peak deviation is only 1.9% of the disk
radius. Don't oversell that curve on stage.

### The visual that does work: the sibling radius profile

Between two *siblings* the geodesic bends hard — up to 28% of the disk radius.
For `Electric & Stovetop Espresso Pots`:

| sibling | start r | dips to | parent r |
|---|---:|---:|---:|
| Drip Coffee Makers | 12.20 | **6.92** | 10.68 |
| Espresso Machines | 12.20 | **6.93** | 10.68 |
| French Presses | 12.20 | **8.07** | 10.68 |

The shortest path between two specific categories passes **shallower than their
own parent**. That is the entire case for hyperbolic space in one picture.

Drawing those arcs on the disk fails, though: the siblings all lie in nearly the
same direction, so the paths leave and return through the same corridor and
overlap into a single stripe. The UI plots **radius against progress along the
path** instead — three clean U-curves dipping below an amber parent line.

### The Gaudi caveat

Say: *"negative curvature is an old idea in this city."* Sagrada Familia uses
hyperboloids and hyperbolic paraboloids — doubly-ruled, negatively curved
surfaces — to build a branching tree of columns out of straight members.

Do **not** say Gaudi used the Poincare disc. Negative Gaussian curvature of a
*surface embedded in R³* is not the same thing as a negatively curved *space*.
The audience will include someone who knows this. Keep it to 20 seconds, as an
intuition pump for "in a negatively curved space there is exponentially more
room as you move outward, which is exactly what a branching tree needs."


## The three things an engineer can use on Monday

1. **`diagnose_language_bias.py`** — points at any existing Qdrant collection
   with a language field and reports same-language *lift* (share in results ÷
   share in corpus). No eval set, no labels, no relevance judgements.

   ```bash
   python3 src/diagnose_language_bias.py \
     --collection my_docs --lang-field locale --url http://localhost:6333
   ```

2. **SHIFT itself** — a mean of differences over parallel pairs. No retraining,
   no new model, no second index. Store several alphas as named vectors and A/B
   them, which is exactly what this demo does (`a000`…`a100`).

   Gotcha worth 30 seconds on stage: E5 models need **separate offsets for the
   `query:` and `passage:` prefixes**. One shared offset silently underperforms.

3. **The prefetch-then-rescore pattern** — usable for *any* non-standard metric,
   not just hyperbolic. Cheap ANN metric for recall, exact metric in a Formula
   Query for precision. The acosh reconstruction is the general trick: if your
   metric is expressible in `ln`/`sqrt`/arithmetic, Qdrant can rescore with it.


---

## The companion page (`site/`)

A single static HTML file plus 44 KB of JSON: the measured results, with the
Poincaré and Euclidean projections of the ATC tree drawn side by side. No build
step, no framework, no dependencies. It exists so the audience has a URL that
still works after the laptop closes.

Every number on the page is read from `site/data/*.json` at runtime — nothing is
hard-coded — so it cannot silently drift from the artifacts it was generated
from.

One rendering detail worth knowing, because it is easy to get wrong in any
hyperbolic viewer: the disk is plotted at **hyperbolic** radius
`2·artanh(||x||)`, not Euclidean norm. Euclidean norms here run 0.9246 to 0.9994,
so all five ATC levels sit within 8% of the boundary and the picture would show
an undifferentiated ring. The caption on the page says so rather than letting
the transform pass unmentioned.

Preview it with any static server:

```bash
python3 -m http.server 4455 --directory site
```

### Deploying to Vercel

The root `vercel.json` sets `outputDirectory` to `site`, so importing this repo
needs no dashboard configuration — framework preset **Other**, no build command.

```bash
npm i -g vercel
vercel          # preview
vercel --prod   # production
```

Or import the GitHub repo at vercel.com/new and accept the defaults.

**Only `site/` is deployable.** The medicine demo cannot go on Vercel and is not
meant to: a 2.1 GB embedding model against a 250 MB function limit, Qdrant local
mode holding an exclusive lock on a 384 MB directory, and OCR that calls a macOS
framework. The demo is offline-first on purpose; the page is the part that
travels.

---

## Repository layout and what is not committed

`.gitignore` excludes about 1.6 GB of regenerable data:

| ignored | size | rebuild with |
|---|---:|---|
| `artifacts/` | 1.5 GB | the `src/**` build scripts |
| `web/med_img/` | 98 MB | `src/med/fetch_photos.py` |
| `web/img/` | 14 MB | `src/fetch_images.py` |

Committed instead: the source registry CSVs and taxonomy files under `data/`
(~37 MB, so the corpus is reproducible from the repo alone) and the small JSON
summaries under `site/data/` that the companion page needs.

First run of either demo embeds its corpus and builds its collections — a few
minutes on an M-series Mac via MPS. After that, startup is seconds.
