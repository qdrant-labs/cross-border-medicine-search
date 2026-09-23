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
1024-dim accuracy. An e5 control with no MRL training drops to 86% — the same
operation, and it only works because the model was trained for it.

**Hyperbolic rerank:** a 10-dimensional Poincaré embedding of the ATC tree,
MAP 0.8314 against 0.7369 for Euclidean at the same dimension. It adds 0.008 on
packaging shots and **0.080** on near-textless tablet blisters — the harder case,
where knowing the drug-classification tree substitutes for having read anything.

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
   live is a genuine first run.
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
