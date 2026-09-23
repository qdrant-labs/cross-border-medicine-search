"""Join the Polish, Spanish and Dutch medicine registries on their ATC code.

Sources, all open government data:
  PL  Rejestr Produktow Leczniczych  -- bulk CSV from dane.gov.pl (22,923 rows)
  ES  CIMA AEMPS REST API            -- live, no key, includes packaging photos
  NL  CBG-MEB Geneesmiddeleninformatiebank -- bulk CSV, 19,484 rows, 92% ATC

Germany and Portugal were investigated and dropped. Recording why, so the search
does not get repeated: Germany's AMIce does carry ATC, but its terms allow
machine-readable storage only transiently during output, so a derived file
cannot be redistributed; the one genuinely open BfArM product file
(verkehrsfaehige_arzneimittel.csv) turns out to be 415 bytes of counts by
authorisation type with no products in it; and the email-gated Referenzdatenbank
per 31b SGB V has 21 columns -- product name, PZN, dosage form, substance name,
strength -- and no ATC field at all, so it cannot be joined to this corpus even
once obtained. Portugal's INFOMED publishes no bulk export and no public API;
bulk access runs through CITS, behind a signed protocol and a fee. Both would
have meant scraping a search UI against its terms.

The join key is deliberately the ATC code, never the brand name. Brand names
collide across borders and sometimes carry *different* active ingredients, so
matching on them would be actively unsafe. ATC is assigned by the WHO and is
stable across countries, which makes the mapping auditable: the UI can show the
full N -> N02 -> N02B -> N02BE -> N02BE01 path behind every result.

The ATC code is also self-describing as a hierarchy -- level boundaries fall at
fixed character offsets -- so the tree needed for the Poincare embedding comes
free from the codes themselves. Only the human-readable level names have to be
harvested, and CIMA returns those alongside each product.
"""
import csv
import json
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
MED = ROOT / "data" / "med"
OUT = ROOT / "artifacts" / "med"

CIMA = "https://cima.aemps.es/cima/rest"
UA = "BarcelonaGeometryDemo/0.1 (Qdrant conference demo; https://qdrant.tech)"

# How many ATC codes to carry into the demo. The full intersection is ~2.8k,
# which is far more than a 20 minute talk needs and turns the CIMA harvest into
# an hour of polite waiting. Ranked by Polish product count, so the codes that
# survive are the ones a person might plausibly hold up to a camera.
TOP_ATC = 220
ES_PER_ATC = 6          # Spanish products kept per ATC code
NL_PER_ATC = 8          # Dutch products kept per ATC code
PHOTOS_FOR = 3          # of those, how many get a detail call for packaging photos

# ATC level 1 is 14 fixed codes. CIMA only returns names from level 3 down, so
# these are spelled out rather than harvested.
L1 = {
    "A": "Alimentary tract and metabolism",
    "B": "Blood and blood forming organs",
    "C": "Cardiovascular system",
    "D": "Dermatologicals",
    "G": "Genito-urinary system and sex hormones",
    "H": "Systemic hormonal preparations",
    "J": "Antiinfectives for systemic use",
    "L": "Antineoplastic and immunomodulating agents",
    "M": "Musculo-skeletal system",
    "N": "Nervous system",
    "P": "Antiparasitic products, insecticides and repellents",
    "R": "Respiratory system",
    "S": "Sensory organs",
    "V": "Various",
}

# ATC is fixed-width per level: N / N02 / N02B / N02BE / N02BE01
CUTS = (1, 3, 4, 5, 7)

# The Polish registry also carries ATCvet codes, which are the human code with
# a "Q" glued on the front: QJ01CA04 is veterinary amoxicillin. They are dropped
# for two reasons. Clinically, this demo answers "what would a pharmacy in
# Barcelona sell me instead" and poultry antibiotics are not an answer to that.
# Structurally, the Q shifts every level boundary by one character, so CUTS
# slices them into codes that do not exist -- QJ0, QJ01CA0 -- which is exactly
# the 65 nodes that came back unnamed from the CIMA index, because there was
# nothing there to name. They contributed 384 documents and zero cross-border
# matches. Note that L1 below has no Q entry: the tree never expected them.


def ancestors(code):
    """Every ATC prefix that is itself a valid level, shortest first."""
    return [code[:c] for c in CUTS if len(code) >= c]


def get(path, **params):
    q = urllib.parse.urlencode(params)
    req = urllib.request.Request(f"{CIMA}/{path}?{q}", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def load_poland():
    """PL registry -> {atc: [product, ...]}, keeping one row per brand name.

    The CSV carries one row per package size, so a single brand appears many
    times over. The demo cares about distinct brands, not pack sizes.
    """
    path = MED / "pl_rpl.csv"
    rows = csv.reader(path.open(encoding="utf-8-sig"), delimiter=";")
    next(rows)
    by_atc, seen = defaultdict(list), set()
    for r in rows:
        if len(r) < 16:
            continue
        atc, name, inn = r[12].strip(), r[1].strip(), r[2].strip()
        if not atc or not name or len(atc) < 7:
            continue
        if atc.startswith("Q"):
            continue
        key = (atc, name.lower())
        if key in seen:
            continue
        seen.add(key)
        by_atc[atc].append({
            "brand": name, "inn": inn, "form": r[8].strip(),
            "strength": r[7].strip(), "holder": r[13].strip()[:60],
        })
    return by_atc


def clean_substance(raw):
    """CBG active ingredients -> a readable INN string.

    The field is uppercase, "#"-separated, and names the salt actually used
    rather than the INN: "LIDOCAINEHYDROCHLORIDE 1-WATER" for what the Polish
    and Spanish registries both call lidocaine. The hydrate suffix is dropped
    and the rest title-cased, which is enough for an embedding to see the
    molecule; stripping the salt itself is not attempted, because doing it by
    string surgery would silently mangle names where the suffix is part of the
    INN, and the cross-border join runs on ATC rather than on this text anyway.
    """
    out = []
    for part in raw.split("#"):
        part = part.strip()
        if not part:
            continue
        for suffix in (" 1-WATER", " 0-WATER", " 2-WATER", " 3-WATER",
                       " 5-WATER", " 7-WATER", " WATERVRIJ"):
            if part.endswith(suffix):
                part = part[: -len(suffix)]
                break
        out.append(part.capitalize())
        if len(out) >= 3:        # Creon lists four enzymes; three is plenty
            break
    return ", ".join(out)


def load_netherlands():
    """NL registry -> {atc: [product, ...]}, one row per distinct product name.

    The ATC column arrives as "C01AA05 - Digoxin", so the code is the part
    before the separator. Rows without one are dropped rather than guessed at:
    7.8% of the file has no ATC, mostly homeopathic RVH registrations, and
    those have no cross-border equivalent to find.
    """
    path = MED / "nl_cbg.csv"
    if not path.exists():
        print(f"NL: {path} missing, skipping", file=sys.stderr)
        return {}
    csv.field_size_limit(10 ** 7)
    by_atc, seen = defaultdict(list), set()
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="|"):
            atc = (r.get("ATC") or "").split(" - ")[0].strip()
            name = (r.get("PRODUCTNAAM") or "").strip()
            if not atc or not name or len(atc) < 7 or atc.startswith("Q"):
                continue
            key = (atc, name.lower())
            if key in seen:
                continue
            seen.add(key)
            by_atc[atc].append({
                "brand": name,
                "inn": clean_substance(r.get("WERKZAMESTOFFEN") or ""),
                "form": (r.get("FARMACEUTISCHEVORM") or "").strip(),
                "strength": (r.get("POTENTIE") or "").strip(),
                "holder": (r.get("HANDELSVERGUNNINGHOUDER") or "").strip()[:60],
                "rvg": (r.get("REGISTRATIENUMMER") or "").strip(),
            })
    return by_atc


def fetch_spain(atc):
    """Spanish products for one ATC code, straight off the list endpoint.

    The list response already carries everything needed -- vtm, dosis and fotos
    -- so there is no per-product detail call. That matters beyond speed:
    "vtm" is the active ingredient, and without it a Spanish record reads
    "ANTIDOL 1 G COMPRIMIDOS", which names the molecule nowhere. No embedding
    model can tell that is paracetamol, so the cross-border match had nothing
    to stand on. The Polish rows carry their INN; this is the Spanish half.
    """
    try:
        d = get("medicamentos", atc=atc, pagina=1)
    except Exception as e:
        print(f"    ES list failed {atc}: {e}", file=sys.stderr)
        return []

    out = []
    for m in (d.get("resultados") or [])[:ES_PER_ATC]:
        # "materialas" is the packaging shot -- that is the thing a person
        # actually points a camera at. The dosage-form shot is the backup.
        photos = [{"kind": f.get("tipo"), "url": f["url"]}
                  for f in (m.get("fotos") or []) if f.get("url")]
        out.append({
            "brand": m.get("nombre", "").strip(),
            "inn": ((m.get("vtm") or {}).get("nombre") or "").strip(),
            "nreg": m.get("nregistro"),
            "otc": (m.get("cpresc") or "").strip(),
            "strength": (m.get("dosis") or "").strip(),
            "form": (m.get("formaFarmaceuticaSimplificada") or {}).get("nombre", ""),
            "generic": bool(m.get("generico")),
            "photos": photos,
        })
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    nl = load_netherlands()
    print(f"NL: {sum(len(v) for v in nl.values())} distinct products "
          f"across {len(nl)} ATC codes")

    # The Spanish half is ~220 polite HTTP calls; the Dutch half is a local
    # file. Re-running the whole harvest to add a local join would be rude to
    # CIMA and slow, so an existing pairs.json is topped up in place.
    existing = OUT / "pairs.json"
    if "--nl-only" in sys.argv:
        if not existing.exists():
            sys.exit("--nl-only needs an existing pairs.json")
        pairs = json.loads(existing.read_text())
        for atc, v in pairs.items():
            v["nl"] = nl.get(atc, [])[:NL_PER_ATC]
        existing.write_text(json.dumps(pairs, ensure_ascii=False, indent=1))
        have = sum(1 for v in pairs.values() if v["nl"])
        print(f"added NL to {have}/{len(pairs)} ATC codes, "
              f"{sum(len(v['nl']) for v in pairs.values())} products")
        return

    pl = load_poland()
    print(f"PL: {sum(len(v) for v in pl.values())} distinct brands "
          f"across {len(pl)} ATC codes")

    targets = sorted(pl, key=lambda a: -len(pl[a]))[:TOP_ATC]
    print(f"harvesting {len(targets)} ATC codes from CIMA\n")

    pairs, matched = {}, 0
    for i, atc in enumerate(targets, 1):
        es = fetch_spain(atc)
        if es:
            matched += 1
        pairs[atc] = {"pl": pl[atc][:12], "es": es,
                      "nl": nl.get(atc, [])[:NL_PER_ATC]}
        if i % 20 == 0 or i == len(targets):
            print(f"  [{i}/{len(targets)}] matched={matched}")
            (OUT / "pairs.json").write_text(json.dumps(pairs, ensure_ascii=False))
        time.sleep(0.2)

    # Level names all come from fill_atc_names.py against the CIMA ATC index.
    # The per-product detail call used to seed some of them, but it covered
    # only levels 3 and below and cost a request per product.
    atc_names = {}

    # Build the tree from the codes themselves.
    nodes = {}
    for atc in pairs:
        for a in ancestors(atc):
            if a in nodes:
                continue
            nodes[a] = {
                "code": a,
                "level": CUTS.index(len(a)) + 1,
                "name": L1.get(a) if len(a) == 1 else atc_names.get(a, ""),
                "parent": ancestors(a)[-2] if len(a) > 1 else None,
            }

    (OUT / "pairs.json").write_text(json.dumps(pairs, ensure_ascii=False, indent=1))
    (OUT / "atc_tree.json").write_text(json.dumps(nodes, ensure_ascii=False, indent=1))

    unnamed = [c for c, n in nodes.items() if not n["name"]]
    both = [a for a, v in pairs.items() if v["pl"] and v["es"]]
    three = [a for a, v in pairs.items() if v["pl"] and v["es"] and v["nl"]]
    photos = sum(len(p["photos"]) for v in pairs.values() for p in v["es"])
    print(f"\nATC codes with products in PL and ES: {len(both)}/{len(pairs)}")
    print(f"ATC codes present in all three:       {len(three)}/{len(pairs)}")
    print(f"tree nodes: {len(nodes)}  (unnamed: {len(unnamed)})")
    print(f"ES packaging photos found: {photos}")


if __name__ == "__main__":
    main()
