"""Cache CC-licensed product photos from Wikimedia Commons, one set per category.

Why Commons: the talk is recorded and published, so every image needs a licence
we can actually show. Commons gives us the licence string and the author in the
same API call, which is what `web/img/manifest.json` carries through to the UI.

Run once. Output is static files, so the demo has zero network dependency on
stage -- which matters because the Qdrant local-mode collection is already well
over its recommended point ceiling and does not need the competition.
"""
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "web" / "img"
TARGETS = Path("/tmp/bgd/targets.json")

API = "https://commons.wikimedia.org/w/api.php"
UA = "BarcelonaGeometryDemo/0.1 (Qdrant conference demo; https://qdrant.tech)"
PER_CAT = 4
THUMB_W = 400

# Commons returns plenty of diagrams, logos and maps for product nouns. We only
# want something that looks like a photo of the object. "screenshot" is here
# because "blender" is also a 3D suite, and it out-ranks the appliance.
BAD = ("logo", "icon", "map", "diagram", "chart", "coat of arms", "flag of",
       "svg", ".pdf", ".ogv", ".webm", ".tif", "screenshot", "poster",
       "painting", "drawing", "stamp", "banknote", "postcard", "patent")
OK_LIC = ("cc", "public domain", "pd-", "cc0")

# A handful of leaves are single generic words that mean something else on
# Commons. Only these get a domain nudge -- applying a hint everywhere made
# results worse ("kitchen espresso machine" returns people standing in kitchens).
HINT = {"Blenders": "kitchen", "Mixers": "kitchen", "Fryers": "deep",
        "Grills": "barbecue", "Hoods": "range", "Steamers": "food",
        "Filters": "camera", "Cases": "camera", "Bags": "camera"}


def api(params):
    q = urllib.parse.urlencode({**params, "format": "json"})
    req = urllib.request.Request(f"{API}?{q}", headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def search(term):
    try:
        d = api({"action": "query", "generator": "search", "gsrsearch": term,
                 "gsrnamespace": 6, "gsrlimit": 14, "prop": "imageinfo",
                 "iiprop": "url|extmetadata", "iiurlwidth": THUMB_W})
    except Exception as e:
        print(f"    search failed: {e}")
        return []
    out = []
    for p in d.get("query", {}).get("pages", {}).values():
        title = p.get("title", "")
        if any(b in title.lower() for b in BAD):
            continue
        ii = (p.get("imageinfo") or [{}])[0]
        thumb = ii.get("thumburl")
        if not thumb:
            continue
        em = ii.get("extmetadata", {})
        lic = (em.get("LicenseShortName", {}).get("value") or "").strip()
        if not any(k in lic.lower() for k in OK_LIC):
            continue
        author = (em.get("Artist", {}).get("value") or "").strip()
        # Artist arrives as HTML; the UI wants a plain credit line.
        for tag in ("<", ">"):
            while tag == "<" and "<" in author and ">" in author:
                author = author[:author.index("<")] + author[author.index(">") + 1:]
        out.append({"title": title, "thumb": thumb, "license": lic,
                    "author": (author or "Unknown")[:60],
                    "page": ii.get("descriptionurl", "")})
        if len(out) >= PER_CAT:
            break
    return out


def singular(w):
    if w.endswith("ies"):
        return w[:-3] + "y"
    if w.endswith(("ses", "xes", "zes", "ches", "shes")):
        return w[:-2]
    if w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def query_for(path):
    """Singular + lowercase. This is not cosmetic: "Bicycle Brake Calipers"
    returns photos of whole bicycle framesets, while "bicycle brake caliper"
    returns photos of calipers. Commons matches description prose, which is
    written in the singular."""
    leaf = path.split(" > ")[-1]
    q = " ".join(singular(w) for w in leaf.lower().replace("&", " ").split())
    hint = HINT.get(leaf)
    return f"{hint} {q}" if hint else q


def main():
    targets = json.loads(TARGETS.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    mpath = OUT / "manifest.json"
    man = json.loads(mpath.read_text()) if mpath.exists() else {}

    todo = [(c, p) for c, p in targets.items() if c not in man]
    print(f"{len(man)} cached, {len(todo)} to fetch")

    for n, (cid, path) in enumerate(todo, 1):
        term = query_for(path)
        hits = search(term)
        d = OUT / str(cid)
        saved = []
        for i, h in enumerate(hits):
            ext = ".jpg" if ".jp" in h["thumb"].lower() else ".png"
            f = d / f"{i}{ext}"
            try:
                d.mkdir(parents=True, exist_ok=True)
                req = urllib.request.Request(h["thumb"], headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=30) as r:
                    data = r.read()
                if len(data) < 1200:      # placeholder / error body
                    continue
                f.write_bytes(data)
                saved.append({"file": f"{cid}/{f.name}", "license": h["license"],
                              "author": h["author"], "page": h["page"]})
            except Exception as e:
                print(f"    dl failed: {e}")
            time.sleep(0.15)
        man[cid] = {"path": path, "query": term, "images": saved}
        print(f"[{n}/{len(todo)}] {term[:44]:46s} {len(saved)} img")
        if n % 10 == 0:
            mpath.write_text(json.dumps(man, indent=1))
        time.sleep(0.25)

    mpath.write_text(json.dumps(man, indent=1))
    tot = sum(len(v["images"]) for v in man.values())
    empty = [v["path"] for v in man.values() if not v["images"]]
    print(f"\ndone: {tot} images across {len(man)} categories")
    print(f"categories with no usable image: {len(empty)}")
    for p in empty[:15]:
        print("   ", p)


if __name__ == "__main__":
    main()
