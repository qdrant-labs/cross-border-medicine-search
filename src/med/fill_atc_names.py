"""Fill in ATC level names the product harvest could not supply.

A product record on CIMA only carries the ATC levels from 3 down, so every
level-2 node in the tree came back nameless. CIMA exposes the full ATC index
separately as "maestra 7", which does have them -- but only behind a filter,
so it has to be walked prefix by prefix rather than downloaded whole.

Names come back in Spanish. That is left as-is: this is the Spanish regulator's
own vocabulary, and the demo is explicitly about the same molecule wearing
different names in different countries.
"""
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "artifacts" / "med"
UA = "BarcelonaGeometryDemo/0.1 (Qdrant conference demo; https://qdrant.tech)"
MAESTRA_ATC = 7


def fetch(prefix):
    """All ATC index entries matching a prefix. Paginated, 200 per page."""
    found, page = {}, 1
    while True:
        q = urllib.parse.urlencode({"maestra": MAESTRA_ATC, "nombre": prefix,
                                    "pagina": page})
        req = urllib.request.Request(
            f"https://cima.aemps.es/cima/rest/maestras?{q}",
            headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                if r.status == 204:
                    return found
                d = json.load(r)
        except Exception as e:
            print(f"  {prefix}: {e}")
            return found
        rows = d.get("resultados") or []
        for x in rows:
            c, n = x.get("codigo"), x.get("nombre")
            if c and n:
                found[c.strip()] = n.strip()
        if len(rows) < d.get("tamanioPagina", 200):
            return found
        page += 1
        time.sleep(0.15)


def main():
    tree = json.loads((OUT / "atc_tree.json").read_text())
    missing = sorted({c for c, n in tree.items() if not n["name"]})
    print(f"{len(missing)} unnamed nodes")

    # Walking the 14 level-1 letters pulls back far more of the index than the
    # nodes we are missing, but it is 14 requests instead of 127.
    names = {}
    for i, letter in enumerate(sorted({c[0] for c in missing}), 1):
        got = fetch(letter)
        names.update(got)
        print(f"  [{i}] {letter}: +{len(got)} (total {len(names)})")
        time.sleep(0.2)

    filled = 0
    for code, node in tree.items():
        if not node["name"] and code in names:
            node["name"] = names[code]
            filled += 1

    (OUT / "atc_tree.json").write_text(
        json.dumps(tree, ensure_ascii=False, indent=1))
    (OUT / "atc_names.json").write_text(
        json.dumps(names, ensure_ascii=False, indent=1))

    still = [c for c, n in tree.items() if not n["name"]]
    print(f"\nfilled {filled}, still unnamed {len(still)}")
    for c in still[:12]:
        print("   ", c, "level", tree[c]["level"])


if __name__ == "__main__":
    main()
