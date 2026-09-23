"""Cache the CIMA packaging photographs locally.

These are the images a person is actually pointing a camera at, so they serve
two roles: the visual result grid, and the reference set the camera frame gets
matched against. Both have to work with the venue wifi unplugged, hence the
local copy.

CIMA advertises these under /fotos/thumbnails/, which is what the product detail
endpoint returns, but the same path under /fotos/full/ serves the original -- 450KB
against 2.5KB. The thumbnails are 200px and Vision reads exactly zero characters
off them, so the full version is not a nicety: the OCR path does not function
without it. They are then downscaled to MAX_PX, which keeps the text legible
while holding the offline bundle to something that fits comfortably on a laptop.

AEMPS publishes these as part of the public medicine register. Provenance is
recorded per file in manifest.json so the UI can attribute them on screen.
"""
import io
import json
import time
import urllib.request
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "artifacts" / "med"
IMG = ROOT / "web" / "med_img"
UA = "BarcelonaGeometryDemo/0.1 (Qdrant conference demo; https://qdrant.tech)"
MAX_PX = 1200
JPEG_Q = 85


def full_url(url):
    return url.replace("/fotos/thumbnails/", "/fotos/full/")


def shrink(data):
    """Downscale to MAX_PX on the long edge, leaving smaller originals alone."""
    im = Image.open(io.BytesIO(data))
    if im.mode != "RGB":
        im = im.convert("RGB")
    if max(im.size) > MAX_PX:
        im.thumbnail((MAX_PX, MAX_PX), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=JPEG_Q, optimize=True)
    return buf.getvalue(), im.size


def main():
    pairs = json.loads((OUT / "pairs.json").read_text())
    IMG.mkdir(parents=True, exist_ok=True)
    mpath = IMG / "manifest.json"
    man = json.loads(mpath.read_text()) if mpath.exists() else {}

    jobs = []
    for atc, v in pairs.items():
        for p in v["es"]:
            for f in p.get("photos", []):
                key = f"{p['nreg']}_{f['kind']}"
                # Entries cached before the switch to /fotos/full/ carry no "px"
                # and are the unreadable 200px thumbnails -- re-fetch those.
                if key not in man or "px" not in man[key]:
                    jobs.append((key, atc, p, f))

    print(f"{len(man)} cached, {len(jobs)} to fetch")
    ok = 0
    for i, (key, atc, p, f) in enumerate(jobs, 1):
        dest = IMG / f"{key}.jpg"
        src = full_url(f["url"])
        try:
            try:
                req = urllib.request.Request(src, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=30) as r:
                    data = r.read()
            except Exception:
                # Not every record has a full-size original; the thumbnail is
                # still worth keeping for the result grid even though OCR
                # will not get anything out of it.
                src = f["url"]
                req = urllib.request.Request(src, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=30) as r:
                    data = r.read()
            if len(data) < 1200:
                continue
            data, size = shrink(data)
            dest.write_bytes(data)
            man[key] = {"file": dest.name, "atc": atc, "nreg": p["nreg"],
                        "brand": p["brand"], "kind": f["kind"],
                        "otc": p.get("otc", ""), "source": src,
                        "px": list(size)}
            ok += 1
        except Exception as e:
            print(f"   {key}: {e}")
        if i % 50 == 0:
            mpath.write_text(json.dumps(man, ensure_ascii=False, indent=1))
            print(f"  [{i}/{len(jobs)}] ok={ok}")
        time.sleep(0.1)

    mpath.write_text(json.dumps(man, ensure_ascii=False, indent=1))
    size = sum(f.stat().st_size for f in IMG.glob("*.jpg"))
    print(f"\n{len(man)} photos cached, {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
