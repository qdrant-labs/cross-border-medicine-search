"""Read the text off a medicine box using Apple's Vision framework.

Vision is the right OCR for this demo for one reason above all others: it runs
entirely on device. No API key, no network, no per-call cost, and the venue wifi
can be unplugged. It is also the same engine behind Live Text, so it is tuned
for exactly this situation -- printed text photographed at an angle, on a glossy
curved carton, under bad lighting.

Polish and Spanish are both in the supported set, which matters because the
whole demo is a Polish box being read in a Spanish room.

Language correction is deliberately OFF. It is built for prose, and medicine
brands are not words: it rewrites "Amotaks" toward "Amoktas", "Betamox" toward
"Betamos". The strings we most need intact are precisely the ones a dictionary
has never seen, so the dictionary is a liability here rather than a help.
"""
import Quartz
import Vision
from Foundation import NSData

LANGS = ["pl-PL", "es-ES", "en-US"]

# Below this, Vision is guessing at glare or package artwork rather than
# reading. Kept low rather than strict: the downstream embedding tolerates a
# stray token far better than it tolerates a missing brand name.
MIN_CONFIDENCE = 0.30


def _cgimage(image_bytes):
    data = NSData.dataWithBytes_length_(image_bytes, len(image_bytes))
    src = Quartz.CGImageSourceCreateWithData(data, None)
    if src is None or Quartz.CGImageSourceGetCount(src) == 0:
        raise ValueError("not a decodable image")
    return Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)


def read(image_bytes, langs=LANGS, min_conf=MIN_CONFIDENCE):
    """Image bytes -> [{text, confidence, area, box}], reading order preserved.

    'area' is the fraction of the frame the text block occupies. Box artwork
    puts the brand in the largest type, so area is a usable proxy for "this is
    the name of the product" versus "this is the dosage small print".

    'box' is [x, y, w, h] as fractions of the frame, with the origin at the
    TOP-left. Vision hands back a bottom-left origin because it inherits Core
    Graphics' convention; every consumer here is a browser, where y grows
    downward, so the flip happens once at the source rather than in each caller.
    """
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    req.setRecognitionLanguages_(list(langs))
    req.setUsesLanguageCorrection_(False)

    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
        _cgimage(image_bytes), None)
    ok, err = handler.performRequests_error_([req], None)
    if not ok:
        raise RuntimeError(f"Vision failed: {err}")

    out = []
    for obs in req.results() or []:
        cands = obs.topCandidates_(1)
        if not cands:
            continue
        c = cands[0]
        if c.confidence() < min_conf:
            continue
        bb = obs.boundingBox()
        w, h = float(bb.size.width), float(bb.size.height)
        out.append({
            "text": c.string(),
            "confidence": round(float(c.confidence()), 3),
            "area": round(w * h, 5),
            "box": [round(float(bb.origin.x), 4),
                    round(1.0 - float(bb.origin.y) - h, 4),
                    round(w, 4), round(h, 4)],
        })
    return out


def query_text(blocks, top_n=6):
    """Collapse OCR blocks into one retrieval string.

    Ordered by area, not by confidence or position. The corpus documents are
    "brand, INN, form" -- brand first -- so leading with the largest type on
    the carton lines the query up with how the documents were written.
    """
    ranked = sorted(blocks, key=lambda b: -b["area"])[:top_n]
    seen, parts = set(), []
    for b in ranked:
        t = b["text"].strip()
        k = t.lower()
        if len(t) < 2 or k in seen:
            continue
        seen.add(k)
        parts.append(t)
    return ", ".join(parts)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    for p in sys.argv[1:]:
        blocks = read(Path(p).read_bytes())
        print(f"\n{p}  ({len(blocks)} blocks)")
        for b in sorted(blocks, key=lambda b: -b["area"])[:8]:
            print(f"   {b['area']:.4f}  {b['confidence']:.2f}  {b['text']}")
        print("  query:", query_text(blocks))
