"""Parse the Google Product Taxonomy into a tree.

Source: https://www.google.com/basepages/producttype/taxonomy-with-ids.en-US.txt
5,595 categories, 7 levels. Same dataset used in the Qdrant article.
"""
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "taxonomy.en-US.txt"


class Taxonomy:
    def __init__(self, path=DATA):
        self.by_id = {}        # id -> node dict
        self.by_path = {}      # full "A > B > C" -> id
        self._load(path)
        self._link()

    def _load(self, path):
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            raw_id, _, full = line.partition(" - ")
            cid = int(raw_id)
            parts = [p.strip() for p in full.split(">")]
            self.by_id[cid] = {
                "id": cid,
                "full": full,
                "name": parts[-1],
                "parts": parts,
                "depth": len(parts) - 1,
                "parent": None,
                "children": [],
            }
            self.by_path[full] = cid

    def _link(self):
        for cid, node in self.by_id.items():
            if node["depth"] == 0:
                continue
            parent_path = " > ".join(node["parts"][:-1])
            pid = self.by_path.get(parent_path)
            if pid is not None:
                node["parent"] = pid
                self.by_id[pid]["children"].append(cid)

    # --- relationships used for training / evaluation ---

    def ancestor_pairs(self):
        """All (descendant, ancestor) transitive pairs. This is the training signal,
        matching the article's 17,312 relationships figure."""
        pairs = []
        for cid, node in self.by_id.items():
            cur = node["parent"]
            while cur is not None:
                pairs.append((cid, cur))
                cur = self.by_id[cur]["parent"]
        return pairs

    def parent_pairs(self):
        """Direct-parent only. The article's harder eval that scored 0.539 MAP."""
        return [(cid, n["parent"]) for cid, n in self.by_id.items() if n["parent"] is not None]

    def stats(self):
        depths = [n["depth"] for n in self.by_id.values()]
        return {
            "categories": len(self.by_id),
            "levels": max(depths) + 1,
            "ancestor_relationships": len(self.ancestor_pairs()),
            "direct_parent_relationships": len(self.parent_pairs()),
            "roots": sum(1 for n in self.by_id.values() if n["depth"] == 0),
        }


if __name__ == "__main__":
    t = Taxonomy()
    for k, v in t.stats().items():
        print(f"{k:32s} {v}")
