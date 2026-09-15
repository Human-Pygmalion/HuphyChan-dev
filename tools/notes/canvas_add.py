"""Add nodes and edges to docs/experiment_map.canvas without rewriting it by hand.

The canvas is an Obsidian JSON file: a list of nodes (a "file" node points at a note,
a "text" node carries Markdown inline) and a list of edges between them. Editing it by
hand means hunting for free coordinates and risking a broken file that Obsidian will
not open, so every session used to write a throwaway script for it. This is that
script, kept in the repo.

It is idempotent: a node or edge whose id already exists is UPDATED, not duplicated.

  python3 tools/notes/canvas_add.py --json '[
    {"id": "n_thing", "text": "**title**\n\nbody", "x": 100, "y": 200, "color": "4"},
    {"id": "n_note", "file": "experiments/2026-09-14_x.md", "x": 600, "y": 200},
    {"id": "e_a_b", "from": "n_thing", "to": "n_note", "label": "왜 이어지는가"}
  ]'
  python3 tools/notes/canvas_add.py --file additions.json      # same payload from a file

An entry with "from"/"to" is an edge; anything else is a node. Node defaults:
width 420, height 200, type "text" (or "file" when "file" is given). Edge defaults:
fromSide "bottom", toSide "top".
"""

import argparse
import json
import pathlib

CANVAS = pathlib.Path(__file__).resolve().parents[2] / "docs" / "experiment_map.canvas"


def build(entry: dict) -> tuple[str, dict]:
  """Returns ("node"|"edge", the object to store)."""
  if "from" in entry and "to" in entry:
    return "edge", {
      "id": entry["id"],
      "fromNode": entry["from"],
      "fromSide": entry.get("fromSide", "bottom"),
      "toNode": entry["to"],
      "toSide": entry.get("toSide", "top"),
      **({"label": entry["label"]} if entry.get("label") else {}),
    }
  node = {
    "id": entry["id"],
    "type": entry.get("type", "file" if "file" in entry else "text"),
    "x": int(entry.get("x", 0)),
    "y": int(entry.get("y", 0)),
    "width": int(entry.get("width", 420)),
    "height": int(entry.get("height", 200)),
  }
  if "file" in entry:
    node["file"] = entry["file"]
  if "text" in entry:
    node["text"] = entry["text"]
  if entry.get("color"):
    node["color"] = str(entry["color"])
  return "node", node


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  g = ap.add_mutually_exclusive_group(required=True)
  g.add_argument("--json", help="the additions as a JSON list")
  g.add_argument("--file", help="a file holding that same JSON list")
  ap.add_argument("--canvas", default=str(CANVAS))
  args = ap.parse_args()

  entries = json.loads(args.json if args.json else pathlib.Path(args.file).read_text())
  canvas = pathlib.Path(args.canvas)
  doc = json.loads(canvas.read_text())
  doc.setdefault("nodes", [])
  doc.setdefault("edges", [])

  added, updated = [], []
  for entry in entries:
    kind, obj = build(entry)
    bucket = doc["nodes"] if kind == "node" else doc["edges"]
    for i, existing in enumerate(bucket):
      if existing.get("id") == obj["id"]:
        bucket[i] = obj
        updated.append(obj["id"])
        break
    else:
      bucket.append(obj)
      added.append(obj["id"])

  canvas.write_text(json.dumps(doc, ensure_ascii=False, indent=2))
  print(f"추가: {added} | 갱신: {updated} | 노드 {len(doc['nodes'])} 엣지 {len(doc['edges'])}")


if __name__ == "__main__":
  main()
