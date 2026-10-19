"""Fills the demo folder with random Unsplash photos via Lorem Picsum.

Lorem Picsum serves Unsplash photos (Unsplash License: free to use, including
commercially) with no rate limit, but they can't be searched by topic — which
is fine: a random mix looks like a real camera roll. Photos are saved as
IMG_XXXX.jpg and their author and source go to tools/provenance_unsplash.json.

Usage: python tools/download_unsplash_photos.py [--n 260]
"""
import argparse
import json
import random
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
DEST = HERE.parent / "demo-folder" / "Photos"
PROVENANCE = HERE / "provenance_unsplash.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=260, help="how many photos (default 260)")
    args = ap.parse_args()

    catalogue = []
    for page in range(1, 12):
        r = requests.get(f"https://picsum.photos/v2/list?page={page}&limit=100", timeout=30)
        r.raise_for_status()
        catalogue += r.json()
    rnd = random.Random(11)
    rnd.shuffle(catalogue)

    DEST.mkdir(parents=True, exist_ok=True)
    taken = {p.stem for p in DEST.glob("IMG_*.jpg")}
    numbers = [n for n in rnd.sample(range(1000, 9999), 3000) if f"IMG_{n}" not in taken]
    provenance = []
    for x in catalogue[: args.n]:
        w, h = int(x["width"]), int(x["height"])
        width = 1280
        height = round(h * width / w)
        try:
            img = requests.get(f"https://picsum.photos/id/{x['id']}/{width}/{height}.jpg", timeout=60)
            img.raise_for_status()
        except Exception as e:
            print("!", x["id"], e)
            continue
        name = f"IMG_{numbers.pop(0)}.jpg"
        (DEST / name).write_bytes(img.content)
        provenance.append({"file": name, "author": x["author"], "source": x["url"], "license": "Unsplash License"})
        if len(provenance) % 25 == 0:
            print(len(provenance), flush=True)
    PROVENANCE.write_text(json.dumps(provenance, ensure_ascii=False, indent=1))
    print("total", len(provenance))


if __name__ == "__main__":
    main()
