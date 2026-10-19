"""Downloads openly licensed photos from Openverse (Flickr and others) into the demo folder.

Only licenses that allow commercial use and modification are requested
(CC BY, CC BY-SA, CC0, public domain). Each photo is saved with a generic
camera-style name (IMG_XXXX.jpg) — finding it without relying on the name is
the whole point of the demo — and its author, license and source are written
to tools/provenance_openverse.json. Re-running resumes where it stopped.

Openverse rate-limits anonymous clients; the script waits as long as the API
asks (Retry-After). Expect it to take a while.

Usage: python tools/download_openverse_photos.py
"""
import json
import random
import re
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
DEST = HERE.parent / "demo-folder" / "Photos"
PROVENANCE = HERE / "provenance_openverse.json"
UA = {"User-Agent": "embeddinggemma2-localsearch/1.0 (github.com/ericrisco/embeddinggemma2-localsearch)"}

# query → how many photos
TOPICS = {
    "dog sleeping on sofa": 10, "cat sleeping on laptop keyboard": 8, "golden retriever park": 8,
    "beach sunset": 10, "mountain lake reflection": 10, "snowy mountain hiking": 8,
    "paella valenciana": 8, "pizza margherita": 8, "latte art coffee cup": 8, "birthday cake candles": 8,
    "city street night rain": 8, "vintage red car": 8, "bicycle parked street": 8, "train station platform": 8,
    "airplane airport terminal": 8, "office meeting whiteboard": 8, "laptop on wooden desk": 8,
    "server rack data center": 10, "children playing playground": 8, "rock concert crowd": 8,
    "football stadium": 8, "modern kitchen interior": 8, "living room interior": 8, "tulip field": 8,
    "autumn forest": 8, "horse meadow": 8, "parrot bird": 8, "sagrada familia barcelona": 8,
    "lisbon tram": 8, "eiffel tower": 6, "bookshelf library": 8, "handwritten notes notebook": 8,
    "receipt paper shop": 6, "whiteboard diagram": 6, "sushi plate": 6, "wedding ceremony": 6,
    "swimming pool": 6, "christmas tree lights": 6, "running marathon": 6, "construction site crane": 6,
}


def get(url, **kw):
    """GET that honours HTTP 429 + Retry-After."""
    for _ in range(8):
        r = requests.get(url, headers=UA, timeout=60, **kw)
        if r.status_code != 429:
            r.raise_for_status()
            return r
        wait = int(r.headers.get("retry-after", 15)) + 2
        print(f"  429, waiting {wait} s", flush=True)
        time.sleep(wait)
    r.raise_for_status()


def search(query: str, n: int) -> list[dict]:
    r = get("https://api.openverse.org/v1/images/", params={
        "q": query, "page_size": n * 2, "license_type": "commercial,modification",
        "extension": "jpg", "mature": "false",
    })
    found = []
    for x in r.json().get("results", []):
        if (x.get("width") or 0) < 450:
            continue
        found.append({
            "query": query, "title": x.get("title"), "author": x.get("creator"),
            "license": f"CC {x.get('license', '').upper()} {x.get('license_version') or ''}".strip().replace("CC CC0", "CC0"),
            "source": x.get("foreign_landing_url"), "image_url": x["url"], "openverse_id": x["id"],
        })
        if len(found) >= n:
            break
    return found


def larger(url: str) -> str:
    # Flickr: ask for the 1024 px version when the URL is the 500 px one.
    m = re.match(r"(https://live\.staticflickr\.com/\d+/\d+_[0-9a-f]+)\.jpg$", url)
    return m.group(1) + "_b.jpg" if m else url


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    provenance = json.loads(PROVENANCE.read_text()) if PROVENANCE.exists() else []
    done = {p["openverse_id"] for p in provenance}
    taken = {p.stem for p in DEST.glob("IMG_*.jpg")}
    numbers = [n for n in random.Random(7).sample(range(1000, 9999), 3000) if f"IMG_{n}" not in taken]
    for query, n in TOPICS.items():
        try:
            photos = search(query, n)
        except Exception as e:
            print(f"! {query}: {e}", file=sys.stderr)
            continue
        for ph in photos:
            if ph["openverse_id"] in done:
                continue
            try:
                try:
                    img = get(larger(ph["image_url"]))
                except Exception:
                    img = get(ph["image_url"])
                if len(img.content) < 15000:
                    raise ValueError("image too small")
            except Exception as e:
                print(f"! {ph['openverse_id']}: {e}", file=sys.stderr)
                continue
            name = f"IMG_{numbers.pop(0)}.jpg"
            (DEST / name).write_bytes(img.content)
            provenance.append({"file": name, **ph})
            done.add(ph["openverse_id"])
            PROVENANCE.write_text(json.dumps(provenance, ensure_ascii=False, indent=1))
            time.sleep(0.3)
        print(f"{query}: {len(photos)}  (total {len(provenance)})", flush=True)
        time.sleep(3.5)


if __name__ == "__main__":
    main()
