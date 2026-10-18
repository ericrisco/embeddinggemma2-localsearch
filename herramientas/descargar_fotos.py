"""Descarga fotos con licencia libre (Openverse: CC BY, BY-SA, CC0, dominio público) para la carpeta de demo.

Guarda cada foto con un nombre genérico (IMG_XXXX.jpg) y deja la procedencia
(autor, licencia, URL) en _trabajo/procedencia_fotos.json.
"""
import json
import random
import sys
import time
from pathlib import Path

import requests

DESTINO = Path(__file__).resolve().parent.parent / "carpeta-demo" / "Fotos"
PROCEDENCIA = Path(__file__).resolve().parent / "procedencia_openverse.json"
UA = {"User-Agent": "demo-buscador-local/1.0 (github.com/ericrisco/buscador-local)"}
API = "https://commons.wikimedia.org/w/api.php"

TEMAS = {
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


def pedir(url, **kw):
    for intento in range(8):
        r = requests.get(url, headers=UA, timeout=60, **kw)
        if r.status_code != 429:
            r.raise_for_status()
            return r
        espera = int(r.headers.get("retry-after", 15)) + 2
        print(f"  429, espero {espera} s", flush=True)
        time.sleep(espera)
    r.raise_for_status()


def buscar(tema: str, n: int) -> list[dict]:
    r = pedir("https://api.openverse.org/v1/images/", params={
        "q": tema, "page_size": n * 2, "license_type": "commercial,modification",
        "extension": "jpg", "mature": "false",
    })
    buenas = []
    for x in r.json().get("results", []):
        if (x.get("width") or 0) < 450:
            continue
        buenas.append({
            "tema": tema, "titulo": x["id"], "url": x["url"], "pagina": x.get("foreign_landing_url"),
            "licencia": f"CC {x.get('license','').upper()} {x.get('license_version') or ''}".strip(),
            "autor": x.get("creator"), "nombre_original": x.get("title"),
        })
        if len(buenas) >= n:
            break
    return buenas


def grande(url: str) -> str:
    # Flickr: pedir la versión de 1024 px si la URL es la de 500 px.
    import re
    m = re.match(r"(https://live\.staticflickr\.com/\d+/\d+_[0-9a-f]+)\.jpg$", url)
    return m.group(1) + "_b.jpg" if m else url


def main() -> None:
    DESTINO.mkdir(parents=True, exist_ok=True)
    procedencia = json.loads(PROCEDENCIA.read_text()) if PROCEDENCIA.exists() else []
    hechas = {p["titulo"] for p in procedencia}
    numeros = random.Random(7).sample(range(1000, 9999), 2000)
    k = len(procedencia)
    for tema, n in TEMAS.items():
        try:
            fotos = buscar(tema, n)
        except Exception as e:
            print(f"! {tema}: {e}", file=sys.stderr)
            continue
        for f in fotos:
            if f["titulo"] in hechas:
                continue
            nombre = f"IMG_{numeros[k]}.jpg"
            try:
                try:
                    img = pedir(grande(f["url"]))
                except Exception:
                    img = pedir(f["url"])
                if len(img.content) < 15000:
                    raise ValueError("imagen demasiado pequeña")
            except Exception as e:
                print(f"! {f['titulo']}: {e}", file=sys.stderr)
                time.sleep(2)
                continue
            (DESTINO / nombre).write_bytes(img.content)
            procedencia.append({**f, "archivo": nombre})
            hechas.add(f["titulo"])
            k += 1
            time.sleep(0.3)
        print(f"{tema}: {len(fotos)}  (total {k})", flush=True)
        time.sleep(3.5)
        PROCEDENCIA.write_text(json.dumps(procedencia, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
