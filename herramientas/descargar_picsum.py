"""Completa la carpeta con fotos de Unsplash vía Lorem Picsum (licencia Unsplash)."""
import json, random, time
from pathlib import Path
import requests

DESTINO = Path(__file__).resolve().parent.parent / "carpeta-demo" / "Fotos"
PROC = Path(__file__).resolve().parent / "procedencia_picsum.json"
N = 260
lista = []
for pag in range(1, 12):
    r = requests.get(f"https://picsum.photos/v2/list?page={pag}&limit=100", timeout=30)
    r.raise_for_status()
    lista += r.json()
rnd = random.Random(11)
rnd.shuffle(lista)
usados = {p.stem for p in DESTINO.glob("IMG_*.jpg")}
numeros = [n for n in rnd.sample(range(1000, 9999), 3000) if f"IMG_{n}" not in usados]
proc = []
for k, x in enumerate(lista[:N]):
    w, h = int(x["width"]), int(x["height"])
    ancho = 1280
    alto = round(h * ancho / w)
    try:
        img = requests.get(f"https://picsum.photos/id/{x['id']}/{ancho}/{alto}.jpg", timeout=60)
        img.raise_for_status()
    except Exception as e:
        print("!", x["id"], e)
        continue
    nombre = f"IMG_{numeros[k]}.jpg"
    (DESTINO / nombre).write_bytes(img.content)
    proc.append({"archivo": nombre, "autor": x["author"], "pagina": x["url"], "licencia": "Unsplash License"})
    if k % 25 == 0:
        print(k, flush=True)
PROC.write_text(json.dumps(proc, ensure_ascii=False, indent=1))
print("total", len(proc))
