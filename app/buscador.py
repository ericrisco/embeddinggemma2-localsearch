#!/usr/bin/env python3
"""Buscador local: eliges una carpeta con fotos y PDFs, la indexas con
EmbeddingGemma 2 y buscas por lo que sale dentro, mezclando texto e imagen.

Todo corre en este ordenador. Internet solo hace falta la primera vez para
descargar el modelo; después la app arranca en modo sin conexión.

Uso:
  .venv/bin/python buscador.py            # abre http://127.0.0.1:8765
  .venv/bin/python buscador.py --puerto 9000 --no-abrir
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

AQUI = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(AQUI / ".cache-hf"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
INDICES = AQUI / "indices"
WEB = AQUI / "web"
MODELO = "google/embeddinggemma-2"

EXT_IMAGEN = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".bmp", ".gif", ".tif", ".tiff"}
EXT_PDF = {".pdf"}
LADO_IMAGEN = 1024
DPI_PAGINA = 110
TROZO = 700  # caracteres por trozo de texto de un PDF
SOLAPE = 120


# ------------------------------------------------------------------ modelo


class Modelo:
    """Carga EmbeddingGemma 2 una vez, en segundo plano, y serializa su uso."""

    def __init__(self) -> None:
        self.estado = "cargando"
        self.error = ""
        self.dispositivo = ""
        self.segundos_carga = 0.0
        self._m = None
        self._candado = threading.Lock()
        threading.Thread(target=self._cargar, daemon=True).start()

    def _cargar(self) -> None:
        t0 = time.perf_counter()
        try:
            import torch
            from sentence_transformers import SentenceTransformer

            disp = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
            kw = dict(device=disp, config_kwargs={"audio_config": None})  # texto + imagen, sin audio
            try:
                # Si ya está descargado, ni siquiera se intenta conectar.
                os.environ["HF_HUB_OFFLINE"] = "1"
                self._m = SentenceTransformer(MODELO, local_files_only=True, **kw)
            except Exception:
                os.environ["HF_HUB_OFFLINE"] = "0"
                self.estado = "descargando"
                self._m = SentenceTransformer(MODELO, **kw)
                os.environ["HF_HUB_OFFLINE"] = "1"
            self.dispositivo = str(self._m.device)
            self._calentar()
            self.segundos_carga = round(time.perf_counter() - t0, 1)
            self.estado = "listo"
        except Exception as e:
            self.estado = "error"
            self.error = f"{type(e).__name__}: {e}"
            traceback.print_exc()

    def _calentar(self) -> None:
        # La primera pasada por la GPU es lenta: se hace aquí y no en la primera búsqueda en cámara.
        from PIL import Image
        import tempfile

        ruta = Path(tempfile.gettempdir()) / "buscador_calentar.jpg"
        Image.new("RGB", (256, 256), (120, 140, 160)).save(ruta)
        for texto in ("calentando", "una búsqueda de prueba algo más larga para calentar el modelo"):
            self._m.encode(texto, prompt_name="SearchQuery")
        self._m.encode([{"image": str(ruta)}])
        self._m.encode(["un documento de prueba"], prompt_name="Document")

    def listo(self) -> bool:
        return self.estado == "listo"

    def imagenes(self, rutas: list[str], lote: int = 8):
        with self._candado:
            return self._m.encode([{"image": r} for r in rutas], batch_size=lote,
                                  normalize_embeddings=True, convert_to_numpy=True)

    def textos(self, textos: list[str], lote: int = 16):
        with self._candado:
            return self._m.encode(textos, prompt_name="Document", batch_size=lote,
                                  normalize_embeddings=True, convert_to_numpy=True)

    def consulta(self, texto: str):
        with self._candado:
            return self._m.encode(texto, prompt_name="SearchQuery",
                                  normalize_embeddings=True, convert_to_numpy=True)


# ------------------------------------------------------------------ índice


def carpeta_indice(carpeta: Path) -> Path:
    return INDICES / hashlib.sha1(str(carpeta).encode()).hexdigest()[:12]


def trocear(texto: str) -> list[str]:
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r"\n{2,}", "\n", texto).strip()
    if len(texto) <= TROZO:
        return [texto] if len(texto) >= 25 else []
    trozos, i = [], 0
    while i < len(texto):
        trozos.append(texto[i : i + TROZO])
        i += TROZO - SOLAPE
    return trozos


class Indice:
    """Una carpeta indexada: elementos (foto o página de PDF) y sus vectores.

    Cada elemento tiene uno o varios vectores: la imagen (foto o página
    renderizada) y, en los PDF, los trozos de texto de la página.
    """

    def __init__(self) -> None:
        self.carpeta: Path | None = None
        self.dir: Path | None = None
        self.elementos: list[dict] = []
        self.filas: list[dict] = []  # {"e": índice de elemento, "m": "imagen"|"texto", "t": trozo}
        self.vectores = None
        self.ensayo: dict = {}
        self.progreso = {"activo": False}

    # ---------------------------------------------------------- cargar

    def cargar(self, carpeta: Path) -> bool:
        import numpy as np

        d = carpeta_indice(carpeta)
        if not (d / "vectores.npy").exists():
            return False
        datos = json.loads((d / "indice.json").read_text())
        self.carpeta, self.dir = carpeta, d
        self.elementos, self.filas = datos["elementos"], datos["filas"]
        self.ensayo = datos.get("ensayo", {})
        self.vectores = np.load(d / "vectores.npy")
        return True

    def resumen(self) -> dict | None:
        if self.carpeta is None:
            return None
        return {"carpeta": str(self.carpeta), "nombre": self.carpeta.name, "ensayo": self.ensayo}

    # ---------------------------------------------------------- indexar

    def indexar(self, carpeta: Path, modelo: Modelo) -> None:
        import numpy as np
        import pymupdf
        from PIL import Image, ImageOps

        try:
            from pillow_heif import register_heif_opener

            register_heif_opener()
        except ImportError:
            pass

        t0 = time.perf_counter()
        p = self.progreso = {
            "activo": True, "fase": "Buscando archivos", "hechos": 0, "total": 0, "actual": "",
            "fotos": 0, "pdfs": 0, "paginas": 0, "inicio": time.time(), "error": "",
        }
        try:
            d = carpeta_indice(carpeta)
            mini = d / "miniaturas"
            mini.mkdir(parents=True, exist_ok=True)
            archivos = sorted(
                a for a in carpeta.rglob("*")
                if a.is_file() and not a.name.startswith(".") and a.suffix.lower() in EXT_IMAGEN | EXT_PDF
            )
            fotos = [a for a in archivos if a.suffix.lower() in EXT_IMAGEN]
            pdfs = [a for a in archivos if a.suffix.lower() in EXT_PDF]
            paginas_total = 0
            for a in pdfs:
                try:
                    with pymupdf.open(a) as doc:
                        paginas_total += doc.page_count
                except Exception:
                    pass
            p.update(fase="Preparando", total=len(fotos) + paginas_total)

            elementos: list[dict] = []
            imagenes_pend: list[tuple[int, str]] = []  # (elemento, ruta de la copia)
            textos_pend: list[tuple[int, str]] = []

            def rel(a: Path) -> str:
                return str(a.relative_to(carpeta))

            for a in fotos:
                p["actual"] = rel(a)
                k = len(elementos)
                copia = mini / f"{k:05d}.jpg"
                try:
                    with Image.open(a) as im:
                        im = ImageOps.exif_transpose(im).convert("RGB")
                        ancho, alto = im.size
                        im.thumbnail((LADO_IMAGEN, LADO_IMAGEN))
                        im.save(copia, "JPEG", quality=86)
                except Exception as e:
                    print(f"salto {a}: {e}", file=sys.stderr)
                    continue
                tipo = "captura" if a.name.lower().startswith(("captura", "screenshot", "screen shot")) else "foto"
                elementos.append({"tipo": tipo, "ruta": str(a), "rel": rel(a), "nombre": a.name,
                                  "mini": copia.name, "ancho": ancho, "alto": alto})
                imagenes_pend.append((k, str(copia)))
                p["fotos"] += 1

            for a in pdfs:
                p["actual"] = rel(a)
                try:
                    doc = pymupdf.open(a)
                except Exception as e:
                    print(f"salto {a}: {e}", file=sys.stderr)
                    continue
                p["pdfs"] += 1
                with doc:
                    for n, pagina in enumerate(doc, start=1):
                        k = len(elementos)
                        copia = mini / f"{k:05d}.jpg"
                        pix = pagina.get_pixmap(dpi=DPI_PAGINA)
                        Image.frombytes("RGB", (pix.width, pix.height), pix.samples).save(copia, "JPEG", quality=86)
                        texto = pagina.get_text()
                        elementos.append({"tipo": "pdf", "ruta": str(a), "rel": rel(a), "nombre": a.name,
                                          "pagina": n, "paginas": doc.page_count, "mini": copia.name,
                                          "ancho": pix.width, "alto": pix.height, "con_texto": len(texto.strip()) > 25})
                        imagenes_pend.append((k, str(copia)))
                        for t in trocear(texto):
                            textos_pend.append((k, t))
                        p["paginas"] += 1

            # Vectores: primero las imágenes (fotos y páginas), luego el texto de los PDF.
            p.update(fase="Leyendo imágenes", hechos=0, total=len(imagenes_pend) + len(textos_pend))
            filas, bloques = [], []
            lote = 8
            for i in range(0, len(imagenes_pend), lote):
                trozo = imagenes_pend[i : i + lote]
                p["actual"] = elementos[trozo[0][0]]["rel"]
                bloques.append(modelo.imagenes([r for _, r in trozo], lote))
                filas += [{"e": k, "m": "imagen"} for k, _ in trozo]
                p["hechos"] += len(trozo)

            p["fase"] = "Leyendo texto de los PDF"
            lote = 16
            for i in range(0, len(textos_pend), lote):
                trozo = textos_pend[i : i + lote]
                p["actual"] = elementos[trozo[0][0]]["rel"]
                bloques.append(modelo.textos([t for _, t in trozo], lote))
                filas += [{"e": k, "m": "texto", "t": t} for k, t in trozo]
                p["hechos"] += len(trozo)

            vectores = np.vstack(bloques).astype("float32")
            segundos = time.perf_counter() - t0
            pico = ram_pico_bytes()
            ensayo = {
                "fecha": time.strftime("%Y-%m-%d %H:%M"),
                "fotos": sum(e["tipo"] == "foto" for e in elementos),
                "capturas": sum(e["tipo"] == "captura" for e in elementos),
                "pdfs": p["pdfs"], "paginas": p["paginas"], "trozos_texto": len(textos_pend),
                "vectores": int(vectores.shape[0]), "dimensiones": int(vectores.shape[1]),
                "segundos": round(segundos, 1), "dispositivo": modelo.dispositivo,
                "ram_pico_gb": round(pico / 1e9, 1) if pico else None,
                "equipo": equipo(),
            }
            np.save(d / "vectores.npy", vectores)
            (d / "indice.json").write_text(json.dumps(
                {"carpeta": str(carpeta), "elementos": elementos, "filas": filas, "ensayo": ensayo},
                ensure_ascii=False))
            (INDICES / "ultima.txt").write_text(str(carpeta))
            self.cargar(carpeta)
            p.update(fase="Listo", activo=False, actual="", hechos=p["total"])
            print(json.dumps(ensayo, ensure_ascii=False, indent=1))
        except Exception as e:
            traceback.print_exc()
            p.update(activo=False, fase="Error", error=f"{type(e).__name__}: {e}")

    # ---------------------------------------------------------- buscar

    def ordenar(self, q, n: int, tipo: str = "todo", excluir: int | None = None) -> list[dict]:
        """Ordena elementos por parecido con el vector de consulta q.

        Los parecidos texto↔texto y texto↔imagen viven en escalas distintas, así
        que cada modalidad se estandariza por separado (z) antes de mezclarlas.
        """
        import numpy as np

        s = self.vectores @ q.reshape(-1)
        mod = np.array([f["m"] == "texto" for f in self.filas])
        z = np.empty_like(s)
        for mascara in (mod, ~mod):
            if mascara.any():
                v = s[mascara]
                z[mascara] = (v - v.mean()) / (v.std() + 1e-6)

        mejor: dict[int, tuple[float, int]] = {}
        mejor_texto: dict[int, int] = {}  # trozo de texto más parecido de cada página
        for i, f in enumerate(self.filas):
            e = f["e"]
            if e == excluir:
                continue
            if e not in mejor or z[i] > mejor[e][0]:
                mejor[e] = (float(z[i]), i)
            if f["m"] == "texto" and (e not in mejor_texto or s[i] > s[mejor_texto[e]]):
                mejor_texto[e] = i

        def pasa(e: int) -> bool:
            t = self.elementos[e]["tipo"]
            return tipo == "todo" or t == tipo or (tipo == "imagen" and t in ("foto", "captura"))

        orden = sorted((e for e in mejor if pasa(e)), key=lambda e: -mejor[e][0])[:n]
        salida = []
        for e in orden:
            puntos, fila = mejor[e]
            f = self.filas[fila]
            salida.append({
                "id": e, **{k: v for k, v in self.elementos[e].items() if k != "ruta"},
                "por": f["m"], "z": round(puntos, 2),
                "fragmento": self.filas[mejor_texto[e]]["t"] if e in mejor_texto else "",
                "similitud": round(float(s[fila]), 3),
            })
        return salida

    def vector_imagen(self, e: int):
        for i, f in enumerate(self.filas):
            if f["e"] == e and f["m"] == "imagen":
                return self.vectores[i]
        raise KeyError(e)


def ram_pico_bytes() -> int:
    try:
        import resource

        pico = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return pico if sys.platform == "darwin" else pico * 1024  # Linux lo da en KB
    except ImportError:  # Windows
        return 0


def equipo() -> str:
    if sys.platform == "darwin":
        try:
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
            ram = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout)
            return f"{chip} · {ram / 2**30:.0f} GB"
        except (OSError, ValueError):
            pass
    import platform

    return f"{platform.system()} {platform.machine()} · {platform.processor() or ''}".strip(" ·")


def abrir_con_sistema(ruta: str, mostrar: bool = False) -> None:
    """Abre un archivo (o lo señala en el explorador) con la app del sistema."""
    if sys.platform == "darwin":
        subprocess.run(["open", "-R", ruta] if mostrar else ["open", ruta])
    elif sys.platform == "win32":
        if mostrar:
            subprocess.run(["explorer", "/select,", ruta])
        else:
            os.startfile(ruta)  # type: ignore[attr-defined]
    else:
        subprocess.run(["xdg-open", str(Path(ruta).parent) if mostrar else ruta])


_red = {"t": 0.0, "ok": False}


def hay_red() -> bool:
    """Solo para el indicador de la esquina: ¿hay conexión ahora mismo?"""
    if time.time() - _red["t"] > 3:
        try:
            socket.create_connection(("1.1.1.1", 53), timeout=0.8).close()
            _red["ok"] = True
        except OSError:
            _red["ok"] = False
        _red["t"] = time.time()
    return _red["ok"]


class SinDialogo(Exception):
    """No hay diálogo nativo: la web pedirá la ruta a mano."""


def elegir_carpeta() -> str | None:
    """Diálogo nativo para elegir carpeta (macOS: Finder; otros: tkinter)."""
    if sys.platform == "darwin":
        r = subprocess.run(
            ["osascript", "-e", 'POSIX path of (choose folder with prompt "Elige la carpeta que quieres indexar")'],
            capture_output=True, text=True,
        )
        return r.stdout.strip().rstrip("/") or None
    try:
        import tkinter
        from tkinter import filedialog

        raiz = tkinter.Tk()
        raiz.withdraw()
        raiz.attributes("-topmost", True)
        ruta = filedialog.askdirectory(title="Elige la carpeta que quieres indexar")
        raiz.destroy()
        return ruta or None
    except Exception as e:
        raise SinDialogo(str(e))


# ------------------------------------------------------------------ servidor


def crear_manejador(modelo: Modelo, indice: Indice):
    class Manejador(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, datos, codigo=200):
            cuerpo = json.dumps(datos, ensure_ascii=False).encode()
            self.send_response(codigo)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(cuerpo)))
            self.end_headers()
            self.wfile.write(cuerpo)

        def _archivo(self, ruta: Path, tipo: str):
            cuerpo = ruta.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", tipo)
            self.send_header("Content-Length", str(len(cuerpo)))
            self.send_header("Cache-Control", "max-age=3600")
            self.end_headers()
            self.wfile.write(cuerpo)

        def _cuerpo(self) -> bytes:
            return self.rfile.read(int(self.headers.get("Content-Length", 0)))

        def _listo(self) -> bool:
            if not modelo.listo():
                self._json({"error": "El modelo aún se está cargando"}, 503)
                return False
            if indice.vectores is None:
                self._json({"error": "Primero elige e indexa una carpeta"}, 409)
                return False
            return True

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path == "/":
                    return self._archivo(WEB / "index.html", "text/html; charset=utf-8")
                if u.path == "/estado":
                    pr = dict(indice.progreso)
                    if pr.get("inicio"):
                        pr["segundos"] = round(time.time() - pr["inicio"], 1)
                    return self._json({
                        "modelo": modelo.estado, "error_modelo": modelo.error, "dispositivo": modelo.dispositivo,
                        "red": hay_red(), "indice": indice.resumen(), "progreso": pr,
                    })
                if u.path.startswith("/mini/"):
                    e = indice.elementos[int(u.path.rsplit("/", 1)[1])]
                    return self._archivo(indice.dir / "miniaturas" / e["mini"], "image/jpeg")
                if u.path.startswith("/archivo/"):
                    e = indice.elementos[int(u.path.rsplit("/", 1)[1])]
                    ruta = Path(e["ruta"])
                    if e["tipo"] == "pdf":
                        return self._archivo(ruta, "application/pdf")
                    return self._archivo(indice.dir / "miniaturas" / e["mini"], "image/jpeg")
                if u.path == "/buscar":
                    if not self._listo():
                        return
                    t0 = time.perf_counter()
                    v = modelo.consulta(q.get("q", ""))
                    res = indice.ordenar(v, int(q.get("n", 30)), q.get("tipo", "todo"))
                    return self._json({"ms": round((time.perf_counter() - t0) * 1000), "resultados": res})
                if u.path == "/parecidas":
                    if not self._listo():
                        return
                    t0 = time.perf_counter()
                    e = int(q["id"])
                    res = indice.ordenar(indice.vector_imagen(e), int(q.get("n", 30)), q.get("tipo", "todo"), excluir=e)
                    return self._json({"ms": round((time.perf_counter() - t0) * 1000), "resultados": res})
                self._json({"error": "no existe"}, 404)
            except (KeyError, IndexError, ValueError) as e:
                self._json({"error": str(e)}, 400)

        def do_POST(self):
            u = urlparse(self.path)
            try:
                if u.path == "/elegir":
                    try:
                        return self._json({"carpeta": elegir_carpeta()})
                    except SinDialogo:
                        return self._json({"manual": True})
                if u.path == "/indexar":
                    datos = json.loads(self._cuerpo() or b"{}")
                    carpeta = Path(datos["carpeta"]).expanduser().resolve()
                    if not carpeta.is_dir():
                        return self._json({"error": "La carpeta no existe"}, 400)
                    if indice.progreso.get("activo"):
                        return self._json({"error": "Ya se está indexando"}, 409)
                    if not datos.get("rehacer") and indice.cargar(carpeta):
                        (INDICES / "ultima.txt").write_text(str(carpeta))
                        return self._json({"ok": True, "ya_estaba": True})
                    if not modelo.listo():
                        return self._json({"error": "El modelo aún se está cargando"}, 503)
                    threading.Thread(target=indice.indexar, args=(carpeta, modelo), daemon=True).start()
                    return self._json({"ok": True})
                if u.path == "/buscar-imagen":
                    if not self._listo():
                        return
                    import tempfile

                    t0 = time.perf_counter()
                    with tempfile.NamedTemporaryFile(suffix=".img", delete=False) as f:
                        f.write(self._cuerpo())
                    from PIL import Image, ImageOps

                    with Image.open(f.name) as im:
                        im = ImageOps.exif_transpose(im).convert("RGB")
                        im.thumbnail((LADO_IMAGEN, LADO_IMAGEN))
                        ruta = f.name + ".jpg"
                        im.save(ruta, "JPEG", quality=88)
                    v = modelo.imagenes([ruta])[0]
                    os.unlink(f.name)
                    os.unlink(ruta)
                    res = indice.ordenar(v, 24, parse_qs(u.query).get("tipo", ["todo"])[0])
                    return self._json({"ms": round((time.perf_counter() - t0) * 1000), "resultados": res})
                if u.path == "/abrir":
                    e = indice.elementos[int(json.loads(self._cuerpo())["id"])]
                    abrir_con_sistema(e["ruta"])
                    return self._json({"ok": True})
                if u.path == "/mostrar":
                    e = indice.elementos[int(json.loads(self._cuerpo())["id"])]
                    abrir_con_sistema(e["ruta"], mostrar=True)
                    return self._json({"ok": True})
                self._json({"error": "no existe"}, 404)
            except Exception as e:
                traceback.print_exc()
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    return Manejador


def abrir_ventana(url: str) -> None:
    """Abre la web como ventana de app (Chrome/Edge en modo --app) o en el navegador."""
    import shutil
    import webbrowser

    candidatos = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        shutil.which("google-chrome") or "", shutil.which("chromium") or "", shutil.which("chromium-browser") or "",
    ]
    for c in candidatos:
        if c and Path(c).exists():
            subprocess.Popen([c, f"--app={url}", "--window-size=1440,900", f"--user-data-dir={AQUI / '.chrome-app'}"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
    webbrowser.open(url)


def main() -> None:
    ap = argparse.ArgumentParser(description="Buscador local de fotos y PDFs con EmbeddingGemma 2.")
    ap.add_argument("--puerto", type=int, default=8765)
    ap.add_argument("--no-abrir", action="store_true", help="no abrir la ventana de Chrome")
    args = ap.parse_args()

    INDICES.mkdir(exist_ok=True)
    modelo = Modelo()
    indice = Indice()
    ultima = INDICES / "ultima.txt"
    if ultima.exists():
        indice.cargar(Path(ultima.read_text().strip()))

    # Solo 127.0.0.1: no es accesible desde fuera de este ordenador.
    servidor = ThreadingHTTPServer(("127.0.0.1", args.puerto), crear_manejador(modelo, indice))
    url = f"http://127.0.0.1:{args.puerto}"
    print(f"Buscador en {url}  (Ctrl+C para parar)", flush=True)
    if not args.no_abrir:
        abrir_ventana(url)
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
