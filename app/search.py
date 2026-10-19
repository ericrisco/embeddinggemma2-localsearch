#!/usr/bin/env python3
"""EmbeddingGemma 2 Local Search: pick a folder of photos and PDFs, index it with
EmbeddingGemma 2, and search by what is *inside* the files, mixing text and
images in a single result list.

Everything runs on this computer. Internet is only needed the first time, to
download the model; after that the app starts in offline mode.

Usage:
  .venv/bin/python search.py                 # opens http://127.0.0.1:8765
  .venv/bin/python search.py --port 9000 --no-window
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

HERE = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(HERE / ".cache-hf"))  # keep the model next to the app
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
INDEXES = HERE / "indexes"
WEB = HERE / "web"
MODEL_ID = "google/embeddinggemma-2"

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".bmp", ".gif", ".tif", ".tiff"}
PDF_EXT = {".pdf"}
IMAGE_SIDE = 1024   # images are downscaled to this before embedding (and for the UI)
PAGE_DPI = 110      # PDF pages are rendered at this resolution
CHUNK = 700         # characters per PDF text chunk
OVERLAP = 120       # overlap between consecutive chunks


# ------------------------------------------------------------------ model


class Model:
    """Loads EmbeddingGemma 2 once, in the background, and serialises access to it."""

    def __init__(self) -> None:
        self.status = "loading"
        self.error = ""
        self.device = ""
        self.load_seconds = 0.0
        self._m = None
        self._lock = threading.Lock()
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self) -> None:
        t0 = time.perf_counter()
        try:
            import torch
            from sentence_transformers import SentenceTransformer

            device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
            kw = dict(device=device, config_kwargs={"audio_config": None})  # text + image encoders only
            try:
                # Already downloaded? Then don't even try to reach the network.
                os.environ["HF_HUB_OFFLINE"] = "1"
                self._m = SentenceTransformer(MODEL_ID, local_files_only=True, **kw)
            except Exception:
                os.environ["HF_HUB_OFFLINE"] = "0"
                self.status = "downloading"
                self._m = SentenceTransformer(MODEL_ID, **kw)
                os.environ["HF_HUB_OFFLINE"] = "1"
            self.device = str(self._m.device)
            self._warm_up()
            self.load_seconds = round(time.perf_counter() - t0, 1)
            self.status = "ready"
        except Exception as e:
            self.status = "error"
            self.error = f"{type(e).__name__}: {e}"
            traceback.print_exc()

    def _warm_up(self) -> None:
        # The first pass through the GPU is slow: do it now, not on the user's first search.
        import tempfile

        from PIL import Image

        path = Path(tempfile.gettempdir()) / "embeddinggemma2_localsearch_warmup.jpg"
        Image.new("RGB", (256, 256), (120, 140, 160)).save(path)
        for text in ("warming up", "a slightly longer test query to warm up the model"):
            self._m.encode(text, prompt_name="SearchQuery")
        self._m.encode([{"image": str(path)}])
        self._m.encode(["a test document"], prompt_name="Document")

    def ready(self) -> bool:
        return self.status == "ready"

    def embed_images(self, paths: list[str], batch: int = 8):
        with self._lock:
            return self._m.encode([{"image": p} for p in paths], batch_size=batch,
                                  normalize_embeddings=True, convert_to_numpy=True)

    def embed_documents(self, texts: list[str], batch: int = 16):
        with self._lock:
            return self._m.encode(texts, prompt_name="Document", batch_size=batch,
                                  normalize_embeddings=True, convert_to_numpy=True)

    def embed_query(self, text: str):
        with self._lock:
            return self._m.encode(text, prompt_name="SearchQuery",
                                  normalize_embeddings=True, convert_to_numpy=True)


# ------------------------------------------------------------------ index


def index_dir(folder: Path) -> Path:
    return INDEXES / hashlib.sha1(str(folder).encode()).hexdigest()[:12]


def chunk_text(text: str) -> list[str]:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text).strip()
    if len(text) <= CHUNK:
        return [text] if len(text) >= 25 else []
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i : i + CHUNK])
        i += CHUNK - OVERLAP
    return chunks


def is_screenshot(name: str) -> bool:
    return name.lower().startswith(("screenshot", "screen shot", "captura"))


class Index:
    """An indexed folder: items (a photo or a PDF page) and their vectors.

    Each item has one or more vectors (rows): the image (photo, screenshot or
    rendered PDF page) and, for PDF pages, one per text chunk.
    """

    def __init__(self) -> None:
        self.folder: Path | None = None
        self.dir: Path | None = None
        self.items: list[dict] = []
        self.rows: list[dict] = []  # {"item": item id, "kind": "image"|"text", "text": chunk}
        self.vectors = None
        self.stats: dict = {}
        self.progress: dict = {"active": False}

    # ---------------------------------------------------------- load

    def load(self, folder: Path) -> bool:
        import numpy as np

        d = index_dir(folder)
        if not (d / "vectors.npy").exists():
            return False
        data = json.loads((d / "index.json").read_text())
        self.folder, self.dir = folder, d
        self.items, self.rows = data["items"], data["rows"]
        self.stats = data.get("stats", {})
        self.vectors = np.load(d / "vectors.npy")
        return True

    def summary(self) -> dict | None:
        if self.folder is None:
            return None
        return {"folder": str(self.folder), "name": self.folder.name, "stats": self.stats}

    # ---------------------------------------------------------- build

    def build(self, folder: Path, model: Model) -> None:
        import numpy as np
        import pymupdf
        from PIL import Image, ImageOps

        try:
            from pillow_heif import register_heif_opener

            register_heif_opener()
        except ImportError:
            pass

        t0 = time.perf_counter()
        p = self.progress = {
            "active": True, "phase": "Finding files", "done": 0, "total": 0, "current": "",
            "images": 0, "pdfs": 0, "pages": 0, "started": time.time(), "error": "",
        }
        try:
            d = index_dir(folder)
            thumbs = d / "thumbs"
            thumbs.mkdir(parents=True, exist_ok=True)
            files = sorted(
                f for f in folder.rglob("*")
                if f.is_file() and not f.name.startswith(".") and f.suffix.lower() in IMAGE_EXT | PDF_EXT
            )
            images = [f for f in files if f.suffix.lower() in IMAGE_EXT]
            pdfs = [f for f in files if f.suffix.lower() in PDF_EXT]
            total_pages = 0
            for f in pdfs:
                try:
                    with pymupdf.open(f) as doc:
                        total_pages += doc.page_count
                except Exception:
                    pass
            p.update(phase="Preparing", total=len(images) + total_pages)

            items: list[dict] = []
            pending_images: list[tuple[int, str]] = []  # (item id, path of the downscaled copy)
            pending_texts: list[tuple[int, str]] = []   # (item id, text chunk)

            def rel(f: Path) -> str:
                return str(f.relative_to(folder))

            for f in images:
                p["current"] = rel(f)
                k = len(items)
                copy = thumbs / f"{k:05d}.jpg"
                try:
                    with Image.open(f) as im:
                        im = ImageOps.exif_transpose(im).convert("RGB")
                        width, height = im.size
                        im.thumbnail((IMAGE_SIDE, IMAGE_SIDE))
                        im.save(copy, "JPEG", quality=86)
                except Exception as e:
                    print(f"skipping {f}: {e}", file=sys.stderr)
                    continue
                items.append({"type": "screenshot" if is_screenshot(f.name) else "photo",
                              "path": str(f), "rel": rel(f), "name": f.name,
                              "thumb": copy.name, "width": width, "height": height})
                pending_images.append((k, str(copy)))
                p["images"] += 1

            for f in pdfs:
                p["current"] = rel(f)
                try:
                    doc = pymupdf.open(f)
                except Exception as e:
                    print(f"skipping {f}: {e}", file=sys.stderr)
                    continue
                p["pdfs"] += 1
                with doc:
                    for n, page in enumerate(doc, start=1):
                        k = len(items)
                        copy = thumbs / f"{k:05d}.jpg"
                        pix = page.get_pixmap(dpi=PAGE_DPI)
                        Image.frombytes("RGB", (pix.width, pix.height), pix.samples).save(copy, "JPEG", quality=86)
                        text = page.get_text()
                        items.append({"type": "pdf", "path": str(f), "rel": rel(f), "name": f.name,
                                      "page": n, "pages": doc.page_count, "thumb": copy.name,
                                      "width": pix.width, "height": pix.height,
                                      "has_text": len(text.strip()) > 25})
                        pending_images.append((k, str(copy)))
                        for chunk in chunk_text(text):
                            pending_texts.append((k, chunk))
                        p["pages"] += 1

            # Vectors: first every image (photos, screenshots, PDF pages), then the PDF text.
            p.update(phase="Reading images", done=0, total=len(pending_images) + len(pending_texts))
            rows, blocks = [], []
            batch = 8
            for i in range(0, len(pending_images), batch):
                part = pending_images[i : i + batch]
                p["current"] = items[part[0][0]]["rel"]
                blocks.append(model.embed_images([path for _, path in part], batch))
                rows += [{"item": k, "kind": "image"} for k, _ in part]
                p["done"] += len(part)

            p["phase"] = "Reading PDF text"
            batch = 16
            for i in range(0, len(pending_texts), batch):
                part = pending_texts[i : i + batch]
                p["current"] = items[part[0][0]]["rel"]
                blocks.append(model.embed_documents([t for _, t in part], batch))
                rows += [{"item": k, "kind": "text", "text": t} for k, t in part]
                p["done"] += len(part)

            vectors = np.vstack(blocks).astype("float32")
            peak = peak_ram_bytes()
            stats = {
                "date": time.strftime("%Y-%m-%d %H:%M"),
                "photos": sum(it["type"] == "photo" for it in items),
                "screenshots": sum(it["type"] == "screenshot" for it in items),
                "pdfs": p["pdfs"], "pages": p["pages"], "text_chunks": len(pending_texts),
                "vectors": int(vectors.shape[0]), "dimensions": int(vectors.shape[1]),
                "seconds": round(time.perf_counter() - t0, 1), "device": model.device,
                "peak_ram_gb": round(peak / 1e9, 1) if peak else None,
                "machine": machine(),
            }
            np.save(d / "vectors.npy", vectors)
            (d / "index.json").write_text(json.dumps(
                {"folder": str(folder), "items": items, "rows": rows, "stats": stats}, ensure_ascii=False))
            (INDEXES / "last.txt").write_text(str(folder))
            self.load(folder)
            p.update(phase="Done", active=False, current="", done=p["total"])
            print(json.dumps(stats, ensure_ascii=False, indent=1))
        except Exception as e:
            traceback.print_exc()
            p.update(active=False, phase="Error", error=f"{type(e).__name__}: {e}")

    # ---------------------------------------------------------- search

    def rank(self, q, n: int, kind: str = "all", exclude: int | None = None) -> list[dict]:
        """Ranks items by similarity to the query vector q.

        Text↔text and text↔image similarities live on different scales (the
        "modality gap"), so each modality is standardised on its own (z-score)
        before mixing them. Each item keeps its best-scoring row.
        """
        import numpy as np

        sims = self.vectors @ q.reshape(-1)
        is_text = np.array([r["kind"] == "text" for r in self.rows])
        z = np.empty_like(sims)
        for mask in (is_text, ~is_text):
            if mask.any():
                v = sims[mask]
                z[mask] = (v - v.mean()) / (v.std() + 1e-6)

        best: dict[int, tuple[float, int]] = {}
        best_text: dict[int, int] = {}  # most similar text chunk of each page, for the snippet
        for i, r in enumerate(self.rows):
            it = r["item"]
            if it == exclude:
                continue
            if it not in best or z[i] > best[it][0]:
                best[it] = (float(z[i]), i)
            if r["kind"] == "text" and (it not in best_text or sims[i] > sims[best_text[it]]):
                best_text[it] = i

        def keep(it: int) -> bool:
            return kind == "all" or self.items[it]["type"] == kind

        order = sorted((it for it in best if keep(it)), key=lambda it: -best[it][0])[:n]
        out = []
        for it in order:
            score, row = best[it]
            out.append({
                "id": it, **{k: v for k, v in self.items[it].items() if k != "path"},
                "matched_by": self.rows[row]["kind"], "z": round(score, 2),
                "snippet": self.rows[best_text[it]]["text"] if it in best_text else "",
                "similarity": round(float(sims[row]), 3),
            })
        return out

    def image_vector(self, item: int):
        for i, r in enumerate(self.rows):
            if r["item"] == item and r["kind"] == "image":
                return self.vectors[i]
        raise KeyError(item)


# ------------------------------------------------------------------ system helpers


def peak_ram_bytes() -> int:
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak if sys.platform == "darwin" else peak * 1024  # Linux reports KB
    except ImportError:  # Windows
        return 0


def machine() -> str:
    if sys.platform == "darwin":
        try:
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
            ram = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout)
            return f"{chip} · {ram / 2**30:.0f} GB"
        except (OSError, ValueError):
            pass
    import platform

    return f"{platform.system()} {platform.machine()} · {platform.processor() or ''}".strip(" ·")


def open_with_system(path: str, reveal: bool = False) -> None:
    """Opens a file with its default app, or reveals it in the file manager."""
    if sys.platform == "darwin":
        subprocess.run(["open", "-R", path] if reveal else ["open", path])
    elif sys.platform == "win32":
        if reveal:
            subprocess.run(["explorer", "/select,", path])
        else:
            os.startfile(path)  # type: ignore[attr-defined]
    else:
        subprocess.run(["xdg-open", str(Path(path).parent) if reveal else path])


_net = {"t": 0.0, "ok": False}


def online() -> bool:
    """Only for the badge in the corner: is there a connection right now?"""
    if time.time() - _net["t"] > 3:
        try:
            socket.create_connection(("1.1.1.1", 53), timeout=0.8).close()
            _net["ok"] = True
        except OSError:
            _net["ok"] = False
        _net["t"] = time.time()
    return _net["ok"]


class NoDialog(Exception):
    """No native folder picker available: the web UI asks for the path instead."""


def pick_folder() -> str | None:
    """Native folder picker (macOS: Finder; elsewhere: tkinter)."""
    if sys.platform == "darwin":
        r = subprocess.run(
            ["osascript", "-e", 'POSIX path of (choose folder with prompt "Choose the folder to index")'],
            capture_output=True, text=True,
        )
        return r.stdout.strip().rstrip("/") or None
    try:
        import tkinter
        from tkinter import filedialog

        root = tkinter.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askdirectory(title="Choose the folder to index")
        root.destroy()
        return path or None
    except Exception as e:
        raise NoDialog(str(e))


# ------------------------------------------------------------------ HTTP server


def make_handler(model: Model, index: Index):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, data, code=200):
            body = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _file(self, path: Path, content_type: str):
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=3600")
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> bytes:
            return self.rfile.read(int(self.headers.get("Content-Length", 0)))

        def _can_search(self) -> bool:
            if not model.ready():
                self._json({"error": "The model is still loading"}, 503)
                return False
            if index.vectors is None:
                self._json({"error": "Pick and index a folder first"}, 409)
                return False
            return True

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path == "/":
                    return self._file(WEB / "index.html", "text/html; charset=utf-8")
                if u.path == "/status":
                    pr = dict(index.progress)
                    if pr.get("started"):
                        pr["seconds"] = round(time.time() - pr["started"], 1)
                    return self._json({
                        "model": model.status, "model_error": model.error, "device": model.device,
                        "online": online(), "index": index.summary(), "progress": pr,
                    })
                if u.path.startswith("/thumb/"):
                    it = index.items[int(u.path.rsplit("/", 1)[1])]
                    return self._file(index.dir / "thumbs" / it["thumb"], "image/jpeg")
                if u.path.startswith("/file/"):
                    it = index.items[int(u.path.rsplit("/", 1)[1])]
                    if it["type"] == "pdf":
                        return self._file(Path(it["path"]), "application/pdf")
                    return self._file(index.dir / "thumbs" / it["thumb"], "image/jpeg")
                if u.path == "/search":
                    if not self._can_search():
                        return
                    t0 = time.perf_counter()
                    v = model.embed_query(q.get("q", ""))
                    res = index.rank(v, int(q.get("n", 24)), q.get("type", "all"))
                    return self._json({"ms": round((time.perf_counter() - t0) * 1000), "results": res})
                if u.path == "/similar":
                    if not self._can_search():
                        return
                    t0 = time.perf_counter()
                    it = int(q["id"])
                    res = index.rank(index.image_vector(it), int(q.get("n", 24)), q.get("type", "all"), exclude=it)
                    return self._json({"ms": round((time.perf_counter() - t0) * 1000), "results": res})
                self._json({"error": "not found"}, 404)
            except (KeyError, IndexError, ValueError) as e:
                self._json({"error": str(e)}, 400)

        def do_POST(self):
            u = urlparse(self.path)
            try:
                if u.path == "/pick-folder":
                    try:
                        return self._json({"folder": pick_folder()})
                    except NoDialog:
                        return self._json({"manual": True})
                if u.path == "/index":
                    data = json.loads(self._body() or b"{}")
                    folder = Path(data["folder"]).expanduser().resolve()
                    if not folder.is_dir():
                        return self._json({"error": "That folder does not exist"}, 400)
                    if index.progress.get("active"):
                        return self._json({"error": "Already indexing"}, 409)
                    if not data.get("rebuild") and index.load(folder):
                        (INDEXES / "last.txt").write_text(str(folder))
                        return self._json({"ok": True, "already_indexed": True})
                    if not model.ready():
                        return self._json({"error": "The model is still loading"}, 503)
                    threading.Thread(target=index.build, args=(folder, model), daemon=True).start()
                    return self._json({"ok": True})
                if u.path == "/search-image":
                    if not self._can_search():
                        return
                    import tempfile

                    from PIL import Image, ImageOps

                    t0 = time.perf_counter()
                    with tempfile.NamedTemporaryFile(suffix=".img", delete=False) as f:
                        f.write(self._body())
                    with Image.open(f.name) as im:
                        im = ImageOps.exif_transpose(im).convert("RGB")
                        im.thumbnail((IMAGE_SIDE, IMAGE_SIDE))
                        jpg = f.name + ".jpg"
                        im.save(jpg, "JPEG", quality=88)
                    v = model.embed_images([jpg])[0]
                    os.unlink(f.name)
                    os.unlink(jpg)
                    res = index.rank(v, 24, parse_qs(u.query).get("type", ["all"])[0])
                    return self._json({"ms": round((time.perf_counter() - t0) * 1000), "results": res})
                if u.path in ("/open", "/reveal"):
                    it = index.items[int(json.loads(self._body())["id"])]
                    open_with_system(it["path"], reveal=u.path == "/reveal")
                    return self._json({"ok": True})
                self._json({"error": "not found"}, 404)
            except Exception as e:
                traceback.print_exc()
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    return Handler


def open_window(url: str) -> None:
    """Opens the UI as an app window (Chrome/Edge --app mode) or in the default browser."""
    import shutil
    import webbrowser

    candidates = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        shutil.which("google-chrome") or "", shutil.which("chromium") or "", shutil.which("chromium-browser") or "",
    ]
    for c in candidates:
        if c and Path(c).exists():
            subprocess.Popen([c, f"--app={url}", "--window-size=1440,900", f"--user-data-dir={HERE / '.chrome-app'}"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
    webbrowser.open(url)


def main() -> None:
    ap = argparse.ArgumentParser(description="Local semantic search over photos and PDFs with EmbeddingGemma 2.")
    ap.add_argument("--port", type=int, default=8765, help="port for the local web UI (default 8765)")
    ap.add_argument("--no-window", action="store_true", help="only start the server, don't open a window")
    args = ap.parse_args()

    INDEXES.mkdir(exist_ok=True)
    model = Model()
    index = Index()
    last = INDEXES / "last.txt"
    if last.exists():
        index.load(Path(last.read_text().strip()))

    # Bound to 127.0.0.1 only: not reachable from other machines on the network.
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(model, index))
    url = f"http://127.0.0.1:{args.port}"
    print(f"EmbeddingGemma 2 Local Search running at {url}  (Ctrl+C to stop)", flush=True)
    if not args.no_window:
        open_window(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
