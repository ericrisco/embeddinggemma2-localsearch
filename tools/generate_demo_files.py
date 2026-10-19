"""Generates the fictional PDFs and screenshots of the demo folder.

Everything is made up (companies, people, amounts, plates). The documents are
in Spanish on purpose: the model is multilingual and the demo video is in
Spanish. Rendered with headless Chrome.

Usage: python tools/generate_demo_files.py
"""
import random
import subprocess
import tempfile
from pathlib import Path

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"  # change on Windows/Linux
ROOT = Path(__file__).resolve().parent.parent / "demo-folder"
DOCS_DIR = ROOT / "Documents"
SCREENSHOTS_DIR = ROOT / "Screenshots"
SCANS_DIR = ROOT / "Scans"
TMP = Path(tempfile.mkdtemp(prefix="gen-docs-"))

BASE_CSS = """
*{box-sizing:border-box} body{font-family:-apple-system,'Helvetica Neue',Arial,sans-serif;color:#1d1d1f;margin:0}
.doc{padding:56px 64px;font-size:13.5px;line-height:1.55}
h1{font-size:26px;margin:0 0 4px} h2{font-size:16px;margin:28px 0 8px} .muted{color:#6e6e73}
table{width:100%;border-collapse:collapse;margin-top:14px} th,td{text-align:left;padding:9px 8px;border-bottom:1px solid #e5e5ea}
th{font-size:11.5px;text-transform:uppercase;letter-spacing:.04em;color:#6e6e73} .r{text-align:right}
.total td{font-weight:700;font-size:15px;border-bottom:none} .head{display:flex;justify-content:space-between;align-items:flex-start}
.logo{font-weight:800;font-size:22px;letter-spacing:-.02em} .box{border:1px solid #e5e5ea;border-radius:10px;padding:14px 16px}
.page{page-break-after:always}
"""


def chrome(*args: str) -> None:
    """Runs headless Chrome. It occasionally hangs, so each call gets a timeout and retries."""
    for attempt in range(3):
        try:
            subprocess.run([CHROME, "--headless=new", "--disable-gpu", *args], check=True, capture_output=True, timeout=60)
            return
        except subprocess.TimeoutExpired:
            print(f"  Chrome timed out, retrying ({attempt + 1}/3)", flush=True)
    raise RuntimeError("headless Chrome kept timing out")


def html(body: str, css: str = "") -> str:
    return f"<!doctype html><html><head><meta charset='utf-8'><style>{BASE_CSS}{css}</style></head><body>{body}</body></html>"


def render_pdf(name: str, content: str) -> None:
    f = TMP / "x.html"
    f.write_text(content)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    chrome("--no-pdf-header-footer", f"--print-to-pdf={DOCS_DIR / name}", f.as_uri())


def screenshot(name: str, content: str, width=1512, height=945, folder=None) -> Path:
    f = TMP / "x.html"
    f.write_text(content)
    folder = folder or SCREENSHOTS_DIR
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / name
    chrome("--hide-scrollbars", "--force-device-scale-factor=1", f"--window-size={width},{height}",
           f"--screenshot={out}", f.as_uri())
    return out


def bars(values, color="#0a84ff", height=180, width=520, labels=None):
    """A small SVG bar chart."""
    top = max(values)
    w = width / len(values)
    rects = []
    for i, v in enumerate(values):
        h = v / top * (height - 30)
        rects.append(f"<rect x='{i*w+8}' y='{height-20-h}' width='{w-16}' height='{h}' rx='4' fill='{color}'/>")
        if labels:
            rects.append(f"<text x='{i*w+w/2}' y='{height-4}' font-size='11' text-anchor='middle' fill='#6e6e73'>{labels[i]}</text>")
    return f"<svg width='{width}' height='{height}'>{''.join(rects)}</svg>"


# ------------------------------------------------------------------ PDFs

def invoice(name, issuer, tax_id, customer, number, date, lines, note=""):
    """A Spanish invoice (factura) with VAT."""
    rows = "".join(f"<tr><td>{d}</td><td class=r>{q}</td><td class=r>{p:,.2f} €</td><td class=r>{q*p:,.2f} €</td></tr>" for d, q, p in lines)
    base = sum(q * p for _, q, p in lines)
    vat = base * 0.21
    render_pdf(name, html(f"""<div class=doc>
<div class=head><div><div class=logo>{issuer}</div><div class=muted>CIF {tax_id}<br>Calle Ficticia 12, 08005 Barcelona</div></div>
<div class=r><h1>FACTURA</h1><div class=muted>Nº {number}<br>Fecha: {date}</div></div></div>
<div class=box style='margin-top:28px;width:50%'><b>Cliente</b><br>{customer}</div>
<table><tr><th>Concepto</th><th class=r>Cant.</th><th class=r>Precio</th><th class=r>Importe</th></tr>{rows}
<tr><td colspan=3 class=r>Base imponible</td><td class=r>{base:,.2f} €</td></tr>
<tr><td colspan=3 class=r>IVA 21 %</td><td class=r>{vat:,.2f} €</td></tr>
<tr class=total><td colspan=3 class=r>TOTAL</td><td class=r>{base+vat:,.2f} €</td></tr></table>
<p class=muted style='margin-top:30px'>{note}</p></div>"""))


def generate_pdfs():
    invoice("factura_2026_0931.pdf", "NubeRack Hosting", "B-12345678", "Laboratorio Demo S.L.", "NR-2026-0931", "01/10/2026",
            [("Servidor dedicado CX-42 (8 vCPU, 32 GB RAM, 240 GB NVMe) — septiembre", 1, 38.90),
             ("Dirección IPv4 adicional", 1, 0.60), ("Copias de seguridad automáticas (20 %)", 1, 7.78),
             ("Tráfico saliente adicional (0 TB)", 0, 1.00)],
            "Pago domiciliado. El servidor se factura por meses naturales.")
    invoice("FAC-000214.pdf", "Electra Mediterránea", "A-87654321", "Laboratorio Demo S.L.", "EM-000214", "12/09/2026",
            [("Término de potencia 4,6 kW · 31 días", 31, 0.11), ("Energía consumida 412 kWh", 412, 0.14), ("Alquiler de contador", 1, 0.81)],
            "Factura de suministro eléctrico. Lectura real del contador.")
    invoice("presupuesto_v3_final.pdf", "Reformas Garbí", "B-11223344", "Marta Puig (ficticio)", "P-2026-077", "03/09/2026",
            [("Demolición de cocina existente y retirada de escombros", 1, 640), ("Muebles de cocina en blanco mate, 4,2 m lineales", 1, 3850),
             ("Encimera de cuarzo gris", 1, 1290), ("Fontanería y electricidad", 1, 980), ("Alicatado con azulejo metro blanco", 18, 42)],
            "Presupuesto de reforma de cocina. Válido 30 días. Plazo estimado: 3 semanas.")

    render_pdf("doc_scan_0007.pdf", html("""<div class=doc>
<h1>Contrato de arrendamiento de vivienda</h1><div class=muted>En Barcelona, a 1 de septiembre de 2026 · Documento de ejemplo, personas ficticias</div>
<h2>Reunidos</h2><p>De una parte, D. Jordi Ferrer (arrendador, ficticio), y de otra, Dña. Laura Sanz (arrendataria, ficticia), reconociéndose capacidad legal suficiente para obligarse.</p>
<h2>Primera — Objeto</h2><p>El arrendador cede en arrendamiento la vivienda situada en Calle del Ejemplo 44, 3º 2ª, de 78 m², con dos dormitorios, cocina equipada y terraza, para uso exclusivo como vivienda habitual.</p>
<h2>Segunda — Duración</h2><p>El contrato tendrá una duración de un año, prorrogable conforme a la normativa vigente.</p>
<h2>Tercera — Renta</h2><p>La renta mensual se fija en 1.150 euros, pagadera por transferencia en los cinco primeros días de cada mes.</p>
<h2>Cuarta — Fianza</h2><p>La arrendataria entrega en este acto una mensualidad en concepto de fianza.</p>
<div class=page></div>
<h2>Quinta — Gastos y suministros</h2><p>Los suministros de agua, luz, gas e internet serán por cuenta de la arrendataria. La comunidad y el IBI, del arrendador.</p>
<h2>Sexta — Mascotas</h2><p>Se permite la tenencia de un perro de raza pequeña, siendo la arrendataria responsable de los daños que pudiera causar.</p>
<h2>Séptima — Obras</h2><p>No se podrán realizar obras sin consentimiento escrito del arrendador.</p>
<p style='margin-top:60px'>Firmado: ____________________ &nbsp;&nbsp;&nbsp;&nbsp; ____________________</p></div>"""))

    render_pdf("boarding_pass.pdf", html("""<div class=doc>
<div class=box style='border-radius:18px;padding:0;overflow:hidden;width:640px'>
<div style='background:#0b3d91;color:#fff;padding:18px 24px;display:flex;justify-content:space-between'><b style='font-size:20px'>AEROMAR ✈ (aerolínea ficticia)</b><span>TARJETA DE EMBARQUE</span></div>
<div style='padding:22px 24px;display:flex;justify-content:space-between;font-size:15px'>
<div><div class=muted>Pasajero</div><b>RISCO / DEMO</b></div><div><div class=muted>Vuelo</div><b>AM 1734</b></div><div><div class=muted>Fecha</div><b>24 OCT 2026</b></div></div>
<div style='padding:0 24px 22px;display:flex;align-items:center;gap:30px'><div style='font-size:52px;font-weight:800'>BCN</div><div style='font-size:28px'>→</div><div style='font-size:52px;font-weight:800'>LIS</div>
<div class=muted>Barcelona El Prat<br>a Lisboa Humberto Delgado</div></div>
<div style='padding:14px 24px;border-top:1px dashed #ccc;display:flex;justify-content:space-between'><span>Puerta <b>B34</b></span><span>Embarque <b>07:25</b></span><span>Asiento <b>14C</b></span><span>Grupo <b>2</b></span></div>
<div style='padding:10px 24px 24px'><div style='height:60px;background:repeating-linear-gradient(90deg,#000 0 2px,#fff 2px 4px,#000 4px 7px,#fff 7px 9px)'></div></div></div></div>"""))

    render_pdf("reserva_confirmada.pdf", html("""<div class=doc><div class=logo>Hotel Miradouro (ficticio)</div><div class=muted>Lisboa · Alfama</div>
<h1 style='margin-top:30px'>Confirmación de reserva</h1>
<table><tr><td>Entrada</td><td>Sábado, 24 de octubre de 2026 (desde las 15:00)</td></tr><tr><td>Salida</td><td>Martes, 27 de octubre de 2026 (antes de las 11:00)</td></tr>
<tr><td>Habitación</td><td>Doble superior con vistas al río Tajo, desayuno incluido</td></tr><tr><td>Huéspedes</td><td>2 adultos</td></tr><tr class=total><td>Total</td><td>486,00 € (pagado)</td></tr></table>
<p class=muted style='margin-top:24px'>Cancelación gratuita hasta 48 horas antes de la llegada. El tranvía 28 para a 200 metros del hotel.</p></div>"""))

    render_pdf("Q3_informe.pdf", html(f"""<div class=doc><div class=logo>Tienda Online Demo</div><h1 style='margin-top:20px'>Informe de ventas · tercer trimestre 2026</h1>
<p>Las ventas han caído por tercer mes consecutivo. La bajada se concentra en la categoría de calzado, mientras que los accesorios se mantienen estables.</p>
<h2>Ventas mensuales (miles de euros)</h2>{bars([148, 141, 132, 118, 97, 84], '#ff453a', labels=['abr','may','jun','jul','ago','sep'])}
<h2>Conclusiones</h2><p>1) Revisar precios de calzado frente a la competencia. 2) Reactivar la newsletter, que lleva dos meses sin enviarse. 3) Lanzar campaña de Black Friday antes de lo previsto.</p></div>"""))

    render_pdf("Q3_report_EN.pdf", html(f"""<div class=doc><div class=logo>Demo Analytics Inc.</div><h1 style='margin-top:20px'>Quarterly growth report — Q3 2026</h1>
<p>Monthly active users grew steadily for the sixth month in a row, driven by the new mobile app.</p>
<h2>Monthly active users (thousands)</h2>{bars([42, 48, 55, 63, 74, 88], '#30d158', labels=['Apr','May','Jun','Jul','Aug','Sep'])}
<p>Churn decreased from 6.1% to 4.3%. Next quarter we will focus on onboarding and pricing experiments.</p></div>"""))

    render_pdf("manual_es.pdf", html("""<div class=doc><h1>Cafetera espresso EX-200 (ficticia)</h1><div class=muted>Manual de usuario · Guía rápida</div>
<svg width='300' height='260' style='float:right;margin:10px'><rect x='60' y='20' width='180' height='200' rx='16' fill='none' stroke='#1d1d1f' stroke-width='4'/>
<rect x='100' y='120' width='100' height='14' fill='#1d1d1f'/><rect x='125' y='150' width='50' height='45' rx='4' fill='none' stroke='#1d1d1f' stroke-width='3'/>
<circle cx='210' cy='60' r='14' fill='none' stroke='#1d1d1f' stroke-width='3'/><text x='150' y='250' font-size='12' text-anchor='middle'>Fig. 1 — Partes de la cafetera</text></svg>
<h2>1. Primer uso</h2><p>Llene el depósito de agua hasta la marca MAX. Encienda la cafetera con el botón superior derecho y espere a que la luz verde deje de parpadear.</p>
<h2>2. Preparar un café</h2><p>Coloque el portafiltro con café molido, presione ligeramente y gire hacia la derecha. Pulse el botón de una taza o de dos tazas.</p>
<h2>3. Descalcificación</h2><p>Cada dos meses, o cuando parpadee la luz naranja, haga pasar una solución descalcificadora por el circuito.</p>
<h2>4. Problemas frecuentes</h2><p>Si el café sale frío, precaliente la taza. Si no sale agua, compruebe que el depósito está bien encajado.</p></div>"""))

    render_pdf("acta_reunion.pdf", html("""<div class=doc><h1>Acta de reunión del equipo de producto</h1><div class=muted>Lunes 5 de octubre de 2026 · Asistentes: Ana, Pau, Lucía, Marc (ficticios)</div>
<h2>Temas tratados</h2><p>1. Migración de la web al servidor propio: Pau confirma que el despliegue con Coolify funciona y que el coste baja a menos de la mitad.</p>
<p>2. Buscador interno: Lucía propone probar un modelo de embeddings para buscar en las fotos del catálogo sin etiquetarlas a mano.</p>
<p>3. Vacaciones de Navidad: se cierra el calendario de guardias antes del 15 de noviembre.</p>
<h2>Tareas</h2><table><tr><th>Tarea</th><th>Responsable</th><th>Fecha</th></tr><tr><td>Copias de seguridad diarias del servidor</td><td>Pau</td><td>12/10</td></tr>
<tr><td>Prototipo del buscador de fotos</td><td>Lucía</td><td>19/10</td></tr><tr><td>Calendario de guardias</td><td>Ana</td><td>15/11</td></tr></table></div>"""))

    render_pdf("receta.pdf", html("""<div class=doc><h1>Paella valenciana de la abuela</h1><div class=muted>Para 4 personas · 1 hora</div>
<h2>Ingredientes</h2><p>400 g de arroz bomba, 1 pollo troceado, medio conejo, 250 g de judía verde plana, 150 g de garrofó, 1 tomate rallado, aceite de oliva, pimentón dulce, azafrán, romero y sal.</p>
<h2>Preparación</h2><p>Dorar la carne en la paella con aceite y sal. Añadir la verdura y sofreír. Incorporar el tomate y el pimentón, cubrir con agua y dejar hervir 25 minutos. Añadir el arroz repartido en cruz, el azafrán y cocer 18 minutos sin remover. Reposar 5 minutos con un paño por encima.</p></div>"""))

    render_pdf("poliza_2026.pdf", html("""<div class=doc><div class=logo>Seguros Tramuntana (ficticio)</div><h1 style='margin-top:20px'>Póliza de seguro de automóvil · Todo riesgo con franquicia</h1>
<table><tr><td>Vehículo</td><td>Turismo eléctrico, matrícula 0000-DEM</td></tr><tr><td>Vigencia</td><td>01/03/2026 – 28/02/2027</td></tr>
<tr><td>Franquicia</td><td>300 € por siniestro</td></tr><tr><td>Asistencia en carretera</td><td>Desde el kilómetro 0, 24 horas</td></tr><tr class=total><td>Prima anual</td><td>612,40 €</td></tr></table>
<h2>Coberturas principales</h2><p>Responsabilidad civil obligatoria y voluntaria, lunas, robo, incendio, daños propios con franquicia, vehículo de sustitución hasta 7 días.</p></div>"""))

    render_pdf("certificado.pdf", html("""<div class=doc style='text-align:center;padding-top:120px'><div class=muted style='letter-spacing:.3em'>CERTIFICADO DE APROVECHAMIENTO</div>
<h1 style='font-size:40px;margin:30px 0'>Curso de Inteligencia Artificial para Directivos</h1><p style='font-size:18px'>Se certifica que <b>Participante de Ejemplo</b> ha completado 20 horas de formación.</p>
<p class=muted style='margin-top:80px'>Barcelona, junio de 2026 · Escuela ficticia</p></div>"""))


# ------------------------------------------------------------------ screenshots

APP = """body{background:#f5f5f7} .top{height:52px;background:#fff;border-bottom:1px solid #e5e5ea;display:flex;align-items:center;padding:0 24px;gap:28px;font-size:14px}
.side{position:fixed;left:0;top:52px;bottom:0;width:220px;background:#fbfbfd;border-right:1px solid #e5e5ea;padding:20px;font-size:14px;line-height:2.4;color:#3a3a3c}
.main{margin-left:220px;padding:32px 40px} .card{background:#fff;border-radius:14px;padding:22px 26px;box-shadow:0 1px 2px #0001;margin-bottom:18px}
.big{font-size:40px;font-weight:700} .kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}"""


def generate_screenshots():
    rnd = random.Random(3)  # fixed seed: same file names on every run

    def name():
        d = rnd.randint(1, 16)
        return f"Screenshot 2026-10-{d:02d} at {rnd.randint(8,23)}.{rnd.randint(0,59):02d}.{rnd.randint(0,59):02d}.png"

    pages = [
        html(f"""<div class=top><b>NubeRack</b><span>Servidores</span><span>Facturación</span><span>Soporte</span></div>
<div class=side>Panel<br>Servidores<br><b>Facturación</b><br>Copias de seguridad<br>Red<br>API</div>
<div class=main><h1>Facturación</h1><div class=kpis><div class=card>Gasto este mes<div class=big>58,82 €</div></div><div class=card>Próxima factura<div class=big>01/11</div></div><div class=card>Servidores activos<div class=big>1</div></div></div>
<div class=card><table><tr><th>Factura</th><th>Fecha</th><th>Concepto</th><th class=r>Total</th></tr><tr><td>NR-2026-0931</td><td>01/10/2026</td><td>Servidor dedicado CX-42</td><td class=r>57,08 €</td></tr>
<tr><td>NR-2026-0812</td><td>01/09/2026</td><td>Servidor dedicado CX-42</td><td class=r>57,08 €</td></tr><tr><td>NR-2026-0698</td><td>01/08/2026</td><td>Servidor dedicado CX-22</td><td class=r>29,40 €</td></tr></table></div></div>""", APP),
        html("""<div style='background:#1e1e1e;color:#d4d4d4;font:15px/1.6 Menlo,monospace;height:945px;padding:24px'>
<span style='color:#6a9955'>~/proyectos/tienda</span> $ npm run build<br><br>&gt; tienda@1.4.0 build<br>&gt; next build<br><br>
<span style='color:#f14c4c'>Error: Cannot find module '@/lib/pagos'</span><br>&nbsp;&nbsp;at Object.&lt;anonymous&gt; (app/checkout/page.tsx:3:1)<br><br>
<span style='color:#f14c4c'>Build failed because of webpack errors</span><br><span style='color:#6a9955'>~/proyectos/tienda</span> $ █</div>"""),
        html("""<div style='background:#1e1e1e;height:945px;display:flex'><div style='width:240px;background:#252526;color:#ccc;font:13px/2 -apple-system;padding:14px'>EXPLORER<br>▾ src<br>&nbsp;&nbsp;buscador.py<br>&nbsp;&nbsp;indice.py<br>&nbsp;&nbsp;servidor.py<br>README.md</div>
<pre style='color:#d4d4d4;font:15px/1.7 Menlo;padding:20px;margin:0'><span style='color:#c586c0'>from</span> sentence_transformers <span style='color:#c586c0'>import</span> SentenceTransformer

modelo = SentenceTransformer(<span style='color:#ce9178'>"google/embeddinggemma-2"</span>)

<span style='color:#569cd6'>def</span> <span style='color:#dcdcaa'>buscar</span>(consulta, vectores, n=<span style='color:#b5cea8'>10</span>):
    q = modelo.encode(consulta, prompt_name=<span style='color:#ce9178'>"SearchQuery"</span>)
    puntos = vectores @ q
    <span style='color:#c586c0'>return</span> puntos.argsort()[::-<span style='color:#b5cea8'>1</span>][:n]</pre></div>"""),
        html("""<div style='background:#efe7de;height:945px;padding:30px 300px;font-size:16px'>
<div style='background:#fff;padding:10px 14px;border-radius:10px;width:fit-content;margin:10px 0'>¿Cenamos el viernes en el japonés de la calle Mallorca? 🍣</div>
<div style='background:#d9fdd3;padding:10px 14px;border-radius:10px;width:fit-content;margin:10px 0 10px auto'>¡Venga! ¿A las 21:00?</div>
<div style='background:#fff;padding:10px 14px;border-radius:10px;width:fit-content;margin:10px 0'>Perfecto, reservo para 4. Trae a Max, que el perro también es bienvenido en la terraza</div>
<div style='background:#d9fdd3;padding:10px 14px;border-radius:10px;width:fit-content;margin:10px 0 10px auto'>Jajaja vale 🐶</div></div>"""),
        html(f"""<div class=top><b>Analíticas</b><span>Resumen</span><span>Contenido</span><span>Audiencia</span></div><div class=side>Resumen<br>Contenido<br>Audiencia<br>Ingresos</div>
<div class=main><h1>Tu canal ha conseguido 48.210 visualizaciones en los últimos 28 días</h1><div class=card>{bars([12,15,14,18,22,25,31,36,41,48],'#ff0033',width=900,height=260)}</div></div>""", APP),
        html("""<div style='padding:30px;font-size:13px'><h2>Gastos del viaje a Lisboa</h2><table style='font-size:14px'><tr><th>Fecha</th><th>Concepto</th><th>Categoría</th><th class=r>Importe</th></tr>
<tr><td>24/10</td><td>Vuelo BCN-LIS</td><td>Transporte</td><td class=r>184,00 €</td></tr><tr><td>24/10</td><td>Hotel Miradouro (3 noches)</td><td>Alojamiento</td><td class=r>486,00 €</td></tr>
<tr><td>25/10</td><td>Cena en Bairro Alto</td><td>Comida</td><td class=r>62,50 €</td></tr><tr><td>26/10</td><td>Tranvía 28 + Sintra</td><td>Transporte</td><td class=r>21,40 €</td></tr><tr class=total><td></td><td>Total</td><td></td><td class=r>753,90 €</td></tr></table></div>"""),
        html("""<div style='padding:40px 200px'><div style='display:flex;gap:40px'><div style='width:520px;height:520px;background:linear-gradient(135deg,#ff9f0a,#ff375f);border-radius:20px;display:flex;align-items:center;justify-content:center;font-size:120px'>👟</div>
<div><div class=muted>Calzado · Running</div><h1 style='font-size:34px'>Zapatillas Aire Pro 3 naranja</h1><div style='font-size:30px;font-weight:700;margin:14px 0'>89,95 € <s class=muted style='font-size:18px'>129,95 €</s></div>
<p>Amortiguación ligera para correr en asfalto. Tallas 38 a 46.</p><div style='background:#1d1d1f;color:#fff;padding:16px;border-radius:12px;text-align:center;margin-top:30px'>Añadir a la cesta</div></div></div></div>"""),
        html("""<div style='padding:50px 120px'><h1 style='text-align:center;font-size:38px'>Elige tu plan</h1><div style='display:grid;grid-template-columns:repeat(3,1fr);gap:24px;margin-top:40px'>
<div class=box style='padding:30px'><h2>Gratis</h2><div class=big>0 €</div><p>1 proyecto<br>100 MB</p></div><div class=box style='padding:30px;border:2px solid #0a84ff'><h2>Pro</h2><div class=big>20 €<span style='font-size:16px'>/mes</span></div><p>Proyectos ilimitados<br>100 GB · Dominio propio</p></div>
<div class=box style='padding:30px'><h2>Equipo</h2><div class=big>99 €<span style='font-size:16px'>/mes</span></div><p>10 usuarios<br>SSO · Soporte prioritario</p></div></div></div>"""),
        html(f"""<div class=top><b>Ventas</b><span>Panel</span></div><div class=side>Panel<br>Pedidos<br>Clientes</div><div class=main><h1>Ventas por mes</h1>
<div class=kpis><div class=card>Ventas septiembre<div class=big style='color:#ff453a'>84.120 €</div>▼ 13 % vs agosto</div><div class=card>Pedidos<div class=big>1.304</div></div><div class=card>Ticket medio<div class=big>64,5 €</div></div></div>
<div class=card>{bars([148,141,132,118,97,84],'#ff453a',width=900,height=280,labels=['abr','may','jun','jul','ago','sep'])}</div></div>""", APP),
        html("""<div style='padding:30px;display:grid;grid-template-columns:repeat(3,1fr);gap:20px;background:#0079bf;height:945px'>
<div style='background:#ebecf0;border-radius:10px;padding:12px'><b>Por hacer</b><div class=card style='margin-top:10px'>Configurar copias de seguridad</div><div class=card>Revisar factura del hosting</div></div>
<div style='background:#ebecf0;border-radius:10px;padding:12px'><b>En curso</b><div class=card style='margin-top:10px'>Buscador de fotos con embeddings</div></div>
<div style='background:#ebecf0;border-radius:10px;padding:12px'><b>Hecho</b><div class=card style='margin-top:10px'>Migrar web a Coolify</div><div class=card>Comprar dominio</div></div></div>"""),
        html("""<div style='height:945px;display:flex;align-items:center;justify-content:center;flex-direction:column;background:#fff'><div style='font-size:140px;font-weight:800;color:#e5e5ea'>404</div><p style='font-size:22px'>Esta página no existe… o se la ha comido el perro.</p></div>"""),
        html("""<div style='padding:30px 60px'><div class=muted>De: reservas@aeromar.example · Para: tú</div><h1>Tu vuelo a Lisboa está confirmado ✈</h1>
<p>Localizador <b>Q7XK2M</b>. Sábado 24 de octubre, salida de Barcelona a las 08:05, llegada a Lisboa a las 09:15 (hora local).</p><p>Puedes hacer el check-in online desde 48 horas antes.</p></div>"""),
        html("""<div style='padding:30px'><h2>Calendario · semana 42</h2><div style='display:grid;grid-template-columns:repeat(5,1fr);gap:10px;font-size:14px'>
<div class=box><b>Lunes</b><p style='background:#e5f1ff;padding:6px;border-radius:6px'>10:00 Reunión de producto</p></div><div class=box><b>Martes</b><p style='background:#ffe9e5;padding:6px;border-radius:6px'>17:00 Dentista</p></div>
<div class=box><b>Miércoles</b><p style='background:#e8f8ec;padding:6px;border-radius:6px'>Grabar vídeo YouTube</p></div><div class=box><b>Jueves</b><p style='background:#e5f1ff;padding:6px;border-radius:6px'>12:00 Llamada con inversores</p></div>
<div class=box><b>Viernes</b><p style='background:#fff4d6;padding:6px;border-radius:6px'>21:00 Cena japonés</p></div></div></div>"""),
        html("""<div style='padding:40px 300px;background:#f5f5f7;height:945px'><div class=card><b>@demo_dev</b> <span class=muted>· 2 h</span><p style='font-size:19px'>He metido toda mi carpeta de capturas en un buscador local y ahora encuentro las cosas escribiendo lo que salía en ellas. Ni un nombre de archivo. Esto es brujería.</p><span class=muted>♥ 1.204 · ↻ 318</span></div></div>"""),
        html("""<div style='padding:40px 400px'><h2>Wi-Fi</h2><div class=card><b>Wi-Fi</b><span style='float:right;background:#30d158;color:#fff;padding:2px 10px;border-radius:10px'>Activado</span></div>
<div class=card>✓ Casa_5G</div><div class=card>Oficina-Invitados 🔒</div><div class=card>Cafeteria_Free</div></div>"""),
    ]
    for p in pages:
        screenshot(name(), p)

    # A "scanned" supermarket receipt: a tilted, grainy image inside a PDF with no text layer.
    t = screenshot("_receipt.png", html("""<div style='width:420px;margin:20px auto;font:15px/1.6 Menlo,monospace;padding:20px'>
<div style='text-align:center'><b>SUPERMERCADO EL PUERTO</b><br>(ficticio) · NIF B-00000000<br>12/10/2026 19:42</div><br>
LECHE ENTERA 1L x2 ....... 2,30<br>PAN DE PAYÉS ............ 1,95<br>TOMATES RAMA 1KG ........ 2,49<br>ACEITE OLIVA 1L ......... 8,90<br>
CAFÉ MOLIDO 250G ........ 3,75<br>COMIDA PERRO 3KG ........ 9,80<br><br><b>TOTAL ................. 29,19 €</b><br>TARJETA ****1234<br><br><div style='text-align:center'>GRACIAS POR SU VISITA</div></div>"""), 480, 560, TMP)
    from PIL import Image, ImageFilter
    import numpy as np
    im = Image.open(t).convert("L").rotate(2.5, expand=True, fillcolor=235)
    arr = np.array(im).astype("float32") * 0.92 + np.random.default_rng(1).normal(0, 9, im.size[::-1])
    im = Image.fromarray(arr.clip(0, 255).astype("uint8")).filter(ImageFilter.GaussianBlur(0.6))
    SCANS_DIR.mkdir(parents=True, exist_ok=True)
    im.convert("RGB").save(SCANS_DIR / "SCAN_20261012_0001.pdf", resolution=150)


if __name__ == "__main__":
    generate_pdfs()
    generate_screenshots()
    print("PDFs:", len(list(DOCS_DIR.glob("*.pdf"))), "Screenshots:", len(list(SCREENSHOTS_DIR.glob("*.png"))))
