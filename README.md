# Buscador local de fotos y PDFs con EmbeddingGemma 2

Eliges una carpeta, se indexa en tu ordenador y buscas escribiendo **lo que sale dentro** de las fotos, las capturas de pantalla y los PDFs. No importa cómo se llamen los archivos.

- «la factura del servidor» → la captura del panel de facturación y el PDF de la factura
- «perro dormido en el sofá» → fotos de perros en sofás (también en inglés: «dog sleeping on the couch»)
- «ticket del supermercado» → un ticket **escaneado** que no tiene ni una letra de texto dentro del PDF

Funciona sin internet: solo hace falta conexión la primera vez, para descargar el modelo. Después puedes quitar el wifi.

El repo incluye una carpeta de prueba con 303 fotos, 15 capturas y 14 PDFs para probarlo en 5 minutos.

## Requisitos

- Python 3.11, 3.12 o 3.13
- ~3 GB libres: 1,4 GB del modelo y ~1 GB de PyTorch
- Probado en un Mac con Apple Silicon (usa la GPU con Metal). En Windows y Linux funciona con GPU NVIDIA (CUDA) o, más despacio, con CPU.

## Instalación

```bash
git clone https://github.com/ericrisco/buscador-local.git
cd buscador-local/app

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

```bash
python buscador.py
```

Se abre una ventana con el buscador. Si tienes Chrome o Edge, es una ventana de app limpia; si no, una pestaña del navegador en `http://127.0.0.1:8765`.

1. La primera vez descarga el modelo `google/embeddinggemma-2` (1,4 GB, sin cuenta de Hugging Face).
2. Pulsa **Elegir carpeta** y elige `carpeta-demo` (o cualquier carpeta tuya) → **Indexar**.
3. Escribe lo que recuerdas y pulsa Enter.

En Mac también puedes abrirlo con doble clic en `app/Abrir buscador.command`, y en Windows con `app/abrir_buscador.bat`, una vez instalado.

### Qué puedes hacer

| Acción | Cómo |
|---|---|
| Buscar por texto | Escribe en la caja. En español, inglés u otros idiomas |
| Buscar con una imagen | Botón **Con una imagen**, o arrastra una foto encima de la ventana |
| Ver fotos parecidas | Abre un resultado → **Buscar parecidas** |
| Filtrar | Todo · Fotos · Capturas · PDFs |
| Ver el PDF en la página exacta | Clic en un resultado de PDF |
| Ver la puntuación de cada resultado | Tecla `s` |
| Ir a la caja de búsqueda | Tecla `/` |

### Búsquedas para probar con `carpeta-demo`

- la factura del servidor
- perro dormido en el sofá
- ticket del supermercado
- algo para desayunar
- gráfico de ventas que bajan
- billete de avión a Lisboa
- cómo quitar la cal de la cafetera
- contrato del piso
- una montaña nevada
- un error al compilar

## Cómo funciona

EmbeddingGemma 2 convierte texto e imágenes en vectores de 768 números **en el mismo espacio**: una frase y una foto que hablan de lo mismo quedan cerca. Buscar es convertir lo que escribes en otro vector y traer los más cercanos.

**Al indexar:**

- **Fotos y capturas:** cada imagen se reduce a 1024 px y pasa por el encoder de imagen → 1 vector.
- **PDFs:** cada página se indexa dos veces:
  - la página renderizada como imagen → 1 vector (así salen los escaneados y las páginas con gráficos);
  - su texto, en trozos de ~700 caracteres → 1 vector por trozo (encoder de texto, prompt `Document`).
- Todo se guarda en `app/indices/<hash de la carpeta>/`: `vectores.npy`, `indice.json` y las miniaturas.

**Al buscar:**

1. Lo que escribes se convierte en un vector (prompt `SearchQuery`).
2. Se compara con todos los vectores del índice (producto escalar, están normalizados).
3. Los parecidos texto↔texto y texto↔imagen salen en escalas distintas. Para poder mezclarlos en una sola lista, cada tipo se estandariza por separado (puntuación z) y cada foto o página se queda con su mejor vector.

**Detalles:**

- Solo se cargan los encoders de texto e imagen: 440 M de parámetros en vez de 740 M, porque el de audio no hace falta.
- El servidor solo escucha en `127.0.0.1`: no es accesible desde otro equipo de la red.
- Nada sale de tu ordenador. El indicador de la esquina comprueba si hay conexión solo para enseñártelo.

### Cifras medidas

En un MacBook Pro M4 Pro con 24 GB:

| | |
|---|---|
| Indexar `carpeta-demo` (303 fotos, 15 capturas, 14 PDFs) | ~3 min |
| RAM durante el indexado | ~1 GB |
| Una búsqueda | 0,03-0,5 s |

Son las de un equipo concreto. Con CPU será bastante más lento.

## Limitaciones

- Es una **demo**, no un producto: no vigila la carpeta. Si añades archivos, vuelve a indexar.
- La puntuación es **parecido**, no probabilidad de acierto. Con búsquedas ambiguas o con matices puede fallar; la ficha del modelo avisa de diferencias de rendimiento entre idiomas.
- Un PDF muy largo genera muchas páginas que indexar. Cuenta con ~0,5 s por página en un Mac M4.
- Vídeo y audio no están incluidos, aunque el modelo los soporta.

## Estructura

```
app/
  buscador.py          servidor local + indexado + búsqueda
  web/index.html       la interfaz (un solo archivo, sin dependencias)
  requirements.txt
  Abrir buscador.command / abrir_buscador.bat
carpeta-demo/          303 fotos, 15 capturas, 14 PDFs (todo ficticio o con licencia libre)
herramientas/          scripts que generaron la carpeta de demo + procedencia de las fotos
CREDITOS.md            autor y licencia de cada foto
```

## Créditos y licencia

- Código: [MIT](LICENSE).
- Modelo: [EmbeddingGemma 2](https://ai.google.dev/gemma/docs/embeddinggemma/model_card_2), de Google, licencia Apache 2.0. No se incluye en el repo: se descarga de Hugging Face.
- Fotos: licencias CC BY, CC BY-SA, CC0 y Unsplash. Autor y enlace de cada una en [CREDITOS.md](CREDITOS.md).
- Capturas y PDFs: generados para la demo. Empresas, personas e importes son inventados.
