from __future__ import annotations

import json
import re
from pathlib import Path

from pypdf import PdfReader

PATRON_ID_ARXIV = re.compile(r"(\d{4}\.\d{5}(?:v\d+)?)")


def _extraer_texto_pdf(ruta_pdf: Path, max_paginas: int) -> str:
    lector = PdfReader(str(ruta_pdf))
    textos: list[str] = []
    paginas = min(max_paginas, len(lector.pages))

    for indice in range(paginas):
        texto = lector.pages[indice].extract_text() or ""
        texto = " ".join(texto.split())
        if texto:
            textos.append(texto)

    return "\n".join(textos).strip()


def _titulo_desde_nombre_archivo(nombre_archivo: str) -> str:
    sin_extension = Path(nombre_archivo).stem
    sin_id = PATRON_ID_ARXIV.sub("", sin_extension)
    titulo = re.sub(r"\s+", " ", sin_id).strip(" -_.")
    return titulo or sin_extension


def _id_desde_nombre_archivo(nombre_archivo: str) -> str:
    coincidencia = PATRON_ID_ARXIV.search(nombre_archivo)
    if coincidencia:
        return coincidencia.group(1)
    return Path(nombre_archivo).stem


def construir_corpus_local_desde_pdfs(
    ruta_carpeta_pdfs: Path,
    ruta_salida_jsonl: Path,
    max_paginas_por_pdf: int = 2,
    max_documentos: int | None = None,
) -> int:
    ruta_salida_jsonl.parent.mkdir(parents=True, exist_ok=True)

    archivos_pdf = sorted(ruta_carpeta_pdfs.glob("*.pdf"))
    if max_documentos is not None:
        archivos_pdf = archivos_pdf[:max_documentos]

    cantidad = 0
    with ruta_salida_jsonl.open("w", encoding="utf-8") as archivo_salida:
        for ruta_pdf in archivos_pdf:
            try:
                texto = _extraer_texto_pdf(ruta_pdf, max_paginas=max_paginas_por_pdf)
            except Exception:
                continue

            if not texto:
                continue

            id_documento = _id_desde_nombre_archivo(ruta_pdf.name)
            titulo = _titulo_desde_nombre_archivo(ruta_pdf.name)
            resumen = texto[:2000]

            registro = {
                "id_documento": id_documento,
                "titulo": titulo,
                "resumen": resumen,
                "autores": [],
                "categorias": ["investigacion_cientifica", "academico"],
                "publicado": "",
                "actualizado": "",
                "url_pdf": "",
                "ruta_pdf_local": str(ruta_pdf),
                "fuente": "local_pdf",
            }

            archivo_salida.write(json.dumps(registro, ensure_ascii=False) + "\n")
            cantidad += 1

    return cantidad
