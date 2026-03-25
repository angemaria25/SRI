from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from modulos.adquisicion_datos.api_arxiv import RecolectorArxiv


def guardar_corpus_jsonl(
    ruta_salida: Path,
    consulta: str,
    total_resultados: int = 500,
    tamano_lote: int = 100,
) -> int:
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    recolector = RecolectorArxiv()

    cantidad = 0
    with ruta_salida.open("w", encoding="utf-8") as archivo:
        for documento in recolector.obtener(
            consulta=consulta,
            total_resultados=total_resultados,
            tamano_lote=tamano_lote,
        ):
            archivo.write(json.dumps(documento.a_diccionario(), ensure_ascii=False) + "\n")
            cantidad += 1

    return cantidad


def cargar_corpus_jsonl(ruta_entrada: Path) -> list[dict]:
    registros: list[dict] = []
    with ruta_entrada.open("r", encoding="utf-8") as archivo:
        for linea in archivo:
            registros.append(json.loads(linea))
    return registros


def generar_estadisticas_corpus(documentos: list[dict]) -> dict:
    categorias = Counter()
    longitudes_resumen = []
    fechas_publicacion = []

    for documento in documentos:
        resumen = (documento.get("resumen") or documento.get("summary") or "").strip()
        longitudes_resumen.append(len(resumen.split()))

        for categoria in documento.get("categorias", []) or documento.get("categories", []):
            categorias[categoria] += 1

        fecha = documento.get("publicado") or documento.get("published")
        if fecha:
            fechas_publicacion.append(fecha)

    promedio_resumen = 0.0
    if longitudes_resumen:
        promedio_resumen = sum(longitudes_resumen) / len(longitudes_resumen)

    return {
        "cantidad_documentos": len(documentos),
        "promedio_palabras_resumen": round(promedio_resumen, 2),
        "top_categorias": categorias.most_common(10),
        "fecha_publicacion_min": min(fechas_publicacion) if fechas_publicacion else None,
        "fecha_publicacion_max": max(fechas_publicacion) if fechas_publicacion else None,
    }


def guardar_estadisticas_corpus(ruta_salida: Path, estadisticas: dict) -> None:
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    ruta_salida.write_text(json.dumps(estadisticas, ensure_ascii=False, indent=2), encoding="utf-8")
