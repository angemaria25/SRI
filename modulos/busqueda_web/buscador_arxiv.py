from __future__ import annotations

import json
from pathlib import Path

from modulos.adquisicion_datos.api_arxiv import PoliticasCrawling, RecolectorArxiv


def _consulta_a_arxiv(consulta_usuario: str) -> str:
    consulta_limpia = " ".join(consulta_usuario.strip().split())
    if not consulta_limpia:
        consulta_limpia = "investigacion cientifica"
    return f"all:{consulta_limpia}"


def buscar_en_arxiv(
    consulta_usuario: str,
    total_resultados: int = 30,
    tamano_lote: int = 30,
) -> list[dict]:
    # Se consulta la API oficial de arXiv para respaldo de resultados.
    recolector = RecolectorArxiv(
        politicas=PoliticasCrawling(validar_robots_txt=False)
    )
    consulta_api = _consulta_a_arxiv(consulta_usuario)

    resultados: list[dict] = []
    for documento in recolector.obtener(
        consulta=consulta_api,
        total_resultados=total_resultados,
        tamano_lote=tamano_lote,
    ):
        resultados.append(documento.a_diccionario())

    return resultados


def guardar_resultados_web_jsonl(ruta_salida: Path, resultados: list[dict]) -> None:
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    with ruta_salida.open("w", encoding="utf-8") as archivo:
        for registro in resultados:
            archivo.write(json.dumps(registro, ensure_ascii=False) + "\n")
