from __future__ import annotations

import argparse
import json
from pathlib import Path

from modulos.adquisicion_datos.constructor_corpus import (
    cargar_corpus_jsonl,
    generar_estadisticas_corpus,
    guardar_estadisticas_corpus,
)
from modulos.adquisicion_datos.corpus_local import construir_corpus_local_desde_pdfs
from modulos.base_vectorial.base_vectorial import BaseVectorialInicial
from modulos.busqueda_web.buscador_arxiv import buscar_en_arxiv, guardar_resultados_web_jsonl
from modulos.indexacion.indice_invertido import IndiceInvertido
from modulos.recuperacion.modelo_lenguaje import RecuperadorModeloLenguaje


def _guardar_json(ruta: Path, datos: dict | list) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")


def _requiere_respaldo_web(resultados_locales: list[dict], minimo_resultados: int = 3) -> bool:
    if len(resultados_locales) < minimo_resultados:
        return True

    puntajes = [float(item.get("puntaje", 0.0)) for item in resultados_locales]
    if not puntajes:
        return True

    if len(set(round(puntaje, 10) for puntaje in puntajes)) <= 1:
        return True

    return False


def _resultados_vectoriales_insuficientes(resultados_vectoriales: list[dict]) -> bool:
    if not resultados_vectoriales:
        return True

    similitudes = [float(item.get("similitud", 0.0)) for item in resultados_vectoriales]
    return max(similitudes, default=0.0) <= 0.0


def ejecutar_flujo_principal(
    ruta_datos: Path,
    ruta_local: Path,
    consulta_usuario: str,
    max_documentos_locales: int = 3000,
    max_documentos_web: int = 30,
) -> None:
    ruta_corpus_local = ruta_datos / "brutos" / "corpus_local.jsonl"
    ruta_corpus_web = ruta_datos / "brutos" / "corpus_web_arxiv.jsonl"

    ruta_indices = ruta_datos / "indices"
    ruta_vectorial = ruta_datos / "base_vectorial"
    ruta_procesados = ruta_datos / "procesados"

    ruta_pdfs_locales = ruta_local / "local_papers"

    cantidad_local = construir_corpus_local_desde_pdfs(
        ruta_carpeta_pdfs=ruta_pdfs_locales,
        ruta_salida_jsonl=ruta_corpus_local,
        max_paginas_por_pdf=2,
        max_documentos=max_documentos_locales,
    )
    print(f"Corpus local construido: {cantidad_local} documentos")

    documentos_locales = cargar_corpus_jsonl(ruta_corpus_local)

    estadisticas_locales = generar_estadisticas_corpus(documentos_locales)
    guardar_estadisticas_corpus(ruta_procesados / "estadisticas_corpus_local.json", estadisticas_locales)

    indice = IndiceInvertido()
    indice.construir(documentos=documentos_locales)
    indice.guardar(ruta_indices)
    print(f"Indice invertido construido: {indice.cantidad_documentos} documentos")

    base_vectorial = BaseVectorialInicial()
    base_vectorial.construir(documentos=documentos_locales)
    base_vectorial.guardar(ruta_vectorial)
    print("Base vectorial inicial generada")

    recuperador = RecuperadorModeloLenguaje(indice=indice, lambda_parametro=0.2)
    resultados_locales = recuperador.buscar(consulta_usuario, top_k=10)

    resultados_vectoriales = base_vectorial.buscar(consulta_usuario, top_k=10)
    resultados_vectoriales_formato = [
        {"doc_id": doc_id, "similitud": similitud}
        for doc_id, similitud in resultados_vectoriales
    ]

    _guardar_json(ruta_procesados / "resultados_locales_modelo_lenguaje.json", resultados_locales)
    _guardar_json(ruta_procesados / "resultados_locales_vectorial.json", resultados_vectoriales_formato)

    salida = {
        "consulta": consulta_usuario,
        "origen_principal": "local",
        "resultados_locales_modelo_lenguaje": resultados_locales,
        "resultados_locales_vectorial": resultados_vectoriales_formato,
        "respaldo_web_activado": False,
        "error_respaldo_web": None,
        "resultados_web": [],
    }

    if _requiere_respaldo_web(resultados_locales) or _resultados_vectoriales_insuficientes(
        resultados_vectoriales_formato
    ):
        print("Resultados locales insuficientes. Se activa busqueda de respaldo en arXiv...")

        try:
            resultados_web = buscar_en_arxiv(
                consulta_usuario=consulta_usuario,
                total_resultados=max_documentos_web,
                tamano_lote=max_documentos_web,
            )

            guardar_resultados_web_jsonl(ruta_corpus_web, resultados_web)
            _guardar_json(ruta_procesados / "resultados_web_arxiv.json", resultados_web)

            salida["respaldo_web_activado"] = True
            salida["resultados_web"] = resultados_web
        except Exception as error:
            salida["error_respaldo_web"] = str(error)
            print(f"No se pudo completar el respaldo web: {error}")

    _guardar_json(ruta_procesados / "respuesta_sistema.json", salida)
    print(f"Resultados locales modelo lenguaje: {len(resultados_locales)}")
    print(f"Resultados locales vectoriales: {len(resultados_vectoriales_formato)}")
    print(f"Respaldo web activado: {salida['respaldo_web_activado']}")


def _parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PaperScan - Recuperacion sobre corpus local y respaldo web")
    parser.add_argument(
        "--consulta",
        type=str,
        default="recuperacion de informacion en investigacion cientifica",
        help="Consulta de usuario para recuperar documentos",
    )
    parser.add_argument(
        "--max-local",
        type=int,
        default=3000,
        help="Cantidad maxima de PDF locales a procesar",
    )
    parser.add_argument(
        "--max-web",
        type=int,
        default=30,
        help="Cantidad maxima de resultados web en respaldo arXiv",
    )
    return parser.parse_args()


if __name__ == "__main__":
    argumentos = _parsear_argumentos()
    raiz_proyecto = Path(__file__).resolve().parent
    ejecutar_flujo_principal(
        ruta_datos=raiz_proyecto / "datos",
        ruta_local=raiz_proyecto / "local",
        consulta_usuario=argumentos.consulta,
        max_documentos_locales=argumentos.max_local,
        max_documentos_web=argumentos.max_web,
    )
