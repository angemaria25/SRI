from __future__ import annotations

import argparse
import json
from pathlib import Path

from modulos.adquisicion_datos.constructor_corpus import (
    cargar_corpus_jsonl,
    generar_estadisticas_corpus,
    guardar_documentos_jsonl,
    guardar_estadisticas_corpus,
)
from modulos.adquisicion_datos.corpus_local import construir_corpus_local_desde_pdfs
from modulos.base_vectorial.base_vectorial import BaseVectorialInicial, BaseVectorialMejorada
from modulos.busqueda_web.buscador_arxiv import buscar_en_arxiv, guardar_resultados_web_jsonl
from modulos.indexacion.indice_invertido import IndiceInvertido
from modulos.recuperacion.modelo_lenguaje import RecuperadorModeloLenguaje
from modulos.rag import generar_respuesta_rag


def _guardar_json(ruta: Path, datos: dict | list) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")


def _requiere_respaldo_web(resultados_locales: list[dict], consulta: str, indice: IndiceInvertido) -> bool:
    if not resultados_locales:
        return True

    from modulos.indexacion.preprocesamiento import PreprocesadorTexto
    prep = PreprocesadorTexto()
    tokens_consulta = prep.tokenizar(consulta)
    
    palabras_conocidas = [t for t in tokens_consulta if t in indice.indice]
    
    cobertura = len(palabras_conocidas) / len(tokens_consulta) if tokens_consulta else 0
    print(f"DEBUG: Cobertura de vocabulario = {cobertura*100:.1f}%")

    if cobertura < 0.4:
        print("-> MOTIVO: La base de datos local no conoce la mayoría de las palabras de la consulta.")
        return True

    mejor_puntaje = float(resultados_locales[0].get("puntaje", -1000))
    print(f"DEBUG: Mejor puntaje = {mejor_puntaje}")

    if mejor_puntaje < -15.0: 
        print("-> MOTIVO: Puntaje de lenguaje insuficiente.")
        return True

    return False

def _resultados_vectoriales_insuficientes(resultados_vectoriales: list[dict], umbral_similitud: float = 0.12) -> bool:
    if not resultados_vectoriales:
        return True

   
    similitudes = [float(item.get("similitud", 0.0)) for item in resultados_vectoriales]
    mejor_similitud = max(similitudes, default=0.0)

    if mejor_similitud < umbral_similitud:
        print(f"Similitud vectorial insuficiente ({mejor_similitud}).")
        return True
        
    return False

def _convertir_resultados_web(resultados_web: list[dict]) -> list[dict]:
    documentos: list[dict] = []
    for documento in resultados_web:
        doc_id = documento.get("id_documento") or documento.get("id") or ""
        if not doc_id:
            continue
        documentos.append(
            {
                "id_documento": str(doc_id),
                "titulo": documento.get("titulo", ""),
                "resumen": documento.get("resumen", ""),
                "autores": documento.get("autores", []),
                "categorias": documento.get("categorias", []),
                "publicado": documento.get("publicado", ""),
                "actualizado": documento.get("actualizado", ""),
                "url_pdf": documento.get("url_pdf", ""),
                "fuente": "web_arxiv",
            }
        )
    return documentos


def _construir_indices_y_vectores(
    documentos: list[dict],
    ruta_indices: Path,
    ruta_vectorial: Path,
    usar_mejorada: bool = True,
) -> tuple[IndiceInvertido, BaseVectorialInicial | BaseVectorialMejorada]:
    indice = IndiceInvertido()
    indice.construir(documentos=documentos)
    indice.guardar(ruta_indices)

    if usar_mejorada:
        base_vectorial: BaseVectorialInicial | BaseVectorialMejorada = BaseVectorialMejorada()
    else:
        base_vectorial = BaseVectorialInicial()

    base_vectorial.construir(documentos=documentos)
    base_vectorial.guardar(ruta_vectorial)
    return indice, base_vectorial


def ejecutar_flujo_principal(
    ruta_datos: Path,
    ruta_local: Path,
    consulta_usuario: str,
    max_documentos_locales: int = 3000,
    max_documentos_web: int = 30,
) -> None:
    ruta_corpus_local = ruta_datos / "brutos" / "corpus_local.jsonl"
    ruta_corpus_web = ruta_datos / "brutos" / "corpus_web_arxiv.jsonl"
    ruta_corpus_combinado = ruta_datos / "brutos" / "corpus_combinado.jsonl"

    ruta_indices = ruta_datos / "indices"
    ruta_vectorial = ruta_datos / "base_vectorial"
    ruta_vectorial_mejorada = ruta_datos / "base_vectorial_mejorada"
    ruta_indices_combinados = ruta_datos / "indices_combinados"
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

    indice, base_vectorial = _construir_indices_y_vectores(
        documentos=documentos_locales,
        ruta_indices=ruta_indices,
        ruta_vectorial=ruta_vectorial_mejorada,
        usar_mejorada=True,
    )
    print(f"Indice invertido construido: {indice.cantidad_documentos} documentos")
    print("Base vectorial mejorada generada")

    recuperador = RecuperadorModeloLenguaje(
        indice=indice,
        suavizado="dirichlet",
        mu_parametro=2000.0,
    )
    resultados_locales = recuperador.buscar(consulta_usuario, top_k=10)

    resultados_vectoriales = base_vectorial.buscar(consulta_usuario, top_k=10)
    resultados_vectoriales_formato = [
        {"doc_id": doc_id, "similitud": similitud}
        for doc_id, similitud in resultados_vectoriales
    ]

    _guardar_json(ruta_procesados / "resultados_locales_modelo_lenguaje.json", resultados_locales)
    _guardar_json(ruta_procesados / "resultados_locales_vectorial.json", resultados_vectoriales_formato)

    respuesta_rag = generar_respuesta_rag(consulta_usuario, resultados_locales)

    salida = {
        "consulta": consulta_usuario,
        "origen_principal": "local",
        "resultados_locales_modelo_lenguaje": resultados_locales,
        "resultados_locales_vectorial": resultados_vectoriales_formato,
        "respuesta_rag": {
            "respuesta": respuesta_rag.respuesta,
            "citas": respuesta_rag.citas,
            "fragmentos": respuesta_rag.fragmentos,
            "modelo": respuesta_rag.modelo,
        },
        "respaldo_web_activado": False,
        "error_respaldo_web": None,
        "resultados_web": [],
    }

    if _requiere_respaldo_web(resultados_locales, consulta_usuario, indice) or _resultados_vectoriales_insuficientes(
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

            documentos_web = _convertir_resultados_web(resultados_web)
            documentos_combinados = documentos_locales + documentos_web
            guardar_documentos_jsonl(ruta_corpus_combinado, documentos_combinados)

            indice_combinado, base_vectorial_combinada = _construir_indices_y_vectores(
                documentos=documentos_combinados,
                ruta_indices=ruta_indices_combinados,
                ruta_vectorial=ruta_vectorial_mejorada,
                usar_mejorada=True,
            )

            recuperador_combinado = RecuperadorModeloLenguaje(
                indice=indice_combinado,
                suavizado="dirichlet",
                mu_parametro=2000.0,
            )
            resultados_combinados = recuperador_combinado.buscar(consulta_usuario, top_k=10)

            resultados_vectoriales_web = base_vectorial_combinada.buscar(consulta_usuario, top_k=10)
            resultados_vectoriales_web_formato = [
                {"doc_id": doc_id, "similitud": similitud}
                for doc_id, similitud in resultados_vectoriales_web
            ]

            respuesta_rag_combinada = generar_respuesta_rag(consulta_usuario, resultados_combinados)

            salida["respaldo_web_activado"] = True
            salida["resultados_web"] = resultados_web
            salida["origen_principal"] = "combinado"
            salida["resultados_combinados_modelo_lenguaje"] = resultados_combinados
            salida["resultados_combinados_vectorial"] = resultados_vectoriales_web_formato
            salida["respuesta_rag"] = {
                "respuesta": respuesta_rag_combinada.respuesta,
                "citas": respuesta_rag_combinada.citas,
                "fragmentos": respuesta_rag_combinada.fragmentos,
                "modelo": respuesta_rag_combinada.modelo,
            }

            _guardar_json(
                ruta_procesados / "resultados_combinados_modelo_lenguaje.json",
                resultados_combinados,
            )
            _guardar_json(
                ruta_procesados / "resultados_combinados_vectorial.json",
                resultados_vectoriales_web_formato,
            )
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
