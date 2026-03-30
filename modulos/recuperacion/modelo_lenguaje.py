from __future__ import annotations

import math
from collections import Counter

from modulos.indexacion.indice_invertido import IndiceInvertido
from modulos.indexacion.preprocesamiento import PreprocesadorTexto


class RecuperadorModeloLenguaje:
    """Modelo probabilistico de lenguaje con suavizado Jelinek-Mercer."""

    def __init__(self, indice: IndiceInvertido, lambda_parametro: float = 0.2) -> None:
        if not (0.0 < lambda_parametro < 1.0):
            raise ValueError("lambda_parametro debe estar en el intervalo (0, 1)")
        self.indice = indice
        self.lambda_parametro = lambda_parametro
        self.preprocesador = PreprocesadorTexto()
        self.conteos_coleccion = self._construir_conteos_coleccion()

    def _construir_conteos_coleccion(self) -> Counter[str]:
        conteos: Counter[str] = Counter()
        for termino, postings in self.indice.indice.items():
            conteos[termino] = sum(int(item["tf"]) for item in postings)
        return conteos

    def buscar(self, consulta: str, top_k: int = 10) -> list[dict]:
        terminos_consulta = self.preprocesador.tokenizar(consulta)
        if not terminos_consulta:
            return []

        terminos_existentes = [
            termino for termino in terminos_consulta if termino in self.indice.indice
        ]
        if not terminos_existentes:
            return []

        puntajes: dict[str, float] = {}
        candidatos = self._candidatos_desde_indice(terminos_existentes)
        if not candidatos:
            return []

        for doc_id in candidatos:
            puntaje_doc = 0.0
            longitud_doc = max(int(self.indice.longitudes_documentos[doc_id]), 1)

            for termino in terminos_existentes:
                tf_doc = self._frecuencia_termino_en_documento(termino, doc_id)
                tf_coleccion = self.conteos_coleccion.get(termino, 0)

                prob_doc = tf_doc / longitud_doc
                prob_coleccion = tf_coleccion / max(self.indice.total_terminos, 1)
                prob_suavizada = (
                    (1 - self.lambda_parametro) * prob_doc
                    + self.lambda_parametro * prob_coleccion
                )

                if prob_suavizada <= 0.0:
                    continue
                puntaje_doc += math.log(prob_suavizada)

            puntajes[doc_id] = puntaje_doc

        ranking = sorted(
            puntajes.items(),
            key=lambda item: (item[1], item[0]),
            reverse=True,
        )[:top_k]
        return [
            {
                "doc_id": doc_id,
                "puntaje": puntaje,
                "documento": self.indice.almacen_documentos.get(doc_id, {}),
            }
            for doc_id, puntaje in ranking
        ]

    def _candidatos_desde_indice(self, terminos_consulta: list[str]) -> set[str]:
        candidatos: set[str] = set()
        for termino in terminos_consulta:
            for posting in self.indice.indice.get(termino, []):
                candidatos.add(posting["doc_id"])
        return candidatos

    def _frecuencia_termino_en_documento(self, termino: str, doc_id: str) -> int:
        postings = self.indice.indice.get(termino, [])
        for posting in postings:
            if posting["doc_id"] == doc_id:
                return int(posting["tf"])
        return 0
