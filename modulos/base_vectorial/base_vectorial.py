from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class BaseVectorialInicial:
    """Base vectorial inicial basada en TF-IDF para Primer Entrega."""

    def __init__(self) -> None:
        self.vectorizador = TfidfVectorizer(max_features=5000)
        self.ids_documentos: list[str] = []
        self.matriz: np.ndarray | None = None

    def construir(
        self,
        documentos: list[dict],
        campos_texto: tuple[str, ...] = ("titulo", "resumen"),
    ) -> None:
        corpus: list[str] = []
        ids_documentos: list[str] = []

        for documento in documentos:
            doc_id = documento.get("id_documento", "")
            if not doc_id:
                continue

            texto = " ".join(documento.get(campo, "") for campo in campos_texto).strip()
            if not texto:
                continue

            ids_documentos.append(doc_id)
            corpus.append(texto)

        if not corpus:
            self.ids_documentos = []
            self.matriz = np.zeros((0, 1), dtype=np.float32)
            return

        matriz_dispersa = self.vectorizador.fit_transform(corpus)
        self.ids_documentos = ids_documentos
        self.matriz = matriz_dispersa.astype(np.float32).toarray()

    def guardar(self, directorio_salida: Path) -> None:
        if self.matriz is None:
            raise ValueError("La base vectorial esta vacia. Debe construirla antes de guardar.")

        directorio_salida.mkdir(parents=True, exist_ok=True)
        np.save(directorio_salida / "vectores.npy", self.matriz)

        metadatos = {
            "ids_documentos": self.ids_documentos,
            "vocabulario": {
                termino: int(indice)
                for termino, indice in self.vectorizador.vocabulary_.items()
            },
            "idf": self.vectorizador.idf_.tolist(),
        }
        (directorio_salida / "metadatos.json").write_text(
            json.dumps(metadatos, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def cargar(cls, directorio_entrada: Path) -> "BaseVectorialInicial":
        instancia = cls()
        metadatos = json.loads((directorio_entrada / "metadatos.json").read_text(encoding="utf-8"))

        instancia.ids_documentos = metadatos["ids_documentos"]
        instancia.matriz = np.load(directorio_entrada / "vectores.npy")
        instancia.vectorizador.vocabulary_ = {
            termino: int(indice) for termino, indice in metadatos["vocabulario"].items()
        }
        instancia.vectorizador.idf_ = np.array(metadatos["idf"], dtype=np.float64)
        instancia.vectorizador._tfidf._idf_diag = None
        return instancia

    def buscar(self, consulta: str, top_k: int = 10) -> list[tuple[str, float]]:
        if self.matriz is None or not self.ids_documentos:
            return []

        vector_consulta = self.vectorizador.transform([consulta]).astype(np.float32).toarray()[0]
        norma_consulta = float(np.linalg.norm(vector_consulta))
        if norma_consulta == 0.0:
            return []

        normas = np.linalg.norm(self.matriz, axis=1) * np.linalg.norm(vector_consulta)
        normas[normas == 0] = 1e-9

        similitudes = (self.matriz @ vector_consulta) / normas
        if np.all(similitudes <= 0.0):
            return []

        indices_top = np.argsort(-similitudes)[:top_k]
        return [
            (self.ids_documentos[indice], float(similitudes[indice]))
            for indice in indices_top
        ]
