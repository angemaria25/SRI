from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


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


class BaseVectorialMejorada:
    """Base vectorial mejorada con TF-IDF disperso y busqueda eficiente."""

    def __init__(self) -> None:
        self.vectorizador = TfidfVectorizer(
            max_features=12000,
            ngram_range=(1, 2),
            min_df=2,
        )
        self.ids_documentos: list[str] = []
        self.matriz: sp.csr_matrix | None = None
        self._indice_nn: NearestNeighbors | None = None

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
            self.matriz = sp.csr_matrix((0, 1), dtype=np.float32)
            self._indice_nn = None
            return

        matriz_dispersa = self.vectorizador.fit_transform(corpus).astype(np.float32)
        self.ids_documentos = ids_documentos
        self.matriz = matriz_dispersa
        self._ajustar_indice()

    def _ajustar_indice(self) -> None:
        if self.matriz is None or self.matriz.shape[0] == 0:
            self._indice_nn = None
            return
        self._indice_nn = NearestNeighbors(metric="cosine", algorithm="brute")
        self._indice_nn.fit(self.matriz)

    def guardar(self, directorio_salida: Path) -> None:
        if self.matriz is None:
            raise ValueError("La base vectorial esta vacia. Debe construirla antes de guardar.")

        directorio_salida.mkdir(parents=True, exist_ok=True)
        sp.save_npz(directorio_salida / "vectores.npz", self.matriz)

        metadatos = {
            "ids_documentos": self.ids_documentos,
            "vocabulario": {
                termino: int(indice)
                for termino, indice in self.vectorizador.vocabulary_.items()
            },
            "idf": self.vectorizador.idf_.tolist(),
            "config": {
                "max_features": 12000,
                "ngram_range": [1, 2],
                "min_df": 2,
            },
        }
        (directorio_salida / "metadatos.json").write_text(
            json.dumps(metadatos, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def cargar(cls, directorio_entrada: Path) -> "BaseVectorialMejorada":
        instancia = cls()
        metadatos = json.loads((directorio_entrada / "metadatos.json").read_text(encoding="utf-8"))

        instancia.ids_documentos = metadatos["ids_documentos"]
        instancia.matriz = sp.load_npz(directorio_entrada / "vectores.npz")
        instancia.vectorizador.vocabulary_ = {
            termino: int(indice) for termino, indice in metadatos["vocabulario"].items()
        }
        instancia.vectorizador.idf_ = np.array(metadatos["idf"], dtype=np.float64)
        instancia.vectorizador._tfidf._idf_diag = None
        instancia._ajustar_indice()
        return instancia

    def buscar(self, consulta: str, top_k: int = 10) -> list[tuple[str, float]]:
        if self.matriz is None or not self.ids_documentos or self._indice_nn is None:
            return []

        vector_consulta = self.vectorizador.transform([consulta]).astype(np.float32)
        if vector_consulta.nnz == 0:
            return []

        distancias, indices = self._indice_nn.kneighbors(
            vector_consulta, n_neighbors=min(top_k, len(self.ids_documentos))
        )
        resultados: list[tuple[str, float]] = []
        for distancia, indice in zip(distancias[0], indices[0]):
            similitud = 1.0 - float(distancia)
            resultados.append((self.ids_documentos[indice], similitud))
        return resultados
