from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from modulos.indexacion.preprocesamiento import PreprocesadorTexto


class IndiceInvertido:
    def __init__(self) -> None:
        self.indice: dict[str, list[dict]] = defaultdict(list)
        self.longitudes_documentos: dict[str, int] = {}
        self.almacen_documentos: dict[str, dict] = {}
        self.total_terminos: int = 0

    def construir(
        self,
        documentos: list[dict],
        campos_texto: tuple[str, ...] = ("titulo", "resumen"),
    ) -> None:
        preprocesador = PreprocesadorTexto()

        for documento in documentos:
            doc_id = documento.get("id_documento", "")
            texto_completo = " ".join(documento.get(campo, "") for campo in campos_texto)
            terminos = preprocesador.tokenizar(texto_completo)

            if not doc_id or not terminos:
                continue

            self.almacen_documentos[doc_id] = documento
            self.longitudes_documentos[doc_id] = len(terminos)
            self.total_terminos += len(terminos)

            tf: dict[str, int] = {}
            for termino in terminos:
                tf[termino] = tf.get(termino, 0) + 1

            for termino, frecuencia in tf.items():
                self.indice[termino].append({"doc_id": doc_id, "tf": frecuencia})

    def guardar(self, directorio_salida: Path) -> None:
        directorio_salida.mkdir(parents=True, exist_ok=True)

        (directorio_salida / "indice_invertido.json").write_text(
            json.dumps(self.indice, ensure_ascii=False), encoding="utf-8"
        )
        (directorio_salida / "longitudes_documentos.json").write_text(
            json.dumps(self.longitudes_documentos, ensure_ascii=False), encoding="utf-8"
        )
        (directorio_salida / "almacen_documentos.json").write_text(
            json.dumps(self.almacen_documentos, ensure_ascii=False), encoding="utf-8"
        )
        (directorio_salida / "estadisticas_coleccion.json").write_text(
            json.dumps({"total_terminos": self.total_terminos}, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def cargar(cls, directorio_salida: Path) -> "IndiceInvertido":
        instancia = cls()
        instancia.indice = json.loads((directorio_salida / "indice_invertido.json").read_text(encoding="utf-8"))
        instancia.longitudes_documentos = json.loads(
            (directorio_salida / "longitudes_documentos.json").read_text(encoding="utf-8")
        )
        instancia.almacen_documentos = json.loads(
            (directorio_salida / "almacen_documentos.json").read_text(encoding="utf-8")
        )
        estadisticas = json.loads((directorio_salida / "estadisticas_coleccion.json").read_text(encoding="utf-8"))
        instancia.total_terminos = int(estadisticas.get("total_terminos", 0))
        return instancia

    @property
    def cantidad_documentos(self) -> int:
        return len(self.longitudes_documentos)
