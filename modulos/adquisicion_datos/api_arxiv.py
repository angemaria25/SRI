from __future__ import annotations

import random
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from typing import Iterator
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests

URL_API_ARXIV = "http://export.arxiv.org/api/query"
ESPACIO_NOMBRES_ATOM = {"atom": "http://www.w3.org/2005/Atom"}
ENCABEZADOS_HTTP = {"User-Agent": "PaperScan/1.0 (proyecto academico SRI)"}


@dataclass
class DocumentoArxiv:
    id_documento: str
    titulo: str
    resumen: str
    autores: list[str]
    categorias: list[str]
    publicado: str
    actualizado: str
    url_pdf: str

    def a_diccionario(self) -> dict:
        return asdict(self)


@dataclass
class PoliticasCrawling:
    demora_segundos: float = 1.0
    tiempo_espera: int = 30
    max_reintentos: int = 4
    tamano_lote_maximo: int = 100
    validar_robots_txt: bool = True


class RecolectorArxiv:
    """Cliente sencillo de arXiv con espera entre solicitudes."""

    def __init__(self, politicas: PoliticasCrawling | None = None) -> None:
        self.politicas = politicas or PoliticasCrawling()
        self._robots_permitido_cache: bool | None = None

    def obtener(
        self,
        consulta: str,
        total_resultados: int = 500,
        tamano_lote: int = 100,
        ordenar_por: str = "submittedDate",
        orden: str = "descending",
    ) -> Iterator[DocumentoArxiv]:
        if self.politicas.validar_robots_txt and not self._api_permitida_por_robots():
            raise RuntimeError("La politica de robots.txt no permite acceder al endpoint de arXiv")

        recuperados = 0
        lote_efectivo = min(tamano_lote, self.politicas.tamano_lote_maximo)

        while recuperados < total_resultados:
            lote_actual = min(lote_efectivo, total_resultados - recuperados)
            parametros = {
                "search_query": consulta,
                "start": recuperados,
                "max_results": lote_actual,
                "sortBy": ordenar_por,
                "sortOrder": orden,
            }

            raiz = self._solicitar_feed_xml(parametros)
            entradas = raiz.findall("atom:entry", ESPACIO_NOMBRES_ATOM)

            if not entradas:
                break

            for entrada in entradas:
                yield self._entrada_a_documento(entrada)

            recuperados += len(entradas)
            time.sleep(self.politicas.demora_segundos)

    def _api_permitida_por_robots(self) -> bool:
        if self._robots_permitido_cache is not None:
            return self._robots_permitido_cache

        try:
            endpoint = urlparse(URL_API_ARXIV)
            robots = RobotFileParser()
            robots.set_url(f"{endpoint.scheme}://{endpoint.netloc}/robots.txt")
            robots.read()
            self._robots_permitido_cache = robots.can_fetch(
                ENCABEZADOS_HTTP["User-Agent"], endpoint.path or "/"
            )
        except Exception:
            # Si robots no se puede consultar, no bloqueamos el proyecto academico.
            self._robots_permitido_cache = True

        return self._robots_permitido_cache

    def _solicitar_feed_xml(self, parametros: dict) -> ET.Element:
        for intento in range(1, self.politicas.max_reintentos + 1):
            try:
                respuesta = requests.get(
                    URL_API_ARXIV,
                    params=parametros,
                    headers=ENCABEZADOS_HTTP,
                    timeout=self.politicas.tiempo_espera,
                )
                respuesta.raise_for_status()
                return ET.fromstring(respuesta.text)
            except (requests.RequestException, ET.ParseError) as error:
                if intento == self.politicas.max_reintentos:
                    raise RuntimeError(f"Fallo al consultar arXiv tras {intento} intentos: {error}") from error
                espera = (2**intento) + random.random()
                time.sleep(espera)

        raise RuntimeError("No fue posible consultar el feed XML de arXiv")

    @staticmethod
    def _entrada_a_documento(entrada: ET.Element) -> DocumentoArxiv:
        id_documento = entrada.findtext(
            "atom:id", default="", namespaces=ESPACIO_NOMBRES_ATOM
        ).split("/")[-1]
        titulo = (entrada.findtext("atom:title", default="", namespaces=ESPACIO_NOMBRES_ATOM) or "").strip()
        resumen = (
            entrada.findtext("atom:summary", default="", namespaces=ESPACIO_NOMBRES_ATOM) or ""
        ).strip()
        publicado = entrada.findtext("atom:published", default="", namespaces=ESPACIO_NOMBRES_ATOM)
        actualizado = entrada.findtext("atom:updated", default="", namespaces=ESPACIO_NOMBRES_ATOM)

        autores = [
            (autor.findtext("atom:name", default="", namespaces=ESPACIO_NOMBRES_ATOM) or "").strip()
            for autor in entrada.findall("atom:author", ESPACIO_NOMBRES_ATOM)
        ]

        categorias = [
            categoria.attrib.get("term", "").strip()
            for categoria in entrada.findall("atom:category", ESPACIO_NOMBRES_ATOM)
        ]

        url_pdf = ""
        for enlace in entrada.findall("atom:link", ESPACIO_NOMBRES_ATOM):
            if enlace.attrib.get("title") == "pdf":
                url_pdf = enlace.attrib.get("href", "")
                break

        return DocumentoArxiv(
            id_documento=id_documento,
            titulo=titulo,
            resumen=resumen,
            autores=autores,
            categorias=categorias,
            publicado=publicado,
            actualizado=actualizado,
            url_pdf=url_pdf,
        )
