from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urldefrag, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup


@dataclass
class PoliticasCrawler:
    profundidad_maxima: int = 2
    max_paginas: int = 200
    demora_segundos: float = 1.5
    tiempo_espera: int = 20
    max_reintentos: int = 3
    respetar_robots_txt: bool = True
    user_agent: str = "PaperScan/1.0 (contacto: tu_correo@dominio.com)"


def _normalizar_url(url: str) -> str:
    url_limpia, _ = urldefrag(url)
    return url_limpia.strip()


def _es_url_valida(url: str) -> bool:
    if not url:
        return False
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _misma_fuente(url: str, dominios_permitidos: set[str]) -> bool:
    if not dominios_permitidos:
        return True
    return urlparse(url).netloc in dominios_permitidos


def _es_recurso_no_html(url: str) -> bool:
    extensiones_bloqueadas = (
        ".pdf",
        ".zip",
        ".rar",
        ".tar",
        ".gz",
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".svg",
        ".mp4",
        ".mp3",
        ".avi",
        ".json",
    )
    return url.lower().endswith(extensiones_bloqueadas)


def _extraer_texto_html(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    titulo = (soup.title.string if soup.title else "") or ""
    texto = soup.get_text(" ", strip=True)
    return titulo.strip(), " ".join(texto.split())


def _extraer_enlaces(html: str, url_base: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    enlaces: list[str] = []
    for enlace in soup.find_all("a", href=True):
        href = enlace.get("href", "").strip()
        if not href or href.startswith("mailto:") or href.startswith("javascript:"):
            continue
        url_abs = _normalizar_url(urljoin(url_base, href))
        if _es_url_valida(url_abs):
            enlaces.append(url_abs)
    return enlaces


class CrawlerWeb:
    def __init__(
        self,
        politicas: PoliticasCrawler,
        dominios_permitidos: set[str],
    ) -> None:
        self.politicas = politicas
        self.dominios_permitidos = dominios_permitidos
        self._robots: dict[str, RobotFileParser] = {}
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": politicas.user_agent})

    def _permitido_por_robots(self, url: str) -> bool:
        if not self.politicas.respetar_robots_txt:
            return True
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        if base not in self._robots:
            parser = RobotFileParser()
            parser.set_url(f"{base}/robots.txt")
            try:
                parser.read()
            except Exception:
                # Si no se puede leer robots.txt, no bloqueamos el proyecto academico.
                parser = RobotFileParser()
                parser.parse([])
            self._robots[base] = parser
        return self._robots[base].can_fetch(self.politicas.user_agent, url)

    def _descargar_html(self, url: str) -> str | None:
        for intento in range(1, self.politicas.max_reintentos + 1):
            try:
                respuesta = self._session.get(url, timeout=self.politicas.tiempo_espera)
                respuesta.raise_for_status()
                tipo = respuesta.headers.get("Content-Type", "")
                if "text/html" not in tipo:
                    return None
                return respuesta.text
            except requests.RequestException:
                if intento == self.politicas.max_reintentos:
                    return None
                time.sleep(2**intento)
        return None

    def crawl(self, semillas: Iterable[str], salida_base: Path) -> int:
        salida_base.mkdir(parents=True, exist_ok=True)
        salida_html = salida_base / "html"
        salida_texto = salida_base / "texto"
        salida_html.mkdir(parents=True, exist_ok=True)
        salida_texto.mkdir(parents=True, exist_ok=True)
        salida_jsonl = salida_base / "registros.jsonl"

        cola: list[tuple[str, int]] = []
        visitados: set[str] = set()

        for url in semillas:
            url_norm = _normalizar_url(url)
            if _es_url_valida(url_norm):
                cola.append((url_norm, 0))

        total_guardados = 0
        print(
            "Iniciando crawler web con "
            f"{len(cola)} semillas, profundidad {self.politicas.profundidad_maxima}, "
            f"maximo {self.politicas.max_paginas} paginas"
        )
        with salida_jsonl.open("a", encoding="utf-8") as archivo:
            while cola and total_guardados < self.politicas.max_paginas:
                url_actual, profundidad = cola.pop(0)
                if url_actual in visitados:
                    continue
                if profundidad > self.politicas.profundidad_maxima:
                    continue
                if not _misma_fuente(url_actual, self.dominios_permitidos):
                    continue
                if _es_recurso_no_html(url_actual):
                    continue
                if not self._permitido_por_robots(url_actual):
                    continue

                visitados.add(url_actual)
                html = self._descargar_html(url_actual)
                if not html:
                    continue

                titulo, texto = _extraer_texto_html(html)
                timestamp = datetime.now(timezone.utc).isoformat()

                nombre_archivo = f"pagina_{total_guardados:04d}.html"
                ruta_html = salida_html / nombre_archivo
                ruta_texto = salida_texto / nombre_archivo.replace(".html", ".txt")

                ruta_html.write_text(html, encoding="utf-8", errors="ignore")
                ruta_texto.write_text(texto, encoding="utf-8", errors="ignore")

                registro = {
                    "url": url_actual,
                    "titulo": titulo,
                    "profundidad": profundidad,
                    "archivo_html": str(ruta_html),
                    "archivo_texto": str(ruta_texto),
                    "fecha_descarga": timestamp,
                }
                archivo.write(json.dumps(registro, ensure_ascii=False) + "\n")
                archivo.flush()
                total_guardados += 1
                print(f"[{total_guardados}] Guardada: {url_actual}")

                enlaces = _extraer_enlaces(html, url_actual)
                for enlace in enlaces:
                    if enlace not in visitados:
                        cola.append((enlace, profundidad + 1))

                time.sleep(self.politicas.demora_segundos)

        return total_guardados


def _parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawler web basico para Corte 1")
    parser.add_argument(
        "--seeds",
        type=str,
        default=(
            "https://arxiv.org/list/cs.IR/recent,"
            "https://arxiv.org/list/cs.CL/recent"
        ),
        help="URLs semilla separadas por coma",
    )
    parser.add_argument("--depth", type=int, default=2, help="Profundidad maxima")
    parser.add_argument("--max-pages", type=int, default=200, help="Maximo de paginas")
    parser.add_argument(
        "--output",
        type=str,
        default="datos/brutos/crawl_web",
        help="Directorio de salida",
    )
    return parser.parse_args()


def main() -> None:
    args = _parsear_argumentos()
    semillas = [url.strip() for url in args.seeds.split(",") if url.strip()]
    dominios = {urlparse(url).netloc for url in semillas if _es_url_valida(url)}

    politicas = PoliticasCrawler(
        profundidad_maxima=max(args.depth, 0),
        max_paginas=max(args.max_pages, 1),
    )
    crawler = CrawlerWeb(politicas=politicas, dominios_permitidos=dominios)

    salida = Path(args.output)
    total = crawler.crawl(semillas=semillas, salida_base=salida)
    print(f"Paginas guardadas: {total}")
    print(f"Salida en: {salida.resolve()}")


if __name__ == "__main__":
    main()
