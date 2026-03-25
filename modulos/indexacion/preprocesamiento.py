from __future__ import annotations

import re
from collections import Counter

PATRON_TOKEN = re.compile(r"[a-zA-Z0-9]+")


class PreprocesadorTexto:
    def __init__(self, longitud_minima: int = 2) -> None:
        self.longitud_minima = longitud_minima

    def tokenizar(self, texto: str) -> list[str]:
        tokens = PATRON_TOKEN.findall(texto.lower())
        return [token for token in tokens if len(token) >= self.longitud_minima]

    def frecuencias(self, texto: str) -> Counter[str]:
        return Counter(self.tokenizar(texto))
