from __future__ import annotations

import re
from collections import Counter

PATRON_TOKEN = re.compile(r"[a-zA-Z0-9]+")


class PreprocesadorTexto:
    def __init__(self, longitud_minima: int = 2) -> None:
        self.longitud_minima = longitud_minima
        self.stop_words = {
            "the", "and", "for", "with", "from", "this", "that", "not", "are", "was",
            "were", "been", "has", "have", "had", "such", "which", "about", "into",
            "using", "used", "paper", "research", "study", "results", "analysis"
        }

    def tokenizar(self, texto: str) -> list[str]:
        tokens = PATRON_TOKEN.findall(texto.lower())
        return [
            t for t in tokens 
            if len(t) >= self.longitud_minima and t not in self.stop_words
        ]

    def frecuencias(self, texto: str) -> Counter[str]:
        return Counter(self.tokenizar(texto))
