"""Modulo RAG con generacion asistida por HuggingFace y fallback extractivo."""

from .rag_extractor import RespuestaRAG, generar_respuesta_rag

__all__ = ["RespuestaRAG", "generar_respuesta_rag"]
