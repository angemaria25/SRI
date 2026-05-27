from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from functools import lru_cache
from typing import Any


MODELO_HUGGINGFACE_POR_DEFECTO = "google/flan-t5-small"
MAX_NUEVOS_TOKENS_POR_DEFECTO = 160


@dataclass
class RespuestaRAG:
    respuesta: str
    citas: list[str]
    fragmentos: list[str]
    modelo: str = ""


def _recortar_palabras(texto: str, max_palabras: int) -> str:
    tokens = texto.split()
    if len(tokens) <= max_palabras:
        return texto
    return " ".join(tokens[:max_palabras]).strip() + "..."


def _extraer_fragmentos_y_citas(
    resultados: list[dict],
    max_documentos: int,
    max_palabras_fragmento: int,
) -> tuple[list[str], list[str]]:
    fragmentos: list[str] = []
    citas: list[str] = []

    for resultado in resultados[:max_documentos]:
        documento = resultado.get("documento", {})
        doc_id = documento.get("id_documento") or resultado.get("doc_id", "")
        titulo = (documento.get("titulo") or "").strip()
        resumen = (documento.get("resumen") or "").strip()

        if not resumen and documento.get("texto"):
            resumen = str(documento.get("texto", "")).strip()

        fragmento = ". ".join(part for part in [titulo, resumen] if part).strip()
        fragmento = _recortar_palabras(fragmento, max_palabras_fragmento)

        if fragmento:
            fragmentos.append(fragmento)
            if doc_id:
                citas.append(str(doc_id))

    return fragmentos, citas


@lru_cache(maxsize=2)
def _cargar_generador_hf(modelo_id: str) -> Any:
    try:
        transformers = import_module("transformers")
    except Exception:
        return None

    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(modelo_id)
        model = transformers.AutoModelForSeq2SeqLM.from_pretrained(modelo_id)
        return transformers.pipeline(
            task="text2text-generation",
            model=model,
            tokenizer=tokenizer,
        )
    except Exception:
        return None


def _generar_respuesta_con_hf(
    consulta: str,
    fragmentos: list[str],
    modelo_id: str,
    max_nuevos_tokens: int,
) -> str | None:
    generador = _cargar_generador_hf(modelo_id)
    if generador is None:
        return None

    contexto = "\n".join(f"- {fragmento}" for fragmento in fragmentos)
    prompt = (
        "Responde en español de forma concisa y académica. "
        "Usa solo la información del contexto. "
        "Si la evidencia es insuficiente, indícalo claramente.\n\n"
        f"Consulta: {consulta}\n"
        f"Contexto:\n{contexto}\n\n"
        "Respuesta:"
    )

    try:
        salida = generador(
            prompt,
            max_new_tokens=max_nuevos_tokens,
            do_sample=False,
            truncation=True,
        )
    except Exception:
        return None

    if not salida:
        return None

    texto = salida[0].get("generated_text") if isinstance(salida, list) else None
    if not texto:
        return None

    return str(texto).strip()


def generar_respuesta_rag(
    consulta: str,
    resultados: list[dict],
    max_documentos: int = 3,
    max_palabras_fragmento: int = 120,
    modelo_hf: str = MODELO_HUGGINGFACE_POR_DEFECTO,
    max_nuevos_tokens: int = MAX_NUEVOS_TOKENS_POR_DEFECTO,
) -> RespuestaRAG:
    if not resultados:
        return RespuestaRAG(
            respuesta=(
                "No se encontraron documentos relevantes para responder la consulta."
            ),
            citas=[],
            fragmentos=[],
            modelo=modelo_hf,
        )

    fragmentos, citas = _extraer_fragmentos_y_citas(
        resultados=resultados,
        max_documentos=max_documentos,
        max_palabras_fragmento=max_palabras_fragmento,
    )

    if not fragmentos:
        return RespuestaRAG(
            respuesta=(
                "No se pudo generar una respuesta con los documentos recuperados."
            ),
            citas=citas,
            fragmentos=[],
            modelo=modelo_hf,
        )

    respuesta_generada = _generar_respuesta_con_hf(
        consulta=consulta,
        fragmentos=fragmentos,
        modelo_id=modelo_hf,
        max_nuevos_tokens=max_nuevos_tokens,
    )

    if respuesta_generada:
        respuesta = (
            f"Consulta: {consulta}.\n"
            f"Respuesta generada con {modelo_hf}:\n"
            f"{respuesta_generada}\n"
            f"Fuentes: {', '.join(citas) if citas else 'sin citas'}"
        )
        modelo_utilizado = modelo_hf
    else:
        respuesta = (
            f"Consulta: {consulta}.\n"
            "Respuesta basada en documentos recuperados:\n"
            + "\n".join(f"- {fragmento}" for fragmento in fragmentos)
            + f"\nFuentes: {', '.join(citas) if citas else 'sin citas'}"
        )
        modelo_utilizado = "extractivo_fallback"

    return RespuestaRAG(
        respuesta=respuesta.strip(),
        citas=citas,
        fragmentos=fragmentos,
        modelo=modelo_utilizado,
    )
