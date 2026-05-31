from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from functools import lru_cache
from typing import Any

import torch


MODELO_HUGGINGFACE_POR_DEFECTO = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NUEVOS_TOKENS_POR_DEFECTO = 256


@dataclass
class RespuestaRAG:
    respuesta: str
    citas: list[str]
    fragmentos: list[str]
    modelo: str = ""
    confianza: float = 0.0


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
def _cargar_generador_hf(modelo_id: str) -> tuple[Any, Any, Any] | None:
    try:
        transformers = import_module("transformers")
    except Exception:
        return None

    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(modelo_id)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            modelo_id,
            torch_dtype=torch.float16,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return model, tokenizer, transformers
    except Exception:
        return None


def _generar_respuesta_con_hf(
    consulta: str,
    fragmentos: list[str],
    modelo_id: str,
    max_nuevos_tokens: int,
) -> tuple[str | None, float]:
    loaded = _cargar_generador_hf(modelo_id)
    if loaded is None:
        return None, 0.0

    model, tokenizer, _ = loaded

    contexto = "\n".join(f"- {fragmento}" for fragmento in fragmentos)

    messages = [
        {
            "role": "system",
            "content": (
                "Eres PaperScan, un asistente de investigación académica experto. "
                "Tu dominio es estrictamente CIENTÍFICO y ACADÉMICO. "
                "\nREGLAS DE COMPORTAMIENTO:\n"
                "1. Si la consulta NO es de naturaleza científica o académica, indica que tu dominio se limita a la investigación.\n"
                "2. Responde en ESPAÑOL usando un tono técnico\n"
                "3. Si los fragmentos mencionan estudios, propuestas o metodologías relacionadas con la consulta, descríbelos de forma informativa.\n"
                "4. Si los fragmentos no guardan relación con el tema de la consulta, indica claramente que no hay evidencia suficiente en la base documental actual."
            ),
        },
        {
            "role": "user",
            "content": f"Consulta: {consulta}\n\nContexto para responder:\n{contexto}",
        },
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)

    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_nuevos_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )
    except Exception:
        return None, 0.0

    generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1] :]
    if generated_ids.shape[0] == 0:
        return "", 0.0

    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    avg_log_prob = 0.0
    if outputs.scores and generated_ids.shape[0] > 0:
        log_probs = []
        for i, score in enumerate(outputs.scores):
            if i < len(generated_ids):
                token_id = generated_ids[i].item()
                log_probs.append(
                    torch.log_softmax(score[0], dim=-1)[token_id].item()
                )
        if log_probs:
            avg_log_prob = sum(log_probs) / len(log_probs)

    return response, avg_log_prob


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

    respuesta_generada, confianza = _generar_respuesta_con_hf(
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
        confianza = 0.0

    return RespuestaRAG(
        respuesta=respuesta.strip(),
        citas=citas,
        fragmentos=fragmentos,
        modelo=modelo_utilizado,
        confianza=confianza,
    )
