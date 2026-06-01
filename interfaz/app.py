import streamlit as st
import sys
import os
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parent.parent))

from modulos.indexacion.indice_invertido import IndiceInvertido
from modulos.recuperacion.modelo_lenguaje import RecuperadorModeloLenguaje
from modulos.base_vectorial.base_vectorial import BaseVectorialMejorada
from modulos.busqueda_web.buscador_arxiv import buscar_en_arxiv
from modulos.rag import generar_respuesta_rag

# Configuración de la página
st.set_page_config(page_title="PaperScan - SRI Académico", page_icon="🎓", layout="wide")

# --- CARGA DE DATOS  ---
@st.cache_resource
def cargar_recursos():
    ruta_datos = Path("datos")
    indice = IndiceInvertido.cargar(ruta_datos / "indices")
    base_vec = BaseVectorialMejorada.cargar(ruta_datos / "base_vectorial_mejorada")
    return indice, base_vec

# --- LÓGICA DE RECOMENDACIÓN (Módulo Opcional) ---
def recomendar_similares(doc_id, base_vec, top_k=5):
    # Encontrar el índice del documento en la base vectorial
    if doc_id in base_vec.ids_documentos:
        idx = base_vec.ids_documentos.index(doc_id)
        vector = base_vec.matriz[idx].toarray() 
        # Usamos la base vectorial para buscar los más cercanos al vector del paper
        resultados = base_vec.buscar(doc_id, top_k=top_k + 1)
        return [res for res in resultados if res[0] != doc_id]
    return []

# --- INTERFAZ ---
st.title("🎓 PaperScan: Buscador Académico Inteligente")
st.markdown("---")

with st.sidebar:
    st.header("Configuración del SRI")
    usar_rag = st.toggle("Activar IA (RAG)", value=True)
    usar_prf = st.toggle("Expansión de consulta (PRF)", value=True)
    top_k = st.slider("Resultados a mostrar", 5, 20, 10)
    st.info("Este sistema utiliza un Modelo de Lenguaje con Suavizado Dirichlet y Posicionamiento por Frescura.")

# Input de búsqueda
query = st.text_input("Introduce tu consulta científica o académica:", placeholder="Ej: Avances en grafeno para baterías")

if query:
    try:
        with st.spinner("Buscando en la base documental local..."):
            indice, base_vec = cargar_recursos()
            
            # 1. Recuperación Local
            recuperador = RecuperadorModeloLenguaje(indice=indice, usar_prf=usar_prf)
            resultados = recuperador.buscar(query, top_k=top_k)

        # 2. Lógica de Respaldo Web 
        if not resultados:
            st.warning("No se encontró información suficiente localmente. Consultando arXiv en tiempo real...")
            resultados_web = buscar_en_arxiv(query, total_resultados=top_k)
            # Convertir formato web a formato compatible
            resultados = [{"doc_id": r["id_documento"], "puntaje": 0, "documento": r} for r in resultados_web]
            st.success(f"Se encontraron {len(resultados)} resultados en la web.")

        if resultados:
            # --- SECCIÓN RAG (IA) ---
            if usar_rag:
                with st.expander("✨ Resumen Inteligente (RAG)", expanded=True):
                    respuesta = generar_respuesta_rag(query, resultados)
                    st.write(respuesta.respuesta)
                    st.caption(f"Modelo utilizado: {respuesta.modelo} | Confianza: {abs(respuesta.confianza):.2f}")

            # --- LISTADO DE RESULTADOS ---
            st.subheader(f"Resultados de búsqueda ({len(resultados)})")
            
            for res in resultados:
                doc = res["documento"]
                with st.container(border=True):
                    col1, col2 = st.columns([0.85, 0.15])
                    
                    with col1:
                        st.markdown(f"### {doc.get('titulo', 'Sin título')}")
                        fecha = doc.get('publicado', 'Fecha desconocida')[:10]
                        st.caption(f"📅 Publicado: {fecha} | 🆔 ID: {res['doc_id']} | 🎯 Score: {res['puntaje']:.2f}")
                        st.write(doc.get('resumen', 'Sin resumen')[:400] + "...")
                        
                        # Botón de Recomendación (Módulo Opcional)
                        if st.button(f"Papers similares a {res['doc_id']}", key=f"rec_{res['doc_id']}"):
                            similares = recomendar_similares(res['doc_id'], base_vec)
                            if similares:
                                st.write("Artículos recomendados:")
                                for s_id, sim in similares:
                                    st.write(f"- {s_id} (Similitud: {sim:.2f})")
                            else:
                                st.write("No se encontraron recomendaciones cercanas.")

                    with col2:
                        url = doc.get('url_pdf', '')
                        if url:
                            st.link_button("📄 Ver PDF", url)
                        else:
                            st.write("PDF Local")

    except Exception as e:
        st.error(f"Error al procesar la búsqueda: {e}")
        st.info("Asegúrate de haber corrido 'main.py' al menos una vez para generar los índices.")

else:
    st.info("Escribe algo arriba para comenzar la investigación.")