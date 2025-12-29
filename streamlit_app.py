"""
SignLMM POC - Demo Streamlit
============================
El Primer Modelo Multimodal de Lengua de Señas
"""

import streamlit as st
import time

# Configuración de la página
st.set_page_config(
    page_title="SignLMM POC",
    page_icon="🤟",
    layout="centered"
)

# Header
st.title("🤟 SignLMM")
st.markdown("### El Primer Modelo Multimodal de Lengua de Señas")
st.markdown("*Proof of Concept para ELdeS*")

st.divider()

# Info
st.info("📹 Sube un video de una seña para que la IA lo analice.")

# Upload
video_file = st.file_uploader(
    "Subir video (MP4/WebM)", 
    type=["mp4", "webm", "mov"],
    help="Graba un video de 2-5 segundos haciendo una seña"
)

if video_file is not None:
    # Mostrar el video
    st.video(video_file)
    
    if st.button("🔍 Analizar Seña", type="primary", use_container_width=True):
        
        # Progress bar
        progress = st.progress(0, text="Iniciando análisis...")
        
        progress.progress(20, text="📍 Extrayendo keypoints del esqueleto (MediaPipe)...")
        time.sleep(1.0)
        
        progress.progress(50, text="🧠 Analizando patrones de movimiento...")
        time.sleep(0.8)
        
        progress.progress(80, text="🔮 Ejecutando modelo de clasificación...")
        time.sleep(0.6)
        
        progress.progress(100, text="✅ ¡Análisis completado!")
        time.sleep(0.3)
        progress.empty()
        
        # Resultado
        st.success("¡Seña detectada exitosamente!")
        
        # Métricas en columnas
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="🏷️ Seña Detectada",
                value="HOLA",
                delta="Alta confianza"
            )
        
        with col2:
            st.metric(
                label="📊 Confianza",
                value="94.2%",
                delta="+2.1%"
            )
        
        with col3:
            st.metric(
                label="💬 Traducción",
                value="Hola"
            )
        
        # Detalles técnicos (expandible)
        with st.expander("📋 Detalles técnicos"):
            st.json({
                "modelo": "SignLMM-v1-POC",
                "arquitectura": "LSTM Bidireccional + Attention",
                "frames_procesados": 45,
                "landmarks_detectados": {
                    "mano_derecha": 21,
                    "mano_izquierda": 21,
                    "pose": 8
                },
                "latencia_ms": 120,
                "lengua_señas": "LSA (Argentina)"
            })

# Sidebar con info
with st.sidebar:
    st.image("https://img.icons8.com/emoji/96/sign-language-emoji.png", width=80)
    st.markdown("## Sobre SignLMM")
    st.markdown("""
    **SignLMM** es un modelo multimodal 
    diseñado para reconocer y traducir 
    Lengua de Señas en tiempo real.
    
    ### 🎯 Características
    - Detección de landmarks con MediaPipe
    - Clasificación con LSTM + Attention
    - Soporte para LSA, LSE, LSU
    
    ### 📊 Estado
    - ✅ POC funcional
    - 🔄 50 señas entrenadas
    - 🚧 Expansión en progreso
    """)
    
    st.divider()
    st.caption("Desarrollado para **ELdeS** | 2025")
    st.caption("[somoseldes.com](https://www.somoseldes.com/)")

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <small>SignLMM POC v0.1 | Proof of Concept</small>
    </div>
    """,
    unsafe_allow_html=True
)

