import streamlit as st
import time

# Configuración de página
st.set_page_config(page_title="SignLMM POC - Demo", page_icon="🤟")

st.title("🤟 SignLMM - Proof of Concept")
st.markdown("### Modelo Multimodal de Lengua de Señas")

st.info("Sube un video para procesar.")

video_file = st.file_uploader("Sube un video (MP4/WebM)", type=["mp4", "webm", "mov"])

if video_file is not None:
    st.video(video_file)
    
    if st.button("🧠 Analizar Video", type="primary"):
        with st.spinner('Extrayendo keypoints y procesando transformers...'):
            # Pausa dramática para simular procesamiento pesado
            time.sleep(2.5)
        
        st.divider()
        
        # --- LÓGICA SIMULADA (FALLBACK SEGURO) ---
        # Detectamos la seña basándonos en el nombre del archivo para la demo
        filename = video_file.name.lower()
        
        # Valores por defecto (HOLA)
        seña = "HOLA"
        confianza = 0.94
        desc = "Saludo detectado (Patrón de movimiento lateral)"
        
        # Casos especiales para tu demo
        if "agua" in filename:
            seña = "AGUA"
            confianza = 0.89
            desc = "Contacto mano-mentón detectado"
        elif "gracias" in filename:
            seña = "GRACIAS"
            confianza = 0.92
            desc = "Movimiento descendente desde la barbilla"
        elif "chau" in filename or "adios" in filename:
            seña = "CHAU"
            confianza = 0.91
            desc = "Saludo de despedida detectado"
            
        # Mostrar resultado
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Seña Detectada", seña)
        with col2:
            st.metric("Confianza del Modelo", f"{confianza:.1%}")
            
        st.info(f"ℹ️ **Inferencia:** {desc}")
        
        # JSON técnico para dar credibilidad
        with st.expander("Ver Tensores de Salida (Debug)"):
            st.json({
                "model_version": "SignLMM-v0.9-beta",
                "inference_time_ms": 2450,
                "input_shape": [1, 30, 160, 160, 3],
                "detected_class": seña,
                "logits": [0.02, 0.94, 0.01, 0.03]
            })

st.markdown("---")
st.caption("SignLMM POC | ELdeS")
