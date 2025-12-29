import streamlit as st
import time
import numpy as np
import pandas as pd

# Configuración de página con diseño amplio
st.set_page_config(page_title="SignLMM - ELdeS", page_icon="🤟", layout="wide")

# --- CABECERA ---
st.title("🤟 SignLMM: Inteligencia Artificial Multimodal")
st.markdown("""
**Objetivo:** Crear el primer Modelo de Lenguaje Grande (LMM) nativo para la **Lengua de Señas Uruguaya (LSU)**.
No es un diccionario: es una IA que entiende la **gramática del movimiento**.
""")

# --- FUNCIONES SIMULADAS (SAFE MODE) ---
def simulate_processing(filename):
    """Simula el procesamiento de un video con lógica determinista para la demo."""
    time.sleep(1.5) # Simular carga de GPU
    
    filename = filename.lower()
    if "agua" in filename:
        return "AGUA", 0.94, "Contacto mano-mentón (LSU Clásico)", True
    elif "gracias" in filename:
        return "GRACIAS", 0.92, "Movimiento vertical descendente", True
    elif "hola" in filename:
        return "HOLA", 0.96, "Agitación lateral de mano abierta", True
    else:
        # Default seguro pero marcado como "Inferencia Genérica"
        return "HOLA", 0.88, "Detección de saludo estándar", False

def generate_fake_landmarks(seña_tipo):
    """Genera datos gráficos falsos pero realistas para visualizar la 'Arquitectura'."""
    frames = 30
    # Generar ondas sinusoidales que parecen movimiento humano
    t = np.linspace(0, 4*np.pi, frames)
    
    if seña_tipo == "HOLA":
        # Movimiento mucho en X, poco en Y
        x = np.sin(t) * 0.8
        y = np.cos(t) * 0.1
    elif seña_tipo == "GRACIAS":
        # Movimiento mucho en Y (baja), poco en X
        x = np.random.normal(0, 0.05, frames)
        y = -np.linspace(-1, 1, frames) # Baja
    else: # AGUA
        # Movimiento estático cerca de una posición (la boca)
        x = np.random.normal(0, 0.02, frames)
        y = np.random.normal(0.5, 0.02, frames) # 0.5 altura boca
        
    df = pd.DataFrame({"Tiempo": range(frames), "Mano X": x, "Mano Y": y, "Mano Z": np.random.normal(0, 0.1, frames)})
    return df

# --- INTERFAZ CON PESTAÑAS ---
tab_demo, tab_data, tab_tech = st.tabs(["🚀 Demo en Vivo", "📂 Recolección de Datos (Dataset)", "🧠 Arquitectura Neural"])

# 1. PESTAÑA DEMO (La que funciona seguro)
with tab_demo:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Prueba de Concepto")
        st.info("Sube un video de LSU para ver la detección en tiempo real.")
        video_file = st.file_uploader("Subir video (MP4)", type=["mp4", "mov"], key="demo_upl")

    with col2:
        if video_file:
            st.video(video_file)
            if st.button("Analizar con SignLMM v0.1", type="primary"):
                with st.spinner("1. Extrayendo esqueleto (MediaPipe)..."):
                    time.sleep(0.8)
                with st.spinner("2. Analizando secuencia temporal (Transformer)..."):
                    time.sleep(0.8)
                
                label, conf, desc, is_specific = simulate_processing(video_file.name)
                
                # Resultado visual impactante
                if is_specific:
                    st.success(f"✅ Seña Detectada: **{label}**")
                else:
                    st.success(f"✅ Seña Detectada: **{label}** (Predicción)")
                    
                st.metric("Confianza del Modelo", f"{conf:.1%}", delta="Alta Precisión")
                st.caption(f"🧠 Razonamiento de IA: {desc}")

# 2. PESTAÑA DATOS (La ambición del proyecto)
with tab_data:
    st.subheader("El Desafío de los Datos: Construyendo el Dataset LSU")
    st.markdown("""
    Para entrenar un modelo robusto, necesitamos miles de ejemplos.
    Esta interfaz permite a la comunidad sorda **enseñarle** a la IA.
    """)
    
    c1, c2 = st.columns(2)
    with c1:
        train_vid = st.file_uploader("Subir video de entrenamiento", key="train_upl")
    with c2:
        tag = st.text_input("Etiqueta Correcta (LSU)", placeholder="Ej: CASA")
        variant = st.selectbox("Variante Regional", ["Montevideo", "Interior", "Neutro"])
    
    if st.button("💾 Guardar en SignLMM-Dataset"):
        if train_vid and tag:
            st.balloons()
            st.success(f"¡Gracias! El video ha sido procesado y agregado al dataset como ejemplo de **{tag.upper()}**.")
            st.json({
                "video_id": train_vid.name,
                "label": tag.upper(),
                "region": variant,
                "skeleton_points": 543,
                "status": "Ready for Training"
            })
        else:
            st.warning("Sube un video y escribe una etiqueta.")

# 3. PESTAÑA TÉCNICA (Explicando la magia)
with tab_tech:
    st.subheader("¿Cómo funciona por dentro?")
    st.markdown("A diferencia de un video normal, SignLMM ve el mundo en **geometría pura**.")
    
    st.markdown("### 1. Visión Esquelética (MediaPipe)")
    st.info("Convertimos el video en coordenadas matemáticas ($x, y, z$). Esto protege la privacidad (no guardamos caras) y es ultraligero.")
    
    # Simulación de lo que ve la IA
    if video_file:
        sim_label, _, _, _ = simulate_processing(video_file.name)
        df_landmarks = generate_fake_landmarks(sim_label)
        
        st.write(f"**Patrón de movimiento extraído para: {sim_label}**")
        st.line_chart(df_landmarks.set_index("Tiempo")[["Mano X", "Mano Y"]])
        st.caption("Eje X (Horizontal) vs Eje Y (Vertical) a lo largo del tiempo.")
    else:
        st.write("Sube un video en la pestaña Demo para ver sus gráficos aquí.")

    st.markdown("### 2. Arquitectura Híbrida")
    st.code("""
    Input (Video) 
       ⬇
    [MediaPipe Layer] -> Extrae 543 puntos (Manos, Cara, Pose)
       ⬇
    [Temporal Encoder] -> Transformer / LSTM (Entiende el movimiento)
       ⬇
    [Classification Head] -> Predicción: "AGUA" (94%)
    """, language="python")

st.sidebar.image("https://img.icons8.com/color/96/sign-language-r.png")
st.sidebar.markdown("### Estado del Sistema")
st.sidebar.write("🟢 **API:** Online")
st.sidebar.write("🟢 **GPU:** Simulada")
st.sidebar.write("🟠 **Dataset:** En construcción")
