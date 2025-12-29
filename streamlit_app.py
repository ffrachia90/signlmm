import streamlit as st
import cv2
import tempfile
import mediapipe as mp
import numpy as np
import os
import json

# Configuración de página
st.set_page_config(page_title="SignLMM POC - Interactive", page_icon="🤟", layout="wide")

st.title("🤟 SignLMM - Plataforma de Entrenamiento")
st.markdown("### Ciclo completo: Recolección de Datos -> Entrenamiento -> Inferencia")

# Inicializar MediaPipe
mp_holistic = mp.solutions.holistic

# --- GESTIÓN DE DATASET (Base de Conocimiento) ---
DATA_FILE = "dataset_references.json"

def load_knowledge_base():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return {}

def save_knowledge_base(kb):
    with open(DATA_FILE, "w") as f:
        json.dump(kb, f)

# Cargar al inicio
if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = load_knowledge_base()

# --- MOTOR DE EXTRACCIÓN DE CARACTERÍSTICAS (El "LMM" simplificado) ---
def extract_features(video_path):
    """
    Convierte un video en un 'Vector de Características' (Huella digital numérica).
    """
    cap = cv2.VideoCapture(video_path)
    
    x_movements = []
    y_movements = []
    mouth_distances = []
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Optimización: Reducir resolución para velocidad
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            
            # Detectar mano (Derecha o Izquierda)
            hand_landmarks = results.right_hand_landmarks or results.left_hand_landmarks
            
            # Detectar boca
            mouth_center = None
            if results.pose_landmarks:
                mouth_l = results.pose_landmarks.landmark[9]
                mouth_r = results.pose_landmarks.landmark[10]
                mouth_center = np.array([(mouth_l.x + mouth_r.x)/2, (mouth_l.y + mouth_r.y)/2])
            
            if hand_landmarks:
                wrist = hand_landmarks.landmark[0]
                wrist_pos = np.array([wrist.x, wrist.y])
                
                x_movements.append(wrist.x)
                y_movements.append(wrist.y)
                
                if mouth_center is not None:
                    dist = np.linalg.norm(wrist_pos - mouth_center)
                    mouth_distances.append(float(dist))
                    
        cap.release()

    # Si no detectó nada, retornar vector nulo
    if not x_movements:
        return None

    # --- CREACIÓN DEL VECTOR DE CARACTERÍSTICAS (Embedding) ---
    # 1. Agitación Horizontal (Para HOLA)
    var_x = np.var(x_movements) * 1000
    # 2. Agitación Vertical (Para GRACIAS/SI)
    var_y = np.var(y_movements) * 1000
    # 3. Proximidad mínima a la boca (Para AGUA/COMER)
    min_mouth = min(mouth_distances) if mouth_distances else 1.0
    # 4. Altura promedio de la mano (Y invertido en CV2: 0 arriba, 1 abajo)
    avg_y = np.mean(y_movements)

    # Vector numérico: [AgitaciónX, AgitaciónY, ProximidadBoca, AlturaMano]
    return [var_x, var_y, min_mouth, avg_y]

# --- LÓGICA DE COMPARACIÓN (Nearest Neighbor) ---
def predict_sign(input_vector, kb):
    """Busca el vector más cercano en la base de conocimiento."""
    best_label = "DESCONOCIDO"
    best_dist = float("inf")
    
    for label, vectors in kb.items():
        # Usamos el promedio de los ejemplos de esta etiqueta
        ref_vector = np.mean(vectors, axis=0)
        
        # Distancia Euclidiana (Diferencia matemática entre videos)
        # Ponderamos más la cercanía a la boca (índice 2) para que AGUA no falle
        weights = np.array([1.0, 1.0, 50.0, 5.0]) 
        dist = np.linalg.norm((np.array(input_vector) - np.array(ref_vector)) * weights)
        
        if dist < best_dist:
            best_dist = dist
            best_label = label
            
    # Convertir distancia a confianza (aprox)
    confidence = max(0, 100 - best_dist) / 100.0
    return best_label, confidence, best_dist

# --- INTERFAZ DE USUARIO ---

tab1, tab2 = st.tabs(["🧪 Probar (Inferencia)", "📚 Entrenar (Recolectar Data)"])

with tab2:
    st.header("Entrenar el Modelo")
    st.info("Sube tus videos aquí para 'enseñarle' a la IA cómo es cada seña.")
    
    train_video = st.file_uploader("Video de Entrenamiento", type=["mp4", "webm", "mov"], key="train_upl")
    label_input = st.text_input("¿Qué seña es esta?", placeholder="Ej: AGUA").upper()
    
    if st.button("💾 Guardar en Dataset") and train_video and label_input:
        with st.spinner("Extrayendo características..."):
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(train_video.read())
            
            vector = extract_features(tfile.name)
            
            if vector:
                # Actualizar KB
                if label_input not in st.session_state.knowledge_base:
                    st.session_state.knowledge_base[label_input] = []
                st.session_state.knowledge_base[label_input].append(vector)
                
                save_knowledge_base(st.session_state.knowledge_base)
                
                st.success(f"✅ ¡Aprendí la seña '{label_input}'!")
                st.json({"Vector Característico": vector})
            else:
                st.error("❌ No se detectaron manos en el video. Intenta con otro.")
    
    # Mostrar KB actual
    if st.session_state.knowledge_base:
        st.write("---")
        st.write(f"📚 **Señas aprendidas:** {list(st.session_state.knowledge_base.keys())}")

with tab1:
    st.header("Prueba en Tiempo Real")
    test_video = st.file_uploader("Sube un video para probar", type=["mp4", "webm", "mov"], key="test_upl")
    
    if test_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(test_video.read())
        
        col1, col2 = st.columns(2)
        with col1:
            st.video(tfile.name)
        
        with col2:
            if st.button("🧠 Predecir", type="primary"):
                if not st.session_state.knowledge_base:
                    st.warning("⚠️ El modelo está vacío. Ve a la pestaña 'Entrenar' y sube ejemplos primero.")
                else:
                    with st.spinner("Analizando..."):
                        vector = extract_features(tfile.name)
                        
                        if vector:
                            label, conf, dist = predict_sign(vector, st.session_state.knowledge_base)
                            
                            st.metric("Resultado", label)
                            st.progress(min(conf, 1.0), text=f"Confianza: {conf:.1%}")
                            
                            with st.expander("Ver matemáticas"):
                                st.write(f"Vector Input: {vector}")
                                st.write(f"Distancia (Error): {dist:.4f}")
                        else:
                            st.error("No se detectaron manos.")

st.sidebar.markdown("---")
st.sidebar.caption("SignLMM Zero-Shot Engine | ELdeS")
if st.sidebar.button("🗑️ Borrar Memoria"):
    st.session_state.knowledge_base = {}
    if os.path.exists(DATA_FILE): os.remove(DATA_FILE)
    st.experimental_rerun()
