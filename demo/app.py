"""
SignLMM POC - Demo Interactiva
==============================
Interfaz web para probar el modelo de reconocimiento de señas.

Uso:
    python demo/app.py
    
    Luego abrir en el navegador: http://localhost:7860
"""

import os
import sys
import tempfile
from pathlib import Path
import json
import pickle

# Agregar directorio padre al path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import gradio as gr

from scripts.extract_landmarks import LandmarkExtractor
from scripts.prepare_dataset import flatten_landmarks, normalize_sequence
from scripts.train_model import SignClassifier
from scripts.translate import get_translator, SimpleTranslator


class SignLMMDemo:
    """Demo interactiva del modelo SignLMM."""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        
        # Cargar modelo y componentes
        self._load_model()
        self._load_translator()
        
        # Extractor de landmarks (obligatorio)
        self.extractor = LandmarkExtractor()
        print("✅ Extractor de landmarks cargado (MediaPipe Tasks)")
    
    def _load_model(self):
        """Carga el modelo entrenado."""
        model_path = self.model_dir / 'best_model.pt'
        
        if not model_path.exists():
            print(f"⚠️  No se encontró modelo en {model_path}")
            print("   Ejecuta primero el entrenamiento o usa el modo demo.")
            self.model = None
            self.label_encoder = None
            return
        
        # Cargar checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint['config']
        
        # Crear modelo
        self.model = SignClassifier(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            num_classes=config['num_classes'],
            dropout=config['dropout']
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Cargar label encoder
        with open(self.model_dir / 'label_encoder.pkl', 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # Cargar estadísticas de normalización
        self.mean = np.load(self.model_dir / 'mean.npy')
        self.std = np.load(self.model_dir / 'std.npy')
        
        print(f"✅ Modelo cargado: {config['num_classes']} clases")
        print(f"   Señas: {list(self.label_encoder.classes_)[:10]}...")
    
    def _load_translator(self):
        """Carga el traductor."""
        # Intentar usar LLM, fallback a traductor simple (sin romper)
        self.translator = get_translator(use_llm=True, provider="openai")
        if isinstance(self.translator, SimpleTranslator):
            print("ℹ️  Traductor: modo simple (sin LLM)")
        else:
            print("✅ Traductor: LLM")
    
    def process_video(self, video_path: str) -> dict:
        """
        Procesa un video y retorna la predicción.
        
        Args:
            video_path: Ruta al archivo de video
            
        Returns:
            dict con predicción, confianza y traducción
        """
        if self.model is None:
            return {
                "error": "Modelo no cargado. Entrena el modelo primero.",
                "seña_detectada": None,
                "confianza": 0,
                "traducción": None
            }
        
        # extractor es obligatorio; si falla, que falle explícito
        
        try:
            # 1. Extraer landmarks
            landmarks_data = self.extractor.extract_from_video(video_path)
            
            # 2. Procesar landmarks
            sequence = []
            for frame_data in landmarks_data['landmarks']:
                flat_features = flatten_landmarks(frame_data)
                sequence.append(flat_features)
            
            if len(sequence) < 5:
                return {
                    "error": "Video muy corto o no se detectaron manos",
                    "seña_detectada": None,
                    "confianza": 0,
                    "traducción": None
                }
            
            # 3. Normalizar secuencia
            normalized = normalize_sequence(sequence, target_length=60)
            
            # 4. Normalizar features
            normalized = (normalized - self.mean) / self.std
            
            # 5. Predecir
            with torch.no_grad():
                input_tensor = torch.FloatTensor(normalized).unsqueeze(0).to(self.device)
                output = self.model(input_tensor)
                probabilities = torch.softmax(output, dim=1)
                
                confidence, predicted_idx = probabilities.max(1)
                predicted_sign = self.label_encoder.inverse_transform([predicted_idx.item()])[0]
                
                # Top 3 predicciones
                top3_probs, top3_indices = torch.topk(probabilities, k=min(3, len(self.label_encoder.classes_)))
                top3_signs = self.label_encoder.inverse_transform(top3_indices[0].cpu().numpy())
                top3 = [
                    {"seña": sign, "confianza": f"{prob:.1%}"}
                    for sign, prob in zip(top3_signs, top3_probs[0].cpu().numpy())
                ]
            
            # 6. Traducir
            translation = self.translator.translate([predicted_sign])
            
            return {
                "seña_detectada": predicted_sign,
                "confianza": f"{confidence.item():.1%}",
                "traducción": translation,
                "top_3": top3,
                "frames_procesados": len(sequence)
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "seña_detectada": None,
                "confianza": 0,
                "traducción": None
            }
    
    def create_interface(self) -> gr.Blocks:
        """Crea la interfaz Gradio."""
        
        # CSS personalizado
        css = """
        .main-title {
            text-align: center;
            margin-bottom: 20px;
        }
        .result-box {
            padding: 20px;
            border-radius: 10px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        """
        
        with gr.Blocks(css=css, title="SignLMM POC") as demo:
            gr.Markdown(
                """
                # 🤟 SignLMM - Proof of Concept
                ### El primer modelo multimodal de Lengua de Señas
                
                Sube un video de una seña y el modelo la reconocerá y traducirá.
                """
            )
            
            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.Video(
                        label="📹 Sube un video de una seña",
                        sources=["upload", "webcam"]
                    )
                    
                    submit_btn = gr.Button(
                        "🔍 Analizar seña",
                        variant="primary",
                        size="lg"
                    )
                    
                    gr.Markdown(
                        """
                        **Tips para mejores resultados:**
                        - Buena iluminación
                        - Fondo liso
                        - Manos y cara visibles
                        - Video de 2-5 segundos
                        """
                    )
                
                with gr.Column(scale=1):
                    # Resultados
                    sign_output = gr.Textbox(
                        label="🏷️ Seña detectada",
                        interactive=False
                    )
                    
                    confidence_output = gr.Textbox(
                        label="📊 Confianza",
                        interactive=False
                    )
                    
                    translation_output = gr.Textbox(
                        label="💬 Traducción al español",
                        interactive=False
                    )
                    
            # details_output eliminado para evitar bug de schema en Gradio
            
            # Función de procesamiento
            def process_and_display(video):
                if video is None:
                    return "No video", "0%", ""
                
                result = self.process_video(video)
                
                if "error" in result and result["error"]:
                    return f"Error: {result['error']}", "0%", ""
                
                return (
                    result.get("seña_detectada", ""),
                    result.get("confianza", "0%"),
                    result.get("traducción", "")
                )
            
            # Conectar eventos
            submit_btn.click(
                fn=process_and_display,
                inputs=[video_input],
                outputs=[sign_output, confidence_output, translation_output]
            )
            
            # Footer
            gr.Markdown(
                """
                ---
                **SignLMM POC** | Desarrollado para ELdeS | 2025
                """
            )
        
        return demo


def create_demo_mode_interface():
    """
    Crea una interfaz de demostración sin modelo entrenado.
    Útil para probar la UI.
    """
    
    # Señas de ejemplo
    EXAMPLE_SIGNS = ["HOLA", "GRACIAS", "YO", "CASA", "TRABAJO", "FAMILIA", "COMER", "BEBER"]
    
    def mock_process(video):
        """Simulación de procesamiento."""
        import random
        
        if video is None:
            return "No video", "0%", ""
        
        # Simular detección
        sign = random.choice(EXAMPLE_SIGNS)
        confidence = random.uniform(0.75, 0.98)
        
        translator = SimpleTranslator()
        translation = translator.translate([sign])
        
        return (
            sign,
            f"{confidence:.1%}",
            translation
        )
    
    with gr.Blocks(title="SignLMM POC - Demo") as demo:
        gr.Markdown(
            """
            # 🤟 SignLMM - MODO DEMO
            ### ⚠️ Usando datos simulados (no hay modelo entrenado)
            
            Esta es una demostración de la interfaz. Para resultados reales, entrena el modelo.
            """
        )
        
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="📹 Video", sources=["upload", "webcam"])
                submit_btn = gr.Button("🔍 Analizar (simulado)", variant="primary")
            
            with gr.Column():
                sign_output = gr.Textbox(label="🏷️ Seña detectada")
                confidence_output = gr.Textbox(label="📊 Confianza")
                translation_output = gr.Textbox(label="💬 Traducción")
        
        submit_btn.click(
            fn=mock_process,
            inputs=[video_input],
            outputs=[sign_output, confidence_output, translation_output]
        )
    
    return demo


def main():
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Demo SignLMM')
    parser.add_argument('--model-dir', default='models', help='Directorio del modelo')
    parser.add_argument('--port', type=int, default=7860, help='Puerto (default: 7860)')
    parser.add_argument('--share', action='store_true', help='Crear URL pública')
    parser.add_argument('--demo-mode', action='store_true', help='Modo demo sin modelo')
    
    args = parser.parse_args()
    
    # FORZAR GRADIO A NO CHEQUEAR LOCALHOST
    os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"
    
    if args.demo_mode:
        print("🎭 Iniciando en modo DEMO (sin modelo)")
        demo = create_demo_mode_interface()
    else:
        print("🚀 Iniciando SignLMM Demo...")
        app = SignLMMDemo(model_dir=args.model_dir)
        demo = app.create_interface()
    
    print(f"🚀 SIRVIENDO EN: http://0.0.0.0:{args.port}")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=False,      # Desactivado para evitar descarga fallida
        show_error=True,
        show_api=False    # Importante para evitar crash de schema
    )


if __name__ == '__main__':
    main()


