import sys
import os
import argparse
from pathlib import Path

# Agregar root al path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Importar nuestros modulos
try:
    from scripts.extract_landmarks import LandmarkExtractor
except ImportError:
    print("❌ Error importando scripts. Asegurate de estar en el root del proyecto.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='SignLMM CLI Demo')
    parser.add_argument('video_path', help='Ruta al video (mp4/webm)')
    args = parser.parse_args()
    
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"❌ El archivo no existe: {video_path}")
        return

    print(f"\n🎥 Procesando video: {video_path.name}")
    print("⏳ Iniciando extractor (MediaPipe)...")
    
    try:
        extractor = LandmarkExtractor()
        
        # 1. Extraer
        data = extractor.extract_from_video(str(video_path))
        num_frames = len(data['landmarks'])
        duration = data['metadata']['duration_ms'] / 1000.0
        
        print(f"✅ Extracción exitosa!")
        print(f"   - Frames: {num_frames}")
        print(f"   - Duración: {duration:.2f}s")
        print(f"   - Resolución: {data['metadata']['width']}x{data['metadata']['height']}")
        
        # 2. Simular predicción (ya que no tenemos modelo entrenado aun)
        print("\n🧠 Analizando movimiento...")
        # (Aquí iría model.predict(data))
        print("ℹ️  Nota: Usando modelo simulado para POC")
        
        print("\n" + "="*40)
        print("🤟 RESULTADO DETECTADO")
        print("="*40)
        print(f"🏷️  SEÑA:      [HOLA]")     # Simulado
        print(f"📊 CONFIANZA: 94.5%")      # Simulado
        print(f"💬 TEXTO:     'Hola'")     # Simulado
        print("="*40 + "\n")

    except Exception as e:
        print(f"❌ Error fatal: {e}")

if __name__ == "__main__":
    main()
