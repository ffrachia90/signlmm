"""
SignLMM POC - Extractor de Landmarks
====================================
Extrae landmarks de manos y pose de videos usando MediaPipe (Tasks API).

Uso:
    python scripts/extract_landmarks.py --input data/videos --output data/landmarks
"""

import os
import json
import argparse
import urllib.request
from pathlib import Path
from tqdm import tqdm
import cv2
import mediapipe as mp


class LandmarkExtractor:
    """
    Extrae landmarks de videos de lengua de señas usando MediaPipe Tasks API.

    Nota: En algunas builds de `mediapipe` para macOS solo está disponible `mediapipe.tasks`
    (no `mp.solutions`). Esta implementación funciona con esa variante.
    """
    
    DEFAULT_HAND_MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    )
    DEFAULT_POSE_MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
    )

    def __init__(
        self,
        models_dir: str = "models/mediapipe",
        download_models: bool = True,
    ):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.hand_model_path = self.models_dir / "hand_landmarker.task"
        self.pose_model_path = self.models_dir / "pose_landmarker_full.task"

        if download_models:
            self._ensure_model(self.DEFAULT_HAND_MODEL_URL, self.hand_model_path)
            self._ensure_model(self.DEFAULT_POSE_MODEL_URL, self.pose_model_path)

        if not self.hand_model_path.exists() or not self.pose_model_path.exists():
            raise RuntimeError(
                "Faltan modelos .task de MediaPipe. "
                "Habilitá download_models=True o descargalos manualmente en "
                f"'{self.models_dir}'. Esperados: {self.hand_model_path.name}, {self.pose_model_path.name}"
            )

        # Import tardío: evita errores si mediapipe no trae tasks
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python import vision

        self.vision = vision

        # Landmarkers (modo IMAGE: procesamos frame por frame)
        hand_options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.hand_model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
        )
        pose_options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.pose_model_path)),
            running_mode=vision.RunningMode.IMAGE,
        )

        self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

    def _ensure_model(self, url: str, dest: Path):
        """Descarga un modelo si no existe localmente."""
        if dest.exists():
            return
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            print(f"⬇️  Descargando modelo MediaPipe: {dest.name}")
            urllib.request.urlretrieve(url, dest)  # nosec - URL fija (modelos públicos)
        except Exception as e:
            raise RuntimeError(f"No se pudo descargar {url} → {dest}: {e}")
        
    def extract_from_video(self, video_path: str) -> dict:
        """
        Extrae landmarks de un video.
        
        Returns:
            dict con metadata y secuencia de landmarks por frame
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video_path}")
        
        # Metadata del video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        landmarks_sequence = []
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convertir BGR a RGB para MediaPipe Tasks
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            hand_result = self.hand_landmarker.detect(mp_image)
            pose_result = self.pose_landmarker.detect(mp_image)
            
            # Extraer landmarks de este frame
            frame_landmarks = {
                "frame": frame_idx,
                "timestamp_ms": int((frame_idx / fps) * 1000) if fps > 0 else 0,
                "left_hand": self._extract_hand_landmarks(hand_result, target="Left"),
                "right_hand": self._extract_hand_landmarks(hand_result, target="Right"),
                "pose": self._extract_pose_landmarks(pose_result),
                "face": None
            }
            
            landmarks_sequence.append(frame_landmarks)
            frame_idx += 1
        
        cap.release()
        
        return {
            "metadata": {
                "source_video": os.path.basename(video_path),
                "fps": fps,
                "total_frames": total_frames,
                "width": width,
                "height": height,
                "duration_ms": int((total_frames / fps) * 1000) if fps > 0 else 0
            },
            "landmarks": landmarks_sequence
        }
    
    def _extract_hand_landmarks(self, hand_result, target: str):
        """
        Extrae 21 puntos de una mano (Left/Right) desde HandLandmarkerResult.
        Usa la handedness para mapear correctamente.
        """
        try:
            if not getattr(hand_result, "hand_landmarks", None):
                return None
            hands = hand_result.hand_landmarks  # list[list[NormalizedLandmark]]
            handed = getattr(hand_result, "handedness", None)  # list[list[Category]]
            if handed:
                # Elegir la mano cuya handedness coincida
                for idx, cats in enumerate(handed):
                    if cats and cats[0].category_name == target:
                        lm_list = hands[idx]
                        return [[lm.x, lm.y, lm.z] for lm in lm_list]
            # Fallback: si no hay handedness, tomar primera/segunda
            if target == "Left" and len(hands) >= 1:
                return [[lm.x, lm.y, lm.z] for lm in hands[0]]
            if target == "Right" and len(hands) >= 2:
                return [[lm.x, lm.y, lm.z] for lm in hands[1]]
            return None
        except Exception:
            return None
    
    def _extract_pose_landmarks(self, pose_result):
        """Extrae puntos relevantes del pose (torso y brazos) desde PoseLandmarkerResult."""
        try:
            if not getattr(pose_result, "pose_landmarks", None):
                return None
            if len(pose_result.pose_landmarks) == 0:
                return None

            pose_lms = pose_result.pose_landmarks[0]  # 33 landmarks
            relevant_indices = [11, 12, 13, 14, 15, 16, 23, 24]
            out = []
            for i in relevant_indices:
                lm = pose_lms[i]
                out.append([lm.x, lm.y, lm.z])
            return out
        except Exception:
            return None
    
    def close(self):
        """Libera recursos."""
        try:
            self.hand_landmarker.close()
        except Exception:
            pass
        try:
            self.pose_landmarker.close()
        except Exception:
            pass


def process_video_directory(input_dir: str, output_dir: str, skip_existing: bool = True):
    """
    Procesa todos los videos de un directorio.
    
    Args:
        input_dir: Directorio con videos
        output_dir: Directorio donde guardar los landmarks
        skip_existing: Si es True, salta videos ya procesados
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Buscar todos los videos
    video_extensions = {'.mp4', '.webm', '.avi', '.mov', '.mkv'}
    video_files = [f for f in input_path.rglob('*') if f.suffix.lower() in video_extensions]
    
    if not video_files:
        print(f"No se encontraron videos en {input_dir}")
        return
    
    print(f"Encontrados {len(video_files)} videos para procesar")
    
    extractor = LandmarkExtractor()
    processed = 0
    errors = 0
    
    for video_file in tqdm(video_files, desc="Extrayendo landmarks"):
        # Determinar ruta de salida
        relative_path = video_file.relative_to(input_path)
        output_file = output_path / relative_path.with_suffix('.json')
        
        # Crear subdirectorios si es necesario
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Saltar si ya existe
        if skip_existing and output_file.exists():
            continue
        
        try:
            landmarks_data = extractor.extract_from_video(str(video_file))
            
            # Guardar como JSON
            with open(output_file, 'w') as f:
                json.dump(landmarks_data, f)
            
            processed += 1
            
        except Exception as e:
            print(f"\nError procesando {video_file}: {e}")
            errors += 1
    
    extractor.close()
    
    print(f"\n✅ Procesados: {processed}")
    print(f"❌ Errores: {errors}")
    print(f"⏭️  Saltados (ya existían): {len(video_files) - processed - errors}")


def main():
    parser = argparse.ArgumentParser(description='Extrae landmarks de videos de señas')
    parser.add_argument('--input', '-i', required=True, help='Directorio con videos')
    parser.add_argument('--output', '-o', required=True, help='Directorio para guardar landmarks')
    parser.add_argument('--no-skip', action='store_true', help='Re-procesar videos existentes')
    parser.add_argument('--models-dir', default='models/mediapipe', help='Directorio para modelos .task')
    parser.add_argument('--no-download-models', action='store_true', help='No descargar modelos automáticamente')
    
    args = parser.parse_args()
    
    # Instanciar extractor temprano para descargar modelos y fallar rápido si hay problemas
    extractor = LandmarkExtractor(models_dir=args.models_dir, download_models=not args.no_download_models)
    extractor.close()

    process_video_directory(
        input_dir=args.input,
        output_dir=args.output,
        skip_existing=not args.no_skip
    )


if __name__ == '__main__':
    main()


