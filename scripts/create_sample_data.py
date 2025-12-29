#!/usr/bin/env python
"""
SignLMM POC - Crear datos de ejemplo
====================================
Genera datos sintéticos para probar el pipeline sin videos reales.

Uso:
    python scripts/create_sample_data.py
"""

import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm


# Señas de ejemplo para la POC
SAMPLE_SIGNS = [
    "HOLA", "CHAU", "GRACIAS", "POR-FAVOR", "PERDON",
    "YO", "TU", "EL", "NOSOTROS", "ELLOS",
    "CASA", "TRABAJO", "FAMILIA", "AMIGO", "COMIDA",
    "IR", "VENIR", "COMER", "BEBER", "DORMIR",
    "SI", "NO", "BIEN", "MAL", "MAS",
    "QUE", "QUIEN", "DONDE", "CUANDO", "COMO",
    "HOY", "AYER", "MANANA", "AHORA", "DESPUES",
    "AGUA", "CAFE", "PAN", "CARNE", "FRUTA",
    "GRANDE", "PEQUENO", "MUCHO", "POCO", "TODO",
    "QUERER", "PODER", "SABER", "TENER", "SER"
]

NUM_SIGNERS = 5
VIDEOS_PER_SIGN = 10


def generate_synthetic_landmarks(
    num_frames: int = 60,
    sign_variation: float = 0.1,
    noise_level: float = 0.02
) -> dict:
    """
    Genera landmarks sintéticos que simulan una seña.
    
    Los landmarks son sintéticos pero tienen estructura similar a MediaPipe.
    """
    # Base positions para manos (21 puntos x 3 coords)
    base_left_hand = np.random.randn(21, 3) * 0.1 + np.array([0.3, 0.5, 0])
    base_right_hand = np.random.randn(21, 3) * 0.1 + np.array([0.7, 0.5, 0])
    
    # Movimiento característico de la seña (sinusoidal con variación)
    t = np.linspace(0, 2 * np.pi, num_frames)
    movement_pattern = np.sin(t + np.random.randn() * sign_variation)
    
    landmarks = []
    
    for frame_idx in range(num_frames):
        # Agregar movimiento
        movement = movement_pattern[frame_idx] * 0.1
        
        # Landmarks con movimiento y ruido
        left_hand = base_left_hand + movement + np.random.randn(21, 3) * noise_level
        right_hand = base_right_hand - movement + np.random.randn(21, 3) * noise_level
        
        # Pose (8 puntos simplificados)
        pose = np.random.randn(8, 3) * 0.1 + np.array([0.5, 0.3, 0])
        pose += np.random.randn(8, 3) * noise_level
        
        frame_data = {
            "frame": frame_idx,
            "timestamp_ms": int((frame_idx / 30) * 1000),
            "left_hand": left_hand.tolist(),
            "right_hand": right_hand.tolist(),
            "pose": pose.tolist(),
            "face": None
        }
        
        landmarks.append(frame_data)
    
    return {
        "metadata": {
            "source_video": "synthetic",
            "fps": 30,
            "total_frames": num_frames,
            "width": 1280,
            "height": 720,
            "duration_ms": int((num_frames / 30) * 1000)
        },
        "landmarks": landmarks
    }


def create_sample_dataset(output_dir: str = "data/landmarks"):
    """Crea un dataset de landmarks sintéticos."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generando datos sintéticos en {output_path}")
    print(f"  - Señas: {len(SAMPLE_SIGNS)}")
    print(f"  - Señantes: {NUM_SIGNERS}")
    print(f"  - Videos por seña: {VIDEOS_PER_SIGN}")
    print(f"  - Total: {len(SAMPLE_SIGNS) * VIDEOS_PER_SIGN} archivos")
    
    total_files = len(SAMPLE_SIGNS) * VIDEOS_PER_SIGN
    
    with tqdm(total=total_files, desc="Generando") as pbar:
        for sign in SAMPLE_SIGNS:
            # Crear subdirectorio por seña
            sign_dir = output_path / sign
            sign_dir.mkdir(exist_ok=True)
            
            for video_idx in range(VIDEOS_PER_SIGN):
                signer_id = f"signer{(video_idx % NUM_SIGNERS) + 1:02d}"
                take_id = f"take{(video_idx // NUM_SIGNERS) + 1:02d}"
                
                # Generar landmarks con variación por seña
                # Cada seña tiene una "semilla" diferente para que sean distinguibles
                np.random.seed(hash(sign) % (2**32) + video_idx)
                
                num_frames = np.random.randint(45, 90)  # Duración variable
                landmarks_data = generate_synthetic_landmarks(
                    num_frames=num_frames,
                    sign_variation=0.2,
                    noise_level=0.03
                )
                
                # Guardar
                filename = f"{sign}_{signer_id}_{take_id}.json"
                filepath = sign_dir / filename
                
                landmarks_data["metadata"]["source_video"] = filename.replace('.json', '.mp4')
                
                with open(filepath, 'w') as f:
                    json.dump(landmarks_data, f)
                
                pbar.update(1)
    
    print(f"\n✅ Dataset sintético creado en {output_path}")
    print(f"   Total archivos: {total_files}")
    
    # Crear archivo de info
    info = {
        "type": "synthetic",
        "num_signs": len(SAMPLE_SIGNS),
        "signs": SAMPLE_SIGNS,
        "num_signers": NUM_SIGNERS,
        "videos_per_sign": VIDEOS_PER_SIGN,
        "total_files": total_files
    }
    
    with open(output_path / "_dataset_info.json", 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"\n📋 Siguiente paso:")
    print(f"   python scripts/prepare_dataset.py -i {output_dir} -o data/processed")


if __name__ == '__main__':
    create_sample_dataset()


