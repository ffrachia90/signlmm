"""
SignLMM POC - Preparador de Dataset
===================================
Convierte landmarks JSON a formato listo para entrenar.

Uso:
    python scripts/prepare_dataset.py --input data/landmarks --output data/processed
"""

import os
import json
import argparse
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import pickle


def parse_filename(filename: str) -> dict:
    """
    Extrae metadata del nombre del archivo.
    
    Formato esperado: {LENGUA}_{SEÑA}_{USER}_{TIMESTAMP}.json
    Ejemplo: LSA_HOLA_user123_20250115143200.json
    
    También soporta formato simple: {SEÑA}_{USER}_{TAKE}.json
    Ejemplo: HOLA_signer01_take02.json
    """
    name = Path(filename).stem
    parts = name.split('_')
    
    if len(parts) >= 4 and parts[0] in ['LSA', 'LSE', 'LSU']:
        # Formato completo
        return {
            'lengua': parts[0],
            'seña': parts[1],
            'user': parts[2],
            'timestamp': '_'.join(parts[3:])
        }
    elif len(parts) >= 2:
        # Formato simple
        return {
            'lengua': 'LSA',  # Default
            'seña': parts[0],
            'user': parts[1] if len(parts) > 1 else 'unknown',
            'timestamp': parts[2] if len(parts) > 2 else 'unknown'
        }
    else:
        return {
            'lengua': 'unknown',
            'seña': name,
            'user': 'unknown',
            'timestamp': 'unknown'
        }


def flatten_landmarks(landmarks_frame: dict) -> np.ndarray:
    """
    Aplana los landmarks de un frame a un vector 1D.
    
    Estructura:
    - left_hand: 21 puntos × 3 coords = 63 valores
    - right_hand: 21 puntos × 3 coords = 63 valores  
    - pose: 8 puntos × 3 coords = 24 valores
    - Total: 150 valores por frame
    """
    features = []
    
    # Mano izquierda (21 × 3 = 63)
    if landmarks_frame.get('left_hand'):
        features.extend(np.array(landmarks_frame['left_hand']).flatten())
    else:
        features.extend([0.0] * 63)
    
    # Mano derecha (21 × 3 = 63)
    if landmarks_frame.get('right_hand'):
        features.extend(np.array(landmarks_frame['right_hand']).flatten())
    else:
        features.extend([0.0] * 63)
    
    # Pose (8 × 3 = 24)
    if landmarks_frame.get('pose'):
        features.extend(np.array(landmarks_frame['pose']).flatten())
    else:
        features.extend([0.0] * 24)
    
    return np.array(features, dtype=np.float32)


def normalize_sequence(sequence: list, target_length: int = 60) -> np.ndarray:
    """
    Normaliza una secuencia a longitud fija.
    
    - Si es más larga: submuestrea uniformemente
    - Si es más corta: repite el último frame (padding)
    """
    current_length = len(sequence)
    
    if current_length == 0:
        # Secuencia vacía, retornar zeros
        return np.zeros((target_length, len(flatten_landmarks({}))), dtype=np.float32)
    
    if current_length == target_length:
        return np.array(sequence, dtype=np.float32)
    
    if current_length > target_length:
        # Submuestrear
        indices = np.linspace(0, current_length - 1, target_length, dtype=int)
        return np.array([sequence[i] for i in indices], dtype=np.float32)
    else:
        # Padding con último frame
        padding = [sequence[-1]] * (target_length - current_length)
        return np.array(sequence + padding, dtype=np.float32)


def load_landmarks_file(filepath: str) -> tuple[np.ndarray, dict]:
    """Carga y procesa un archivo de landmarks."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Extraer y aplanar landmarks por frame
    sequence = []
    for frame_data in data.get('landmarks', []):
        flat_features = flatten_landmarks(frame_data)
        sequence.append(flat_features)
    
    return sequence, data.get('metadata', {})


def prepare_dataset(
    input_dir: str,
    output_dir: str,
    sequence_length: int = 60,
    test_size: float = 0.2,
    min_samples_per_class: int = 5,
    random_seed: int = 42
):
    """
    Prepara el dataset completo para entrenamiento.
    
    Args:
        input_dir: Directorio con archivos JSON de landmarks
        output_dir: Directorio donde guardar el dataset procesado
        sequence_length: Longitud normalizada de cada secuencia
        test_size: Proporción para test set
        min_samples_per_class: Mínimo de muestras para incluir una clase
        random_seed: Semilla para reproducibilidad
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Buscar todos los archivos de landmarks
    landmark_files = list(input_path.rglob('*.json'))
    
    if not landmark_files:
        print(f"No se encontraron archivos JSON en {input_dir}")
        return
    
    print(f"Encontrados {len(landmark_files)} archivos de landmarks")
    
    # Cargar todos los datos
    X = []  # Features
    y = []  # Labels (señas)
    metadata_list = []
    
    for filepath in tqdm(landmark_files, desc="Cargando landmarks"):
        try:
            # Extraer seña del nombre del archivo
            file_info = parse_filename(filepath.name)
            seña = file_info['seña']
            
            # Cargar landmarks
            sequence, meta = load_landmarks_file(str(filepath))
            
            if len(sequence) < 5:  # Muy pocos frames, saltar
                continue
            
            # Normalizar longitud
            normalized = normalize_sequence(sequence, target_length=sequence_length)
            
            X.append(normalized)
            y.append(seña)
            metadata_list.append({
                'file': filepath.name,
                **file_info,
                **meta
            })
            
        except Exception as e:
            print(f"\nError procesando {filepath}: {e}")
    
    if not X:
        print("No se pudieron procesar archivos")
        return
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    
    print(f"\nDataset cargado: {X.shape[0]} muestras, {X.shape[1]} frames, {X.shape[2]} features")
    
    # Contar muestras por clase
    class_counts = Counter(y)
    print(f"Clases encontradas: {len(class_counts)}")
    
    # Filtrar clases con pocas muestras
    valid_classes = {cls for cls, count in class_counts.items() if count >= min_samples_per_class}
    
    if len(valid_classes) < len(class_counts):
        print(f"Filtrando clases con menos de {min_samples_per_class} muestras...")
        mask = np.array([label in valid_classes for label in y])
        X = X[mask]
        y = y[mask]
        metadata_list = [m for m, valid in zip(metadata_list, mask) if valid]
        print(f"Muestras después del filtro: {len(X)}")
    
    # Codificar labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"Clases finales: {len(label_encoder.classes_)}")
    print(f"Señas: {list(label_encoder.classes_)}")
    
    # Split train/test
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y_encoded, np.arange(len(X)),
        test_size=test_size,
        random_state=random_seed,
        stratify=y_encoded
    )
    
    print(f"\nTrain: {len(X_train)} muestras")
    print(f"Test: {len(X_test)} muestras")
    
    # Guardar dataset
    np.save(output_path / 'X_train.npy', X_train)
    np.save(output_path / 'X_test.npy', X_test)
    np.save(output_path / 'y_train.npy', y_train)
    np.save(output_path / 'y_test.npy', y_test)
    
    # Guardar label encoder y metadata
    with open(output_path / 'label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump({
            'num_classes': len(label_encoder.classes_),
            'classes': list(label_encoder.classes_),
            'sequence_length': sequence_length,
            'feature_dim': X.shape[2],
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'class_distribution': {str(k): int(v) for k, v in class_counts.items()}
        }, f, indent=2)
    
    # Guardar estadísticas para normalización
    mean = X_train.mean(axis=(0, 1))
    std = X_train.std(axis=(0, 1)) + 1e-8  # Evitar división por cero
    np.save(output_path / 'mean.npy', mean)
    np.save(output_path / 'std.npy', std)
    
    print(f"\n✅ Dataset guardado en {output_path}")
    print(f"   - X_train.npy: {X_train.shape}")
    print(f"   - X_test.npy: {X_test.shape}")
    print(f"   - label_encoder.pkl")
    print(f"   - metadata.json")


def main():
    parser = argparse.ArgumentParser(description='Prepara dataset para entrenamiento')
    parser.add_argument('--input', '-i', required=True, help='Directorio con landmarks JSON')
    parser.add_argument('--output', '-o', required=True, help='Directorio para dataset procesado')
    parser.add_argument('--seq-length', type=int, default=60, help='Longitud de secuencia (default: 60)')
    parser.add_argument('--test-size', type=float, default=0.2, help='Proporción test set (default: 0.2)')
    parser.add_argument('--min-samples', type=int, default=5, help='Mínimo muestras por clase (default: 5)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    prepare_dataset(
        input_dir=args.input,
        output_dir=args.output,
        sequence_length=args.seq_length,
        test_size=args.test_size,
        min_samples_per_class=args.min_samples,
        random_seed=args.seed
    )


if __name__ == '__main__':
    main()


