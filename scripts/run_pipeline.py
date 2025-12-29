#!/usr/bin/env python
"""
SignLMM POC - Pipeline Completo
===============================
Ejecuta todo el pipeline de una vez: extract → prepare → train

Uso:
    python scripts/run_pipeline.py --videos data/videos
    
    O paso a paso:
    python scripts/run_pipeline.py --step extract
    python scripts/run_pipeline.py --step prepare
    python scripts/run_pipeline.py --step train
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list, description: str):
    """Ejecuta un comando y muestra el resultado."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Comando: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    
    if result.returncode != 0:
        print(f"\n❌ Error en: {description}")
        sys.exit(1)
    
    print(f"\n✅ Completado: {description}")


def main():
    parser = argparse.ArgumentParser(description='Pipeline completo SignLMM')
    parser.add_argument('--videos', default='data/videos', help='Directorio con videos')
    parser.add_argument('--step', choices=['all', 'extract', 'prepare', 'train', 'demo'], 
                       default='all', help='Paso a ejecutar')
    parser.add_argument('--epochs', type=int, default=100, help='Epochs de entrenamiento')
    parser.add_argument('--skip-existing', action='store_true', help='Saltar videos ya procesados')
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    python = sys.executable
    
    steps = []
    
    if args.step in ['all', 'extract']:
        steps.append({
            'cmd': [python, 'scripts/extract_landmarks.py', 
                   '-i', args.videos, 
                   '-o', 'data/landmarks'],
            'desc': 'Paso 1: Extrayendo landmarks de videos'
        })
    
    if args.step in ['all', 'prepare']:
        steps.append({
            'cmd': [python, 'scripts/prepare_dataset.py',
                   '-i', 'data/landmarks',
                   '-o', 'data/processed'],
            'desc': 'Paso 2: Preparando dataset'
        })
    
    if args.step in ['all', 'train']:
        steps.append({
            'cmd': [python, 'scripts/train_model.py',
                   '-d', 'data/processed',
                   '-o', 'models',
                   '--epochs', str(args.epochs)],
            'desc': 'Paso 3: Entrenando modelo'
        })
    
    if args.step == 'demo':
        steps.append({
            'cmd': [python, 'demo/app.py'],
            'desc': 'Ejecutando demo'
        })
    
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   🤟 SignLMM POC - Pipeline de Entrenamiento             ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    for step in steps:
        run_command(step['cmd'], step['desc'])
    
    print(f"""
    
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   ✅ Pipeline completado exitosamente!                   ║
    ║                                                           ║
    ║   Para ejecutar la demo:                                  ║
    ║   python demo/app.py                                      ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)


if __name__ == '__main__':
    main()


