# 🤟 SignLMM POC

**Proof of Concept del primer modelo multimodal de Lengua de Señas**

Desarrollado para **ELdeS** - Plataforma de aprendizaje de lengua de señas.

---

## 📋 Descripción

Este proyecto es una POC (Proof of Concept) para crear un modelo que pueda:

1. **Reconocer** señas a partir de video
2. **Clasificar** qué seña se está realizando
3. **Traducir** las señas detectadas a español natural

## 🏗️ Arquitectura

```
Video de señas
      │
      ▼
┌─────────────────┐
│  MediaPipe      │  Extrae landmarks de manos y pose
│  (Landmarks)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LSTM           │  Clasifica la secuencia de landmarks
│  Bidireccional  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLM            │  Traduce glosas a español natural
│  (GPT/Gemini)   │
└────────┬────────┘
         │
         ▼
   "Hola, ¿cómo estás?"
```

## 📁 Estructura del proyecto

```
signlmm-poc/
├── data/
│   ├── videos/          # Videos originales de señas
│   ├── landmarks/       # Landmarks extraídos (JSON)
│   └── processed/       # Dataset listo para entrenar
├── scripts/
│   ├── extract_landmarks.py   # Extrae landmarks de videos
│   ├── prepare_dataset.py     # Prepara dataset para entrenar
│   ├── train_model.py         # Entrena el clasificador
│   └── translate.py           # Traduce glosas a español
├── models/              # Modelos entrenados
├── demo/
│   └── app.py          # Aplicación web demo
├── config/
│   └── config.yaml     # Configuración
├── requirements.txt
└── README.md
```

## 🚀 Instalación

### 1. Clonar y crear entorno

```bash
cd signlmm-poc
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Configurar API keys (opcional, para traducción LLM)

```bash
cp .env.example .env
# Editar .env con tu API key de OpenAI o Gemini
```

## 📊 Pipeline completo

### Paso 1: Preparar videos

Organiza tus videos en `data/videos/` con el formato:
```
{SEÑA}_{usuario}_{toma}.mp4

Ejemplos:
HOLA_user01_take01.mp4
GRACIAS_user01_take01.mp4
YO_user02_take01.mp4
```

### Paso 2: Extraer landmarks

```bash
python scripts/extract_landmarks.py \
    --input data/videos \
    --output data/landmarks
```

### Paso 3: Preparar dataset

```bash
python scripts/prepare_dataset.py \
    --input data/landmarks \
    --output data/processed
```

### Paso 4: Entrenar modelo

```bash
python scripts/train_model.py \
    --data data/processed \
    --output models \
    --epochs 100
```

### Paso 5: Ejecutar demo

```bash
python demo/app.py
```

Abrir en el navegador: http://localhost:7860

## 🎮 Modo demo (sin modelo)

Para probar la interfaz sin entrenar el modelo:

```bash
python demo/app.py --demo-mode
```

## 📈 Métricas esperadas

| Dataset | Accuracy esperada |
|---------|-------------------|
| 500 videos (50 señas) | 70-80% |
| 2,500 videos (50 señas) | 85-90% |
| 10,000 videos (100 señas) | 90-95% |

## 🔧 Configuración

Editar `config/config.yaml` para ajustar:

- Parámetros del modelo (hidden_size, num_layers, etc.)
- Longitud de secuencia normalizada
- Learning rate y epochs
- Proveedor de LLM para traducción

## 📝 Formato de datos

### Videos de entrada
- Formato: MP4, WebM, AVI
- Resolución: Mínimo 720p
- FPS: 30
- Duración: 2-10 segundos
- Contenido: Una seña por video, manos y cara visibles

### Landmarks extraídos (JSON)
```json
{
  "metadata": {
    "source_video": "HOLA_user01_take01.mp4",
    "fps": 30,
    "total_frames": 90
  },
  "landmarks": [
    {
      "frame": 0,
      "left_hand": [[x, y, z], ...],
      "right_hand": [[x, y, z], ...],
      "pose": [[x, y, z], ...]
    }
  ]
}
```

## 🤝 Contribuir

1. Fork del repositorio
2. Crear branch: `git checkout -b feature/nueva-funcionalidad`
3. Commit: `git commit -am 'Agregar nueva funcionalidad'`
4. Push: `git push origin feature/nueva-funcionalidad`
5. Pull Request

## 📄 Licencia

Propiedad de ELdeS. Uso interno.

## 👥 Equipo

- **ELdeS** - Plataforma de aprendizaje de lengua de señas
- **Fernando Frachia** - Desarrollo POC

---

**SignLMM POC** | 2025 | 🇦🇷🇪🇸🇺🇾


