# Video-Annotation-System 

Sistema avanzado para análisis de videos y extracción de características de movimiento humano usando Computer Vision y MediaPipe.

## Características

### 1. Video Analyzer
- Análisis exhaustivo de metadatos técnicos
- Histogramas de distribución de color
- Detección automática de cambios de escena
- Análisis de movimiento con flujo óptico
- Reportes en CSV/JSON y visualizaciones

### 2. Landmark Extractor
- Extracción de pose, manos y rostro con MediaPipe
- Múltiples formatos de salida (CSV, Parquet, NPY)
- Visualización opcional con landmarks dibujados
- Procesamiento batch optimizado
- Manejo robusto de errores y detección fallida

### 3. Landmark Processor
- Normalización espacial avanzada (3 métodos)
- Filtrado temporal (Moving Average, Savitzky-Golay, Gaussian)
- Interpolación inteligente de valores faltantes
- Cálculo de velocidades y aceleraciones
- Detección de outliers con Z-score
- Cálculo de ángulos articulares

## Instalación

```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/tu-repositorio.git
cd tu-repositorio

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

## Uso

### Análisis de Videos

```bash
python src/video_analyzer.py \
    --input_dir ./data/videos \
    --output_dir ./reports
```

### Extracción de Landmarks

```bash
# Básico (solo pose)
python src/landmark_extractor.py \
    --input_dir ./data/videos \
    --output_dir ./data/landmarks \
    --format parquet

# Con manos y rostro + visualización
python src/landmark_extractor.py \
    --input_dir ./data/videos \
    --output_dir ./data/landmarks \
    --extract_hands \
    --extract_face \
    --visualize
```

### Preprocesamiento de Landmarks

```bash
python src/landmark_processor.py \
    --input_file ./data/landmarks/video_landmarks.parquet \
    --output_file ./data/processed/video_processed.parquet \
    --normalize_method shoulder_hip \
    --filter_type savgol \
    --window_size 7
```

## Opciones de Configuración

### Video Analyzer
- `--input_dir`: Carpeta con videos de entrada
- `--output_dir`: Carpeta para reportes

### Landmark Extractor
- `--frame_skip`: Procesar 1 de cada N frames (default: 1)
- `--format`: Formato de salida: csv, parquet, npy (default: parquet)
- `--visualize`: Generar videos con landmarks
- `--extract_hands`: Extraer landmarks de manos
- `--extract_face`: Extraer landmarks faciales
- `--min_confidence`: Umbral de confianza (default: 0.5)

### Landmark Processor
- `--fps`: FPS del video original (default: 30)
- `--normalize_method`: shoulder_hip, torso, body_bbox
- `--filter_type`: moving_avg, savgol, gaussian
- `--window_size`: Tamaño de ventana para filtrado (default: 7)
- `--no_derivatives`: Desactivar cálculo de velocidades
- `--no_angles`: Desactivar cálculo de ángulos

## Formato de Salida

### Landmarks Extraídos
Cada frame contiene:
- `video_name`, `frame_index`, `timestamp`
- `pose_x_{i}`, `pose_y_{i}`, `pose_z_{i}`, `pose_vis_{i}` (33 landmarks)
- Opcionalmente: hand y face landmarks

### Landmarks Procesados
Además de los anteriores:
- `norm_x_{i}`, `norm_y_{i}`, `norm_z_{i}`: Coordenadas normalizadas
- `vel_x_{i}`, `vel_y_{i}`,
