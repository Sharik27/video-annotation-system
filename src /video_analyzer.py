"""
video_analyzer.py
==================
Análisis exploratorio avanzado de videos usando OpenCV.

Características:
- Análisis de metadatos técnicos
- Histogramas de color y distribución de intensidad
- Detección de cambios de escena
- Análisis de movimiento con flujo óptico
- Generación de reportes en CSV y JSON
- Visualizaciones avanzadas

Uso:
    python src/video_analyzer.py --input_dir ./data/videos --output_dir ./reports
"""

import cv2
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import argparse


class VideoAnalyzer:
    """Clase para análisis exploratorio de videos."""
    
    def __init__(self, output_dir: str = "./reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        
    def extract_technical_metadata(self, video_path: str) -> Optional[Dict]:
        """Extrae metadatos técnicos del video."""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"⚠️  No se pudo abrir: {video_path}")
            return None
        
        # Obtener propiedades básicas
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec = int(cap.get(cv2.CAP_PROP_FOURCC))
        
        # Calcular duración y tamaño
        duration = total_frames / fps if fps > 0 else 0
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        
        # Decodificar codec
        codec_str = "".join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])
        
        cap.release()
        
        return {
            "filename": os.path.basename(video_path),
            "filepath": str(video_path),
            "width": width,
            "height": height,
            "resolution": f"{width}x{height}",
            "fps": round(fps, 2),
            "total_frames": total_frames,
            "duration_seconds": round(duration, 2),
            "file_size_mb": round(file_size_mb, 2),
            "codec": codec_str,
            "aspect_ratio": round(width / height, 2) if height > 0 else 0
        }
    
    def analyze_color_distribution(self, video_path: str, 
                                   sample_frames: int = 50) -> Dict:
        """Analiza la distribución de colores en el video."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Seleccionar frames uniformemente distribuidos
        frame_indices = np.linspace(0, total_frames - 1, 
                                   min(sample_frames, total_frames), 
                                   dtype=int)
        
        mean_colors = []
        std_colors = []
        histograms = {'red': [], 'green': [], 'blue': []}
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Calcular estadísticas de color
            mean_colors.append(frame.mean(axis=(0, 1)))
            std_colors.append(frame.std(axis=(0, 1)))
            
            # Calcular histogramas por canal
            for i, color in enumerate(['blue', 'green', 'red']):
                hist = cv2.calcHist([frame], [i], None, [256], [0, 256])
                histograms[color].append(hist.flatten())
        
        cap.release()
        
        # Promediar resultados
        mean_color = np.mean(mean_colors, axis=0)
        std_color = np.mean(std_colors, axis=0)
        
        return {
            "mean_blue": round(float(mean_color[0]), 2),
            "mean_green": round(float(mean_color[1]), 2),
            "mean_red": round(float(mean_color[2]), 2),
            "std_blue": round(float(std_color[0]), 2),
            "std_green": round(float(std_color[1]), 2),
            "std_red": round(float(std_color[2]), 2),
            "avg_brightness": round(float(np.mean(mean_color)), 2),
            "color_variance": round(float(np.mean(std_color)), 2)
        }
    
    def detect_scene_changes(self, video_path: str, 
                            threshold: float = 30.0) -> Dict:
        """Detecta cambios de escena usando diferencias entre frames."""
        cap = cv2.VideoCapture(video_path)
        prev_frame = None
        frame_differences = []
        scene_changes = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convertir a escala de grises
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None:
                # Calcular diferencia absoluta
                diff = cv2.absdiff(prev_frame, gray)
                mean_diff = diff.mean()
                frame_differences.append(mean_diff)
                
                # Detectar cambio de escena
                if mean_diff > threshold:
                    scene_changes.append(frame_idx)
            
            prev_frame = gray.copy()
            frame_idx += 1
        
        cap.release()
        
        return {
            "num_scene_changes": len(scene_changes),
            "avg_frame_difference": round(float(np.mean(frame_differences)), 2),
            "max_frame_difference": round(float(np.max(frame_differences)), 2),
            "scene_change_frames": scene_changes[:10]  # Primeros 10
        }
    
    def analyze_motion(self, video_path: str, skip_frames: int = 5) -> Dict:
        """Analiza el movimiento usando flujo óptico."""
        cap = cv2.VideoCapture(video_path)
        ret, prev_frame = cap.read()
        
        if not ret:
            cap.release()
            return {"motion_score": 0, "avg_motion_magnitude": 0}
        
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        motion_magnitudes = []
        frame_count = 0
        
        while True:
            # Saltar frames para acelerar
            for _ in range(skip_frames):
                cap.grab()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calcular flujo óptico
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            # Calcular magnitud del movimiento
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            motion_magnitudes.append(magnitude.mean())
            
            prev_gray = gray.copy()
            frame_count += 1
        
        cap.release()
        
        avg_motion = np.mean(motion_magnitudes) if motion_magnitudes else 0
        motion_score = np.percentile(motion_magnitudes, 90) if motion_magnitudes else 0
        
        return {
            "motion_score": round(float(motion_score), 2),
            "avg_motion_magnitude": round(float(avg_motion), 2),
            "motion_std": round(float(np.std(motion_magnitudes)), 2)
        }
    
    def analyze_video(self, video_path: str) -> Dict:
        """Realiza análisis completo de un video."""
        print(f"📹 Analizando: {os.path.basename(video_path)}")
        
        result = {"timestamp": datetime.now().isoformat()}
        
        # Metadatos técnicos
        metadata = self.extract_technical_metadata(video_path)
        if metadata is None:
            return None
        result.update(metadata)
        
        # Análisis de color
        print("  → Analizando distribución de color...")
        color_info = self.analyze_color_distribution(video_path)
        result.update(color_info)
        
        # Detección de escenas
        print("  → Detectando cambios de escena...")
        scene_info = self.detect_scene_changes(video_path)
        result.update(scene_info)
        
        # Análisis de movimiento
        print("  → Analizando movimiento...")
        motion_info = self.analyze_motion(video_path)
        result.update(motion_info)
        
        print(f"  ✓ Completado\n")
        return result
    
    def analyze_directory(self, input_dir: str, 
                         extensions: Tuple[str, ...] = ('.mp4', '.avi', '.mov')):
        """Analiza todos los videos en un directorio."""
        video_files = []
        input_path = Path(input_dir)
        
        for ext in extensions:
            video_files.extend(input_path.glob(f"*{ext}"))
        
        if not video_files:
            print(f"⚠️  No se encontraron videos en {input_dir}")
            return
        
        print(f"\n{'='*60}")
        print(f"Encontrados {len(video_files)} videos para analizar")
        print(f"{'='*60}\n")
        
        for video_file in video_files:
            result = self.analyze_video(str(video_file))
            if result:
                self.results.append(result)
    
    def generate_reports(self):
        """Genera reportes en CSV y JSON."""
        if not self.results:
            print("⚠️  No hay resultados para generar reportes")
            return
        
        # Guardar CSV
        df = pd.DataFrame(self.results)
        csv_path = self.output_dir / "video_analysis_report.csv"
        df.to_csv(csv_path, index=False)
        print(f"📊 CSV generado: {csv_path}")
        
        # Guardar JSON detallado
        json_path = self.output_dir / "video_analysis_report.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"📄 JSON generado: {json_path}")
        
        # Estadísticas descriptivas
        print("\n" + "="*60)
        print("ESTADÍSTICAS DESCRIPTIVAS")
        print("="*60)
        print(df[['duration_seconds', 'fps', 'file_size_mb', 
                  'avg_brightness', 'motion_score']].describe())
    
    def create_visualizations(self):
        """Crea visualizaciones de los resultados."""
        if not self.results:
            return
        
        df = pd.DataFrame(self.results)
        sns.set_style("whitegrid")
        
        # 1. Distribución de duraciones
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        axes[0, 0].hist(df['duration_seconds'], bins=15, 
                       color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Distribución de Duraciones', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Duración (segundos)')
        axes[0, 0].set_ylabel('Frecuencia')
        
        # 2. Brillo vs Movimiento
        scatter = axes[0, 1].scatter(df['avg_brightness'], df['motion_score'],
                                     c=df['duration_seconds'], cmap='viridis',
                                     s=100, alpha=0.6, edgecolors='black')
        axes[0, 1].set_title('Brillo vs Movimiento', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Brillo Promedio')
        axes[0, 1].set_ylabel('Score de Movimiento')
        plt.colorbar(scatter, ax=axes[0, 1], label='Duración (s)')
        
        # 3. Cambios de escena
        axes[1, 0].bar(range(len(df)), df['num_scene_changes'], 
                      color='coral', edgecolor='black')
        axes[1, 0].set_title('Cambios de Escena por Video', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Índice de Video')
        axes[1, 0].set_ylabel('Número de Cambios')
        
        # 4. Tamaño de archivo vs Duración
        axes[1, 1].scatter(df['duration_seconds'], df['file_size_mb'],
                          color='green', s=100, alpha=0.6, edgecolors='black')
        axes[1, 1].set_title('Tamaño vs Duración', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Duración (segundos)')
        axes[1, 1].set_ylabel('Tamaño (MB)')
        
        plt.tight_layout()
        viz_path = self.output_dir / "video_analysis_visualizations.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"📈 Visualizaciones guardadas: {viz_path}")
        plt.close()
        
        # Gráfico de color promedio
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(df))
        width = 0.25
        
        ax.bar(x - width, df['mean_red'], width, label='Rojo', color='red', alpha=0.7)
        ax.bar(x, df['mean_green'], width, label='Verde', color='green', alpha=0.7)
        ax.bar(x + width, df['mean_blue'], width, label='Azul', color='blue', alpha=0.7)
        
        ax.set_xlabel('Videos')
        ax.set_ylabel('Intensidad Promedio')
        ax.set_title('Distribución de Color RGB por Video', fontsize=14, fontweight='bold')
        ax.legend()
        ax.set_xticks(x)
        ax.set_xticklabels([f"V{i+1}" for i in range(len(df))], rotation=45)
        
        plt.tight_layout()
        color_path = self.output_dir / "color_distribution.png"
        plt.savefig(color_path, dpi=300, bbox_inches='tight')
        print(f"🎨 Distribución de color guardada: {color_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Análisis exploratorio avanzado de videos"
    )
    parser.add_argument('--input_dir', type=str, default='./data/videos',
                       help='Directorio con videos de entrada')
    parser.add_argument('--output_dir', type=str, default='./reports',
                       help='Directorio para guardar reportes')
    
    args = parser.parse_args()
    
    # Crear analizador
    analyzer = VideoAnalyzer(output_dir=args.output_dir)
    
    # Analizar videos
    analyzer.analyze_directory(args.input_dir)
    
    # Generar reportes
    analyzer.generate_reports()
    
    # Crear visualizaciones
    analyzer.create_visualizations()
    
    print(f"\n{'='*60}")
    print("✅ Análisis completado exitosamente")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
