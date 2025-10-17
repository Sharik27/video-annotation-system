"""
landmark_processor.py
=====================
Preprocesador avanzado de landmarks.

Características:
- Normalización espacial por múltiples métodos
- Filtrado temporal (Moving Average, Savitzky-Golay, Kalman)
- Interpolación de valores faltantes
- Cálculo de características derivadas (velocidad, aceleración)
- Detección y corrección de outliers
- Suavizado adaptativo

Uso:
    python src/landmark_processor.py --input_file ./data/landmarks/video_landmarks.parquet --output_file ./data/processed/video_processed.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
from scipy import signal, interpolate
from scipy.spatial import distance
import argparse
import json
from datetime import datetime


class LandmarkProcessor:
    """Procesador avanzado de landmarks."""
    
    def __init__(self, num_pose_landmarks: int = 33):
        self.num_pose_landmarks = num_pose_landmarks
        
        # Índices clave de MediaPipe Pose
        self.LANDMARK_INDICES = {
            'nose': 0,
            'left_eye': 2,
            'right_eye': 5,
            'left_ear': 7,
            'right_ear': 8,
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28
        }
    
    def load_landmarks(self, file_path: str) -> pd.DataFrame:
        """Carga landmarks desde archivo."""
        path = Path(file_path)
        
        if path.suffix == '.csv':
            return pd.read_csv(file_path)
        elif path.suffix == '.parquet':
            return pd.read_parquet(file_path)
        elif path.suffix == '.npy':
            data = np.load(file_path)
            # Cargar nombres de columnas
            columns_file = str(file_path).replace('.npy', '_columns.json')
            with open(columns_file, 'r') as f:
                columns = json.load(f)
            return pd.DataFrame(data, columns=columns)
        else:
            raise ValueError(f"Formato no soportado: {path.suffix}")
    
    def interpolate_missing_values(self, df: pd.DataFrame, 
                                   method: str = 'cubic') -> pd.DataFrame:
        """
        Interpola valores faltantes en coordenadas de landmarks.
        
        Args:
            method: 'linear', 'cubic', 'nearest'
        """
        df_interp = df.copy()
        
        # Identificar columnas de coordenadas
        coord_cols = [col for col in df.columns if any(
            col.startswith(prefix) for prefix in ['pose_x_', 'pose_y_', 'pose_z_']
        )]
        
        for col in coord_cols:
            # Interpolar solo si hay valores válidos
            if df[col].notna().sum() > 2:
                # Usar interpolate de pandas
                df_interp[col] = df[col].interpolate(method=method, limit_direction='both')
                
                # Si aún quedan NaN, usar forward/backward fill
                df_interp[col] = df_interp[col].fillna(method='ffill').fillna(method='bfill')
        
        return df_interp
    
    def normalize_landmarks(self, df: pd.DataFrame, 
                           method: str = 'shoulder_hip') -> pd.DataFrame:
        """
        Normaliza landmarks espacialmente.
        
        Args:
            method: 'shoulder_hip', 'torso', 'body_bbox'
        """
        df_norm = df.copy()
        
        # Calcular factores de normalización por frame
        scales = []
        centers_x = []
        centers_y = []
        
        for idx, row in df.iterrows():
            if method == 'shoulder_hip':
                scale, center = self._compute_shoulder_hip_normalization(row)
            elif method == 'torso':
                scale, center = self._compute_torso_normalization(row)
            elif method == 'body_bbox':
                scale, center = self._compute_bbox_normalization(row)
            else:
                raise ValueError(f"Método desconocido: {method}")
            
            scales.append(scale)
            centers_x.append(center[0])
            centers_y.append(center[1])
        
        df_norm['normalization_scale'] = scales
        df_norm['center_x'] = centers_x
        df_norm['center_y'] = centers_y
        
        # Aplicar normalización
        for i in range(self.num_pose_landmarks):
            x_col = f'pose_x_{i}'
            y_col = f'pose_y_{i}'
            z_col = f'pose_z_{i}'
            
            if x_col in df.columns:
                df_norm[f'norm_x_{i}'] = (df[x_col] - df_norm['center_x']) / df_norm['normalization_scale']
                df_norm[f'norm_y_{i}'] = (df[y_col] - df_norm['center_y']) / df_norm['normalization_scale']
                df_norm[f'norm_z_{i}'] = df[z_col] / df_norm['normalization_scale']
        
        return df_norm
    
    def _compute_shoulder_hip_normalization(self, row: pd.Series) -> Tuple[float, Tuple[float, float]]:
        """Normalización basada en distancia entre hombros y centro del cuerpo."""
        ls_idx = self.LANDMARK_INDICES['left_shoulder']
        rs_idx = self.LANDMARK_INDICES['right_shoulder']
        lh_idx = self.LANDMARK_INDICES['left_hip']
        rh_idx = self.LANDMARK_INDICES['right_hip']
        
        # Obtener coordenadas
        coords = []
        for idx in [ls_idx, rs_idx, lh_idx, rh_idx]:
            x = row.get(f'pose_x_{idx}', np.nan)
            y = row.get(f'pose_y_{idx}', np.nan)
            coords.append([x, y])
        
        coords = np.array(coords)
        
        # Si hay valores faltantes, usar escala por defecto
        if np.isnan(coords).any():
            return 1.0, (0.5, 0.5)
        
        # Calcular distancia entre hombros como escala
        shoulder_dist = distance.euclidean(coords[0], coords[1])
        scale = shoulder_dist if shoulder_dist > 1e-6 else 1.0
        
        # Centro: promedio de hombros y caderas
        center = np.mean(coords, axis=0)
        
        return scale, tuple(center)
    
    def _compute_torso_normalization(self, row: pd.Series) -> Tuple[float, Tuple[float, float]]:
        """Normalización basada en altura del torso."""
        nose_idx = self.LANDMARK_INDICES['nose']
        lh_idx = self.LANDMARK_INDICES['left_hip']
        rh_idx = self.LANDMARK_INDICES['right_hip']
        
        nose_y = row.get(f'pose_y_{nose_idx}', np.nan)
        lh_y = row.get(f'pose_y_{lh_idx}', np.nan)
        rh_y = row.get(f'pose_y_{rh_idx}', np.nan)
        
        if np.isnan([nose_y, lh_y, rh_y]).any():
            return 1.0, (0.5, 0.5)
        
        hip_avg_y = (lh_y + rh_y) / 2
        torso_height = abs(nose_y - hip_avg_y)
        scale = torso_height if torso_height > 1e-6 else 1.0
        
        # Centro: punto medio del torso
        center_x = (row.get(f'pose_x_{lh_idx}', 0.5) + row.get(f'pose_x_{rh_idx}', 0.5)) / 2
        center_y = (nose_y + hip_avg_y) / 2
        
        return scale, (center_x, center_y)
    
    def _compute_bbox_normalization(self, row: pd.Series) -> Tuple[float, Tuple[float, float]]:
        """Normalización basada en bounding box del cuerpo."""
        xs = []
        ys = []
        
        for i in range(self.num_pose_landmarks):
            x = row.get(f'pose_x_{i}', np.nan)
            y = row.get(f'pose_y_{i}', np.nan)
            if not np.isnan(x) and not np.isnan(y):
                xs.append(x)
                ys.append(y)
        
        if len(xs) < 2:
            return 1.0, (0.5, 0.5)
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        width = max_x - min_x
        height = max_y - min_y
        scale = max(width, height)
        scale = scale if scale > 1e-6 else 1.0
        
        center = ((min_x + max_x) / 2, (min_y + max_y) / 2)
        
        return scale, center
    
    def apply_temporal_filter(self, df: pd.DataFrame, 
                             filter_type: str = 'savgol',
                             window_size: int = 7) -> pd.DataFrame:
        """
        Aplica filtrado temporal para suavizar landmarks.
        
        Args:
            filter_type: 'moving_avg', 'savgol', 'gaussian'
            window_size: Tamaño de ventana para filtrado
        """
        df_filtered = df.copy()
        
        # Columnas a filtrar
        norm_cols = [col for col in df.columns if col.startswith('norm_')]
        
        for col in norm_cols:
            values = df[col].values
            
            if filter_type == 'moving_avg':
                # Media móvil
                filtered = pd.Series(values).rolling(
                    window=window_size, 
                    min_periods=1, 
                    center=True
                ).mean().values
                
            elif filter_type == 'savgol':
                # Savitzky-Golay filter
                if len(values) > window_size:
                    polyorder = min(3, window_size - 1)
                    filtered = signal.savgol_filter(
                        values, 
                        window_length=window_size if window_size % 2 == 1 else window_size + 1,
                        polyorder=polyorder,
                        mode='nearest'
                    )
                else:
                    filtered = values
                    
            elif filter_type == 'gaussian':
                # Gaussian filter
                sigma = window_size / 6
                filtered = signal.gaussian_filter1d(values, sigma=sigma)
            
            else:
                raise ValueError(f"Filtro desconocido: {filter_type}")
            
            df_filtered[col] = filtered
        
        return df_filtered
    
    def compute_velocities(self, df: pd.DataFrame, fps: float = 30.0) -> pd.DataFrame:
        """Calcula velocidades de landmarks."""
        df_vel = df.copy()
        
        for i in range(self.num_pose_landmarks):
            x_col = f'norm_x_{i}'
            y_col = f'norm_y_{i}'
            
            if x_col in df.columns:
                # Calcular diferencias temporales
                dx = df[x_col].diff() * fps
                dy = df[y_col].diff() * fps
                
                # Magnitud de velocidad
                velocity = np.sqrt(dx**2 + dy**2)
                
                df_vel[f'vel_x_{i}'] = dx
                df_vel[f'vel_y_{i}'] = dy
                df_vel[f'vel_mag_{i}'] = velocity
        
        return df_vel
    
    def compute_accelerations(self, df: pd.DataFrame, fps: float = 30.0) -> pd.DataFrame:
        """Calcula aceleraciones de landmarks."""
        df_acc = df.copy()
        
        for i in range(self.num_pose_landmarks):
            vel_x_col = f'vel_x_{i}'
            vel_y_col = f'vel_y_{i}'
            
            if vel_x_col in df.columns:
                # Derivada de velocidad = aceleración
                ax = df[vel_x_col].diff() * fps
                ay = df[vel_y_col].diff() * fps
                
                acceleration = np.sqrt(ax**2 + ay**2)
                
                df_acc[f'acc_x_{i}'] = ax
                df_acc[f'acc_y_{i}'] = ay
                df_acc[f'acc_mag_{i}'] = acceleration
        
        return df_acc
    
    def detect_outliers(self, df: pd.DataFrame, 
                       threshold: float = 3.0) -> pd.DataFrame:
        """
        Detecta outliers usando Z-score.
        
        Args:
            threshold: Número de desviaciones estándar para considerar outlier
        """
        df_clean = df.copy()
        
        norm_cols = [col for col in df.columns if col.startswith('norm_')]
        
        outlier_mask = np.zeros(len(df), dtype=bool)
        
        for col in norm_cols:
            values = df[col].values
            mean = np.nanmean(values)
            std = np.nanstd(values)
            
            if std > 1e-6:
                z_scores = np.abs((values - mean) / std)
                outlier_mask |= (z_scores > threshold)
        
        df_clean['is_outlier'] = outlier_mask
        
        # Opcional: marcar como NaN los outliers para luego interpolar
        for col in norm_cols:
            df_clean.loc[outlier_mask, col] = np.nan
        
        return df_clean
    
    def compute_joint_angles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula ángulos de articulaciones clave."""
        df_angles = df.copy()
        
        # Definir tríos de puntos para ángulos
        angle_defs = {
            'left_elbow': ('left_shoulder', 'left_elbow', 'left_wrist'),
            'right_elbow': ('right_shoulder', 'right_elbow', 'right_wrist'),
            'left_knee': ('left_hip', 'left_knee', 'left_ankle'),
            'right_knee': ('right_hip', 'right_knee', 'right_ankle'),
        }
        
        for angle_name, (p1, p2, p3) in angle_defs.items():
            angles = []
            
            for idx, row in df.iterrows():
                angle = self._compute_angle(row, p1, p2, p3)
                angles.append(angle)
            
            df_angles[f'angle_{angle_name}'] = angles
        
        return df_angles
    
    def _compute_angle(self, row: pd.Series, 
                       point1: str, point2: str, point3: str) -> float:
        """Calcula ángulo entre tres puntos."""
        idx1 = self.LANDMARK_INDICES[point1]
        idx2 = self.LANDMARK_INDICES[point2]
        idx3 = self.LANDMARK_INDICES[point3]
        
        # Obtener coordenadas normalizadas
        p1 = np.array([
            row.get(f'norm_x_{idx1}', np.nan),
            row.get(f'norm_y_{idx1}', np.nan)
        ])
        p2 = np.array([
            row.get(f'norm_x_{idx2}', np.nan),
            row.get(f'norm_y_{idx2}', np.nan)
        ])
        p3 = np.array([
            row.get(f'norm_x_{idx3}', np.nan),
            row.get(f'norm_y_{idx3}', np.nan)
        ])
        
        if np.isnan([p1, p2, p3]).any():
            return np.nan
        
        # Vectores
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Ángulo usando producto punto
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        return np.degrees(angle)
    
    def process_landmarks(self, 
                         input_file: str,
                         output_file: str,
                         fps: float = 30.0,
                         normalize_method: str = 'shoulder_hip',
                         filter_type: str = 'savgol',
                         window_size: int = 7,
                         compute_derivatives: bool = True,
                         compute_angles: bool = True) -> Dict:
        """
        Pipeline completo de procesamiento.
        """
        print(f"\n{'='*60}")
        print(f"Procesando: {Path(input_file).name}")
        print(f"{'='*60}\n")
        
        # 1. Cargar datos
        print("📂 Cargando landmarks...")
        df = self.load_landmarks(input_file)
        original_shape = df.shape
        print(f"   Dimensiones originales: {original_shape}")
        
        # 2. Interpolar valores faltantes
        print("🔧 Interpolando valores faltantes...")
        df = self.interpolate_missing_values(df, method='cubic')
        
        # 3. Normalizar landmarks
        print(f"📏 Normalizando landmarks (método: {normalize_method})...")
        df = self.normalize_landmarks(df, method=normalize_method)
        
        # 4. Detectar outliers
        print("🔍 Detectando outliers...")
        df = self.detect_outliers(df, threshold=3.0)
        outliers_count = df['is_outlier'].sum()
        print(f"   Outliers detectados: {outliers_count}")
        
        # Re-interpolar después de marcar outliers
        if outliers_count > 0:
            df = self.interpolate_missing_values(df, method='cubic')
        
        # 5. Aplicar filtrado temporal
        print(f"🎚️  Aplicando filtro temporal ({filter_type})...")
        df = self.apply_temporal_filter(df, filter_type=filter_type, 
                                       window_size=window_size)
        
        # 6. Calcular derivadas
        if compute_derivatives:
            print("📈 Calculando velocidades y aceleraciones...")
            df = self.compute_velocities(df, fps=fps)
            df = self.compute_accelerations(df, fps=fps)
        
        # 7. Calcular ángulos de articulaciones
        if compute_angles:
            print("📐 Calculando ángulos articulares...")
            df = self.compute_joint_angles(df)
        
        # 8. Guardar resultados
        print(f"💾 Guardando resultados...")
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix == '.csv':
            df.to_csv(output_file, index=False)
        elif output_path.suffix == '.parquet':
            df.to_parquet(output_file, index=False, compression='snappy')
        
        final_shape = df.shape
        print(f"   Dimensiones finales: {final_shape}")
        print(f"   Archivo guardado: {output_file}")
        
        # Estadísticas
        stats = {
            'input_file': input_file,
            'output_file': output_file,
            'original_shape': original_shape,
            'final_shape': final_shape,
            'outliers_detected': int(outliers_count),
            'normalization_method': normalize_method,
            'filter_type': filter_type,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"\n✅ Procesamiento completado\n")
        
        return stats


def main():
    parser = argparse.ArgumentParser(
        description="Preprocesamiento avanzado de landmarks"
    )
    parser.add_argument('--input_file', type=str, required=True,
                       help='Archivo de landmarks de entrada')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Archivo de salida para landmarks procesados')
    parser.add_argument('--fps', type=float, default=30.0,
                       help='FPS del video original')
    parser.add_argument('--normalize_method', type=str, default='shoulder_hip',
                       choices=['shoulder_hip', 'torso', 'body_bbox'],
                       help='Método de normalización')
    parser.add_argument('--filter_type', type=str, default='savgol',
                       choices=['moving_avg', 'savgol', 'gaussian'],
                       help='Tipo de filtro temporal')
    parser.add_argument('--window_size', type=int, default=7,
                       help='Tamaño de ventana para filtrado')
    parser.add_argument('--no_derivatives', action='store_true',
                       help='No calcular velocidades y aceleraciones')
    parser.add_argument('--no_angles', action='store_true',
                       help='No calcular ángulos articulares')
    
    args = parser.parse_args()
    
    # Crear procesador
    processor = LandmarkProcessor()
    
    # Procesar landmarks
    stats = processor.process_landmarks(
        input_file=args.input_file,
        output_file=args.output_file,
        fps=args.fps,
        normalize_method=args.normalize_method,
        filter_type=args.filter_type,
        window_size=args.window_size,
        compute_derivatives=not args.no_derivatives,
        compute_angles=not args.no_angles
    )
    
    # Guardar estadísticas
    stats_file = Path(args.output_file).parent / "processing_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"📊 Estadísticas guardadas: {stats_file}\n")


if __name__ == "__main__":
    main()
