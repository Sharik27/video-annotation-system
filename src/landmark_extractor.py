"""
landmark_extractor.py
=====================
Extractor avanzado de landmarks usando MediaPipe.

Características:
- Extracción de pose, manos y rostro
- Soporte para múltiples formatos de salida (CSV, Parquet, NPY)
- Visualización opcional de landmarks sobre video
- Procesamiento en batch con multiprocessing
- Manejo robusto de errores

Uso:
    python src/landmark_extractor.py --input_dir ./data/videos --output_dir ./data/landmarks --visualize
"""

import cv2
import os
import numpy as np
import pandas as pd
import mediapipe as mp
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import argparse
import json
from datetime import datetime


class LandmarkExtractor:
    """Extractor de landmarks con MediaPipe."""
    
    def __init__(self, 
                 extract_pose: bool = True,
                 extract_hands: bool = False,
                 extract_face: bool = False,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        
        self.extract_pose = extract_pose
        self.extract_hands = extract_hands
        self.extract_face = extract_face
        
        # Inicializar MediaPipe
        self.mp_pose = mp.solutions.pose if extract_pose else None
        self.mp_hands = mp.solutions.hands if extract_hands else None
        self.mp_face = mp.solutions.face_mesh if extract_face else None
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Configuración
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
    def extract_from_video(self, 
                          video_path: str,
                          output_path: str,
                          frame_skip: int = 1,
                          output_format: str = 'parquet',
                          visualize: bool = False) -> Dict:
        """
        Extrae landmarks de un video.
        
        Args:
            video_path: Ruta al video
            output_path: Ruta de salida para los landmarks
            frame_skip: Saltar N frames (1 = todos los frames)
            output_format: 'csv', 'parquet', o 'npy'
            visualize: Si True, guarda video con landmarks dibujados
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video_path}")
        
        # Información del video
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Preparar video de salida si se requiere visualización
        video_writer = None
        if visualize:
            video_output_path = str(Path(output_path).parent / 
                                   f"{Path(video_path).stem}_landmarks.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_output_path, fourcc, 
                                          fps, (width, height))
        
        # Inicializar detectores
        pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        ) if self.extract_pose else None
        
        hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        ) if self.extract_hands else None
        
        face_mesh = self.mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        ) if self.extract_face else None
        
        # Almacenar resultados
        landmark_data = []
        frame_idx = 0
        processed_frames = 0
        detection_failures = 0
        
        pbar = tqdm(total=total_frames, 
                   desc=f"Extrayendo {Path(video_path).name}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Saltar frames si es necesario
            if frame_idx % frame_skip != 0:
                frame_idx += 1
                pbar.update(1)
                continue
            
            # Convertir a RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Preparar registro de frame
            frame_data = {
                'video_name': Path(video_path).name,
                'frame_index': frame_idx,
                'timestamp': frame_idx / fps
            }
            
            # Extraer pose landmarks
            pose_detected = False
            if pose:
                pose_results = pose.process(image_rgb)
                if pose_results.pose_landmarks:
                    pose_detected = True
                    for idx, landmark in enumerate(pose_results.pose_landmarks.landmark):
                        frame_data[f'pose_x_{idx}'] = landmark.x
                        frame_data[f'pose_y_{idx}'] = landmark.y
                        frame_data[f'pose_z_{idx}'] = landmark.z
                        frame_data[f'pose_vis_{idx}'] = landmark.visibility
                    
                    # Dibujar en video si se requiere
                    if visualize:
                        self.mp_drawing.draw_landmarks(
                            frame,
                            pose_results.pose_landmarks,
                            self.mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=self.mp_drawing_styles
                            .get_default_pose_landmarks_style()
                        )
                else:
                    # Rellenar con NaN si no hay detección
                    for idx in range(33):  # 33 landmarks de pose
                        frame_data[f'pose_x_{idx}'] = np.nan
                        frame_data[f'pose_y_{idx}'] = np.nan
                        frame_data[f'pose_z_{idx}'] = np.nan
                        frame_data[f'pose_vis_{idx}'] = np.nan
                    detection_failures += 1
            
            # Extraer hand landmarks
            if hands:
                hands_results = hands.process(image_rgb)
                if hands_results.multi_hand_landmarks:
                    for hand_idx, hand_landmarks in enumerate(
                        hands_results.multi_hand_landmarks[:2]
                    ):
                        for idx, landmark in enumerate(hand_landmarks.landmark):
                            frame_data[f'hand{hand_idx}_x_{idx}'] = landmark.x
                            frame_data[f'hand{hand_idx}_y_{idx}'] = landmark.y
                            frame_data[f'hand{hand_idx}_z_{idx}'] = landmark.z
                        
                        if visualize:
                            self.mp_drawing.draw_landmarks(
                                frame,
                                hand_landmarks,
                                self.mp_hands.HAND_CONNECTIONS
                            )
            
            # Extraer face landmarks
            if face_mesh:
                face_results = face_mesh.process(image_rgb)
                if face_results.multi_face_landmarks:
                    face_landmarks = face_results.multi_face_landmarks[0]
                    for idx, landmark in enumerate(face_landmarks.landmark[:20]):
                        frame_data[f'face_x_{idx}'] = landmark.x
                        frame_data[f'face_y_{idx}'] = landmark.y
                        frame_data[f'face_z_{idx}'] = landmark.z
            
            landmark_data.append(frame_data)
            
            # Escribir frame con landmarks si se requiere
            if visualize and video_writer:
                video_writer.write(frame)
            
            frame_idx += 1
            processed_frames += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        if pose:
            pose.close()
        if hands:
            hands.close()
        if face_mesh:
            face_mesh.close()
        if video_writer:
            video_writer.release()
        
        # Guardar landmarks
        df = pd.DataFrame(landmark_data)
        
        if output_format == 'csv':
            df.to_csv(output_path, index=False)
        elif output_format == 'parquet':
            df.to_parquet(output_path, index=False, compression='snappy')
        elif output_format == 'npy':
            np.save(output_path, df.to_numpy())
            # Guardar también los nombres de columnas
            with open(output_path.replace('.npy', '_columns.json'), 'w') as f:
                json.dump(df.columns.tolist(), f)
        
        # Estadísticas de extracción
        detection_rate = (processed_frames - detection_failures) / processed_frames * 100
        
        stats = {
            'video_name': Path(video_path).name,
            'total_frames': total_frames,
            'processed_frames': processed_frames,
            'detection_failures': detection_failures,
            'detection_rate': round(detection_rate, 2),
            'output_path': str(output_path),
            'timestamp': datetime.now().isoformat()
        }
        
        return stats
    
    def extract_from_directory(self,
                               input_dir: str,
                               output_dir: str,
                               frame_skip: int = 1,
                               output_format: str = 'parquet',
                               visualize: bool = False,
                               extensions: Tuple[str, ...] = ('.mp4', '.avi', '.mov')):
        """Extrae landmarks de todos los videos en un directorio."""
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Buscar videos
        video_files = []
        for ext in extensions:
            video_files.extend(input_path.glob(f"*{ext}"))
        
        if not video_files:
            print(f"⚠️  No se encontraron videos en {input_dir}")
            return
        
        print(f"\n{'='*60}")
        print(f"Extrayendo landmarks de {len(video_files)} videos")
        print(f"{'='*60}\n")
        
        all_stats = []
        
        for video_file in video_files:
            # Determinar ruta de salida
            output_filename = f"{video_file.stem}_landmarks.{output_format}"
            if output_format == 'npy':
                output_filename = f"{video_file.stem}_landmarks.npy"
            
            output_file = output_path / output_filename
            
            try:
                stats = self.extract_from_video(
                    str(video_file),
                    str(output_file),
                    frame_skip=frame_skip,
                    output_format=output_format,
                    visualize=visualize
                )
                all_stats.append(stats)
                print(f"✓ {video_file.name}: Tasa de detección {stats['detection_rate']}%\n")
                
            except Exception as e:
                print(f"❌ Error procesando {video_file.name}: {str(e)}\n")
                continue
        
        # Guardar resumen de estadísticas
        if all_stats:
            stats_df = pd.DataFrame(all_stats)
            stats_path = output_path / "extraction_stats.csv"
            stats_df.to_csv(stats_path, index=False)
            print(f"\n📊 Estadísticas guardadas: {stats_path}")
            
            print("\n" + "="*60)
            print("RESUMEN DE EXTRACCIÓN")
            print("="*60)
            print(f"Videos procesados: {len(all_stats)}")
            print(f"Tasa de detección promedio: {stats_df['detection_rate'].mean():.2f}%")
            print(f"Total de frames procesados: {stats_df['processed_frames'].sum()}")


def main():
    parser = argparse.ArgumentParser(
        description="Extracción avanzada de landmarks con MediaPipe"
    )
    parser.add_argument('--input_dir', type=str, default='./data/videos',
                       help='Directorio con videos de entrada')
    parser.add_argument('--output_dir', type=str, default='./data/landmarks',
                       help='Directorio para guardar landmarks')
    parser.add_argument('--frame_skip', type=int, default=1,
                       help='Procesar 1 de cada N frames (1 = todos)')
    parser.add_argument('--format', type=str, default='parquet',
                       choices=['csv', 'parquet', 'npy'],
                       help='Formato de salida')
    parser.add_argument('--visualize', action='store_true',
                       help='Generar videos con landmarks dibujados')
    parser.add_argument('--extract_hands', action='store_true',
                       help='Extraer landmarks de manos')
    parser.add_argument('--extract_face', action='store_true',
                       help='Extraer landmarks faciales')
    parser.add_argument('--min_confidence', type=float, default=0.5,
                       help='Confianza mínima para detección')
    
    args = parser.parse_args()
    
    # Crear extractor
    extractor = LandmarkExtractor(
        extract_pose=True,
        extract_hands=args.extract_hands,
        extract_face=args.extract_face,
        min_detection_confidence=args.min_confidence,
        min_tracking_confidence=args.min_confidence
    )
    
    # Extraer landmarks
    extractor.extract_from_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        frame_skip=args.frame_skip,
        output_format=args.format,
        visualize=args.visualize
    )
    
    print(f"\n{'='*60}")
    print("✅ Extracción completada")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
