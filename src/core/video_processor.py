import cv2
import numpy as np
import mediapipe as mp
from .face_analyzer import FaceAnalyzer
from typing import Optional, Tuple
from collections import deque

class VideoProcessor:
    def __init__(self):
        self.cap = None
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 0 for short-range, 1 for full-range
            min_detection_confidence=0.7
        )
        
        # Initialize face analyzer for detailed analysis
        self.face_analyzer = FaceAnalyzer(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        self.process_interval = 1.0 / 30  # Default 30 FPS processing
        self.last_process_time = 0
        
        # Initialize temporal smoothing with optimized parameters
        self.smoothing_window = 3  # Reduced window size for less ghosting
        self.frame_buffer = deque(maxlen=self.smoothing_window)
        self.last_valid_frame = None
        self.last_detection_time = 0
        self.detection_interval = 1.0 / 15  # Separate interval for detection to reduce double boxes
        
    def start_capture(self, camera_id: int = 0) -> bool:
        try:
            self.cap = cv2.VideoCapture(camera_id)
            if self.cap.isOpened():
                # Set optimal camera parameters
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                return True
            return False
        except Exception as e:
            print(f"Error starting camera: {str(e)}")
            return False
    
    def stop_capture(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.frame_buffer.clear()
            self.last_valid_frame = None
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        if self.cap is None or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return self.last_valid_frame
        
        # Apply temporal smoothing
        self.frame_buffer.append(frame)
        if len(self.frame_buffer) < self.smoothing_window:
            return frame
        
        # Process frame with facial analysis
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        if (current_time - self.last_process_time) >= self.process_interval:
            try:
                # Apply weighted temporal smoothing to reduce jitter
                weights = np.linspace(0.5, 1.0, len(self.frame_buffer))
                weights = weights / np.sum(weights)
                smoothed_frame = np.average(self.frame_buffer, axis=0, weights=weights).astype(np.uint8)
                
                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(smoothed_frame, cv2.COLOR_BGR2RGB)
                
                # Only run detection at specified interval to prevent double boxes
                should_detect = (current_time - self.last_detection_time) >= self.detection_interval
                annotated_frame = smoothed_frame.copy()
                
                if should_detect:
                    # Detect faces first
                    detection_results = self.face_detection.process(rgb_frame)
                    self.last_detection_time = current_time
                    
                    if detection_results.detections:
                        for detection in detection_results.detections:
                            self.mp_drawing.draw_detection(annotated_frame, detection)
                            
                        # Perform detailed facial analysis on detected faces
                        processed_frame, _ = self.face_analyzer.analyze_frame(annotated_frame)
                        self.last_process_time = current_time
                        self.last_valid_frame = processed_frame
                        return processed_frame
                
                # If no detection needed or no faces found, return the annotated frame
                self.last_valid_frame = annotated_frame
                return annotated_frame
            except Exception as e:
                print(f"Error processing frame: {str(e)}")
                return frame
        
        # Return the most recent valid frame, with proper null checking
        if self.last_valid_frame is not None:
            return self.last_valid_frame
        return frame
    
    def set_detection_confidence(self, confidence: float) -> None:
        if hasattr(self.face_analyzer, 'face_mesh'):
            self.face_analyzer = FaceAnalyzer(
                min_detection_confidence=confidence,
                min_tracking_confidence=confidence
            )
    
    def set_processing_interval(self, fps: float) -> None:
        if fps > 0:
            self.process_interval = 1.0 / fps
            # Adjust smoothing window based on FPS
            self.smoothing_window = max(2, min(5, int(fps / 6)))
            self.frame_buffer = deque(maxlen=self.smoothing_window)